# code for p3 task 2 (taken from p2 juypter notebook stuff)
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import root, brentq

HERE = os.path.dirname(os.path.abspath(__file__))
np.seterr(all="ignore")

# constants
RHO, NU, G = 1.225, 1.46e-5, 9.81
S, B = 0.633, 1.59
AR = B ** 2 / S
W_P1, M_WING_P1, T_P1 = 19.78, 0.503, 0.12      # P1 weight
V_CR, V_STALL = 8.35, 7.0                       # P1 speeds
CD_OTHER = 0.020
E_FOAM, RHO_FOAM = 19.3e6, 25.2
ETA_CHECK, SOFT_LIMIT, SOFT_FACTOR = 0.9, 0.20, 1.05
STALL_TOL = 0.998
POST_SLOPE = 0.003
RE_EXP = -0.55
NETA = 400
TIP_ETA = 0.95
ETA = (np.arange(NETA) + 0.5) / NETA
NT = 40
PHI = np.arange(1, NT + 1) * np.pi / (2 * NT)
NN = 2 * np.arange(NT) + 1
ETA_J = np.cos(PHI)
S_J = np.sin(np.outer(PHI, NN))
SPHI = np.sin(PHI)
PHI_E = np.arccos(ETA)
S_E = np.sin(np.outer(PHI_E, NN))
SPHI_E = np.sin(PHI_E)
R2D = 180 / np.pi

P2 = dict(cb_cr=1.0, ct_cr=0.7, th_b=-4.45, th_t=-4.45, eta_b=0.5)
BASE = dict(P2)


# readingnXFOIL polars
def read_xfoil(path):
    txt = open(path).read()
    naca = re.search(r"NACA\s+(\d{4})", txt).group(1)
    m = re.search(r"Re\s*=\s*([\d.]+)\s*e\s*(\d+)", txt)
    Re = float(m.group(1)) * 10 ** int(m.group(2))
    rows = []
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) == 9:
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                pass
    d = np.array(rows)
    d = d[np.argsort(d[:, 0], kind="stable")]
    _, i = np.unique(d[:, 0], return_index=True)
    return naca, Re, d[i]


class Section:
    def __init__(s, path, name, cruise_path=None, V_cr_chord=None):
        s.name = name
        s.naca, s.Re_stall, d = read_xfoil(path)
        s.t_pct = float(s.naca[2:]); s.m_pct = float(s.naca[0])
        a, cl, cd = d[:, 0], d[:, 1], d[:, 2]
        k = int(np.argmax(np.where(a <= 12.0, cl, -9)))     # first stall peak
        s.apk, s.clmax = a[k], cl[k]
        pre = a <= s.apk
        s.a_pre, s.cl_pre = a[pre], np.maximum.accumulate(cl[pre])
        s.cd_pre_stall = cd[pre]
        fit = (a >= 0) & (a <= 5)
        s.a0_deg = np.polyfit(a[fit], cl[fit], 1)[0]
        near0 = pre & (a > -6)
        s.aL0 = float(np.interp(0.0, cl[near0], a[near0]))
        s.cd_cruise = None
        if cruise_path and os.path.exists(cruise_path):
            _, s.Re_cruise, dc = read_xfoil(cruise_path)
            ac, clc, cdc = dc[:, 0], dc[:, 1], dc[:, 2]
            kc = int(np.argmax(np.where(ac <= 12.0, clc, -9)))
            pc = ac <= ac[kc]
            s.cl_c, s.cd_c = np.maximum.accumulate(clc[pc]), cdc[pc]
        else:
            s.Re_cruise = None

    def lift(s, al):
        al = np.asarray(al, float)
        out = np.interp(al, s.a_pre, s.cl_pre)
        out = np.where(al < s.a_pre[0], s.cl_pre[0] + s.a0_deg * (al - s.a_pre[0]), out)
        out = np.where(al > s.apk, s.clmax + POST_SLOPE * (al - s.apk), out)
        return out

    def cd_stall(s, cl):
        return np.interp(cl, s.cl_pre, s.cd_pre_stall)

    def cd_cr(s, cl, chord_here):
        if s.cd_cruise is not None or s.Re_cruise:
            return np.interp(cl, s.cl_c, s.cd_c)
        Re_cr = V_CR * chord_here / NU
        return np.interp(cl, s.cl_pre, s.cd_pre_stall) * (Re_cr / s.Re_stall) ** RE_EXP


def load_sections():
    names = ("root", "break", "tip")
    out = []
    for n in names:
        out.append(Section(os.path.join(HERE, "polars", f"{n}_stall.txt"), n,
                           os.path.join(HERE, "polars", f"{n}_cruise.txt")))
    return out


# wing
class Wing:
    def __init__(s, cb_cr, ct_cr, th_b, th_t, eta_b, secs):
        s.eb, s.cb_cr, s.ct_cr, s.th, s.secs = eta_b, cb_cr, ct_cr, (0.0, th_b, th_t), secs
        ar = eta_b * (1 + cb_cr) / 2 + (1 - eta_b) * (cb_cr + ct_cr) / 2
        s.cr = S / (B * ar)

    def _pw(s, eta, v): return np.interp(eta, [0, s.eb, 1.0], v)
    def chord(s, eta): return s.cr * s._pw(eta, [1.0, s.cb_cr, s.ct_cr])
    def twist_deg(s, eta): return s._pw(eta, s.th)
    def pw(s, eta):
        eta = np.asarray(eta, float)
        w = np.where(eta <= s.eb, eta / s.eb, (eta - s.eb) / (1 - s.eb))
        p = (eta > s.eb).astype(int)
        return p, w

    def blend3(s, v3, eta):
        p, w = s.pw(eta); i = np.arange(len(p))
        return (1 - w) * v3[p, i] + w * v3[p + 1, i]

    def lift(s, al_deg, eta):
        v3 = np.array([sec.lift(al_deg) for sec in s.secs]); return s.blend3(v3, eta)

    def clmax(s, eta):
        v = np.array([[sec.clmax] * len(eta) for sec in s.secs]); return s.blend3(v, eta)

    def aL0(s, eta):
        v = np.array([[sec.aL0] * len(eta) for sec in s.secs]); return s.blend3(v, eta)

    def cd(s, cl, eta, mode):
        c = s.chord(eta)
        if mode == "stall":
            v3 = np.array([sec.cd_stall(cl) for sec in s.secs])
        else:
            ends = [s.cr, s.cr * s.cb_cr, s.cr * s.ct_cr]
            v3 = np.array([sec.cd_cr(cl, ends[i]) for i, sec in enumerate(s.secs)])
        return s.blend3(v3, eta)

    def mass(s, t_pct):
        c0 = S / B
        k = M_WING_P1 / (RHO_FOAM * T_P1 * c0 ** 2 * B)
        return RHO_FOAM * k * (t_pct / 100) * B * np.sum(s.chord(ETA) ** 2) / NETA


# lifting line
class LL:
    def __init__(s, wing):
        s.w = wing
        s.c_j, s.th_j = wing.chord(ETA_J), wing.twist_deg(ETA_J)
        s.c_e, s.th_e = wing.chord(ETA), wing.twist_deg(ETA)
        s.aL0_j = wing.aL0(ETA_J)
        s.k_j = 4 * B / s.c_j
        s.k_e = 4 * B / s.c_e

    def residual(s, A, al):
        ai = (S_J @ (NN * A)) / SPHI * R2D
        ae = al + s.th_j - ai
        return s.k_j * (S_J @ A) - s.w.lift(ae, ETA_J)

    def guess(s, al):
        a0 = 6.0
        M = S_J * (4 * B / (a0 * s.c_j))[:, None] + S_J * NN[None, :] / SPHI[:, None]
        return np.linalg.solve(M, (al + s.th_j - s.aL0_j) / R2D)

    def solve(s, al, A0=None):
        A0 = s.guess(al) if A0 is None else A0
        r = root(s.residual, A0, args=(al,), method="hybr", tol=1e-11)
        ok = r.success and np.max(np.abs(s.residual(r.x, al))) < 1e-6
        return r.x, ok

    def step(s, al, Aprev, alprev):
        A, ok = s.solve(al, Aprev)
        if ok or Aprev is None:
            return A, ok
        Ac = Aprev
        for a in np.linspace(alprev, al, 6)[1:]:
            Ac, ok = s.solve(a, Ac)
            if not ok: return Ac, False
        return Ac, ok

    @staticmethod
    def CL(A): return np.pi * AR * A[0]
    @staticmethod
    def CDi(A): return np.pi * AR * np.sum(NN * A ** 2)

    def at_eta(s, A, al):
        ai = (S_E @ (NN * A)) / SPHI_E * R2D
        ae = al + s.th_e - ai
        return s.k_e * (S_E @ A), ae

    def sweep(s, alphas):
        A, prev = None, None
        out = dict(al=[], CL=[], A=[], ok=[])
        for al in alphas:
            A, ok = s.step(al, A, prev if prev is not None else al)
            if not ok:
                A_good = out["A"][-1] if out["A"] else None
                A = A_good
                out["al"].append(al); out["CL"].append(np.nan); out["A"].append(A_good); out["ok"].append(False)
                prev = out["al"][-2] if len(out["al"]) > 1 else al
                continue
            out["al"].append(al); out["CL"].append(s.CL(A)); out["A"].append(A); out["ok"].append(True)
            prev = al
        out["al"], out["CL"], out["ok"] = map(np.array, (out["al"], out["CL"], out["ok"]))
        return out

    def alpha_for_CL(s, CLt, lo=-4.0, hi=12.0):
        f = lambda a: s.CL(s.solve(a)[0]) - CLt
        return brentq(f, lo, hi, xtol=1e-6)


# analysis
def stall_front(ll, sw):
    cm = ll.w.clmax(ETA)
    ok = sw["ok"]
    CLs = sw["CL"][ok]
    ratio = []
    for A, al in zip([a for a, o in zip(sw["A"], ok) if o], sw["al"][ok]):
        _, ae = ll.at_eta(A, al)
        ratio.append(ll.w.lift(ae, ETA) / cm)
    ratio = np.array(ratio)
    star = np.full(NETA, np.nan)
    for j in range(NETA):
        idx = np.where(ratio[:, j] >= STALL_TOL)[0]
        if len(idx) == 0: continue
        i = idx[0]
        if i == 0: star[j] = CLs[0]; continue
        r0, r1 = ratio[i - 1, j], ratio[i, j]
        star[j] = CLs[i - 1] + (CLs[i] - CLs[i - 1]) * (STALL_TOL - r0) / (r1 - r0)
    return star


def linear_front(ll, CL1=0.35, CL2=0.75):
    sol = []
    for CLt in (CL1, CL2):
        a = ll.alpha_for_CL(CLt, -2.0, 10.0); A, _ = ll.solve(a); cl, _ = ll.at_eta(A, a); sol.append((ll.CL(A), cl))
    dcl = (sol[1][1] - sol[0][1]) / (sol[1][0] - sol[0][0])
    clb = sol[0][1] - sol[0][0] * dcl
    return (ll.w.clmax(ETA) - clb) / dcl


def evaluate(design, secs, d_alpha=0.25, a_max=17.0, full=True):
    wing = Wing(design["cb_cr"], design["ct_cr"], design["th_b"], design["th_t"], design["eta_b"], secs)
    ll = LL(wing)
    r = dict(wing=wing, ll=ll, design=design)
    r["m_wing"] = wing.mass(np.mean([s.t_pct for s in secs]))
    r["W"] = (W_P1 - M_WING_P1 * G) + r["m_wing"] * G
    r["CLmax_req"] = 2 * r["W"] / (RHO * V_STALL ** 2 * S)
    r["CL_cr"] = 2 * r["W"] / (RHO * V_CR ** 2 * S)
    sw = ll.sweep(np.arange(-3.0, a_max + 1e-9, d_alpha))
    r["sweep"] = sw
    star = stall_front(ll, sw)
    r["CLstar"] = star
    chk = ETA <= ETA_CHECK
    sc = np.where(chk & np.isfinite(star), star, np.inf)
    i = int(np.argmin(sc))
    r["CLmax"], r["eta_first"] = star[i], ETA[i]
    r["CLstar_nl"] = star
    lin = linear_front(ll)
    r["CLstar"] = lin
    il = int(np.argmin(np.where(chk, lin, np.inf)))
    r["CLmax_lin"], r["eta_first_lin"] = lin[il], ETA[il]
    r["soft"] = float(np.mean(lin[chk] <= SOFT_FACTOR * r["CLmax_lin"]))
    # cruise
    a_cr = ll.alpha_for_CL(r["CL_cr"])
    A_cr, _ = ll.solve(a_cr)
    cl_cr, _ = ll.at_eta(A_cr, a_cr)
    r["alpha_cr"], r["cl_cr_span"] = a_cr, cl_cr
    r["CDi"] = ll.CDi(A_cr)
    r["CDp"] = (B / S) * np.sum(wing.cd(cl_cr, ETA, "cruise") * wing.chord(ETA)) / NETA
    r["e"] = r["CL_cr"] ** 2 / (np.pi * AR * r["CDi"])
    r["CD"] = CD_OTHER + r["CDp"] + r["CDi"]
    r["D"] = 0.5 * RHO * V_CR ** 2 * S * r["CD"]
    r["V_stall_ach"] = np.sqrt(2 * r["W"] / (RHO * S * r["CLmax"]))
    # structure
    t_r, eps_r = secs[0].t_pct / 100, secs[0].m_pct / 100
    lam = 2 * S / (B * wing.cr) - 1
    r["lam"] = lam
    r["delta_b"] = (0.018 * (r["W"] - r["m_wing"] * G) / (E_FOAM * t_r * (t_r ** 2 + eps_r ** 2))
                    * (1 + lam) ** 3 * (1 + 2 * lam) * AR ** 3 / S)
    r["h_root"] = 1000 * wing.cr * secs[0].t_pct / 100
    r["h_break"] = 1000 * wing.cr * wing.cb_cr * secs[1].t_pct / 100
    r["h_tip"] = 1000 * wing.cr * wing.ct_cr * secs[2].t_pct / 100
    r["panel_in"] = [design["eta_b"] * B / 2 / 0.0254, (1 - design["eta_b"]) * B / 2 / 0.0254]
    r["R"] = dict(R1=r["CLmax"] >= r["CLmax_req"],
                  R2=(r["soft"] <= SOFT_LIMIT) and (r["eta_first_lin"] <= 0.05),
                  R3=max(r["h_root"], r["h_break"]) <= 50.0,
                  R4=max(r["panel_in"]) <= 24.0,
                  R5=r["delta_b"] <= 0.1,
                  R6=r["h_tip"] >= 15.0)
    return r


# gives cruise cl for the 3 sections
def section_cruise_cl(r, etas=None):
    ll, wing = r["ll"], r["wing"]
    A, _ = ll.solve(r["alpha_cr"])
    etas = etas or (("root", 0.0), ("break", wing.eb), ("tip", TIP_ETA))
    out = []
    for name, eta in etas:
        e = np.array([eta]); phi = np.arccos(e)
        c = float(wing.chord(e)[0])
        cl = float(4 * B / c * (np.sin(np.outer(phi, NN)) @ A)[0])
        cd = float(wing.cd(np.array([cl]), e, "cruise")[0])
        out.append((name, eta, c, cl, cd))
    return out


def show(r, tag=""):
    print(f"--- {tag}")
    for k in ("cr", ):
        pass
    for k in ("m_wing", "W", "CLmax_req", "CL_cr", "CLmax", "eta_first", "CLmax_lin", "soft", "alpha_cr", "CDi", "CDp",
              "e", "CD", "D", "V_stall_ach", "delta_b", "h_root", "h_break", "h_tip"):
        print(f"  {k:12s} = {r[k]:.5g}")
    for nm, eta, c, cl, cd in section_cruise_cl(r):
        print(f"  cruise cl at {nm:5s} (eta={eta:.2f}, c={c:.4f} m): cl = {cl:.4f}, cd = {cd:.5f}")
    print("  ", {k: ("ok" if v else "FAIL") for k, v in r["R"].items()},
          " sweep converged:", int(r["sweep"]["ok"].sum()), "/", len(r["sweep"]["ok"]))


# twist search
def twist_search(secs):
    best, rows = None, []
    for tb in (-6, -5, -4.45, -4, -3, -2, -1, 0):
        for tt in (-6, -5, -4.45, -4, -3, -2, -1, 0, 1, 2):
            d = dict(BASE, th_b=tb, th_t=tt)
            try:
                r = evaluate(d, secs, d_alpha=0.5, a_max=15.0)
            except Exception:
                continue
            rows.append((tb, tt, r["CLmax"], r["CLmax_req"], r["soft"], r["eta_first_lin"], r["CD"], r["R"]["R1"] and r["R"]["R2"]))
            if r["R"]["R1"] and r["R"]["R2"] and (best is None or r["CD"] < best[0]):
                best = (r["CD"], tb, tt)
    return best, rows


# figures + latex
def figures(r, secs):
    ll, wing = r["ll"], r["wing"]
    sw = r["sweep"]; ok = sw["ok"]
    # Fig 3
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.plot(sw["al"][ok], sw["CL"][ok], lw=2, label=r"$C_L$ (nonlinear lifting line)")
    lin = np.polyfit(sw["al"][ok][(sw["CL"][ok] > 0.2) & (sw["CL"][ok] < 0.9)], sw["CL"][ok][(sw["CL"][ok] > 0.2) & (sw["CL"][ok] < 0.9)], 1)
    aa = sw["al"][ok]
    ax.plot(aa, np.polyval(lin, aa), "--", color="gray", lw=1, label="linear extension")
    ax.axhline(r["CLmax_req"], color="tab:green", ls=":", label=fr"$C_{{L\max,req}}={r['CLmax_req']:.3f}$")
    ax.axhline(r["CL_cr"], color="tab:orange", ls=":", label=fr"$C_{{L,cr}}={r['CL_cr']:.3f}$")
    a_ms = np.interp(r["CLmax"], sw["CL"][ok], sw["al"][ok])
    ax.plot(a_ms, r["CLmax"], "ko", label=fr"first stall, $C_{{L\max}}={r['CLmax']:.3f}$")
    ax.set_xlabel(r"root angle of attack $\alpha$ (deg)"); ax.set_ylabel(r"$C_L$")
    ax.set_ylim(-0.2, 2.0); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig3_CL_alpha.png"), dpi=200); plt.close(fig)

    # Fig 4
    CLg = np.linspace(0.1, r["CLmax"], 50)
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for mode, V, st, lab in (("stall", V_STALL, "--", "stall Re numbers"),
                             ("cruise", V_CR, "-", "cruise Re numbers")):
        xs = []
        for c in CLg:
            a = float(np.interp(c, sw['CL'][ok], sw['al'][ok])); A, _ = ll.solve(a); clx, _ = ll.at_eta(A, a)
            cdp = (B / S) * np.sum(wing.cd(clx, ETA, mode) * wing.chord(ETA)) / NETA
            xs.append(cdp + ll.CDi(A))
        ax.plot(xs, CLg, st, lw=2, label=lab)
    ax.plot(r["CDp"] + r["CDi"], r["CL_cr"], "ro", ms=7, label="cruise point")
    ax.set_xlabel(r"wing drag $C_{D,p}+C_{D,i}$"); ax.set_ylabel(r"$C_L$")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig4_CL_CD.png"), dpi=200); plt.close(fig)

    # Fig 5
    a_ms = np.interp(r["CLmax"], sw["CL"][ok], sw["al"][ok])
    A, _ = ll.solve(a_ms)
    cl, ae = ll.at_eta(A, a_ms)
    clm = wing.clmax(ETA)
    stalled = cl >= STALL_TOL * clm
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.plot(ETA, cl, lw=2, label=r"$c_l$ at $C_{L\max}$")
    ax.plot(ETA, clm, "--k", lw=1.5, label=r"$c_{l,\max}$ (XFOIL, local Re)")
    soft = (r["CLstar"] <= SOFT_FACTOR * r["CLmax_lin"]) & (ETA <= ETA_CHECK)
    ax.fill_between(ETA, 0, 1, where=soft, transform=ax.get_xaxis_transform(), color="tab:red", alpha=0.10,
                    label=fr"$C_L^*<1.05\,C_{{L\max}}$ ({100 * r['soft']:.0f}% of checked span)")
    ax.plot(ETA[stalled], cl[stalled], "rs", ms=6, label="stalled")
    ax.plot(ETA[~stalled][::22], cl[~stalled][::22], "o", mfc="none", color="tab:blue", ms=5, label="not stalled")
    ax.axvline(ETA_CHECK, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$\eta=2y/b$"); ax.set_ylabel(r"section $c_l$"); ax.set_ylim(0, 1.8)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig5_cl_span.png"), dpi=200); plt.close(fig)

    # Stall front plot
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(ETA, r["CLstar"] / r["CLmax_lin"], lw=2)
    ax.axhline(1.05, ls="--", color="k", lw=1)
    ax.axvline(ETA_CHECK, color="gray", ls=":", lw=1)
    ax.set_ylim(0.95, 1.6); ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$C_L^*/C_{L\max}$"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_stallfront.png"), dpi=200); plt.close(fig)


def write_tex(r, secs, base, name="results.tex"):
    f = lambda x, n=3: f"{x:.{n}f}"
    L = []
    add = lambda k, v: L.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    wing, d = r["wing"], r["design"]
    cruise_real = all(s.Re_cruise for s in secs)
    L.append("\\newif\\ifcruisepolars"); L.append("\\cruisepolarstrue" if cruise_real else "\\cruisepolarsfalse")
    vals = dict(CLmaxlin=f(r["CLmax_lin"]), CLmax=f(r["CLmax"]), CLmaxreq=f(r["CLmax_req"]), CLcr=f(r["CL_cr"]), etafirst=f(r["eta_first"], 4),
                soft=f(100 * r["soft"], 1), softlimit=f(100 * SOFT_LIMIT, 0), CDi=f(r["CDi"], 5), CDp=f(r["CDp"], 5),
                CD=f(r["CD"], 5), Dcr=f(r["D"], 3), espan=f(r["e"], 4), Wnew=f(r["W"], 2), mwing=f(r["m_wing"], 3),
                Vstallach=f(r["V_stall_ach"], 2), deltab=f(r["delta_b"], 4), lam=f(r["lam"], 3),
                croot=f(wing.cr, 4), cbreak=f(wing.cr * wing.cb_cr, 4), ctip=f(wing.cr * wing.ct_cr, 4),
                hroot=f(r["h_root"], 1), hbreak=f(r["h_break"], 1), htip=f(r["h_tip"], 1), panelin=f(r["panel_in"][0], 1),
                alphacr=f(r["alpha_cr"], 2), thb=f(d["th_b"], 2), tht=f(d["th_t"], 2),
                Vstall=f(V_STALL, 2), Vcr=f(V_CR, 2))
    for k, v in vals.items(): add(k, v)
    for nm, eta, c, cl, cd in section_cruise_cl(r):
        add("clcr" + nm, f(cl, 3)); add("cdcr" + nm, f(cd, 5))
    names = ["one", "two", "three", "four", "five", "six"]
    for i, k in enumerate(["R1", "R2", "R3", "R4", "R5", "R6"]):
        add("status" + names[i], "met" if r["R"][k] else "\\textbf{NOT met}")
    # section table rows
    rows = []
    for s, nm in zip(secs, ("Root", "Break", "Tip")):
        rows.append(f"{nm} & NACA {s.naca} & {s.Re_stall / 1e3:.0f}k & {s.aL0:.2f} & {s.a0_deg * R2D:.2f} & {s.clmax:.3f} & {s.apk:.2f} \\\\")
    L.append("\\newcommand{\\sectionrows}{" + " ".join(rows) + "}")
    # changes
    ch = []
    def row(n, o, nw, why): ch.append(f"{n} & {o} & {nw} & {why} \\\\")
    row("Root section camber $m_r$ (\\%)", "4.2", f"{secs[0].m_pct:.0f}", "NACA 4412 chosen in Task 1")
    row("Break section camber $m_b$ (\\%)", "4.6", f"{secs[1].m_pct:.0f}", "NACA 5412 chosen in Task 1")
    row("Tip section camber $m_t$ (\\%)", "4.6", f"{secs[2].m_pct:.0f}", "NACA 5412 chosen in Task 1")
    for key, nm in (("th_b", "$\\theta_b$ (deg)"), ("th_t", "$\\theta_t$ (deg)")):
        if abs(d[key] - base[key]) > 1e-9:
            row(nm, f"{base[key]}", f"{d[key]}", "tuned with XFOIL sections for R1/R2 + lower drag")
    row("Wing mass (kg)", "0.503", f"{r['m_wing']:.3f}", "P1 mass model with P3 chords")
    row("Weight $W$ (N)", f"{W_P1:.2f}", f"{r['W']:.2f}", "follows from wing mass")
    row("$C_{L\\max,req}$", "1.04", f"{r['CLmax_req']:.3f}", "new $W$; no 10\\% trim allowance")
    L.append("\\newcommand{\\changerows}{" + " ".join(ch) + "}")
    open(os.path.join(HERE, name), "w").write("\n".join(L) + "\n")

DESIGN = dict(BASE, th_b=-2.0, th_t=-4.0)

if __name__ == "__main__":
    import sys
    secs = load_sections()
    for s_ in secs:
        print(f"{s_.name:6s} NACA {s_.naca}  Re={s_.Re_stall:.3g}  aL0={s_.aL0:.3f} deg  slope={s_.a0_deg * R2D:.3f}/rad  "
              f"clmax={s_.clmax:.4f} @ {s_.apk} deg")
    r0 = evaluate(BASE, secs)
    show(r0, "P2 twist (baseline)")
    if "--search" in sys.argv:
        best, rows = twist_search(secs)
        print("best twist:", best)
    r = evaluate(DESIGN, secs)
    show(r, "P3 design")
    figures(r, secs)
    write_tex(r, secs, BASE)
    extra = dict(CLmaxbase=f"{r0['CLmax']:.3f}", softbase=f"{100*r0['soft']:.1f}", CDbase=f"{r0['CD']:.5f}",
                 Dbase=f"{r0['D']:.3f}", tmaxroot=f"{100*50.0/(1000*r['wing'].cr):.1f}")
    with open(os.path.join(HERE, "results.tex"), "a") as fh:
        for k, v in extra.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("wrote results.tex and figures")
