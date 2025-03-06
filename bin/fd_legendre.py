#!/usr/bin/env python3

import os,argparse,configparser

import torch
from matplotlib import pyplot as plt

from   Tools.Legendre     import LegendreDecompFn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
end    = 1        # Maximum value to evaluate
Npt    = int(1e3)  # Number of points to use

toplot = {   1 : {'color': 'orange'},
             3 : {'color': 'red'},
             2 : {'color': 'green'},
             4 : {'color': 'purple'},
             5 : {'color': 'brown'},
             0 : {'color': 'blue'},
         }

torch.set_printoptions(precision=3, linewidth=160)

######
# Parse command line parameters and config files
######
ArgP      = argparse.ArgumentParser(description='=== Functional Decomposition Legendre Polynomial Plotter ===')
ArgP.add_argument('--base', type=str, default=".",                  help="FD base directory.")
ArgP.add_argument('--show', action="store_true",                    help="Display plots interactively.")
ArgP.add_argument('--logx', action="store_true",                    help="Set x axis to log.")
ArgP.add_argument('--save', type=str, default="Output/orthexp.pdf", help="Filename to save plot.")
ArgC      = ArgP.parse_args()

Config    = configparser.ConfigParser()
Config.optionxform = str
Config.read( os.path.join(ArgC.base, "base.conf") )

PlotStyle = Config.get("General", "PlotStyle")
try:            plt.style.use( PlotStyle )
except IOError: plt.style.use( os.path.join(ArgC.base, PlotStyle) )


x      = torch.linspace(-1, end, Npt, device=device)
w      = torch.ones((Npt,), device=device) / Npt

Decomp = LegendreDecompFn( x=x, w=w, Nbasis=max(toplot.keys())+1, device=device )

######
for D in Decomp:
    if D.N in toplot:
        plt.plot(x.cpu(), Decomp.Values().cpu(), zorder=-D.N, label='$E_{%d}\\left(z\\right)$' % D.N, **toplot[D.N])
    print(D.Values())
    print("%d: %f" % (D.N, Decomp.Moment()))

plt.xlabel("z")
plt.ylabel("Arbitrary Units")
plt.xlim( -1, end)
plt.ylim(-1.02, 1.02)

if ArgC.logx:
    plt.xscale('log')
plt.legend()
plt.tight_layout()

try:
    plt.savefig(ArgC.save)
except IOError:
    print("Directory for save does not exist or cannot be written to.")
    
if ArgC.show:
    plt.show()
