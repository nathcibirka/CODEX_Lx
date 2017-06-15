# CODEX_Lx
CODEX luminosity function

Updated to include curvature, effective dark energy of state (both via geometry) and massive neutrinos (via recipe by Costanzi et al 1311.1514)

Note: to use this code, you have to add lookup functions for Omega_k, w0 and wa to the CLASS python wrapper. Go to your class/python/classy.pyx file and add the following function definitions:


    def Omega_k(self):
        return self.ba.Omega0_k

    def w0_fld(self):
        return self.ba.w0_fld

    def wa_fld(self):
        return self.ba.wa_fld


And recompile via make class