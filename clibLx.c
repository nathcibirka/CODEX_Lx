/* External C librabry to compute CODEX cluster counts */

// Original by Steffen Hagstotz
// steffen.hagstotz@gmail.com

/* --------------------- 

Computes N(etaOb, Texp, z)

// compile for MacOS gcc-4.8 -dynamiclib -o clibLx.so clibLx.c -I/sw/include -lgsl -lgslcblas -lm -O3 -fopenmp
// for gcc 4.8:		 gcc-4.8 -dynamiclib -o szlib.dylib szlib.c -I/sw/include -lgsl -lgslcblas -lm -O3 -std=c99 -fopenmp

// compile for Linux gcc -shared -o clibLx.so -fPIC clibLx.c -I/sw/include -lgsl -lgslcblas -lm -O3 - fopenmp
// 					 gcc -shared -o szlib.so -fPIC szlib.c -lgsl -lm -O3 -std=c99 -lgslcblas -fopenmp

Note: The following functions have to be called first:
setup_cosmo()
setup_nuisance()
spline_init()
setup_sigma_2d_spline()
setup_data()

------------------------ /*

/* --- includes --- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_const_mksa.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>

#include <omp.h>

#define NEVAL 1000 // number of gsl_integrate workspaces
#define epsrel 1e-4 // relative integration error for dV(M) integration

#define ROW 100 
#define COL 2
#define ROW2 61 
#define COL2 61
#define ROWzm 231 
#define COLeta 18
#define ROWz 11
#define ROWm 21 

#define Mstep 100 // length of sigma(M) array
#define zstep 32 // length of sigma(M,z) array

// #define G_NEWTON GSL_CONST_MKSA_GRAVITATIONAL_CONSTANT
// #define M_SUN GSL_CONST_MKSA_SOLAR_MASS

#define nLxbins 8
#define nzbins 10
#define nMbins 10
#define nLambins 8
#define netaObbins 14
#define nTexp 100

// #define sn_threshold 6. // S/N threshold for cluster detection. Planck 2013: 7 / Planck 2015: 6

#define rhocrit 2.7751973751261264e11
#define Omegam 0.28
#define Omegal 0.72
#define Omegak 0.0

#define A_codex 3.046 // codex area in steradians correspondng to 10,000 degrees

#define Texp_min 0.02699
#define Texp_max 2405.6976
#define dAdT_min 0.0
#define dAdT_max 0.0036
#define z_min 0.1
#define z_max 0.6
#define log10M_min 13.5
#define log10M_max 15.5
#define etaOb_min 4.0
#define etaOb_max 21.0
#define lnM_min log(3.2e13)
#define lnM_max log(1e16)
#define lnLx_min log(1e41)
#define lnLx_max log(1e46)
#define lnetaOb_min log(4.0)
#define lnetaOb_max log(1004)
#define lnTexp_min log(5.0)
#define lnTexp_max log(1e3)

/* global variables */
double h;
double Omega_m;
double Omega_nu;
double Omega_k;

// dark energy equation of state
double w0 = -1.;
double wa = 0.;

/* bins for likelihood */
double Lx_vec[9] = {1e42, 5e42, 1e43, 5e43, 1e44, 5e44, 1e45, 5e45, 1e46};
double log10Lx_vec[5] = {42.0, 43.0, 44.0, 45.0, 46.0};
double z_vec[11] = {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6};
double etaOb_vec[97] = { 4.,    5.,    6.,    7.,    8.,    9.,   10.,   11.,   12.,
         13.,   14.,   15.,   16.,   17.,   18.,   19.,   20.,   21.,
         22.,   23.,   24.,   25.,   26.,   27.,   28.,   29.,   30.,
         31.,   32.,   33.,   34.,   35.,   36.,   37.,   38.,   39.,
         40.,   41.,   42.,   43.,   44.,   45.,   46.,   47.,   48.,
         49.,   50.,   51.,   52.,   53.,   54.,   55.,   56.,   57.,
         58.,   59.,   60.,   61.,   62.,   63.,   64.,   65.,   66.,
         67.,   68.,   69.,   70.,   71.,   72.,   73.,   74.,   75.,
         76.,   77.,   78.,   79.,   80.,   81.,   82.,   83.,   84.,
         85.,   86.,   87.,   88.,   89.,   90.,   91.,   92.,   93.,
         94.,   95.,   96.,   97.,   98.,   99.,  100.};
double Lambda_vec[9] = { 10.,   30.,   50.,   70.,   90.,  110.,  130.,  150., 310. };
double M_vec[20] = {3.20000000e+13,   4.06385253e+13,   5.16090544e+13,
         6.55411208e+13,   8.32342031e+13,   1.05703602e+14,
         1.34238704e+14,   1.70476968e+14,   2.16497893e+14,
         2.74942347e+14,   3.49164110e+14,   4.43422330e+14,
         5.63125924e+14,   7.15143974e+14,   9.08199890e+14,
         1.15337201e+15,   1.46472930e+15,   1.86013871e+15,
         2.36229044e+15,   3.00000000e+15};

double Afz[ROW][COL];
double Kcor[ROW2][COL2];
double probDec[ROWzm][COLeta]; // 231 x 18
double probDec3d[ROWz][ROWm][COLeta]; // 11 x 21 x 18
double dummy;

double Texp[100];
double dA_dTexp[100];

double lnM_array[Mstep];
double lnR_array[Mstep];
double z_array[zstep];
double sigma_array[Mstep];
double sigma2d_array[Mstep*zstep];

double array_check(int a, int b);
double interp2d(double f, double z);
double interp1d(double T_exp);
double interp3d(double z, double M, int etaOb);
double gsl_sf_gamma(double gam_arg);
double theory_counts[10][70][97];
int data_counts[nzbins][netaObbins];

// define splines
gsl_interp_accel *acc_norm_spline;	// spline for norm
gsl_spline *norm_spline;
gsl_interp_accel *acc_a_spline;	// spline for a
gsl_spline *a_spline;
gsl_interp_accel *acc_b_spline;	// spline for b
gsl_spline *b_spline;
gsl_interp_accel *acc_c_spline;	// spline for c
gsl_spline *c_spline;


gsl_interp_accel *acc_sigma_M_spline;	// spline for simga(M), replaced by sigma(lnR,z) spline
gsl_spline *sigma_M_spline;

gsl_interp_accel *acc_sigma_R_spline;	// spline for simga(R), replaced by sigma(lnR,z) spline
gsl_spline *sigma_R_spline;

gsl_spline2d *sigma_lnR_z_spline;	// spline for sigma(M,z)
gsl_interp_accel *acc_z_lnR_lnR;
gsl_interp_accel *acc_z_lnR_z;


/* setup functions, must be called first from Python. Initalise splines, read binned data and prepare arrays */
void setup_cosmo(double cosmoh, double cosmoOmega_m, double cosmoOmega_k, double cosmoOmega_nu, double cosmow0, double cosmowa);
void setup_data(double *data);
void setup_sigma_spline(double *min, double *sigmain);
void setup_sigma2d_spline(double *min, double *zin, double *sigmain);
void spline_init(char *path);
void setup_areaflux(char *path);
void setup_kcorr(char *path);

/* cosmology */
double E(double z);
double Omega_m_z(double z);
double dvolume(double z);
double volumez(double z, double Texp);
double distance(double z);
double aux_distance(double z, void *params);

/* Tinker multiplicity and mass function */
double ftinker2008(double sigma, double Delta, double z);
double dndlnM(double lnM, double z, double Delta);

/* spline evaluation functions for debugging only */
// double noisemap_spline_eval(double theta, double skypatch);
// double sigma_lnMz_spline_eval(double lnM, double z);
// double sigma_lnM_spline_eval(double lnM);
// double sigma_lnR_spline_eval(double lnR);
// double sigma_lnR_deriv_spline_eval(double lnR);

/* mass-observable relations and completeness */
//double completeness_2d(double Mtrue, double z, double Ltrue, int etaOb, double Texp);
double transfM(double Mtrue, double z);
double transfML(double M);
double transfLMLx(double lambd, double z);
double transf_etaTrue(double Ltrue, double Texp, double z, double M);

double P_L_M(double Mtrue, double Ltrue, double z);
double P_L_M_cov(double Mtrue, double Ltrue, double z);
double errorF(double lnLambdamin, double lnLambdamax, double M, double z);
double p_eta(double Ltrue, double Texp, unsigned int etaOb, double z, double M);
double p_eta_norm(double Lx, double Texp, double etaOb, double z, double M);
double p_eta_cont(double Lx, double Texp, double etaOb, double z, double M);
double Lx_eta(double etaOb, double Texp, double z);

/* functions for theoretical number count integration */
double aux_counts(double zmin, double zmax, int index_q);
double aux_dz_integrand_gsl(double z, void *params);
double aux_dz_integrand_scipy(int n, double args[n]);
double aux_dz_integrand(double z, double qmin, double qmax);
double aux_dlnM_integrand_gsl(double lnM, void *params);
double aux_dlnM_integrand(int n, double args[n]);

/* Monte Carlo tehoretical number count integration */
double int_etaOb(double Lx, double Texp, double z, double M);
double MC_counts(double etaOb, double z, double Texp);
double aux_MC_counts(double *variables, size_t dim, void *params);

/* struct for integration parameters */
struct BPARAM{
	double lnM;
	double etaOb;
	double Texp;
	double lnLx;
	double z;
	double lnLammin;
	double lnLammax;

};

/* get current cosmological parameters from MontePython 
 * -----------------------
 * read Hubble parameter h, Omega_m, Omega_k, Omega_nu, w0/wa for dark energy equation of state
 */
void setup_cosmo(double cosmoh,
				 double cosmoOmega_m,
				 double cosmoOmega_k,
				 double cosmoOmega_nu,
				 double cosmow0,
				 double cosmowa){
	h = cosmoh;
	Omega_m = cosmoOmega_m;
	Omega_k = cosmoOmega_k;
	Omega_nu = cosmoOmega_nu;
	w0 = cosmow0;
	wa = cosmowa;
}


/* get binned cluster data from python and save in data_counts array */
void setup_data(double *data){

	for (int index_Lx = 0; index_Lx < netaObbins; ++index_Lx)
	{
		for (int index_z = 0; index_z < nzbins; ++index_z)
		{
			data_counts[index_Lx][index_z] = data[index_Lx * (nzbins) + index_z];
		}
	}
}


/* setup sigma(M,z) spline */
void setup_sigma2d_spline(double *min, double *zin, double *sigmain){
	
	// build lnsigma(lnR,z) spline
	for (int j = 0; j < zstep; ++j)
	{
		z_array[j] = zin[j];

		for (int i = 0; i < Mstep; ++i)
			{
				sigma2d_array[j+i*zstep] = log(sigmain[j+i*zstep]);
			}

	}

	// initialise lnR array
	for (int i = 0; i < Mstep; ++i)
	{
		lnM_array[i] = log(min[i]);
		lnR_array[i] = log(pow((3. * min[i] / 4. / M_PI / rhocrit / (Omega_m - Omega_nu)),1./3.));
		// fprintf(stdout, "%f \n", lnR_array[i]);
	}

	gsl_spline2d_init(sigma_lnR_z_spline, z_array, lnR_array, sigma2d_array, zstep, Mstep);
}


/* calculate E(z) = H(z)/H0 without radiation and assuming w_DE = w0 + (1.-a) * wa */
double E(double z){
	double scale = 1.+z;
    double w_eff = w0 + (1. - scale) * wa;
    double Esq = Omega_m * pow(scale,3) + (1.-Omega_k-Omega_m) * pow(scale,-3*(1.+w_eff) ) + Omega_k * pow(scale,2);
	// double Esq=Omega_m * pow(scale,3)+(1.-Omega_m);
	return sqrt(Esq);
}

/* calculate Omega_m(z) */
double Omega_m_z(double z){
	return(Omega_m * pow((1.+z),3) / pow(E(z),2));
}

/* differential cosmological volume element dV/dz in Mpc**3 */
double dvolume(double z){

	double result;

	result=2997.9/h/E(z)*pow(distance(z),2);

	return(result);
	
}

/* volume for computation of N(Lx^ob), no integral over z */
double volumez(double z,  double Texp){

	double dA_sky = interp1d(Texp)/41253.; 
	double V = dvolume(z) * dA_sky;//((4*M_PI/3)*(pow(distance(z + dz),3)) - (4*M_PI/3)*(pow(distance(z),3))) * dA_sky;
	return V;
}	
	

/* comoving distance in Mpc */
double distance(double z){

	double dist,result,error;
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(NEVAL);
	gsl_function F;
	
	F.function = &aux_distance;
	F.params = NULL;
	
	gsl_integration_qag(&F,0.0,z,0.0,1e-4,NEVAL,GSL_INTEG_GAUSS15,w,&result,&error);
	
	gsl_integration_workspace_free(w);

	// take care of curvature
    if (Omega_k < 0){
      dist = 2997.9/h/sqrt(-Omega_k) * sin(sqrt(-Omega_k) * result/2997.9*h);
    }
    if (Omega_k > 0){
      dist = 2997.9/h/sqrt(Omega_k) * sinh(sqrt(Omega_k) * result/2997.9*h);
    }
    if (Omega_k == 0){
      dist = result;
    }

	return(result);
}

/* integrand for comoving distance */
double aux_distance(double z, void *params){

	double result;
	// D_hubble = c/H_0 = 2997.9/h Mpc
	result = 2997.9/E(z)/h;

	return(result);
}

/* setup splines */
void spline_init(char *path){

	// allocate spline memory for Tinker splines
	norm_spline = gsl_spline_alloc(gsl_interp_cspline,9);
	acc_norm_spline = gsl_interp_accel_alloc();

	a_spline = gsl_spline_alloc(gsl_interp_cspline,9);
	acc_a_spline = gsl_interp_accel_alloc();

	b_spline = gsl_spline_alloc(gsl_interp_cspline,9);
	acc_b_spline = gsl_interp_accel_alloc();

	c_spline = gsl_spline_alloc(gsl_interp_cspline,9);
	acc_c_spline = gsl_interp_accel_alloc();

	// values from Tinker08
	double delta_vec[9] = { log10(200), log10(300), log10(400), log10(600), log10(800), log10(1200), log10(1600), log10(2400), log10(3200) };
	double norm_vec[9] = { 0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260 };
	double a_vec[9] = { 1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66 };
	double b_vec[9] = { 2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41 };
	double c_vec[9] = { 1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44 };

	// setup splines
	gsl_spline_init(norm_spline,delta_vec,norm_vec,9);
	gsl_spline_init(a_spline,delta_vec,a_vec,9);
	gsl_spline_init(b_spline,delta_vec,b_vec,9);
	gsl_spline_init(c_spline,delta_vec,c_vec,9);

	sigma_lnR_z_spline = gsl_spline2d_alloc(gsl_interp2d_bicubic, zstep, Mstep);

	acc_z_lnR_lnR = gsl_interp_accel_alloc();
	acc_z_lnR_z = gsl_interp_accel_alloc();


}

/* reads area_flux table */
void setup_areaflux(char *path){

	FILE *file;
	char fname[256];

	//fprintf(stdout, "%s/%s\n",path,"area_flux.dat");
	sprintf(fname,"%s/%s",path,"area_flux.dat");
	file = fopen(fname,"r");

	for (int j = 0; j < 100; ++j)
	{
		fscanf(file," %le %le\n", &Afz[j][0], &Afz[j][1]);

	}
	for (int i = 0; i < 100; ++i)
	{
		Texp[i] = 4*1.2e-11 / pow(10,Afz[i][0]);
		if (Texp[i] < 5){dA_dTexp[i] = 0;}
		else {dA_dTexp[i] = Afz[i][1];} /* degree */
		
	}	
}

/* reads k-correction table */
void setup_kcorr(char *path){

	FILE *file2;
	char fname2[256];
	
	sprintf(fname2,"%s/%s",path,"kcorr.dat");
	file2 = fopen(fname2,"r");

	for (int j = 0; j < 61; ++j)
	{
		for (int i = 0; i < 61; ++i)
		{
			fscanf(file2," %le\n", &Kcor[j][i]);	
		}
	}
}

/* reads detection probability table */
void setup_detecProb(char *path){

	FILE *file3;
	char fname3[256];
	
	sprintf(fname3,"%s/%s",path,"all_prob.cat");
	file3 = fopen(fname3,"r");

	for (int zm = 0; zm < 231; ++zm)
	{
		for (int e = 0; e < 18; ++e)
		{
			fscanf(file3," %le\n", &probDec[zm][e]);
		}
	}
/* Convert 2D in 3D [z][M][etaOb] */
	for (int i = 0; i < 11; ++i)
	{
		int j = i*21;
		int k = j + 21;

		for (int a = j; a < k; ++a)
		{
			for (int b = 0; b < 18; ++b)
			{
				probDec3d[i][a-j][b] = probDec[a][b];
			}
		}
	}
}

/* 1D linear interpolation routine, requires Texp[100] dA_dTexp[100] and  from setup_areaflux */
double interp1d(double T_exp){

	// check boundaries
	if (T_exp > Texp_max){T_exp = Texp_max;}
	if (T_exp < Texp_min){T_exp = Texp_min;}

	double result;
	double nf; //fractional parts
	int n; //integer parts
	double slope;

	//get Texp index
	int diff = 9999;
	int ibest = -1;
	for (int i = 0; i < 100; i++)
	{	
		if (abs(Texp[i] - T_exp) < diff)
		{
			ibest = i;
			diff = abs(Texp[i] - T_exp);
		}
	}
	n = ibest;
	
	slope = (dA_dTexp[n] - dA_dTexp[n-1])/(Texp[n] - Texp[n-1]);

	//calculate interpolated value
	result = slope*(T_exp - Texp[n-1]) + dA_dTexp[n-1]; 
	return (result);
}

/* 3d linear interpolation routine, requires filled table probDec3d[z][M][etaOb] */
double interp3d(double z, double M, int etaOb){

	double log10M = log10(M);
	double z_vec[11] = { 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60 };
	double log10M_vec[21] = { 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0, 15.1, 15.2, 15.3, 15.4, 15.5 };
	double etaOb_vec1[18] = { 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };

	// check boundaries
	if (z > z_max){z = z_max;}
	if (z < z_min){z = z_min;}
	if (log10M > log10M_max){log10M = log10M_max;}
	if (log10M < log10M_min){log10M = log10M_min;}
	if (etaOb > etaOb_max){etaOb = etaOb_max;}
	if (etaOb < etaOb_min){etaOb = etaOb_min;}

	double result;
	double lf, nf, mf, c00, c01, c10, c11, c0, c1; //fractional parts
	int l, m, n; //integer parts

	//get lower index
	l = (int) ((z - z_min) / (0.05));
	m = (int) ((log10M - log10M_min)/(0.1));
	n = (int) ((etaOb - etaOb_min) / (1.));
	
	lf = (z - z_vec[l]) / (z_vec[l+1] - z_vec[l]);
	mf = (log10M - log10M_vec[m]) / (log10M_vec[m+1] - log10M_vec[m]);
	nf = (etaOb - etaOb_vec1[n]) / (etaOb_vec1[n+1] - etaOb_vec1[n]);
	
	c00 = probDec3d[l][m][n]*(1- lf) + probDec3d[l+1][m][n]*lf;
	c01 = probDec3d[l][m][n+1]*(1- lf) + probDec3d[l+1][m][n+1]*lf;
	c10 = probDec3d[l][m+1][n]*(1- lf) + probDec3d[l+1][m+1][n]*lf;
	c11 = probDec3d[l][m+1][n+1]*(1- lf) + probDec3d[l+1][m+1][n+1]*lf;

	c0 = c00 * (1.-mf) + c10*mf;
	c1 = c01 * (1.-mf) + c11*mf;

	//calculate interpolated value
	result = c0 * (1. - nf) + c1*nf;
	return (result);
}

/* Scaling relation M, z --> Lx */
double transfM(double M, double z){

	// M must be M200c
    	double zcor2xlum = sqrt(Omegam * (pow((1. + z), 3)) + Omegak * (pow((1. + z), 2)) + Omegal);
    	double t = (log10( M * zcor2xlum ) - 13.56) / 1.69;  //t = 10**t
    	double xl = (pow(10,(43.48 + 2.63 * (t - 0.48) - 0.15 + 0.07))) * zcor2xlum;

	return (xl);  
}

/* Scaling relation M --> Lambda */
double transfML(double M){
    return (log(30.) + (1./1.3) * log(M/(pow(10., 14.2)))); //Simet+16 for M200c
}

/* Relation Lx_true, Texp, z --> etaTrue */
double transf_etaTrue(double Ltrue, double Texp, double z, double M){

	double t, riw, rjw, akcor, vcor, xl, lumdist, Dl, fl, etaTrue;
	int iw, jw;

    // from M to temperature
	double zcor2xlum = sqrt(Omegam * (pow((1. + z), 3)) + Omegak * (pow((1. + z), 2)) + Omegal);
	t = pow(10,((log10( M * zcor2xlum ) - 13.56) / 1.69));  //0.2 + 6*(pow(10,((log10(Ltrue)-44.45)/2.1)));

    // k correction
	riw = ((log10(t)+1.5)/0.05);
	iw = (int) riw;

	if (iw <= 0){iw = 0;}
	if (iw >= 61){iw = 60;}

	rjw = z/0.05 - 1;
	jw = (int) rjw;

	if (jw <= 0){jw = 0;}
	if (jw >= 61){jw = 60;}

	akcor = Kcor[iw][jw];

	if ((0 < iw) && (iw < 60)){
		if (riw > iw){
			akcor = akcor + (riw - iw) * (Kcor[iw+1][jw] - Kcor[iw][jw]);
		}
		else{
			akcor = akcor + (riw-iw)*(Kcor[iw-1][jw] - Kcor[iw][jw]);
		}
	}
	if (rjw > jw){
		if (jw < 61){
			akcor = akcor + (rjw - jw)*(Kcor[iw][jw+1] - Kcor[iw][jw]);
        	}
		else{
            		if (jw > 1){akcor = akcor + (rjw - jw)*(Kcor[iw][jw-1] - Kcor[iw][jw]);}
		}
	}
        
	if (akcor > 10.){akcor = 10.;}

	vcor = 1. + 0.07 / (z - 0.05); // assume 1 just for know
	xl = Ltrue;
	lumdist = distance(z) * (1.+z);
	Dl = lumdist * 3.0857e24;

     // apply k and v correction and derive flux from Lx
	fl = xl / (akcor*vcor*4.*M_PI*Dl*Dl);

 	etaTrue = (fl / 1.2e-11) * Texp;

	return (etaTrue);
}

/* Log-normal scatter Lx_true - M_true */
double P_L_M(double Mtrue, double Ltrue, double z){
	
	double sigL, P;
	sigL = 0.46 * (1. - 0.61 * z); //0.4;
	P = 1. / (sqrt(2. * M_PI) * Ltrue * sigL) * exp(-((log(Ltrue) - log(transfM(Mtrue, z)))*(log(Ltrue) - log(transfM(Mtrue, z))))/(2.*(sigL*sigL)));
	
	return (P);
}

/* new scatter in Lx-M including Alexi's covariance, instead of P_L_M */
double P_L_M_cov(double Mtrue, double Ltrue, double z){

    double sglx, xx1, rcut, mcut, srm, srmt, vcov, r, rn, x, rm, rsgm, xrc, xxrc, wrc, P;

    sglx = 0.46 * (1. - 0.61 * z); //0.6;
    xx1 = log(Ltrue) - log(transfM(Mtrue, z));

    rcut = 30;
    mcut = 1.58e14/h * (pow((rcut/30.),1.33)); //2.21e+14 * (pow((rcut/40.),1.33));

    if (z <= 0.3){srm = 0.1;}
    if (z > 0.3){srm = 4.33 * pow(z, 4.17);}
    srmt = sqrt(pow(srm, 2) + pow(0.35, 2));

    vcov = -0.2 * 0.35/srmt;
    r = vcov;
    rn = sqrt(1 - r*r);
    x = log(Mtrue) - log(mcut);
    rm = 1.3 * sqrt(srm*srm + 0.35*0.35);
    rsgm = x/rm;
    xrc = rsgm/rn;

    xxrc = xrc - r * xx1/sglx/rn;

    wrc = 1. - 0.5*(gsl_sf_erfc(xxrc/sqrt(2.0)));

    P = 1. / (sqrt(2. * M_PI) * Ltrue * sglx) * exp(-(xx1*xx1)/(2.*(sglx*sglx)));

    return (P * wrc);
}


/* Complementary error function Lambda_true - M_true */
double errorF(double lnLambdamin, double lnLambdamax, double M, double z){

	double lnLamM, sigLM, sigT, x1, x2, result;
	lnLamM = transfML(M);
	// before I was using sig = 0.3
	if (z <= 0.3){sigLM = 0.1;}
	if (z > 0.3){sigLM = 4.33*pow(z, 4.17);}
	sigT = sqrt(pow(sigLM, 2) + pow(0.35, 2));
	x1 = (lnLambdamin - lnLamM) / (sqrt(2) * sigT);
	x2 = (lnLambdamax - lnLamM) / (sqrt(2) * sigT);
	result = (1./2.) * (gsl_sf_erfc(x1) - gsl_sf_erfc(x2));
	return (result);
}

/* Poisson distribution */
double p_eta(double Ltrue, double Texp, unsigned int etaOb, double z, double M){
	
	double etaTrue, poisson_pdf;
	etaTrue = transf_etaTrue(Ltrue, Texp, z, M);
	poisson_pdf = gsl_ran_poisson_pdf(etaOb, etaTrue);
	return (poisson_pdf);

}
/* normal approximation for the Poisson distribution */
double p_eta_norm(double Lx, double Texp, double etaOb, double z, double M){
	
	double etaTrue, poisson_pdf, sigma, y;
	etaTrue = transf_etaTrue(Lx, Texp, z, M);
	sigma = sqrt(etaTrue);	
	y = (etaOb - etaTrue);
	poisson_pdf = gsl_ran_gaussian_pdf(y, sigma);
	return (poisson_pdf);
}

/* continuous approximation for the Poisson distribution */
double p_eta_cont(double Lx, double Texp, double etaOb, double z, double M){
	
	double etaTrue, prob, sigma, y, gam_arg;  
	
	etaTrue = transf_etaTrue(Lx, Texp, z, M);
	sigma = sqrt(etaTrue);	
	y = (etaOb - etaTrue);
	gam_arg = etaOb+1.0;
	
	if (etaTrue < 100.0)
	{
		if (etaOb > 150.0){prob = 0.0;}
		else{prob = (exp(-etaTrue) * pow(etaTrue, etaOb)) / (gsl_sf_gamma(gam_arg));}
	}
	else{prob = gsl_ran_gaussian_pdf(y, sigma);}
	
	return (prob);
}


/* halo multiplicity function from Tinker et al, 2008 */
double ftinker2008(double sigma, double Delta, double z)
{
	double norm,a,b,c;
	double result;

	double aux_alpha = -pow((0.75 / log10(Delta/75.)),1.2);
	double alpha = pow(10.,aux_alpha);
	// catch values outside of interpolation range 
	if (Delta > 3200.){Delta = 3200.;}
	if (Delta < 200.){Delta = 200.;}

	norm = gsl_spline_eval(norm_spline,log10(Delta),acc_norm_spline) * pow(1.+z,-0.14);
	a = gsl_spline_eval(a_spline,log10(Delta),acc_a_spline) * pow(1.+z,-0.06);
	b = gsl_spline_eval(b_spline,log10(Delta),acc_b_spline) * pow(1.+z,-alpha);
	c = gsl_spline_eval(c_spline,log10(Delta),acc_c_spline);

	result = norm * (1. + pow(sigma/b,-a)) * exp(-c/pow(sigma,2));
	return(result);
}

/* 
 * halo mass function from Tinker et al, 2008 
 * -----------------------
 * Requires:	spline_init() and setup_sigma_2d_spline() to be called first
 * input:		lnM in M_sun/h
 *				z redshift
 *				Delta overdensity of clusters compared to rho_crit
 * Output:		dndlnM number of clusters per Mpc/h**3 and lnM
 */

double dndlnM(double lnM, double z, double Delta){
	double result;
	double deriv;
	double M, R, sigma, f;
	double rho_m;

	M = exp(lnM);
	rho_m = rhocrit * (Omega_m - Omega_nu);

	Delta = Delta / Omega_m_z(z);
	R = pow((3. * M / 4. / M_PI / rho_m),(1./3.));

	sigma = exp(gsl_spline2d_eval(sigma_lnR_z_spline, z, log(R), acc_z_lnR_z, acc_z_lnR_lnR));
	f = ftinker2008(sigma, Delta, z);

	// deriv = gsl_spline_eval_deriv(sigma_R_spline,log(R),acc_sigma_R_spline);
	deriv = gsl_spline2d_eval_deriv_y(sigma_lnR_z_spline,z,log(R),acc_z_lnR_z,acc_z_lnR_lnR);

	result = -f * deriv / 4. / M_PI / pow(R,3);
	return(result);
}

double Lx_eta(double etaOb, double Texp, double z){

	double vcor, Dl, fl, lx_init, k_init, lx, t_, riw, rjw, akcor; 
	int iw, jw;

	vcor = 1. + 0.07 / (z - 0.05);
	Dl = (distance(z) * (1.+z)) * 3.0857e24;

	fl = (etaOb/Texp) * 1.2e-11;

	lx_init = 1.;
	k_init = 1.6;
	lx = fl * (k_init*vcor*4.*M_PI*Dl*Dl);

	while(abs(lx - lx_init) > (0.01*lx)){
		lx_init = lx;
		t_ =  pow(10,((log10(lx_init) - 43.48 + 0.15 - 0.07)/2.63 + 0.48));

		riw = ((log10(t_)+1.5)/0.05);

		iw = (int) riw;

		if (iw <= 0){iw = 0;}
		if (iw >= 61){iw = 60;}

		rjw = z/0.05 - 1;
		jw = (int) rjw;

		if (jw <= 0){jw = 0;}
		if (jw >= 61){jw = 60;}

		akcor = Kcor[iw][jw];

		if ((0 < iw) && (iw < 60)){
			if (riw > iw){
				akcor = akcor + (riw - iw) * (Kcor[iw+1][jw] - Kcor[iw][jw]);
			}
			else{
				akcor = akcor + (riw-iw)*(Kcor[iw-1][jw] - Kcor[iw][jw]);
			}
		}
		if (rjw > jw){
			if (jw < 61){
				akcor = akcor + (rjw - jw)*(Kcor[iw][jw+1] - Kcor[iw][jw]);
			}
			else{
		    		if (jw > 1){akcor = akcor + (rjw - jw)*(Kcor[iw][jw-1] - Kcor[iw][jw]);}
			}
		}
		
		if (akcor > 10.){akcor = 10.;}

		lx = fl * (akcor*vcor*4.*M_PI*Dl*Dl);
	}

	return lx;
}

/* Monte Carlo integration for binned theoretical number counts */
double MC_counts(double etaOb, double z, double Texp){

	double result,aux,error;
	// dimension of the integral
	int dim = 2;
	// integral boundaries: lnM, lnLtrue, lnTexp, lnetaOb, z
	double lowerlimits[2] = { lnM_min, lnLx_min };
	double upperlimits[2] = { lnM_max, lnLx_max };
	struct BPARAM *bparam = (struct BPARAM *)calloc(3,sizeof(struct BPARAM));
	bparam-> etaOb = etaOb;
	bparam-> z = z;
 	bparam-> Texp = Texp;

	const gsl_rng_type *T;
	gsl_rng *r;
	gsl_monte_function F = { &aux_MC_counts, dim, bparam};
	size_t calls = 1e5;  // number of calls
  
	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
  
	gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(dim);
  
	// warm up with 1000 samples
	 gsl_monte_vegas_integrate (&F, lowerlimits, upperlimits, dim, 1000, r, s, &aux, &error);
  
	// for real
	gsl_monte_vegas_integrate (&F, lowerlimits, upperlimits, dim, calls, r, s, &result, &error);
	gsl_monte_vegas_free (s);
	gsl_rng_free(r);
  
	return(result);
}


/* integrand for theoretical number count Monte Carlo integration */
double aux_MC_counts(double *variables, size_t dim, void *params){

	double result;
	double lnM, lnLx;
	//double lnLammin = log(30);
	//double lnLammax = log(350);

	struct BPARAM *bparam = (struct BPARAM *)params; 
	double Texp = bparam->Texp;
	double etaOb = bparam->etaOb;
	double z = bparam->z;

	lnM = variables[0];
	lnLx = variables[1];


	result = pow(h,3) * volumez(z, Texp) * dndlnM(lnM, z, 200.) * P_L_M_cov(exp(lnM)/h, exp(lnLx)/(pow(h,2)), z) * exp(lnLx)/(pow(h,2)) * interp3d(z, exp(lnM)/h, etaOb) * p_eta_cont(exp(lnLx)/(pow(h,2)), Texp, etaOb, z, exp(lnM)/h); //* errorF(lnLammin, lnLammax, exp(lnM), z);

	return(result);
}

/* calculates log-likelihood in bins of Lx and z */
double loglkl(){

	FILE *fp;
	fp = fopen("N_Lx_new.txt", "w+");

	double result;
	double etaOb, z_, Texp, LxOb;
	result = 0.;

	#pragma omp parallel for private(z_, Texp, etaOb) shared(result) schedule(dynamic)

	for (int index_z = 0; index_z < 10; ++index_z)
	{
		z_ = z_vec[index_z];

		for (int index_T = 0; index_T < 70; ++index_T)
		{
			Texp = 4.*1.2e-11 / pow(10,Afz[index_T][0]);


			for (int index_etaOb = 0; index_etaOb < 97; ++index_etaOb)
			{
				etaOb = etaOb_vec[index_etaOb];
	
				theory_counts[index_z][index_T][index_etaOb] = MC_counts(etaOb, z_, Texp);

				LxOb = Lx_eta(etaOb, Texp, z_);

				#pragma omp atomic
				result += theory_counts[index_z][index_T][index_etaOb];
				//result1 = MC_counts(lnLammin, lnLammax, zmin, zmax); 
				//theory_counts[index_z][index_Lx];
				//printf("bin: %le %le %le %le\n", zmin,zmax, lnLammin, lnLammax);
				printf("%le %le %le %le %le\n", z_, Texp, etaOb, LxOb, theory_counts[index_z][index_T][index_etaOb]);
				fprintf(fp, "%le %le %le %le %le\n", z_, Texp, etaOb, LxOb, theory_counts[index_z][index_T][index_etaOb]);
				//result += (data_counts[index_Lx][index_z] * log(theory_counts[index_Lx][index_z]) - theory_counts[index_Lx][index_z] - gsl_sf_lnfact(data_counts[index_Lx][index_z]));
			}
		
	        }
	}
	printf("%le\n", result);
	fclose(fp);
	return(result);
}




