//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef TABLE_HELPER_HPP
#define TABLE_HELPER_HPP

#include <vector>
#include <string>
#include <boost/version.hpp>
#include <boost/lexical_cast.hpp>

//
// Also include headers for whatever else we may be testing:
//
#ifdef TEST_LIBSTDCXX
#include <tr1/cmath>
#include <stdexcept>
#endif
#ifdef TEST_GSL
#include <gsl/gsl_sf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_version.h>

void gsl_handler(const char * reason, const char * file, int line, int gsl_errno)
{
   if(gsl_errno == GSL_ERANGE) return; // handle zero or infinity in our test code.
#ifdef DISTRIBUTIONS_TEST
   return;
#else
   throw std::domain_error(reason);
#endif
}

struct gsl_error_handler_setter
{
   gsl_error_handler_t * old_handler;
   gsl_error_handler_setter()
   {
      old_handler = gsl_set_error_handler(gsl_handler);
   }
   ~gsl_error_handler_setter()
   {
      gsl_set_error_handler(old_handler);
   }
};

static const gsl_error_handler_setter handler;

#endif

#ifdef TEST_RMATH
// Rmath overloads ftrunc, leading to strange errors from GCC unless we include this:
#include <boost/math/special_functions.hpp>
#define MATHLIB_STANDALONE
#include <Rmath.h>
#endif

#ifdef TEST_DCDFLIB
extern "C" {
   extern void cdfbet(int*, double*, double*, double*, double*, double*, double*, int*, double*);
   extern void cdfbin(int*, double*, double*, double*, double*, double*, double*, int*, double*);
   extern void cdfchi(int*, double*, double*, double*, double*, int*, double*);
   extern void cdfchn(int*, double*, double*, double*, double*, double*, int*, double*);
   extern void cdff(int*, double*, double*, double*, double*, double*, int*, double*);
   extern void cdffnc(int*, double*, double*, double*, double*, double*, double*, int*s, double*);
   extern void cdfgam(int*, double*, double*, double*, double*, double*, int*, double*);
   extern void cdfnbn(int*, double*, double*, double*, double*, double*, double*, int*, double*);
   extern void cdfnor(int*, double*, double*, double*, double*, double*, int*, double*);
   extern void cdfpoi(int*, double*, double*, double*, double*, int*, double*);
   extern void cdft(int*, double*, double*, double*, double*, int*, double*);
   //extern void cdftnc(int*, double*, double*, double*, double*, double*, int*, double*);
}

inline double dcdflib_beta_cdf(double x, double a, double b)
{
   int what = 1;
   int status = 0;
   double p, q, bound, y(1-x);
   cdfbet(&what, &p, &q, &x, &y, &a, &b, &status, &bound);
   return p;
}

inline double dcdflib_beta_quantile(double p, double a, double b)
{
   int what = 2;
   int status = 0;
   double x, y, bound, q(1 - p);
   cdfbet(&what, &p, &q, &x, &y, &a, &b, &status, &bound);
   return x;
}

inline double dcdflib_binomial_cdf(double x, double s, double sf)
{
   int what = 1;
   int status = 0;
   double p, q, bound, sfc(1-sf);
   cdfbin(&what, &p, &q, &x, &s, &sf, &sfc, &status, &bound);
   return p;
}

inline double dcdflib_binomial_quantile(double p, double s, double sf)
{
   int what = 2;
   int status = 0;
   double x, bound, q(1 - p), sfc(1-sf);
   cdfbin(&what, &p, &q, &x, &s, &sf, &sfc, &status, &bound);
   return x;
}

inline double dcdflib_chi_cdf(double x, double df)
{
   int what = 1;
   int status = 0;
   double p, q, bound;
   cdfchi(&what, &p, &q, &x, &df, &status, &bound);
   return p;
}

inline double dcdflib_chi_quantile(double p, double df)
{
   int what = 2;
   int status = 0;
   double x, bound, q(1 - p);
   cdfchi(&what, &p, &q, &x, &df, &status, &bound);
   return x;
}

inline double dcdflib_chi_n_cdf(double x, double df, double nc)
{
   int what = 1;
   int status = 0;
   double p, q, bound;
   cdfchn(&what, &p, &q, &x, &df, &nc, &status, &bound);
   return p;
}

inline double dcdflib_chi_n_quantile(double p, double df, double nc)
{
   int what = 2;
   int status = 0;
   double x, bound, q(1 - p);
   cdfchn(&what, &p, &q, &x, &df, &nc, &status, &bound);
   return x;
}

inline double dcdflib_f_cdf(double x, double df1, double df2)
{
   int what = 1;
   int status = 0;
   double p, q, bound;
   cdff(&what, &p, &q, &x, &df1, &df2, &status, &bound);
   return p;
}

inline double dcdflib_f_quantile(double p, double df1, double df2)
{
   int what = 2;
   int status = 0;
   double x, bound, q(1 - p);
   cdff(&what, &p, &q, &x, &df1, &df2, &status, &bound);
   return x;
}

inline double dcdflib_f_n_cdf(double x, double df1, double df2, double nc)
{
   int what = 1;
   int status = 0;
   double p, q, bound;
   cdffnc(&what, &p, &q, &x, &df1, &df2, &nc, &status, &bound);
   return p;
}

inline double dcdflib_f_n_quantile(double p, double df1, double df2, double nc)
{
   int what = 2;
   int status = 0;
   double x, bound, q(1 - p);
   cdffnc(&what, &p, &q, &x, &df1, &df2, &nc, &status, &bound);
   return x;
}

inline double dcdflib_gamma_cdf(double x, double shape, double scale)
{
   int what = 1;
   int status = 0;
   double p, q, bound;
   scale = 1 / scale;
   cdfgam(&what, &p, &q, &x, &shape, &scale, &status, &bound);
   return p;
}

inline double dcdflib_gamma_quantile(double p, double shape, double scale)
{
   int what = 2;
   int status = 0;
   double x, bound, q(1 - p);
   scale = 1 / scale;
   cdfgam(&what, &p, &q, &x, &shape, &scale, &status, &bound);
   return x;
}

inline double dcdflib_nbin_cdf(double x, double r, double sf)
{
   int what = 1;
   int status = 0;
   double p, q, bound, sfc(1 - sf);
   cdfnbn(&what, &p, &q, &x, &r, &sf, &sfc, &status, &bound);
   return p;
}

inline double dcdflib_nbin_quantile(double p, double r, double sf)
{
   int what = 2;
   int status = 0;
   double x, bound, q(1 - p), sfc(1 - sf);
   cdfnbn(&what, &p, &q, &x, &r, &sf, &sfc, &status, &bound);
   return x;
}

inline double dcdflib_norm_cdf(double x, double mean, double sd)
{
   int what = 1;
   int status = 0;
   double p, q, bound;
   cdfnor(&what, &p, &q, &x, &mean, &sd, &status, &bound);
   return p;
}

inline double dcdflib_norm_quantile(double p, double mean, double sd)
{
   int what = 2;
   int status = 0;
   double x, bound, q(1 - p);
   cdfnor(&what, &p, &q, &x, &mean, &sd, &status, &bound);
   return x;
}

inline double dcdflib_poisson_cdf(double x, double param)
{
   int what = 1;
   int status = 0;
   double p, q, bound;
   cdfpoi(&what, &p, &q, &x, &param, &status, &bound);
   return p;
}

inline double dcdflib_poisson_quantile(double p, double param)
{
   int what = 2;
   int status = 0;
   double x, bound, q(1 - p);
   cdfpoi(&what, &p, &q, &x, &param, &status, &bound);
   return x;
}

inline double dcdflib_t_cdf(double x, double param)
{
   int what = 1;
   int status = 0;
   double p, q, bound;
   cdft(&what, &p, &q, &x, &param, &status, &bound);
   return p;
}

inline double dcdflib_t_quantile(double p, double param)
{
   int what = 2;
   int status = 0;
   double x, bound, q(1 - p);
   cdft(&what, &p, &q, &x, &param, &status, &bound);
   return x;
}
/*
inline double dcdflib_t_n_cdf(double x, double param, double nc)
{
   int what = 1;
   int status = 0;
   double p, q, bound;
   cdftnc(&what, &p, &q, &x, &param, &nc, &status, &bound);
   return p;
}

inline double dcdflib_t_n_quantile(double p, double param, double nc)
{
   int what = 2;
   int status = 0;
   double x, bound, q(1 - p);
   cdftnc(&what, &p, &q, &x, &param, &nc, &status, &bound);
   return x;
}
*/
#endif

extern std::vector<std::vector<double> > data;

void report_execution_time(double t, std::string table, std::string row, std::string heading);
std::string get_compiler_options_name();

inline std::string boost_name()
{
   return "boost " + boost::lexical_cast<std::string>(BOOST_VERSION / 100000) + "." + boost::lexical_cast<std::string>((BOOST_VERSION / 100) % 1000);
}

inline std::string compiler_name()
{
#ifdef COMPILER_NAME
   return COMPILER_NAME;
#else
   return BOOST_COMPILER;
#endif
}

inline std::string platform_name()
{
#ifdef _WIN32
   return "Windows x64";
#else
   return BOOST_PLATFORM;
#endif
}

#endif // TABLE_HELPER_HPP

