//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_ERF_OTHER_HOOKS_HPP
#define BOOST_MATH_TEST_ERF_OTHER_HOOKS_HPP

#ifdef TEST_CEPHES
namespace other{
extern "C" {
   double jv(double, double);
   float jvf(float, float);
   long double jvl(long double, long double);
   double yv(double, double);
   float yvf(float, float);
   long double yvl(long double, long double);
}
inline float cyl_bessel_j(float a, float x)
{ return jvf(a, x); }
inline double cyl_bessel_j(double a, double x)
{ return jv(a, x); }
inline long double cyl_bessel_j(long double a, long double x)
{
#ifdef _MSC_VER
   return jv((double)a, x); 
#else
   return jvl(a, x); 
#endif
}
inline float cyl_neumann(float a, float x)
{ return yvf(a, x); }
inline double cyl_neumann(double a, double x)
{ return yv(a, x); }
inline long double cyl_neumann(long double a, long double x)
{
#ifdef _MSC_VER
   return yv((double)a, x); 
#else
   return yvl(a, x); 
#endif
}
}
#define TEST_OTHER
#endif

#ifdef TEST_GSL
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_message.h>

namespace other{
inline float cyl_bessel_j(float a, float x)
{ return (float)gsl_sf_bessel_Jnu(a, x); }
inline double cyl_bessel_j(double a, double x)
{ return gsl_sf_bessel_Jnu(a, x); }
inline long double cyl_bessel_j(long double a, long double x)
{ return gsl_sf_bessel_Jnu(a, x); }

inline float cyl_bessel_i(float a, float x)
{ return (float)gsl_sf_bessel_Inu(a, x); }
inline double cyl_bessel_i(double a, double x)
{ return gsl_sf_bessel_Inu(a, x); }
inline long double cyl_bessel_i(long double a, long double x)
{ return gsl_sf_bessel_Inu(a, x); }

inline float cyl_bessel_k(float a, float x)
{ return (float)gsl_sf_bessel_Knu(a, x); }
inline double cyl_bessel_k(double a, double x)
{ return gsl_sf_bessel_Knu(a, x); }
inline long double cyl_bessel_k(long double a, long double x)
{ return gsl_sf_bessel_Knu(a, x); }

inline float cyl_neumann(float a, float x)
{ return (float)gsl_sf_bessel_Ynu(a, x); }
inline double cyl_neumann(double a, double x)
{ return gsl_sf_bessel_Ynu(a, x); }
inline long double cyl_neumann(long double a, long double x)
{ return gsl_sf_bessel_Ynu(a, x); }
}
#define TEST_OTHER
#endif

#ifdef TEST_OTHER
namespace other{
   boost::math::concepts::real_concept cyl_bessel_j(boost::math::concepts::real_concept, boost::math::concepts::real_concept){ return 0; }
   boost::math::concepts::real_concept cyl_bessel_i(boost::math::concepts::real_concept, boost::math::concepts::real_concept){ return 0; }
   boost::math::concepts::real_concept cyl_bessel_k(boost::math::concepts::real_concept, boost::math::concepts::real_concept){ return 0; }
   boost::math::concepts::real_concept cyl_neumann(boost::math::concepts::real_concept, boost::math::concepts::real_concept){ return 0; }
}
#endif


#endif



