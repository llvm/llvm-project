//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_ERF_OTHER_HOOKS_HPP
#define BOOST_MATH_TEST_ERF_OTHER_HOOKS_HPP

#ifdef TEST_NATIVE
namespace other{
inline float erf(float a)
{
   return ::erff(a);
}
inline float erfc(float a)
{
   return ::erfcf(a);
}
inline double erf(double a)
{
   return ::erf(a);
}
inline double erfc(double a)
{
   return ::erfc(a);
}
inline long double erf(long double a)
{
   return ::erfl(a);
}
inline long double erfc(long double a)
{
   return ::erfcl(a);
}
}
#define TEST_OTHER
#endif

#ifdef TEST_CEPHES
namespace other{
extern "C" {
   double erf(double);
   float erff(float);
   long double erfl(long double);
}
inline float erf(float a)
{ return erff(a); }
inline long double erf(long double a)
{
#ifdef _MSC_VER
   return erf((double)a); 
#else
   return erfl(a); 
#endif
}
extern "C" {
   double erfc(double);
   float erfcf(float);
   long double erfcl(long double);
}
inline float erfc(float a)
{ return erfcf(a); }
inline long double erfc(long double a)
{
#ifdef _MSC_VER
   return erfc((double)a); 
#else
   return erfcl(a); 
#endif
}
}
#define TEST_OTHER
#endif

#ifdef TEST_GSL
#include <gsl/gsl_sf_erf.h>

namespace other{
inline float erf(float a)
{ return (float)gsl_sf_erf(a); }
inline double erf(double a)
{ return gsl_sf_erf(a); }
inline long double erf(long double a)
{ return gsl_sf_erf(a); }
inline float erfc(float a)
{ return (float)gsl_sf_erfc(a); }
inline double erfc(double a)
{ return gsl_sf_erfc(a); }
inline long double erfc(long double a)
{ return gsl_sf_erfc(a); }
}
#define TEST_OTHER
#endif

#ifdef TEST_OTHER
namespace other{
   boost::math::concepts::real_concept erf(boost::math::concepts::real_concept){ return 0; }
   boost::math::concepts::real_concept erfc(boost::math::concepts::real_concept){ return 0; }
}
#endif


#endif


