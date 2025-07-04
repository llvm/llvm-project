//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_ZETA_OTHER_HOOKS_HPP
#define BOOST_MATH_TEST_ZETA_OTHER_HOOKS_HPP


#ifdef TEST_CEPHES
namespace other{
extern "C" {
   double zetac(double);
   float zetacf(float);
   long double zetacl(long double);
}
inline float zeta(float a)
{ return 1 + zetac(a); }
inline double zeta(double a)
{ return 1 + zetac(a); }
inline long double zeta(long double a)
{
#ifdef _MSC_VER
   return 1 + zetac((double)a); 
#else
   return zetacl(a); 
#endif
}
}
#define TEST_OTHER
#endif

#ifdef TEST_GSL
#include <gsl/gsl_sf_zeta.h>

namespace other{
inline float zeta(float a)
{ return (float)gsl_sf_zeta(a); }
inline double zeta(double a)
{ return gsl_sf_zeta(a); }
inline long double zeta(long double a)
{ return gsl_sf_zeta(a); }
}
#define TEST_OTHER
#endif

#ifdef TEST_OTHER
namespace other{
   boost::math::concepts::real_concept zeta(boost::math::concepts::real_concept){ return 0; }
}
#endif


#endif


