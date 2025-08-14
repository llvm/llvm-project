//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_LEGENDRE_OTHER_HOOKS_HPP
#define BOOST_MATH_TEST_LEGENDRE_OTHER_HOOKS_HPP

#ifdef TEST_GSL
#include <gsl/gsl_sf_legendre.h>

namespace other{
inline float legendre_p(int l, float a)
{ return (float)gsl_sf_legendre_Pl (l, a); }
inline double legendre_p(int l, double a)
{ return gsl_sf_legendre_Pl (l, a); }
inline long double legendre_p(int l, long double a)
{ return gsl_sf_legendre_Pl (l, a); }

inline float legendre_q(int l, float a)
{ return (float)gsl_sf_legendre_Ql (l, a); }
inline double legendre_q(int l, double a)
{ return gsl_sf_legendre_Ql (l, a); }
inline long double legendre_q(int l, long double a)
{ return gsl_sf_legendre_Ql (l, a); }

}
#define TEST_OTHER
#endif

#ifdef TEST_OTHER
namespace other{
   boost::math::concepts::real_concept legendre_p(int, boost::math::concepts::real_concept){ return 0; }
   boost::math::concepts::real_concept legendre_q(int, boost::math::concepts::real_concept){ return 0; }
}
#endif


#endif


