//  (C) Copyright John Maddock 2008.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_NCBETA_OTHER_HOOKS_HPP
#define BOOST_MATH_TEST_NCBETA_OTHER_HOOKS_HPP


#ifdef TEST_R
#define MATHLIB_STANDALONE
#include <rmath.h>
namespace other{
inline float ncbeta_cdf(float a, float b, float nc, float x)
{ 
   return (float)pnbeta(x, a, b, nc, 1, 0);
}
inline double ncbeta_cdf(double a, double b, double nc, double x)
{
   return pnbeta(x, a, b, nc, 1, 0);
}
inline long double ncbeta_cdf(long double a, long double b, long double nc, long double x)
{ 
   return pnbeta((double)x, (double)a, (double)b, (double)nc, 1, 0);
}
}
#define TEST_OTHER
#endif

#ifdef TEST_OTHER
namespace other{
   boost::math::concepts::real_concept ncbeta_cdf(boost::math::concepts::real_concept, boost::math::concepts::real_concept, boost::math::concepts::real_concept){ return 0; }
}
#endif


#endif


