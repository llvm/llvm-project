//  (C) Copyright John Maddock 2008.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_NCCS_OTHER_HOOKS_HPP
#define BOOST_MATH_TEST_NCCS_OTHER_HOOKS_HPP


#ifdef TEST_R
#define MATHLIB_STANDALONE
#include <rmath.h>
namespace other{
inline float nccs_cdf(float df, float nc, float x)
{ 
   return (float)pnchisq(x, df, nc, 1, 0);
}
inline double nccs_cdf(double df, double nc, double x)
{
   return pnchisq(x, df, nc, 1, 0);
}
inline long double nccs_cdf(long double df, long double nc, long double x)
{ 
   return pnchisq((double)x, (double)df, (double)nc, 1, 0);
}
}
#define TEST_OTHER
#endif

#ifdef TEST_CDFLIB
#include <cdflib.h>
namespace other{
inline double nccs_cdf(double df, double nc, double x)
{
   int kind(1), status(0);
   double p, q, bound(0);
   cdfchn(&kind, &p, &q, &x, &df, &nc, &status, &bound);
   return p;
}
inline float nccs_cdf(float df, float nc, float x)
{ 
   return (double)nccs_cdf((double)df, (double)nc, (double)x);
}
inline long double nccs_cdf(long double df, long double nc, long double x)
{ 
   return nccs_cdf((double)df, (double)nc, (double)x);
}
}
#define TEST_OTHER
#endif

#ifdef TEST_OTHER
namespace other{
   boost::math::concepts::real_concept nccs_cdf(boost::math::concepts::real_concept, boost::math::concepts::real_concept, boost::math::concepts::real_concept){ return 0; }
}
#endif


#endif


