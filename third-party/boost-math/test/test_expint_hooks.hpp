//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_ZETA_OTHER_HOOKS_HPP
#define BOOST_MATH_TEST_ZETA_OTHER_HOOKS_HPP


#ifdef TEST_CEPHES
namespace other{
extern "C" {
   double expn(int, double);
   float expnf(int, float);
   long double expnl(int, long double);
}
inline float expint(unsigned n, float a)
{ return expnf(n, a); }
inline double expint(unsigned n, double a)
{ return expn(n, a); }
inline long double expint(unsigned n, long double a)
{
#ifdef _MSC_VER
   return expn(n, (double)a); 
#else
   return expnl(n, a); 
#endif
}
// Ei is not supported:
template <class T>
inline T expint(T){ return 0; }
}
#define TEST_OTHER
#endif

#ifdef TEST_GSL
#include <gsl/gsl_sf_expint.h>

namespace other{
inline float expint(float a)
{ return (float)gsl_sf_expint_Ei(a); }
inline double expint(double a)
{ return gsl_sf_expint_Ei(a); }
inline long double expint(long double a)
{ return gsl_sf_expint_Ei(a); }
// En is not supported:
template <class T>
inline T expint(unsigned, T){ return 0; }
}
#define TEST_OTHER
#endif

#ifdef TEST_SPECFUN
namespace other{
extern "C" int calcei_(double *arg, double *result, int*);
inline float expint(float a)
{ 
   double r, a_(a);
   int v = 1;
   calcei_(&a_, &r, &v); 
   return (float)r;
}
inline double expint(double a)
{ 
   double r, a_(a);
   int v = 1;
   calcei_(&a_, &r, &v); 
   return r;
}
inline long double expint(long double a)
{ 
   double r, a_(a);
   int v = 1;
   calcei_(&a_, &r, &v); 
   return r;
}
// En is not supported:
template <class T>
inline T expint(unsigned, T){ return 0; }
}
#define TEST_OTHER
#endif

#ifdef TEST_OTHER
namespace other{
   boost::math::concepts::real_concept expint(unsigned, boost::math::concepts::real_concept){ return 0; }
   boost::math::concepts::real_concept expint(boost::math::concepts::real_concept){ return 0; }
}
#endif


#endif


