//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_BETA_OTHER_HOOKS_HPP
#define BOOST_MATH_TEST_BETA_OTHER_HOOKS_HPP

#ifdef TEST_CEPHES
namespace other{
extern "C" {
   double beta(double, double);
   float betaf(float, float);
   long double betal(long double, long double);

   double incbet(double, double, double);
   float incbetf(float, float, float);
   long double incbetl(long double, long double, long double);
}
inline float beta(float a, float b)
{ return betaf(a, b); }
inline long double beta(long double a, long double b)
{
#ifdef _MSC_VER
   return beta((double)a, (double)b); 
#else
   return betal(a, b); 
#endif
}
inline float ibeta(float a, float b, float x)
{ return incbetf(a, b, x); }
inline double ibeta(double a, double b, double x)
{ return incbet(a, b, x); }
inline long double ibeta(long double a, long double b, long double x)
{
#ifdef _MSC_VER
   return incbet((double)a, (double)b, (double)x); 
#else
   return incbetl(a, b); 
#endif
}
}
#define TEST_OTHER
#endif

#ifdef TEST_GSL
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_message.h>

namespace other{
inline float beta(float a, float b)
{ return (float)gsl_sf_beta(a, b); }
inline double beta(double a, double b)
{ return gsl_sf_beta(a, b); }
inline long double beta(long double a, long double b)
{ return gsl_sf_beta(a, b); }

inline float ibeta(float a, float b, float x)
{ return (float)gsl_sf_beta_inc(a, b, x); }
inline double ibeta(double a, double b, double x)
{ return gsl_sf_beta_inc(a, b, x); }
inline long double ibeta(long double a, long double b, long double x)
{
   return gsl_sf_beta_inc((double)a, (double)b, (double)x); 
}
}
#define TEST_OTHER
#endif

#ifdef TEST_BRATIO
namespace other{
extern "C" int bratio_(double*a, double*b, double*x, double*y, double*w, double*w1, int*ierr);

inline float ibeta(float a, float b, float x)
{
   double a_ = a;
   double b_ = b;
   double x_ = x;
   double y_ = 1-x_;
   double w, w1;
   int ierr = 0;
   bratio_(&a_, &b_, &x_, &y_, &w, &w1, &ierr); 
   return w;
}
inline double ibeta(double a, double b, double x)
{ 
   double a_ = a;
   double b_ = b;
   double x_ = x;
   double y_ = 1-x_;
   double w, w1;
   int ierr = 0;
   bratio_(&a_, &b_, &x_, &y_, &w, &w1, &ierr); 
   return w;
}
inline long double ibeta(long double a, long double b, long double x)
{
   double a_ = a;
   double b_ = b;
   double x_ = x;
   double y_ = 1-x_;
   double w, w1;
   int ierr = 0;
   bratio_(&a_, &b_, &x_, &y_, &w, &w1, &ierr); 
   return w;
}
}
#define TEST_OTHER
#endif

#ifdef TEST_OTHER
namespace other{
   boost::math::concepts::real_concept beta(boost::math::concepts::real_concept, boost::math::concepts::real_concept){ return 0; }
   boost::math::concepts::real_concept ibeta(boost::math::concepts::real_concept, boost::math::concepts::real_concept, boost::math::concepts::real_concept){ return 0; }
}
#endif


#endif


