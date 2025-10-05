//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_GAMMA_OTHER_HOOKS_HPP
#define BOOST_MATH_TEST_GAMMA_OTHER_HOOKS_HPP

#ifdef TEST_CEPHES
namespace other{
extern "C" {
   double gamma(double);
   float gammaf(float);
   long double gammal(long double);
   double lgam(double);
   float lgamf(float);
   long double lgaml(long double);
   float igamf(float, float);
   double igam(double, double);
   long double igaml(long double, long double);
   float igamcf(float, float);
   double igamc(double, double);
   long double igamcl(long double, long double);
}
inline float tgamma(float x)
{ return gammaf(x); }
inline double tgamma(double x)
{ return gamma(x); }
inline long double tgamma(long double x)
{
#ifdef _MSC_VER
   return gamma((double)x); 
#else
   return gammal(x); 
#endif
}
inline float lgamma(float x)
{ return lgamf(x); }
inline double lgamma(double x)
{ return lgam(x); }
inline long double lgamma(long double x)
{
#ifdef _MSC_VER
   return lgam((double)x); 
#else
   return lgaml(x); 
#endif
}
inline float gamma_q(float x, float y)
{ return igamcf(x, y); }
inline double gamma_q(double x, double y)
{ return igamc(x, y); }
inline long double gamma_q(long double x, long double y)
{
#ifdef _MSC_VER
   return igamc((double)x, (double)y); 
#else
   return igamcl(x, y); 
#endif
}
inline float gamma_p(float x, float y)
{ return igamf(x, y); }
inline double gamma_p(double x, double y)
{ return igam(x, y); }
inline long double gamma_p(long double x, long double y)
{
#ifdef _MSC_VER
   return igam((double)x, (double)y); 
#else
   return igaml(x, y); 
#endif
}
}
#define TEST_OTHER
#endif

#ifdef TEST_NATIVE
#include <math.h>
namespace other{
#if defined(__FreeBSD__)
// no float version:
inline float tgamma(float x)
{ return ::tgamma(x); }
#else
inline float tgamma(float x)
{ return ::tgammaf(x); }
#endif
inline double tgamma(double x)
{ return ::tgamma(x); }
inline long double tgamma(long double x)
{ 
#if defined(__CYGWIN__) || defined(__FreeBSD__)
   // no long double versions:
   return ::tgamma(x);
#else
   return ::tgammal(x);
#endif
}
#if defined(__FreeBSD__)
inline float lgamma(float x)
{ return ::lgamma(x); }
#else
inline float lgamma(float x)
{ return ::lgammaf(x); }
#endif
inline double lgamma(double x)
{ return ::lgamma(x); }
inline long double lgamma(long double x)
{ 
#if defined(__CYGWIN__) || defined(__FreeBSD__) 
   // no long double versions:
   return ::lgamma(x); 
#else
   return ::lgammal(x); 
#endif
}
}
#define TEST_OTHER
#endif

#ifdef TEST_GSL
#define TEST_OTHER
#include <gsl/gsl_sf_gamma.h>

namespace other{
float tgamma(float z)
{
   return (float)gsl_sf_gamma(z);
}
double tgamma(double z)
{
   return gsl_sf_gamma(z);
}
long double tgamma(long double z)
{
   return gsl_sf_gamma(z);
}
float lgamma(float z)
{
   return (float)gsl_sf_lngamma(z);
}
double lgamma(double z)
{
   return gsl_sf_lngamma(z);
}
long double lgamma(long double z)
{
   return gsl_sf_lngamma(z);
}
inline float gamma_q(float x, float y)
{ return (float)gsl_sf_gamma_inc_Q(x, y); }
inline double gamma_q(double x, double y)
{ return gsl_sf_gamma_inc_Q(x, y); }
inline long double gamma_q(long double x, long double y)
{ return gsl_sf_gamma_inc_Q(x, y); }
inline float gamma_p(float x, float y)
{ return (float)gsl_sf_gamma_inc_P(x, y); }
inline double gamma_p(double x, double y)
{ return gsl_sf_gamma_inc_P(x, y); }
inline long double gamma_p(long double x, long double y)
{ return gsl_sf_gamma_inc_P(x, y); }
}
#endif

#ifdef TEST_DCDFLIB
#define TEST_OTHER
#include <dcdflib.h>

namespace other{
float tgamma(float z)
{
   double v = z;
   return (float)gamma_x(&v);
}
double tgamma(double z)
{
   return gamma_x(&z);
}
long double tgamma(long double z)
{
   double v = z;
   return gamma_x(&v);
}
float lgamma(float z)
{
   double v = z;
   return (float)gamma_log(&v);
}
double lgamma(double z)
{
   double v = z;
   return gamma_log(&v);
}
long double lgamma(long double z)
{
   double v = z;
   return gamma_log(&v);
}
inline double gamma_q(double x, double y)
{ 
   double ans, qans;
   int i = 0;
   gamma_inc (&x, &y, &ans, &qans, &i); 
   return qans;
}
inline float gamma_q(float x, float y)
{ 
   return (float)gamma_q((double)x, (double)y); 
}
inline long double gamma_q(long double x, long double y)
{ 
   return gamma_q((double)x, (double)y); 
}
inline double gamma_p(double x, double y)
{ 
   double ans, qans;
   int i = 0;
   gamma_inc (&x, &y, &ans, &qans, &i); 
   return ans;
}
inline float gamma_p(float x, float y)
{ 
   return (float)gamma_p((double)x, (double)y); 
}
inline long double gamma_p(long double x, long double y)
{ 
   return gamma_p((double)x, (double)y); 
}

inline double gamma_q_inv(double x, double y)
{ 
   double ans, p, nul;
   int i = 0;
   p = 1 - y;
   nul = 0;
   gamma_inc_inv (&x, &ans, &nul, &p, &y, &i); 
   return ans;
}
inline float gamma_q_inv(float x, float y)
{ 
   return (float)gamma_q_inv((double)x, (double)y); 
}
inline long double gamma_q_inv(long double x, long double y)
{ 
   return gamma_q_inv((double)x, (double)y); 
}
inline double gamma_p_inv(double x, double y)
{ 
   double ans, p, nul;
   int i = 0;
   p = 1 - y;
   nul = 0;
   gamma_inc_inv (&x, &ans, &nul, &y, &p, &i); 
   return ans;
}
inline float gamma_p_inv(float x, float y)
{ 
   return (float)gamma_p_inv((double)x, (double)y); 
}
inline long double gamma_p_inv(long double x, long double y)
{ 
   return gamma_p_inv((double)x, (double)y); 
}

}
#endif

#ifdef TEST_OTHER
namespace other{
   boost::math::concepts::real_concept tgamma(boost::math::concepts::real_concept){ return 0; }
   boost::math::concepts::real_concept lgamma(boost::math::concepts::real_concept){ return 0; }
   boost::math::concepts::real_concept gamma_q(boost::math::concepts::real_concept, boost::math::concepts::real_concept){ return 0; }
   boost::math::concepts::real_concept gamma_p(boost::math::concepts::real_concept, boost::math::concepts::real_concept){ return 0; }
   boost::math::concepts::real_concept gamma_p_inv(boost::math::concepts::real_concept x, boost::math::concepts::real_concept y){ return 0; }
   boost::math::concepts::real_concept gamma_q_inv(boost::math::concepts::real_concept x, boost::math::concepts::real_concept y){ return 0; }
}
#endif

#endif


