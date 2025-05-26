//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define L22
//#include "../tools/ntl_rr_lanczos.hpp"
//#include "../tools/ntl_rr_digamma.hpp"
#include "multiprecision.hpp"
#include <boost/math/tools/polynomial.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/special_functions/zeta.hpp>
#include <boost/math/special_functions/expint.hpp>
#include <boost/math/special_functions/lambert_w.hpp>

#include <cmath>


mp_type f(const mp_type& x, int variant)
{
   static const mp_type tiny = boost::math::tools::min_value<mp_type>() * 64;
   switch(variant)
   {
   case 0:
      {
      mp_type x_ = sqrt(x == 0 ? 1e-80 : x);
      return boost::math::erf(x_) / x_;
      }
   case 1:
      {
      mp_type x_ = 1 / x;
      return boost::math::erfc(x_) * x_ / exp(-x_ * x_);
      }
   case 2:
      {
      return boost::math::erfc(x) * x / exp(-x * x);
      }
   case 3:
      {
         mp_type y(x);
         if(y == 0) 
            y += tiny;
         return boost::math::lgamma(y+2) / y - 0.5;
      }
   case 4:
      //
      // lgamma in the range [2,3], use:
      //
      // lgamma(x) = (x-2) * (x + 1) * (c + R(x - 2))
      //
      // Works well at 80-bit long double precision, but doesn't
      // stretch to 128-bit precision.
      //
      if(x == 0)
      {
         return boost::lexical_cast<mp_type>("0.42278433509846713939348790991759756895784066406008") / 3;
      }
      return boost::math::lgamma(x+2) / (x * (x+3));
   case 5:
      {
         //
         // lgamma in the range [1,2], use:
         //
         // lgamma(x) = (x - 1) * (x - 2) * (c + R(x - 1))
         //
         // works well over [1, 1.5] but not near 2 :-(
         //
         mp_type r1 = boost::lexical_cast<mp_type>("0.57721566490153286060651209008240243104215933593992");
         mp_type r2 = boost::lexical_cast<mp_type>("0.42278433509846713939348790991759756895784066406008");
         if(x == 0)
         {
            return r1;
         }
         if(x == 1)
         {
            return r2;
         }
         return boost::math::lgamma(x+1) / (x * (x - 1));
      }
   case 6:
      {
         //
         // lgamma in the range [1.5,2], use:
         //
         // lgamma(x) = (2 - x) * (1 - x) * (c + R(2 - x))
         //
         // works well over [1.5, 2] but not near 1 :-(
         //
         mp_type r1 = boost::lexical_cast<mp_type>("0.57721566490153286060651209008240243104215933593992");
         mp_type r2 = boost::lexical_cast<mp_type>("0.42278433509846713939348790991759756895784066406008");
         if(x == 0)
         {
            return r2;
         }
         if(x == 1)
         {
            return r1;
         }
         return boost::math::lgamma(2-x) / (x * (x - 1));
      }
   case 7:
      {
         //
         // erf_inv in range [0, 0.5]
         //
         mp_type y = x;
         if(y == 0)
            y = boost::math::tools::epsilon<mp_type>() / 64;
         return boost::math::erf_inv(y) / (y * (y+10));
      }
   case 8:
      {
         // 
         // erfc_inv in range [0.25, 0.5]
         // Use an y-offset of 0.25, and range [0, 0.25]
         // abs error, auto y-offset.
         //
         mp_type y = x;
         if(y == 0)
            y = boost::lexical_cast<mp_type>("1e-5000");
         return sqrt(-2 * log(y)) / boost::math::erfc_inv(y);
      }
   case 9:
      {
         mp_type x2 = x;
         if(x2 == 0)
            x2 = boost::lexical_cast<mp_type>("1e-5000");
         mp_type y = exp(-x2*x2); // sqrt(-log(x2)) - 5;
         return boost::math::erfc_inv(y) / x2;
      }
   case 10:
      {
         //
         // Digamma over the interval [1,2], set x-offset to 1
         // and optimise for absolute error over [0,1].
         //
         int current_precision = get_working_precision();
         if(current_precision < 1000)
            set_working_precision(1000);
         //
         // This value for the root of digamma is calculated using our
         // differentiated lanczos approximation.  It agrees with Cody
         // to ~ 25 digits and to Morris to 35 digits.  See:
         // TOMS ALGORITHM 708 (Didonato and Morris).
         // and Math. Comp. 27, 123-127 (1973) by Cody, Strecok and Thacher.
         //
         //mp_type root = boost::lexical_cast<mp_type>("1.4616321449683623412626595423257213234331845807102825466429633351908372838889871");
         //
         // Actually better to calculate the root on the fly, it appears to be more
         // accurate: convergence is easier with the 1000-bit value, the approximation
         // produced agrees with functions.mathworld.com values to 35 digits even quite
         // near the root.
         //
         static boost::math::tools::eps_tolerance<mp_type> tol(1000);
         static std::uintmax_t max_iter = 1000;
         mp_type (*pdg)(mp_type) = &boost::math::digamma;
         static const mp_type root = boost::math::tools::bracket_and_solve_root(pdg, mp_type(1.4), mp_type(1.5), true, tol, max_iter).first;

         mp_type x2 = x;
         double lim = 1e-65;
         if(fabs(x2 - root) < lim)
         {
            //
            // This is a problem area:
            // x2-root suffers cancellation error, so does digamma.
            // That gets compounded again when Remez calculates the error
            // function.  This cludge seems to stop the worst of the problems:
            //
            static const mp_type a = boost::math::digamma(root - lim) / -lim;
            static const mp_type b = boost::math::digamma(root + lim) / lim;
            mp_type fract = (x2 - root + lim) / (2*lim);
            mp_type r = (1-fract) * a + fract * b;
            std::cout << "In root area: " << r;
            return r;
         }
         mp_type result =  boost::math::digamma(x2) / (x2 - root);
         if(current_precision < 1000)
            set_working_precision(current_precision);
         return result;
      }
   case 11:
      // expm1:
      if(x == 0)
      {
         static mp_type lim = 1e-80;
         static mp_type a = boost::math::expm1(-lim);
         static mp_type b = boost::math::expm1(lim);
         static mp_type l = (b-a) / (2 * lim);
         return l;
      }
      return boost::math::expm1(x) / x;
   case 12:
      // demo, and test case:
      return exp(x);
   case 13:
      // K(k):
      {
      return boost::math::ellint_1(x);
      }
   case 14:
      // K(k)
      {
      return boost::math::ellint_1(1-x) / log(x);
   }
   case 15:
      // E(k)
      {
         // x = 1-k^2
         mp_type z = 1 - x * log(x);
         return boost::math::ellint_2(sqrt(1-x)) / z;
      }
   case 16:
      // Bessel I0(x) over [0,16]:
      {
         return boost::math::cyl_bessel_i(0, sqrt(x));
      }
   case 17:
      // Bessel I0(x) over [16,INF]
      {
         mp_type z = 1 / (mp_type(1)/16 - x);
         return boost::math::cyl_bessel_i(0, z) * sqrt(z) / exp(z);
      }
   case 18:
      // Zeta over [0, 1]
      {
         return boost::math::zeta(1 - x) * x - x;
      }
   case 19:
      // Zeta over [1, n]
      {
         return boost::math::zeta(x) - 1 / (x - 1);
      }
   case 20:
      // Zeta over [a, b] : a >> 1
      {
         return log(boost::math::zeta(x) - 1);
      }
   case 21:
      // expint[1] over [0,1]:
      {
         mp_type tiny = boost::lexical_cast<mp_type>("1e-5000");
         mp_type z = (x <= tiny) ? tiny : x;
         return boost::math::expint(1, z) - z + log(z);
      }
   case 22:
      // expint[1] over [1,N],
      // Note that x varies from [0,1]:
      {
         mp_type z = 1 / x;
         return boost::math::expint(1, z) * exp(z) * z;
      }
   case 23:
      // expin Ei over [0,R]
      {
         static const mp_type root = 
            boost::lexical_cast<mp_type>("0.372507410781366634461991866580119133535689497771654051555657435242200120636201854384926049951548942392");
         mp_type z = x < (std::numeric_limits<long double>::min)() ? (std::numeric_limits<long double>::min)() : x;
         return (boost::math::expint(z) - log(z / root)) / (z - root);
      }
   case 24:
      // Expint Ei for large x:
      {
         static const mp_type root = 
            boost::lexical_cast<mp_type>("0.372507410781366634461991866580119133535689497771654051555657435242200120636201854384926049951548942392");
         mp_type z = x < (std::numeric_limits<long double>::min)() ? (std::numeric_limits<long double>::max)() : mp_type(1 / x);
         return (boost::math::expint(z) - z) * z * exp(-z);
         //return (boost::math::expint(z) - log(z)) * z * exp(-z);
      }
   case 25:
      // Expint Ei for large x:
      {
         return (boost::math::expint(x) - x) * x * exp(-x);
      }
   case 26:
      {
         //
         // erf_inv in range [0, 0.5]
         //
         mp_type y = x;
         if(y == 0)
            y = boost::math::tools::epsilon<mp_type>() / 64;
         y = sqrt(y);
         return boost::math::erf_inv(y) / (y);
      }
   case 28:
      {
         // log1p over [-0.5,0.5]
         mp_type y = x;
         if(fabs(y) < 1e-100)
            y = (y == 0) ? 1e-100 : boost::math::sign(y) * 1e-100;
         return (boost::math::log1p(y) - y + y * y / 2) / (y);
      }
   case 29:
      {
         // cbrt over [0.5, 1]
         return boost::math::cbrt(x);
      }
   case 30:
   {
      // trigamma over [x,y]
      mp_type y = x;
      y = sqrt(y);
      return boost::math::trigamma(x) * (x * x);
   }
   case 31:
   {
      // trigamma over [x, INF]
      if(x == 0) return 1;
      mp_type y = (x == 0) ? (std::numeric_limits<double>::max)() / 2 : mp_type(1/x);
      return boost::math::trigamma(y) * y;
   }
   case 32:
   {
      // I0 over [N, INF]
      // Don't need to go past x = 1/1000 = 1e-3 for double, or
      // 1/15000 = 0.0006 for long double, start at 1/7.75=0.13
      mp_type arg = 1 / x;
      return sqrt(arg) * exp(-arg) * boost::math::cyl_bessel_i(0, arg);
   }
   case 33:
   {
      // I0 over [0, N]
      mp_type xx = sqrt(x) * 2;
      return (boost::math::cyl_bessel_i(0, xx) - 1) / x;
   }
   case 34:
   {
      // I1 over [0, N]
      mp_type xx = sqrt(x) * 2;
      return (boost::math::cyl_bessel_i(1, xx) * 2 / xx - 1 - x / 2) / (x * x);
   }
   case 35:
   {
      // I1 over [N, INF]
      mp_type xx = 1 / x;
      return boost::math::cyl_bessel_i(1, xx) * sqrt(xx) * exp(-xx);
   }
   case 36:
   {
      // K0 over [0, 1]
      mp_type xx = sqrt(x);
      return boost::math::cyl_bessel_k(0, xx) + log(xx) * boost::math::cyl_bessel_i(0, xx);
   }
   case 37:
   {
      // K0 over [1, INF]
      mp_type xx = 1 / x;
      return boost::math::cyl_bessel_k(0, xx) * exp(xx) * sqrt(xx);
   }
   case 38:
   {
      // K1 over [0, 1]
      mp_type xx = sqrt(x);
      return (boost::math::cyl_bessel_k(1, xx) - log(xx) * boost::math::cyl_bessel_i(1, xx) - 1 / xx) / xx;
   }
   case 39:
   {
      // K1 over [1, INF]
      mp_type xx = 1 / x;
      return boost::math::cyl_bessel_k(1, xx) * sqrt(xx) * exp(xx);
   }
   // Lambert W0
   case 40:
      return boost::math::lambert_w0(x);
   case 41:
   {
      if (x == 0)
         return 1;
      return boost::math::lambert_w0(x) / x;
   }
   case 42:
   {
      static const mp_type e1 = exp(mp_type(-1));
      return x / -boost::math::lambert_w0(-e1 + x);
   }
   case 43:
   {
      mp_type xx = 1 / x;
      return 1 / boost::math::lambert_w0(xx);
   }
   case 44:
   {
      mp_type ex = exp(x);
      return boost::math::lambert_w0(ex) - x;
   }
   }
   return 0;
}

void show_extra(
   const boost::math::tools::polynomial<mp_type>& n, 
   const boost::math::tools::polynomial<mp_type>& d, 
   const mp_type& x_offset, 
   const mp_type& y_offset, 
   int variant)
{
   switch(variant)
   {
   default:
      // do nothing here...
      ;
   }
}

