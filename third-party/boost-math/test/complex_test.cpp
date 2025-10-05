//  (C) Copyright John Maddock 2005.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/mpl/if.hpp>
#include <boost/math/tools/assert.hpp>
#include <boost/math/complex.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <typeinfo>

#ifdef BOOST_NO_STDC_NAMESPACE
namespace std{ using ::sqrt; using ::tan; using ::tanh; }
#endif

#ifndef VERBOSE
#undef BOOST_TEST_MESSAGE
#define BOOST_TEST_MESSAGE(x)
#endif

#ifdef _MSC_VER
#pragma warning (disable:C4996)
#elif __GNUC__ >= 5
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__clang__)
#  pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

//
// check_complex:
// Verifies that expected value "a" and found value "b" have a relative error
// less than "max_error" epsilons.  Note that relative error is calculated for
// the complex number as a whole; this means that the error in the real or 
// imaginary parts alone can be much higher than max_error when the real and 
// imaginary parts are of very different magnitudes.  This is important, because
// the Hull et al analysis of the acos and asin algorithms requires that very small
// real/imaginary components can be safely ignored if they are negligible compared
// to the other component.
//
template <class T>
bool check_complex(const std::complex<T>& a, const std::complex<T>& b, int max_error)
{
   //
   // a is the expected value, b is what was actually found,
   // compute | (a-b)/b | and compare with max_error which is the 
   // multiple of E to permit:
   //
   bool result = true;
   static const std::complex<T> zero(0);
   static const T eps = std::pow(static_cast<T>(std::numeric_limits<T>::radix), static_cast<T>(1 - std::numeric_limits<T>::digits));
   if(a == zero)
   {
      if(b != zero)
      {
         if(boost::math::fabs(b) > eps)
         {
            result = false;
            BOOST_ERROR("Expected {0,0} but got: " << b);
         }
         else
         {
            BOOST_TEST_MESSAGE("Expected {0,0} but got: " << b);
         }
      }
      return result;
   }
   else if(b == zero)
   {
      if(boost::math::fabs(a) > eps)
      {
         BOOST_ERROR("Found {0,0} but expected: " << a);
         return false;;
      }
      else
      {
         BOOST_TEST_MESSAGE("Found {0,0} but expected: " << a);
      }
   }

   if((boost::math::isnan)(a.real()))
   {
      BOOST_ERROR("Found non-finite value for real part: " << a);
   }
   if((boost::math::isnan)(a.imag()))
   {
      BOOST_ERROR("Found non-finite value for imaginary part: " << a);
   }

   T rel = boost::math::fabs((b-a)/b) / eps;
   if( rel > max_error)
   {
      result = false;
      BOOST_ERROR("Error in result exceeded permitted limit of " << max_error << " (actual relative error was " << rel << "e).  Found " << b << " expected " << a);
   }
   return result;
}

//
// test_inverse_trig:
// This is nothing more than a sanity check, computes trig(atrig(z)) 
// and compare the result to z.  Note that:
//
// atrig(trig(z)) != z
//
// for certain z because the inverse trig functions are multi-valued, this 
// essentially rules this out as a testing method.  On the other hand:
//
// trig(atrig(z))
//
// can vary compare to z by an arbitrarily large amount.  For one thing we 
// have no control over the implementation of the trig functions, for another
// even if both functions were accurate to 1ulp (as accurate as transcendental
// number can get, thanks to the "table makers dilemma"), the errors can still
// be arbitrarily large - often the inverse trig functions will map a very large
// part of the complex domain into a small output domain, so you can never get
// back exactly where you started from.  Consequently these tests are no more than
// sanity checks (just verifies that signs are correct and so on).
//
template <class T>
void test_inverse_trig(T)
{
   using namespace std;

   static const T interval = static_cast<T>(2.0L/128.0L);

   T x, y;

   std::cout << std::setprecision(std::numeric_limits<T>::digits10+2);

   for(x = -1; x <= 1; x += interval)
   {
      for(y = -1; y <= 1; y += interval)
      {
         // acos:
         std::complex<T> val(x, y), inter, result;
         inter = boost::math::acos(val);
         result = cos(inter);
         if(!check_complex(val, result, 50))
         {
            std::cout << "Error in testing inverse complex cos for type " << typeid(T).name() << std::endl;
            std::cout << "   val=             " << val << std::endl;
            std::cout << "   acos(val) =      " << inter << std::endl;
            std::cout << "   cos(acos(val)) = " << result << std::endl;
         }
         // asin:
         inter = boost::math::asin(val);
         result = sin(inter);
         if(!check_complex(val, result, 5))
         {
            std::cout << "Error in testing inverse complex sin for type " << typeid(T).name() << std::endl;
            std::cout << "   val=             " << val << std::endl;
            std::cout << "   asin(val) =      " << inter << std::endl;
            std::cout << "   sin(asin(val)) = " << result << std::endl;
         }
      }
   }

   static const T interval2 = static_cast<T>(3.0L/256.0L);
   for(x = -3; x <= 3; x += interval2)
   {
      for(y = -3; y <= 3; y += interval2)
      {
         // asinh:
         std::complex<T> val(x, y), inter, result;
         inter = boost::math::asinh(val);
         result = sinh(inter);
         if(!check_complex(val, result, 5))
         {
            std::cout << "Error in testing inverse complex sinh for type " << typeid(T).name() << std::endl;
            std::cout << "   val=               " << val << std::endl;
            std::cout << "   asinh(val) =       " << inter << std::endl;
            std::cout << "   sinh(asinh(val)) = " << result << std::endl;
         }
         // acosh:
         if(!((y == 0) && (x <= 1))) // can't test along the branch cut
         {
            inter = boost::math::acosh(val);
            result = cosh(inter);
            if(!check_complex(val, result, 60))
            {
               std::cout << "Error in testing inverse complex cosh for type " << typeid(T).name() << std::endl;
               std::cout << "   val=               " << val << std::endl;
               std::cout << "   acosh(val) =       " << inter << std::endl;
               std::cout << "   cosh(acosh(val)) = " << result << std::endl;
            }
         }
         //
         // There is a problem in testing atan and atanh:
         // The inverse functions map a large input range to a much
         // smaller output range, so at the extremes too rather different
         // inputs may map to the same output value once rounded to N places.
         // Consequently tan(atan(z)) can suffer from arbitrarily large errors
         // even if individually they each have a small error bound.  On the other
         // hand we can't test atan(tan(z)) either because atan is multi-valued, so
         // round-tripping in this direction isn't always possible.
         // The following heuristic is designed to make the best of a bad job,
         // using atan(tan(z)) where possible and tan(atan(z)) when it's not.
         //
         static const int tanh_error = 20;
         if((0 != x) && (0 != y) && ((std::fabs(y) < 1) || (std::fabs(x) < 1)))
         {
            // atanh:
            val = boost::math::atanh(val);
            inter = tanh(val);
            result = boost::math::atanh(inter);
            if(!check_complex(val, result, tanh_error))
            {
               std::cout << "Error in testing inverse complex tanh for type " << typeid(T).name() << std::endl;
               std::cout << "   val=               " << val << std::endl;
               std::cout << "   tanh(val) =        " << inter << std::endl;
               std::cout << "   atanh(tanh(val)) = " << result << std::endl;
            }
            // atan:
            if(!((x == 0) && (std::fabs(y) == 1))) // we can't test infinities here
            {
               val = std::complex<T>(x, y);
               val = boost::math::atan(val);
               inter = tan(val);
               result = boost::math::atan(inter);
               if(!check_complex(val, result, tanh_error))
               {
                  std::cout << "Error in testing inverse complex tan for type " << typeid(T).name() << std::endl;
                  std::cout << "   val=               " << val << std::endl;
                  std::cout << "   tan(val) =         " << inter << std::endl;
                  std::cout << "   atan(tan(val)) =   " << result << std::endl;
               }
            }
         }
         else
         {
            // atanh:
            inter = boost::math::atanh(val);
            result = tanh(inter);
            if(!check_complex(val, result, tanh_error))
            {
               std::cout << "Error in testing inverse complex atanh for type " << typeid(T).name() << std::endl;
               std::cout << "   val=                 " << val << std::endl;
               std::cout << "   atanh(val) =         " << inter << std::endl;
               std::cout << "   tanh(atanh(val)) =   " << result << std::endl;
            }
            // atan:
            if(!((x == 0) && (std::fabs(y) == 1))) // we can't test infinities here
            {
               inter = boost::math::atan(val);
               result = tan(inter);
               if(!check_complex(val, result, tanh_error))
               {
                  std::cout << "Error in testing inverse complex atan for type " << typeid(T).name() << std::endl;
                  std::cout << "   val=                 " << val << std::endl;
                  std::cout << "   atan(val) =          " << inter << std::endl;
                  std::cout << "   tan(atan(val)) =     " << result << std::endl;
               }
            }
         }
      }
   }
}

//
// check_spots:
// Various spot values, mostly the C99 special cases (infinities and NAN's).
// TODO: add spot checks for the Wolfram spot values.
//
template <class T>
void check_spots(const T&)
{
   typedef std::complex<T> ct;
   ct result;
   static const T two = 2.0;
   T eps = std::pow(two, T(1-std::numeric_limits<T>::digits)); // numeric_limits<>::epsilon way too small to be useful on Darwin.
   static const T zero = 0;
   static const T mzero = -zero;
   static const T one = 1;
   static const T pi = boost::math::constants::pi<T>();
   static const T half_pi = boost::math::constants::half_pi<T>();
   static const T quarter_pi = half_pi / 2;
   static const T three_quarter_pi = boost::math::constants::three_quarters_pi<T>();
   T infinity = std::numeric_limits<T>::infinity();
   bool test_infinity = std::numeric_limits<T>::has_infinity;
   T nan = 0;
   bool test_nan = false;
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x564))
   // numeric_limits reports that a quiet NaN is present
   // but an attempt to access it will terminate the program!!!!
   if(std::numeric_limits<T>::has_quiet_NaN)
      nan = std::numeric_limits<T>::quiet_NaN();
   if((boost::math::isnan)(nan))
      test_nan = true;
#endif
#if defined(__DECCXX) && !defined(_IEEE_FP)
   // Tru64 cxx traps infinities unless the -ieee option is used:
   test_infinity = false;
#endif

   //
   // C99 spot tests for acos:
   //
   result = boost::math::acos(ct(zero));
   check_complex(ct(half_pi), result, 2);
   
   result = boost::math::acos(ct(mzero));
   check_complex(ct(half_pi), result, 2);
   
   result = boost::math::acos(ct(zero, mzero));
   check_complex(ct(half_pi), result, 2);
   
   result = boost::math::acos(ct(mzero, mzero));
   check_complex(ct(half_pi), result, 2);
   
   if(test_nan)
   {
      result = boost::math::acos(ct(zero,nan));
      BOOST_CHECK_CLOSE(result.real(), half_pi, eps*200);
      BOOST_CHECK((boost::math::isnan)(result.imag()));
   
      result = boost::math::acos(ct(mzero,nan));
      BOOST_CHECK_CLOSE(result.real(), half_pi, eps*200);
      BOOST_CHECK((boost::math::isnan)(result.imag()));
   }
   if(test_infinity)
   {
      result = boost::math::acos(ct(zero, infinity));
      BOOST_CHECK_CLOSE(result.real(), half_pi, eps*200);
      BOOST_CHECK(result.imag() == -infinity);

      result = boost::math::acos(ct(zero, -infinity));
      BOOST_CHECK_CLOSE(result.real(), half_pi, eps*200);
      BOOST_CHECK(result.imag() == infinity);
   }

   if(test_nan)
   {
      result = boost::math::acos(ct(one, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));
   }
   if(test_infinity)
   {
      result = boost::math::acos(ct(-infinity, one));
      BOOST_CHECK_CLOSE(result.real(), pi, eps*200);
      BOOST_CHECK(result.imag() == -infinity);

      result = boost::math::acos(ct(infinity, one));
      BOOST_CHECK(result.real() == 0);
      BOOST_CHECK(result.imag() == -infinity);

      result = boost::math::acos(ct(-infinity, -one));
      BOOST_CHECK_CLOSE(result.real(), pi, eps*200);
      BOOST_CHECK(result.imag() == infinity);

      result = boost::math::acos(ct(infinity, -one));
      BOOST_CHECK(result.real() == 0);
      BOOST_CHECK(result.imag() == infinity);

      result = boost::math::acos(ct(-infinity, infinity));
      BOOST_CHECK_CLOSE(result.real(), three_quarter_pi, eps*200);
      BOOST_CHECK(result.imag() == -infinity);

      result = boost::math::acos(ct(infinity, infinity));
      BOOST_CHECK_CLOSE(result.real(), quarter_pi, eps*200);
      BOOST_CHECK(result.imag() == -infinity);

      result = boost::math::acos(ct(-infinity, -infinity));
      BOOST_CHECK_CLOSE(result.real(), three_quarter_pi, eps*200);
      BOOST_CHECK(result.imag() == infinity);

      result = boost::math::acos(ct(infinity, -infinity));
      BOOST_CHECK_CLOSE(result.real(), quarter_pi, eps*200);
      BOOST_CHECK(result.imag() == infinity);
   }
   if(test_nan)
   {
      result = boost::math::acos(ct(infinity, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK(std::fabs(result.imag()) == infinity);

      result = boost::math::acos(ct(-infinity, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK(std::fabs(result.imag()) == infinity);

      result = boost::math::acos(ct(nan, zero));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::acos(ct(nan, -zero));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::acos(ct(nan, one));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::acos(ct(nan, -one));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::acos(ct(nan, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::acos(ct(nan, infinity));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK(result.imag() == -infinity);
      
      result = boost::math::acos(ct(nan, -infinity));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK(result.imag() == infinity);
   }
   if(boost::math::signbit(mzero))
   {
      result = boost::math::acos(ct(-1.25f, zero));
      BOOST_CHECK(result.real() > 0);
      BOOST_CHECK(result.imag() < 0);
      result = boost::math::asin(ct(-1.75f, mzero));
      BOOST_CHECK(result.real() < 0);
      BOOST_CHECK(result.imag() < 0);
      result = boost::math::atan(ct(mzero, -1.75f));
      BOOST_CHECK(result.real() < 0);
      BOOST_CHECK(result.imag() < 0);

      result = boost::math::acos(ct(zero, zero));
      BOOST_CHECK(result.real() > 0);
      BOOST_CHECK(result.imag() == 0);
      BOOST_CHECK((boost::math::signbit)(result.imag()));
      result = boost::math::acos(ct(zero, mzero));
      BOOST_CHECK(result.real() > 0);
      BOOST_CHECK(result.imag() == 0);
      BOOST_CHECK(0 == (boost::math::signbit)(result.imag()));
      result = boost::math::acos(ct(mzero, zero));
      BOOST_CHECK(result.real() > 0);
      BOOST_CHECK(result.imag() == 0);
      BOOST_CHECK((boost::math::signbit)(result.imag()));
      result = boost::math::acos(ct(mzero, mzero));
      BOOST_CHECK(result.real() > 0);
      BOOST_CHECK(result.imag() == 0);
      BOOST_CHECK(0 == (boost::math::signbit)(result.imag()));
   }

   //
   // C99 spot tests for acosh:
   //
   result = boost::math::acosh(ct(zero, zero));
   BOOST_CHECK(result.real() == 0);
   BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);

   result = boost::math::acosh(ct(zero, mzero));
   BOOST_CHECK(result.real() == 0);
   BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);

   result = boost::math::acosh(ct(mzero, zero));
   BOOST_CHECK(result.real() == 0);
   BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);
   
   result = boost::math::acosh(ct(mzero, mzero));
   BOOST_CHECK(result.real() == 0);
   BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);
   
   if(test_infinity)
   {
      result = boost::math::acosh(ct(one, infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);

      result = boost::math::acosh(ct(one, -infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);
   }

   if(test_nan)
   {
      result = boost::math::acosh(ct(one, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));
   }
   if(test_infinity)
   {
      result = boost::math::acosh(ct(-infinity, one));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), pi, eps*200);
      
      result = boost::math::acosh(ct(infinity, one));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK(result.imag() == 0);
      
      result = boost::math::acosh(ct(-infinity, -one));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), -pi, eps*200);
      
      result = boost::math::acosh(ct(infinity, -one));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK(result.imag() == 0);
      
      result = boost::math::acosh(ct(-infinity, infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), three_quarter_pi, eps*200);
      
      result = boost::math::acosh(ct(infinity, infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), quarter_pi, eps*200);
      
      result = boost::math::acosh(ct(-infinity, -infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), -three_quarter_pi, eps*200);
      
      result = boost::math::acosh(ct(infinity, -infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), -quarter_pi, eps*200);
   }
   
   if(test_nan)
   {
      result = boost::math::acosh(ct(infinity, nan));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK((boost::math::isnan)(result.imag()));
      
      result = boost::math::acosh(ct(-infinity, nan));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK((boost::math::isnan)(result.imag()));
      
      result = boost::math::acosh(ct(nan, one));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));
      
      result = boost::math::acosh(ct(nan, infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK((boost::math::isnan)(result.imag()));
      
      result = boost::math::acosh(ct(nan, -one));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));
      
      result = boost::math::acosh(ct(nan, -infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK((boost::math::isnan)(result.imag()));
      
      result = boost::math::acosh(ct(nan, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));
   }
   if(boost::math::signbit(mzero))
   {
      result = boost::math::acosh(ct(-2.5f, zero));
      BOOST_CHECK(result.real() > 0);
      BOOST_CHECK(result.imag() > 0);
   }
   //
   // C99 spot checks for asinh:
   //
   result = boost::math::asinh(ct(zero, zero));
   BOOST_CHECK(result.real() == 0);
   BOOST_CHECK(result.imag() == 0);

   result = boost::math::asinh(ct(mzero, zero));
   BOOST_CHECK(result.real() == 0);
   BOOST_CHECK(result.imag() == 0);

   result = boost::math::asinh(ct(zero, mzero));
   BOOST_CHECK(result.real() == 0);
   BOOST_CHECK(result.imag() == 0);

   result = boost::math::asinh(ct(mzero, mzero));
   BOOST_CHECK(result.real() == 0);
   BOOST_CHECK(result.imag() == 0);

   if(test_infinity)
   {
      result = boost::math::asinh(ct(one, infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);
      
      result = boost::math::asinh(ct(one, -infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);
      
      result = boost::math::asinh(ct(-one, -infinity));
      BOOST_CHECK(result.real() == -infinity);
      BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);
      
      result = boost::math::asinh(ct(-one, infinity));
      BOOST_CHECK(result.real() == -infinity);
      BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);
   }

   if(test_nan)
   {
      result = boost::math::asinh(ct(one, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::asinh(ct(-one, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::asinh(ct(zero, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));
   }

   if(test_infinity)
   {
      result = boost::math::asinh(ct(infinity, one));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK(result.imag() == 0);
      
      result = boost::math::asinh(ct(infinity, -one));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK(result.imag() == 0);
      
      result = boost::math::asinh(ct(-infinity, -one));
      BOOST_CHECK(result.real() == -infinity);
      BOOST_CHECK(result.imag() == 0);
      
      result = boost::math::asinh(ct(-infinity, one));
      BOOST_CHECK(result.real() == -infinity);
      BOOST_CHECK(result.imag() == 0);
      
      result = boost::math::asinh(ct(infinity, infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), quarter_pi, eps*200);
      
      result = boost::math::asinh(ct(infinity, -infinity));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK_CLOSE(result.imag(), -quarter_pi, eps*200);
      
      result = boost::math::asinh(ct(-infinity, -infinity));
      BOOST_CHECK(result.real() == -infinity);
      BOOST_CHECK_CLOSE(result.imag(), -quarter_pi, eps*200);
      
      result = boost::math::asinh(ct(-infinity, infinity));
      BOOST_CHECK(result.real() == -infinity);
      BOOST_CHECK_CLOSE(result.imag(), quarter_pi, eps*200);
   }

   if(test_nan)
   {
      result = boost::math::asinh(ct(infinity, nan));
      BOOST_CHECK(result.real() == infinity);
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::asinh(ct(-infinity, nan));
      BOOST_CHECK(result.real() == -infinity);
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::asinh(ct(nan, zero));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK(result.imag() == 0);

      result = boost::math::asinh(ct(nan,  mzero));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK(result.imag() == 0);

      result = boost::math::asinh(ct(nan, one));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::asinh(ct(nan,  -one));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::asinh(ct(nan,  nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::asinh(ct(nan, infinity));
      BOOST_CHECK(std::fabs(result.real()) == infinity);
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::asinh(ct(nan,  -infinity));
      BOOST_CHECK(std::fabs(result.real()) == infinity);
      BOOST_CHECK((boost::math::isnan)(result.imag()));
   }
   if(boost::math::signbit(mzero))
   {
      result = boost::math::asinh(ct(zero, 1.5f));
      BOOST_CHECK(result.real() > 0);
      BOOST_CHECK(result.imag() > 0);
   }
   
   //
   // C99 special cases for atanh:
   //
   result = boost::math::atanh(ct(zero, zero));
   BOOST_CHECK(result.real() == zero);
   BOOST_CHECK(result.imag() == zero);

   result = boost::math::atanh(ct(mzero, zero));
   BOOST_CHECK(result.real() == zero);
   BOOST_CHECK(result.imag() == zero);

   result = boost::math::atanh(ct(zero, mzero));
   BOOST_CHECK(result.real() == zero);
   BOOST_CHECK(result.imag() == zero);

   result = boost::math::atanh(ct(mzero, mzero));
   BOOST_CHECK(result.real() == zero);
   BOOST_CHECK(result.imag() == zero);

   if(test_nan)
   {
      result = boost::math::atanh(ct(zero, nan));
      BOOST_CHECK(result.real() == zero);
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::atanh(ct(-zero, nan));
      BOOST_CHECK(result.real() == zero);
      BOOST_CHECK((boost::math::isnan)(result.imag()));
   }

   if(test_infinity)
   {
      result = boost::math::atanh(ct(one, zero));
      BOOST_CHECK_EQUAL(result.real(), infinity);
      BOOST_CHECK_EQUAL(result.imag(), zero);

      result = boost::math::atanh(ct(-one, zero));
      BOOST_CHECK_EQUAL(result.real(), -infinity);
      BOOST_CHECK_EQUAL(result.imag(), zero);

      result = boost::math::atanh(ct(-one, -zero));
      BOOST_CHECK_EQUAL(result.real(), -infinity);
      BOOST_CHECK_EQUAL(result.imag(), zero);

      result = boost::math::atanh(ct(one, -zero));
      BOOST_CHECK_EQUAL(result.real(), infinity);
      BOOST_CHECK_EQUAL(result.imag(), zero);

      result = boost::math::atanh(ct(pi, infinity));
      BOOST_CHECK_EQUAL(result.real(), zero);
      BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);

      result = boost::math::atanh(ct(pi, -infinity));
      BOOST_CHECK_EQUAL(result.real(), zero);
      BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);

      result = boost::math::atanh(ct(-pi, -infinity));
      BOOST_CHECK_EQUAL(result.real(), zero);
      BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);

      result = boost::math::atanh(ct(-pi, infinity));
      BOOST_CHECK_EQUAL(result.real(), zero);
      BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);
   }
   if(test_nan)
   {
      result = boost::math::atanh(ct(pi, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::atanh(ct(-pi, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));
   }

   if(test_infinity)
   {
      result = boost::math::atanh(ct(infinity, pi));
      BOOST_CHECK(result.real() == zero);
      BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);

      result = boost::math::atanh(ct(infinity, -pi));
      BOOST_CHECK_EQUAL(result.real(), zero);
      BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);

      result = boost::math::atanh(ct(-infinity, -pi));
      BOOST_CHECK_EQUAL(result.real(), zero);
      BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);

      result = boost::math::atanh(ct(-infinity, pi));
      BOOST_CHECK_EQUAL(result.real(), zero);
      BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);

      result = boost::math::atanh(ct(infinity, infinity));
      BOOST_CHECK_EQUAL(result.real(), zero);
      BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);

      result = boost::math::atanh(ct(infinity, -infinity));
      BOOST_CHECK_EQUAL(result.real(), zero);
      BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);

      result = boost::math::atanh(ct(-infinity, -infinity));
      BOOST_CHECK_EQUAL(result.real(), zero);
      BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);

      result = boost::math::atanh(ct(-infinity, infinity));
      BOOST_CHECK_EQUAL(result.real(), zero);
      BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);
   }

   if(test_nan)
   {
      result = boost::math::atanh(ct(infinity, nan));
      BOOST_CHECK(result.real() == 0);
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::atanh(ct(-infinity, nan));
      BOOST_CHECK(result.real() == 0);
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::atanh(ct(nan, pi));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::atanh(ct(nan, -pi));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

      result = boost::math::atanh(ct(nan, infinity));
      BOOST_CHECK(result.real() == 0);
      BOOST_CHECK_CLOSE(result.imag(), half_pi, eps*200);

      result = boost::math::atanh(ct(nan, -infinity));
      BOOST_CHECK(result.real() == 0);
      BOOST_CHECK_CLOSE(result.imag(), -half_pi, eps*200);

      result = boost::math::atanh(ct(nan, nan));
      BOOST_CHECK((boost::math::isnan)(result.real()));
      BOOST_CHECK((boost::math::isnan)(result.imag()));

   }
   if(boost::math::signbit(mzero))
   {
      result = boost::math::atanh(ct(-2.0f, mzero));
      BOOST_CHECK(result.real() < 0);
      BOOST_CHECK(result.imag() < 0);
   }
}

//
// test_boundaries:
// This is an accuracy test, sets the real and imaginary components
// of the input argument to various "boundary conditions" that exist
// inside the implementation.  Then computes the result at double precision
// and again at float precision.  The double precision result will be
// computed using the "regular" code, where as the float precision versions
// will calculate the result using the "exceptional value" handlers, so
// we end up comparing the values calculated by two different methods.
//
const float boundaries[] = {
   0,
   1,
   2,
   (std::numeric_limits<float>::max)(),
   (std::numeric_limits<float>::min)(),
   std::numeric_limits<float>::epsilon(),
   std::sqrt((std::numeric_limits<float>::max)()) / 8,
   static_cast<float>(4) * std::sqrt((std::numeric_limits<float>::min)()),
   0.6417F,
   1.5F,
   std::sqrt((std::numeric_limits<float>::max)()) / 2,
   std::sqrt((std::numeric_limits<float>::min)()),
   1.0F / 0.3F,
};

void do_test_boundaries(float x, float y)
{
   std::complex<float> r1 = boost::math::asin(std::complex<float>(x, y));
   std::complex<double> dr = boost::math::asin(std::complex<double>(x, y));
   std::complex<float> r2(static_cast<float>(dr.real()), static_cast<float>(dr.imag()));
   check_complex(r2, r1, 5);
   r1 = boost::math::acos(std::complex<float>(x, y));
   dr = boost::math::acos(std::complex<double>(x, y));
   r2 = std::complex<float>(std::complex<double>(dr.real(), dr.imag()));
   check_complex(r2, r1, 5);
   r1 = boost::math::atanh(std::complex<float>(x, y));
   dr = boost::math::atanh(std::complex<double>(x, y));
   r2 = std::complex<float>(std::complex<double>(dr.real(), dr.imag()));
   check_complex(r2, r1, 5);
}

void test_boundaries(float x, float y)
{
   do_test_boundaries(x, y);
   do_test_boundaries(-x, y); 
   do_test_boundaries(-x, -y);
   do_test_boundaries(x, -y);
}

void test_boundaries(float x)
{
   for(unsigned i = 0; i < sizeof(boundaries)/sizeof(float); ++i)
   {
      test_boundaries(x, boundaries[i]);
      test_boundaries(x, boundaries[i] + std::numeric_limits<float>::epsilon()*boundaries[i]);
      test_boundaries(x, boundaries[i] - std::numeric_limits<float>::epsilon()*boundaries[i]);
   }
}

void test_boundaries()
{
   for(unsigned i = 0; i < sizeof(boundaries)/sizeof(float); ++i)
   {
      test_boundaries(boundaries[i]);
      test_boundaries(boundaries[i] + std::numeric_limits<float>::epsilon()*boundaries[i]);
      test_boundaries(boundaries[i] - std::numeric_limits<float>::epsilon()*boundaries[i]);//here
   }
}


BOOST_AUTO_TEST_CASE( test_main )
{
   std::cout << "Running complex trig sanity checks for type float." << std::endl;
   test_inverse_trig(float(0));
   std::cout << "Running complex trig sanity checks for type double." << std::endl;
   test_inverse_trig(double(0));
   //test_inverse_trig((long double)(0));

   std::cout << "Running complex trig spot checks for type float." << std::endl;
   check_spots(float(0));
   std::cout << "Running complex trig spot checks for type double." << std::endl;
   check_spots(double(0));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   std::cout << "Running complex trig spot checks for type long double." << std::endl;
   check_spots((long double)(0));
#endif

   std::cout << "Running complex trig boundary and accuracy tests." << std::endl;
   test_boundaries();
}



