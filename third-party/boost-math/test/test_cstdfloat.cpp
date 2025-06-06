//  Copyright Christopher Kormanyos 2014.
//  Copyright John Maddock 2014.
//  Copyright Paul A. Bristow 2014.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <complex>
#include <limits>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <boost/cstdfloat.hpp>
#include <boost/math/tools/big_constant.hpp>

#ifdef _MSC_VER
#  pragma warning(disable : 4127) // conditional expression is constant.
#  pragma warning(disable : 4512) // assignment operator could not be generated.
#  pragma warning(disable : 4996) // use -D_SCL_SECURE_NO_WARNINGS.
#endif

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/scoped_array.hpp>

//
// We need to define an iostream operator for __float128 in order to
// compile this test:
//

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the implementation of floating-point typedefs having
// specified widths, as implemented in <boost/cstdfloat.hpp> and described
// in N3626 (proposed for C++14).

// For more information on <boost/cstdfloat.hpp> and the corresponding
// proposal of "Floating-Point Typedefs Having Specified Widths",
// see: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3626.pdf

// The tests:
//
// Perform sanity checks on boost::float16_t, boost::float32_t,
// boost::float64_t, boost::float80_t, and boost::float128_t when
// these types are available. In the sanity checks, we verify the
// formal behavior of the types and the macros for creating literal
// floating-point constants.
//
// An extended check is included for boost::float128_t. This checks
// the functionality of <cmath>, I/O stream operations, and <complex>
// functions for boost::float128_t.

// For some reason the (x != x) check fails on Mingw:
#if !defined(__MINGW64__)
#define TEST_CSTDFLOAT_SANITY_CHECK_NAN(the_digits)                                                  \
  {                                                                                                  \
    using std::sqrt;                                                                                 \
    const float_type x = sqrt(float_type(test_cstdfloat::minus_one));                                \
    const bool the_nan_test = (   std::numeric_limits<float_type>::has_quiet_NaN                     \
                               && (x != x));                                                         \
    BOOST_CHECK_EQUAL( the_nan_test, true );                                                         \
  }  
#else
#define TEST_CSTDFLOAT_SANITY_CHECK_NAN(the_digits)                                                  
#endif

#define TEST_CSTDFLOAT_SANITY_CHECK(the_digits)                                                      \
void sanity_check_##the_digits##_func()                                                              \
{                                                                                                    \
  typedef boost::float##the_digits##_t float_type;                                                   \
                                                                                                     \
  constexpr int my_digits10 = std::numeric_limits<float_type>::digits10;              \
                                                                                                     \
  {                                                                                                  \
    constexpr float_type x =                                                          \
      BOOST_FLOAT##the_digits##_C(0.33333333333333333333333333333333333333333);                      \
    std::stringstream ss;                                                                            \
    ss << std::setprecision(my_digits10 - 1)                                                         \
       << x;                                                                                         \
    std::string str = "0.";                                                                          \
    str += std::string(std::string::size_type(my_digits10 - 1), char('3'));                          \
    BOOST_CHECK_EQUAL( ss.str(), str );                                                              \
  }                                                                                                  \
  {                                                                                                  \
    constexpr float_type x =                                                          \
      BOOST_FLOAT##the_digits##_C(0.66666666666666666666666666666666666666666);                      \
    std::stringstream ss;                                                                            \
    ss << std::setprecision(my_digits10 - 1)                                                         \
       << x;                                                                                         \
    std::string str = "0.";                                                                          \
    str += std::string(std::string::size_type(my_digits10 - 2), char('6'));                          \
    str += "7";                                                                                      \
    BOOST_CHECK_EQUAL( ss.str(), str );                                                              \
  }                                                                                                  \
  {                                                                                                  \
    const float_type x = BOOST_FLOAT##the_digits##_C(1.0) / test_cstdfloat::zero;                    \
    const bool the_inf_test = (   std::numeric_limits<float_type>::has_infinity                      \
                               && (x == std::numeric_limits<float_type>::infinity()));               \
    BOOST_CHECK_EQUAL( the_inf_test, true );                                                         \
  }                                                                                                  \
  TEST_CSTDFLOAT_SANITY_CHECK_NAN(the_digits)\
  {                                                                                                  \
    const bool the_lim_test =                                                                        \
      (std::numeric_limits<boost::floatmax_t>::digits >= std::numeric_limits<float_type>::digits);   \
    BOOST_CHECK_EQUAL( the_lim_test, true );                                                         \
  }                                                                                                  \
}

namespace test_cstdfloat
{
#if defined(BOOST_FLOAT128_C)

   template <class T, class U>
   void test_less(T a, U b)
   {
      BOOST_CHECK(a < b);
      BOOST_CHECK(a <= b);
      BOOST_CHECK(!(a > b));
      BOOST_CHECK(!(a >= b));
      BOOST_CHECK(!(a == b));
      BOOST_CHECK((a != b));

      BOOST_CHECK(b > a);
      BOOST_CHECK(b >= a);
      BOOST_CHECK(!(b < a));
      BOOST_CHECK(!(b <= a));
      BOOST_CHECK(!(b == a));
      BOOST_CHECK((b != a));

      BOOST_CHECK(std::isless(a, b));
      BOOST_CHECK(std::islessequal(a, b));
      BOOST_CHECK(!std::isgreater(a, b));
      BOOST_CHECK(!std::isgreaterequal(a, b));
      BOOST_CHECK(std::islessgreater(a, b));

      BOOST_CHECK(!std::isless(b, a));
      BOOST_CHECK(!std::islessequal(b, a));
      BOOST_CHECK(std::isgreater(b, a));
      BOOST_CHECK(std::isgreaterequal(b, a));
      BOOST_CHECK(std::islessgreater(b, a));
   }
   template <class T, class U>
   void test_equal(T a, U b)
   {
      BOOST_CHECK(!(a < b));
      BOOST_CHECK(a <= b);
      BOOST_CHECK(!(a > b));
      BOOST_CHECK((a >= b));
      BOOST_CHECK((a == b));
      BOOST_CHECK(!(a != b));

      BOOST_CHECK(!(b > a));
      BOOST_CHECK(b >= a);
      BOOST_CHECK(!(b < a));
      BOOST_CHECK((b <= a));
      BOOST_CHECK((b == a));
      BOOST_CHECK(!(b != a));

      BOOST_CHECK(!std::isless(a, b));
      BOOST_CHECK(std::islessequal(a, b));
      BOOST_CHECK(!std::isgreater(a, b));
      BOOST_CHECK(std::isgreaterequal(a, b));
      BOOST_CHECK(!std::islessgreater(a, b));

      BOOST_CHECK(!std::isless(b, a));
      BOOST_CHECK(std::islessequal(b, a));
      BOOST_CHECK(!std::isgreater(b, a));
      BOOST_CHECK(std::isgreaterequal(b, a));
      BOOST_CHECK(!std::islessgreater(b, a));
   }
   template <class T, class U>
   void test_unordered(T a, U b)
   {
      BOOST_CHECK(!(a < b));
      BOOST_CHECK(!(a <= b));
      BOOST_CHECK(!(a > b));
      BOOST_CHECK(!(a >= b));
      BOOST_CHECK(!(a == b));
      BOOST_CHECK((a != b));

      BOOST_CHECK(!(b > a));
      BOOST_CHECK(!(b >= a));
      BOOST_CHECK(!(b < a));
      BOOST_CHECK(!(b <= a));
      BOOST_CHECK(!(b == a));
      BOOST_CHECK((b != a));

      BOOST_CHECK(!std::isless(a, b));
      BOOST_CHECK(!std::islessequal(a, b));
      BOOST_CHECK(!std::isgreater(a, b));
      BOOST_CHECK(!std::isgreaterequal(a, b));
      BOOST_CHECK(!std::islessgreater(a, b));

      BOOST_CHECK(!std::isless(b, a));
      BOOST_CHECK(!std::islessequal(b, a));
      BOOST_CHECK(!std::isgreater(b, a));
      BOOST_CHECK(!std::isgreaterequal(b, a));
      BOOST_CHECK(!std::islessgreater(b, a));
   }

   template <class T>
   void test()
   {
      //
      // Basic sanity checks for C99 functions which are just imported versions
      // from Boost.Math.  These should still be found via ADL so no using declarations here...
      //
      T val = 2;
      BOOST_CHECK(std::signbit(val) == 0);
      BOOST_CHECK(std::signbit(val + 2) == 0);
      val = -val;
      BOOST_CHECK(std::signbit(val));
      BOOST_CHECK(std::signbit(val * 2));

      T s = 2;
      val = 3;
      BOOST_CHECK_EQUAL(std::copysign(val, s), 3);
      BOOST_CHECK_EQUAL(std::copysign(val, s * -2), -3);
      BOOST_CHECK_EQUAL(std::copysign(-2 * val, s), 6);
      BOOST_CHECK_EQUAL(std::copysign(-2 * val, 2 * s), 6);
      s = -2;
      BOOST_CHECK_EQUAL(std::copysign(val, s), -3);
      BOOST_CHECK_EQUAL(std::copysign(val, s * -2), 3);
      BOOST_CHECK_EQUAL(std::copysign(-2 * val, s), -6);
      BOOST_CHECK_EQUAL(std::copysign(-2 * val, 2 * s), -6);
      val = -3;
      BOOST_CHECK_EQUAL(std::copysign(val, s), -3);
      BOOST_CHECK_EQUAL(std::copysign(val, s * -2), 3);
      BOOST_CHECK_EQUAL(std::copysign(-2 * val, s), -6);
      BOOST_CHECK_EQUAL(std::copysign(-2 * val, 2 * s), -6);
      s = 0;
      BOOST_CHECK_EQUAL(std::copysign(val, s), 3);
      BOOST_CHECK_EQUAL(std::copysign(val, s * -2), -3);

      BOOST_CHECK_EQUAL(std::copysign(-2 * val, s), 6);
      BOOST_CHECK_EQUAL(std::copysign(-2 * val, 2 * s), 6);
      // Things involving signed zero, need to detect it first:

      val = 3;
      BOOST_CHECK_EQUAL(std::fpclassify(val), FP_NORMAL);
      BOOST_CHECK_EQUAL(std::fpclassify(val * 3), FP_NORMAL);
      BOOST_CHECK(!std::isinf(val));
      BOOST_CHECK(!std::isinf(val + 2));
      BOOST_CHECK(!std::isnan(val));
      BOOST_CHECK(!std::isnan(val + 2));
      BOOST_CHECK(std::isnormal(val));
      BOOST_CHECK(std::isnormal(val + 2));
      val = -3;
      BOOST_CHECK_EQUAL(std::fpclassify(val), FP_NORMAL);
      BOOST_CHECK_EQUAL(std::fpclassify(val * 3), FP_NORMAL);
      BOOST_CHECK(!std::isinf(val));
      BOOST_CHECK(!std::isinf(val + 2));
      BOOST_CHECK(!std::isnan(val));
      BOOST_CHECK(!std::isnan(val + 2));
      BOOST_CHECK(std::isnormal(val));
      BOOST_CHECK(std::isnormal(val + 2));
      val = 0;
      BOOST_CHECK_EQUAL(std::fpclassify(val), FP_ZERO);
      BOOST_CHECK_EQUAL(std::fpclassify(val * 3), FP_ZERO);
      BOOST_CHECK(!std::isinf(val));
      BOOST_CHECK(!std::isinf(val + 2));
      BOOST_CHECK(!std::isnan(val));
      BOOST_CHECK(!std::isnan(val + 2));
      BOOST_CHECK(!std::isnormal(val));
      BOOST_CHECK(!std::isnormal(val * 2));
      BOOST_CHECK(!std::isnormal(val * -2));
      if (std::numeric_limits<T>::has_infinity)
      {
         val = std::numeric_limits<T>::infinity();
         BOOST_CHECK_EQUAL(std::fpclassify(val), FP_INFINITE);
         BOOST_CHECK_EQUAL(std::fpclassify(val * 3), FP_INFINITE);
         BOOST_CHECK(std::isinf(val));
         BOOST_CHECK(std::isinf(val + 2));
         BOOST_CHECK(!std::isnan(val));
         BOOST_CHECK(!std::isnan(val + 2));
         BOOST_CHECK(!std::isnormal(val));
         BOOST_CHECK(!std::isnormal(val + 2));
         val = -val;
         BOOST_CHECK_EQUAL(std::fpclassify(val), FP_INFINITE);
         BOOST_CHECK_EQUAL(std::fpclassify(val * 3), FP_INFINITE);
         BOOST_CHECK(std::isinf(val));
         BOOST_CHECK(std::isinf(val + 2));
         BOOST_CHECK(!std::isnan(val));
         BOOST_CHECK(!std::isnan(val + 2));
         BOOST_CHECK(!std::isnormal(val));
         BOOST_CHECK(!std::isnormal(val + 2));
      }
      if (std::numeric_limits<T>::has_quiet_NaN)
      {
         val = std::numeric_limits <T>::quiet_NaN();
         BOOST_CHECK_EQUAL(std::fpclassify(val), FP_NAN);
         BOOST_CHECK_EQUAL(std::fpclassify(val * 3), FP_NAN);
         BOOST_CHECK(!std::isinf(val));
         BOOST_CHECK(!std::isinf(val + 2));
         BOOST_CHECK(std::isnan(val));
         BOOST_CHECK(std::isnan(val + 2));
         BOOST_CHECK(!std::isnormal(val));
         BOOST_CHECK(!std::isnormal(val + 2));
      }
      s = 8 * std::numeric_limits<T>::epsilon();
      val = 2.5;

      BOOST_CHECK_CLOSE_FRACTION(std::asinh(val), T(BOOST_MATH_LARGEST_FLOAT_C(1.6472311463710957106248586104436196635044144301932365282203100930843983757633104078778420255069424907777006132075516484778755360595913172299093829522950397895699619540523579875476513967578478619028438291006578604823887119907434)), s);
      BOOST_CHECK_CLOSE_FRACTION(std::acosh(val), T(BOOST_MATH_LARGEST_FLOAT_C(1.5667992369724110786640568625804834938620823510926588639329459980122148134693922696279968499622201141051039184050936311066453565386393240356562374302417843319480223211857615778787272615171906055455922537080327062362258846337050)), s);
      val = 0.5;
      BOOST_CHECK_CLOSE_FRACTION(std::atanh(val), T(BOOST_MATH_LARGEST_FLOAT_C(0.5493061443340548456976226184612628523237452789113747258673471668187471466093044834368078774068660443939850145329789328711840021129652599105264009353836387053015813845916906835896868494221804799518712851583979557605727959588753)), s);
      val = 55.25;
      BOOST_CHECK_CLOSE_FRACTION(std::cbrt(val), T(BOOST_MATH_LARGEST_FLOAT_C(3.8087058015466360309383876359583281382991983919300128125378938779672144843676192684301168479657279498120767424724024965319869248797423276064015643361426189576415670917818313417529572608229017809069355688606687557031643655896118)), s);
      val = 2.75;
      BOOST_CHECK_CLOSE_FRACTION(std::erf(val), T(BOOST_MATH_LARGEST_FLOAT_C(0.9998993780778803631630956080249130432349352621422640655161095794654526422025908961447328296681056892975214344779300734620255391682713519265048496199034963706976420982849598189071465666866369396765001072187538732800143945532487)), s);
      BOOST_CHECK_CLOSE_FRACTION(std::erfc(val), T(BOOST_MATH_LARGEST_FLOAT_C(0.0001006219221196368369043919750869567650647378577359344838904205345473577974091038552671703318943107024785655220699265379744608317286480734951503800965036293023579017150401810928534333133630603234998927812461267199856054467512)), s);
      val = 0.125;
      BOOST_CHECK_CLOSE_FRACTION(std::expm1(val), T(BOOST_MATH_LARGEST_FLOAT_C(0.1331484530668263168290072278117938725655031317451816259128200360788235778800483865139399907949417285732315270156473075657048210452584733998785564025916995261162759280700397984729320345630340659469435372721057879969170503978449)), s);

      val = 20;
      s = 2;
      BOOST_CHECK_EQUAL(std::fdim(val, s), 18);
      BOOST_CHECK_EQUAL(std::fdim(s, val), 0);
      BOOST_CHECK_EQUAL(std::fdim(val, s * 2), 16);
      BOOST_CHECK_EQUAL(std::fdim(s * 2, val), 0);
      BOOST_CHECK_EQUAL(std::fdim(val, 2), 18);
      BOOST_CHECK_EQUAL(std::fdim(2, val), 0);

      BOOST_CHECK_EQUAL(std::fmax(val, s), val);
      BOOST_CHECK_EQUAL(std::fmax(s, val), val);
      BOOST_CHECK_EQUAL(std::fmax(val * 2, s), val * 2);
      BOOST_CHECK_EQUAL(std::fmax(val, s * 2), val);
      BOOST_CHECK_EQUAL(std::fmax(val * 2, s * 2), val * 2);
      BOOST_CHECK_EQUAL(std::fmin(val, s), s);
      BOOST_CHECK_EQUAL(std::fmin(s, val), s);
      BOOST_CHECK_EQUAL(std::fmin(val * 2, s), s);
      BOOST_CHECK_EQUAL(std::fmin(val, s * 2), s * 2);
      BOOST_CHECK_EQUAL(std::fmin(val * 2, s * 2), s * 2);

      BOOST_CHECK_EQUAL(std::fmax(val, 2), val);
      BOOST_CHECK_EQUAL(std::fmax(val, 2.0), val);
      BOOST_CHECK_EQUAL(std::fmax(20, s), val);
      BOOST_CHECK_EQUAL(std::fmax(20.0, s), val);
      BOOST_CHECK_EQUAL(std::fmin(val, 2), s);
      BOOST_CHECK_EQUAL(std::fmin(val, 2.0), s);
      BOOST_CHECK_EQUAL(std::fmin(20, s), s);
      BOOST_CHECK_EQUAL(std::fmin(20.0, s), s);
      if (std::numeric_limits<T>::has_quiet_NaN)
      {
         BOOST_CHECK_EQUAL(std::fmax(val, std::numeric_limits<T>::quiet_NaN()), val);
         BOOST_CHECK_EQUAL(std::fmax(std::numeric_limits<T>::quiet_NaN(), val), val);
         BOOST_CHECK_EQUAL(std::fmin(val, std::numeric_limits<T>::quiet_NaN()), val);
         BOOST_CHECK_EQUAL(std::fmin(std::numeric_limits<T>::quiet_NaN(), val), val);
      }
      if (std::numeric_limits<double>::has_quiet_NaN)
      {
         BOOST_CHECK_EQUAL(std::fmax(val, std::numeric_limits<double>::quiet_NaN()), val);
         BOOST_CHECK_EQUAL(std::fmax(std::numeric_limits<double>::quiet_NaN(), val), val);
         BOOST_CHECK_EQUAL(std::fmin(val, std::numeric_limits<double>::quiet_NaN()), val);
         BOOST_CHECK_EQUAL(std::fmin(std::numeric_limits<double>::quiet_NaN(), val), val);
      }

      test_less(s, val);
      test_less(2, val);
      test_less(s, 20);
      test_less(s + 0, val);
      test_less(s, val * 1);
      test_less(s * 1, val * 1);
      test_less(s * 1, 20);
      test_less(s + 2, val * 2);

      test_equal(val, val);
      test_equal(20, val);
      test_equal(val, 20);
      test_equal(val + 0, val);
      test_equal(val, val * 1);
      test_equal(val * 1, val * 1);
      test_equal(val * 1, 20);
      test_equal(val * 20, val * 20);

      if (std::numeric_limits<T>::has_quiet_NaN)
      {
         s = std::numeric_limits<T>::quiet_NaN();
         test_unordered(s, val);
         test_unordered(s, 20);
         test_unordered(s + 0, val);
         test_unordered(s, val * 1);
         test_unordered(s * 1, val * 1);
         test_unordered(s * 1, 20);
         test_unordered(s + 2, val * 2);
         if (std::numeric_limits<double>::has_quiet_NaN)
         {
            double n = std::numeric_limits<double>::quiet_NaN();
            test_unordered(n, val);
         }
      }

      T tol = 8 * std::numeric_limits<T>::epsilon();
      s = 2;

      BOOST_CHECK_CLOSE_FRACTION(T(std::hypot(val, s)), T(BOOST_MATH_LARGEST_FLOAT_C(20.099751242241780540438529825519152373890046940052754581145656594656982463103940762472355384907904704732599006530)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::hypot(val, 2)), T(BOOST_MATH_LARGEST_FLOAT_C(20.099751242241780540438529825519152373890046940052754581145656594656982463103940762472355384907904704732599006530)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::hypot(val, 2.0)), T(BOOST_MATH_LARGEST_FLOAT_C(20.099751242241780540438529825519152373890046940052754581145656594656982463103940762472355384907904704732599006530)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::hypot(20, s)), T(BOOST_MATH_LARGEST_FLOAT_C(20.099751242241780540438529825519152373890046940052754581145656594656982463103940762472355384907904704732599006530)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::hypot(20.0, s)), T(BOOST_MATH_LARGEST_FLOAT_C(20.099751242241780540438529825519152373890046940052754581145656594656982463103940762472355384907904704732599006530)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::hypot(val * 1, s)), T(BOOST_MATH_LARGEST_FLOAT_C(20.099751242241780540438529825519152373890046940052754581145656594656982463103940762472355384907904704732599006530)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::hypot(val * 1, s * 1)), T(BOOST_MATH_LARGEST_FLOAT_C(20.099751242241780540438529825519152373890046940052754581145656594656982463103940762472355384907904704732599006530)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::hypot(val * 1, 2)), T(BOOST_MATH_LARGEST_FLOAT_C(20.099751242241780540438529825519152373890046940052754581145656594656982463103940762472355384907904704732599006530)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::hypot(val * 1, 2.0)), T(BOOST_MATH_LARGEST_FLOAT_C(20.099751242241780540438529825519152373890046940052754581145656594656982463103940762472355384907904704732599006530)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::hypot(20, s * 1)), T(BOOST_MATH_LARGEST_FLOAT_C(20.099751242241780540438529825519152373890046940052754581145656594656982463103940762472355384907904704732599006530)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::hypot(20.0, s * 1)), T(BOOST_MATH_LARGEST_FLOAT_C(20.099751242241780540438529825519152373890046940052754581145656594656982463103940762472355384907904704732599006530)), tol);

      BOOST_CHECK_CLOSE_FRACTION(std::lgamma(val), T(BOOST_MATH_LARGEST_FLOAT_C(39.339884187199494036224652394567381081691457206897853119937969989377572554993874476249340525204204720861169039582)), tol);
      BOOST_CHECK_CLOSE_FRACTION(std::lgamma(val + 0), T(BOOST_MATH_LARGEST_FLOAT_C(39.339884187199494036224652394567381081691457206897853119937969989377572554993874476249340525204204720861169039582)), tol);

      BOOST_CHECK_EQUAL(std::lrint(val), 20);
      BOOST_CHECK_EQUAL(std::lrint(val * 2), 40);
      BOOST_CHECK_EQUAL(std::llrint(val), 20);
      BOOST_CHECK_EQUAL(std::llrint(val * 2), 40);

      val = 0.125;
      BOOST_CHECK_CLOSE_FRACTION(std::log1p(val), T(BOOST_MATH_LARGEST_FLOAT_C(0.117783035656383454538794109470521705068480712564733141107348638794807720528133786929641528638208114949935615070)), tol);
      BOOST_CHECK_CLOSE_FRACTION(std::log1p(val + 0), T(BOOST_MATH_LARGEST_FLOAT_C(0.117783035656383454538794109470521705068480712564733141107348638794807720528133786929641528638208114949935615070)), tol);
      val = 20;
      BOOST_CHECK_CLOSE_FRACTION(T(std::log2(val)), T(BOOST_MATH_LARGEST_FLOAT_C(4.321928094887362347870319429489390175864831393024580612054756395815934776608625215850139743359370155099657371710)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::log2(val + 0)), T(BOOST_MATH_LARGEST_FLOAT_C(4.321928094887362347870319429489390175864831393024580612054756395815934776608625215850139743359370155099657371710)), tol);

      BOOST_CHECK_EQUAL(T(std::nearbyint(val)), 20);
      BOOST_CHECK_EQUAL(T(std::nearbyint(val + 0.25)), 20);
      BOOST_CHECK_EQUAL(T(std::rint(val)), 20);
      BOOST_CHECK_EQUAL(T(std::rint(val + 0.25)), 20);

      BOOST_CHECK_GT(std::nextafter(val, T(200)), val);
      BOOST_CHECK_GT(std::nextafter(val + 0, T(200)), val);
      BOOST_CHECK_GT(std::nextafter(val + 0, T(200) + 1), val);
      BOOST_CHECK_GT(std::nextafter(val, T(200) + 1), val);

      BOOST_CHECK_GT(std::nexttoward(val, T(200)), val);
      BOOST_CHECK_GT(std::nexttoward(val + 0, T(200)), val);
      BOOST_CHECK_GT(std::nexttoward(val + 0, T(200) + 1), val);
      BOOST_CHECK_GT(std::nexttoward(val, T(200) + 1), val);

      val = 21;
      s = 5;
      BOOST_CHECK_EQUAL(T(std::remainder(val, s)), 1);
      BOOST_CHECK_EQUAL(T(std::remainder(val, 5)), 1);
      BOOST_CHECK_EQUAL(T(std::remainder(21, s)), 1);
      BOOST_CHECK_EQUAL(T(std::remainder(val * 1, s)), 1);
      BOOST_CHECK_EQUAL(T(std::remainder(val * 1, s * 1)), 1);
      BOOST_CHECK_EQUAL(T(std::remainder(val, s * 1)), 1);
      BOOST_CHECK_EQUAL(T(std::remainder(val * 1, 5)), 1);
      BOOST_CHECK_EQUAL(T(std::remainder(21, s * 1)), 1);
      int i(0);
      BOOST_CHECK_EQUAL(T(std::remquo(val, s, &i)), 1);
      BOOST_CHECK_EQUAL(i, 4);
      i = 0;
      BOOST_CHECK_EQUAL(T(std::remquo(val, 5, &i)), 1);
      BOOST_CHECK_EQUAL(i, 4);
      i = 0;
      BOOST_CHECK_EQUAL(T(std::remquo(21, s, &i)), 1);
      BOOST_CHECK_EQUAL(i, 4);
      i = 0;
      BOOST_CHECK_EQUAL(T(std::remquo(val * 1, s, &i)), 1);
      BOOST_CHECK_EQUAL(i, 4);
      i = 0;
      BOOST_CHECK_EQUAL(T(std::remquo(val * 1, s * 1, &i)), 1);
      BOOST_CHECK_EQUAL(i, 4);
      i = 0;
      BOOST_CHECK_EQUAL(T(std::remquo(val, s * 1, &i)), 1);
      BOOST_CHECK_EQUAL(i, 4);
      i = 0;
      BOOST_CHECK_EQUAL(T(std::remquo(val * 1, 5, &i)), 1);
      BOOST_CHECK_EQUAL(i, 4);
      i = 0;
      BOOST_CHECK_EQUAL(T(std::remquo(21, s * 1, &i)), 1);
      BOOST_CHECK_EQUAL(i, 4);
      i = 0;
      val = 5.25;
      tol = 3000;

      BOOST_CHECK_CLOSE_FRACTION(std::tgamma(val), T(BOOST_MATH_LARGEST_FLOAT_C(35.211611852799685705225257690531248115026311138908448314086859575901217653313145619623624570033258659272301335544)), tol);
      BOOST_CHECK_CLOSE_FRACTION(std::tgamma(val + 1), T(BOOST_MATH_LARGEST_FLOAT_C(184.86096222719834995243260287528905260388813347926935364895601277348139267989401450302402899267460796117958201160)), tol);

      BOOST_CHECK_CLOSE_FRACTION(T(std::exp2(val)), T(BOOST_MATH_LARGEST_FLOAT_C(38.054627680087074134959999057935229289375106958842157216608071191022933383261349115865003025220405558913196632792)), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::exp2(val + 1)), T(BOOST_MATH_LARGEST_FLOAT_C(76.109255360174148269919998115870458578750213917684314433216142382045866766522698231730006050440811117826393265585)), tol);
      
      val = 15;
      BOOST_CHECK_CLOSE_FRACTION(T(std::exp2(val)), T(32768uL), tol);
      BOOST_CHECK_CLOSE_FRACTION(T(std::exp2(val + 1)), T(65536uL), tol);

      i = std::fpclassify(val) + std::isgreaterequal(val, s) + std::islessequal(val, s) + std::isnan(val) + std::isunordered(val, s)
         + std::isfinite(val) + std::isinf(val) + std::islessgreater(val, s) + std::isnormal(val) + std::signbit(val) + std::isgreater(val, s) + std::isless(val, s);
   }

#endif

   int zero;
   int minus_one;

#if defined(BOOST_FLOATMAX_C)
   constexpr int has_floatmax_t = 1;
#else
   constexpr int has_floatmax_t = 0;
#endif

#if defined(BOOST_FLOAT16_C)
   TEST_CSTDFLOAT_SANITY_CHECK(16)
#endif

#if defined(BOOST_FLOAT32_C)
      TEST_CSTDFLOAT_SANITY_CHECK(32)
#endif

#if defined(BOOST_FLOAT64_C)
      TEST_CSTDFLOAT_SANITY_CHECK(64)
#endif

#if defined(BOOST_FLOAT80_C)
      TEST_CSTDFLOAT_SANITY_CHECK(80)
#endif

#if defined(BOOST_FLOAT128_C)
      TEST_CSTDFLOAT_SANITY_CHECK(128)

      void extend_check_128_func()
   {
      test<boost::float128_t>();
   }
#endif // defined (BOOST_FLOAT128_C)
}

BOOST_AUTO_TEST_CASE(test_main)
{
   test_cstdfloat::zero = 0;
   test_cstdfloat::minus_one = -1;

   // Perform basic sanity checks that verify both the existence of the proper
   // floating-point literal macros as well as the correct digit handling
   // for a given floating-point typedef having specified width.

   BOOST_CHECK_EQUAL(test_cstdfloat::has_floatmax_t, 1);

#if defined(BOOST_FLOAT16_C)
   test_cstdfloat::sanity_check_16_func();
#endif

#if defined(BOOST_FLOAT32_C)
   test_cstdfloat::sanity_check_32_func();
#endif

#if defined(BOOST_FLOAT64_C)
   test_cstdfloat::sanity_check_64_func();
#endif

#if defined(BOOST_FLOAT80_C)
   test_cstdfloat::sanity_check_80_func();
#endif

#if defined(BOOST_FLOAT128_C)
   test_cstdfloat::sanity_check_128_func();

   // Perform an extended check of boost::float128_t including
   // a variety of functions from the C++ standard library.
   test_cstdfloat::extend_check_128_func();
#endif // defined (BOOST_FLOAT128_C)
}
