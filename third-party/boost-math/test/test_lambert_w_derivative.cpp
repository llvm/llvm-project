// Copyright Paul A. Bristow 2016, 2017, 2018.
// Copyright John Maddock 2016.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// test_lambert_w.cpp
//! \brief Basic sanity tests for Lambert W derivative.

#include <climits>
#include <cfloat>
#if defined(BOOST_MATH_TEST_FLOAT128) && (LDBL_MANT_DIG > 100)
//
// Mixing __float128 and long double results in:
// error: __float128 and long double cannot be used in the same expression
// whenever long double is a [possibly quasi-] quad precision type.
// 
#undef BOOST_MATH_TEST_FLOAT128
#endif

#ifdef BOOST_MATH_TEST_FLOAT128
#include <boost/cstdfloat.hpp> // For float_64_t, float128_t. Must be first include!
#endif // #ifdef #ifdef BOOST_MATH_TEST_FLOAT128
// Needs gnu++17 for BOOST_HAS_FLOAT128
#include <boost/config.hpp>   // for BOOST_MSVC definition etc.
#include <boost/version.hpp>   // for BOOST_MSVC versions.

// Boost macros
#define BOOST_TEST_MAIN
#define BOOST_LIB_DIAGNOSTIC "on" // Report library file details.
#include <boost/test/included/unit_test.hpp> // Boost.Test
// #include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/array.hpp>
#include <boost/type_traits/is_constructible.hpp>

#ifdef BOOST_MATH_TEST_MULTIPRECISION
#include <boost/multiprecision/cpp_dec_float.hpp> // boost::multiprecision::cpp_dec_float_50
using boost::multiprecision::cpp_dec_float_50;

#include <boost/multiprecision/cpp_bin_float.hpp>
using boost::multiprecision::cpp_bin_float_quad;

#ifdef BOOST_MATH_TEST_FLOAT128

#ifdef BOOST_HAS_FLOAT128
// Including this header below without float128 triggers:
// fatal error C1189: #error:  "Sorry compiler is neither GCC, not Intel, don't know how to configure this header."
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif // ifdef BOOST_HAS_FLOAT128
#endif // #ifdef #ifdef BOOST_MATH_TEST_FLOAT128

#endif //   #ifdef BOOST_MATH_TEST_MULTIPRECISION

//#include <boost/fixed_point/fixed_point.hpp> // If available.

#include <boost/math/concepts/real_concept.hpp> // for real_concept tests.
#include <boost/math/special_functions/fpclassify.hpp> // isnan, isfinite.
#include <boost/math/special_functions/next.hpp> // float_next, float_prior
using boost::math::float_next;
using boost::math::float_prior;
#include <boost/math/special_functions/ulp.hpp>  // ulp

#include <boost/math/tools/test_value.hpp>  // for create_test_value and macro BOOST_MATH_TEST_VALUE.
#include <boost/math/policies/policy.hpp>
using boost::math::policies::digits2;
using boost::math::policies::digits10;
#include <boost/math/special_functions/lambert_w.hpp> // For Lambert W lambert_w function.
using boost::math::lambert_wm1;
using boost::math::lambert_w0;

#include <limits>
#include <cmath>
#include <typeinfo>
#include <iostream>
#include <exception>

std::string show_versions(void);

BOOST_AUTO_TEST_CASE( Derivatives_of_lambert_w )
{
  std::cout << "Macro BOOST_MATH_LAMBERT_W_DERIVATIVES to test 1st derivatives is defined." << std::endl;
  BOOST_TEST_MESSAGE("\nTest Lambert W function 1st differentials.");

  using boost::math::constants::exp_minus_one;
  using boost::math::lambert_w0_prime;
  using boost::math::lambert_wm1_prime;

  // Derivatives
  // https://www.wolframalpha.com/input/?i=derivative+of+productlog(0,+x)
  //  d/dx(W_0(x)) = W(x)/(x W(x) + x)
  // https://www.wolframalpha.com/input/?i=derivative+of+productlog(-1,+x)
  // d/dx(W_(-1)(x)) = (W_(-1)(x))/(x W_(-1)(x) + x)

  // 55 decimal digit values added to allow future testing using multiprecision.

  typedef double RealType;

  int epsilons = 1;
  RealType tolerance = boost::math::tools::epsilon<RealType>() * epsilons; // 2 eps as a fraction.

  // derivative of productlog(-1, x)   at x = -0.1 == -13.8803
  // (derivative of productlog(-1, x) ) at x = N[-0.1, 55] - but the result disappears!
  // (derivative of N[productlog(-1, x), 55] ) at x = N[-0.1, 55]

  // W0 branch
  BOOST_CHECK_CLOSE_FRACTION(lambert_w0_prime(BOOST_MATH_TEST_VALUE(RealType, -0.2)),
   // BOOST_MATH_TEST_VALUE(RealType, 1.7491967609218355),
    BOOST_MATH_TEST_VALUE(RealType, 1.7491967609218358355273514903396335693828167746571404),
    tolerance); //                  1.7491967609218358355273514903396335693828167746571404

    BOOST_CHECK_CLOSE_FRACTION(lambert_w0_prime(BOOST_MATH_TEST_VALUE(RealType, 10.)),
    BOOST_MATH_TEST_VALUE(RealType, 0.063577133469345105142021311010780887641928338458371618),
    tolerance);

// W-1 branch
  BOOST_CHECK_CLOSE_FRACTION(lambert_wm1_prime(BOOST_MATH_TEST_VALUE(RealType, -0.1)),
    BOOST_MATH_TEST_VALUE(RealType, -13.880252213229780748699361486619519025203815492277715),
    tolerance);
  // Lambert W_prime -13.880252213229780748699361486619519025203815492277715, double -13.880252213229781

  BOOST_CHECK_CLOSE_FRACTION(lambert_wm1_prime(BOOST_MATH_TEST_VALUE(RealType, -0.2)),
    BOOST_MATH_TEST_VALUE(RealType, -8.2411940564179044961885598641955579728547896392013239),
    tolerance);
  // Lambert W_prime -8.2411940564179044961885598641955579728547896392013239, double -8.2411940564179051

  // Lambert W_prime 0.063577133469345105142021311010780887641928338458371618, double 0.063577133469345098
}; // BOOST_AUTO_TEST_CASE("Derivatives of lambert_w")


