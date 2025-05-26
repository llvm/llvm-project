// Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning(disable : 4756) // overflow in constant arithmetic
// Constants are too big for float case, but this doesn't matter for test.
#endif

#include <boost/math/tools/config.hpp>

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp>
#endif

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/heuman_lambda.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, typename T>
void do_test_heuman_lambda(const T& data, const char* type_name, const char* test)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(HEUMAN_LAMBDA_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

#ifdef HEUMAN_LAMBDA_FUNCTION_TO_TEST
   value_type(*fp2)(value_type, value_type) = HEUMAN_LAMBDA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
    value_type (*fp2)(value_type, value_type) = boost::math::ellint_d<value_type, value_type>;
#else
   value_type(*fp2)(value_type, value_type) = boost::math::heuman_lambda;
#endif
    boost::math::tools::test_result<value_type> result;

    result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp2, 1, 0),
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "heuman_lambda", test);

   std::cout << std::endl;
#endif
}

template <typename T>
void test_spots(T, const char* type_name)
{
    BOOST_MATH_STD_USING
    // Function values calculated on http://functions.wolfram.com/
    // Note that Mathematica's EllipticE accepts k^2 as the second parameter.
    static const std::array<std::array<T, 3>, 5> data1 = {{
       { { SC_(0.25), SC_(0.5), SC_(0.231195544262270355901990821099667428154924832224446817213200) } },
       { { SC_(-0.25), SC_(0.5), SC_(-0.231195544262270355901990821099667428154924832224446817213200) } },
        { { SC_(0), SC_(0.5), SC_(0) } },
        { { SC_(1), T(0.5), SC_(0.792745183008071035953588061452801838417979005666066982987549) } },
        { { SC_(1), T(0), SC_(0.841470984807896506652502321630298999622563060798371065672751) } },
    }};

    do_test_heuman_lambda<T>(data1, type_name, "Elliptic Integral Jacobi Zeta: Mathworld Data");

#include "heuman_lambda_data.ipp"

    do_test_heuman_lambda<T>(heuman_lambda_data, type_name, "Elliptic Integral Heuman Lambda: Random Data");
}

