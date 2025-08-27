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
#include <boost/math/special_functions/ellint_d.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, typename T>
void do_test_ellint_d2(const T& data, const char* type_name, const char* test)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(ELLINT_D2_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

#ifdef ELLINT_D2_FUNCTION_TO_TEST
   value_type(*fp2)(value_type, value_type) = ELLINT_D2_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
    value_type (*fp2)(value_type, value_type) = boost::math::ellint_d<value_type, value_type>;
#else
    value_type (*fp2)(value_type, value_type) = boost::math::ellint_d;
#endif
    boost::math::tools::test_result<value_type> result;

    result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp2, 1, 0),
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "ellint_d", test);

   std::cout << std::endl;
#endif
}

template <class Real, typename T>
void do_test_ellint_d1(T& data, const char* type_name, const char* test)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(ELLINT_D1_FUNCTION_TO_TEST))
   typedef Real                   value_type;
    boost::math::tools::test_result<value_type> result;

   std::cout << "Testing: " << test << std::endl;

#ifdef ELLINT_D1_FUNCTION_TO_TEST
   value_type(*fp1)(value_type) = ELLINT_D1_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   value_type (*fp1)(value_type) = boost::math::ellint_d<value_type>;
#else
   value_type (*fp1)(value_type) = boost::math::ellint_d;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp1, 0),
      extract_result<Real>(1));
   handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "ellint_d (complete)", test);

   std::cout << std::endl;
#endif
}

template <typename T>
void test_spots(T, const char* type_name)
{
    BOOST_MATH_STD_USING
    // Function values calculated on http://functions.wolfram.com/
    // Note that Mathematica's EllipticE accepts k^2 as the second parameter.
    static const std::array<std::array<T, 3>, 12> data1 = {{
       { { SC_(0.5), SC_(0.5), SC_(0.040348098248931543984282958654503585) } },
        {{ SC_(0), SC_(0.5), SC_(0) }},
        { { SC_(1), SC_(0.5), SC_(0.28991866293419922467977188008516755) } },
        { { SC_(1), T(1), SC_(0.38472018607562056416055864584160775) } },
        { { SC_(-1), T(1), SC_(-0.38472018607562056416055864584160775) } },
        { { SC_(-1), T(0.5), SC_(-0.28991866293419922467977188008516755) } },
        { { SC_(-10), T(0.5), SC_(-5.2996914501577855803123384771117708) } },
        { { SC_(10), SC_(-0.5), SC_(5.2996914501577855803123384771117708) } },
        { { SC_(0.125), SC_(1.5), SC_(0.000655956467603362564458676111698495009248974444516843) } },
        { { SC_(1.208925819614629174706176e24) /* 2^80 */, SC_(0.5), SC_(672000998924580555450487.42418840712)}},
    }};

    do_test_ellint_d2<T>(data1, type_name, "Elliptic Integral E: Mathworld Data");

#include "ellint_d2_data.ipp"

    do_test_ellint_d2<T>(ellint_d2_data, type_name, "Elliptic Integral D: Random Data");

    // Function values calculated on http://functions.wolfram.com/
    // Note that Mathematica's EllipticE accepts k^2 as the second parameter.
    static const std::array<std::array<T, 2>, 3> data2 = {{
       { { SC_(0.5), SC_(0.87315258189267554964563356323264341) } },
       { { SC_(1.0) / 1024, SC_(0.78539844427788694671464428063604776) } },
       { { boost::math::tools::root_epsilon<T>(), SC_(0.78539816339744830961566084581987572) } }
    }};

    do_test_ellint_d1<T>(data2, type_name, "Elliptic Integral E: Mathworld Data");

#include "ellint_d_data.ipp"

    do_test_ellint_d1<T>(ellint_d_data, type_name, "Elliptic Integral D: Random Data");

    #ifdef BOOST_MATH_NO_EXCEPTIONS
    BOOST_MATH_CHECK_THROW(boost::math::ellint_d(T(1)), std::domain_error);
    BOOST_MATH_CHECK_THROW(boost::math::ellint_d(T(-1)), std::domain_error);
    BOOST_MATH_CHECK_THROW(boost::math::ellint_d(T(1.5)), std::domain_error);
    BOOST_MATH_CHECK_THROW(boost::math::ellint_d(T(-1.5)), std::domain_error);
    BOOST_IF_CONSTEXPR(std::numeric_limits<T>::has_infinity)
    {
       BOOST_CHECK_EQUAL(boost::math::ellint_d(T(0.5), std::numeric_limits<T>::infinity()), std::numeric_limits<T>::infinity());
    }
    BOOST_MATH_CHECK_THROW(boost::math::ellint_d(T(1.5), T(1.0)), std::domain_error);
    #endif
}

