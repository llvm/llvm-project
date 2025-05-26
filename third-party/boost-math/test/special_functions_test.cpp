// test file for special functions.

//  (C) Copyright Hubert Holin 2003.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <iomanip>


#include <boost/mpl/list.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_log.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/math/tools/test.hpp>


template<typename T>
struct string_type_name;

#define DEFINE_TYPE_NAME(Type)              \
template<> struct string_type_name<Type>    \
{                                           \
    static char const * _()                 \
    {                                       \
        return #Type;                       \
    }                                       \
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
DEFINE_TYPE_NAME(float);
DEFINE_TYPE_NAME(double);
DEFINE_TYPE_NAME(long double);

typedef boost::mpl::list<float,double,long double>  test_types;
#else
DEFINE_TYPE_NAME(float);
DEFINE_TYPE_NAME(double);

typedef boost::mpl::list<float,double>  test_types;
#endif

// Apple GCC 4.0 uses the "double double" format for its long double,
// which means that epsilon is VERY small but useless for
// comparisons. So, don't do those comparisons.
#if (defined(__APPLE_CC__) && defined(__GNUC__) && __GNUC__ == 4) || defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
typedef boost::mpl::list<float,double>  near_eps_test_types;
#else
typedef boost::mpl::list<float,double,long double>  near_eps_test_types;
#endif

#include "sinc_test.hpp"
#include "sinhc_test.hpp"
#include "atanh_test.hpp"
#include "asinh_test.hpp"
#include "acosh_test.hpp"



boost::unit_test::test_suite *    init_unit_test_suite(int, char *[])
{
    ::boost::unit_test::unit_test_log.
        set_threshold_level(::boost::unit_test::log_messages);
    
    boost::unit_test::test_suite *    test =
        BOOST_TEST_SUITE("special_functions_test");
    
    BOOST_TEST_MESSAGE("Results of special functions test.");
    BOOST_TEST_MESSAGE(" ");
    BOOST_TEST_MESSAGE("(C) Copyright Hubert Holin 2003-2005.");
    BOOST_TEST_MESSAGE("Distributed under the Boost Software License, Version 1.0.");
    BOOST_TEST_MESSAGE("(See accompanying file LICENSE_1_0.txt or copy at");
    BOOST_TEST_MESSAGE("http://www.boost.org/LICENSE_1_0.txt)");
    BOOST_TEST_MESSAGE(" ");
    
#define BOOST_SPECIAL_FUNCTIONS_COMMON_GENERATOR(fct)   \
    test->add(BOOST_TEST_CASE_TEMPLATE(fct##_test, test_types));

#define BOOST_SPECIAL_FUNCTIONS_COMMON_GENERATOR_NEAR_EPS(fct)   \
    test->add(BOOST_TEST_CASE_TEMPLATE(fct##_test, near_eps_test_types));
    
    
#define BOOST_SPECIAL_FUNCTIONS_COMMON_TEST             \
    BOOST_SPECIAL_FUNCTIONS_COMMON_GENERATOR(atanh)     \
    BOOST_SPECIAL_FUNCTIONS_COMMON_GENERATOR(asinh)     \
    BOOST_SPECIAL_FUNCTIONS_COMMON_GENERATOR(acosh)     \
    BOOST_SPECIAL_FUNCTIONS_COMMON_GENERATOR(sinc_pi)   \
    BOOST_SPECIAL_FUNCTIONS_COMMON_GENERATOR(sinhc_pi)
    
#define BOOST_SPECIAL_FUNCTIONS_TEMPLATE_TEMPLATE_TEST          \
    BOOST_SPECIAL_FUNCTIONS_COMMON_GENERATOR_NEAR_EPS(sinc_pi_complex)   \
    BOOST_SPECIAL_FUNCTIONS_COMMON_GENERATOR_NEAR_EPS(sinhc_pi_complex)
    
    
#ifdef  BOOST_NO_TEMPLATE_TEMPLATES

#define BOOST_SPECIAL_FUNCTIONS_TEST    \
    BOOST_SPECIAL_FUNCTIONS_COMMON_TEST \
    BOOST_TEST_MESSAGE("Warning: no template templates; curtailed functionality.");
    
#else   /* BOOST_NO_TEMPLATE_TEMPLATES */

#define BOOST_SPECIAL_FUNCTIONS_TEST                \
    BOOST_SPECIAL_FUNCTIONS_COMMON_TEST             \
    BOOST_SPECIAL_FUNCTIONS_TEMPLATE_TEMPLATE_TEST
    
#endif  /* BOOST_NO_TEMPLATE_TEMPLATES */
    
    
    BOOST_SPECIAL_FUNCTIONS_TEST
    
    
#undef  BOOST_SPECIAL_FUNCTIONS_TEST
    
#undef  BOOST_SPECIAL_FUNCTIONS_TEMPLATE_TEMPLATE_TEST

#undef  BOOST_SPECIAL_FUNCTIONS_COMMON_TEST
    
#undef  BOOST_SPECIAL_FUNCTIONS_COMMON_GENERATOR

#undef  BOOST_SPECIAL_FUNCTIONS_COMMON_GENERATOR_NEAR_EPS
    
#ifdef    BOOST_SPECIAL_FUNCTIONS_TEST_VERBOSE
        
    using    ::std::numeric_limits;
    
    BOOST_TEST_MESSAGE("epsilon");
    
    BOOST_TEST_MESSAGE( ::std::setw(15) << numeric_limits<float>::epsilon()
                << ::std::setw(15) << numeric_limits<double>::epsilon()
                << ::std::setw(15) << numeric_limits<long double>::epsilon());
    
    BOOST_TEST_MESSAGE(" ");
    
    test->add(BOOST_TEST_CASE(atanh_manual_check));
    test->add(BOOST_TEST_CASE(asinh_manual_check));
    test->add(BOOST_TEST_CASE(acosh_manual_check));
    test->add(BOOST_TEST_CASE(sinc_pi_manual_check));
    test->add(BOOST_TEST_CASE(sinhc_pi_manual_check));
    
#endif    /* BOOST_SPECIAL_FUNCTIONS_TEST_VERBOSE */
    
    return test;
}

#undef DEFINE_TYPE_NAME
