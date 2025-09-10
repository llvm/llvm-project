// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
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
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, typename T>
void do_test_ellint_f(T& data, const char* type_name, const char* test)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(ELLINT_1_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

#ifdef ELLINT_1_FUNCTION_TO_TEST
   value_type(*fp2)(value_type, value_type) = ELLINT_1_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
    value_type (*fp2)(value_type, value_type) = boost::math::ellint_1<value_type, value_type>;
#else
    value_type (*fp2)(value_type, value_type) = boost::math::ellint_1;
#endif
    boost::math::tools::test_result<value_type> result;

    result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp2, 1, 0),
      extract_result<Real>(2));
    handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "ellint_1", test);

   std::cout << std::endl;
#endif
}

template <class Real, typename T>
void do_test_ellint_k(T& data, const char* type_name, const char* test)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(ELLINT_1C_FUNCTION_TO_TEST))
   typedef Real                   value_type;
    boost::math::tools::test_result<value_type> result;

   std::cout << "Testing: " << test << std::endl;

#ifdef ELLINT_1C_FUNCTION_TO_TEST
   value_type(*fp1)(value_type) = ELLINT_1C_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   value_type (*fp1)(value_type) = boost::math::ellint_1<value_type>;
#else
   value_type (*fp1)(value_type) = boost::math::ellint_1;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp1, 0),
      extract_result<Real>(1));
   handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "ellint_1 (complete)", test);

   std::cout << std::endl;
#endif
}

template <typename T>
void test_spots(T, const char* type_name)
{
    // Function values calculated on http://functions.wolfram.com/
    // Note that Mathematica's EllipticF accepts k^2 as the second parameter.
    static const std::array<std::array<typename table_type<T>::type, 3>, 22> data1 = {{
        {{ SC_(0.0), SC_(0.0), SC_(0.0) }},
        {{ SC_(-10.0), SC_(0.0), SC_(-10.0) }},
        {{ SC_(-1.0), SC_(-1.0), SC_(-1.2261911708835170708130609674719067527242483502207) }},
        {{ SC_(-4.0), SC_(0.875), SC_(-5.3190556182262405182189463092940736859067548232647) }},
        {{ SC_(8.0), SC_(-0.625), SC_(9.0419973860310100524448893214394562615252527557062) }},
        {{ SC_(1e-05), SC_(0.875), SC_(0.000010000000000127604166668510945638036143355898993088) }},
        {{ SC_(1e+05), SC_(0.009765625) /*T(10)/1024*/, SC_(100002.38431454899771096037307519328741455615271038) }},
        {{ SC_(1e-20), SC_(1.0), SC_(1.0000000000000000000000000000000000000000166666667e-20) }},
        {{ SC_(1e-20), SC_(1e-20), SC_(1.000000000000000e-20) }},
        {{ SC_(1e+20), SC_(0.390625) /*T(400)/1024*/, SC_(1.0418143796499216839719289963154558027005142709763e20) }},
        {{ SC_(1e+50), SC_(0.875), SC_(1.3913251718238765549409892714295358043696028445944e50) }},
        {{ SC_(2.0), SC_(0.5), SC_(2.1765877052210673672479877957388515321497888026770) }},
        {{ SC_(4.0), SC_(0.5), SC_(4.2543274975235836861894752787874633017836785640477) }},
        {{ SC_(6.0), SC_(0.5), SC_(6.4588766202317746302999080620490579800463614807916) }},
        {{ SC_(10.0), SC_(0.5), SC_(10.697409951222544858346795279378531495869386960090) }},
        {{ SC_(-2.0), SC_(0.5), SC_(-2.1765877052210673672479877957388515321497888026770) }},
        {{ SC_(-4.0), SC_(0.5), SC_(-4.2543274975235836861894752787874633017836785640477) }},
        {{ SC_(-6.0), SC_(0.5), SC_(-6.4588766202317746302999080620490579800463614807916) }},
        {{ SC_(-10.0), SC_(0.5), SC_(-10.697409951222544858346795279378531495869386960090) }},
        // Some values where k is > 1:
        {{ SC_(0.1538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538), SC_(1.1538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538), SC_(0.154661869446904722070471580919758948531148566762183486996920)}},
        {{ SC_(0.1538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538), SC_(1.461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461), SC_(0.155166467455029577314314021156113481657713115640002027219)}},
        {{ SC_(0.1538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538), SC_(2.461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461538461), SC_(0.15776272074094290829870142225970052217542486917945444918)}},
    }};

    do_test_ellint_f<T>(data1, type_name, "Elliptic Integral F: Mathworld Data");

#include "ellint_f_data.ipp"

    do_test_ellint_f<T>(ellint_f_data, type_name, "Elliptic Integral F: Random Data");

    // Function values calculated on http://functions.wolfram.com/
    // Note that Mathematica's EllipticK accepts k^2 as the second parameter.
    static const std::array<std::array<typename table_type<T>::type, 2>, 9> data2 = {{
        {{ SC_(0.0), SC_(1.5707963267948966192313216916397514420985846996876) }},
        {{ SC_(0.125), SC_(1.5769867712158131421244030532288080803822271060839) }},
        {{ SC_(0.25), SC_(1.5962422221317835101489690714979498795055744578951) }},
        {{ SC_(0.29296875) /*T(300)/1024*/, SC_(1.6062331054696636704261124078746600894998873503208) }},
        {{ SC_(0.390625) /*T(400)/1024*/, SC_(1.6364782007562008756208066125715722889067992997614) }},
        {{ SC_(-0.5), SC_(1.6857503548125960428712036577990769895008008941411) }},
        {{ SC_(-0.75), SC_(1.9109897807518291965531482187613425592531451316788) }},
        {{ SC_(0.875) /*1-T(1)/8*/, SC_(2.185488469278223686913080323730158689730428415766) }},
        {{ SC_(0.9990234375) /*1-T(1)/1024*/, SC_(4.5074135978990422666372495313621124487894807327687) }},
    }};

    do_test_ellint_k<T>(data2, type_name, "Elliptic Integral K: Mathworld Data");

#include "ellint_k_data.ipp"

    do_test_ellint_k<T>(ellint_k_data, type_name, "Elliptic Integral K: Random Data");

    //
    // Test error handling:
    //
    #ifndef BOOST_MATH_NO_EXCEPTIONS
    BOOST_CHECK_GE(boost::math::ellint_1(T(1)), boost::math::tools::max_value<T>());
    BOOST_CHECK_GE(boost::math::ellint_1(T(-1)), boost::math::tools::max_value<T>());
    BOOST_CHECK_THROW(boost::math::ellint_1(T(1.0001)), std::domain_error);
    BOOST_CHECK_THROW(boost::math::ellint_1(T(-1.0001)), std::domain_error);
    BOOST_CHECK_THROW(boost::math::ellint_1(T(2.2), T(0.5)), std::domain_error);
    BOOST_CHECK_THROW(boost::math::ellint_1(T(-2.2), T(0.5)), std::domain_error);
    #endif
}

