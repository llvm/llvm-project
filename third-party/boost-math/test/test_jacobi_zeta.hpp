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
#include <boost/math/special_functions/jacobi_zeta.hpp>
#include <boost/math/constants/constants.hpp>
//#include <boost/math/special_functions/next.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, typename T>
void do_test_jacobi_zeta(const T& data, const char* type_name, const char* test)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(JACOBI_ZETA_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

#ifdef JACOBI_ZETA_FUNCTION_TO_TEST
   value_type(*fp2)(value_type, value_type) = JACOBI_ZETA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
    value_type (*fp2)(value_type, value_type) = boost::math::ellint_d<value_type, value_type>;
#else
   value_type(*fp2)(value_type, value_type) = boost::math::jacobi_zeta;
#endif
    boost::math::tools::test_result<value_type> result;

    result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp2, 1, 0),
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "jacobi_zeta", test);

   std::cout << std::endl;
#endif
}

template <typename T>
void test_spots(T, const char* type_name)
{
    BOOST_MATH_STD_USING
    // Function values calculated on http://functions.wolfram.com/
    // Note that Mathematica's EllipticE accepts k^2 as the second parameter.
    static const std::array<std::array<T, 3>, 18> data1 = {{
       { { SC_(0.5), SC_(0.5), SC_(0.055317014255129651475392155709691519) } },
       { { SC_(-0.5), SC_(0.5), SC_(-0.055317014255129651475392155709691519) } },
        { { SC_(0), SC_(0.5), SC_(0) } },
        { { SC_(1), T(0.5), SC_(0.061847782565098669252626761181452815) } },
//        { { boost::math::float_prior(boost::math::constants::half_pi<T>()), T(0.5), SC_(0) } },
        { { SC_(1), T(0), SC_(0) } },
        { { SC_(1), T(1), SC_(0.84147098480789650665250232163029900) } },
        { { SC_(2), T(0.5), SC_(-0.051942537457672732722176231281435254) } },
        { { SC_(5), T(0.5), SC_(-0.037609329968145259476447488930872898) } },
        { { SC_(0.5), SC_(1), SC_(0.479425538604203000273287935215571388081803367940600675188616) } },
       { { boost::math::constants::half_pi<T>() - static_cast<T>(1) / 1024, SC_(1), SC_(0.999999523162879692486369202949889069215510235208243466564977) } },
       { { boost::math::constants::half_pi<T>() + static_cast<T>(1) / 1024, SC_(1), SC_(-0.999999523162879692486369202949889069215510235208243466564977) } },
       { { SC_(2), SC_(1), SC_(-0.90929742682568169539601986591174484270225497144789026837897) } },
       { { SC_(3), SC_(1), SC_(-0.14112000805986722210074480280811027984693326425226558415188) } },
       { { SC_(4), SC_(1), SC_(0.756802495307928251372639094511829094135912887336472571485416) } },
        { { SC_(-0.5), SC_(1), SC_(-0.479425538604203000273287935215571388081803367940600675188616) } },
       { { SC_(-2), SC_(1), SC_(0.90929742682568169539601986591174484270225497144789026837897) } },
       { { SC_(-3), SC_(1), SC_(0.14112000805986722210074480280811027984693326425226558415188) } },
       { { SC_(-4), SC_(1), SC_(-0.756802495307928251372639094511829094135912887336472571485416) } },
    }};

    do_test_jacobi_zeta<T>(data1, type_name, "Elliptic Integral Jacobi Zeta: Mathworld Data");

#include "jacobi_zeta_data.ipp"

    do_test_jacobi_zeta<T>(jacobi_zeta_data, type_name, "Elliptic Integral Jacobi Zeta: Random Data");

#include "jacobi_zeta_big_phi.ipp"

    do_test_jacobi_zeta<T>(jacobi_zeta_big_phi, type_name, "Elliptic Integral Jacobi Zeta: Large Phi Values");
}

