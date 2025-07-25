// Copyright Nick Thompson, 2017
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE adaptive_gauss_kronrod_quadrature_test

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#if __has_include(<stdfloat>)
# include <stdfloat>
#endif

#if !defined(BOOST_NO_CXX11_DECLTYPE) && !defined(BOOST_NO_CXX11_TRAILING_RESULT_TYPES) && !defined(BOOST_NO_SFINAE_EXPR)

#include <boost/math/concepts/real_concept.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/type_index.hpp>

#if !defined(TEST1) && !defined(TEST1A) && !defined(TEST2) && !defined(TEST3)
#  define TEST1
#  define TEST1A
#  define TEST2
#  define TEST3
#endif

#ifdef _MSC_VER
#pragma warning(disable:4127)  // Conditional expression is constant
#endif

using std::expm1;
using std::atan;
using std::tan;
using std::log;
using std::log1p;
using std::asinh;
using std::atanh;
using std::sqrt;
using std::isnormal;
using std::abs;
using std::sinh;
using std::tanh;
using std::cosh;
using std::pow;
using std::exp;
using std::sin;
using std::cos;
using std::string;
using boost::math::quadrature::gauss_kronrod;
using boost::math::constants::pi;
using boost::math::constants::half_pi;
using boost::math::constants::two_div_pi;
using boost::math::constants::two_pi;
using boost::math::constants::half;
using boost::math::constants::third;
using boost::math::constants::half;
using boost::math::constants::third;
using boost::math::constants::catalan;
using boost::math::constants::ln_two;
using boost::math::constants::root_two;
using boost::math::constants::root_two_pi;
using boost::math::constants::root_pi;
using boost::multiprecision::cpp_bin_float_quad;

template <class Real>
Real get_termination_condition()
{
   return boost::math::tools::epsilon<Real>() * 1000;
}


template<class Real, unsigned Points>
void test_linear()
{
    std::cout << "Testing linear functions are integrated properly by gauss_kronrod on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = boost::math::tools::epsilon<Real>() * 10;
    Real error;
    auto f = [](const Real& x)
    {
       return 5*x + 7;
    };
    Real L1;
    Real Q = gauss_kronrod<Real, Points>::integrate(f, (Real) 0, (Real) 1, 15, get_termination_condition<Real>(), &error, &L1);
    BOOST_CHECK_CLOSE_FRACTION(Q, 9.5, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, 9.5, tol);
    BOOST_CHECK_LE(fabs(error / Q), get_termination_condition<Real>());
    BOOST_CHECK_GE(fabs(error), fabs(Q - 9.5));
}

template<class Real, unsigned Points>
void test_quadratic()
{
    std::cout << "Testing quadratic functions are integrated properly by gauss-kronrod on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = boost::math::tools::epsilon<Real>() * 10;
    Real error;

    auto f = [](const Real& x) { return 5*x*x + 7*x + 12; };
    Real L1;
    Real Q = gauss_kronrod<Real, Points>::integrate(f, 0, 1, 15, get_termination_condition<Real>(), &error, &L1);
    BOOST_CHECK_CLOSE_FRACTION(Q, (Real) 17 + half<Real>()*third<Real>(), tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, (Real) 17 + half<Real>()*third<Real>(), tol);
    BOOST_CHECK_LE(fabs(error / Q), get_termination_condition<Real>());
    BOOST_CHECK_GE(fabs(error), fabs(Q - ((Real)17 + half<Real>()*third<Real>())));
}

// Examples taken from
//http://crd-legacy.lbl.gov/~dhbailey/dhbpapers/quadrature.pdf
template<class Real, unsigned Points>
void test_ca()
{
    std::cout << "Testing integration of C(a) on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = boost::math::tools::epsilon<Real>() * 10;
    Real L1;
    Real error;

    auto f1 = [](const Real& x) { return atan(x)/(x*(x*x + 1)) ; };
    Real Q = gauss_kronrod<Real, Points>::integrate(f1, 0, 1, 15, get_termination_condition<Real>(), &error, &L1);
    Real Q_expected = pi<Real>()*ln_two<Real>()/8 + catalan<Real>()*half<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
    BOOST_CHECK_LE(fabs(error / Q), get_termination_condition<Real>());
    BOOST_CHECK_GE(fabs(error), fabs(Q - Q_expected));

    auto f2 = [](Real x)->Real { Real t0 = x*x + 1; Real t1 = sqrt(t0); return atan(t1)/(t0*t1); };
    Q = gauss_kronrod<Real, Points>::integrate(f2, 0 , 1, 15, get_termination_condition<Real>(), &error, &L1);
    Q_expected = pi<Real>()/4 - pi<Real>()/root_two<Real>() + 3*atan(root_two<Real>())/root_two<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
    BOOST_CHECK_LE(fabs(error / Q), get_termination_condition<Real>());
    BOOST_CHECK_GE(fabs(error), fabs(Q - Q_expected));

    auto f5 = [](Real t)->Real { return t*t*log(t)/((t*t - 1)*(t*t*t*t + 1)); };
    Q = gauss_kronrod<Real, Points>::integrate(f5, 0, 1, 25);
    Q_expected = pi<Real>()*pi<Real>()*(2 - root_two<Real>())/32;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 100 * tol);
}

template<class Real, unsigned Points>
void test_three_quadrature_schemes_examples()
{
    std::cout << "Testing integral in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = boost::math::tools::epsilon<Real>() * 10;
    Real Q;
    Real Q_expected;

    // Example 1:
    auto f1 = [](const Real& t) { return t*boost::math::log1p(t); };
    Q = gauss_kronrod<Real, Points>::integrate(f1, 0 , 1);
    Q_expected = half<Real>()*half<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);


    // Example 2:
    auto f2 = [](const Real& t) { return t*t*atan(t); };
    Q = gauss_kronrod<Real, Points>::integrate(f2, 0, 1);
    Q_expected = (pi<Real>() -2 + 2*ln_two<Real>())/12;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 2 * tol);

    // Example 3:
    auto f3 = [](const Real& t) { return exp(t)*cos(t); };
    Q = gauss_kronrod<Real, Points>::integrate(f3, 0, half_pi<Real>());
    Q_expected = boost::math::expm1(half_pi<Real>())*half<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

    // Example 4:
    auto f4 = [](Real x)->Real { Real t0 = sqrt(x*x + 2); return atan(t0)/(t0*(x*x+1)); };
    Q = gauss_kronrod<Real, Points>::integrate(f4, 0, 1);
    Q_expected = 5*pi<Real>()*pi<Real>()/96;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
}


template<class Real, unsigned Points>
void test_integration_over_real_line()
{
    std::cout << "Testing integrals over entire real line in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = boost::math::tools::epsilon<Real>() * 10;
    Real Q;
    Real Q_expected;
    Real L1;
    Real error;

    auto f1 = [](const Real& t) { return 1/(1+t*t);};
    Q = gauss_kronrod<Real, Points>::integrate(f1, -boost::math::tools::max_value<Real>(), boost::math::tools::max_value<Real>(), 15, get_termination_condition<Real>(), &error, &L1);
    Q_expected = pi<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
    BOOST_CHECK_LE(fabs(error / Q), get_termination_condition<Real>());

    auto f4 = [](const Real& t) { return 1/cosh(t);};
    Q = gauss_kronrod<Real, Points>::integrate(f4, -boost::math::tools::max_value<Real>(), boost::math::tools::max_value<Real>(), 15, get_termination_condition<Real>(), &error, &L1);
    Q_expected = pi<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
    BOOST_CHECK_LE(fabs(error / Q), get_termination_condition<Real>());

}

template<class Real, unsigned Points>
void test_right_limit_infinite()
{
    std::cout << "Testing right limit infinite for Gauss-Kronrod in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = boost::math::tools::epsilon<Real>() * 10;
    Real Q;
    Real Q_expected;
    Real L1;
    Real error;

    // Example 11:
    auto f1 = [](const Real& t) { return 1/(1+t*t);};
    Q = gauss_kronrod<Real, Points>::integrate(f1, 0, boost::math::tools::max_value<Real>(), 15, get_termination_condition<Real>(), &error, &L1);
    Q_expected = half_pi<Real>();
    BOOST_CHECK_CLOSE(Q, Q_expected, 100*tol);
    BOOST_CHECK_LE(fabs(error / Q), get_termination_condition<Real>());

    auto f4 = [](const Real& t) { return 1/(1+t*t); };
    Q = gauss_kronrod<Real, Points>::integrate(f4, 1, boost::math::tools::max_value<Real>(), 15, get_termination_condition<Real>(), &error, &L1);
    Q_expected = pi<Real>()/4;
    BOOST_CHECK_CLOSE(Q, Q_expected, 100*tol);
    BOOST_CHECK_LE(fabs(error / Q), get_termination_condition<Real>());
}

template<class Real, unsigned Points>
void test_left_limit_infinite()
{
    std::cout << "Testing left limit infinite for Gauss-Kronrod in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = boost::math::tools::epsilon<Real>() * 10;
    Real Q;
    Real Q_expected;

    // Example 11:
    auto f1 = [](const Real& t) { return 1/(1+t*t);};
    Q = gauss_kronrod<Real, Points>::integrate(f1, -boost::math::tools::max_value<Real>(), 0);
    Q_expected = half_pi<Real>();
    BOOST_CHECK_CLOSE(Q, Q_expected, 300*tol);
}

BOOST_AUTO_TEST_CASE(gauss_quadrature_test)
{
#ifdef TEST1
    std::cout << "Testing with 15 point Gauss-Kronrod rule:\n";
    test_linear<double, 15>();
    test_quadratic<double, 15>();
    test_ca<double, 15>();
    test_three_quadrature_schemes_examples<double, 15>();
    test_integration_over_real_line<double, 15>();
    test_right_limit_infinite<double, 15>();
    test_left_limit_infinite<double, 15>();

    //  test one case where we do not have pre-computed constants:
    std::cout << "Testing with 17 point Gauss-Kronrod rule:\n";
    test_linear<double, 17>();
    test_quadratic<double, 17>();
    test_ca<double, 17>();
    test_three_quadrature_schemes_examples<double, 17>();
    test_integration_over_real_line<double, 17>();
    test_right_limit_infinite<double, 17>();
    test_left_limit_infinite<double, 17>();

    #ifdef __STDCPP_FLOAT64_T__
    test_linear<std::float64_t, 15>();
    test_quadratic<std::float64_t, 15>();
    test_ca<std::float64_t, 15>();
    test_three_quadrature_schemes_examples<std::float64_t, 15>();
    test_integration_over_real_line<std::float64_t, 15>();
    test_right_limit_infinite<std::float64_t, 15>();
    test_left_limit_infinite<std::float64_t, 15>();
    #endif

#endif
#ifdef TEST1A
#if LDBL_MANT_DIG < 100 // If we have too many digits in a long double, we get build errors due to a constexpr issue.
    std::cout << "Testing with 21 point Gauss-Kronrod rule:\n";
    test_linear<cpp_bin_float_quad, 21>();
    test_quadratic<cpp_bin_float_quad, 21>();
    test_ca<cpp_bin_float_quad, 21>();
    test_three_quadrature_schemes_examples<cpp_bin_float_quad, 21>();
    test_integration_over_real_line<cpp_bin_float_quad, 21>();
    test_right_limit_infinite<cpp_bin_float_quad, 21>();
    test_left_limit_infinite<cpp_bin_float_quad, 21>();

    std::cout << "Testing with 31 point Gauss-Kronrod rule:\n";
    test_linear<cpp_bin_float_quad, 31>();
    test_quadratic<cpp_bin_float_quad, 31>();
    test_ca<cpp_bin_float_quad, 31>();
    test_three_quadrature_schemes_examples<cpp_bin_float_quad, 31>();
    test_integration_over_real_line<cpp_bin_float_quad, 31>();
    test_right_limit_infinite<cpp_bin_float_quad, 31>();
    test_left_limit_infinite<cpp_bin_float_quad, 31>();
#endif
#endif
#ifdef TEST2
#if LDBL_MANT_DIG < 100 // If we have too many digits in a long double, we get build errors due to a constexpr issue.
    std::cout << "Testing with 41 point Gauss-Kronrod rule:\n";
    test_linear<cpp_bin_float_quad, 41>();
    test_quadratic<cpp_bin_float_quad, 41>();
    test_ca<cpp_bin_float_quad, 41>();
    test_three_quadrature_schemes_examples<cpp_bin_float_quad, 41>();
    test_integration_over_real_line<cpp_bin_float_quad, 41>();
    test_right_limit_infinite<cpp_bin_float_quad, 41>();
    test_left_limit_infinite<cpp_bin_float_quad, 41>();

    std::cout << "Testing with 51 point Gauss-Kronrod rule:\n";
    test_linear<cpp_bin_float_quad, 51>();
    test_quadratic<cpp_bin_float_quad, 51>();
    test_ca<cpp_bin_float_quad, 51>();
    test_three_quadrature_schemes_examples<cpp_bin_float_quad, 51>();
    test_integration_over_real_line<cpp_bin_float_quad, 51>();
    test_right_limit_infinite<cpp_bin_float_quad, 51>();
    test_left_limit_infinite<cpp_bin_float_quad, 51>();
#endif
#endif
#ifdef TEST3
#if LDBL_MANT_DIG < 100 // If we have too many digits in a long double, we get build errors due to a constexpr issue.
    std::cout << "Testing with 61 point Gauss-Kronrod rule:\n";
    test_linear<cpp_bin_float_quad, 61>();
    test_quadratic<cpp_bin_float_quad, 61>();
    test_ca<cpp_bin_float_quad, 61>();
    test_three_quadrature_schemes_examples<cpp_bin_float_quad, 61>();
    test_integration_over_real_line<cpp_bin_float_quad, 61>();
    test_right_limit_infinite<cpp_bin_float_quad, 61>();
    test_left_limit_infinite<cpp_bin_float_quad, 61>();
#endif
#endif
}

#else

int main() { return 0; }

#endif
