// Copyright Nick Thompson, 2017
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE tanh_sinh_quadrature_test

#include <complex>
//#include <boost/multiprecision/mpc.hpp>
#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#if !defined(BOOST_NO_CXX11_DECLTYPE) && !defined(BOOST_NO_CXX11_TRAILING_RESULT_TYPES) && !defined(BOOST_NO_SFINAE_EXPR)

#include <boost/math/concepts/real_concept.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <boost/math/tools/test_value.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_complex.hpp>
#include <boost/type_index.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/complex128.hpp>
#endif

#ifdef _MSC_VER
#pragma warning(disable:4127)  // Conditional expression is constant
#endif

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

#if !defined(TEST1) && !defined(TEST2) && !defined(TEST3)
#  define TEST1
#  define TEST2
#  define TEST3
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
using boost::math::quadrature::gauss;
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

//
// Error rates depend only on the number of points in the approximation, not the type being tested,
// define all our expected errors here:
//

enum
{
   test_ca_error_id,
   test_ca_error_id_2,
   test_three_quad_error_id,
   test_three_quad_error_id_2,
   test_integration_over_real_line_error_id,
   test_right_limit_infinite_error_id,
   test_left_limit_infinite_error_id
};

template <unsigned Points>
double expected_error(unsigned)
{
   return 0; // placeholder, all tests will fail
}

template <>
double expected_error<7>(unsigned id)
{
   switch (id)
   {
   case test_ca_error_id:
      return 1e-7;
   case test_ca_error_id_2:
      return 2e-5;
   case test_three_quad_error_id:
      return 1e-8;
   case test_three_quad_error_id_2:
      return 3.5e-3;
   case test_integration_over_real_line_error_id:
      return 6e-3;
   case test_right_limit_infinite_error_id:
   case test_left_limit_infinite_error_id:
      return 1e-5;
   }
   return 0;  // placeholder, all tests will fail
}

template <>
double expected_error<9>(unsigned id)
{
   switch (id)
   {
   case test_ca_error_id:
      return 1e-7;
   case test_ca_error_id_2:
      return 2e-5;
   case test_three_quad_error_id:
      return 1e-8;
   case test_three_quad_error_id_2:
      return 3.5e-3;
   case test_integration_over_real_line_error_id:
      return 6e-3;
   case test_right_limit_infinite_error_id:
   case test_left_limit_infinite_error_id:
      return 1e-5;
   }
   return 0;  // placeholder, all tests will fail
}

template <>
double expected_error<10>(unsigned id)
{
   switch (id)
   {
   case test_ca_error_id:
      return 1e-12;
   case test_ca_error_id_2:
      return 3e-6;
   case test_three_quad_error_id:
      return 2e-13;
   case test_three_quad_error_id_2:
      return 2e-3;
   case test_integration_over_real_line_error_id:
      return 6e-3;  // doesn't get any better with more points!
   case test_right_limit_infinite_error_id:
   case test_left_limit_infinite_error_id:
      return 5e-8;
   }
   return 0;  // placeholder, all tests will fail
}

template <>
double expected_error<15>(unsigned id)
{
   switch (id)
   {
   case test_ca_error_id:
      return 6e-20;
   case test_ca_error_id_2:
      return 3e-7;
   case test_three_quad_error_id:
      return 1e-19;
   case test_three_quad_error_id_2:
      return 6e-4;
   case test_integration_over_real_line_error_id:
      return 6e-3;  // doesn't get any better with more points!
   case test_right_limit_infinite_error_id:
   case test_left_limit_infinite_error_id:
      return 5e-11;
   }
   return 0;  // placeholder, all tests will fail
}

template <>
double expected_error<20>(unsigned id)
{
   switch (id)
   {
   case test_ca_error_id:
      return 1e-26;
   case test_ca_error_id_2:
      return 1e-7;
   case test_three_quad_error_id:
      return 3e-27;
   case test_three_quad_error_id_2:
      return 3e-4;
   case test_integration_over_real_line_error_id:
      return 5e-5;  // doesn't get any better with more points!
   case test_right_limit_infinite_error_id:
   case test_left_limit_infinite_error_id:
      return 1e-15;
   }
   return 0;  // placeholder, all tests will fail
}

template <>
double expected_error<25>(unsigned id)
{
   switch (id)
   {
   case test_ca_error_id:
      return 5e-33;
   case test_ca_error_id_2:
      return 1e-8;
   case test_three_quad_error_id:
      return 1e-32;
   case test_three_quad_error_id_2:
      return 3e-4;
   case test_integration_over_real_line_error_id:
      return 1e-14;
   case test_right_limit_infinite_error_id:
   case test_left_limit_infinite_error_id:
      return 3e-19;
   }
   return 0;  // placeholder, all tests will fail
}

template <>
double expected_error<30>(unsigned id)
{
   switch (id)
   {
   case test_ca_error_id:
      return 2e-34;
   case test_ca_error_id_2:
      return 5e-9;
   case test_three_quad_error_id:
      return 4e-34;
   case test_three_quad_error_id_2:
      return 1e-4;
   case test_integration_over_real_line_error_id:
      return 1e-16;
   case test_right_limit_infinite_error_id:
   case test_left_limit_infinite_error_id:
      return 3e-23;
   }
   return 0;  // placeholder, all tests will fail
}


template<class Real, unsigned Points>
void test_linear()
{
    std::cout << "Testing linear functions are integrated properly by gauss on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = boost::math::tools::epsilon<Real>() * 10;
    auto f = [](const Real& x)
    {
       return 5*x + 7;
    };
    Real L1;
    Real Q = gauss<Real, Points>::integrate(f, (Real) 0, (Real) 1, &L1);
    BOOST_CHECK_CLOSE_FRACTION(Q, 9.5, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, 9.5, tol);
    Q = gauss<Real, Points>::integrate(f, (Real) 0, (Real) 0, &L1);
    BOOST_CHECK_CLOSE(Q, 0, tol);
    Q = gauss<Real, Points>::integrate(f, (Real) 1, (Real) 0, &L1);
    BOOST_CHECK_CLOSE_FRACTION(Q, -9.5, tol);
}

template<class Real, unsigned Points>
void test_quadratic()
{
    std::cout << "Testing quadratic functions are integrated properly by Gaussian quadrature on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = boost::math::tools::epsilon<Real>() * 10;

    auto f = [](const Real& x) { return 5*x*x + 7*x + 12; };
    Real L1;
    Real Q = gauss<Real, Points>::integrate(f, 0, 1, &L1);
    BOOST_CHECK_CLOSE_FRACTION(Q, (Real) 17 + half<Real>()*third<Real>(), tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, (Real) 17 + half<Real>()*third<Real>(), tol);
}

// Examples taken from
//http://crd-legacy.lbl.gov/~dhbailey/dhbpapers/quadrature.pdf
template<class Real, unsigned Points>
void test_ca()
{
    std::cout << "Testing integration of C(a) on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = expected_error<Points>(test_ca_error_id);
    Real L1;

    auto f1 = [](const Real& x) { 
      if (x == 0) {
         return static_cast<Real>(1);
      }
      return atan(x)/(x*(x*x + 1)) ;
    };
    Real Q = gauss<Real, Points>::integrate(f1, 0, 1, &L1);
    Real Q_expected = pi<Real>()*ln_two<Real>()/8 + catalan<Real>()*half<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);

    auto f2 = [](Real x)->Real { Real t0 = x*x + 1; Real t1 = sqrt(t0); return atan(t1)/(t0*t1); };
    Q = gauss<Real, Points>::integrate(f2, 0 , 1, &L1);
    Q_expected = pi<Real>()/4 - pi<Real>()/root_two<Real>() + 3*atan(root_two<Real>())/root_two<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);

    tol = expected_error<Points>(test_ca_error_id_2);
    auto f5 = [](Real t)->Real { return t*t*log(t)/((t*t - 1)*(t*t*t*t + 1)); };
    Q = gauss<Real, Points>::integrate(f5, 0 , 1);
    Q_expected = pi<Real>()*pi<Real>()*(2 - root_two<Real>())/32;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
}

template<class Real, unsigned Points>
void test_three_quadrature_schemes_examples()
{
    std::cout << "Testing integral in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = expected_error<Points>(test_three_quad_error_id);
    Real Q;
    Real Q_expected;

    // Example 1:
    auto f1 = [](const Real& t) { return t*boost::math::log1p(t); };
    Q = gauss<Real, Points>::integrate(f1, 0 , 1);
    Q_expected = half<Real>()*half<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);


    // Example 2:
    auto f2 = [](const Real& t) { return t*t*atan(t); };
    Q = gauss<Real, Points>::integrate(f2, 0 , 1);
    Q_expected = (pi<Real>() -2 + 2*ln_two<Real>())/12;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 2 * tol);

    // Example 3:
    auto f3 = [](const Real& t) { return exp(t)*cos(t); };
    Q = gauss<Real, Points>::integrate(f3, 0, half_pi<Real>());
    Q_expected = boost::math::expm1(half_pi<Real>())*half<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

    // Example 4:
    auto f4 = [](Real x)->Real { Real t0 = sqrt(x*x + 2); return atan(t0)/(t0*(x*x+1)); };
    Q = gauss<Real, Points>::integrate(f4, 0 , 1);
    Q_expected = 5*pi<Real>()*pi<Real>()/96;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

    tol = expected_error<Points>(test_three_quad_error_id_2);
    // Example 5:
    auto f5 = [](const Real& t) { return sqrt(t)*log(t); };
    Q = gauss<Real, Points>::integrate(f5, 0 , 1);
    Q_expected = -4/ (Real) 9;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

    // Example 6:
    auto f6 = [](const Real& t) { return sqrt(1 - t*t); };
    Q = gauss<Real, Points>::integrate(f6, 0 , 1);
    Q_expected = pi<Real>()/4;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
}


template<class Real, unsigned Points>
void test_integration_over_real_line()
{
    std::cout << "Testing integrals over entire real line in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = expected_error<Points>(test_integration_over_real_line_error_id);
    Real Q;
    Real Q_expected;
    Real L1;

    auto f1 = [](const Real& t) { return 1/(1+t*t);};
    Q = gauss<Real, Points>::integrate(f1, -boost::math::tools::max_value<Real>(), boost::math::tools::max_value<Real>(), &L1);
    Q_expected = pi<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
}

template<class Real, unsigned Points>
void test_right_limit_infinite()
{
    std::cout << "Testing right limit infinite for Gaussian quadrature in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = expected_error<Points>(test_right_limit_infinite_error_id);
    Real Q;
    Real Q_expected;
    Real L1;

    // Example 11:
    auto f1 = [](const Real& t) { return 1/(1+t*t);};
    Q = gauss<Real, Points>::integrate(f1, 0, boost::math::tools::max_value<Real>(), &L1);
    Q_expected = half_pi<Real>();
    BOOST_CHECK_CLOSE(Q, Q_expected, 100*tol);

    auto f4 = [](const Real& t) { return 1/(1+t*t); };
    Q = gauss<Real, Points>::integrate(f4, 1, boost::math::tools::max_value<Real>(), &L1);
    Q_expected = pi<Real>()/4;
    BOOST_CHECK_CLOSE(Q, Q_expected, 100*tol);
}

template<class Real, unsigned Points>
void test_left_limit_infinite()
{
    std::cout << "Testing left limit infinite for Gaussian quadrature in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = expected_error<Points>(test_left_limit_infinite_error_id);
    Real Q;
    Real Q_expected;

    // Example 11:
    auto f1 = [](const Real& t) { return 1/(1+t*t);};
    Q = gauss<Real, Points>::integrate(f1, -boost::math::tools::max_value<Real>(), Real(0));
    Q_expected = half_pi<Real>();
    BOOST_CHECK_CLOSE(Q, Q_expected, 100*tol);
}

template<class Complex>
void test_complex_lambert_w()
{
    #ifndef BOOST_MATH_STANDALONE
    std::cout << "Testing that complex-valued integrands are integrated correctly by Gaussian quadrature on type " << boost::typeindex::type_id<Complex>().pretty_name() << "\n";
    typedef typename Complex::value_type Real;
    Real tol = 10e-9;
    using boost::math::constants::pi;
    Complex z{2, 3};
    auto lw = [&z](Real v)->Complex {
      using std::cos;
      using std::sin;
      using std::exp;
      Real sinv = sin(v);
      Real cosv = cos(v);

      Real cotv = cosv/sinv;
      Real cscv = 1/sinv;
      Real t = (1-v*cotv)*(1-v*cotv) + v*v;
      Real x = v*cscv*exp(-v*cotv);
      Complex den = z + x;
      Complex num = t*(z/pi<Real>());
      Complex res = num/den;
      return res;
    };

    //N[ProductLog[2+3*I], 150]
    Complex Q = gauss<Real, 30>::integrate(lw, (Real) 0, pi<Real>());
    BOOST_CHECK_CLOSE_FRACTION(Q.real(), BOOST_MATH_TEST_VALUE(Real, 1.0900765344857908463017778267816696498710210863535777805644), tol);
    BOOST_CHECK_CLOSE_FRACTION(Q.imag(), BOOST_MATH_TEST_VALUE(Real, 0.5301397207748388014268602135741217419287056313827031782979), tol);
    #endif
}

BOOST_AUTO_TEST_CASE(gauss_quadrature_test)
{
  
#ifdef TEST1

#ifdef __STDCPP_FLOAT64_T__
    test_linear<std::float64_t, 7>();
    test_quadratic<std::float64_t, 7>();
    test_ca<std::float64_t, 7>();
    test_three_quadrature_schemes_examples<std::float64_t, 7>();
    test_integration_over_real_line<std::float64_t, 7>();
    test_right_limit_infinite<std::float64_t, 7>();
    test_left_limit_infinite<std::float64_t, 7>();

    test_linear<std::float64_t, 9>();
    test_quadratic<std::float64_t, 9>();
    test_ca<std::float64_t, 9>();
    test_three_quadrature_schemes_examples<std::float64_t, 9>();
    test_integration_over_real_line<std::float64_t, 9>();
    test_right_limit_infinite<std::float64_t, 9>();
    test_left_limit_infinite<std::float64_t, 9>();
#else
    test_linear<double, 7>();
    test_quadratic<double, 7>();
    test_ca<double, 7>();
    test_three_quadrature_schemes_examples<double, 7>();
    test_integration_over_real_line<double, 7>();
    test_right_limit_infinite<double, 7>();
    test_left_limit_infinite<double, 7>();

    test_linear<double, 9>();
    test_quadratic<double, 9>();
    test_ca<double, 9>();
    test_three_quadrature_schemes_examples<double, 9>();
    test_integration_over_real_line<double, 9>();
    test_right_limit_infinite<double, 9>();
    test_left_limit_infinite<double, 9>();
#endif

#if LDBL_MANT_DIG < 100 && defined(BOOST_MATH_RUN_MP_TESTS)
    test_linear<cpp_bin_float_quad, 10>();
    test_quadratic<cpp_bin_float_quad, 10>();
    test_ca<cpp_bin_float_quad, 10>();
    test_three_quadrature_schemes_examples<cpp_bin_float_quad, 10>();
    test_integration_over_real_line<cpp_bin_float_quad, 10>();
    test_right_limit_infinite<cpp_bin_float_quad, 10>();
    test_left_limit_infinite<cpp_bin_float_quad, 10>();
#endif
#endif
#ifdef TEST2
#if LDBL_MANT_DIG < 100 && defined(BOOST_MATH_RUN_MP_TESTS)
    test_linear<cpp_bin_float_quad, 15>();
    test_quadratic<cpp_bin_float_quad, 15>();
    test_ca<cpp_bin_float_quad, 15>();
    test_three_quadrature_schemes_examples<cpp_bin_float_quad, 15>();
    test_integration_over_real_line<cpp_bin_float_quad, 15>();
    test_right_limit_infinite<cpp_bin_float_quad, 15>();
    test_left_limit_infinite<cpp_bin_float_quad, 15>();

    test_linear<cpp_bin_float_quad, 20>();
    test_quadratic<cpp_bin_float_quad, 20>();
    test_ca<cpp_bin_float_quad, 20>();
    test_three_quadrature_schemes_examples<cpp_bin_float_quad, 20>();
    test_integration_over_real_line<cpp_bin_float_quad, 20>();
    test_right_limit_infinite<cpp_bin_float_quad, 20>();
    test_left_limit_infinite<cpp_bin_float_quad, 20>();

    test_linear<cpp_bin_float_quad, 25>();
    test_quadratic<cpp_bin_float_quad, 25>();
    test_ca<cpp_bin_float_quad, 25>();
    test_three_quadrature_schemes_examples<cpp_bin_float_quad, 25>();
    test_integration_over_real_line<cpp_bin_float_quad, 25>();
    test_right_limit_infinite<cpp_bin_float_quad, 25>();
    test_left_limit_infinite<cpp_bin_float_quad, 25>();

    test_linear<cpp_bin_float_quad, 30>();
    test_quadratic<cpp_bin_float_quad, 30>();
    test_ca<cpp_bin_float_quad, 30>();
    test_three_quadrature_schemes_examples<cpp_bin_float_quad, 30>();
    test_integration_over_real_line<cpp_bin_float_quad, 30>();
    test_right_limit_infinite<cpp_bin_float_quad, 30>();
    test_left_limit_infinite<cpp_bin_float_quad, 30>();

#endif
#endif
#ifdef TEST3
#if LDBL_MANT_DIG < 100 && defined(BOOST_MATH_RUN_MP_TESTS)
    test_left_limit_infinite<cpp_bin_float_quad, 30>();
#endif
    test_complex_lambert_w<std::complex<double>>();
    test_complex_lambert_w<std::complex<long double>>();
#ifdef BOOST_HAS_FLOAT128
    test_left_limit_infinite<boost::multiprecision::float128, 30>();
    test_complex_lambert_w<boost::multiprecision::complex128>();
#endif
#if LDBL_MANT_DIG < 100 && defined(BOOST_MATH_RUN_MP_TESTS)
    test_complex_lambert_w<boost::multiprecision::cpp_complex_quad>();
#endif
#endif
}

#else

int main() { return 0; }

#endif
