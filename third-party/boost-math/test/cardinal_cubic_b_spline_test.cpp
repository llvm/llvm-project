// Copyright Nick Thompson, 2017
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE test_cubic_b_spline
#include <cmath>
#include <random>
#include <functional>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/type_index.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <boost/math/interpolators/detail/cardinal_cubic_b_spline_detail.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using std::sin;
using boost::multiprecision::cpp_bin_float_50;
using boost::math::constants::third;
using boost::math::constants::half;

template<class Real>
void test_b3_spline()
{
    std::cout << "Testing evaluation of spline basis functions on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    // Outside the support:
    Real eps = std::numeric_limits<Real>::epsilon();
    BOOST_CHECK_SMALL(boost::math::interpolators::detail::b3_spline<Real>(Real(2.5)), (Real) 0);
    BOOST_CHECK_SMALL(boost::math::interpolators::detail::b3_spline<Real>(Real(-2.5)), (Real) 0);
    BOOST_CHECK_SMALL(boost::math::interpolators::detail::b3_spline_prime<Real>(Real(2.5)), (Real) 0);
    BOOST_CHECK_SMALL(boost::math::interpolators::detail::b3_spline_prime<Real>(Real(-2.5)), (Real) 0);
    BOOST_CHECK_SMALL(boost::math::interpolators::detail::b3_spline_double_prime<Real>(Real(2.5)), (Real) 0);
    BOOST_CHECK_SMALL(boost::math::interpolators::detail::b3_spline_double_prime<Real>(Real(-2.5)), (Real) 0);


    // On the boundary of support:
    BOOST_CHECK_SMALL(boost::math::interpolators::detail::b3_spline<Real>(2), (Real) 0);
    BOOST_CHECK_SMALL(boost::math::interpolators::detail::b3_spline<Real>(-2), (Real) 0);
    BOOST_CHECK_SMALL(boost::math::interpolators::detail::b3_spline_prime<Real>(2), (Real) 0);
    BOOST_CHECK_SMALL(boost::math::interpolators::detail::b3_spline_prime<Real>(-2), (Real) 0);

    // Special values:
    BOOST_CHECK_CLOSE(boost::math::interpolators::detail::b3_spline<Real>(-1), third<Real>()*half<Real>(), eps);
    BOOST_CHECK_CLOSE(boost::math::interpolators::detail::b3_spline<Real>( 1), third<Real>()*half<Real>(), eps);
    BOOST_CHECK_CLOSE(boost::math::interpolators::detail::b3_spline<Real>(0), 2*third<Real>(), eps);

    BOOST_CHECK_CLOSE(boost::math::interpolators::detail::b3_spline_prime<Real>(-1), half<Real>(), eps);
    BOOST_CHECK_CLOSE(boost::math::interpolators::detail::b3_spline_prime<Real>( 1), -half<Real>(), eps);
    BOOST_CHECK_SMALL(boost::math::interpolators::detail::b3_spline_prime<Real>(0), eps);

    // Properties: B3 is an even function, B3' is an odd function.
    for (size_t i = 1; i < 200; ++i)
    {
        Real arg = i*Real(0.01);
        BOOST_CHECK_CLOSE(boost::math::interpolators::detail::b3_spline<Real>(arg), boost::math::interpolators::detail::b3_spline<Real>(arg), eps);
        BOOST_CHECK_CLOSE(boost::math::interpolators::detail::b3_spline_prime<Real>(-arg), -boost::math::interpolators::detail::b3_spline_prime<Real>(arg), eps);
        BOOST_CHECK_CLOSE(boost::math::interpolators::detail::b3_spline_double_prime<Real>(-arg), boost::math::interpolators::detail::b3_spline_double_prime<Real>(arg), eps);
    }

}
/*
 * This test ensures that the interpolant s(x_j) = f(x_j) at all grid points.
 */
template<class Real>
void test_interpolation_condition()
{
    using std::sqrt;
    std::cout << "Testing interpolation condition for cubic b splines on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    std::random_device rd;
    std::mt19937 gen(rd());
    boost::random::uniform_real_distribution<Real> dis(1, 10);
    std::vector<Real> v(5000);
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = dis(gen);
    }

    Real step = static_cast<Real>(0.01);
    Real a = Real(5);
    boost::math::interpolators::cardinal_cubic_b_spline<Real> spline(v.data(), v.size(), a, step);

    for (size_t i = 0; i < v.size(); ++i)
    {
        Real y = spline(i*step + a);
        // This seems like a very large tolerance, but I don't know of any other interpolators
        // that will be able to do much better on random data.
        BOOST_CHECK_CLOSE(y, v[i], 10000*sqrt(std::numeric_limits<Real>::epsilon()));
    }

}


template<class Real>
void test_constant_function()
{
    std::cout << "Testing that constants are interpolated correctly by cubic b splines on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    std::vector<Real> v(500);
    const Real constant = Real(50.2);
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = constant;
    }

    Real step = static_cast<Real>(0.02);
    Real a = 5;
    boost::math::interpolators::cardinal_cubic_b_spline<Real> spline(v.data(), v.size(), a, step);

    for (size_t i = 0; i < v.size(); ++i)
    {
        // Do not test at interpolation point; we already know it works there:
        Real y = spline(i*step + a + Real(0.001));
        BOOST_CHECK_CLOSE(y, constant, 10*std::numeric_limits<Real>::epsilon());
        Real y_prime = spline.prime(i*step + a + Real(0.002));
        BOOST_CHECK_SMALL(y_prime, 5000*std::numeric_limits<Real>::epsilon());
        Real y_double_prime = spline.double_prime(i*step + a + Real(0.002));
        BOOST_CHECK_SMALL(y_double_prime, 5000*std::numeric_limits<Real>::epsilon());

    }

    // Test that correctly specified left and right-derivatives work properly:
    spline = boost::math::interpolators::cardinal_cubic_b_spline<Real>(v.data(), v.size(), a, step, 0, 0);

    for (size_t i = 0; i < v.size(); ++i)
    {
        Real y = spline(i*step + a + Real(0.002));
        BOOST_CHECK_CLOSE(y, constant, std::numeric_limits<Real>::epsilon());
        Real y_prime = spline.prime(i*step + a + Real(0.002));
        BOOST_CHECK_SMALL(y_prime, std::numeric_limits<Real>::epsilon());
    }

    //
    // Again with iterator constructor:
    //
    boost::math::interpolators::cardinal_cubic_b_spline<Real> spline2(v.begin(), v.end(), a, step);

    for (size_t i = 0; i < v.size(); ++i)
    {
       // Do not test at interpolation point; we already know it works there:
       Real y = spline2(i*step + a + Real(0.001));
       BOOST_CHECK_CLOSE(y, constant, 10 * std::numeric_limits<Real>::epsilon());
       Real y_prime = spline2.prime(i*step + a + Real(0.002));
       BOOST_CHECK_SMALL(y_prime, 5000 * std::numeric_limits<Real>::epsilon());
    }

    // Test that correctly specified left and right-derivatives work properly:
    spline2 = boost::math::interpolators::cardinal_cubic_b_spline<Real>(v.begin(), v.end(), a, step, 0, 0);

    for (size_t i = 0; i < v.size(); ++i)
    {
       Real y = spline2(i*step + a + Real(0.002));
       BOOST_CHECK_CLOSE(y, constant, std::numeric_limits<Real>::epsilon());
       Real y_prime = spline2.prime(i*step + a + Real(0.002));
       BOOST_CHECK_SMALL(y_prime, std::numeric_limits<Real>::epsilon());
    }
}


template<class Real>
void test_affine_function()
{
    using std::sqrt;
    std::cout << "Testing that affine functions are interpolated correctly by cubic b splines on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    std::vector<Real> v(500);
    Real a = 10;
    Real b = 8;
    Real step = static_cast<Real>(0.005);

    auto f = [a, b](Real x) { return a*x + b; };
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = f(i*step);
    }

    boost::math::interpolators::cardinal_cubic_b_spline<Real> spline(v.data(), v.size(), 0, step);

    for (size_t i = 0; i < v.size() - 1; ++i)
    {
        Real arg = static_cast<Real>(i*step + Real(0.0001));
        Real y = spline(arg);
        BOOST_CHECK_CLOSE(y, f(arg), sqrt(std::numeric_limits<Real>::epsilon()));
        Real y_prime = spline.prime(arg);
        BOOST_CHECK_CLOSE(y_prime, a, 100*sqrt(std::numeric_limits<Real>::epsilon()));
    }

    // Test that correctly specified left and right-derivatives work properly:
    spline = boost::math::interpolators::cardinal_cubic_b_spline<Real>(v.data(), v.size(), 0, step, a, a);

    for (size_t i = 0; i < v.size() - 1; ++i)
    {
        Real arg = static_cast<Real>(i*step + Real(0.0001));
        Real y = spline(arg);
        BOOST_CHECK_CLOSE(y, f(arg), sqrt(std::numeric_limits<Real>::epsilon()));
        Real y_prime = spline.prime(arg);
        BOOST_CHECK_CLOSE(y_prime, a, 100*sqrt(std::numeric_limits<Real>::epsilon()));
    }
}


template<class Real>
void test_quadratic_function()
{
    using std::sqrt;
    std::cout << "Testing that quadratic functions are interpolated correctly by cubic b splines on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    std::vector<Real> v(500);
    Real a = static_cast<Real>(1.2);
    Real b = static_cast<Real>(-3.4);
    Real c = static_cast<Real>(-8.6);
    Real step = static_cast<Real>(0.01);

    auto f = [a, b, c](Real x) { return a*x*x + b*x + c; };
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = f(i*step);
    }

    boost::math::interpolators::cardinal_cubic_b_spline<Real> spline(v.data(), v.size(), 0, step);

    for (size_t i = 0; i < v.size() -1; ++i)
    {
        Real arg = static_cast<Real>(i*step + Real(0.001));
        Real y = spline(arg);
        BOOST_CHECK_CLOSE(y, f(arg), Real(0.1));
        Real y_prime = spline.prime(arg);
        BOOST_CHECK_CLOSE(y_prime, 2*a*arg + b, Real(2.0));
    }
}


template<class Real>
void test_circ_conic_function()
{
    using std::sqrt;
    std::cout << "Testing that conic section of a circle is interpolated correctly by cubic b splines on type " << boost::typeindex::type_id<Real>().pretty_name() << '\n';
    std::vector<Real> v(500);
    Real cv = Real(0.1);
    Real w = Real(2.0);
    Real step = Real(2 * w / (v.size() - 1));

    auto f = [cv](Real x) { return cv * x * x / (1 + sqrt(1 - cv * cv * x * x)); };
    auto df = [cv](Real x) { return cv * x / sqrt(1 - cv * cv * x * x); };

    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = f(-w + i * step);
    }

    boost::math::interpolators::cardinal_cubic_b_spline<Real> spline(v.data(), v.size(), -w, step);

    Real tol = 100 * sqrt(std::numeric_limits<Real>::epsilon());
    if ((std::numeric_limits<Real>::digits > 100) || !std::numeric_limits<Real>::digits)
       tol *= 500000;
    // First check derivatives exactly at end points
    BOOST_CHECK_CLOSE(spline.prime(-w), df(-w), tol);
    BOOST_CHECK_CLOSE(spline.prime(w), df(w), tol);

    for (size_t i = 0; i < v.size() - 1; ++i)
    {
        Real arg = -w + i * step + Real(0.001);
        Real y = spline(arg);
        BOOST_CHECK_CLOSE(y, f(arg), Real(2.0));
        Real y_prime = spline.prime(arg);
        BOOST_CHECK_CLOSE(y_prime, df(arg), Real(1.0));
    }
}


template<class Real>
void test_trig_function()
{
    std::cout << "Testing that sine functions are interpolated correctly by cubic b splines on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    std::mt19937 gen;
    std::vector<Real> v(500);
    Real x0 = 1;
    Real step = Real(0.125);

    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = sin(x0 + step * i);
    }

    boost::math::interpolators::cardinal_cubic_b_spline<Real> spline(v.data(), v.size(), x0, step);

    boost::random::uniform_real_distribution<Real> abscissa(x0, x0 + 499 * step);

    for (size_t i = 0; i < v.size(); ++i)
    {
        Real x = abscissa(gen);
        Real y = spline(x);
        BOOST_CHECK_CLOSE(y, Real(sin(x)), Real(1.0));
        auto y_prime = spline.prime(x);
        BOOST_CHECK_CLOSE(y_prime, Real(cos(x)), Real(2.0));
    }
}

template<class Real>
void test_copy_move()
{
    std::cout << "Testing that copy/move operation succeed on cubic b spline\n";
    std::vector<Real> v(500);
    Real x0 = 1;
    Real step = static_cast<Real>(0.125);
    constexpr Real tol = Real(0.01);

    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = sin(x0 + step * i);
    }

    boost::math::interpolators::cardinal_cubic_b_spline<Real> spline(v.data(), v.size(), x0, step);


    // Default constructor should compile so that splines can be member variables:
    boost::math::interpolators::cardinal_cubic_b_spline<Real> d;
    d = boost::math::interpolators::cardinal_cubic_b_spline<Real>(v.data(), v.size(), x0, step);
    BOOST_CHECK_CLOSE(d(x0), Real(sin(x0)), tol);
    // Passing to lambda should compile:
    auto f = [=](Real x) { return d(x); };
    // Make sure this variable is used.
    BOOST_CHECK_CLOSE(f(x0), Real(sin(x0)), tol);

    // Move operations should compile.
    auto s = std::move(spline);

    // Copy operations should compile:
    boost::math::interpolators::cardinal_cubic_b_spline<Real> c = d;
    BOOST_CHECK_CLOSE(c(x0), Real(sin(x0)), tol);

    // Test with std::bind:
    auto h = std::bind(&boost::math::interpolators::cardinal_cubic_b_spline<Real>::operator(), &s, std::placeholders::_1);
    BOOST_CHECK_CLOSE(h(x0), Real(sin(x0)), tol);
}

template<class Real>
void test_outside_interval()
{
    std::cout << "Testing that the spline can be evaluated outside the interpolation interval\n";
    std::vector<Real> v(400);
    Real x0 = 1;
    Real step = Real(0.125);

    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = sin(x0 + step * i);
    }

    boost::math::interpolators::cardinal_cubic_b_spline<Real> spline(v.data(), v.size(), x0, step);

    // There's no test here; it simply does it's best to be an extrapolator.
    //
    std::ostream cnull(0);
    cnull << spline(0);
    cnull << spline(2000);
}

BOOST_AUTO_TEST_CASE(test_cubic_b_spline)
{
    #ifdef __STDCPP_FLOAT32_T__

    test_b3_spline<std::float32_t>();
    test_interpolation_condition<std::float32_t>();
    test_constant_function<std::float32_t>();
    test_affine_function<std::float32_t>();
    test_quadratic_function<std::float32_t>();
    test_circ_conic_function<std::float32_t>();
    test_trig_function<std::float32_t>();

    #else
    
    test_b3_spline<float>();
    test_interpolation_condition<float>();
    test_constant_function<float>();
    test_affine_function<float>();
    test_quadratic_function<float>();
    test_circ_conic_function<float>();
    test_trig_function<float>();

    #endif

    #ifdef __STDCPP_FLOAT64_T__

    test_b3_spline<std::float64_t>();
    test_interpolation_condition<std::float64_t>();
    test_constant_function<std::float64_t>();
    test_affine_function<std::float64_t>();
    test_quadratic_function<std::float64_t>();
    test_circ_conic_function<std::float64_t>();
    test_trig_function<std::float64_t>();
    test_copy_move<std::float64_t>();
    test_outside_interval<std::float64_t>();

    #else

    test_b3_spline<double>();
    test_interpolation_condition<double>();
    test_constant_function<double>();
    test_affine_function<double>();
    test_quadratic_function<double>();
    test_circ_conic_function<double>();
    test_trig_function<double>();
    test_copy_move<double>();
    test_outside_interval<double>();

    #endif

    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS

    test_b3_spline<long double>();
    test_interpolation_condition<long double>();
    test_constant_function<long double>();
    test_affine_function<long double>();
    test_quadratic_function<long double>();
    test_circ_conic_function<long double>();
    test_trig_function<long double>();

    #endif

    test_b3_spline<cpp_bin_float_50>();
    test_interpolation_condition<cpp_bin_float_50>();
    test_constant_function<cpp_bin_float_50>();
    test_affine_function<cpp_bin_float_50>();
    test_quadratic_function<cpp_bin_float_50>();
    test_trig_function<cpp_bin_float_50>();
}
