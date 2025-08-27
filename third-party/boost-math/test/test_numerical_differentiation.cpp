//  (C) Copyright Nick Thompson, 2018
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE numerical_differentiation_test

#include <cmath>
#include <limits>
#include <iostream>
#include <boost/type_index.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/bessel_prime.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/math/differentiation/finite_difference.hpp>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using std::abs;
using std::pow;
using boost::math::differentiation::finite_difference_derivative;
using boost::math::differentiation::complex_step_derivative;
using boost::math::cyl_bessel_j;
using boost::math::cyl_bessel_j_prime;
using boost::math::constants::half;

template<class Real, size_t order>
void test_order(size_t points_to_test)
{
    std::cout << "Testing order " <<  order  << " derivative error estimate on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    //std::cout << std::fixed << std::scientific;
    auto f = [](Real t) { return boost::math::cyl_bessel_j<Real>(1, t); };
    Real min = Real(-100000.0);
    Real max = -min;
    Real x = min;
    Real max_error = 0;
    Real max_relative_error_in_error = 0;
    size_t j = 0;
    size_t failures = 0;
    while (j < points_to_test)
    {
        x = min + (Real) 2*j*max/ (Real) points_to_test;
        Real error_estimate;
        Real computed = finite_difference_derivative<decltype(f), Real, order>(f, x, &error_estimate);
        Real expected = (Real) cyl_bessel_j_prime<Real>(1, x);
        Real error = abs(computed - expected);
        // The error estimate is provided under the assumption that the function is evaluated to 1 ULP.
        // Presumably no one will be too offended by this estimate being off by a factor of 2 or so.
        if (error > 2*error_estimate)
        {
            ++failures;
            Real relative_error_in_error = abs(error - error_estimate)/ error;
            if (relative_error_in_error > max_relative_error_in_error)
            {
                max_relative_error_in_error = relative_error_in_error;
            }
            if (relative_error_in_error > 2)
            {
                throw std::logic_error("Relative error in error is too high!");
            }
        }
        if (error > max_error)
        {
            max_error = error;
        }
        ++j;
    }
    //std::cout << "Maximum error :" << max_error << "\n";
    //std::cout <<  "Error estimate failed " << failures << " times out of " << points_to_test << "\n";
    //std::cout << "Failure rate: " << (double) failures / (double) points_to_test << "\n";
    //std::cout << "Maximum error in estimated error = " << max_relative_error_in_error << "\n";
    //Real convergence_rate = (Real) order/ (Real) (order + 1);
    //std::cout << "eps^(order/order+1) = " << pow(std::numeric_limits<Real>::epsilon(), convergence_rate) << "\n\n\n";

    bool max_error_good = max_error < 2*sqrt(std::numeric_limits<Real>::epsilon());
    BOOST_TEST(max_error_good);

    bool error_estimate_good = max_relative_error_in_error < (Real) 2;
    BOOST_TEST(error_estimate_good);

    double failure_rate = (double) failures / (double) points_to_test;
    BOOST_CHECK_SMALL(failure_rate, 0.05);
}

template<class Real>
void test_bessel()
{
    std::cout << "Testing numerical derivatives of Bessel's function on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10);

    Real eps = std::numeric_limits<Real>::epsilon();
    Real x = static_cast<Real>(25.1);
    auto f = [](Real t) { return boost::math::cyl_bessel_j(12, t); };

    Real computed = finite_difference_derivative<decltype(f), Real, 1>(f, x);
    Real expected = cyl_bessel_j_prime(12, x);
    Real error_estimate = Real(4*abs(f(x))*sqrt(eps));
    //std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    //std::cout << "cyl_bessel_j_prime: " << expected << std::endl;
    //std::cout << "First order fd    : " << computed << std::endl;
    //std::cout << "Error             : " << abs(computed - expected) << std::endl;
    //std::cout << "a prior error est : " << error_estimate << std::endl;

    BOOST_CHECK_CLOSE_FRACTION(expected, computed, 10*error_estimate);

    computed = finite_difference_derivative<decltype(f), Real, 2>(f, x);
    expected = cyl_bessel_j_prime(12, x);
    error_estimate = abs(f(x))*pow(eps, boost::math::constants::two_thirds<Real>());
    //std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    //std::cout << "cyl_bessel_j_prime: " << expected << std::endl;
    //std::cout << "Second order fd   : " << computed << std::endl;
    //std::cout << "Error             : " << abs(computed - expected) << std::endl;
    //std::cout << "a prior error est : " << error_estimate << std::endl;

    BOOST_CHECK_CLOSE_FRACTION(expected, computed, 50*error_estimate);

    computed = finite_difference_derivative<decltype(f), Real, 4>(f, x);
    expected = cyl_bessel_j_prime(12, x);
    error_estimate = abs(f(x))*pow(eps, (Real) 4 / (Real) 5);
    //std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    //std::cout << "cyl_bessel_j_prime: " << expected << std::endl;
    //std::cout << "Fourth order fd   : " << computed << std::endl;
    //std::cout << "Error             : " << abs(computed - expected) << std::endl;
    //std::cout << "a prior error est : " << error_estimate << std::endl;

    BOOST_CHECK_CLOSE_FRACTION(expected, computed, 25*error_estimate);


    computed = finite_difference_derivative<decltype(f), Real, 6>(f, x);
    expected = cyl_bessel_j_prime(12, x);
    error_estimate = abs(f(x))*pow(eps, (Real)  6/ (Real) 7);
    //std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    //std::cout << "cyl_bessel_j_prime: " << expected << std::endl;
    //std::cout << "Sixth order fd    : " << computed << std::endl;
    //std::cout << "Error             : " << abs(computed - expected) << std::endl;
    //std::cout << "a prior error est : " << error_estimate << std::endl;

    BOOST_CHECK_CLOSE_FRACTION(expected, computed, 100*error_estimate);

    computed = finite_difference_derivative<decltype(f), Real, 8>(f, x);
    expected = cyl_bessel_j_prime(12, x);
    error_estimate = abs(f(x))*pow(eps, (Real)  8/ (Real) 9);
    //std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    //std::cout << "cyl_bessel_j_prime: " << expected << std::endl;
    //std::cout << "Eighth order fd   : " << computed << std::endl;
    //std::cout << "Error             : " << abs(computed - expected) << std::endl;
    //std::cout << "a prior error est : " << error_estimate << std::endl;

    BOOST_CHECK_CLOSE_FRACTION(expected, computed, 25*error_estimate);
}

// Example of a function which is subject to catastrophic cancellation using finite-differences, but is almost perfectly stable using complex step:
template<class RealOrComplex>
RealOrComplex moler_example(RealOrComplex x)
{
    using std::sin;
    using std::cos;
    using std::exp;

    RealOrComplex cosx = cos(x);
    RealOrComplex sinx = sin(x);
    return exp(x)/(cosx*cosx*cosx + sinx*sinx*sinx);
}

template<class RealOrComplex>
RealOrComplex moler_example_derivative(RealOrComplex x)
{
    using std::sin;
    using std::cos;
    using std::exp;

    RealOrComplex expx = exp(x);
    RealOrComplex cosx = cos(x);
    RealOrComplex sinx = sin(x);
    RealOrComplex coscubed_sincubed = cosx*cosx*cosx + sinx*sinx*sinx;
    return (expx/coscubed_sincubed)*(1 - 3*(sinx*sinx*cosx - sinx*cosx*cosx)/ (coscubed_sincubed));
}


template<class Real>
void test_complex_step()
{
    using std::abs;
    using std::complex;
    using std::isfinite;
    using std::isnormal;
    std::cout << "Testing numerical derivatives of Bessel's function on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    Real x = -100;
    while ( x < 100 )
    {
        if (!isfinite(moler_example(x)))
        {
            x += 1;
            continue;
        }
        Real expected = moler_example_derivative<Real>(x);
        Real computed = complex_step_derivative(moler_example<complex<Real>>, x);
        if (!isfinite(expected))
        {
            x += 1;
            continue;
        }
        if (abs(expected) <= std::numeric_limits<Real>::epsilon())
        {
            bool issmall = computed < std::numeric_limits<Real>::epsilon();
            BOOST_TEST(issmall);
        }
        else
        {
            BOOST_CHECK_CLOSE_FRACTION(expected, computed, 200*std::numeric_limits<Real>::epsilon());
        }
        x += 1;
    }
}


BOOST_AUTO_TEST_CASE(numerical_differentiation_test)
{
    constexpr size_t points_to_test = 1000;
    
    #ifdef __STDCPP_FLOAT32_T__
    test_complex_step<std::float32_t>();
    test_bessel<std::float32_t>();
    test_order<std::float32_t, 1>(points_to_test);
    test_order<std::float32_t, 2>(points_to_test);
    test_order<std::float32_t, 4>(points_to_test);
    test_order<std::float32_t, 6>(points_to_test);
    test_order<std::float32_t, 8>(points_to_test);
    #else
    test_complex_step<float>();
    test_bessel<float>();
    test_order<float, 1>(points_to_test);
    test_order<float, 2>(points_to_test);
    test_order<float, 4>(points_to_test);
    test_order<float, 6>(points_to_test);
    test_order<float, 8>(points_to_test);
    #endif

    #ifdef __STDCPP_FLOAT64_T__
    test_complex_step<std::float64_t>();
    test_bessel<std::float64_t>();
    test_order<std::float64_t, 1>(points_to_test);
    test_order<std::float64_t, 2>(points_to_test);
    test_order<std::float64_t, 4>(points_to_test);
    test_order<std::float64_t, 6>(points_to_test);
    test_order<std::float64_t, 8>(points_to_test);
    #else
    test_complex_step<double>();
    test_bessel<double>();
    test_order<double, 1>(points_to_test);   
    test_order<double, 2>(points_to_test);
    test_order<double, 4>(points_to_test);
    test_order<double, 6>(points_to_test);
    test_order<double, 8>(points_to_test);
    #endif
}
