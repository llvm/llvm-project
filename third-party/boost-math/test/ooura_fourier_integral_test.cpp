// Copyright Nick Thompson, 2019
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
#define BOOST_TEST_MODULE test_ooura_fourier_transform

#include <cmath>
#include <iostream>
#include <boost/type_index.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/quadrature/ooura_fourier_integrals.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

using boost::math::quadrature::ooura_fourier_sin;
using boost::math::quadrature::ooura_fourier_cos;
using boost::math::constants::pi;


float float_tol = 10*std::numeric_limits<float>::epsilon();
ooura_fourier_sin<float> float_sin_integrator(float_tol);

double double_tol = 10*std::numeric_limits<double>::epsilon();
ooura_fourier_sin<double> double_sin_integrator(double_tol);

long double long_double_tol = 10*std::numeric_limits<long double>::epsilon();
ooura_fourier_sin<long double> long_double_sin_integrator(long_double_tol);

template<class Real>
auto get_sin_integrator() {
    if constexpr (std::is_same_v<Real, float>) {
        return float_sin_integrator;
    }
    if constexpr (std::is_same_v<Real, double>) {
        return double_sin_integrator;
    }
    if constexpr (std::is_same_v<Real, long double>) {
        return long_double_sin_integrator;
    }
}

ooura_fourier_cos<float> float_cos_integrator(float_tol);
ooura_fourier_cos<double> double_cos_integrator(double_tol);
ooura_fourier_cos<long double> long_double_cos_integrator(long_double_tol);

template<class Real>
auto get_cos_integrator() {
    if constexpr (std::is_same_v<Real, float>) {
        return float_cos_integrator;
    }
    if constexpr (std::is_same_v<Real, double>) {
        return double_cos_integrator;
    }
    if constexpr (std::is_same_v<Real, long double>) {
        return long_double_cos_integrator;
    }
}


template<class Real>
void test_ooura_eta()
{
    using boost::math::quadrature::detail::ooura_eta;
    std::cout << "Testing eta function on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    {
        Real x = 0;
        Real alpha = 7;
        auto [eta, eta_prime] = ooura_eta(x, alpha);
        BOOST_CHECK_SMALL(eta, (std::numeric_limits<Real>::min)());
        BOOST_CHECK_CLOSE_FRACTION(eta_prime, 2 + alpha + Real(1)/Real(4), 10*std::numeric_limits<Real>::epsilon());
    }

    {
        Real alpha = 4;
        for (Real z = 0.125; z < 500; z += 0.125) {
            Real x = std::log(z);
            auto [eta, eta_prime] = ooura_eta(x, alpha);
            BOOST_CHECK_CLOSE_FRACTION(eta, 2*x + alpha*(1-1/z) + (z-1)/4, 10*std::numeric_limits<Real>::epsilon());
            BOOST_CHECK_CLOSE_FRACTION(eta_prime, 2 + alpha/z + z/4, 10*std::numeric_limits<Real>::epsilon());
        }
    }
}

template<class Real>
void test_ooura_sin_nodes_and_weights()
{
    using boost::math::quadrature::detail::ooura_sin_node_and_weight;
    using boost::math::quadrature::detail::ooura_eta;
    using std::exp;
    std::cout << "Testing nodes and weights on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    {
        long n = 1;
        Real alpha = 1;
        Real h = 1;
        auto [node, weight] = ooura_sin_node_and_weight(n, h, alpha);
        Real expected_node = pi<Real>()/(1-exp(-ooura_eta(n*h, alpha).first));
        BOOST_CHECK_CLOSE_FRACTION(node,  expected_node,10*std::numeric_limits<Real>::epsilon());
    }
}

template<class Real>
void test_ooura_alpha() {
    std::cout << "Testing Ooura alpha on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    using std::sqrt;
    using std::log1p;
    using boost::math::quadrature::detail::calculate_ooura_alpha;
    Real alpha = calculate_ooura_alpha(Real(1));
    Real expected = 1/sqrt(16 + 4*log1p(pi<Real>()));
    BOOST_CHECK_CLOSE_FRACTION(alpha, expected, 10*std::numeric_limits<Real>::epsilon());
}

void test_node_weight_precision_agreement()
{
    using std::abs;
    using boost::math::quadrature::detail::ooura_sin_node_and_weight;
    using boost::math::quadrature::detail::ooura_eta;
    using boost::multiprecision::cpp_bin_float_quad;
    std::cout << "Testing agreement in two different precisions of nodes and weights\n";
    cpp_bin_float_quad alpha_quad = 1;
    long int_max = 128;
    cpp_bin_float_quad h_quad = 1/cpp_bin_float_quad(int_max);
    double alpha_dbl = 1;
    double h_dbl = static_cast<double>(h_quad);
    std::cout << std::fixed;
    for (long n = -1; n > -6*int_max; --n) {
        auto [node_dbl, weight_dbl] = ooura_sin_node_and_weight(n, h_dbl, alpha_dbl);
        auto p = ooura_sin_node_and_weight(n, h_quad, alpha_quad);
        double node_quad = static_cast<double>(p.first);
        double weight_quad = static_cast<double>(p.second);
        auto node_dist = abs(boost::math::float_distance(node_quad, node_dbl));
        if ( (weight_quad < 0 && weight_dbl > 0) || (weight_dbl < 0 && weight_quad > 0) ){
            std::cout << "Weights at different precisions have different signs!\n";
        } else {
            auto weight_dist = abs(boost::math::float_distance(weight_quad, weight_dbl));
            if (weight_dist > 100) {
                std::cout << std::fixed;
                std::cout <<"n =" << n << ", x = " << n*h_dbl << ", node distance = " << node_dist << ", weight distance = " << weight_dist << "\n";
                std::cout << std::scientific;
                std::cout << "computed weight = " << weight_dbl << ", actual weight = " << weight_quad << "\n";
            }
        }
    }

}

template<class Real>
void test_sinc()
{
    std::cout << "Testing sinc integral on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    using std::numeric_limits;
    Real tol = 50*numeric_limits<Real>::epsilon();
    auto integrator = get_sin_integrator<Real>();
    auto f = [](Real x)->Real { return 1/x; };
    Real omega = 1;
    while (omega < 10)
    {
        auto [Is, err] = integrator.integrate(f, omega);
        BOOST_CHECK_CLOSE_FRACTION(Is, pi<Real>()/2, tol);

        auto [Isn, errn] = integrator.integrate(f, -omega);
        BOOST_CHECK_CLOSE_FRACTION(Isn, -pi<Real>()/2, tol);
        omega += 1;
    }
}


template<class Real>
void test_exp()
{
    std::cout << "Testing exponential integral on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    using std::exp;
    using std::numeric_limits;
    Real tol = 50*numeric_limits<Real>::epsilon();
    auto integrator = get_sin_integrator<Real>();
    auto f = [](Real x)->Real {return exp(-x);};
    Real omega = 1;
    while (omega < 5)
    {
        auto [Is, err] = integrator.integrate(f, omega);
        Real exact = omega/(1+omega*omega);
        BOOST_CHECK_CLOSE_FRACTION(Is, exact, tol);
        omega += 1;
    }
}


template<class Real>
void test_root()
{
    std::cout << "Testing integral of sin(kx)/sqrt(x) on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    using std::sqrt;
    using std::numeric_limits;
    Real tol = 10*numeric_limits<Real>::epsilon();
    auto integrator = get_sin_integrator<Real>();
    auto f = [](Real x)->Real { return 1/sqrt(x);};
    Real omega = 1;
    while (omega < 5) {
        auto [Is, err] = integrator.integrate(f, omega);
        Real exact = sqrt(pi<Real>()/(2*omega));
        BOOST_CHECK_CLOSE_FRACTION(Is, exact, 10*tol);
        omega += 1;
    }
}

// See: https://scicomp.stackexchange.com/questions/32790/numerical-evaluation-of-highly-oscillatory-integral/32799#32799
template<class Real>
Real asymptotic(Real lambda) {
    using std::sin;
    using std::cos;
    using boost::math::constants::pi;
    Real I1 = cos(lambda - pi<Real>()/4)*sqrt(2*pi<Real>()/lambda);
    Real I2 = sin(lambda - pi<Real>()/4)*sqrt(2*pi<Real>()/(lambda*lambda*lambda))/8;
    return I1 + I2;
}

template<class Real>
void test_double_osc()
{
    std::cout << "Testing double oscillation on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    using std::sqrt;
    using std::numeric_limits;
    auto integrator = get_sin_integrator<Real>();
    Real lambda = 7;
    auto f = [&lambda](Real x)->Real { return cos(lambda*cos(x))/x; };
    Real omega = 1;
    auto [Is, err] = integrator.integrate(f, omega);
    Real exact = asymptotic(lambda);
    BOOST_CHECK_CLOSE_FRACTION(2*Is, exact, 0.05);
}

template<class Real>
void test_zero_integrand()
{
    // Make sure relative error tolerance doesn't break on zero integrand:
    std::cout << "Testing zero integrand on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    using std::sqrt;
    using std::numeric_limits;
    auto integrator = get_sin_integrator<Real>();
    auto f = [](Real /* x */)->Real { return Real(0); };
    Real omega = 1;
    auto [Is, err] = integrator.integrate(f, omega);
    Real exact = 0;
    BOOST_CHECK_EQUAL(Is, exact);
}


// This works, but doesn't recover the precision you want in a unit test:
// template<class Real>
// void test_log()
// {
//     std::cout << "Testing integral of log(x)sin(x) on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
//     using std::log;
//     using std::exp;
//     using std::numeric_limits;
//     using boost::math::constants::euler;
//     Real tol = 1000*numeric_limits<Real>::epsilon();
//     auto f = [](Real x)->Real { return exp(-100*numeric_limits<Real>::epsilon()*x)*log(x);};
//     Real omega = 1;
//     Real Is = ooura_fourier_sin<decltype(f), Real>(f, omega, sqrt(numeric_limits<Real>::epsilon())/100);
//     BOOST_CHECK_CLOSE_FRACTION(Is, -euler<Real>(), tol);
// }


template<class Real>
void test_cos_integral1()
{
    std::cout << "Testing integral of cos(x)/(x*x+1) on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    using std::exp;
    using boost::math::constants::half_pi;
    using boost::math::constants::e;
    using std::numeric_limits;
    Real tol = 10*numeric_limits<Real>::epsilon();

    auto integrator = get_cos_integrator<Real>();
    auto f = [](Real x)->Real { return 1/(x*x+1);};
    Real omega = 1;
    auto [Is, err] = integrator.integrate(f, omega);
    Real exact = half_pi<Real>()/e<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Is, exact, tol);
}

template<class Real>
void test_cos_integral2()
{
    std::cout << "Testing integral of exp(-a*x) on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    using std::exp;
    using boost::math::constants::half_pi;
    using boost::math::constants::e;
    using std::numeric_limits;
    Real tol = 10*numeric_limits<Real>::epsilon();

    auto integrator = get_cos_integrator<Real>();
    for (Real a = 1; a < 5; ++a) {
        auto f = [&a](Real x)->Real { return exp(-a*x);};
        for(Real omega = 1; omega < 3; ++omega) {
            auto [Is, err] = integrator.integrate(f, omega);
            Real exact = a/(a*a+omega*omega);
            BOOST_CHECK_CLOSE_FRACTION(Is, exact, 50*tol);
        }
    }
}

template<class Real>
void test_nodes()
{
    std::cout << "Testing nodes and weights on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    auto sin_integrator = get_sin_integrator<Real>();

    auto const & big_nodes = sin_integrator.big_nodes();
    for (auto & node_row : big_nodes) {
        Real t0 = node_row[0];
        for (size_t i = 1; i < node_row.size(); ++i) {
            Real t1 = node_row[i];
            BOOST_CHECK(t1 > t0);
            t0 = t1;
        }
    }

    auto const & little_nodes = sin_integrator.little_nodes();
    for (auto & node_row : little_nodes) {
        Real t0 = node_row[0];
        for (size_t i = 1; i < node_row.size(); ++i) {
            Real t1 = node_row[i];
            BOOST_CHECK(t1 < t0);
            t0 = t1;
        }
    }
}


BOOST_AUTO_TEST_CASE(ooura_fourier_transform_test)
{
    test_cos_integral1<float>();
    test_cos_integral1<double>();
    test_cos_integral1<long double>();

    test_cos_integral2<float>();
    test_cos_integral2<double>();
    test_cos_integral2<long double>();

    //test_node_weight_precision_agreement();
    test_zero_integrand<float>();
    test_zero_integrand<double>();

    test_ooura_eta<float>();
    test_ooura_eta<double>();
    test_ooura_eta<long double>();

    test_ooura_sin_nodes_and_weights<float>();
    test_ooura_sin_nodes_and_weights<double>();
    test_ooura_sin_nodes_and_weights<long double>();

    test_ooura_alpha<float>();
    test_ooura_alpha<double>();
    test_ooura_alpha<long double>();

    test_sinc<float>();
    test_sinc<double>();
    test_sinc<long double>();

    test_exp<float>();
    test_exp<double>();
    test_exp<long double>();

    test_root<float>();
    test_root<double>();

    test_double_osc<float>();
    test_double_osc<double>();
    // Takes too long!
    //test_double_osc<long double>();

    // This test should be last:
    test_nodes<float>();
    test_nodes<double>();
    test_nodes<long double>();
}
