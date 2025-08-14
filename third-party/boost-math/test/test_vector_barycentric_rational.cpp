// Copyright Nick Thompson, 2019
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE vector_barycentric_rational

#include <cmath>
#include <random>
#include <array>
#include <Eigen/Dense>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/type_index.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/math/interpolators/vector_barycentric_rational.hpp>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using std::sqrt;
using std::abs;
using std::numeric_limits;

template<class Real>
void test_agreement_with_1d()
{
    std::cout << "Testing with 1D interpolation on type "
              << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    std::mt19937 gen(4723);
    boost::random::uniform_real_distribution<Real> dis(0.1f, 1);
    std::vector<Real> t(100);
    std::vector<Eigen::Vector2d> y(100);
    t[0] = dis(gen);
    y[0][0] = dis(gen);
    y[0][1] = dis(gen);
    for (size_t i = 1; i < t.size(); ++i)
    {
        t[i] = t[i-1] + dis(gen);
        y[i][0] = dis(gen);
        y[i][1] = dis(gen);
    }

    std::vector<Eigen::Vector2d> y_copy = y;
    std::vector<Real> t_copy = t;
    std::vector<Real> t_copy0 = t;
    std::vector<Real> t_copy1 = t;

    std::vector<Real> y_copy0(y.size());
    std::vector<Real> y_copy1(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        y_copy0[i] = y[i][0];
        y_copy1[i] = y[i][1];
    }

    boost::random::uniform_real_distribution<Real> dis2(t[0], t[t.size()-1]);
    boost::math::interpolators::vector_barycentric_rational<decltype(t), decltype(y)> interpolator(std::move(t), std::move(y));
    boost::math::interpolators::barycentric_rational<Real> scalar_interpolator0(std::move(t_copy0), std::move(y_copy0));
    boost::math::interpolators::barycentric_rational<Real> scalar_interpolator1(std::move(t_copy1), std::move(y_copy1));


    Eigen::Vector2d z;

    size_t samples = 0;
    while (samples++ < 1000)
    {
        Real t = dis2(gen);
        interpolator(z, t);
        BOOST_CHECK_CLOSE(z[0], scalar_interpolator0(t), 10000*numeric_limits<Real>::epsilon());
        BOOST_CHECK_CLOSE(z[1], scalar_interpolator1(t), 10000*numeric_limits<Real>::epsilon());
    }
}


template<class Real>
void test_interpolation_condition_eigen()
{
    std::cout << "Testing interpolation condition for barycentric interpolation on Eigen vectors of type "
              << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    std::mt19937 gen(4723);
    boost::random::uniform_real_distribution<Real> dis(0.1f, 1);
    std::vector<Real> t(100);
    std::vector<Eigen::Vector2d> y(100);
    t[0] = dis(gen);
    y[0][0] = dis(gen);
    y[0][1] = dis(gen);
    for (size_t i = 1; i < t.size(); ++i)
    {
        t[i] = t[i-1] + dis(gen);
        y[i][0] = dis(gen);
        y[i][1] = dis(gen);
    }

    std::vector<Eigen::Vector2d> y_copy = y;
    std::vector<Real> t_copy = t;
    boost::math::interpolators::vector_barycentric_rational<decltype(t), decltype(y)> interpolator(std::move(t), std::move(y));

    Eigen::Vector2d z;
    for (size_t i = 0; i < t_copy.size(); ++i)
    {
        interpolator(z, t_copy[i]);
        BOOST_CHECK_CLOSE(z[0], y_copy[i][0], 100*numeric_limits<Real>::epsilon());
        BOOST_CHECK_CLOSE(z[1], y_copy[i][1], 100*numeric_limits<Real>::epsilon());
    }
}

template<class Real>
void test_interpolation_condition_std_array()
{
    std::cout << "Testing interpolation condition for barycentric interpolation on std::array vectors of type "
              << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    std::mt19937 gen(4723);
    boost::random::uniform_real_distribution<Real> dis(0.1f, 1);
    std::vector<Real> t(100);
    std::vector<std::array<Real, 2>> y(100);
    t[0] = dis(gen);
    y[0][0] = dis(gen);
    y[0][1] = dis(gen);
    for (size_t i = 1; i < t.size(); ++i)
    {
        t[i] = t[i-1] + dis(gen);
        y[i][0] = dis(gen);
        y[i][1] = dis(gen);
    }

    std::vector<std::array<Real, 2>> y_copy = y;
    std::vector<Real> t_copy = t;
    boost::math::interpolators::vector_barycentric_rational<decltype(t), decltype(y)> interpolator(std::move(t), std::move(y));

    std::array<Real, 2> z;
    for (size_t i = 0; i < t_copy.size(); ++i)
    {
        interpolator(z, t_copy[i]);
        BOOST_CHECK_CLOSE(z[0], y_copy[i][0], 100*numeric_limits<Real>::epsilon());
        BOOST_CHECK_CLOSE(z[1], y_copy[i][1], 100*numeric_limits<Real>::epsilon());
    }
}


template<class Real>
void test_interpolation_condition_ublas()
{
    std::cout << "Testing interpolation condition for barycentric interpolation ublas vectors of type "
              << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    std::mt19937 gen(4723);
    boost::random::uniform_real_distribution<Real> dis(0.1f, 1);
    std::vector<Real> t(100);
    std::vector<boost::numeric::ublas::vector<Real>> y(100);
    t[0] = dis(gen);
    y[0].resize(2);
    y[0][0] = dis(gen);
    y[0][1] = dis(gen);
    for (size_t i = 1; i < t.size(); ++i)
    {
        t[i] = t[i-1] + dis(gen);
        y[i].resize(2);
        y[i][0] = dis(gen);
        y[i][1] = dis(gen);
    }

    std::vector<Real> t_copy = t;
    std::vector<boost::numeric::ublas::vector<Real>> y_copy = y;

    boost::math::interpolators::vector_barycentric_rational<decltype(t), decltype(y)> interpolator(std::move(t), std::move(y));

    boost::numeric::ublas::vector<Real> z(2);
    for (size_t i = 0; i < t_copy.size(); ++i)
    {
        interpolator(z, t_copy[i]);
        BOOST_CHECK_CLOSE(z[0], y_copy[i][0], 100*numeric_limits<Real>::epsilon());
        BOOST_CHECK_CLOSE(z[1], y_copy[i][1], 100*numeric_limits<Real>::epsilon());
    }
}

template<class Real>
void test_interpolation_condition_high_order()
{
    std::cout << "Testing interpolation condition in high order for barycentric interpolation on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";
    std::mt19937 gen(5);
    boost::random::uniform_real_distribution<Real> dis(0.1f, 1);
    std::vector<Real> t(100);
    std::vector<Eigen::Vector2d> y(100);
    t[0] = dis(gen);
    y[0][0] = dis(gen);
    y[0][1] = dis(gen);
    for (size_t i = 1; i < t.size(); ++i)
    {
        t[i] = t[i-1] + dis(gen);
        y[i][0] = dis(gen);
        y[i][1] = dis(gen);
    }

    std::vector<Eigen::Vector2d> y_copy = y;
    std::vector<Real> t_copy = t;
    boost::math::interpolators::vector_barycentric_rational<decltype(t), decltype(y)> interpolator(std::move(t), std::move(y), 5);

    Eigen::Vector2d z;
    for (size_t i = 0; i < t_copy.size(); ++i)
    {
        interpolator(z, t_copy[i]);
        BOOST_CHECK_CLOSE(z[0], y_copy[i][0], 100*numeric_limits<Real>::epsilon());
        BOOST_CHECK_CLOSE(z[1], y_copy[i][1], 100*numeric_limits<Real>::epsilon());
    }
}


template<class Real>
void test_constant_eigen()
{
    std::cout << "Testing that constants are interpolated correctly using barycentric interpolation on Eigen vectors of type "
              << boost::typeindex::type_id<Real>().pretty_name() << "\n";

    std::mt19937 gen(6);
    boost::random::uniform_real_distribution<Real> dis(0.1f, 1);
    std::vector<Real> t(100);
    std::vector<Eigen::Vector2d> y(100);
    t[0] = dis(gen);
    Real constant0 = dis(gen);
    Real constant1 = dis(gen);
    y[0][0] = constant0;
    y[0][1] = constant1;
    for (size_t i = 1; i < t.size(); ++i)
    {
        t[i] = t[i-1] + dis(gen);
        y[i][0] = constant0;
        y[i][1] = constant1;
    }

    std::vector<Eigen::Vector2d> y_copy = y;
    std::vector<Real> t_copy = t;
    boost::math::interpolators::vector_barycentric_rational<decltype(t), decltype(y)> interpolator(std::move(t), std::move(y));

    Eigen::Vector2d z;
    for (size_t i = 0; i < t_copy.size(); ++i)
    {
        // Don't evaluate the constant at x[i]; that's already tested in the interpolation condition test.
        Real t = t_copy[i] + dis(gen);
        z = interpolator(t);
        BOOST_CHECK_CLOSE(z[0], constant0, 100*sqrt(numeric_limits<Real>::epsilon()));
        BOOST_CHECK_CLOSE(z[1], constant1, 100*sqrt(numeric_limits<Real>::epsilon()));
        Eigen::Vector2d zprime = interpolator.prime(t);
        Real zero_0 = zprime[0];
        Real zero_1 = zprime[1];
        BOOST_CHECK_SMALL(zero_0, sqrt(numeric_limits<Real>::epsilon()));
        BOOST_CHECK_SMALL(zero_1, sqrt(numeric_limits<Real>::epsilon()));
    }
}


template<class Real>
void test_constant_std_array()
{
    std::cout << "Testing that constants are interpolated correctly using barycentric interpolation on std::array vectors of type "
              << boost::typeindex::type_id<Real>().pretty_name() << "\n";

    std::mt19937 gen(6);
    boost::random::uniform_real_distribution<Real> dis(0.1f, 1);
    std::vector<Real> t(100);
    std::vector<std::array<Real, 2>> y(100);
    t[0] = dis(gen);
    Real constant0 = dis(gen);
    Real constant1 = dis(gen);
    y[0][0] = constant0;
    y[0][1] = constant1;
    for (size_t i = 1; i < t.size(); ++i)
    {
        t[i] = t[i-1] + dis(gen);
        y[i][0] = constant0;
        y[i][1] = constant1;
    }

    std::vector<std::array<Real,2>> y_copy = y;
    std::vector<Real> t_copy = t;
    boost::math::interpolators::vector_barycentric_rational<decltype(t), decltype(y)> interpolator(std::move(t), std::move(y));

    std::array<Real, 2> z;
    for (size_t i = 0; i < t_copy.size(); ++i)
    {
        // Don't evaluate the constant at x[i]; that's already tested in the interpolation condition test.
        Real t = t_copy[i] + dis(gen);
        z = interpolator(t);
        BOOST_CHECK_CLOSE(z[0], constant0, 100*sqrt(numeric_limits<Real>::epsilon()));
        BOOST_CHECK_CLOSE(z[1], constant1, 100*sqrt(numeric_limits<Real>::epsilon()));
        std::array<Real, 2> zprime = interpolator.prime(t);
        Real zero_0 = zprime[0];
        Real zero_1 = zprime[1];
        BOOST_CHECK_SMALL(zero_0, sqrt(numeric_limits<Real>::epsilon()));
        BOOST_CHECK_SMALL(zero_1, sqrt(numeric_limits<Real>::epsilon()));
    }
}


template<class Real>
void test_constant_high_order()
{
    std::cout << "Testing that constants are interpolated correctly using barycentric interpolation on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";

    std::mt19937 gen(6);
    boost::random::uniform_real_distribution<Real> dis(0.1f, 1);
    std::vector<Real> t(100);
    std::vector<Eigen::Vector2d> y(100);
    t[0] = dis(gen);
    Real constant0 = dis(gen);
    Real constant1 = dis(gen);
    y[0][0] = constant0;
    y[0][1] = constant1;
    for (size_t i = 1; i < t.size(); ++i)
    {
        t[i] = t[i-1] + dis(gen);
        y[i][0] = constant0;
        y[i][1] = constant1;
    }

    std::vector<Eigen::Vector2d> y_copy = y;
    std::vector<Real> t_copy = t;
    boost::math::interpolators::vector_barycentric_rational<decltype(t), decltype(y)> interpolator(std::move(t), std::move(y), 5);

    Eigen::Vector2d z;
    for (size_t i = 0; i < t_copy.size(); ++i)
    {
        // Don't evaluate the constant at x[i]; that's already tested in the interpolation condition test.
        Real t = t_copy[i] + dis(gen);
        z = interpolator(t);
        BOOST_CHECK_CLOSE(z[0], constant0, 100*sqrt(numeric_limits<Real>::epsilon()));
        BOOST_CHECK_CLOSE(z[1], constant1, 100*sqrt(numeric_limits<Real>::epsilon()));
        Eigen::Vector2d zprime = interpolator.prime(t);
        Real zero_0 = zprime[0];
        Real zero_1 = zprime[1];
        BOOST_CHECK_SMALL(zero_0, sqrt(numeric_limits<Real>::epsilon()));
        BOOST_CHECK_SMALL(zero_1, sqrt(numeric_limits<Real>::epsilon()));
    }
}


template<class Real>
void test_weights()
{
    std::cout << "Testing weights are calculated correctly using barycentric interpolation on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";

    std::mt19937 gen(9);
    boost::random::uniform_real_distribution<Real> dis(0.005, 0.01);
    std::vector<Real> t(100);
    std::vector<Eigen::Vector2d> y(100);
    t[0] = dis(gen);
    y[0][0] = dis(gen);
    y[0][1] = dis(gen);
    for (size_t i = 1; i < t.size(); ++i)
    {
        t[i] = t[i-1] + dis(gen);
        y[i][0] = dis(gen);
        y[i][1] = dis(gen);
    }

    std::vector<Eigen::Vector2d> y_copy = y;
    std::vector<Real> t_copy = t;
    boost::math::interpolators::detail::vector_barycentric_rational_imp<decltype(t), decltype(y)> interpolator(std::move(t), std::move(y), 1);

    for (size_t i = 1; i < t_copy.size() - 1; ++i)
    {
        Real w = interpolator.weight(i);
        Real w_expect = 1/(t_copy[i] - t_copy[i - 1]) + 1/(t_copy[i+1] - t_copy[i]);
        if (i % 2 == 0)
        {
            BOOST_CHECK_CLOSE(w, -w_expect, 0.00001);
        }
        else
        {
            BOOST_CHECK_CLOSE(w, w_expect, 0.00001);
        }
    }
}


BOOST_AUTO_TEST_CASE(vector_barycentric_rational)
{
    #ifdef __STDCPP_FLOAT64_T__

    test_weights<std::float64_t>();
    test_constant_eigen<std::float64_t>();
    test_constant_std_array<std::float64_t>();
    test_constant_high_order<std::float64_t>();
    test_interpolation_condition_eigen<std::float64_t>();
    test_interpolation_condition_ublas<std::float64_t>();
    test_interpolation_condition_std_array<std::float64_t>();
    test_interpolation_condition_high_order<std::float64_t>();
    test_agreement_with_1d<std::float64_t>();

    #else

    test_weights<double>();
    test_constant_eigen<double>();
    test_constant_std_array<double>();
    test_constant_high_order<double>();
    test_interpolation_condition_eigen<double>();
    test_interpolation_condition_ublas<double>();
    test_interpolation_condition_std_array<double>();
    test_interpolation_condition_high_order<double>();
    test_agreement_with_1d<double>();

    #endif
}
