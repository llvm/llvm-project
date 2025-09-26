/*
 *  (C) Copyright Nick Thompson 2018.
 *  (C) Copyright Matt Borland 2021.
 *  Use, modification and distribution are subject to the
 *  Boost Software License, Version 1.0. (See accompanying file
 *  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <forward_list>
#include <algorithm>
#include <random>
#include <tuple>
#include <cmath>
#include "math_unit_test.hpp"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/statistics/bivariate_statistics.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_complex.hpp>
#include <boost/math/statistics/univariate_statistics.hpp>

using boost::multiprecision::cpp_bin_float_50;
using boost::multiprecision::cpp_complex_50;

/*
 * Test checklist:
 * 1) Does it work with multiprecision?
 * 2) Does it work with .cbegin()/.cend() if the data is not altered?
 * 3) Does it work with ublas and std::array? (Checking Eigen and Armadillo will make the CI system really unhappy.)
 * 4) Does it work with std::forward_list if a forward iterator is all that is required?
 * 5) Does it work with complex data if complex data is sensible?
 */

using  boost::math::statistics::means_and_covariance;
using  boost::math::statistics::covariance;

#ifndef BOOST_NO_CXX17_HDR_EXECUTION
#include <execution>

template<typename Real, typename ExecutionPolicy>
void test_covariance(ExecutionPolicy&& exec)
{
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10+1);
    Real tol = std::numeric_limits<Real>::epsilon();
    using std::abs;

    // Covariance of a single thing is zero:
    std::array<Real, 1> u1{8};
    std::array<Real, 1> v1{17};
    std::tuple<Real, Real, Real> temp = means_and_covariance(exec, u1, v1);
    Real mu_u1 = std::get<0>(temp);
    Real mu_v1 = std::get<1>(temp);
    Real cov1 = std::get<2>(temp);

    CHECK_LE(abs(cov1), tol);
    CHECK_LE(abs(mu_u1 - 8), tol);
    CHECK_LE(abs(mu_v1 - 17), tol);


    std::array<Real, 2> u2{8, 4};
    std::array<Real, 2> v2{3, 7};
    temp = means_and_covariance(exec, u2, v2);
    Real mu_u2 = std::get<0>(temp);
    Real mu_v2 = std::get<1>(temp);
    Real cov2 = std::get<2>(temp);

    CHECK_LE(abs(cov2+4), tol);
    CHECK_LE(abs(mu_u2 - 6), tol);
    CHECK_LE(abs(mu_v2 - 5), tol);

    std::vector<Real> u3{1,2,3};
    std::vector<Real> v3{1,1,1};

    temp = means_and_covariance(exec, u3, v3);
    Real mu_u3 = std::get<0>(temp);
    Real mu_v3 = std::get<1>(temp);
    Real cov3 = std::get<2>(temp);

    // Since v is constant, covariance(u,v) = 0 against everything any u:
    CHECK_LE(abs(cov3), tol);
    CHECK_LE(abs(mu_u3 - 2), tol);
    CHECK_LE(abs(mu_v3 - 1), tol);
    // Make sure we pull the correct symbol out of means_and_covariance:
    cov3 = covariance(exec, u3, v3);
    CHECK_LE(abs(cov3), tol);

    cov3 = covariance(exec, v3, u3);
    // Covariance is symmetric: cov(u,v) = cov(v,u)
    CHECK_LE(abs(cov3), tol);

    // cov(u,u) = sigma(u)^2:
    cov3 = covariance(exec, u3, u3);
    Real expected = Real(2)/Real(3);

    CHECK_LE(abs(cov3 - expected), tol);

    std::mt19937 gen(15);
    // Can't template standard library on multiprecision, so use double and cast back:
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    std::vector<Real> u(500);
    std::vector<Real> v(500);
    for(size_t i = 0; i < u.size(); ++i)
    {
        u[i] = (Real) dis(gen);
        v[i] = (Real) dis(gen);
    }

    Real mu_u = boost::math::statistics::mean(u);
    Real mu_v = boost::math::statistics::mean(v);
    Real sigma_u_sq = boost::math::statistics::variance(u);
    Real sigma_v_sq = boost::math::statistics::variance(v);

    temp = means_and_covariance(exec, u, v);
    Real mu_u_ = std::get<0>(temp);
    Real mu_v_ = std::get<1>(temp);
    Real cov_uv = std::get<2>(temp);

    CHECK_LE(abs(mu_u - mu_u_), tol);
    CHECK_LE(abs(mu_v - mu_v_), tol);

    // Cauchy-Schwartz inequality:
    CHECK_LE(cov_uv*cov_uv, sigma_u_sq*sigma_v_sq);
    // cov(X, X) = sigma(X)^2:
    Real cov_uu = covariance(exec, u, u);
    CHECK_LE(abs(cov_uu - sigma_u_sq), tol);
    Real cov_vv = covariance(exec, v, v);
    CHECK_LE(abs(cov_vv - sigma_v_sq), tol);
}

template<typename Z, typename ExecutionPolicy>
void test_integer_covariance(ExecutionPolicy&& exec)
{
    std::cout << std::setprecision(std::numeric_limits<double>::digits10+1);
    double tol = std::numeric_limits<double>::epsilon();
    using std::abs;

    // Covariance of a single thing is zero:
    std::array<Z, 1> u1{8};
    std::array<Z, 1> v1{17};
    std::tuple<double, double, double> temp = means_and_covariance(exec, u1, v1);
    double mu_u1 = std::get<0>(temp);
    double mu_v1 = std::get<1>(temp);
    double cov1 = std::get<2>(temp);

    CHECK_LE(abs(cov1), tol);
    CHECK_LE(abs(mu_u1 - 8), tol);
    CHECK_LE(abs(mu_v1 - 17), tol);


    std::array<Z, 2> u2{8, 4};
    std::array<Z, 2> v2{3, 7};
    temp = means_and_covariance(exec, u2, v2);
    double mu_u2 = std::get<0>(temp);
    double mu_v2 = std::get<1>(temp);
    double cov2 = std::get<2>(temp);

    CHECK_LE(abs(cov2+4), tol);
    CHECK_LE(abs(mu_u2 - 6), tol);
    CHECK_LE(abs(mu_v2 - 5), tol);

    std::vector<Z> u3{1,2,3};
    std::vector<Z> v3{1,1,1};

    temp = means_and_covariance(exec, u3, v3);
    double mu_u3 = std::get<0>(temp);
    double mu_v3 = std::get<1>(temp);
    double cov3 = std::get<2>(temp);

    // Since v is constant, covariance(u,v) = 0 against everything any u:
    CHECK_LE(abs(cov3), tol);
    CHECK_LE(abs(mu_u3 - 2), tol);
    CHECK_LE(abs(mu_v3 - 1), tol);
    // Make sure we pull the correct symbol out of means_and_covariance:
    cov3 = covariance(exec, u3, v3);
    CHECK_LE(abs(cov3), tol);

    cov3 = covariance(exec, v3, u3);
    // Covariance is symmetric: cov(u,v) = cov(v,u)
    CHECK_LE(abs(cov3), tol);

    // cov(u,u) = sigma(u)^2:
    cov3 = covariance(exec, u3, u3);
    double expected = double(2)/double(3);

    CHECK_LE(abs(cov3 - expected), tol);

    std::mt19937 gen(15);
    // Can't template standard library on multiprecision, so use double and cast back:
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    std::vector<Z> u(500);
    std::vector<Z> v(500);
    for(size_t i = 0; i < u.size(); ++i)
    {
        u[i] = (Z) dis(gen);
        v[i] = (Z) dis(gen);
    }

    double mu_u = boost::math::statistics::mean(u);
    double mu_v = boost::math::statistics::mean(v);
    double sigma_u_sq = boost::math::statistics::variance(u);
    double sigma_v_sq = boost::math::statistics::variance(v);

    temp = means_and_covariance(exec, u, v);
    double mu_u_ = std::get<0>(temp);
    double mu_v_ = std::get<1>(temp);
    double cov_uv = std::get<2>(temp);

    CHECK_LE(abs(mu_u - mu_u_), tol);
    CHECK_LE(abs(mu_v - mu_v_), tol);

    // Cauchy-Schwartz inequality:
    CHECK_LE(cov_uv*cov_uv, sigma_u_sq*sigma_v_sq);
    // cov(X, X) = sigma(X)^2:
    double cov_uu = covariance(exec, u, u);
    CHECK_LE(abs(cov_uu - sigma_u_sq), tol);
    double cov_vv = covariance(exec, v, v);
    CHECK_LE(abs(cov_vv - sigma_v_sq), tol);
}

template<typename Real, typename ExecutionPolicy>
void test_correlation_coefficient(ExecutionPolicy&& exec)
{
    using boost::math::statistics::correlation_coefficient;
    using std::abs;
    using std::sqrt;

    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> u{1};
    std::vector<Real> v{1};
    Real rho_uv = correlation_coefficient(exec, u, v);
    CHECK_NAN(rho_uv);

    u = {1,1};
    v = {1,1};
    rho_uv = correlation_coefficient(exec, u, v);
    CHECK_NAN(rho_uv);

    u = {1, 2, 3};
    v = {1, 2, 3};
    rho_uv = correlation_coefficient(exec, u, v);
    CHECK_LE(abs(rho_uv - 1), tol);

    u = {1, 2, 3};
    v = {-1, -2, -3};
    rho_uv = correlation_coefficient(exec, u, v);
    CHECK_LE(abs(rho_uv + 1), tol);

    rho_uv = correlation_coefficient(exec, v, u);
    CHECK_LE(abs(rho_uv + 1), tol);

    u = {1, 2, 3};
    v = {0, 0, 0};
    rho_uv = correlation_coefficient(exec, v, u);
    CHECK_NAN(rho_uv);

    u = {1, 2, 3};
    v = {0, 0, 3};
    rho_uv = correlation_coefficient(exec, v, u);
    // mu_u = 2, sigma_u^2 = 2/3, mu_v = 1, sigma_v^2 = 2, cov(u,v) = 1.
    CHECK_LE(abs(rho_uv - sqrt(Real(3))/Real(2)), tol);
}

template<typename Z, typename ExecutionPolicy>
void test_integer_correlation_coefficient(ExecutionPolicy&& exec)
{
    using boost::math::statistics::correlation_coefficient;
    using std::abs;
    using std::sqrt;

    double tol = std::numeric_limits<double>::epsilon();
    std::vector<Z> u{1};
    std::vector<Z> v{1};
    double rho_uv = correlation_coefficient(exec, u, v);
    CHECK_NAN(rho_uv);

    u = {1,1};
    v = {1,1};
    rho_uv = correlation_coefficient(exec, u, v);
    CHECK_NAN(rho_uv);

    u = {1, 2, 3};
    v = {1, 2, 3};
    rho_uv = correlation_coefficient(exec, u, v);
    CHECK_LE(abs(rho_uv - 1.0), tol);

    rho_uv = correlation_coefficient(exec, v, u);
    CHECK_LE(abs(rho_uv - 1.0), tol);

    u = {1, 2, 3};
    v = {0, 0, 0};
    rho_uv = correlation_coefficient(exec, v, u);
    CHECK_NAN(rho_uv);

    u = {1, 2, 3};
    v = {0, 0, 3};
    rho_uv = correlation_coefficient(exec, v, u);
    // mu_u = 2, sigma_u^2 = 2/3, mu_v = 1, sigma_v^2 = 2, cov(u,v) = 1.
    CHECK_LE(abs(rho_uv - sqrt(double(3))/double(2)), tol);
}

int main()
{
    test_covariance<float>(std::execution::seq);
    test_covariance<float>(std::execution::par);
    test_covariance<double>(std::execution::seq);
    test_covariance<double>(std::execution::par);
    test_covariance<long double>(std::execution::seq);
    test_covariance<long double>(std::execution::par);
    test_covariance<cpp_bin_float_50>(std::execution::seq);
    test_covariance<cpp_bin_float_50>(std::execution::par);
    
    test_integer_covariance<int>(std::execution::seq);
    test_integer_covariance<int>(std::execution::par);
    test_integer_covariance<int32_t>(std::execution::seq);
    test_integer_covariance<int32_t>(std::execution::par);
    test_integer_covariance<int64_t>(std::execution::seq);
    test_integer_covariance<int64_t>(std::execution::par);
    test_integer_covariance<uint32_t>(std::execution::seq);
    test_integer_covariance<uint32_t>(std::execution::par);

    test_correlation_coefficient<float>(std::execution::seq);
    test_correlation_coefficient<float>(std::execution::par);
    test_correlation_coefficient<double>(std::execution::seq);
    test_correlation_coefficient<double>(std::execution::par);
    test_correlation_coefficient<long double>(std::execution::seq);
    test_correlation_coefficient<long double>(std::execution::par);
    test_correlation_coefficient<cpp_bin_float_50>(std::execution::seq);
    test_correlation_coefficient<cpp_bin_float_50>(std::execution::par);
    
    test_integer_correlation_coefficient<int>(std::execution::seq);
    test_integer_correlation_coefficient<int>(std::execution::par);
    test_integer_correlation_coefficient<int32_t>(std::execution::seq);
    test_integer_correlation_coefficient<int32_t>(std::execution::par);
    test_integer_correlation_coefficient<int64_t>(std::execution::seq);
    test_integer_correlation_coefficient<int64_t>(std::execution::par);
    test_integer_correlation_coefficient<uint32_t>(std::execution::seq);
    test_integer_correlation_coefficient<uint32_t>(std::execution::par);
    
    return boost::math::test::report_errors();
}

#else

template<typename Real>
void test_covariance()
{
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10+1);
    Real tol = std::numeric_limits<Real>::epsilon();
    using std::abs;

    // Covariance of a single thing is zero:
    std::array<Real, 1> u1{8};
    std::array<Real, 1> v1{17};
    std::tuple<Real, Real, Real> temp = means_and_covariance(u1, v1);
    Real mu_u1 = std::get<0>(temp);
    Real mu_v1 = std::get<1>(temp);
    Real cov1 = std::get<2>(temp);

    CHECK_LE(abs(cov1), tol);
    CHECK_LE(abs(mu_u1 - 8), tol);
    CHECK_LE(abs(mu_v1 - 17), tol);


    std::array<Real, 2> u2{8, 4};
    std::array<Real, 2> v2{3, 7};
    temp = means_and_covariance(u2, v2);
    Real mu_u2 = std::get<0>(temp);
    Real mu_v2 = std::get<1>(temp);
    Real cov2 = std::get<2>(temp);

    CHECK_LE(abs(cov2+4), tol);
    CHECK_LE(abs(mu_u2 - 6), tol);
    CHECK_LE(abs(mu_v2 - 5), tol);

    std::vector<Real> u3{1,2,3};
    std::vector<Real> v3{1,1,1};

    temp = means_and_covariance(u3, v3);
    Real mu_u3 = std::get<0>(temp);
    Real mu_v3 = std::get<1>(temp);
    Real cov3 = std::get<2>(temp);

    // Since v is constant, covariance(u,v) = 0 against everything any u:
    CHECK_LE(abs(cov3), tol);
    CHECK_LE(abs(mu_u3 - 2), tol);
    CHECK_LE(abs(mu_v3 - 1), tol);
    // Make sure we pull the correct symbol out of means_and_covariance:
    cov3 = covariance(u3, v3);
    CHECK_LE(abs(cov3), tol);

    cov3 = covariance(v3, u3);
    // Covariance is symmetric: cov(u,v) = cov(v,u)
    CHECK_LE(abs(cov3), tol);

    // cov(u,u) = sigma(u)^2:
    cov3 = covariance(u3, u3);
    Real expected = Real(2)/Real(3);

    CHECK_LE(abs(cov3 - expected), tol);

    std::mt19937 gen(15);
    // Can't template standard library on multiprecision, so use double and cast back:
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    std::vector<Real> u(500);
    std::vector<Real> v(500);
    for(size_t i = 0; i < u.size(); ++i)
    {
        u[i] = (Real) dis(gen);
        v[i] = (Real) dis(gen);
    }

    Real mu_u = boost::math::statistics::mean(u);
    Real mu_v = boost::math::statistics::mean(v);
    Real sigma_u_sq = boost::math::statistics::variance(u);
    Real sigma_v_sq = boost::math::statistics::variance(v);

    temp = means_and_covariance(u, v);
    Real mu_u_ = std::get<0>(temp);
    Real mu_v_ = std::get<1>(temp);
    Real cov_uv = std::get<2>(temp);

    CHECK_LE(abs(mu_u - mu_u_), tol);
    CHECK_LE(abs(mu_v - mu_v_), tol);

    // Cauchy-Schwartz inequality:
    CHECK_LE(cov_uv*cov_uv, sigma_u_sq*sigma_v_sq);
    // cov(X, X) = sigma(X)^2:
    Real cov_uu = covariance(u, u);
    CHECK_LE(abs(cov_uu - sigma_u_sq), tol);
    Real cov_vv = covariance(v, v);
    CHECK_LE(abs(cov_vv - sigma_v_sq), tol);
}

template<typename Z>
void test_integer_covariance()
{
    std::cout << std::setprecision(std::numeric_limits<double>::digits10+1);
    double tol = std::numeric_limits<double>::epsilon();
    using std::abs;

    // Covariance of a single thing is zero:
    std::array<Z, 1> u1{8};
    std::array<Z, 1> v1{17};
    std::tuple<double, double, double> temp = means_and_covariance(u1, v1);
    double mu_u1 = std::get<0>(temp);
    double mu_v1 = std::get<1>(temp);
    double cov1 = std::get<2>(temp);

    CHECK_LE(abs(cov1), tol);
    CHECK_LE(abs(mu_u1 - 8), tol);
    CHECK_LE(abs(mu_v1 - 17), tol);


    std::array<Z, 2> u2{8, 4};
    std::array<Z, 2> v2{3, 7};
    temp = means_and_covariance(u2, v2);
    double mu_u2 = std::get<0>(temp);
    double mu_v2 = std::get<1>(temp);
    double cov2 = std::get<2>(temp);

    CHECK_LE(abs(cov2+4), tol);
    CHECK_LE(abs(mu_u2 - 6), tol);
    CHECK_LE(abs(mu_v2 - 5), tol);

    std::vector<Z> u3{1,2,3};
    std::vector<Z> v3{1,1,1};

    temp = means_and_covariance(u3, v3);
    double mu_u3 = std::get<0>(temp);
    double mu_v3 = std::get<1>(temp);
    double cov3 = std::get<2>(temp);

    // Since v is constant, covariance(u,v) = 0 against everything any u:
    CHECK_LE(abs(cov3), tol);
    CHECK_LE(abs(mu_u3 - 2), tol);
    CHECK_LE(abs(mu_v3 - 1), tol);
    // Make sure we pull the correct symbol out of means_and_covariance:
    cov3 = covariance(u3, v3);
    CHECK_LE(abs(cov3), tol);

    cov3 = covariance(v3, u3);
    // Covariance is symmetric: cov(u,v) = cov(v,u)
    CHECK_LE(abs(cov3), tol);

    // cov(u,u) = sigma(u)^2:
    cov3 = covariance(u3, u3);
    double expected = double(2)/double(3);

    CHECK_LE(abs(cov3 - expected), tol);

    std::mt19937 gen(15);
    // Can't template standard library on multiprecision, so use double and cast back:
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    std::vector<Z> u(500);
    std::vector<Z> v(500);
    for(size_t i = 0; i < u.size(); ++i)
    {
        u[i] = (Z) dis(gen);
        v[i] = (Z) dis(gen);
    }

    double mu_u = boost::math::statistics::mean(u);
    double mu_v = boost::math::statistics::mean(v);
    double sigma_u_sq = boost::math::statistics::variance(u);
    double sigma_v_sq = boost::math::statistics::variance(v);

    temp = means_and_covariance(u, v);
    double mu_u_ = std::get<0>(temp);
    double mu_v_ = std::get<1>(temp);
    double cov_uv = std::get<2>(temp);

    CHECK_LE(abs(mu_u - mu_u_), tol);
    CHECK_LE(abs(mu_v - mu_v_), tol);

    // Cauchy-Schwartz inequality:
    CHECK_LE(cov_uv*cov_uv, sigma_u_sq*sigma_v_sq);
    // cov(X, X) = sigma(X)^2:
    double cov_uu = covariance(u, u);
    CHECK_LE(abs(cov_uu - sigma_u_sq), tol);
    double cov_vv = covariance(v, v);
    CHECK_LE(abs(cov_vv - sigma_v_sq), tol);
}

template<typename Real>
void test_correlation_coefficient()
{
    using boost::math::statistics::correlation_coefficient;
    using std::abs;
    using std::sqrt;

    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> u{1};
    std::vector<Real> v{1};
    Real rho_uv = correlation_coefficient(u, v);
    CHECK_NAN(rho_uv);

    u = {1,1};
    v = {1,1};
    rho_uv = correlation_coefficient(u, v);
    CHECK_NAN(rho_uv);

    u = {1, 2, 3};
    v = {1, 2, 3};
    rho_uv = correlation_coefficient(u, v);
    CHECK_LE(abs(rho_uv - 1), tol);

    u = {1, 2, 3};
    v = {-1, -2, -3};
    rho_uv = correlation_coefficient(u, v);
    CHECK_LE(abs(rho_uv + 1), tol);

    rho_uv = correlation_coefficient(v, u);
    CHECK_LE(abs(rho_uv + 1), tol);

    u = {1, 2, 3};
    v = {0, 0, 0};
    rho_uv = correlation_coefficient(v, u);
    CHECK_NAN(rho_uv);

    u = {1, 2, 3};
    v = {0, 0, 3};
    rho_uv = correlation_coefficient(v, u);
    // mu_u = 2, sigma_u^2 = 2/3, mu_v = 1, sigma_v^2 = 2, cov(u,v) = 1.
    CHECK_LE(abs(rho_uv - sqrt(Real(3))/Real(2)), tol);
}

template<typename Z>
void test_integer_correlation_coefficient()
{
    using boost::math::statistics::correlation_coefficient;
    using std::abs;
    using std::sqrt;

    double tol = std::numeric_limits<double>::epsilon();
    std::vector<Z> u{1};
    std::vector<Z> v{1};
    double rho_uv = correlation_coefficient(u, v);
    CHECK_NAN(rho_uv);

    u = {1,1};
    v = {1,1};
    rho_uv = correlation_coefficient(u, v);
    CHECK_NAN(rho_uv);

    u = {1, 2, 3};
    v = {1, 2, 3};
    rho_uv = correlation_coefficient(u, v);
    CHECK_LE(abs(rho_uv - 1.0), tol);

    rho_uv = correlation_coefficient(v, u);
    CHECK_LE(abs(rho_uv - 1.0), tol);

    u = {1, 2, 3};
    v = {0, 0, 0};
    rho_uv = correlation_coefficient(v, u);
    CHECK_NAN(rho_uv);

    u = {1, 2, 3};
    v = {0, 0, 3};
    rho_uv = correlation_coefficient(v, u);
    // mu_u = 2, sigma_u^2 = 2/3, mu_v = 1, sigma_v^2 = 2, cov(u,v) = 1.
    CHECK_LE(abs(rho_uv - sqrt(double(3))/double(2)), tol);
}

int main()
{
    test_covariance<float>();
    test_covariance<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_covariance<long double>();
#endif
    test_covariance<cpp_bin_float_50>();

    test_integer_covariance<int>();
    test_integer_covariance<int32_t>();
    test_integer_covariance<int64_t>();
    test_integer_covariance<uint32_t>();

    test_correlation_coefficient<float>();
    test_correlation_coefficient<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_correlation_coefficient<long double>();
#endif
    test_correlation_coefficient<cpp_bin_float_50>();

    test_integer_correlation_coefficient<int>();
    test_integer_correlation_coefficient<int32_t>();
    test_integer_correlation_coefficient<int64_t>();
    test_integer_correlation_coefficient<uint32_t>();

    return boost::math::test::report_errors();
}

#endif
