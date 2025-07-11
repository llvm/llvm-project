/*
 *  (C) Copyright Nick Thompson 2018.
 *  (C) Copyright Matt Borland 2020.
 *  Use, modification and distribution are subject to the
 *  Boost Software License, Version 1.0. (See accompanying file
 *  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <vector>
#include <array>
#include <list>
#include <forward_list>
#include <algorithm>
#include <random>
#include <iostream>
#include <boost/core/lightweight_test.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/statistics/univariate_statistics.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_complex.hpp>

// Support compilers with P0024R2 implemented without linking TBB
// https://en.cppreference.com/w/cpp/compiler_support
#ifndef BOOST_NO_CXX17_HDR_EXECUTION
#include <execution>
#endif

using boost::multiprecision::cpp_bin_float_50;
using boost::multiprecision::cpp_complex_50;
using std::abs;

/*
 * Test checklist:
 * 1) Does it work with multiprecision?
 * 2) Does it work with .cbegin()/.cend() if the data is not altered?
 * 3) Does it work with ublas and std::array? (Checking Eigen and Armadillo will make the CI system really unhappy.)
 * 4) Does it work with std::forward_list if a forward iterator is all that is required?
 * 5) Does it work with complex data if complex data is sensible?
 */

 // To stress test, set global_seed = 0, global_size = huge.
 static constexpr size_t global_seed = 0;
 static constexpr size_t global_size = 128;

template<class T>
std::vector<T> generate_random_vector(size_t size, size_t seed)
{
    if (seed == 0)
    {
        std::random_device rd;
        seed = rd();
    }
    std::vector<T> v(size);

    std::mt19937 gen(seed);

    if constexpr (std::is_floating_point<T>::value)
    {
        std::normal_distribution<T> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
         v[i] = dis(gen);
        }
        return v;
    }
    else if constexpr (std::is_integral<T>::value)
    {
        // Rescaling by larger than 2 is UB!
        std::uniform_int_distribution<T> dis(std::numeric_limits<T>::lowest()/2, (std::numeric_limits<T>::max)()/2);
        for (size_t i = 0; i < v.size(); ++i)
        {
         v[i] = dis(gen);
        }
        return v;
    }
    else if constexpr (boost::is_complex<T>::value)
    {
        std::normal_distribution<typename T::value_type> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = {dis(gen), dis(gen)};
        }
        return v;
    }
    else if constexpr (boost::multiprecision::number_category<T>::value == boost::multiprecision::number_kind_complex)
    {
        std::normal_distribution<long double> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = {dis(gen), dis(gen)};
        }
        return v;
    }
    else if constexpr (boost::multiprecision::number_category<T>::value == boost::multiprecision::number_kind_floating_point)
    {
        std::normal_distribution<long double> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = dis(gen);
        }
        return v;
    }
    else
    {
        BOOST_MATH_ASSERT_MSG(false, "Could not identify type for random vector generation.");
        return v;
    }
}

template<class Z, class ExecutionPolicy>
void test_integer_mean(ExecutionPolicy&& exec)
{
    double tol = 100*std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,2,3,4,5};
    double mu = boost::math::statistics::mean(exec, v);
    BOOST_TEST(abs(mu - 3) < tol);

    // Work with std::array?
    std::array<Z, 5> w{1,2,3,4,5};
    mu = boost::math::statistics::mean(exec, w);
    BOOST_TEST(abs(mu - 3) < tol);

    v = generate_random_vector<Z>(global_size, global_seed);
    Z scale = 2;

    double m1 = scale*boost::math::statistics::mean(exec, v);
    for (auto & x : v)
    {
        x *= scale;
    }
    double m2 = boost::math::statistics::mean(exec, v);
    BOOST_TEST(abs(m1 - m2) < tol*abs(m1));
}

template<class RandomAccessContainer>
auto naive_mean(RandomAccessContainer const & v)
{
    typename RandomAccessContainer::value_type sum = 0;
    for (auto & x : v)
    {
        sum += x;
    }
    return sum/v.size();
}

template<class Real, class ExecutionPolicy>
void test_mean(ExecutionPolicy&& exec)
{
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,2,3,4,5};
    Real mu = boost::math::statistics::mean(exec, v.begin(), v.end());
    BOOST_TEST(abs(mu - 3) < tol);

    // Does range call work?
    mu = boost::math::statistics::mean(exec, v);
    BOOST_TEST(abs(mu - 3) < tol);

    // Can we successfully average only part of the vector?
    mu = boost::math::statistics::mean(exec, v.begin(), v.begin() + 3);
    BOOST_TEST(abs(mu - 2) < tol);

    // Does it work when we const qualify?
    mu = boost::math::statistics::mean(exec, v.cbegin(), v.cend());
    BOOST_TEST(abs(mu - 3) < tol);

    // Does it work for std::array?
    std::array<Real, 7> u{1,2,3,4,5,6,7};
    mu = boost::math::statistics::mean(exec, u.begin(), u.end());
    BOOST_TEST(abs(mu - 4) < 10*tol);

    // Does it work for a forward iterator?
    std::forward_list<Real> l{1,2,3,4,5,6,7};
    mu = boost::math::statistics::mean(exec, l.begin(), l.end());
    BOOST_TEST(abs(mu - 4) < tol);

    // Does it work with ublas vectors?
    boost::numeric::ublas::vector<Real> w(7);
    for (size_t i = 0; i < w.size(); ++i)
    {
        w[i] = i+1;
    }
    mu = boost::math::statistics::mean(exec, w.cbegin(), w.cend());
    BOOST_TEST(abs(mu - 4) < tol);

    v = generate_random_vector<Real>(global_size, global_seed);
    Real scale = 2;
    Real m1 = scale*boost::math::statistics::mean(exec, v);
    for (auto & x : v)
    {
        x *= scale;
    }
    Real m2 = boost::math::statistics::mean(exec, v);
    BOOST_TEST(abs(m1 - m2) < tol*abs(m1));

    // Stress test:
    for (size_t i = 1; i < 30; ++i)
    {
        v = generate_random_vector<Real>(i, 12803);
        auto naive_ = naive_mean(v);
        auto higham_ = boost::math::statistics::mean(exec, v);
        if (abs(higham_ - naive_) >= 100*tol*abs(naive_))
        {
            std::cout << std::hexfloat;
            std::cout << "Terms = " << v.size() << "\n";
            std::cout << "higham = " << higham_ << "\n";
            std::cout << "naive_ = " << naive_ << "\n";
        }
        BOOST_TEST(abs(higham_ - naive_) < 100*tol*abs(naive_));
    }
}

template<class Complex, class ExecutionPolicy>
void test_complex_mean(ExecutionPolicy&& exec)
{
    typedef typename Complex::value_type Real;
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Complex> v{{0,1},{0,2},{0,3},{0,4},{0,5}};
    auto mu = boost::math::statistics::mean(exec, v.begin(), v.end());
    BOOST_TEST(abs(mu.imag() - 3) < tol);
    BOOST_TEST(abs(mu.real()) < tol);

    // Does range work?
    mu = boost::math::statistics::mean(exec, v);
    BOOST_TEST(abs(mu.imag() - 3) < tol);
    BOOST_TEST(abs(mu.real()) < tol);
}

template<class Real, class ExecutionPolicy>
void test_variance(ExecutionPolicy&& exec)
{
    Real tol = 10*std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,1,1,1,1,1};
    Real sigma_sq = boost::math::statistics::variance(exec, v.begin(), v.end());
    BOOST_TEST(abs(sigma_sq) < tol);

    sigma_sq = boost::math::statistics::variance(exec, v);
    BOOST_TEST(abs(sigma_sq) < tol);

    Real s_sq = boost::math::statistics::sample_variance(exec, v);
    BOOST_TEST(abs(s_sq) < tol);

    // Fails with assertion
    //std::vector<Real> u{1};
    //sigma_sq = boost::math::statistics::variance(exec, u.cbegin(), u.cend());
    //BOOST_TEST(abs(sigma_sq) < tol);

    std::array<Real, 8> w{0,1,0,1,0,1,0,1};
    sigma_sq = boost::math::statistics::variance(exec, w.begin(), w.end());
    BOOST_TEST(abs(sigma_sq - 1.0/4.0) < tol);

    sigma_sq = boost::math::statistics::variance(exec, w);
    BOOST_TEST(abs(sigma_sq - 1.0/4.0) < tol);

    std::forward_list<Real> l{0,1,0,1,0,1,0,1};
    sigma_sq = boost::math::statistics::variance(exec, l.begin(), l.end());
    BOOST_TEST(abs(sigma_sq - 1.0/4.0) < tol);

    v = generate_random_vector<Real>(global_size, global_seed);
    Real scale = 2;
    Real m1 = scale*scale*boost::math::statistics::variance(exec, v);
    for (auto & x : v)
    {
        x *= scale;
    }
    Real m2 = boost::math::statistics::variance(exec, v);
    BOOST_TEST(abs(m1 - m2) < tol*abs(m1));

    // Wikipedia example for a variance of N sided die:
    // https://en.wikipedia.org/wiki/Variance
    for (size_t j = 16; j < 2048; j *= 2)
    {
        v.resize(1024);
        Real n = v.size();
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = i + 1;
        }

        sigma_sq = boost::math::statistics::variance(exec, v);

        BOOST_TEST(abs(sigma_sq - (n*n-1)/Real(12)) <= tol*sigma_sq);
    }

}

template<class Z, class ExecutionPolicy>
void test_integer_variance(ExecutionPolicy&& exec)
{
    double tol = 10*std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,1,1,1,1,1};
    double sigma_sq = boost::math::statistics::variance(exec, v);
    BOOST_TEST(abs(sigma_sq) < tol);

    std::forward_list<Z> l{0,1,0,1,0,1,0,1};
    sigma_sq = boost::math::statistics::variance(exec, l.begin(), l.end());
    BOOST_TEST(abs(sigma_sq - 1.0/4.0) < tol);

    v = generate_random_vector<Z>(global_size, global_seed);
    Z scale = 2;
    double m1 = scale*scale*boost::math::statistics::variance(exec, v);
    for (auto & x : v)
    {
        x *= scale;
    }
    double m2 = boost::math::statistics::variance(exec, v);
    BOOST_TEST(abs(m1 - m2) < tol*abs(m1));
}

template<class Z, class ExecutionPolicy>
void test_integer_skewness(ExecutionPolicy&& exec)
{
    double tol = 10*std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,1,1};
    double skew = boost::math::statistics::skewness(exec, v);
    BOOST_TEST(abs(skew) < tol);

    // Dataset is symmetric about the mean:
    v = {1,2,3,4,5};
    skew = boost::math::statistics::skewness(exec, v);
    BOOST_TEST(abs(skew) < tol);

    v = {0,0,0,0,5};
    // mu = 1, sigma^2 = 4, sigma = 2, skew = 3/2
    skew = boost::math::statistics::skewness(exec, v);
    BOOST_TEST(abs(skew - 3.0/2.0) < tol);

    std::forward_list<Z> v2{0,0,0,0,5};
    skew = boost::math::statistics::skewness(exec, v);
    BOOST_TEST(abs(skew - 3.0/2.0) < tol);


    v = generate_random_vector<Z>(global_size, global_seed);
    Z scale = 2;
    double m1 = boost::math::statistics::skewness(exec, v);
    for (auto & x : v)
    {
        x *= scale;
    }
    double m2 = boost::math::statistics::skewness(exec, v);
    BOOST_TEST(abs(m1 - m2) < 2*tol*abs(m1));
}

template<class Real, class ExecutionPolicy>
void test_skewness(ExecutionPolicy&& exec)
{
    Real tol = 10*std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,1,1};
    Real skew = boost::math::statistics::skewness(exec, v);
    BOOST_TEST(abs(skew) < tol);

    // Dataset is symmetric about the mean:
    v = {1,2,3,4,5};
    skew = boost::math::statistics::skewness(exec, v);
    BOOST_TEST(abs(skew) < tol);

    v = {0,0,0,0,5};
    // mu = 1, sigma^2 = 4, sigma = 2, skew = 3/2
    skew = boost::math::statistics::skewness(exec, v);
    BOOST_TEST(abs(skew - Real(3)/Real(2)) < tol);

    std::array<Real, 5> w1{0,0,0,0,5};
    skew = boost::math::statistics::skewness(exec, w1);
    BOOST_TEST(abs(skew - Real(3)/Real(2)) < tol);

    std::forward_list<Real> w2{0,0,0,0,5};
    skew = boost::math::statistics::skewness(exec, w2);
    BOOST_TEST(abs(skew - Real(3)/Real(2)) < tol);

    v = generate_random_vector<Real>(global_size, global_seed);
    Real scale = 2;
    Real m1 = boost::math::statistics::skewness(exec, v);
    for (auto & x : v)
    {
        x *= scale;
    }
    Real m2 = boost::math::statistics::skewness(exec, v);
    BOOST_TEST(abs(m1 - m2) < 2*tol*abs(m1));
}

template<class Real, class ExecutionPolicy>
void test_kurtosis(ExecutionPolicy&& exec)
{
    Real tol = 10*std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,1,1};
    Real kurt = boost::math::statistics::kurtosis(exec, v);
    BOOST_TEST(abs(kurt) < tol);

    v = {1,2,3,4,5};
    // mu =1, sigma^2 = 2, kurtosis = 17/10
    kurt = boost::math::statistics::kurtosis(exec, v);
    BOOST_TEST(abs(kurt - Real(17)/Real(10)) < tol);

    v = {0,0,0,0,5};
    // mu = 1, sigma^2 = 4, sigma = 2, skew = 3/2, kurtosis = 13/4
    kurt = boost::math::statistics::kurtosis(exec, v);
    BOOST_TEST(abs(kurt - Real(13)/Real(4)) < tol);

    std::array<Real, 5> v1{0,0,0,0,5};
    kurt = boost::math::statistics::kurtosis(exec, v1);
    BOOST_TEST(abs(kurt - Real(13)/Real(4)) < tol);

    std::forward_list<Real> v2{0,0,0,0,5};
    kurt = boost::math::statistics::kurtosis(exec, v2);
    BOOST_TEST(abs(kurt - Real(13)/Real(4)) < tol);

    std::vector<Real> v3(10000);
    std::mt19937 gen(42);
    std::normal_distribution<long double> dis(0, 1);
    for (size_t i = 0; i < v3.size(); ++i) {
        v3[i] = dis(gen);
    }
    kurt = boost::math::statistics::kurtosis(exec, v3);
    BOOST_TEST(abs(kurt - 3) < 0.1);

    std::uniform_real_distribution<long double> udis(-1, 3);
    for (size_t i = 0; i < v3.size(); ++i) {
        v3[i] = udis(gen);
    }
    auto excess_kurtosis = boost::math::statistics::excess_kurtosis(exec, v3);
    BOOST_TEST(abs(excess_kurtosis + 6.0/5.0) < 0.2);

    v = generate_random_vector<Real>(global_size, global_seed);
    Real scale = 2;
    Real m1 = boost::math::statistics::kurtosis(exec, v);
    for (auto & x : v)
    {
        x *= scale;
    }
    Real m2 = boost::math::statistics::kurtosis(exec, v);
    BOOST_TEST(abs(m1 - m2) < tol*abs(m1));

    // This test only passes when there are a large number of samples.
    // Otherwise, the distribution doesn't generate enough outliers to give,
    // or generates too many, giving pretty wildly different values of kurtosis on different runs.
    // However, by kicking up the samples to 1,000,000, I got very close to 6 for the excess kurtosis on every run.
    // The CI system, however, would die on a million long doubles.
    //v3.resize(1000000);
    //std::exponential_distribution<long double> edis(0.1);
    //for (size_t i = 0; i < v3.size(); ++i) {
    //    v3[i] = edis(gen);
    //}
    //excess_kurtosis = boost::math::statistics::kurtosis(v3) - 3;
    //BOOST_TEST(abs(excess_kurtosis - 6.0) < 0.2);
}

template<class Z, class ExecutionPolicy>
void test_integer_kurtosis(ExecutionPolicy&& exec)
{
    double tol = 10*std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,1,1};
    double kurt = boost::math::statistics::kurtosis(exec, v);
    BOOST_TEST(abs(kurt) < tol);

    v = {1,2,3,4,5};
    // mu =1, sigma^2 = 2, kurtosis = 17/10
    kurt = boost::math::statistics::kurtosis(exec, v);
    BOOST_TEST(abs(kurt - 17.0/10.0) < tol);

    v = {0,0,0,0,5};
    // mu = 1, sigma^2 = 4, sigma = 2, skew = 3/2, kurtosis = 13/4
    kurt = boost::math::statistics::kurtosis(exec, v);
    BOOST_TEST(abs(kurt - 13.0/4.0) < tol);

    v = generate_random_vector<Z>(global_size, global_seed);
    Z scale = 2;
    double m1 = boost::math::statistics::kurtosis(exec, v);
    for (auto & x : v)
    {
        x *= scale;
    }
    double m2 = boost::math::statistics::kurtosis(exec, v);
    BOOST_TEST(abs(m1 - m2) < tol*abs(m1));
}

template<class Real, class ExecutionPolicy>
void test_first_four_moments(ExecutionPolicy&& exec)
{
    Real tol = 10*std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,1,1};
    auto [M1_1, M2_1, M3_1, M4_1] = boost::math::statistics::first_four_moments(exec, v);
    BOOST_TEST(abs(M1_1 - 1) < tol);
    BOOST_TEST(abs(M2_1) < tol);
    BOOST_TEST(abs(M3_1) < tol);
    BOOST_TEST(abs(M4_1) < tol);

    v = {1, 2, 3, 4, 5};
    auto [M1_2, M2_2, M3_2, M4_2] = boost::math::statistics::first_four_moments(exec, v);
    BOOST_TEST(abs(M1_2 - Real(3)) < tol);
    BOOST_TEST(abs(M2_2 - Real(2)) < tol);
    BOOST_TEST(abs(M3_2) < tol);
    BOOST_TEST(abs(M4_2 - Real(34)/Real(5)) < tol);
}

template<class Real, class ExecutionPolicy>
void test_median(ExecutionPolicy&& exec)
{
    std::mt19937 g(12);
    std::vector<Real> v{1,2,3,4,5,6,7};

    Real m = boost::math::statistics::median(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 4);

    std::shuffle(v.begin(), v.end(), g);
    // Does range call work?
    m = boost::math::statistics::median(exec, v);
    BOOST_TEST_EQ(m, 4);

    v = {1,2,3,3,4,5};
    m = boost::math::statistics::median(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 3);
    std::shuffle(v.begin(), v.end(), g);
    m = boost::math::statistics::median(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 3);

    v = {1};
    m = boost::math::statistics::median(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 1);

    v = {1,1};
    m = boost::math::statistics::median(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 1);

    v = {2,4};
    m = boost::math::statistics::median(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 3);

    v = {1,1,1};
    m = boost::math::statistics::median(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 1);

    v = {1,2,3};
    m = boost::math::statistics::median(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 2);
    std::shuffle(v.begin(), v.end(), g);
    m = boost::math::statistics::median(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 2);

    // Does it work with std::array?
    std::array<Real, 3> w{1,2,3};
    m = boost::math::statistics::median(exec, w);
    BOOST_TEST_EQ(m, 2);

    // Does it work with ublas?
    boost::numeric::ublas::vector<Real> w1(3);
    w1[0] = 1;
    w1[1] = 2;
    w1[2] = 3;
    m = boost::math::statistics::median(exec, w);
    BOOST_TEST_EQ(m, 2);
}

template<class Real, class ExecutionPolicy>
void test_median_absolute_deviation(ExecutionPolicy&& exec)
{
    std::vector<Real> v{-1, 2, -3, 4, -5, 6, -7};

    Real m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end(), 0);
    BOOST_TEST_EQ(m, 4);

    std::mt19937 g(12);
    std::shuffle(v.begin(), v.end(), g);
    m = boost::math::statistics::median_absolute_deviation(exec, v, 0);
    BOOST_TEST_EQ(m, 4);

    v = {1, -2, -3, 3, -4, -5};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end(), 0);
    BOOST_TEST_EQ(m, 3);
    std::shuffle(v.begin(), v.end(), g);
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end(), 0);
    BOOST_TEST_EQ(m, 3);

    v = {-1};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end(), 0);
    BOOST_TEST_EQ(m, 1);

    v = {-1, 1};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end(), 0);
    BOOST_TEST_EQ(m, 1);
    m = boost::math::statistics::median_absolute_deviation(exec, v, 0);
    BOOST_TEST_EQ(m, 1);


    v = {2, -4};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end(), 0);
    BOOST_TEST_EQ(m, 3);

    v = {1, -1, 1};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end(), 0);
    BOOST_TEST_EQ(m, 1);

    v = {1, 2, -3};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end(), 0);
    BOOST_TEST_EQ(m, 2);
    std::shuffle(v.begin(), v.end(), g);
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end(), 0);
    BOOST_TEST_EQ(m, 2);

    std::array<Real, 3> w{1, 2, -3};
    m = boost::math::statistics::median_absolute_deviation(exec, w, 0);
    BOOST_TEST_EQ(m, 2);

    // boost.ublas vector?
    boost::numeric::ublas::vector<Real> u(6);
    u[0] = 1;
    u[1] = 2;
    u[2] = -3;
    u[3] = 1;
    u[4] = 2;
    u[5] = -3;
    m = boost::math::statistics::median_absolute_deviation(exec, u, 0);
    BOOST_TEST_EQ(m, 2);


    v = {-1, 2, -3, 4, -5, 6, -7};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 4);

    g = std::mt19937(12);
    std::shuffle(v.begin(), v.end(), g);
    m = boost::math::statistics::median_absolute_deviation(exec, v);
    BOOST_TEST_EQ(m, 4);

    v = {1, -2, -3, 3, -4, -5};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 2);
    std::shuffle(v.begin(), v.end(), g);
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 2);

    v = {-1};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 0);

    v = {-1, 1};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 1);

    m = boost::math::statistics::median_absolute_deviation(exec, v);
    BOOST_TEST_EQ(m, 1);


    v = {2, -4};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 3);

    v = {1, -1, 1};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 0);

    v = {1, 2, -3};
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 1);
    std::shuffle(v.begin(), v.end(), g);
    m = boost::math::statistics::median_absolute_deviation(exec, v.begin(), v.end());
    BOOST_TEST_EQ(m, 1);

    w = {1, 2, -3};
    m = boost::math::statistics::median_absolute_deviation(exec, w);
    BOOST_TEST_EQ(m, 1);

    // boost.ublas vector?
    boost::numeric::ublas::vector<Real> u2(6);
    u2[0] = 1;
    u2[1] = 2;
    u2[2] = -3;
    u2[3] = 1;
    u2[4] = 2;
    u2[5] = -3;
    m = boost::math::statistics::median_absolute_deviation(exec, u2);
    BOOST_TEST_EQ(m, 1);
}


template<class Real, class ExecutionPolicy>
void test_sample_gini_coefficient(ExecutionPolicy&& exec)
{
    Real tol = 10*std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,0,0};
    Real gini = boost::math::statistics::sample_gini_coefficient(exec, v.begin(), v.end());
    BOOST_TEST(abs(gini - 1) < tol);

    gini = boost::math::statistics::sample_gini_coefficient(exec, v);
    BOOST_TEST(abs(gini - 1) < tol);

    v[0] = 1;
    v[1] = 1;
    v[2] = 1;
    gini = boost::math::statistics::sample_gini_coefficient(exec, v.begin(), v.end());
    BOOST_TEST(abs(gini) < tol);

    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
    gini = boost::math::statistics::sample_gini_coefficient(exec, v.begin(), v.end());
    BOOST_TEST(abs(gini) < tol);

    std::array<Real, 3> w{0,0,0};
    gini = boost::math::statistics::sample_gini_coefficient(exec, w);
    BOOST_TEST(abs(gini) < tol);
}


template<class Real, class ExecutionPolicy>
void test_gini_coefficient(ExecutionPolicy&& exec)
{
    Real tol = 10*std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,0,0};
    Real gini = boost::math::statistics::gini_coefficient(exec, v.begin(), v.end());
    Real expected = Real(2)/Real(3);
    BOOST_TEST(abs(gini - expected) < tol);

    gini = boost::math::statistics::gini_coefficient(exec, v);
    BOOST_TEST(abs(gini - expected) < tol);

    v[0] = 1;
    v[1] = 1;
    v[2] = 1;
    gini = boost::math::statistics::gini_coefficient(exec, v.begin(), v.end());
    BOOST_TEST(abs(gini) < tol);

    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
    gini = boost::math::statistics::gini_coefficient(exec, v.begin(), v.end());
    BOOST_TEST(abs(gini) < tol);

    std::array<Real, 3> w{0,0,0};
    gini = boost::math::statistics::gini_coefficient(exec, w);
    BOOST_TEST(abs(gini) < tol);

    boost::numeric::ublas::vector<Real> w1(3);
    w1[0] = 1;
    w1[1] = 1;
    w1[2] = 1;
    gini = boost::math::statistics::gini_coefficient(exec, w1);
    BOOST_TEST(abs(gini) < tol);

    std::mt19937 gen(18);
    // Gini coefficient for a uniform distribution is (b-a)/(3*(b+a));
    std::uniform_real_distribution<long double> dis(0, 3);
    expected = (dis.b() - dis.a())/(3*(dis.b()+ dis.a()));
    v.resize(1024);
    for(size_t i = 0; i < v.size(); ++i)
    {
        v[i] = dis(gen);
    }
    gini = boost::math::statistics::gini_coefficient(exec, v);
    BOOST_TEST(abs(gini - expected) < Real(0.03));
}

template<class Z, class ExecutionPolicy>
void test_integer_gini_coefficient(ExecutionPolicy&& exec)
{
    double tol = std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,0,0};
    double gini = boost::math::statistics::gini_coefficient(exec, v.begin(), v.end());
    double expected = 2.0/3.0;
    BOOST_TEST(abs(gini - expected) < tol);

    gini = boost::math::statistics::gini_coefficient(exec, v);
    BOOST_TEST(abs(gini - expected) < tol);

    v[0] = 1;
    v[1] = 1;
    v[2] = 1;
    gini = boost::math::statistics::gini_coefficient(exec, v.begin(), v.end());
    BOOST_TEST(abs(gini) < tol);

    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
    gini = boost::math::statistics::gini_coefficient(exec, v.begin(), v.end());
    BOOST_TEST(abs(gini) < tol);

    std::array<Z, 3> w{0,0,0};
    gini = boost::math::statistics::gini_coefficient(exec, w);
    BOOST_TEST(abs(gini) < tol);

    boost::numeric::ublas::vector<Z> w1(3);
    w1[0] = 1;
    w1[1] = 1;
    w1[2] = 1;
    gini = boost::math::statistics::gini_coefficient(exec, w1);
    BOOST_TEST(abs(gini) < tol);
}

template<typename Real, typename ExecutionPolicy>
void test_interquartile_range(ExecutionPolicy&& exec)
{
    std::mt19937 gen(486);
    Real iqr;
    // Taken from Wikipedia's example:
    std::vector<Real> v{7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177};

    // Q1 = 31, Q3 = 119, Q3 - Q1 = 88.
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 88);

    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 88);

    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 88);

    std::fill(v.begin(), v.end(), 1);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 0);

    v = {1,2,3};
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 2);
    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 2);

    v = {0, 3, 5};
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 5);
    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 5);

    v = {1,2,3,4};
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 2);
    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 2);

    v = {1,2,3,4,5};
    // Q1 = 1.5, Q3 = 4.5
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 3);
    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 3);

    v = {1,2,3,4,5,6};
    // Q1 = 2, Q3 = 5
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 3);
    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 3);

    v = {1,2,3, 4, 5,6,7};
    // Q1 = 2, Q3 = 6
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 4);
    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 4);

    v = {1,2,3,4,5,6,7,8};
    // Q1 = 2.5, Q3 = 6.5
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 4);
    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 4);

    v = {1,2,3,4,5,6,7,8,9};
    // Q1 = 2.5, Q3 = 7.5
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 5);
    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 5);

    v = {1,2,3,4,5,6,7,8,9,10};
    // Q1 = 3, Q3 = 8
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 5);
    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 5);

    v = {1,2,3,4,5,6,7,8,9,10,11};
    // Q1 = 3, Q3 = 9
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 6);
    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 6);

    v = {1,2,3,4,5,6,7,8,9,10,11,12};
    // Q1 = 3.5, Q3 = 9.5
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 6);
    std::shuffle(v.begin(), v.end(), gen);
    iqr = boost::math::statistics::interquartile_range(exec, v);
    BOOST_TEST_EQ(iqr, 6);
}

template<class Z, class ExecutionPolicy>
void test_integer_mode(ExecutionPolicy&& exec)
{
    std::vector<Z> modes;
    std::vector<Z> v {1, 2, 2, 3, 4, 5};
    const Z ref = 2;

    // Does iterator call work?
    boost::math::statistics::mode(exec, v.begin(), v.end(), std::back_inserter(modes));
    BOOST_TEST_EQ(ref, modes[0]);

    // Does container call work?
    modes.clear();
    boost::math::statistics::mode(exec, v, std::back_inserter(modes));
    BOOST_TEST_EQ(ref, modes[0]);

    // Does it work with part of a vector?
    modes.clear();
    boost::math::statistics::mode(exec, v.begin(), v.begin() + 3, std::back_inserter(modes));
    BOOST_TEST_EQ(ref, modes[0]);

    // Does it work with const qualification? Only if pre-sorted
    modes.clear();
    boost::math::statistics::mode(exec, v.cbegin(), v.cend(), std::back_inserter(modes));
    BOOST_TEST_EQ(ref, modes[0]);

    // Does it work with std::array?
    modes.clear();
    std::array<Z, 6> u {1, 2, 2, 3, 4, 5};
    boost::math::statistics::mode(exec, u, std::back_inserter(modes));
    BOOST_TEST_EQ(ref, modes[0]);

    // Does it work with a bi-modal distribuition?
    modes.clear();
    std::vector<Z> w {1, 2, 2, 3, 3, 4, 5};
    boost::math::statistics::mode(exec, w.begin(), w.end(), std::back_inserter(modes));
    BOOST_TEST_EQ(modes.size(), 2);

    // Does it work with an empty vector?
    modes.clear();
    std::vector<Z> x {};
    boost::math::statistics::mode(exec, x, std::back_inserter(modes));
    BOOST_TEST_EQ(modes.size(), 0);

    // Does it work with a one item vector
    modes.clear();
    x.push_back(2);
    boost::math::statistics::mode(exec, x, std::back_inserter(modes));
    BOOST_TEST_EQ(ref, modes[0]);

    // Does it work with a doubly linked list
    modes.clear();
    std::list<Z> dl {1, 2, 2, 3, 4, 5};
    boost::math::statistics::mode(exec, dl, std::back_inserter(modes));
    BOOST_TEST_EQ(ref, modes[0]);

    // Does it work with a singly linked list
    modes.clear();
    std::forward_list<Z> fl {1, 2, 2, 3, 4, 5};
    boost::math::statistics::mode(exec, fl, std::back_inserter(modes));
    BOOST_TEST_EQ(ref, modes[0]);

    // Does the returning a list work?
    auto return_modes = boost::math::statistics::mode(exec, fl);
    BOOST_TEST_EQ(ref, return_modes.front());

    auto return_modes_2 = boost::math::statistics::mode(exec, fl.begin(), fl.end());
    BOOST_TEST_EQ(ref, return_modes_2.front());
}

template<class Real, class ExecutionPolicy>
void test_mode(ExecutionPolicy&& exec)
{
    std::vector<Real> v {Real(2.0), Real(2.0), Real(2.001), Real(3.2), Real(3.3), Real(2.1)};
    std::vector<Real> modes;

    boost::math::statistics::mode(exec, v, std::back_inserter(modes));
    BOOST_TEST_EQ(Real(2.0), modes[0]);

    // Bi-modal
    modes.clear();
    std::vector<Real> v2 {Real(2.0), Real(2.0), Real(2.0001), Real(2.0001), Real(3.2), Real(1.9999)};
    boost::math::statistics::mode(exec, v2, std::back_inserter(modes));
    BOOST_TEST_EQ(modes.size(), 2);
}

int main()
{
    // Support compilers with P0024R2 implemented without linking TBB
    // https://en.cppreference.com/w/cpp/compiler_support
#ifndef BOOST_NO_CXX17_HDR_EXECUTION

    test_mean<float>(std::execution::seq);
    test_mean<float>(std::execution::par);
    test_mean<double>(std::execution::seq);
    test_mean<double>(std::execution::par);
    test_mean<long double>(std::execution::seq);
    test_mean<long double>(std::execution::par);
    test_mean<cpp_bin_float_50>(std::execution::seq);
    test_mean<cpp_bin_float_50>(std::execution::par);

    test_integer_mean<unsigned>(std::execution::seq);
    test_integer_mean<unsigned>(std::execution::par);
    test_integer_mean<int>(std::execution::seq);
    test_integer_mean<int>(std::execution::par);

    test_complex_mean<std::complex<float>>(std::execution::seq);
    test_complex_mean<std::complex<float>>(std::execution::par);
    test_complex_mean<cpp_complex_50>(std::execution::seq);
    test_complex_mean<cpp_complex_50>(std::execution::par);

    test_variance<float>(std::execution::seq);
    test_variance<float>(std::execution::par);
    test_variance<double>(std::execution::seq);
    test_variance<double>(std::execution::par);
    test_variance<long double>(std::execution::seq);
    test_variance<long double>(std::execution::par);
    test_variance<cpp_bin_float_50>(std::execution::seq);
    test_variance<cpp_bin_float_50>(std::execution::par);

    test_integer_variance<unsigned>(std::execution::seq);
    test_integer_variance<unsigned>(std::execution::par);
    test_integer_variance<int>(std::execution::seq);
    test_integer_variance<int>(std::execution::par);

    test_skewness<float>(std::execution::seq);
    test_skewness<float>(std::execution::par);
    test_skewness<double>(std::execution::seq);
    test_skewness<double>(std::execution::par);
    test_skewness<long double>(std::execution::seq);
    test_skewness<long double>(std::execution::par);
    test_skewness<cpp_bin_float_50>(std::execution::seq);
    test_skewness<cpp_bin_float_50>(std::execution::par);

    test_integer_skewness<int>(std::execution::seq);
    test_integer_skewness<int>(std::execution::par);
    test_integer_skewness<unsigned>(std::execution::seq);
    test_integer_skewness<unsigned>(std::execution::par);

    test_first_four_moments<float>(std::execution::seq);
    test_first_four_moments<float>(std::execution::par);
    test_first_four_moments<double>(std::execution::seq);
    test_first_four_moments<double>(std::execution::par);
    
    test_first_four_moments<long double>(std::execution::seq);
    test_first_four_moments<long double>(std::execution::par);
    test_first_four_moments<cpp_bin_float_50>(std::execution::seq);
    test_first_four_moments<cpp_bin_float_50>(std::execution::par);

    test_kurtosis<float>(std::execution::seq);
    test_kurtosis<float>(std::execution::par);
    test_kurtosis<double>(std::execution::seq);
    test_kurtosis<double>(std::execution::par);
    test_kurtosis<long double>(std::execution::seq);
    test_kurtosis<long double>(std::execution::par);
    // Kinda expensive:
    //test_kurtosis<cpp_bin_float_50>(std::execution::seq);
    //test_kurtosis<cpp_bin_float_50>(std::execution::par);

    test_integer_kurtosis<int>(std::execution::seq);
    test_integer_kurtosis<int>(std::execution::par);
    test_integer_kurtosis<unsigned>(std::execution::seq);
    test_integer_kurtosis<unsigned>(std::execution::par);

    test_median<float>(std::execution::seq);
    test_median<float>(std::execution::par);
    test_median<double>(std::execution::seq);
    test_median<double>(std::execution::par);
    test_median<long double>(std::execution::seq);
    test_median<long double>(std::execution::par);
    test_median<cpp_bin_float_50>(std::execution::seq);
    test_median<cpp_bin_float_50>(std::execution::par);
    test_median<int>(std::execution::seq);
    test_median<int>(std::execution::par);

    test_median_absolute_deviation<float>(std::execution::seq);
    test_median_absolute_deviation<float>(std::execution::par);
    test_median_absolute_deviation<double>(std::execution::seq);
    test_median_absolute_deviation<double>(std::execution::par);
    test_median_absolute_deviation<long double>(std::execution::seq);
    test_median_absolute_deviation<long double>(std::execution::par);
    test_median_absolute_deviation<cpp_bin_float_50>(std::execution::seq);
    test_median_absolute_deviation<cpp_bin_float_50>(std::execution::par);

    test_gini_coefficient<float>(std::execution::seq);
    test_gini_coefficient<float>(std::execution::par);
    test_gini_coefficient<double>(std::execution::seq);
    test_gini_coefficient<double>(std::execution::par);
    test_gini_coefficient<long double>(std::execution::seq);
    test_gini_coefficient<long double>(std::execution::par);
    test_gini_coefficient<cpp_bin_float_50>(std::execution::seq);
    test_gini_coefficient<cpp_bin_float_50>(std::execution::par);

    test_integer_gini_coefficient<unsigned>(std::execution::seq);
    test_integer_gini_coefficient<unsigned>(std::execution::par);
    test_integer_gini_coefficient<int>(std::execution::seq);
    test_integer_gini_coefficient<int>(std::execution::par);

    test_sample_gini_coefficient<float>(std::execution::seq);
    test_sample_gini_coefficient<float>(std::execution::par);
    test_sample_gini_coefficient<double>(std::execution::seq);
    test_sample_gini_coefficient<double>(std::execution::par);

    test_sample_gini_coefficient<long double>(std::execution::seq);
    test_sample_gini_coefficient<long double>(std::execution::par);
    test_sample_gini_coefficient<cpp_bin_float_50>(std::execution::seq);
    test_sample_gini_coefficient<cpp_bin_float_50>(std::execution::par);

    test_interquartile_range<double>(std::execution::seq);
    test_interquartile_range<double>(std::execution::par);
    test_interquartile_range<cpp_bin_float_50>(std::execution::seq);
    test_interquartile_range<cpp_bin_float_50>(std::execution::par);

    test_integer_mode<int>(std::execution::seq);
    test_integer_mode<int>(std::execution::par);
    test_integer_mode<int32_t>(std::execution::seq);
    test_integer_mode<int32_t>(std::execution::par);
    test_integer_mode<int64_t>(std::execution::seq);
    test_integer_mode<int64_t>(std::execution::par);
    test_integer_mode<uint32_t>(std::execution::seq);
    test_integer_mode<uint32_t>(std::execution::par);

    test_mode<float>(std::execution::seq);
    test_mode<float>(std::execution::par);
    test_mode<double>(std::execution::seq);
    test_mode<double>(std::execution::par);
    test_mode<cpp_bin_float_50>(std::execution::seq);
    test_mode<cpp_bin_float_50>(std::execution::par);

    #endif // Compiler guard

    return boost::report_errors();
}
