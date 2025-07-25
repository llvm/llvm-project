/*
 *  (C) Copyright Nick Thompson 2018.
 *  Use, modification and distribution are subject to the
 *  Boost Software License, Version 1.0. (See accompanying file
 *  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <vector>
#include <array>
#include <forward_list>
#include <algorithm>
#include <random>
#include <cmath>
#include <cfloat>
#include <boost/core/lightweight_test.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/statistics/univariate_statistics.hpp>
#include <boost/math/statistics/signal_statistics.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_complex.hpp>

using std::abs;
using boost::multiprecision::cpp_bin_float_50;
using boost::multiprecision::cpp_complex_50;
using boost::math::constants::two_pi;

/*
 * Test checklist:
 * 1) Does it work with multiprecision?
 * 2) Does it work with .cbegin()/.cend() if the data is not altered?
 * 3) Does it work with ublas and std::array? (Checking Eigen and Armadillo will make the CI system really unhappy.)
 * 4) Does it work with std::forward_list if a forward iterator is all that is required?
 * 5) Does it work with complex data if complex data is sensible?
 * 6) Does it work with integer data if sensible?
 */

template<class Real>
void test_hoyer_sparsity()
{
    using std::sqrt;
    Real tol = 5*std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,0,0};
    Real hs = boost::math::statistics::hoyer_sparsity(v.begin(), v.end());
    BOOST_TEST(abs(hs - 1) < tol);

    hs = boost::math::statistics::hoyer_sparsity(v);
    BOOST_TEST(abs(hs - 1) < tol);

    // Does it work with constant iterators?
    hs = boost::math::statistics::hoyer_sparsity(v.cbegin(), v.cend());
    BOOST_TEST(abs(hs - 1) < tol);

    v[0] = 1;
    v[1] = 1;
    v[2] = 1;
    hs = boost::math::statistics::hoyer_sparsity(v.cbegin(), v.cend());
    BOOST_TEST(abs(hs) < tol);

    std::array<Real, 3> w{1,1,1};
    hs = boost::math::statistics::hoyer_sparsity(w);
    BOOST_TEST(abs(hs) < tol);

    // Now some statistics:
    // If x_i ~ Unif(0,1), E[x_i] = 1/2, E[x_i^2] = 1/3.
    // Therefore, E[||x||_1] = N/2, E[||x||_2] = sqrt(N/3),
    // and hoyer_sparsity(x) is close to (1-sqrt(3)/2)/(1-1/sqrt(N))
    std::mt19937 gen(82);
    std::uniform_real_distribution<long double> dis(0, 1);
    v.resize(5000);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = dis(gen);
    }
    hs = boost::math::statistics::hoyer_sparsity(v);
    Real expected = (1.0 - boost::math::constants::root_three<Real>()/2)/(1.0 - 1.0/sqrt(v.size()));
    BOOST_TEST(abs(expected - hs) < 0.01);

    // Does it work with a forward list?
    std::forward_list<Real> u1{1, 1, 1};
    hs = boost::math::statistics::hoyer_sparsity(u1);
    BOOST_TEST(abs(hs) < tol);

    // Does it work with a boost ublas vector?
    boost::numeric::ublas::vector<Real> u2(3);
    u2[0] = 1;
    u2[1] = 1;
    u2[2] = 1;
    hs = boost::math::statistics::hoyer_sparsity(u2);
    BOOST_TEST(abs(hs) < tol);

}

template<class Z>
void test_integer_hoyer_sparsity()
{
    using std::sqrt;
    double tol = 5*std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,0,0};
    double hs = boost::math::statistics::hoyer_sparsity(v);
    BOOST_TEST(abs(hs - 1) < tol);

    v[0] = 1;
    v[1] = 1;
    v[2] = 1;
    hs = boost::math::statistics::hoyer_sparsity(v);
    BOOST_TEST(abs(hs) < tol);
}


template<class Complex>
void test_complex_hoyer_sparsity()
{
    typedef typename Complex::value_type Real;
    using std::sqrt;
    Real tol = 5*std::numeric_limits<Real>::epsilon();
    std::vector<Complex> v{{0,1}, {0, 0}, {0,0}};
    Real hs = boost::math::statistics::hoyer_sparsity(v.begin(), v.end());
    BOOST_TEST(abs(hs - 1) < tol);

    hs = boost::math::statistics::hoyer_sparsity(v);
    BOOST_TEST(abs(hs - 1) < tol);

    // Does it work with constant iterators?
    hs = boost::math::statistics::hoyer_sparsity(v.cbegin(), v.cend());
    BOOST_TEST(abs(hs - 1) < tol);

    // All are the same magnitude:
    v[0] = {0, 1};
    v[1] = {1, 0};
    v[2] = {0,-1};
    hs = boost::math::statistics::hoyer_sparsity(v.cbegin(), v.cend());
    BOOST_TEST(abs(hs) < tol);
}


template<class Real>
void test_absolute_gini_coefficient()
{
    using boost::math::statistics::absolute_gini_coefficient;
    using boost::math::statistics::sample_absolute_gini_coefficient;
    using std::abs;
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{-1,0,0};
    Real gini = sample_absolute_gini_coefficient(v.begin(), v.end());
    BOOST_TEST(abs(gini - 1) < tol);

    gini = absolute_gini_coefficient(v);
    BOOST_TEST(abs(gini - Real(2)/Real(3)) < tol);

    v[0] = 1;
    v[1] = -1;
    v[2] = 1;
    gini = absolute_gini_coefficient(v.begin(), v.end());
    BOOST_TEST(abs(gini) < tol);
    gini = sample_absolute_gini_coefficient(v.begin(), v.end());
    BOOST_TEST(abs(gini) < tol);

    std::vector<std::complex<Real>> w(128);
    std::complex<Real> i{0,1};
    for(size_t k = 0; k < w.size(); ++k)
    {
        w[k] = exp(i*static_cast<Real>(k)/static_cast<Real>(w.size()));
    }
    gini = absolute_gini_coefficient(w.begin(), w.end());
    BOOST_TEST(abs(gini) < tol);
    gini = sample_absolute_gini_coefficient(w.begin(), w.end());
    BOOST_TEST(abs(gini) < tol);

    // The population Gini index is invariant under "cloning": If w = v \oplus v, then G(w) = G(v).
    // We use the sample Gini index, so we need to rescale
    std::vector<Real> u(1000);
    std::mt19937 gen(35);
    std::uniform_real_distribution<long double> dis(0, 50);
    for (size_t i = 0; i < u.size()/2; ++i)
    {
        u[i] = dis(gen);
    }
    for (size_t i = 0; i < u.size()/2; ++i)
    {
        u[i + u.size()/2] = u[i];
    }
    Real population_gini1 = absolute_gini_coefficient(u.begin(), u.begin() + u.size()/2);
    Real population_gini2 = absolute_gini_coefficient(u.begin(), u.end());

    BOOST_TEST(abs(population_gini1 - population_gini2) < 10*tol);

    // The Gini coefficient of a uniform distribution is (b-a)/(3*(b+a)), see https://en.wikipedia.org/wiki/Gini_coefficient
    Real expected = (dis.b() - dis.a() )/(3*(dis.a() + dis.b()));

    BOOST_TEST(abs(expected - population_gini1) < 0.01);

    std::exponential_distribution<long double> exp_dis(1);
    for (size_t i = 0; i < u.size(); ++i)
    {
        u[i] = exp_dis(gen);
    }
    population_gini2 = absolute_gini_coefficient(u);

    std::cout << population_gini2 << std::endl;
    BOOST_TEST(abs(population_gini2 - 0.5) < 0.012);
}


template<class Real>
void test_oracle_snr()
{
    using std::abs;
    Real tol = 100*std::numeric_limits<Real>::epsilon();
    size_t length = 100;
    std::vector<Real> signal(length, 1);
    std::vector<Real> noisy_signal = signal;

    noisy_signal[0] += 1;
    Real snr = boost::math::statistics::oracle_snr(signal, noisy_signal);
    Real snr_db = boost::math::statistics::oracle_snr_db(signal, noisy_signal);
    BOOST_TEST(abs(snr - length) < tol);
    BOOST_TEST(abs(snr_db - 10*log10(length)) < tol);
}

template<class Z>
void test_integer_oracle_snr()
{
    double tol = std::numeric_limits<double>::epsilon();
    size_t length = 100;
    std::vector<Z> signal(length, 1);
    std::vector<Z> noisy_signal = signal;

    noisy_signal[0] += 1;
    double snr = boost::math::statistics::oracle_snr(signal, noisy_signal);
    double snr_db = boost::math::statistics::oracle_snr_db(signal, noisy_signal);
    BOOST_TEST(abs(snr - length) < tol);
    BOOST_TEST(abs(snr_db - 10*log10(length)) < tol);
}

template<class Complex>
void test_complex_oracle_snr()
{
    using Real = typename Complex::value_type;
    using std::abs;
    using std::log10;
    Real tol = 100*std::numeric_limits<Real>::epsilon();
    size_t length = 100;
    std::vector<Complex> signal(length, {1,0});
    std::vector<Complex> noisy_signal = signal;

    noisy_signal[0] += Complex(1,0);
    Real snr = boost::math::statistics::oracle_snr(signal, noisy_signal);
    Real snr_db = boost::math::statistics::oracle_snr_db(signal, noisy_signal);
    BOOST_TEST(abs(snr - length) < tol);
    BOOST_TEST(abs(snr_db - 10*log10(length)) < tol);
}

template<class Real>
void test_m2m4_snr_estimator()
{
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> signal(5000, 1);
    std::vector<Real> x(signal.size());
    std::mt19937 gen(18);
    std::normal_distribution<Real> dis{0, 1.0};

    for (size_t i = 0; i < x.size(); ++i)
    {
        signal[i] = 5*sin(100*6.28*i/x.size());
        x[i] = signal[i] + dis(gen);
    }

    // Kurtosis of a sine wave is 1.5:
    auto m2m4_db = boost::math::statistics::m2m4_snr_estimator_db(x, 1.5);
    auto oracle_snr_db = boost::math::statistics::mean_invariant_oracle_snr_db(signal, x);
    BOOST_TEST(abs(m2m4_db - oracle_snr_db) < 0.2);

    std::uniform_real_distribution<Real> uni_dis{-1,1};
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = signal[i] + uni_dis(gen);
    }

    // Kurtosis of continuous uniform distribution over [-1,1] is 1.8:
    m2m4_db = boost::math::statistics::m2m4_snr_estimator_db(x, 1.5, 1.8);
    oracle_snr_db = boost::math::statistics::mean_invariant_oracle_snr_db(signal, x);
    // The performance depends on the exact numbers generated by the distribution, but this isn't bad:
    BOOST_TEST(abs(m2m4_db - oracle_snr_db) < 0.2);

    // The SNR estimator should be scale invariant.
    // If x has snr y, then kx should have snr y.
    Real ka = 1.5;
    Real kw = 1.8;
    auto m2m4 = boost::math::statistics::m2m4_snr_estimator(x.begin(), x.end(), ka, kw);
    for(size_t i = 0; i < x.size(); ++i)
    {
        x[i] *= 4096;
    }
    auto m2m4_2 = boost::math::statistics::m2m4_snr_estimator(x.begin(), x.end(), ka, kw);
    BOOST_TEST(abs(m2m4 - m2m4_2) < tol);
}

int main()
{
    test_absolute_gini_coefficient<float>();
    test_absolute_gini_coefficient<double>();
    test_absolute_gini_coefficient<long double>();

    test_hoyer_sparsity<float>();
    test_hoyer_sparsity<double>();
    test_hoyer_sparsity<long double>();
    test_hoyer_sparsity<cpp_bin_float_50>();

    test_integer_hoyer_sparsity<int>();
    test_integer_hoyer_sparsity<unsigned>();

    test_complex_hoyer_sparsity<std::complex<float>>();
    test_complex_hoyer_sparsity<std::complex<double>>();
    test_complex_hoyer_sparsity<std::complex<long double>>();
    test_complex_hoyer_sparsity<cpp_complex_50>();

    test_oracle_snr<float>();
    test_oracle_snr<double>();
    test_oracle_snr<long double>();
    test_oracle_snr<cpp_bin_float_50>();

    test_integer_oracle_snr<int>();
    test_integer_oracle_snr<unsigned>();

    test_complex_oracle_snr<std::complex<float>>();
    test_complex_oracle_snr<std::complex<double>>();
    test_complex_oracle_snr<std::complex<long double>>();
    test_complex_oracle_snr<cpp_complex_50>();

    test_m2m4_snr_estimator<float>();
    test_m2m4_snr_estimator<double>();
    test_m2m4_snr_estimator<long double>();

    return boost::report_errors();
}
