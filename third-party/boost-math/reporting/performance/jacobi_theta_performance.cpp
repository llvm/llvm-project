//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <benchmark/benchmark.h>
#include <boost/math/special_functions/jacobi_theta.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::float128;
using boost::multiprecision::cpp_bin_float_50;
using boost::multiprecision::cpp_bin_float_100;
using boost::math::jacobi_theta1;
using boost::math::jacobi_theta1tau;

template<class Real>
void JacobiTheta1(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_real_distribution<long double> unif(0,0.01);

    Real x = static_cast<Real>(unif(mt));
    Real q = static_cast<Real>(unif(mt));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(jacobi_theta1(x, q));
        x += std::numeric_limits<Real>::epsilon();
    }
}

BENCHMARK_TEMPLATE(JacobiTheta1, float);
BENCHMARK_TEMPLATE(JacobiTheta1, double);
BENCHMARK_TEMPLATE(JacobiTheta1, long double);
BENCHMARK_TEMPLATE(JacobiTheta1, float128);
BENCHMARK_TEMPLATE(JacobiTheta1, number<mpfr_float_backend<100>>);
BENCHMARK_TEMPLATE(JacobiTheta1, number<mpfr_float_backend<200>>);
BENCHMARK_TEMPLATE(JacobiTheta1, number<mpfr_float_backend<300>>);
BENCHMARK_TEMPLATE(JacobiTheta1, number<mpfr_float_backend<400>>);
BENCHMARK_TEMPLATE(JacobiTheta1, number<mpfr_float_backend<1000>>);
BENCHMARK_TEMPLATE(JacobiTheta1, cpp_bin_float_50);
BENCHMARK_TEMPLATE(JacobiTheta1, cpp_bin_float_100);

template<class Real>
void JacobiTheta1Tau(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_real_distribution<long double> unif(0,0.01);

    Real x = static_cast<Real>(unif(mt));
    Real q = static_cast<Real>(unif(mt));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(jacobi_theta1tau(x, q));
        x += std::numeric_limits<Real>::epsilon();
    }
}

BENCHMARK_TEMPLATE(JacobiTheta1Tau, float);
BENCHMARK_TEMPLATE(JacobiTheta1Tau, double);
BENCHMARK_TEMPLATE(JacobiTheta1Tau, long double);
BENCHMARK_TEMPLATE(JacobiTheta1Tau, float128);
BENCHMARK_TEMPLATE(JacobiTheta1Tau, number<mpfr_float_backend<100>>);
BENCHMARK_TEMPLATE(JacobiTheta1Tau, number<mpfr_float_backend<200>>);
BENCHMARK_TEMPLATE(JacobiTheta1Tau, number<mpfr_float_backend<300>>);
BENCHMARK_TEMPLATE(JacobiTheta1Tau, number<mpfr_float_backend<400>>);
BENCHMARK_TEMPLATE(JacobiTheta1Tau, number<mpfr_float_backend<1000>>);
BENCHMARK_TEMPLATE(JacobiTheta1Tau, cpp_bin_float_50);
BENCHMARK_TEMPLATE(JacobiTheta1Tau, cpp_bin_float_100);

BENCHMARK_MAIN();
