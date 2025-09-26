//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <benchmark/benchmark.h>
#include <boost/math/special_functions/rsqrt.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::float128;
using boost::multiprecision::cpp_bin_float_50;
using boost::multiprecision::cpp_bin_float_100;
using boost::math::rsqrt;

template<class Real>
void Rsqrt(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_real_distribution<long double> unif(1,100);

    Real x = static_cast<Real>(unif(mt));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(rsqrt(x));
        x += std::numeric_limits<Real>::epsilon();
    }
}

BENCHMARK_TEMPLATE(Rsqrt, float);
BENCHMARK_TEMPLATE(Rsqrt, double);
BENCHMARK_TEMPLATE(Rsqrt, long double);
BENCHMARK_TEMPLATE(Rsqrt, float128);
BENCHMARK_TEMPLATE(Rsqrt, number<mpfr_float_backend<100>>);
BENCHMARK_TEMPLATE(Rsqrt, number<mpfr_float_backend<200>>);
BENCHMARK_TEMPLATE(Rsqrt, number<mpfr_float_backend<300>>);
BENCHMARK_TEMPLATE(Rsqrt, number<mpfr_float_backend<400>>);
BENCHMARK_TEMPLATE(Rsqrt, number<mpfr_float_backend<1000>>);
BENCHMARK_TEMPLATE(Rsqrt, cpp_bin_float_50);
BENCHMARK_TEMPLATE(Rsqrt, cpp_bin_float_100);

BENCHMARK_MAIN();
