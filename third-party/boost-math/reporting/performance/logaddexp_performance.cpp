//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <limits>
#include <benchmark/benchmark.h>
#include <boost/math/special_functions/logaddexp.hpp>

using boost::math::logaddexp;

template <typename Real>
void logaddexp_performance(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937_64 mt {rd()};
    std::uniform_real_distribution<long double> dist(1e-50, 5e-50);

    Real x = static_cast<Real>(dist(mt));
    Real y = static_cast<Real>(dist(mt));

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(logaddexp(x, y));
        x += std::numeric_limits<Real>::epsilon();
        y += std::numeric_limits<Real>::epsilon();
    }
}

BENCHMARK_TEMPLATE(logaddexp_performance, float);
BENCHMARK_TEMPLATE(logaddexp_performance, double);
BENCHMARK_TEMPLATE(logaddexp_performance, long double);

BENCHMARK_MAIN();
