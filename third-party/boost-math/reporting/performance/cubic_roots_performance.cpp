//  (C) Copyright Nick Thompson 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <array>
#include <vector>
#include <iostream>
#include <benchmark/benchmark.h>
#include <boost/math/tools/cubic_roots.hpp>

using boost::math::tools::cubic_roots;

template<class Real>
void CubicRoots(benchmark::State& state)
{
    std::random_device rd;
    auto seed = rd();
    // This seed generates 3 real roots:
    //uint32_t seed = 416683252;
    std::mt19937_64 mt(seed);
    std::uniform_real_distribution<Real> unif(-10, 10);

    Real a = unif(mt);
    Real b = unif(mt);
    Real c = unif(mt);
    Real d = unif(mt);
    for (auto _ : state)
    {
        auto roots = cubic_roots(a,b,c,d);
        benchmark::DoNotOptimize(roots[0]);
    }
}

BENCHMARK_TEMPLATE(CubicRoots, float);
BENCHMARK_TEMPLATE(CubicRoots, double);
BENCHMARK_TEMPLATE(CubicRoots, long double);

BENCHMARK_MAIN();
