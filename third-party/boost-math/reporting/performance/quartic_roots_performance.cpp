//  (C) Copyright Nick Thompson 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <array>
#include <vector>
#include <iostream>
#include <benchmark/benchmark.h>
#include <boost/math/tools/quartic_roots.hpp>

using boost::math::tools::quartic_roots;

template<class Real>
void QuarticRoots(benchmark::State& state)
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
    Real e = unif(mt);
    for (auto _ : state)
    {
        auto roots = quartic_roots(a,b,c,d, e);
        benchmark::DoNotOptimize(roots[0]);
    }
}

BENCHMARK_TEMPLATE(QuarticRoots, float);
BENCHMARK_TEMPLATE(QuarticRoots, double);
BENCHMARK_TEMPLATE(QuarticRoots, long double);

BENCHMARK_MAIN();
