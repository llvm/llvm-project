//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <random>
#include <limits>
#include <boost/math/tools/color_maps.hpp>
#include <benchmark/benchmark.h>

template <typename Real>
void viridis_bm(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<Real> dist(0, 0.125);
    Real x = dist(gen);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::tools::viridis(x));
        x += std::numeric_limits<Real>::epsilon();
    }
}

BENCHMARK_TEMPLATE(viridis_bm, float);
BENCHMARK_TEMPLATE(viridis_bm, double);


BENCHMARK_MAIN();
