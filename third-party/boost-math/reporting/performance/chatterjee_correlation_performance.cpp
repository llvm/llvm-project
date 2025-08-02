//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>
#include <algorithm>
#include <boost/math/tools/random_vector.hpp>
#include <boost/math/statistics/chatterjee_correlation.hpp>
#include <benchmark/benchmark.h>

using boost::math::generate_random_vector;

template <typename T>
void chatterjee_correlation(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> u = generate_random_vector<T>(size, seed);
    std::vector<T> v = generate_random_vector<T>(size, seed);

    std::sort(u.begin(), u.end());

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::chatterjee_correlation(u, v));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(chatterjee_correlation, float)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity()->UseRealTime();
BENCHMARK_TEMPLATE(chatterjee_correlation, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity()->UseRealTime();

BENCHMARK_MAIN();
