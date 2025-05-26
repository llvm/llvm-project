//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>
#include <benchmark/benchmark.h>
#include <boost/math/special_functions/logsumexp.hpp>
#include <boost/math/tools/random_vector.hpp>

using boost::math::logsumexp;
using boost::math::generate_random_vector;

template <typename Real>
void logsumexp_performance(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<Real> test_set = generate_random_vector<Real>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(logsumexp(test_set));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(logsumexp_performance, float)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity()->UseRealTime();
BENCHMARK_TEMPLATE(logsumexp_performance, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity()->UseRealTime();
BENCHMARK_TEMPLATE(logsumexp_performance, long double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity()->UseRealTime();

BENCHMARK_MAIN();
