//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>
#include <boost/math/tools/random_vector.hpp>
#include <boost/math/statistics/bivariate_statistics.hpp>
#include <benchmark/benchmark.h>

using boost::math::generate_random_vector;

template<typename T>
void seq_covariance(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> u = generate_random_vector<T>(size, seed);
    std::vector<T> v = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::covariance(std::execution::seq, u, v));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void par_covariance(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> u = generate_random_vector<T>(size, seed);
    std::vector<T> v = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::covariance(std::execution::par, u, v));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void seq_correlation(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> u = generate_random_vector<T>(size, seed);
    std::vector<T> v = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::correlation_coefficient(std::execution::seq, u, v));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void par_correlation(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> u = generate_random_vector<T>(size, seed);
    std::vector<T> v = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::correlation_coefficient(std::execution::par, u, v));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(seq_covariance, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(par_covariance, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

BENCHMARK_TEMPLATE(seq_correlation, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(par_correlation, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

BENCHMARK_MAIN();
