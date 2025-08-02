// Copyright 2020 Matt Borland
//
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/statistics/univariate_statistics.hpp>
#include <boost/math/tools/assert.hpp>
#include <boost/math/tools/complex.hpp>
#include <benchmark/benchmark.h>
#include <vector>
#include <algorithm>
#include <random>
#include <execution>
#include <iostream>
#include <iterator>

template<class T>
std::vector<T> generate_random_vector(std::size_t size, std::size_t seed)
{
    if (seed == 0)
    {
        std::random_device rd;
        seed = rd();
    }
    std::vector<T> v(size);

    std::mt19937 gen(seed);

    if constexpr (std::is_floating_point<T>::value)
    {
        std::normal_distribution<T> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = dis(gen);
        }
        return v;
    }
    else if constexpr (std::is_integral<T>::value)
    {
        // Rescaling by larger than 2 is UB!
        std::uniform_int_distribution<T> dis(std::numeric_limits<T>::lowest()/2, (std::numeric_limits<T>::max)()/2);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = dis(gen);
        }
        return v;
    }
    else if constexpr (boost::math::tools::is_complex_type<T>::value)
    {
        std::normal_distribution<typename T::value_type> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = {dis(gen), dis(gen)};
        }
        return v;
    }
    else if constexpr (boost::multiprecision::number_category<T>::value == boost::multiprecision::number_kind_complex)
    {
        std::normal_distribution<long double> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = {dis(gen), dis(gen)};
        }
        return v;
    }
    else if constexpr (boost::multiprecision::number_category<T>::value == boost::multiprecision::number_kind_floating_point)
    {
        std::normal_distribution<long double> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = dis(gen);
        }
        return v;
    }
    else
    {
        BOOST_MATH_ASSERT_MSG(false, "Could not identify type for random vector generation.");
        return v;
    }
}

template<typename T>
void mean(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::mean(std::execution::seq, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void parallel_mean(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::mean(std::execution::par, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void variance(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::variance(std::execution::seq, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void parallel_variance(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::variance(std::execution::par, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void skewness(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::skewness(std::execution::seq, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void parallel_skewness(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::skewness(std::execution::par, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void first_four_moments(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::first_four_moments(std::execution::seq, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void parallel_first_four_moments(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::first_four_moments(std::execution::par, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void kurtosis(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::kurtosis(std::execution::seq, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void parallel_kurtosis(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::kurtosis(std::execution::par, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void median(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::median(std::execution::seq, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void parallel_median(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::median(std::execution::par, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void median_absolute_deviation(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::median_absolute_deviation(std::execution::seq, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void parallel_median_absolute_deviation(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::median_absolute_deviation(std::execution::par, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void gini_coefficient(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::gini_coefficient(std::execution::seq, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void parallel_gini_coefficient(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::gini_coefficient(std::execution::par, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void interquartile_range(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::interquartile_range(std::execution::seq, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void parallel_interquartile_range(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::interquartile_range(std::execution::par, test_set));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void mode(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);
    std::vector<T> modes;

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::mode(std::execution::seq, test_set, std::back_inserter(modes)));
    }
    state.SetComplexityN(state.range(0));
}

template<typename T>
void parallel_mode(benchmark::State& state)
{
    constexpr std::size_t seed {};
    const std::size_t size = state.range(0);
    std::vector<T> test_set = generate_random_vector<T>(size, seed);
    std::vector<T> modes;

    for(auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::statistics::mode(std::execution::par, test_set, std::back_inserter(modes)));
    }
    state.SetComplexityN(state.range(0));
}

// Mean
BENCHMARK_TEMPLATE(mean, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_mean, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(mean, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_mean, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

// Variance
BENCHMARK_TEMPLATE(variance, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_variance, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(variance, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_variance, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

// Skewness
BENCHMARK_TEMPLATE(skewness, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_skewness, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(skewness, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_skewness, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

// First four moments
BENCHMARK_TEMPLATE(first_four_moments, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_first_four_moments, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(first_four_moments, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_first_four_moments, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

// Kurtosis
BENCHMARK_TEMPLATE(kurtosis, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_kurtosis, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(kurtosis, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_kurtosis, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

// Median
BENCHMARK_TEMPLATE(median, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_median, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(median, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_median, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

// Median absolute deviation
BENCHMARK_TEMPLATE(median_absolute_deviation, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_median_absolute_deviation, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(median_absolute_deviation, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_median_absolute_deviation, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

// Gini Coefficient
BENCHMARK_TEMPLATE(gini_coefficient, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_gini_coefficient, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(gini_coefficient, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_gini_coefficient, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

// Interquartile Range - Only floating point values implemented
BENCHMARK_TEMPLATE(interquartile_range, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_interquartile_range, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

// Mode
BENCHMARK_TEMPLATE(mode, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_mode, int)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(mode, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();
BENCHMARK_TEMPLATE(parallel_mode, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity(benchmark::oN)->UseRealTime();

BENCHMARK_MAIN();
