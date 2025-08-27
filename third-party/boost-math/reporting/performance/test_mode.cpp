//  (C) Copyright Nick Thompson and Matt Borland 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <boost/math/statistics/univariate_statistics.hpp>
#include <benchmark/benchmark.h>

template <class Z>
void test_mode(benchmark::State& state)
{
    using boost::math::statistics::sorted_mode;
    
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<> dist {1, 10};

    auto gen = [&dist, &mt](){return dist(mt);};

    std::vector<Z> v(state.range(0));
    std::generate(v.begin(), v.end(), gen);

    for (auto _ : state)
    {
        std::vector<Z> modes;
        benchmark::DoNotOptimize(sorted_mode(v.begin(), v.end(), std::back_inserter(modes)));
    }

    state.SetComplexityN(state.range(0));
}

template <class Z>
void sequential_test_mode(benchmark::State& state)
{
    using boost::math::statistics::sorted_mode;

    std::vector<Z> v(state.range(0));
    
    size_t current_num {1};
    // produces {1, 2, 3, 4, 5...}
    for(size_t i {}; i < v.size(); ++i)
    {
        v[i] = current_num;
        ++current_num;
    }

    for (auto _ : state)
    {
        std::vector<Z> modes;
        benchmark::DoNotOptimize(sorted_mode(v, std::back_inserter(modes)));
    }

    state.SetComplexityN(state.range(0));
}

template <class Z>
void sequential_pairs_test_mode(benchmark::State& state)
{
    using boost::math::statistics::sorted_mode;

    std::vector<Z> v(state.range(0));
    
    size_t current_num {1};
    size_t current_num_counter {};
    // produces {1, 1, 2, 2, 3, 3, ...}
    for(size_t i {}; i < v.size(); ++i)
    {
        v[i] = current_num;
        ++current_num_counter;
        if(current_num_counter > 2)
        {
            ++current_num;
            current_num_counter = 0;
        }
    }

    for (auto _ : state)
    {
        std::vector<Z> modes;
        benchmark::DoNotOptimize(sorted_mode(v, std::back_inserter(modes)));
    }

    state.SetComplexityN(state.range(0));
}

template <class Z>
void sequential_multiple_test_mode(benchmark::State& state)
{
    using boost::math::statistics::sorted_mode;

    std::vector<Z> v(state.range(0));
    
    size_t current_num {1};
    size_t current_num_counter {};
    // produces {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, ...}
    for(size_t i {}; i < v.size(); ++i)
    {
        v[i] = current_num;
        ++current_num_counter;
        if(current_num_counter > current_num)
        {
            ++current_num;
            current_num_counter = 0;
        }
    }

    for (auto _ : state)
    {
        std::vector<Z> modes;
        benchmark::DoNotOptimize(sorted_mode(v, std::back_inserter(modes)));
    }

    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(test_mode, int32_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();
BENCHMARK_TEMPLATE(test_mode, int64_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();
BENCHMARK_TEMPLATE(test_mode, uint32_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();
BENCHMARK_TEMPLATE(sequential_test_mode, int32_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();
BENCHMARK_TEMPLATE(sequential_test_mode, int64_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();
BENCHMARK_TEMPLATE(sequential_test_mode, uint32_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();
BENCHMARK_TEMPLATE(sequential_pairs_test_mode, int32_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();
BENCHMARK_TEMPLATE(sequential_pairs_test_mode, int64_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();
BENCHMARK_TEMPLATE(sequential_pairs_test_mode, uint32_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();
BENCHMARK_TEMPLATE(sequential_multiple_test_mode, int32_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();
BENCHMARK_TEMPLATE(sequential_multiple_test_mode, int64_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();
BENCHMARK_TEMPLATE(sequential_multiple_test_mode, uint32_t)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity();

BENCHMARK_MAIN();
