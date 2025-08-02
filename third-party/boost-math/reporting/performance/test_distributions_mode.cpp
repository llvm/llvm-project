//  (C) Copyright Victor Ananyev 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <vector>
#include <boost/math/distributions.hpp>
#include <benchmark/benchmark.h>


template <class Z, template<typename> class dist >
void test_mode_2param(benchmark::State& state)
{
    using boost::math::normal_distribution;

    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::normal_distribution<Z> noise(0., 1E-6);

    for (auto _ : state)
    {
        state.PauseTiming();
        Z p1 = state.range(0) + noise(mt);
        Z p2 = state.range(1) + noise(mt);
        dist<Z> the_dist(p1, p2);
        state.ResumeTiming();
        try {
            benchmark::DoNotOptimize(mode(the_dist));
        }
        catch (boost::wrapexcept<boost::math::evaluation_error>& e) {
            state.SkipWithError(e.what());
            break;
        }
    }
}


static void fixed_ratio_2args(benchmark::internal::Benchmark* b, long double left_div_right, std::vector<int64_t> lefts) {
    for (const long double &left: lefts) {
        b->Args({static_cast<int64_t>(left), static_cast<int64_t>((left/left_div_right))});
    }
}


using boost::math::non_central_chi_squared_distribution;

BENCHMARK_TEMPLATE(test_mode_2param, long double, non_central_chi_squared_distribution)->ArgsProduct({
    {2, 15, 50},
    benchmark::CreateRange(4, 1024, /*multi=*/2)
})->Name("fixed_k");

BENCHMARK_TEMPLATE(test_mode_2param, long double, non_central_chi_squared_distribution)->ArgsProduct({
    benchmark::CreateRange(4, 4096, /*multi=*/2),
    {1, 30, 100, 500}
})->Name("fixed_nc");

BENCHMARK_TEMPLATE(test_mode_2param, long double, non_central_chi_squared_distribution)
    -> Apply([](benchmark::internal::Benchmark*b) {
                fixed_ratio_2args(b, 0.05, benchmark::CreateRange(4, 4096, /*multi=*/2));
    }) -> Name("fixed_scale_0_05");

BENCHMARK_TEMPLATE(test_mode_2param, long double, non_central_chi_squared_distribution)
    -> Apply([](benchmark::internal::Benchmark*b) {
                fixed_ratio_2args(b, 0.15, benchmark::CreateRange(4, 4096, /*multi=*/2));
    }) -> Name("fixed_scale_0_15");

BENCHMARK_TEMPLATE(test_mode_2param, long double, non_central_chi_squared_distribution)
    -> Apply([](benchmark::internal::Benchmark*b) {
                fixed_ratio_2args(b, 0.25, benchmark::CreateRange(4, 4096, /*multi=*/2));
    }) -> Name("fixed_scale_0_25");

BENCHMARK_MAIN();
