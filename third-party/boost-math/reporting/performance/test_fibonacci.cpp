//  Copyright Madhur Chauhan 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <boost/math/special_functions/fibonacci.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <utility>

using T = boost::multiprecision::mpz_int;

auto fib_rec(unsigned long long n) -> std::pair<T, T> {
    if (n == 0) return {0, 1};
    auto p = fib_rec(n >> 1);
    T c = p.first * (2 * p.second - p.first);
    T d = p.first * p.first + p.second * p.second;
    return (n & 1) ? std::make_pair(d, c + d) : std::make_pair(c, d);
}

static void recursive_slow(benchmark::State &state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(fib_rec(state.range(0)).first);
    state.SetComplexityN(state.range(0));
}
constexpr int bm_start = 1 << 3, bm_end = 1 << 22;
BENCHMARK(recursive_slow)->Range(bm_start, bm_end)->Complexity();

static void iterative_fast(benchmark::State &state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(boost::math::fibonacci<T>(state.range(0)));
    state.SetComplexityN(state.range(0));
}
BENCHMARK(iterative_fast)->Range(bm_start, bm_end)->Complexity();

BENCHMARK_MAIN();

/*
Expected output:

CPU Caches:
  L1 Data 32K (x4)
  L1 Instruction 32K (x4)
  L2 Unified 256K (x4)
  L3 Unified 8192K (x4)
Load Average: 0.96, 0.80, 0.74
-----------------------------------------------------------------
Benchmark                       Time             CPU   Iterations
-----------------------------------------------------------------
recursive_slow/8             3669 ns         3594 ns       190414
recursive_slow/64            6213 ns         6199 ns       116423
recursive_slow/512           8999 ns         8990 ns        78773
recursive_slow/4096         14529 ns        13984 ns        51206
recursive_slow/32768        74183 ns        73039 ns         8539
recursive_slow/262144     1297806 ns      1291304 ns          569
recursive_slow/2097152   23166565 ns     22751898 ns           30
recursive_slow/4194304   47038831 ns     46546938 ns           16
recursive_slow_BigO          0.51 NlgN       0.51 NlgN 
recursive_slow_RMS              5 %             5 %    
iterative_fast/8             1413 ns         1399 ns       493692
iterative_fast/64            2408 ns         2394 ns       298417
iterative_fast/512           4181 ns         4132 ns       170957
iterative_fast/4096          7627 ns         7558 ns        93554
iterative_fast/32768        65080 ns        64791 ns        11289
iterative_fast/262144     1207873 ns      1200725 ns          557
iterative_fast/2097152   19627331 ns     19510132 ns           36
iterative_fast/4194304   42351871 ns     42240620 ns           17
iterative_fast_BigO          0.46 NlgN       0.45 NlgN 
iterative_fast_RMS              5 %             5 %
*/
