//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>

using namespace boost::math::constants;
using boost::multiprecision::mpfr_float;

void LaplaceLimit(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(laplace_limit<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(LaplaceLimit)->RangeMultiplier(2)->Range(128, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);

void Dottie(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(dottie<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(Dottie)->RangeMultiplier(2)->Range(512, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);

void ReciprocalFibonacci(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(reciprocal_fibonacci<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(ReciprocalFibonacci)->RangeMultiplier(2)->Range(512, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);


void Pi(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(pi<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(Pi)->RangeMultiplier(2)->Range(512, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);

void Gauss(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(gauss<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(Gauss)->RangeMultiplier(2)->Range(512, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);

void Exp1(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(e<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(Exp1)->RangeMultiplier(2)->Range(512, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);

void Catalan(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(catalan<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(Catalan)->RangeMultiplier(2)->Range(512, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);

void Plastic(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(plastic<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(Plastic)->RangeMultiplier(2)->Range(512, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);

void RootTwo(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(root_two<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(RootTwo)->RangeMultiplier(2)->Range(512, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);

void ZetaThree(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(zeta_three<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(ZetaThree)->RangeMultiplier(2)->Range(512, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);


void Euler(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(euler<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(Euler)->RangeMultiplier(2)->Range(512, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);


void LnTwo(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(ln_two<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(LnTwo)->RangeMultiplier(2)->Range(512, 1<<20)->Complexity()->Unit(benchmark::kMicrosecond);

void Glaisher(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(glaisher<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(Glaisher)->RangeMultiplier(2)->Range(512, 4096)->Complexity()->Unit(benchmark::kMicrosecond);


void Khinchin(benchmark::State& state)
{
    mpfr_float::default_precision(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(khinchin<mpfr_float>());
    }
    state.SetComplexityN(state.range(0));
}

// There is a performance bug in the Khinchin constant:
BENCHMARK(Khinchin)->RangeMultiplier(2)->Range(512, 512)->Complexity()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
