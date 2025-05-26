//  (C) Copyright Nick Thompson, John Maddock 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <array>
#include <benchmark/benchmark.h>
#include <boost/math/tools/estrin.hpp>
#include <boost/math/tools/rational.hpp>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

using boost::math::tools::evaluate_polynomial_estrin;
using boost::math::tools::evaluate_polynomial;

template <class Real> void HornerRealCoeffsRealArg(benchmark::State &state) {
  long n = state.range(0);
  std::random_device rd;
  auto seed = rd();
  std::mt19937_64 mt(seed);
  std::uniform_real_distribution<Real> unif(-10, 10);

  std::vector<Real> coeffs(n);
  for (auto &c : coeffs) {
    c = unif(mt);
  }
  Real x = unif(mt);
  // Prevent the compiler from hoisting the evaluation out of the loop:
  Real fudge = std::sqrt(std::numeric_limits<Real>::epsilon());
  for (auto _ : state) {
    Real output = evaluate_polynomial(coeffs.data(), x, n);
    benchmark::DoNotOptimize(output);
    x += fudge;
  }
  state.SetComplexityN(state.range(0));
}

template <class Real> void HornerRealCoeffsComplexArg(benchmark::State &state) {
  long n = state.range(0);
  std::random_device rd;
  auto seed = rd();
  std::mt19937_64 mt(seed);
  std::uniform_real_distribution<Real> unif(-10, 10);

  std::vector<Real> coeffs(n);
  for (auto &c : coeffs) {
    c = unif(mt);
  }
  std::complex<Real> x{unif(mt), unif(mt)};
  Real fudge = std::sqrt(std::numeric_limits<Real>::epsilon());
  for (auto _ : state) {
    std::complex<Real> output = evaluate_polynomial(coeffs.data(), x, n);
    benchmark::DoNotOptimize(output);
    x += fudge;
  }
  state.SetComplexityN(state.range(0));
}

template <class Real> void EstrinRealCoeffsRealArg(benchmark::State &state) {
  long n = state.range(0);
  std::random_device rd;
  auto seed = rd();
  std::mt19937_64 mt(seed);
  std::uniform_real_distribution<Real> unif(-10, 10);

  std::vector<Real> c(n);
  for (auto &d : c) {
    d = unif(mt);
  }
  Real x = unif(mt);
  Real fudge = std::sqrt(std::numeric_limits<Real>::epsilon());
  for (auto _ : state) {
    Real output = evaluate_polynomial_estrin(c, x);
    benchmark::DoNotOptimize(output);
    x += fudge;
  }
  state.SetComplexityN(state.range(0));
}

template <class Real> void EstrinRealCoeffsRealArgWithScratch(benchmark::State &state) {
  long n = state.range(0);
  std::random_device rd;
  auto seed = rd();
  std::mt19937_64 mt(seed);
  std::uniform_real_distribution<Real> unif(-10, 10);

  std::vector<Real> c(n);
  std::vector<Real> scratch((n + 1) / 2);
  for (auto &d : c) {
    d = unif(mt);
  }
  Real x = unif(mt);
  Real fudge = std::sqrt(std::numeric_limits<Real>::epsilon());
  for (auto _ : state) {
    Real output = evaluate_polynomial_estrin(c, scratch, x);
    benchmark::DoNotOptimize(output);
    x += fudge;
  }
  state.SetComplexityN(state.range(0));
}

template <class Real> void EstrinRealCoeffsComplexArgWithScratch(benchmark::State &state) {
  long n = state.range(0);
  std::random_device rd;
  auto seed = rd();
  std::mt19937_64 mt(seed);
  std::uniform_real_distribution<Real> unif(-10, 10);

  std::vector<Real> c(n);
  std::vector<std::complex<Real>> scratch((n + 1) / 2);
  for (auto &d : c) {
    d = unif(mt);
  }
  std::complex<Real> z{unif(mt), unif(mt)};
  Real fudge = std::sqrt(std::numeric_limits<Real>::epsilon());
  for (auto _ : state) {
    auto output = evaluate_polynomial_estrin(c, scratch, z);
    benchmark::DoNotOptimize(output);
    z += fudge;
  }
  state.SetComplexityN(state.range(0));
}

template <class Real, size_t n> void EstrinRealCoeffsRealArgStdArray(benchmark::State &state) {
  std::random_device rd;
  auto seed = rd();
  std::mt19937_64 mt(seed);
  std::uniform_real_distribution<Real> unif(-10, 10);

  std::array<Real, n> c;
  for (auto &d : c) {
    d = unif(mt);
  }
  Real x = unif(mt);
  Real fudge = std::sqrt(std::numeric_limits<Real>::epsilon());
  for (auto _ : state) {
    Real output = evaluate_polynomial_estrin(c, x);
    benchmark::DoNotOptimize(output);
    x += fudge;
  }
}

template <class Real, size_t n> void HornerRealCoeffsRealArgStdArray(benchmark::State &state) {
  std::random_device rd;
  auto seed = rd();
  std::mt19937_64 mt(seed);
  std::uniform_real_distribution<Real> unif(-10, 10);

  std::array<Real, n> c;
  for (auto &d : c) {
    d = unif(mt);
  }
  Real x = unif(mt);
  Real fudge = std::sqrt(std::numeric_limits<Real>::epsilon());
  for (auto _ : state) {
    Real output = boost::math::tools::evaluate_polynomial(c, x);
    benchmark::DoNotOptimize(output);
    x += fudge;
  }
}

template <class Real, size_t n> void EstrinRealCoeffsComplexArgStdArray(benchmark::State &state) {
  std::random_device rd;
  auto seed = rd();
  std::mt19937_64 mt(seed);
  std::uniform_real_distribution<Real> unif(-10, 10);

  std::array<Real, n> c;
  for (auto &d : c) {
    d = unif(mt);
  }
  std::complex<Real> z{unif(mt), unif(mt)};
  Real fudge = std::sqrt(std::numeric_limits<Real>::epsilon());
  for (auto _ : state) {
    auto output = evaluate_polynomial_estrin(c, z);
    benchmark::DoNotOptimize(output);
    z += fudge;
  }
}

BENCHMARK_TEMPLATE(HornerRealCoeffsRealArg, float)->RangeMultiplier(2)->Range(1, 1 << 15)->Complexity(benchmark::oN);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgWithScratch, float)->RangeMultiplier(2)->Range(1, 1 << 15)->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 2);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 3);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 4);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 5);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 8);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 9);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 16);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 17);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 32);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 33);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 64);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, float, 65);

BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 2);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 3);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 4);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 5);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 8);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 9);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 16);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 17);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 32);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 33);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 64);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, float, 65);

BENCHMARK_TEMPLATE(HornerRealCoeffsRealArg, double)->RangeMultiplier(2)->Range(1, 1 << 15)->Complexity(benchmark::oN);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgWithScratch, double)->RangeMultiplier(2)->Range(1, 1 << 15)->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 2);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 3);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 4);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 5);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 8);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 9);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 16);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 17);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 32);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 33);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 64);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgStdArray, double, 65);

BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 2);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 3);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 4);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 5);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 8);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 9);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 16);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 17);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 32);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 33);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 64);
BENCHMARK_TEMPLATE(HornerRealCoeffsRealArgStdArray, double, 65);

BENCHMARK_TEMPLATE(HornerRealCoeffsRealArg, double)->DenseRange(64, 128, 8)->Complexity(benchmark::oN);
BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArgWithScratch, double)->DenseRange(64, 128, 8)->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(HornerRealCoeffsComplexArg, float)->RangeMultiplier(2)->Range(1, 1 << 15)->Complexity(benchmark::oN);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgWithScratch, float)->RangeMultiplier(2)->Range(1, 1 << 15)->Complexity(benchmark::oN);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, float, 2);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, float, 4);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, float, 8);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, float, 16);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, float, 32);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, float, 64);

BENCHMARK_TEMPLATE(HornerRealCoeffsComplexArg, double)->RangeMultiplier(2)->Range(1, 1 << 15)->Complexity(benchmark::oN);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgWithScratch, double)->RangeMultiplier(2)->Range(1, 1 << 15)->Complexity(benchmark::oN);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, double, 2);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, double, 4);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, double, 8);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, double, 16);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, double, 32);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgStdArray, double, 64);

BENCHMARK_TEMPLATE(HornerRealCoeffsComplexArg, double)->DenseRange(64, 128, 8)->Complexity(benchmark::oN);
BENCHMARK_TEMPLATE(EstrinRealCoeffsComplexArgWithScratch, double)->DenseRange(64, 128, 8)->Complexity(benchmark::oN);

// These just tell us what we already know: Allocation is expensive!
// BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArg, float)->RangeMultiplier(2)->Range(1, 1<<15)->Complexity(benchmark::oN);
// BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArg, double)->RangeMultiplier(2)->Range(1, 1 << 15)->Complexity(benchmark::oN);
// BENCHMARK_TEMPLATE(EstrinRealCoeffsRealArg, double)->DenseRange(64, 128, 8)->Complexity(benchmark::oN);

BENCHMARK_MAIN();
