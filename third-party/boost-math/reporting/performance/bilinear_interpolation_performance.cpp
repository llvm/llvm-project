//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <benchmark/benchmark.h>
#include <boost/math/interpolators/bilinear_uniform.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

using boost::math::interpolators::bilinear_uniform;

template<class Real>
void Bilinear(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_real_distribution<Real> unif(0, 10);

    std::vector<double> v(state.range(0)*state.range(0), std::numeric_limits<Real>::quiet_NaN());

    for (auto& x : v) {
        x = unif(mt);
    }

    auto ub = bilinear_uniform<decltype(v)>(std::move(v), state.range(0), state.range(0));
    Real x = static_cast<Real>(unif(mt));
    Real y = static_cast<Real>(unif(mt));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(ub(x, y));
        x += std::numeric_limits<Real>::epsilon();
        y += std::numeric_limits<Real>::epsilon();
    }
     state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(Bilinear, double)->RangeMultiplier(2)->Range(1 << 6, 1 << 20)->Complexity();

BENCHMARK_MAIN();
