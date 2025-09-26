//  (C) Copyright Nick Thompson 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <array>
#include <vector>
#include <benchmark/benchmark.h>
#include <boost/math/interpolators/bezier_polynomial.hpp>

using boost::math::interpolators::bezier_polynomial;

template<class Real>
void BezierPolynomial(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_real_distribution<Real> unif(0, 10);

    std::vector<std::array<Real, 3>> v(state.range(0));

    for (size_t i = 0; i < v.size(); ++i) {
        v[i][0] = unif(mt);
        v[i][1] = unif(mt);
        v[i][2] = unif(mt);
    }

    auto bp = bezier_polynomial(std::move(v));
    Real t = 0;
    for (auto _ : state)
    {
        auto p = bp(t);
        benchmark::DoNotOptimize(p[0]);
        t += std::numeric_limits<Real>::epsilon();
    }
     state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BezierPolynomial, double)->DenseRange(2, 30)->Complexity();

BENCHMARK_MAIN();
