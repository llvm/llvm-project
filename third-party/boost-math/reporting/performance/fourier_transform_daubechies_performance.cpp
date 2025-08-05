//  (C) Copyright Nick Thompson 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <array>
#include <vector>
#include <iostream>
#include <benchmark/benchmark.h>
#include <boost/math/special_functions/fourier_transform_daubechies.hpp>

using boost::math::fourier_transform_daubechies_scaling;

template<class Real, size_t p>
void FourierTransformDaubechiesScaling(benchmark::State& state)
{
    std::random_device rd;
    auto seed = rd();
    std::mt19937_64 mt(seed);
    std::uniform_real_distribution<Real> unif(0, 10);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(fourier_transform_daubechies_scaling<Real, p>(unif(mt)));
    }
}
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, float, 1);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, float, 2);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, float, 3);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, float, 4);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, float, 5);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, float, 6);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, float, 7);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, float, 8);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, float, 9);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, float, 10);

BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, double, 1);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, double, 2);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, double, 3);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, double, 4);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, double, 5);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, double, 6);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, double, 7);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, double, 8);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, double, 9);
BENCHMARK_TEMPLATE(FourierTransformDaubechiesScaling, double, 10);


BENCHMARK_MAIN();
