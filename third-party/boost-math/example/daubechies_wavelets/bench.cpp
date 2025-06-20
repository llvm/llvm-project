/*
 * Copyright Nick Thompson, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <cmath>
#include <random>
#include <benchmark/benchmark.h>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/math/special_functions/daubechies_scaling.hpp>
#include <boost/math/quadrature/wavelet_transforms.hpp>
#include <boost/math/interpolators/cubic_hermite.hpp>
#include <boost/math/interpolators/detail/quintic_hermite_detail.hpp>
#include <boost/math/interpolators/detail/septic_hermite_detail.hpp>

double exponential(benchmark::IterationCount j)
{
    return std::pow(2, j);
}


template<typename Real, int p>
void DyadicGrid(benchmark::State & state)
{
    int j = state.range(0);
    size_t s = 0;
    for (auto _ : state)
    {
        auto v = boost::math::daubechies_scaling_dyadic_grid<Real, 4, 0>(j);
        benchmark::DoNotOptimize(v[0]);
        s = v.size();
    }

    state.counters["RAM"] = s*sizeof(Real);
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(DyadicGrid, double, 4)->DenseRange(3, 22, 1)->Unit(benchmark::kMillisecond)->Complexity(exponential);
//BENCHMARK_TEMPLATE(DyadicGrid, double, 8)->DenseRange(3, 22, 1)->Unit(benchmark::kMillisecond)->Complexity(exponential);
//BENCHMARK_TEMPLATE(DyadicGrid, double, 11)->DenseRange(3,22,1)->Unit(benchmark::kMillisecond)->Complexity(exponential);

uint64_t s[2] = { 0x41, 0x29837592 };

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t next(void) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
    s[1] = rotl(s1, 36); // c

    return result;
}

double uniform() {
    return next()*(1.0/18446744073709551616.0);
}

template<typename Real, int p>
void ScalingEvaluation(benchmark::State & state)
{
    auto phi = boost::math::daubechies_scaling<Real, p>();
    Real x = 0;
    Real step = std::numeric_limits<Real>::epsilon();
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(phi(x));
        x += step;
    }
}


BENCHMARK_TEMPLATE(ScalingEvaluation, double, 2);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 3);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 4);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 5);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 6);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 7);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 8);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 9);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 10);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 11);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 12);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 13);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 14);
BENCHMARK_TEMPLATE(ScalingEvaluation, double, 15);

BENCHMARK_TEMPLATE(ScalingEvaluation, float, 2);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 3);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 4);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 5);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 6);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 7);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 8);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 9);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 10);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 11);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 12);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 13);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 14);
BENCHMARK_TEMPLATE(ScalingEvaluation, float, 15);


template<typename Real, int p>
void ScalingConstructor(benchmark::State & state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::daubechies_scaling<Real, p>());
    }
}

BENCHMARK_TEMPLATE(ScalingConstructor, float, 2)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(ScalingConstructor, double, 2)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(ScalingConstructor, long double, 2)->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(ScalingConstructor, float, 3)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(ScalingConstructor, double, 3)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(ScalingConstructor, long double, 3)->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(ScalingConstructor, float, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(ScalingConstructor, double, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(ScalingConstructor, long double, 4)->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(ScalingConstructor, float, 5)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(ScalingConstructor, double, 5)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(ScalingConstructor, long double, 5)->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(ScalingConstructor, float, 11)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(ScalingConstructor, double, 11)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(ScalingConstructor, long double, 11)->Unit(benchmark::kMillisecond);

template<typename Real>
void CubicHermite(benchmark::State & state)
{
    using boost::math::interpolators::cubic_hermite;
    auto n = state.range(0);
    std::vector<Real> x(n);
    std::vector<Real> y(n);
    std::vector<Real> dydx(n);
    std::random_device rd;
    boost::random::uniform_real_distribution<Real> dis(Real(0), Real(1));
    x[0] = dis(rd);
    y[0] = dis(rd);
    dydx[0] = dis(rd);
    for (size_t i = 1; i < y.size(); ++i)
    {
        x[i] = x[i-1] + dis(rd);
        y[i] = dis(rd);
        dydx[i] = dis(rd);
    }
    Real x0 = x.front();
    Real xf = x.back();

    auto qh = cubic_hermite(std::move(x), std::move(y), std::move(dydx));
    Real t = x0;
    Real step = uniform()*(xf-x0)/2048;
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(qh(t));
        t += step;
        if (t >= xf)
        {
            t = x0;
            step = uniform()*(xf-x0)/2048;
        }
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(CubicHermite, double)->RangeMultiplier(2)->Range(1<<8, 1<<20)->Complexity(benchmark::oLogN);

template<typename Real>
void CardinalCubicHermite(benchmark::State & state)
{
    using boost::math::interpolators::detail::cardinal_cubic_hermite_detail;
    auto n = state.range(0);
    std::vector<Real> y(n);
    std::vector<Real> dydx(n);
    std::random_device rd;
    boost::random::uniform_real_distribution<Real> dis(Real(0), Real(1));
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = uniform();
        dydx[i] = uniform();
    }

    Real dx = Real(1)/Real(8);
    Real x0 = 0;
    Real xf = x0 + (y.size()-1)*dx;

    auto qh = cardinal_cubic_hermite_detail(std::move(y), std::move(dydx), x0, dx);
    Real x = x0;
    Real step = uniform()*(xf-x0)/2048;
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(qh.unchecked_evaluation(x));
        x += step;
        if (x >= xf)
        {
            x = x0;
            step = uniform()*(xf-x0)/2048;
        }
    }
    state.SetComplexityN(state.range(0));
}

template<typename Real>
void CardinalCubicHermiteAOS(benchmark::State & state)
{
    auto n = state.range(0);
    std::vector<std::array<Real, 2>> dat(n);
    std::random_device rd;
    boost::random::uniform_real_distribution<Real> dis(Real(0), Real(1));
    for (size_t i = 0; i < dat.size(); ++i)
    {
        dat[i][0] = uniform();
        dat[i][1] = uniform();
    }

    using boost::math::interpolators::detail::cardinal_cubic_hermite_detail_aos;
    Real dx = Real(1)/Real(8);
    Real x0 = 0;
    Real xf = x0 + (dat.size()-1)*dx;
    auto qh = cardinal_cubic_hermite_detail_aos(std::move(dat), x0, dx);
    Real x = 0;
    Real step = uniform()*(xf-x0)/2048;
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(qh.unchecked_evaluation(x));
        x += step;
        if (x >= xf)
        {
            x = x0;
            step = uniform()*(xf-x0)/2048;
        }
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(CardinalCubicHermiteAOS, double)->RangeMultiplier(2)->Range(1<<8, 1<<21)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(CardinalCubicHermite, double)->RangeMultiplier(2)->Range(1<<8, 1<<21)->Complexity(benchmark::o1);

template<class Real>
void SineEvaluation(benchmark::State& state)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<Real> x_dis(0, 3.14159);

    Real x = x_dis(gen);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(std::sin(x));
        x += std::numeric_limits<Real>::epsilon();
    }
}

BENCHMARK_TEMPLATE(SineEvaluation, float);
BENCHMARK_TEMPLATE(SineEvaluation, double);
BENCHMARK_TEMPLATE(SineEvaluation, long double);

template<class Real>
void ExpEvaluation(benchmark::State& state)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<Real> x_dis(0, 3.14159);

    Real x = x_dis(gen);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(std::exp(x));
        x += std::numeric_limits<Real>::epsilon();
    }
}

BENCHMARK_TEMPLATE(ExpEvaluation, float);
BENCHMARK_TEMPLATE(ExpEvaluation, double);
BENCHMARK_TEMPLATE(ExpEvaluation, long double);

template<class Real>
void PowEvaluation(benchmark::State& state)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<Real> x_dis(0, 3.14159);

    Real x = x_dis(gen);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(std::pow(x, x+1));
        x += std::numeric_limits<Real>::epsilon();
    }
}

BENCHMARK_TEMPLATE(PowEvaluation, float);
BENCHMARK_TEMPLATE(PowEvaluation, double);
BENCHMARK_TEMPLATE(PowEvaluation, long double);


template<typename Real>
void CardinalQuinticHermite(benchmark::State & state)
{
    using boost::math::interpolators::detail::cardinal_quintic_hermite_detail;
    auto n = state.range(0);
    std::vector<Real> y(n);
    std::vector<Real> dydx(n);
    std::vector<Real> d2ydx2(n);
    std::random_device rd;
    boost::random::uniform_real_distribution<Real> dis(Real(0), Real(1));
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = uniform();
        dydx[i] = uniform();
        d2ydx2[i] = uniform();
    }

    Real dx = Real(1)/Real(8);
    Real x0 = 0;
    Real xf = x0 + (y.size()-1)*dx;

    auto qh = cardinal_quintic_hermite_detail(std::move(y), std::move(dydx), std::move(d2ydx2), x0, dx);
    Real x = 0;
    Real step = uniform()*(xf-x0)/2048;
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(qh.unchecked_evaluation(x));
        x += step;
        if (x >= xf)
        {
            x = x0;
            step = uniform()*(xf-x0)/2048;
        }
    }
    state.SetComplexityN(state.range(0));
}

template<typename Real>
void CardinalQuinticHermiteAOS(benchmark::State & state)
{
    auto n = state.range(0);
    std::vector<std::array<Real, 3>> dat(n);
    std::random_device rd;
    boost::random::uniform_real_distribution<Real> dis(Real(0), Real(1));
    for (size_t i = 0; i < dat.size(); ++i)
    {
        dat[i][0] = uniform();
        dat[i][1] = uniform();
        dat[i][2] = uniform();
    }

    using boost::math::interpolators::detail::cardinal_quintic_hermite_detail_aos;
    Real dx = Real(1)/Real(8);
    Real x0 = 0;
    Real xf = x0 + (dat.size()-1)*dx;
    auto qh = cardinal_quintic_hermite_detail_aos(std::move(dat), x0, dx);
    Real x = x0;
    Real step = uniform()*(xf-x0)/2048;
    for (auto _ : state) {
        benchmark::DoNotOptimize(qh.unchecked_evaluation(x));
        x += step;
        if (x >= xf)
        {
            x = x0;
            step = uniform()*(xf-x0)/2048;
        }
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(CardinalQuinticHermiteAOS, double)->RangeMultiplier(2)->Range(1<<8, 1<<22)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(CardinalQuinticHermite, double)->RangeMultiplier(2)->Range(1<<8, 1<<22)->Complexity(benchmark::o1);

template<typename Real>
void SepticHermite(benchmark::State & state)
{
    using boost::math::interpolators::detail::septic_hermite_detail;
    auto n = state.range(0);
    std::vector<Real> x(n);
    std::vector<Real> y(n);
    std::vector<Real> dydx(n);
    std::vector<Real> d2ydx2(n);
    std::vector<Real> d3ydx3(n);
    std::random_device rd;
    boost::random::uniform_real_distribution<Real> dis(Real(0), Real(1));
    Real x0 = dis(rd);
    x[0] = x0;
    for (decltype(n) i = 1; i < n; ++i)
    {
        x[i] = x[i-1] + dis(rd);
    }
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = dis(rd);
        dydx[i] = dis(rd);
        d2ydx2[i] = dis(rd);
        d3ydx3[i] = dis(rd);
    }

    Real xf = x.back();

    auto sh = septic_hermite_detail(std::move(x), std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3));
    Real t = x0;
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(sh(t));
        t += xf/128;
        if (t >= xf)
        {
            t = x0;
        }
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(SepticHermite, double)->RangeMultiplier(2)->Range(1<<8, 1<<20)->Complexity();


template<typename Real>
void CardinalSepticHermite(benchmark::State & state)
{
    using boost::math::interpolators::detail::cardinal_septic_hermite_detail;
    auto n = state.range(0);
    std::vector<Real> y(n);
    std::vector<Real> dydx(n);
    std::vector<Real> d2ydx2(n);
    std::vector<Real> d3ydx3(n);
    std::random_device rd;
    boost::random::uniform_real_distribution<Real> dis(Real(0), Real(1));
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = dis(rd);
        dydx[i] = dis(rd);
        d2ydx2[i] = dis(rd);
        d3ydx3[i] = dis(rd);
    }

    Real dx = Real(1)/Real(8);
    Real x0 = 0;
    Real xf = x0 + (y.size()-1)*dx;

    auto sh = cardinal_septic_hermite_detail(std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3), x0, dx);
    Real x = 0;
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(sh.unchecked_evaluation(x));
        x += xf/128;
        if (x >= xf)
        {
            x = x0;
        }
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(CardinalSepticHermite, double)->RangeMultiplier(2)->Range(1<<8, 1<<20)->Complexity();

template<typename Real>
void CardinalSepticHermiteAOS(benchmark::State & state)
{
    using boost::math::interpolators::detail::cardinal_septic_hermite_detail_aos;
    auto n = state.range(0);
    std::vector<std::array<Real, 4>> data(n);
    std::random_device rd;
    boost::random::uniform_real_distribution<Real> dis(Real(0), Real(1));
    for (size_t i = 0; i < data.size(); ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            data[i][j] = dis(rd);
        }
    }

    Real dx = Real(1)/Real(8);
    Real x0 = 0;
    Real xf = x0 + (data.size()-1)*dx;

    auto sh = cardinal_septic_hermite_detail_aos(std::move(data), x0, dx);
    Real x = 0;
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(sh.unchecked_evaluation(x));
        x += xf/128;
        if (x >= xf)
        {
            x = x0;
        }
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(CardinalSepticHermiteAOS, double)->RangeMultiplier(2)->Range(1<<8, 1<<20)->Complexity();


template<typename Real, int p>
void WaveletTransform(benchmark::State & state)
{
    auto psi = boost::math::daubechies_wavelet<Real, p>();
    auto f = [](Real t) {
        return std::exp(-t*t);
    };

    auto Wf = boost::math::quadrature::daubechies_wavelet_transform(f, psi);
    for (auto _ : state)
    {
        Real s = 1 + uniform();
        Real t = uniform();
        benchmark::DoNotOptimize(Wf(s, t));
    }
}

BENCHMARK_TEMPLATE(WaveletTransform, float, 3);
BENCHMARK_TEMPLATE(WaveletTransform, float, 4);
BENCHMARK_TEMPLATE(WaveletTransform, float, 5);
BENCHMARK_TEMPLATE(WaveletTransform, float, 6);
BENCHMARK_TEMPLATE(WaveletTransform, float, 7);
BENCHMARK_TEMPLATE(WaveletTransform, float, 8);
BENCHMARK_TEMPLATE(WaveletTransform, float, 9);
BENCHMARK_TEMPLATE(WaveletTransform, float, 10);
BENCHMARK_TEMPLATE(WaveletTransform, float, 11);
BENCHMARK_TEMPLATE(WaveletTransform, float, 12);
BENCHMARK_TEMPLATE(WaveletTransform, float, 13);
BENCHMARK_TEMPLATE(WaveletTransform, float, 14);
BENCHMARK_TEMPLATE(WaveletTransform, float, 15);
BENCHMARK_TEMPLATE(WaveletTransform, float, 16);
BENCHMARK_TEMPLATE(WaveletTransform, float, 17);
BENCHMARK_TEMPLATE(WaveletTransform, float, 18);
BENCHMARK_TEMPLATE(WaveletTransform, float, 19);

BENCHMARK_TEMPLATE(WaveletTransform, double, 4);
BENCHMARK_TEMPLATE(WaveletTransform, double, 8);
BENCHMARK_TEMPLATE(WaveletTransform, double, 12);
BENCHMARK_TEMPLATE(WaveletTransform, double, 15);
BENCHMARK_TEMPLATE(WaveletTransform, double, 19);

BENCHMARK_MAIN();
