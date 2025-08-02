//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>
#include <random>
#include <benchmark/benchmark.h>
#include <boost/math/special_functions/chebyshev.hpp>


template<class Real>
void ChebyshevClenshaw(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_real_distribution<Real> unif(-1,1);
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = unif(mt);
    }

    using boost::math::chebyshev_clenshaw_recurrence;
    Real x = unif(mt);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(chebyshev_clenshaw_recurrence(v.data(), v.size(), x));
    }
    state.SetComplexityN(state.range(0));
}

template<class Real>
void TranslatedChebyshevClenshaw(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_real_distribution<Real> unif(-1,1);
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = unif(mt);
    }

    using boost::math::detail::unchecked_chebyshev_clenshaw_recurrence;
    Real x = unif(mt);
    Real a = -2;
    Real b = 5;
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(unchecked_chebyshev_clenshaw_recurrence(v.data(), v.size(), a, b, x));
    }
    state.SetComplexityN(state.range(0));
}


BENCHMARK_TEMPLATE(TranslatedChebyshevClenshaw, double)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity(benchmark::oN);
BENCHMARK_TEMPLATE(ChebyshevClenshaw, double)->RangeMultiplier(2)->Range(1<<1, 1<<22)->Complexity(benchmark::oN);



BENCHMARK_MAIN();
