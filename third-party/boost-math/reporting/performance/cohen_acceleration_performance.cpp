//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <iostream>
#include <iomanip>
#include <benchmark/benchmark.h>
#include <boost/math/tools/cohen_acceleration.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/core/demangle.hpp>

using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::float128;
using boost::multiprecision::cpp_bin_float_50;
using boost::multiprecision::cpp_bin_float_100;
using boost::math::tools::cohen_acceleration;
using boost::math::constants::pi;

template<typename Real>
class G {
public:
    G(){
        k_ = 0;
    }
    
    Real operator()() {
        k_ += 1;
        return 1/(k_*k_);
    }

private:
    Real k_;
};


template<class Real>
void CohenAcceleration(benchmark::State& state)
{
    using std::abs;
    Real x = pi<Real>()*pi<Real>()/12;
    for (auto _ : state)
    {
        auto g = G<Real>();
        x = cohen_acceleration(g);
        benchmark::DoNotOptimize(x);
    }
    if (abs(x - pi<Real>()*pi<Real>()/12) > 16*std::numeric_limits<Real>::epsilon())
    {
        std::cerr << std::setprecision(std::numeric_limits<Real>::max_digits10);
        std::cerr << "Cohen acceleration computed " << x << " on type " << boost::core::demangle(typeid(Real).name()) << "\n";
        std::cerr << "But expected value is       " << pi<Real>()*pi<Real>()/12 << "\n";
    }
}

BENCHMARK_TEMPLATE(CohenAcceleration, float);
BENCHMARK_TEMPLATE(CohenAcceleration, double);
BENCHMARK_TEMPLATE(CohenAcceleration, long double);
BENCHMARK_TEMPLATE(CohenAcceleration, float128);
BENCHMARK_TEMPLATE(CohenAcceleration, cpp_bin_float_50);
BENCHMARK_TEMPLATE(CohenAcceleration, cpp_bin_float_100);
BENCHMARK_TEMPLATE(CohenAcceleration, number<mpfr_float_backend<100>>);
BENCHMARK_TEMPLATE(CohenAcceleration, number<mpfr_float_backend<200>>);
BENCHMARK_TEMPLATE(CohenAcceleration, number<mpfr_float_backend<300>>);
BENCHMARK_TEMPLATE(CohenAcceleration, number<mpfr_float_backend<400>>);
BENCHMARK_TEMPLATE(CohenAcceleration, number<mpfr_float_backend<1000>>);


template<class Real>
void NaiveSum(benchmark::State& state)
{
    using std::abs;
    Real x = pi<Real>()*pi<Real>()/12;
    for (auto _ : state)
    {
        auto g = G<Real>();
        Real term = g();
        x = term;
        bool even = false;
        while (term > std::numeric_limits<Real>::epsilon()/2) {
            term = g();
            if (even) {
                x += term;
                even = false;
            } else {
                x -= term;
                even = true;
            }
        }
        benchmark::DoNotOptimize(x);
    }
    // The accuracy tests don't pass because the sum is ill-conditioned:
    /*if (abs(x - pi<Real>()*pi<Real>()/12) > 16*std::numeric_limits<Real>::epsilon())
    {
        std::cerr << std::setprecision(std::numeric_limits<Real>::max_digits10);
        std::cerr << "Cohen acceleration computed " << x << " on type " << boost::core::demangle(typeid(Real).name()) << "\n";
        std::cerr << "But expected value is       " << pi<Real>()*pi<Real>()/12 << "\n";
    }*/
}

BENCHMARK_TEMPLATE(NaiveSum, float);
BENCHMARK_TEMPLATE(NaiveSum, double);
BENCHMARK_TEMPLATE(NaiveSum, long double);
BENCHMARK_TEMPLATE(NaiveSum, float128);
BENCHMARK_TEMPLATE(NaiveSum, cpp_bin_float_50);
BENCHMARK_TEMPLATE(NaiveSum, cpp_bin_float_100);
BENCHMARK_TEMPLATE(NaiveSum, number<mpfr_float_backend<100>>);
BENCHMARK_TEMPLATE(NaiveSum, number<mpfr_float_backend<200>>);
BENCHMARK_TEMPLATE(NaiveSum, number<mpfr_float_backend<300>>);
BENCHMARK_TEMPLATE(NaiveSum, number<mpfr_float_backend<400>>);
BENCHMARK_TEMPLATE(NaiveSum, number<mpfr_float_backend<1000>>);

BENCHMARK_MAIN();
