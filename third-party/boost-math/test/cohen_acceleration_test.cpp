/*
 * Copyright Nick Thompson, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <boost/math/tools/cohen_acceleration.hpp>
#include <boost/math/constants/constants.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <cmath>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using boost::math::tools::cohen_acceleration;
using boost::multiprecision::cpp_bin_float_100;
using boost::math::constants::pi;
using std::log;

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

template<typename Real>
void test_pisq_div12()
{
    auto g = G<Real>();
    Real x = cohen_acceleration(g);
    CHECK_ULP_CLOSE(pi<Real>()*pi<Real>()/12, x, 3);
}

template<typename Real>
class Divergent {
public:
    Divergent(){
        k_ = 0;
    }

    // See C3 of: https://people.mpim-bonn.mpg.de/zagier/files/exp-math-9/fulltext.pdf
    Real operator()() {
        using std::log;
        k_ += 1;
        return log(k_);
    }

private:
    Real k_;
};

template<typename Real>
void test_divergent()
{
    auto g = Divergent<Real>();
    Real x = -cohen_acceleration(g);
    CHECK_ULP_CLOSE(log(pi<Real>()/2)/2, x, (std::numeric_limits<Real>::digits > 100 ? 350 : 150));
}

int main()
{
    #ifdef __STDCPP_FLOAT32_T__
    test_pisq_div12<std::float32_t>();
    test_divergent<std::float32_t>();
    #else
    test_pisq_div12<float>();
    test_divergent<float>();
    #endif
    
    #ifdef __STDCPP_FLOAT64_T__
    test_pisq_div12<std::float64_t>();
    test_divergent<std::float64_t>();
    #else
    test_divergent<double>();
    test_pisq_div12<double>();
    #endif

    test_divergent<long double>();
    test_pisq_div12<long double>();

    #ifdef BOOST_HAS_FLOAT128
    test_pisq_div12<float128>();
    test_divergent<float128>();
    #endif
    
    return boost::math::test::report_errors();
}
