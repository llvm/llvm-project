// Copyright 2020, Madhur Chauhan

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is an example to calculate Reciprocal Fibonacci Constant (A079586 in the OEIS)
// compile with flags: -std=c++11 -lmpfr

//[fibonacci_eg

#include <boost/math/special_functions/fibonacci.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <iomanip>
#include <iostream>

int main() {
    using Real = boost::multiprecision::mpfr_float_1000;
    boost::math::fibonacci_generator<Real> gen;
    gen.set(1); // start producing values from 1st fibonacci number
    Real ans = 0;
    const int ITR = 1000;
    for (int i = 0; i < ITR; ++i) {
        ans += 1.0 / gen();
    }
    std::cout << std::setprecision(1000) << "Reciprocal fibonacci constant after "
              << ITR << " iterations is: " << ans << std::endl;
}

//]
