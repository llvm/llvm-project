//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/math/tools/luroth_expansion.hpp>
#include <boost/math/constants/constants.hpp>

#ifndef BOOST_MATH_STANDALONE
#include <boost/multiprecision/mpfr.hpp>
using boost::multiprecision::mpfr_float;
#endif // BOOST_MATH_STANDALONE

using boost::math::constants::pi;
using boost::math::tools::luroth_expansion;

int main() {
    #ifndef BOOST_MATH_STANDALONE
    using Real = mpfr_float;
    mpfr_float::default_precision(1024);
    #else
    using Real = long double;
    #endif
    
    auto luroth = luroth_expansion(pi<Real>());
    std::cout << luroth << "\n";
}
