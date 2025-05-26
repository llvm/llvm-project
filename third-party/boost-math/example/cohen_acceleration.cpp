//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <iostream>
#include <iomanip>
#include <boost/math/tools/cohen_acceleration.hpp>
#include <boost/math/constants/constants.hpp>

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

int main() {
    using Real = float;
    auto g = G<Real>();
    Real computed = cohen_acceleration(g);
    Real expected = pi<Real>()*pi<Real>()/12;
    std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10);

    std::cout << "Computed = " << computed << " = " << std::hexfloat << computed <<  "\n";
    std::cout << std::defaultfloat;
    std::cout << "Expected = " << expected << " = " << std::hexfloat << expected << "\n";

    // Compute with a specified number of terms:
    // Make sure to reset g:
    g = G<Real>();
    computed = cohen_acceleration(g, 5);
    std::cout << std::defaultfloat;
    std::cout << "Computed = " << computed << " = " << std::hexfloat << computed << " using 5 terms.\n";
}
