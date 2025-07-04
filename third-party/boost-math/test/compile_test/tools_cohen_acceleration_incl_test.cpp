//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
//
#include <boost/math/tools/cohen_acceleration.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

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

void compile_and_link_test()
{    
    auto f_g = G<float>();
    check_result<float>(boost::math::tools::cohen_acceleration(f_g));

    auto d_g = G<double>();
    check_result<double>(boost::math::tools::cohen_acceleration(d_g));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    auto ld_g = G<long double>();
    check_result<long double>(boost::math::tools::cohen_acceleration(ld_g));
#endif
}
