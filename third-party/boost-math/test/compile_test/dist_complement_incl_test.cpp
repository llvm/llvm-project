//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/distributions/complement.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/distributions/complement.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   boost::math::complement(f, f);
   boost::math::complement(f, f, d);
   boost::math::complement(f, f, d, l);
   boost::math::complement(f, f, d, l, i);
}
