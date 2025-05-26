//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define TEST_COMPLEX

#include "compile_test/instantiate.hpp"

void other_test()
{
   instantiate(double(0));
   instantiate_mixed(double(0));
}


