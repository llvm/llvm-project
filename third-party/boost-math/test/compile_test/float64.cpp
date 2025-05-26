//  Copyright Matt Borland 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

#ifdef __STDCPP_FLOAT64_T__

#define TEST_COMPLEX

#include "instantiate.hpp"

int main(int argc, char* [])
{
   if(argc > 10000)
   {
      instantiate(0.0F64);
      instantiate_mixed(0.0F64);
   }
}

#else

int main(int, char*[])
{
    return 0;
}

#endif
