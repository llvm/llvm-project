//  Copyright John Maddock 2014.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_STANDALONE

#define BOOST_MATH_ASSERT_UNDEFINED_POLICY false

#include <boost/cstdfloat.hpp>
#include "poison.hpp"
#include "instantiate.hpp"


int main(int
#ifdef BOOST_FLOAT64_C
   argc
#endif
  , char*[])
{
#ifdef BOOST_FLOAT64_C
   if(argc > 1000)
      instantiate(BOOST_FLOAT64_C(1.23));
#endif
}

#else
int main(void) { return 0; }
#endif
