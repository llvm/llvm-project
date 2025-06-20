//  Copyright John Maddock 2017.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <fftw3.h>

int main()
{
   fftwq_plan plan;  // early versions don't have this it seems.

   fftw_cleanup();
   fftwf_cleanup();
   fftwl_cleanup();


   return 0;
}

