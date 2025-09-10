//  Copyright John Maddock 2008.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstddef> // See https://gcc.gnu.org/gcc-4.9/porting_to.html
#include <gmp.h>
#include <boost/config.hpp>

#ifdef __GNUC__
#pragma message "__GNU_MP_VERSION=" BOOST_STRINGIZE(__GNU_MP_VERSION)
#pragma message "__GNU_MP_VERSION_MINOR=" BOOST_STRINGIZE(__GNU_MP_VERSION_MINOR)
#endif

#if (__GNU_MP_VERSION < 4) || ((__GNU_MP_VERSION == 4) && (__GNU_MP_VERSION_MINOR < 2))
#error "Incompatible GMP version"
#endif

int main()
{
   void* (*alloc_func_ptr)(size_t);
   void* (*realloc_func_ptr)(void*, size_t, size_t);
   void (*free_func_ptr)(void*, size_t);

   mp_get_memory_functions(&alloc_func_ptr, &realloc_func_ptr, &free_func_ptr);

   mpz_t integ;
   mpz_init(integ);
   if (integ[0]._mp_d)
      mpz_clear(integ);

   return 0;
}
