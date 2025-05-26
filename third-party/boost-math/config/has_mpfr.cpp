//  Copyright John Maddock 2012.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstddef> // See https://gcc.gnu.org/gcc-4.9/porting_to.html
#include <mpfr.h>
#include <boost/config.hpp>

#ifdef __GNUC__
#pragma message "MPFR_VERSION_STRING=" MPFR_VERSION_STRING
#endif

#if (__GNU_MP_VERSION < 4) || ((__GNU_MP_VERSION == 4) && (__GNU_MP_VERSION_MINOR < 2))
#error "Incompatible GMP version"
#endif

#if (MPFR_VERSION < 3)
#error "Incompatible MPFR version"
#endif

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

   mpfr_buildopt_tls_p();

   mpfr_t t;
   mpfr_init2(t, 128);
   if (t[0]._mpfr_d)
      mpfr_clear(t);

   return 0;
}
