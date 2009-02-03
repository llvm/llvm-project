/*
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 ******************************************************************************
 *
 * Description:
 *  This header file includes the infamous alloc.h header file if the
 *  autoconf system has found it.  It hides all of the autoconf details
 *  from the rest of the application source code.
 */

#ifndef _CONFIG_ALLOC_H
#define _CONFIG_ALLOC_H

#include "llvm/Config/config.h"

/*
 * This is a modified version of that suggested by the Autoconf manual.
 *  1) The #pragma is indented so that pre-ANSI C compilers ignore it.
 *  2) If alloca.h cannot be found, then try stdlib.h.  Some platforms
 *     (notably FreeBSD) defined alloca() there.
 */
#ifdef _MSC_VER
#include <malloc.h>
#define alloca _alloca
#elif defined(HAVE_ALLOCA_H)
#include <alloca.h>
#elif defined(__MINGW32__) && defined(HAVE_MALLOC_H)
#include <malloc.h>
#elif !defined(__GNUC__)
# ifdef _AIX
#   pragma alloca
# else
#   ifndef alloca
      char * alloca ();
#   endif
# endif
#else
# ifdef HAVE_STDLIB_H
#   include <stdlib.h>
# else
#   error "The function alloca() is required but not found!"
# endif
#endif

#endif
