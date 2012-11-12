/*===-- include/Support/DataTypes.h - Define fixed size types -----*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file contains definitions to figure out the size of _HOST_ data types.*|
|* This file is important because different host OS's define different macros,*|
|* which makes portability tough.  This file exports the following            *|
|* definitions:                                                               *|
|*                                                                            *|
|*   [u]int(32|64)_t : typedefs for signed and unsigned 32/64 bit system types*|
|*   [U]INT(8|16|32|64)_(MIN|MAX) : Constants for the min and max values.     *|
|*                                                                            *|
|* No library is required when using these functions.                         *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*/

/* Please leave this file C-compatible. */

/* Please keep this file in sync with DataTypes.h.in */

#ifndef SUPPORT_DATATYPES_H
#define SUPPORT_DATATYPES_H

#cmakedefine HAVE_SYS_TYPES_H ${HAVE_SYS_TYPES_H}
#cmakedefine HAVE_INTTYPES_H ${HAVE_INTTYPES_H}
#cmakedefine HAVE_STDINT_H ${HAVE_STDINT_H}
#cmakedefine HAVE_UINT64_T ${HAVE_UINT64_T}
#cmakedefine HAVE_U_INT64_T ${HAVE_U_INT64_T}

#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif

#ifndef _MSC_VER

/* Note that this header's correct operation depends on __STDC_LIMIT_MACROS
   being defined.  We would define it here, but in order to prevent Bad Things
   happening when system headers or C++ STL headers include stdint.h before we
   define it here, we define it on the g++ command line (in Makefile.rules). */
#if !defined(__STDC_LIMIT_MACROS)
# error "Must #define __STDC_LIMIT_MACROS before #including Support/DataTypes.h"
#endif

#if !defined(__STDC_CONSTANT_MACROS)
# error "Must #define __STDC_CONSTANT_MACROS before " \
        "#including Support/DataTypes.h"
#endif

/* Note that <inttypes.h> includes <stdint.h>, if this is a C99 system. */
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#endif

#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif

#ifdef _AIX
#include "llvm/Support/AIXDataTypesFix.h"
#endif

/* Handle incorrect definition of uint64_t as u_int64_t */
#ifndef HAVE_UINT64_T
#ifdef HAVE_U_INT64_T
typedef u_int64_t uint64_t;
#else
# error "Don't have a definition for uint64_t on this platform"
#endif
#endif

#else /* _MSC_VER */
/* Visual C++ doesn't provide standard integer headers, but it does provide
   built-in data types. */
#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif
#include <stdlib.h>
#include <stddef.h>
#include <sys/types.h>
#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed int ssize_t;
#ifndef INT8_MAX
# define INT8_MAX 127
#endif
#ifndef INT8_MIN
# define INT8_MIN -128
#endif
#ifndef UINT8_MAX
# define UINT8_MAX 255
#endif
#ifndef INT16_MAX
# define INT16_MAX 32767
#endif
#ifndef INT16_MIN
# define INT16_MIN -32768
#endif
#ifndef UINT16_MAX
# define UINT16_MAX 65535
#endif
#ifndef INT32_MAX
# define INT32_MAX 2147483647
#endif
#ifndef INT32_MIN
/* MSC treats -2147483648 as -(2147483648U). */
# define INT32_MIN (-INT32_MAX - 1)
#endif
#ifndef UINT32_MAX
# define UINT32_MAX 4294967295U
#endif
/* Certain compatibility updates to VC++ introduce the `cstdint'
 * header, which defines the INT*_C macros. On default installs they
 * are absent. */
#ifndef INT8_C
# define INT8_C(C)   C##i8
#endif
#ifndef UINT8_C
# define UINT8_C(C)  C##ui8
#endif
#ifndef INT16_C
# define INT16_C(C)  C##i16
#endif
#ifndef UINT16_C
# define UINT16_C(C) C##ui16
#endif
#ifndef INT32_C
# define INT32_C(C)  C##i32
#endif
#ifndef UINT32_C
# define UINT32_C(C) C##ui32
#endif
#ifndef INT64_C
# define INT64_C(C)  C##i64
#endif
#ifndef UINT64_C
# define UINT64_C(C) C##ui64
#endif

#ifndef PRId64
# define PRId64 "I64d"
#endif
#ifndef PRIi64
# define PRIi64 "I64i"
#endif
#ifndef PRIo64
# define PRIo64 "I64o"
#endif
#ifndef PRIu64
# define PRIu64 "I64u"
#endif
#ifndef PRIx64
# define PRIx64 "I64x"
#endif
#ifndef PRIX64
# define PRIX64 "I64X"
#endif

#endif /* _MSC_VER */

/* Set defaults for constants which we cannot find. */
#if !defined(INT64_MAX)
# define INT64_MAX 9223372036854775807LL
#endif
#if !defined(INT64_MIN)
# define INT64_MIN ((-INT64_MAX)-1)
#endif
#if !defined(UINT64_MAX)
# define UINT64_MAX 0xffffffffffffffffULL
#endif

#if __GNUC__ > 3
#define END_WITH_NULL __attribute__((sentinel))
#else
#define END_WITH_NULL
#endif

#ifndef HUGE_VALF
#define HUGE_VALF (float)HUGE_VAL
#endif

#endif  /* SUPPORT_DATATYPES_H */
