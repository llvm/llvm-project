#ifndef __HOSTRPC_H__
#define __HOSTRPC_H__

/*
 *    hostrpc.h:  This header contains the enum for all the
 *                implemented services in hostrpc.  This header is
 *                included by both device stubs and host routines.
 *                It also includes the version, release, and patch
 *                identification for hostrpc.

MIT License

Copyright © 2020 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#if defined(__cplusplus)
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif

#define NOINLINE __attribute__((noinline))

#include <stdint.h>
#include <stdio.h>

//  These are the interfaces for the device stubs */
EXTERN int fprintf(FILE *, const char *, ...);
EXTERN char *fprintf_allocate(uint32_t bufsz);
EXTERN int printf(const char *, ...);
EXTERN char *printf_allocate(uint32_t bufsz);
EXTERN int printf_execute(char *bufptr, uint32_t bufsz);

EXTERN char *hostrpc_varfn_uint_allocate(uint32_t bufsz);
EXTERN char *hostrpc_varfn_uint64_allocate(uint32_t bufsz);
EXTERN char *hostrpc_varfn_double_allocate(uint32_t bufsz);
EXTERN uint32_t hostrpc_varfn_uint_execute(char *bufptr, uint32_t bufsz);
EXTERN uint64_t hostrpc_varfn_uint64_execute(char *bufptr, uint32_t bufsz);
EXTERN double hostrpc_varfn_double_execute(char *bufptr, uint32_t bufsz);

EXTERN uint32_t __strlen_max(char *instr, uint32_t maxstrlen);

EXTERN int vector_product_zeros(int N, int *A, int *B, int *C);

typedef uint32_t hostrpc_varfn_uint_t(void *, ...);
typedef uint64_t hostrpc_varfn_uint64_t(void *, ...);
typedef double hostrpc_varfn_double_t(void *, ...);

EXTERN void hostrpc_fptr0(void *fptr);
EXTERN uint32_t hostrpc_varfn_uint(void *fnptr, ...);
EXTERN uint64_t hostrpc_varfn_uint64(void *fnptr, ...);
EXTERN double hostrpc_varfn_double(void *fnptr, ...);

// Please update at least the patch level when adding a new service.
// This will ensure that applications that use a new device stub do not
// try to use backlevel hostrpc host runtimes that do not have the
// implmented host version of the service.
//
#define HOSTRPC_VERSION 0
#define HOSTRPC_RELEASE 7
#define HOSTRPC_PATCH 1
// HOSTRPC_VRM fits in two bytes allowing for 64 patches, 64 releases, and 15
// versions
#define HOSTRPC_VRM                                                            \
  ((HOSTRPC_VERSION * 4096) + (HOSTRPC_RELEASE * 64) + HOSTRPC_PATCH)
#define HOSTRPC_VERSION_RELEASE ((HOSTRPC_VERSION * 64) + HOSTRPC_RELEASE)
typedef short hostcall_version_t;

#define PACK_VERS(x) ((uint32_t)HOSTRPC_VRM << 16) | ((uint32_t)x)

enum hostcall_service_id {
  HOSTRPC_SERVICE_UNUSED,
  HOSTRPC_SERVICE_TERMINATE,
  HOSTRPC_SERVICE_PRINTF,
  HOSTRPC_SERVICE_MALLOC,
  HOSTRPC_SERVICE_MALLOC_PRINTF,
  HOSTRPC_SERVICE_FREE,
  HOSTRPC_SERVICE_DEMO,
  HOSTRPC_SERVICE_FUNCTIONCALL,
  HOSTRPC_SERVICE_VARFNUINT,
  HOSTRPC_SERVICE_VARFNUINT64,
  HOSTRPC_SERVICE_VARFNDOUBLE,
  HOSTRPC_SERVICE_FPRINTF,
  HOSTRPC_SERVICE_FTNASSIGN,
  HOSTRPC_SERVICE_SANITIZER
};
typedef enum hostcall_service_id hostcall_service_id_t;

#endif // __HOSTRPC_H__
