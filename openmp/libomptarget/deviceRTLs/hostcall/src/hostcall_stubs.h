#ifndef HOSTCALL_STUBS_H
#define HOSTCALL_STUBS_H

#if defined(__cplusplus)
#define EXTERN extern "C" __attribute__((device))
#else
#define EXTERN __attribute__((device))
#endif

#include <stdint.h>

//  These are the interfaces for the device stubs */
EXTERN int printf(const char *, ...);
EXTERN char *printf_allocate(uint32_t bufsz);
EXTERN char *global_allocate(uint32_t bufsz);
EXTERN int global_free(char *ptr);
EXTERN int printf_execute(char *bufptr, uint32_t bufsz);

EXTERN char *hostrpc_varfn_uint_allocate(uint32_t bufsz);
EXTERN char *hostrpc_varfn_uint64_allocate(uint32_t bufsz);
EXTERN char *hostrpc_varfn_double_allocate(uint32_t bufsz);

EXTERN uint32_t __strlen_max(char*instr, uint32_t maxstrlen);

EXTERN int vector_product_zeros(int N, int *A, int *B, int *C);

typedef uint32_t hostrpc_varfn_uint_t(void *, ...);
typedef uint64_t hostrpc_varfn_uint64_t(void *, ...);
typedef double hostrpc_varfn_double_t(void *, ...);

hostrpc_varfn_uint_t hostrpc_varfn_uint;
hostrpc_varfn_uint64_t hostrpc_varfn_uint64;
hostrpc_varfn_double_t hostrpc_varfn_double;

//  These decls are device and host for host fallback.
#pragma omp declare target
EXTERN void hostrpc_fptr0(void *fptr);
EXTERN uint32_t hostrpc_varfn_uint_execute(char *bufptr, uint32_t bufsz);
EXTERN uint64_t hostrpc_varfn_uint64_execute(char *bufptr, uint32_t bufsz);
EXTERN double hostrpc_varfn_double_execute(char *bufptr, uint32_t bufsz);
#pragma omp end declare target

typedef struct hostcall_result_s{
  uint64_t arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7;
} hostcall_result_t;

EXTERN hostcall_result_t hostcall_invoke(uint32_t id,
    uint64_t arg0, uint64_t arg1, uint64_t arg2, uint64_t arg3,
    uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7);

#endif
