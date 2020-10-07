///
///  hostrpc.cpp: definitions of device stubs and host fallback functions
///

#include "../hostrpc/src/hostrpc.h"
// ---------------------------------------------------

#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdarg.h>

// This is the host fallback version of hostrpc_fptr0
EXTERN void hostrpc_fptr0(void *fnptr) {
  void (*fptr)() = (void (*)())fnptr;
  (*fptr)();
}

typedef uint hostrpc_varfn_uint_t(void *, ...);
typedef uint64_t hostrpc_varfn_uint64_t(void *, ...);
typedef double hostrpc_varfn_double_t(void *, ...);

// These functions definitions are called directly
// if the target region is run on the host as a fallback.

EXTERN uint hostrpc_varfn_uint(void *fnptr, ...) {
  hostrpc_varfn_uint_t *local_fnptr = (hostrpc_varfn_uint_t *)fnptr;
  uint rc = local_fnptr(fnptr);
  return rc;
}
EXTERN uint64_t hostrpc_varfn_uint64(void *fnptr, ...) {
  hostrpc_varfn_uint64_t *local_fnptr = (hostrpc_varfn_uint64_t *)fnptr;
  uint64_t rc = local_fnptr(fnptr);
  return rc;
}
EXTERN double hostrpc_varfn_double(void *fnptr, ...) {
  hostrpc_varfn_double_t *local_fnptr = (hostrpc_varfn_double_t *)fnptr;
  double rc = local_fnptr(fnptr);
  return rc;
}
int vector_product_zeros(int N, int *A, int *B, int *C) {
  int zeros = 0;
  for (int i = 0; i < N; i++) {
    C[i] = A[i] * B[i];
    if (C[i] == 0)
      zeros++;
  }
  return zeros;
}
static void _error(const char *fname) {
  printf("ERROR: Calls to function %s are for device only execution\n", fname);
}
EXTERN char *printf_allocate(uint32_t bufsz) {
  _error((char *)"printf_allocate");
  return NULL;
}
EXTERN int printf_execute(char *bufptr, uint32_t bufsz) {
  _error("printf_allocate");
  return 0;
}
EXTERN char *hostrpc_varfn_uint_allocate(uint32_t bufsz) {
  _error("hostrpc_varfn_uint_allocate");
  return NULL;
}
EXTERN char *hostrpc_varfn_uint64_allocate(uint32_t bufsz) {
  _error("hostrpc_varfn_uint64_allocate");
  return NULL;
}
EXTERN char *hostrpc_varfn_double_allocate(uint32_t bufsz) {
  _error("hostrpc_varfn_double_allocate");
  return NULL;
}
EXTERN uint32_t hostrpc_varfn_uint_execute(char *bufptr, uint32_t bufsz) {
  _error("hostrpc_varfn_uint_execute");
  return 0;
}
EXTERN uint64_t hostrpc_varfn_uint64_execute(char *bufptr, uint32_t bufsz) {
  _error("hostrpc_varfn_uint64_execute");
  return 0;
}
EXTERN double hostrpc_varfn_double_execute(char *bufptr, uint32_t bufsz) {
  _error("hostrpc_varfn_double_execute");
  return 0;
}

EXTERN char *global_allocate(uint32_t bufsz) {
  printf("HOST FALLBACK EXECUTION OF global_allocate not yet implemented\n");
  return NULL;
}
EXTERN int global_free(char *ptr) {
  printf("HOST FALLBACK EXECUTION OF global_free not yet implemented\n");
  return 0;
}

// NOTE: if you add a new interface above, also add it to
// libomptarget/hostrpc/src/exports and to libomptarget/hostrpc/src/hostrpc.cpp

