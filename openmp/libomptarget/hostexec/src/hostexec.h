//
// hostexec.h: Headers for hostexec device stubs
//
#ifndef __HOSTEXEC_H__
#define __HOSTEXEC_H__

#if defined(__cplusplus)
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif

#include <stdint.h>
#include <stdio.h>

typedef void hostexec_t(void *, ...);
typedef uint32_t hostexec_uint_t(void *, ...);
typedef uint64_t hostexec_uint64_t(void *, ...);
typedef double hostexec_double_t(void *, ...);
typedef float hostexec_float_t(void *, ...);
typedef int hostexec_int_t(void *, ...);
typedef long hostexec_long_t(void *, ...);

#if defined(__NVPTX__) || defined(__AMDGCN__)

// Device interfaces for user-callable hostexec functions
EXTERN void hostexec(void *fnptr, ...);
EXTERN uint32_t hostexec_uint(void *fnptr, ...);
EXTERN uint64_t hostexec_uint64(void *fnptr, ...);
EXTERN double hostexec_double(void *fnptr, ...);
EXTERN float hostexec_float(void *fnptr, ...);
EXTERN int hostexec_int(void *fnptr, ...);
EXTERN long hostexec_long(void *fnptr, ...);

#else

//  On host pass, simply drop the hostexec wrapper. Technically,
//  host passes should not see these hostexec functions
#define hostexec_uint(fn, ...) fn(fn, __VA_ARGS__)
#define hostexec_uint64(fn, ...) fn(fn, __VA_ARGS__)
#define hostexec_double(fn, ...) fn(fn, __VA_ARGS__)
#define hostexec_float(fn, ...) fn(fn, __VA_ARGS__)
#define hostexec_int(fn, ...) fn(fn, __VA_ARGS__)
#define hostexec_long(fn, ...) fn(fn, __VA_ARGS__)
#define hostexec(fn, ...) fn(fn, __VA_ARGS__)

#endif

#endif // __HOSTEXEC_H__
