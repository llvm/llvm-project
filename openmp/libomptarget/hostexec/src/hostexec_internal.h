#ifndef __HOSTEXEC_INTERNAL_H__
#define __HOSTEXEC_INTERNAL_H__

/*
 *   hostexec_internal.h:

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

#include "hostexec.h"
#include <stdint.h>
#include <stdio.h>

//  These are the interfaces for the device stubs emitted
//  by EmitHostexecAllocAndExecFns in CGGPUBuiltin.cpp
EXTERN char *printf_allocate(uint32_t bufsz);
EXTERN int printf_execute(char *bufptr, uint32_t bufsz);
EXTERN char *fprintf_allocate(uint32_t bufsz);
EXTERN int fprintf_execute(char *bufptr, uint32_t bufsz);
EXTERN char *hostexec_allocate(uint32_t bufsz);
EXTERN void hostexec_execute(char *bufptr, uint32_t bufsz);
EXTERN char *hostexec_uint_allocate(uint32_t bufsz);
EXTERN uint32_t hostexec_uint_execute(char *bufptr, uint32_t bufsz);
EXTERN char *hostexec_uint64_allocate(uint32_t bufsz);
EXTERN uint64_t hostexec_uint64_execute(char *bufptr, uint32_t bufsz);
EXTERN char *hostexec_double_allocate(uint32_t bufsz);
EXTERN double hostexec_double_execute(char *bufptr, uint32_t bufsz);
EXTERN char *hostexec_int_allocate(uint32_t bufsz);
EXTERN int hostexec_int_execute(char *bufptr, uint32_t bufsz);
EXTERN char *hostexec_long_allocate(uint32_t bufsz);
EXTERN long hostexec_long_execute(char *bufptr, uint32_t bufsz);
EXTERN char *hostexec_float_allocate(uint32_t bufsz);
EXTERN float hostexec_float_execute(char *bufptr, uint32_t bufsz);

// This device runtime utility function is needed for variable length strings.
EXTERN uint32_t __strlen_max(char *instr, uint32_t maxstrlen);

// The version release and patch level (VRM) are embedded in every packet with
// the service id (sid) and checked by the host runtime on the first packet.
// The runtime fails if VR is different or device VRM > host VRM.
// Runtime warns if device VRM < host VRM i.e old compiler with newer runtime.
// See check_version in services/<arch>_hostexec.cpp.
#define HOSTEXEC_VERSION 0
#define HOSTEXEC_RELEASE 1
#define HOSTEXEC_PATCH 0
// HOSTEXEC_VRM uses 2 bytes allowing 64 patches, 64 releases, 15 versions
#define HOSTEXEC_VRM                                                           \
  ((HOSTEXEC_VERSION * 4096) + (HOSTEXEC_RELEASE * 64) + HOSTEXEC_PATCH)
#define HOSTEXEC_VERSION_RELEASE ((HOSTEXEC_VERSION * 64) + HOSTEXEC_RELEASE)

// This macro packs VRM and sid id into the 1st 4 bytes of the packet.
#define PACK_VERS(x) ((uint32_t)HOSTEXEC_VRM << 16) | ((uint32_t)x)

// The host runtime for host services is linked statically into the
// libomptarget plugin i.e.  libomptarget.rtl.amdgcn or libomptarget.rtl.cuda
// Typically these are part of the compiler installation so VRM checking
// would not be necessary. However, compiled applications that dynamically
// link to libomptarget and the plugin may might get an old runtime.
// so VRM checking is
//
// Please update at least the patch level when adding a new service id (sid)
// below. This ensures that applications that use a new device stub do not
// try to use backlevel host runtimes that do not have a valid VRM.
enum hostexec_sid {
  HOSTEXEC_SID_UNUSED,
  HOSTEXEC_SID_TERMINATE,
  HOSTEXEC_SID_DEVICE_MALLOC, // Device global memory
  HOSTEXEC_SID_HOST_MALLOC,   // shared or managed memory
  HOSTEXEC_SID_FREE,
  HOSTEXEC_SID_PRINTF,
  HOSTEXEC_SID_FPRINTF,
  HOSTEXEC_SID_FTNASSIGN,
  HOSTEXEC_SID_SANITIZER,
  HOSTEXEC_SID_UINT,
  HOSTEXEC_SID_UINT64,
  HOSTEXEC_SID_DOUBLE,
  HOSTEXEC_SID_INT,
  HOSTEXEC_SID_LONG,
  HOSTEXEC_SID_FLOAT,
  HOSTEXEC_SID_VOID,
};

#endif // __HOSTEXEC_INTERNAL_H__
