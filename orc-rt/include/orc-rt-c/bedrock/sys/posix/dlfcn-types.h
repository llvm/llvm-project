/*===--------- dlfcn-types.h - dlfcn.h types and constants --------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* The dlfcn.h types and constants used by the ORC runtime's dlfcn APIs:      *|
|* orc_rt_Dl_info, and ORC_RT_RTLD_* for the mode flags and special handles.  *|
|*                                                                            *|
|* Where the host provides <dlfcn.h>, these are the host's own definitions,   *|
|* so that the runtime's dlfcn APIs use the same Dl_info layout and RTLD_*    *|
|* values as the host's. Elsewhere the runtime provides its own definitions,  *|
|* which are ORC runtime ABI.                                                 *|
|*                                                                            *|
|* On glibc and musl, Dl_info and RTLD_DEFAULT / RTLD_NEXT are only declared  *|
|* when _GNU_SOURCE is defined (musl also accepts _BSD_SOURCE), so C code     *|
|* that includes this header there must define it. (C++ compilers define      *|
|* _GNU_SOURCE by default on Linux.)                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_BEDROCK_SYS_POSIX_DLFCN_TYPES_H
#define ORC_RT_C_BEDROCK_SYS_POSIX_DLFCN_TYPES_H

#include "orc-rt-c/config.h"

#if ORC_RT_TARGET_POSIX

#include <dlfcn.h>

/* If this fails to compile on glibc or musl, define _GNU_SOURCE (see above). */
typedef Dl_info orc_rt_Dl_info;

#define ORC_RT_RTLD_LAZY RTLD_LAZY
#define ORC_RT_RTLD_NOW RTLD_NOW
#define ORC_RT_RTLD_GLOBAL RTLD_GLOBAL
#define ORC_RT_RTLD_LOCAL RTLD_LOCAL
#define ORC_RT_RTLD_DEFAULT RTLD_DEFAULT
#define ORC_RT_RTLD_NEXT RTLD_NEXT

#else /* !ORC_RT_TARGET_POSIX */

#include <stdint.h>

/* The fields and field order used by every common POSIX implementation. */
typedef struct {
  const char *dli_fname; /* Path of the image containing the address. */
  void *dli_fbase;       /* Base address of that image. */
  const char *dli_sname; /* Name of the nearest symbol below the address. */
  void *dli_saddr;       /* Address of that symbol. */
} orc_rt_Dl_info;

/* Values follow Darwin's. */
#define ORC_RT_RTLD_LAZY 0x1
#define ORC_RT_RTLD_NOW 0x2
#define ORC_RT_RTLD_LOCAL 0x4
#define ORC_RT_RTLD_GLOBAL 0x8
/* C++ casts in C++, so that the handles don't trigger -Wold-style-cast. */
#ifdef __cplusplus
#define ORC_RT_RTLD_DEFAULT reinterpret_cast<void *>(static_cast<intptr_t>(-2))
#define ORC_RT_RTLD_NEXT reinterpret_cast<void *>(static_cast<intptr_t>(-1))
#else
#define ORC_RT_RTLD_DEFAULT ((void *)(intptr_t)-2)
#define ORC_RT_RTLD_NEXT ((void *)(intptr_t)-1)
#endif

#endif /* ORC_RT_TARGET_POSIX */

#endif /* ORC_RT_C_BEDROCK_SYS_POSIX_DLFCN_TYPES_H */
