/*===------------ dlfcn-host.h - Call host dlfcn.h APIs -----------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* The dlfcn-host.h APIs always call through to the host's native dlfcn.h     *|
|* APIs (the host must be a POSIX system), bypassing the ORC runtime. They    *|
|* are the counterpart to the spire/sys/posix/dlfcn-jit.h APIs: JIT'd code    *|
|* can use them to reach host libraries directly, e.g. when the usual dlfcn.h *|
|* names (dlopen, dlsym, etc.) have been aliased to the dlfcn-jit.h APIs.     *|
|*                                                                            *|
|* Caller-sensitive arguments resolve relative to the ORC runtime, not to the *|
|* JIT'd code calling these APIs, since the host's dynamic loader sees the    *|
|* runtime as the caller. E.g. dlsym(RTLD_NEXT, ...) searches the images      *|
|* loaded after the runtime's image, and dlopen resolves @loader_path and     *|
|* @rpath (Darwin) or $ORIGIN and DT_RUNPATH (ELF) relative to the runtime's  *|
|* image.                                                                     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_BEDROCK_HOST_POSIX_DLFCN_HOST_H
#define ORC_RT_C_BEDROCK_HOST_POSIX_DLFCN_HOST_H

#include "orc-rt-c/support/Compiler.h"

#include <dlfcn.h>

ORC_RT_C_EXTERN_C_BEGIN

ORC_RT_C_EXPORT void *orc_rt_dlopen_host(const char *path,
                                         int mode) ORC_RT_C_NOTHROW;
ORC_RT_C_EXPORT int orc_rt_dlclose_host(void *handle) ORC_RT_C_NOTHROW;
ORC_RT_C_EXPORT void *orc_rt_dlsym_host(void *handle,
                                        const char *symbol) ORC_RT_C_NOTHROW;
ORC_RT_C_EXPORT int orc_rt_dladdr_host(const void *addr,
                                       Dl_info *info) ORC_RT_C_NOTHROW;
/* Returns char * to match dlerror's declaration in <dlfcn.h>. As with dlerror,
   the string must not be modified or freed. */
ORC_RT_C_EXPORT char *orc_rt_dlerror_host(void) ORC_RT_C_NOTHROW;

ORC_RT_C_EXTERN_C_END

#endif /* ORC_RT_C_BEDROCK_HOST_POSIX_DLFCN_HOST_H */
