/*===------- DlfcnLoaderAdapter.h - dlfcn.h APIs via an adapter ---*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* C entry points that forward dlfcn.h-style calls to a DlfcnLoaderAdapter    *|
|* instance, and the instance symbol through which JIT'd code reaches a       *|
|* session's adapter.                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_BEDROCK_SYS_POSIX_DLFCNLOADERADAPTER_H
#define ORC_RT_C_BEDROCK_SYS_POSIX_DLFCNLOADERADAPTER_H

#include "orc-rt-c/bedrock/sys/posix/dlfcn-types.h"
#include "orc-rt-c/support/Compiler.h"

ORC_RT_C_EXTERN_C_BEGIN

typedef struct orc_rt_OpaqueDlfcnLoaderAdapter *orc_rt_DlfcnLoaderAdapterRef;

/* The instance symbol through which JIT'd code reaches a session's
   DlfcnLoaderAdapter: its address is the adapter's
   orc_rt_DlfcnLoaderAdapterRef. An adapter adds it to the instance symbol
   table passed to its constructor, and it becomes visible to JIT'd code when
   that table is published to the session. Code that refers to it fails to
   link in any session without an adapter. */
#define ORC_RT_DLFCN_LOADER_ADAPTER_INSTANCE orc_rt_DlfcnLoaderAdapter_Instance

ORC_RT_C_EXPORT void *
orc_rt_DlfcnLoaderAdapter_dlopen(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                 const char *Path, int Mode) ORC_RT_C_NOTHROW;

ORC_RT_C_EXPORT int
orc_rt_DlfcnLoaderAdapter_dlclose(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                  void *Handle) ORC_RT_C_NOTHROW;

ORC_RT_C_EXPORT void *
orc_rt_DlfcnLoaderAdapter_dlsym(orc_rt_DlfcnLoaderAdapterRef Ldr, void *Handle,
                                const char *Symbol) ORC_RT_C_NOTHROW;

ORC_RT_C_EXPORT int
orc_rt_DlfcnLoaderAdapter_dladdr(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                 const void *Addr,
                                 orc_rt_Dl_info *Info) ORC_RT_C_NOTHROW;

ORC_RT_C_EXPORT char *orc_rt_DlfcnLoaderAdapter_dlerror(
    orc_rt_DlfcnLoaderAdapterRef Ldr) ORC_RT_C_NOTHROW;

ORC_RT_C_EXTERN_C_END

#endif /* ORC_RT_C_BEDROCK_SYS_POSIX_DLFCNLOADERADAPTER_H */
