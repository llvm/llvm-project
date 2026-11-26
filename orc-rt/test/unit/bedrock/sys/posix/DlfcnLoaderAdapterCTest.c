/*===- DlfcnLoaderAdapterCTest.c - Call the C API from C ----------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Calls each orc_rt_DlfcnLoaderAdapter_* entry point from C, for             *|
|* DlfcnLoaderAdapterTest.cpp. This also checks that the C header compiles as *|
|* C.                                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* glibc only declares Dl_info with _GNU_SOURCE (see dlfcn-types.h). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "orc-rt-c/bedrock/sys/posix/DlfcnLoaderAdapter.h"

void *
orc_rt_test_DlfcnLoaderAdapter_dlopenFromC(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                           const char *Path, int Mode) {
  return orc_rt_DlfcnLoaderAdapter_dlopen(Ldr, Path, Mode);
}

int orc_rt_test_DlfcnLoaderAdapter_dlcloseFromC(
    orc_rt_DlfcnLoaderAdapterRef Ldr, void *Handle) {
  return orc_rt_DlfcnLoaderAdapter_dlclose(Ldr, Handle);
}

void *
orc_rt_test_DlfcnLoaderAdapter_dlsymFromC(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                          void *Handle, const char *Symbol) {
  return orc_rt_DlfcnLoaderAdapter_dlsym(Ldr, Handle, Symbol);
}

int orc_rt_test_DlfcnLoaderAdapter_dladdrFromC(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                               const void *Addr,
                                               orc_rt_Dl_info *Info) {
  return orc_rt_DlfcnLoaderAdapter_dladdr(Ldr, Addr, Info);
}

char *
orc_rt_test_DlfcnLoaderAdapter_dlerrorFromC(orc_rt_DlfcnLoaderAdapterRef Ldr) {
  return orc_rt_DlfcnLoaderAdapter_dlerror(Ldr);
}
