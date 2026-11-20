//===- dlfcn-host.cpp - Call host dlfcn.h APIs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Call through to the host dlfcn.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-c/bedrock/host/posix/dlfcn-host.h"

extern "C" {

void *orc_rt_dlopen_host(const char *path, int mode) noexcept {
  return dlopen(path, mode);
}

int orc_rt_dlclose_host(void *handle) noexcept { return dlclose(handle); }

void *orc_rt_dlsym_host(void *handle, const char *symbol) noexcept {
  return dlsym(handle, symbol);
}

int orc_rt_dladdr_host(const void *addr, Dl_info *info) noexcept {
  return dladdr(addr, info);
}

char *orc_rt_dlerror_host(void) noexcept { return dlerror(); }

} // extern "C"
