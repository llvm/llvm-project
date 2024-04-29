//===----------- rtl.cpp - Target independent OpenMP target RTL -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Initialization and tear down of the offload runtime.
//
//===----------------------------------------------------------------------===//

#include "OpenMP/OMPT/Callback.h"
#include "PluginManager.h"

#include "Shared/Debug.h"
#include "Shared/Profile.h"

#ifdef OMPT_SUPPORT
extern void llvm::omp::target::ompt::connectLibrary();
#endif

static std::mutex PluginMtx;
static uint32_t RefCount = 0;

void initRuntime() {
  std::scoped_lock<decltype(PluginMtx)> Lock(PluginMtx);
  Profiler::get();
  TIMESCOPE();

  if (PM == nullptr)
    PM = new PluginManager();

  RefCount++;
  if (RefCount == 1) {
    DP("Init offload library!\n");
#ifdef OMPT_SUPPORT
    // Initialize OMPT first
    llvm::omp::target::ompt::connectLibrary();
#endif

    PM->init();
    PM->registerDelayedLibraries();
  }
}

void deinitRuntime() {
  std::scoped_lock<decltype(PluginMtx)> Lock(PluginMtx);
  assert(PM && "Runtime not initialized");

  if (RefCount == 1) {
    DP("Deinit offload library!\n");
    delete PM;
    PM = nullptr;
  }

  RefCount--;
}

// HACK: These depricated device stubs still needs host versions for fallback
// FIXME: Deprecate upstream, change test cases to use malloc & free directly
extern "C" char *global_allocate(uint32_t sz) { return (char *)malloc(sz); }
extern "C" int global_free(void *ptr) {
  free(ptr);
  return 0;
}
