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
using namespace llvm::omp::target::debug;

static std::mutex PluginMtx;
static uint32_t RefCount = 0;
std::atomic<bool> RTLAlive{false};
std::atomic<int> RTLOngoingSyncs{0};

void initRuntime() {
  std::scoped_lock<decltype(PluginMtx)> Lock(PluginMtx);
  Profiler::get();
  TIMESCOPE();

  if (PM == nullptr)
    PM = new PluginManager();

  RefCount++;
  if (RefCount == 1) {
    ODBG(ODT_Init) << "Init offload library!";
#ifdef OMPT_SUPPORT
    // Initialize OMPT first
    llvm::omp::target::ompt::connectLibrary();
#endif

    PM->init();
    PM->registerDelayedLibraries();

    // RTL initialization is complete
    RTLAlive = true;
  }
}

void deinitRuntime() {
  std::scoped_lock<decltype(PluginMtx)> Lock(PluginMtx);
  assert(PM && "Runtime not initialized");

  if (RefCount == 1) {
    ODBG(ODT_Deinit) << "Deinit offload library!";
    // RTL deinitialization has started
    RTLAlive = false;
    while (RTLOngoingSyncs > 0) {
      ODBG(ODT_Sync) << "Waiting for ongoing syncs to finish, count:"
                     << RTLOngoingSyncs.load();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    PM->deinit();
    delete PM;
    PM = nullptr;
  }

  RefCount--;
}
