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

#include "OpenMP/OffloadRTL.h"
#include "OpenMP/OMPT/Callback.h"
#include "PluginManager.h"

#include "Shared/Debug.h"
#include "Shared/Profile.h"

using namespace llvm::omp::target::debug;
using llvm::omp::target::OmpPluginManager;

static std::mutex &getPluginMutex() {
  static std::mutex Mutex;
  return Mutex;
}
static uint32_t RefCount = 0;
std::atomic<bool> RTLAlive{false};
std::atomic<int> RTLOngoingSyncs{0};
OmpPluginManager *PM = nullptr;

/// Check deleted and deprecated features, such as environment variables.
static void checkRuntimeEnvironment() {
  const char *ShmemEnvarName = "LIBOMPTARGET_SHARED_MEMORY_SIZE";
  if (std::getenv(ShmemEnvarName))
    MESSAGE("Warning: %s is no longer valid. Please use OpenMP clause "
            "'dyn_groupprivate' instead.\n",
            ShmemEnvarName);
}

void initRuntime() {
  std::scoped_lock<std::mutex> Lock(getPluginMutex());
  Profiler::get();
  TIMESCOPE();

  checkRuntimeEnvironment();

  RefCount++;
  if (RefCount == 1) {
    assert(PM == nullptr);
    PM = new llvm::omp::target::OmpPluginManager();

    ODBG(ODT_Init) << "Init offload library!";
#ifdef OMPT_SUPPORT
    // Initialize OMPT first
    llvm::omp::target::ompt::connectLibrary();
#endif

    PM->init();
    PM->registerDelayedLibraries();
    // After all plugins are initialized, register atExit cleanup handlers.
    std::atexit([]() {
      // Interop cleanup should be done before the plugins are deinitialized as
      // the backend libraries may be already unloaded.
      if (PM)
        PM->InteropTbl.clear();
    });

    // RTL initialization is complete
    RTLAlive = true;
  }
}

void deinitRuntime() {
  std::scoped_lock<std::mutex> Lock(getPluginMutex());
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
