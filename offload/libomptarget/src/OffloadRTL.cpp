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
#include "OffloadPolicy.h"
#include "OpenMP/OMPT/Callback.h"
#include "PluginManager.h"

#include "Profile.h"
#include "Shared/Debug.h"

#include <thread>

using namespace llvm::omp::target::debug;
using llvm::omp::target::OmpPluginManager;

static std::mutex &getPluginMutex() {
  static std::mutex Mutex;
  return Mutex;
}
/// Guards TornDown and PM's creation against the exit handler.
static std::mutex &getTeardownMutex() {
  static std::mutex Mutex;
  return Mutex;
}
static uint32_t RefCount = 0;
static bool TornDown = false;
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

/// Idempotent teardown; the teardown mutex must be held.
static void teardownRuntimeLocked() {
  if (TornDown || !PM)
    return;
  TornDown = true;

  ODBG(ODT_Deinit) << "Tearing down offload library!";
  RTLAlive = false;
  while (RTLOngoingSyncs > 0) {
    ODBG(ODT_Sync) << "Waiting for ongoing syncs to finish, count:"
                   << RTLOngoingSyncs.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  // Backend libraries may be unloaded after deinit.
  PM->InteropTbl.clear();
  PM->deinit();
}

void teardownRuntime() {
  std::scoped_lock<std::mutex> Lock(getTeardownMutex());
  teardownRuntimeLocked();
}

/// Exit handlers registered by onPlatformsInitialized() that have not run yet.
static std::atomic<int> PendingExitHandlers{0};

/// Exit-time fallback if deinitRuntime() has not run. A module's unregister
/// handler is registered after the instance from its __tgt_register_lib, so
/// only the last instance to run (the earliest registered) tears down.
static void exitTeardown() {
  if (--PendingExitHandlers == 0)
    teardownRuntime();
}

void llvm::omp::target::OmpPluginManager::onPlatformsInitialized() {
  ++PendingExitHandlers;
  if (std::atexit(exitTeardown))
    --PendingExitHandlers;
}

void initRuntime() {
  std::scoped_lock<std::mutex> Lock(getPluginMutex());
  Profiler::get();
  TIMESCOPE();

  checkRuntimeEnvironment();

  RefCount++;
  if (RefCount == 1) {
    assert(!PM && "Runtime already initialized");
    {
      std::scoped_lock<std::mutex> TeardownLock(getTeardownMutex());
      PM = new llvm::omp::target::OmpPluginManager();
      TornDown = false;
    }

    ODBG(ODT_Init) << "Init offload library!";
#ifdef OMPT_SUPPORT
    // Initialize OMPT first
    llvm::omp::target::ompt::connectLibrary();
#endif

    if (!OffloadPolicy::isOffloadDisabled())
      PM->init();
    else
      ODBG(ODT_Init) << "Offload is disabled. Skipping plugin initialization";
    PM->registerDelayedLibraries();

    // RTL initialization is complete
    RTLAlive = true;
  }
}

void deinitRuntime() {
  std::scoped_lock<std::mutex> Lock(getPluginMutex());
  assert(PM && "Runtime not initialized");

  if (RefCount == 1) {
    ODBG(ODT_Deinit) << "Deinit offload library!";
    teardownRuntime();
    // TornDown is set, so exit handlers no longer touch PM.
    delete PM;
    PM = nullptr;
  }

  RefCount--;
}
