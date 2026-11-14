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

#include "Shared/Debug.h"
#include "Shared/Profile.h"

#include <thread>

using namespace llvm::omp::target::debug;
using llvm::omp::target::OmpPluginManager;

static std::mutex &getPluginMutex() {
  static std::mutex Mutex;
  return Mutex;
}
/// Guards PM and TornDown against the shutdown callback.
static std::mutex &getTeardownMutex() {
  static std::mutex Mutex;
  return Mutex;
}
static uint32_t RefCount = 0;
static bool TornDown = false;
// A shutdown callback is pending in liboffload; register at most one.
static bool ShutdownCallbackPending = false;
static void teardownRuntime();
static void shutdownCallback(void *);
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

/// Idempotent teardown, called from deinitRuntime() and the shutdown callback.
/// Must not call olShutDown or take the plugin mutex.
static void teardownRuntimeLocked() {
  if (!PM || TornDown)
    return;
  TornDown = true;

  ODBG(ODT_Deinit) << "Tearing down offload library!";
  // RTL deinitialization has started
  RTLAlive = false;
  while (RTLOngoingSyncs > 0) {
    ODBG(ODT_Sync) << "Waiting for ongoing syncs to finish, count:"
                   << RTLOngoingSyncs.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  // Must precede device deinit: the backends may be unloaded afterwards.
  PM->InteropTbl.clear();
  PM->deinitDevices();
}

static void teardownRuntime() {
  std::scoped_lock<std::mutex> Lock(getTeardownMutex());
  teardownRuntimeLocked();
}

static void shutdownCallback(void *) {
  std::scoped_lock<std::mutex> Lock(getTeardownMutex());
  // liboffload drops the callback once it ran.
  ShutdownCallbackPending = false;
  teardownRuntimeLocked();
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
    // Tear down before liboffload deinitializes the platforms at exit.
    if (PM->isLibOffloadInitialized()) {
      std::scoped_lock<std::mutex> TeardownLock(getTeardownMutex());
      if (!ShutdownCallbackPending) {
        if (ol_result_t Res =
                olRegisterShutdownCallback(shutdownCallback, nullptr))
          REPORT() << "Failed to register shutdown callback: " << Res->Details;
        else
          ShutdownCallbackPending = true;
      }
    }

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
    PM->deinit();
    std::scoped_lock<std::mutex> TeardownLock(getTeardownMutex());
    delete PM;
    PM = nullptr;
  }

  RefCount--;
}
