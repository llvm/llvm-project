//===-------- RuntimeState.cpp ------------------------------------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains managing the OpenACC runtime state.
//
//===----------------------------------------------------------------------===//

#include "RuntimeState.h"
#include "Debug.h"
#include "DeviceManager.h"
#include "Logger.h"
#include "QueueManager.h"

#include <cassert>
#include <mutex>

namespace llvm::acc::target {
PluginManager *PM = nullptr;
} // namespace llvm::acc::target

using namespace llvm::acc::target;
using namespace llvm::acc::target::debug;

static std::mutex &getInitMutex() {
  static std::mutex Mutex;
  return Mutex;
}
static int32_t InitRefCount = 0;

void initAccRuntime() {
  FUNC_LOGGER();
  std::scoped_lock<std::mutex> Lock(getInitMutex());
  assert(InitRefCount >= 0 && "Ref count cannot be negative.");

  if (InitRefCount++ == 0) {
    ODBG(ADT_Init) << "Init OpenACC runtime";
    assert(!PM && "PM is present");
    PM = new PluginManager();
    PM->init();
    // TODO Blindly register all rtls for now. In reality we should only be
    // initializing the requested types in case we come from __tgt_acc_init(),
    // or only the ones we have device code for.
    PM->initializeAllDevices();
    PM->registerDelayedLibraries();

    DM = new DeviceManagerTy();
    DM->init();

    QueueManager = new QueueManagerTy();
    QueueManager->init();
  }
  DM->refreshDeviceMapping(/*UpdateDeviceType=*/true);
}

void deinitAccRuntime() {
  FUNC_LOGGER();
  std::scoped_lock<std::mutex> Lock(getInitMutex());
  assert(PM && "OpenACC runtime not initialized");
  assert(InitRefCount > 0 && "Ref count cannot be negative.");

  if (--InitRefCount == 0) {
    ODBG(ADT_Init) << "Deinit OpenACC runtime";
    QueueManager->deinit();
    delete QueueManager;
    QueueManager = nullptr;

    DM->deinit();
    delete DM;
    DM = nullptr;

    PM->deinit();
    delete PM;
    PM = nullptr;
  }
}
