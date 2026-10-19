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

#include <cassert>
#include <mutex>

namespace llvm::acc::target {
extern PluginManager *PM = nullptr;
} // namespace llvm::acc::target

using namespace llvm::acc::target;

static std::mutex &getInitMutex() {
  static std::mutex Mutex;
  return Mutex;
}
static int32_t InitRefCount = 0;

void initAccRuntime() {
  std::scoped_lock<std::mutex> Lock(getInitMutex());
  assert(InitRefCount >= 0);

  if (InitRefCount++ == 0) {
    ODBG(ADT_Init) << "Init OpenACC runtime";
    assert(!PM && "PM is present");
    PM = new PluginManager();
    PM->init();
    PM->initializeAllDevices();
  }
}

void deinitAccRuntime() {
  std::scoped_lock<std::mutex> Lock(getInitMutex());
  assert(PM && "OpenACC runtime not initialized");
  assert(InitRefCount > 0);

  if (--InitRefCount == 0) {
    ODBG(ADT_Init) << "Deinit OpenACC runtime";
    PM->deinit();
    delete PM;
    PM = nullptr;
  }
}
