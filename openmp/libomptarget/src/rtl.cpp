//===----------- rtl.cpp - Target independent OpenMP target RTL -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality for handling RTL plugins.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/OffloadBinary.h"
#include "llvm/OffloadArch/OffloadArch.h"

#include "DeviceImage.h"
#include "OpenMP/OMPT/Callback.h"
#include "PluginManager.h"
#include "device.h"
#include "private.h"
#include "rtl.h"

#include "Shared/Debug.h"
#include "Shared/Profile.h"
#include "Shared/Utils.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

using namespace llvm;
using namespace llvm::sys;
using namespace llvm::omp::target;

#ifdef OMPT_SUPPORT
extern void ompt::connectLibrary();
#endif

__attribute__((constructor(101))) void init() {
  DP("Init target library!\n");

  PM = new PluginManager();

#ifdef OMPT_SUPPORT
  // Initialize OMPT first
  ompt::connectLibrary();
#endif

  PM->init();

  Profiler::get();
  PM->registerDelayedLibraries();
}

__attribute__((destructor(101))) void deinit() {
  DP("Deinit target library!\n");
  delete PM;
}
