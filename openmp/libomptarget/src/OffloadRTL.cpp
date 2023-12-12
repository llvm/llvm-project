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

__attribute__((constructor(101))) void init() {
  Profiler::get();
  TIMESCOPE();

  DP("Init offload library!\n");

  PM = new PluginManager();

#ifdef OMPT_SUPPORT
  // Initialize OMPT first
  llvm::omp::target::ompt::connectLibrary();
#endif

  PM->init();

  PM->registerDelayedLibraries();
}

__attribute__((destructor(101))) void deinit() {
  DP("Deinit offload library!\n");
  delete PM;
}
