//===-------- OffloadRTL.h ---------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains OpenMP specific PluginManager state.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OPENMP_OFFLOADRTL_H
#define OMPTARGET_OPENMP_OFFLOADRTL_H

#include "PluginManager.h"

#include <atomic>

extern PluginManager *PM;
extern std::atomic<bool> RTLAlive; // Indicates if the RTL has been initialized
extern std::atomic<int> RTLOngoingSyncs; // Counts ongoing external syncs

/// Initialize the plugin manager and OpenMP runtime.
void initRuntime();

/// Deinitialize the plugin and delete it.
void deinitRuntime();

#endif // OMPTARGET_OPENMP_OFFLOADRTL_H
