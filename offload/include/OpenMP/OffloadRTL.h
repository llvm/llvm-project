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

#ifndef _OPENMP_OMPPLUGINMANAGER_H_
#define _OPENMP_OMPPLUGINMANAGER_H_

#include "OpenMP/InteropAPI.h"
#include "PluginManager.h"

namespace llvm::omp::target {
class OmpPluginManager : public PluginManager {
public:
  /// Table of cached implicit interop objects
  InteropTblTy InteropTbl;
};
} // namespace llvm::omp::target

extern llvm::omp::target::OmpPluginManager *PM;
extern std::atomic<bool> RTLAlive; // Indicates if the RTL has been initialized
extern std::atomic<int> RTLOngoingSyncs; // Counts ongoing external syncs

#endif // _OPENMP_OMPPLUGINMANAGER_H_
