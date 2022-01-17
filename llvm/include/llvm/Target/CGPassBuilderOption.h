//===- CGPassBuilderOption.h - Options for pass builder ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CCState and CCValAssign classes, used for lowering
// and implementing calling conventions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_CGPASSBUILDEROPTION_H
#define LLVM_TARGET_CGPASSBUILDEROPTION_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Target/TargetOptions.h"

namespace llvm {

enum class RunOutliner { TargetDefault, AlwaysOutline, NeverOutline };
enum class RegAllocType { Default, Basic, Fast, Greedy, PBQP };
enum class CFLAAType { None, Steensgaard, Andersen, Both };

// Not one-on-one but mostly corresponding to commandline options in
// TargetPassConfig.cpp.
struct CGPassBuilderOption {
  Optional<bool> OptimizeRegAlloc;
  Optional<bool> EnableIPRA;
  bool DebugPM = false;
  bool DisableVerify = false;
  bool EnableImplicitNullChecks = false;
  bool EnableBlockPlacementStats = false;
  bool MISchedPostRA = false;
  bool EarlyLiveIntervals = false;

  bool DisableLSR = false;
  bool DisableCGP = false;
  bool PrintLSR = false;
  bool DisableMergeICmps = false;
  bool DisablePartialLibcallInlining = false;
  bool DisableConstantHoisting = false;
  bool PrintISelInput = false;
  bool PrintGCInfo = false;
  bool RequiresCodeGenSCCOrder = false;

  RunOutliner EnableMachineOutliner = RunOutliner::TargetDefault;
  RegAllocType RegAlloc = RegAllocType::Default;
  CFLAAType UseCFLAA = CFLAAType::None;
  Optional<GlobalISelAbortMode> EnableGlobalISelAbort;

  Optional<bool> VerifyMachineCode;
  Optional<bool> EnableFastISelOption;
  Optional<bool> EnableGlobalISelOption;
};

CGPassBuilderOption getCGPassBuilderOption();

} // namespace llvm

#endif // LLVM_TARGET_CGPASSBUILDEROPTION_H
