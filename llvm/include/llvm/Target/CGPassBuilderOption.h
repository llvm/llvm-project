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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetOptions.h"
#include <optional>

namespace llvm {

enum class RunOutliner {
  TargetDefault,
  AlwaysOutline,
  OptimisticPGO,
  ConservativePGO,
  NeverOutline
};
enum class RegAllocType { Unset, Default, Basic, Fast, Greedy, PBQP };

class RegAllocTypeParser : public cl::parser<RegAllocType> {
public:
  RegAllocTypeParser(cl::Option &O) : cl::parser<RegAllocType>(O) {}
  void initialize() {
    cl::parser<RegAllocType>::initialize();
    addLiteralOption("default", RegAllocType::Default,
                     "Default register allocator");
    addLiteralOption("pbqp", RegAllocType::PBQP, "PBQP register allocator");
    addLiteralOption("fast", RegAllocType::Fast, "Fast register allocator");
    addLiteralOption("basic", RegAllocType::Basic, "Basic register allocator");
    addLiteralOption("greedy", RegAllocType::Greedy,
                     "Greedy register allocator");
  }
};

// Not one-on-one but mostly corresponding to commandline options in
// TargetPassConfig.cpp.
struct CGPassBuilderOption {
  std::optional<bool> OptimizeRegAlloc;
  std::optional<bool> EnableIPRA;
  bool DebugPM = false;
  bool DisableVerify = false;
  bool EnableImplicitNullChecks = false;
  bool EnableBlockPlacementStats = false;
  bool EnableGlobalMergeFunc = false;
  bool EnableMachineFunctionSplitter = false;
  bool EnableSinkAndFold = false;
  bool EnableTailMerge = true;
  /// Enable LoopTermFold immediately after LSR.
  bool EnableLoopTermFold = false;
  bool MISchedPostRA = false;
  bool EarlyLiveIntervals = false;
  bool GCEmptyBlocks = false;

  bool DisableLSR = false;
  bool DisableCGP = false;
  bool DisableMergeICmps = false;
  bool DisablePartialLibcallInlining = false;
  bool DisableConstantHoisting = false;
  bool DisableSelectOptimize = true;
  bool DisableAtExitBasedGlobalDtorLowering = false;
  bool DisableExpandReductions = false;
  bool DisableRAFSProfileLoader = false;
  bool DisableCFIFixup = false;
  bool PrintAfterISel = false;
  bool PrintISelInput = false;
  bool RequiresCodeGenSCCOrder = false;

  RunOutliner EnableMachineOutliner = RunOutliner::TargetDefault;
  RegAllocType RegAlloc = RegAllocType::Unset;
  std::optional<GlobalISelAbortMode> EnableGlobalISelAbort;
  std::string FSProfileFile;
  std::string FSRemappingFile;

  std::optional<bool> VerifyMachineCode;
  std::optional<bool> EnableFastISelOption;
  std::optional<bool> EnableGlobalISelOption;
  std::optional<bool> DebugifyAndStripAll;
  std::optional<bool> DebugifyCheckAndStripAll;
};

LLVM_ABI CGPassBuilderOption getCGPassBuilderOption();

} // namespace llvm

#endif // LLVM_TARGET_CGPASSBUILDEROPTION_H
