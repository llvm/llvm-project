//===- CGPassBuilderOption.h - Options for pass builder ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the options influencing building of codegen pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_CGPASSBUILDEROPTION_H
#define LLVM_CODEGEN_CGPASSBUILDEROPTION_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Target/TargetOptions.h"
#include <vector>

namespace llvm {
class TargetMachine;

enum class RunOutliner { TargetDefault, AlwaysOutline, NeverOutline };
enum class RegAllocType { Default, Basic, Fast, Greedy, PBQP };
enum class CFLAAType { None, Steensgaard, Andersen, Both };

// Not one-on-one but mostly corresponding to commandline options in
// TargetPassConfig.cpp
struct CGPassBuilderOption {
  // Enable optimized register allocation compilation path
  Optional<bool> OptimizeRegAlloc;

  // Enable interprocedural register allocation to reduce load/store at
  // procedure calls
  Optional<bool> EnableIPRA;

  // Enable debug logging of pass pipeline
  bool DebugPM = false;

  // Disable machine function verification
  bool DisableVerify = false;

  // Fold null checks into faulting memory operations
  bool EnableImplicitNullChecksPass = false;

  // Collect probability-driven block placement stats
  bool EnableMachineBlockPlacementStatsPass = false;

  // Run MachineScheduler post regalloc (independent of preRA sched)
  bool EnablePostMachineSchedulerPass = false;

  // Run live interval analysis earlier in the pipeline
  bool EnableLiveIntervalsPass = false;

  // Disable Loop Strength Reduction Pass
  bool DisableLoopStrengthReducePass = false;

  // Disable Codegen Prepare
  bool DisableCodeGenPreparePass = false;

  // Disable MergeICmps Pass
  bool DisableMergeICmpsPass = false;

  // Disable Partial Libcall Inlining Pass
  bool DisablePartiallyInlineLibCallsPass = false;

  // Disable ConstantHoisting Pass
  bool DisableConstantHoistingPass = false;

  // Print LLVM IR produced by the loop-reduce pass
  bool PrintAfterLSR = false;

  // Print LLVM IR input to isel pass
  bool PrintISelInput = false;

  // Dump garbage collector data
  bool PrintGCInfo = false;

  // Enable codegen in SCC order.
  bool RequiresCodeGenSCCOrder = false;

  // Enable the machine outliner
  RunOutliner EnableMachineOutliner = RunOutliner::TargetDefault;

  // Register allocator to use
  RegAllocType RegAlloc = RegAllocType::Default;

  // Experimental option to use CFL-AA in codegen
  CFLAAType UseCFLAA = CFLAAType::None;

  // Enable abort calls when "global" instruction selection fails to
  // lower/select an instruction
  Optional<GlobalISelAbortMode> EnableGlobalISelAbort;

  // Verify generated machine code"
  Optional<bool> VerifyMachineCode;

  // Enable the "fast" instruction selector
  Optional<bool> EnableFastISelOption;

  // Enable the "global" instruction selector
  Optional<bool> EnableGlobalISelOption;
};

CGPassBuilderOption getCGPassBuilderOption();

} // namespace llvm

#endif // LLVM_CODEGEN_CGPASSBUILDEROPTION_H
