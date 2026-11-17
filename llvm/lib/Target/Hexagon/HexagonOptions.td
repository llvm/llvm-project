//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def HexagonOptions : OptionsStruct;

// HexagonISelLowering.cpp
defm : BoolField<"constant-loads-to-imm", "true",
  "Convert constant loads to immediate values">;
defm : BoolField<"enable-hexagon-sdnode-sched", "false",
  "Enable Hexagon SDNode scheduling">;
defm : BoolField<"hexagon-align-loads", "false",
  "Rewrite unaligned loads as a pair of aligned loads">;
defm : BoolField<"hexagon-args-min-alignment", "true",
  "Enable minimum alignment of 1 for arguments passed by value on stack">;
defm : BoolField<"hexagon-emit-jump-tables", "true",
  "Control jump table emission on Hexagon target">;
defm : ValueField<"minimum-jump-tables", "int", "5",
  "Set minimum jump tables">;

// HexagonISelLoweringHVX.cpp
defm : BoolField<"hexagon-fp-fast-convert", "false",
  "Enable FP fast conversion routine">;
defm : OptionalValueField<"hexagon-hvx-widen", "unsigned",
  "Lower threshold (in bytes) for widening to HVX vectors">;
defm : BoolField<"hexagon-qf-fast-convert", "false",
  "Enable XQF fast conversion routine">;

// HexagonFrameLowering.cpp
defm : BoolField<"enable-save-restore-long", "false",
  "Enable long calls for save-restore stubs">;
defm : BoolField<"enable-stackovf-sanitizer", "false",
  "Enable runtime checks for stack overflow">;
defm : BoolField<"hexagon-dealloc-ret", "true",
  "Enable Dealloc Return for Hexagon target">;
defm : BoolField<"hexagon-fp-elim", "true",
  "Refrain from using FP whenever possible">;
defm : BoolField<"hexagon-opt-spill", "true",
  "Optimize spill slots">;
defm : BoolField<"hexagon-shrink-frame", "true",
  "Enable stack frame shrink wrapping">;
defm : ValueField<"number-scavenger-slots", "unsigned", "2",
  "Set the number of scavenger slots">;
defm : OptionalValueField<"shrink-frame-limit", "unsigned",
  "Max count of stack frame shrink-wraps">;
defm : ValueField<"spill-func-threshold", "int", "6",
  "Specify O2(not Os) spill func threshold">;
defm spill_func_threshold_os : ValueField<"spill-func-threshold-Os", "int",
  "1", "Specify Os spill func threshold">;
let Condition = AssertsOnly in
defm : OptionalValueField<"spill-opt-max", "unsigned", "">;

// HexagonSubtarget.cpp
defm : OptionalBoolField<"enable-bsb-sched", "">;
defm : BoolField<"enable-cur-sched", "true",
  "Enable the scheduler to generate .cur">;
defm : BoolField<"hexagon-check-bank-conflict", "true",
  "Enable checking for cache bank conflicts">;
defm : OptionalBoolField<"hexagon-long-calls",
  "If present, forces/disables the use of long calls">;
defm : BoolField<"hexagon-misched", "true",
  "Enable Hexagon MI Scheduling">;
defm : BoolField<"hexagon-pred-calls", "false",
  "Consider calls to be predicable">;
defm : BoolField<"sched-preds-closer", "true", "">;
defm : BoolField<"sched-retval-optimization", "true", "">;

// HexagonHardwareLoops.cpp
// Option to create preheader only for a specific function.
let Condition = AssertsOnly in
defm : ValueField<"hexagon-hwloop-phfn", "StringRef", "\"\"", "">;
// Option to create a preheader if one doesn't exist.
defm : BoolField<"hexagon-hwloop-preheader", "true",
  "Add a preheader to a hardware loop if one doesn't exist">;
let Condition = AssertsOnly in
defm : ValueField<"hexagon-max-hwloop", "int", "-1", "">;
// Turn it off by default. If a preheader block is not created here, the
// software pipeliner may be unable to find a block suitable to serve as
// a preheader. In that case SWP will not run.
defm : BoolField<"hwloop-spec-preheader", "false",
  "Allow speculation of preheader instructions">;

// HexagonEarlyIfConv.cpp
defm : BoolField<"enable-hexagon-br-prob", "true",
  "Enable branch probability info">;
defm : ValueField<"eif-limit", "unsigned", "6",
  "Size limit in Hexagon early if-conversion">;
defm : BoolField<"eif-loop-exit", "true",
  "Convert branches that may exit the loop">;

// HexagonConstExtenders.cpp
defm : OptionalValueField<"hexagon-cext-limit", "unsigned",
  "Maximum number of replacements">;
defm : ValueField<"hexagon-cext-threshold", "unsigned", "3",
  "Minimum number of extenders to trigger replacement">;

// HexagonPeephole.cpp
defm : BoolField<"enable-hexagon-peephole", "true",
  "Enable Peephole Optimization">;
defm : BoolField<"hexagon-fuse-intrinsic-vminub", "true",
  "Fuse A2_vminub/C2_cmpgtup into A6_vminub_RdP">;
defm : BoolField<"hexagon-opt-ext-to-64", "false",
  "Enable Optimization of extensions to i64">;
defm : BoolField<"hexagon-optszext", "false",
  "Enable Optimization of Sign/Zero Extends">;
defm : BoolField<"hexagon-pnotp", "true",
  "Enable Optimization of PNotP">;

// HexagonRegisterInfo.cpp
defm : ValueField<"hexagon-frame-index-reuse-limit", "unsigned", "~0U",
  "Limit on the number of reused registers in frame index elimination">;
defm : ValueField<"hexagon-frame-index-search-range", "unsigned", "32",
  "Limit on instruction search range in frame index elimination">;

// TODO: Remove the old spellings
def : AliasOf<"disable-hexagon-dealloc-ret", "hexagon-dealloc-ret=false">;
def : AliasOf<"disable-hexagon-misched", "hexagon-misched=false">;
def : AliasOf<"disable-hexagon-opt-ext-to-64", "hexagon-opt-ext-to-64=false">;
def : AliasOf<"disable-hexagon-optszext", "hexagon-optszext=false">;
def : AliasOf<"disable-hexagon-peephole", "enable-hexagon-peephole=false">;
def : AliasOf<"disable-hexagon-pnotp", "hexagon-pnotp=false">;
def : AliasOf<"eif-no-loop-exit", "eif-loop-exit=false">;
def : AliasOf<"hexagon-disable-args-min-alignment", "hexagon-args-min-alignment=false">;
