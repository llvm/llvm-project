//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def ARMOptions : OptionsStruct<prefix = "arm-">;

// ARMCallLowering.cpp
// Whether Big-endian GISel is enabled, defaults to off, can be enabled for
// testing.
defm : BoolField<"arm-gisel-bigendian", "false",
  "Enable Global-ISel Big Endian Lowering">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"enable-arm-gisel-bigendian", "arm-gisel-bigendian">;

// ARMConstantIslandPass.cpp
defm : BoolField<"arm-adjust-jump-tables", "true",
  "Adjust basic block layout to better use TB[BH]">;
defm : ValueField<"arm-constant-island-max-iteration", "unsigned", "30",
  "The max number of iteration for converge">;
defm : BoolField<"arm-synthesize-thumb-1-tbb", "true",
  "Use compressed jump tables in Thumb-1 by synthesizing an equivalent to the TBB/TBH instructions">;

// ARMExpandPseudoInsts.cpp
defm : BoolField<"verify-arm-pseudo-expand", "false",
  "Verify machine code after expanding ARM pseudos">;

// ARMFrameLowering.cpp
defm : BoolField<"align-neon-spills", "true",
  "Align ARM NEON spills in prolog and epilog">;

// ARMHazardRecognizer.cpp
defm : OptionalBoolField<"arm-assume-itcm-bankconflict", "">;
defm : OptionalValueField<"arm-data-bank-mask", "int", "">;

// ARMISelDAGToDAG.cpp
defm : BoolField<"arm-shifter-op", "true",
  "Enable isel of shifter-op">;

// ARMISelLowering.cpp
defm : BoolField<"arm-interworking", "true",
  "Enable / disable ARM interworking (for debugging only)">;
defm : ValueField<"arm-max-base-updates-to-check", "unsigned", "64",
  "Maximum number of base-updates to check generating postindex">;
// FIXME: set to true by default once PR32780 is fixed
defm : BoolField<"arm-promote-constant", "false",
  "Enable / disable promotion of unnamed_addr constants into constant pools">;
defm : ValueField<"arm-promote-constant-max-size", "unsigned", "64",
  "Maximum size of constant to promote into a constant pool">;
defm : ValueField<"arm-promote-constant-max-total", "unsigned", "128",
  "Maximum size of ALL constants to promote into a constant pool">;
defm : ValueField<"mve-max-interleave-factor", "unsigned", "2",
  "Maximum interleave factor for MVE VLDn to generate">;

// ARMLoadStoreOptimizer.cpp
// This switch disables formation of double/multi instructions that could
// potentially lead to (new) alignment traps even with CCR.UNALIGN_TRP
// disabled. This can be used to create libraries that are robust even when
// users provoke undefined behaviour by supplying misaligned pointers. See
// mayCombineMisaligned().
defm : BoolField<"arm-assume-misaligned-load-store", "false",
  "Be more conservative in ARM load/store opt">;
// Limit the number of instructions to be rescheduled.
// FIXME: tune this limit, and/or come up with some better heuristics.
defm : ValueField<"arm-prera-ldst-opt-reorder-limit", "unsigned", "8", "">;

// ARMLowOverheadLoops.cpp
defm : BoolField<"arm-loloops-tailpred", "true",
  "Enable tail-predication in the ARM LowOverheadLoop pass">;
defm : BoolField<"arm-omit-dls", "true",
  "Enable omitting 'dls lr, lr' instructions">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"arm-loloops-disable-tailpred", "arm-loloops-tailpred=false">;

// ARMParallelDSP.cpp
defm : ValueField<"arm-parallel-dsp-load-limit", "unsigned", "16",
  "Limit the number of loads analysed">;
defm : BoolField<"disable-arm-parallel-dsp", "false",
  "Disable the ARM Parallel DSP pass">;

// ARMSelectionDAGInfo.cpp
defm : EnumField<"arm-memtransfer-tploop", "TPLoopMemTransfer", "TPLoopMemTransfer::ForceDisabled",
  "force-disabled,force-enabled,allow", ["TPLoopMemTransfer::ForceDisabled", "TPLoopMemTransfer::ForceEnabled", "TPLoopMemTransfer::Allow"],
  "Control conversion of memcpy to Tail predicated loops (WLSTP)">;

// ARMSubtarget.cpp
// Use the fast-isel, even for subtargets where it is not currently supported
// (for testing only).
defm : BoolField<"arm-force-fast-isel", "false", "">;
defm : BoolField<"arm-restrict-it", "false",
  "Disallow complex IT blocks">;
defm : BoolField<"arm-use-mulops", "true", "">;
def : AliasOf<"arm-default-it", "arm-restrict-it=false">;

// ARMTargetMachine.cpp
defm : BoolField<"arm-a15-sd-optimization", "true",
  "Enable optimization of S->D register accesses on A15">;
defm : BoolField<"arm-atomic-cfg-tidy", "true",
  "Run SimplifyCFG after expanding atomic operations to make use of cmpxchg flow-based information">;
// FIXME: Unify control over GlobalMerge.
defm : OptionalBoolField<"arm-global-merge",
  "Enable the global merge pass">;
defm : BoolField<"arm-load-store-opt", "true",
  "Enable ARM load/store optimization pass">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"disable-a15-sd-optimization", "arm-a15-sd-optimization=false">;

// ARMTargetTransformInfo.cpp
defm : BoolField<"arm-allow-wlsloops", "true",
  "Enable the generation of WLS loops">;
defm : ValueField<"arm-force-unroll-threshold", "int", "12",
  "Threshold for forced unrolling of small loops in Arm architecture">;
defm : BoolField<"arm-loloops", "true",
  "Enable the generation of low-overhead loops">;
defm : BoolField<"arm-maskedldst", "true",
  "Enable the generation of masked loads and stores">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"disable-arm-loloops", "arm-loloops=false">;
def : AliasOf<"enable-arm-maskedldst", "arm-maskedldst">;

// MLxExpansionPass.cpp
defm : BoolField<"arm-expand-all-fp-mlx", "false", "">;
defm : ValueField<"arm-expand-limit", "unsigned", "~0U", "">;

// MVEGatherScatterLowering.cpp
defm : BoolField<"arm-maskedgatscat", "true",
  "Enable the generation of masked gathers and scatters">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"enable-arm-maskedgatscat", "arm-maskedgatscat">;

// MVELaneInterleavingPass.cpp
defm : BoolField<"mve-interleave", "true",
  "Enable interleave MVE vector operation lowering">;

// MVETPAndVPTOptimisationsPass.cpp
defm : BoolField<"arm-merge-loopenddec", "true",
  "Enable merging Loop End and Dec instructions">;
defm : BoolField<"arm-set-lr-predicate", "true",
  "Enable setting lr as a predicate in tail predication regions">;

// MVETailPredication.cpp
defm : EnumField<"tail-predication", "TailPredication", "TailPredication::Enabled",
  "disabled,enabled-no-reductions,enabled,force-enabled-no-reductions,force-enabled",
  ["TailPredication::Disabled", "TailPredication::EnabledNoReductions", "TailPredication::Enabled", "TailPredication::ForceEnabledNoReductions", "TailPredication::ForceEnabled"],
  "MVE tail-predication pass options">;

// Thumb2InstrInfo.cpp
defm : BoolField<"arm-prefer-no-csel", "false",
  "Prefer predicated Move to CSEL">;

// Thumb2SizeReduction.cpp
defm : ValueField<"t2-reduce-limit", "int", "-1", "">;
defm : ValueField<"t2-reduce-limit2", "int", "-1", "">;
defm : ValueField<"t2-reduce-limit3", "int", "-1", "">;
