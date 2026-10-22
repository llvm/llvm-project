//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def MipsOptions : OptionsStruct<prefix = "mips-">;

// Mips16ISelLowering.cpp
defm : BoolField<"mips16-dont-expand-cond-pseudo", "false",
  "Don't expand conditional move related pseudos for Mips 16">;

// MipsBranchExpansion.cpp
defm : BoolField<"force-mips-long-branch", "false",
  "MIPS: Expand all branches to long format">;
defm : BoolField<"skip-mips-long-branch", "false",
  "MIPS: Skip branch expansion pass">;

// MipsConstantIslandPass.cpp
// FIXME: This option should be removed once it has received sufficient testing.
defm : BoolField<"mips-align-constant-islands", "true",
  "Align constant islands in code">;
// For testing purposes we tell it to not use relaxed load forms so that it will
// split blocks.
defm : BoolField<"mips-constant-islands-no-load-relaxation", "false",
  "Don't relax loads to long loads - for testing purposes">;
// Rather than do make check tests with huge amounts of code, we force the test
// to use this amount.
defm : ValueField<"mips-constant-islands-small-offset", "int", "0",
  "Make small offsets be this amount for testing purposes">;

// MipsDelaySlotFiller.cpp
defm : BoolField<"disable-mips-delay-filler", "false",
  "Fill all delay slots with NOPs">;
defm : BoolField<"disable-mips-df-backward-search", "false",
  "Disallow MIPS delay filler to search backward">;
defm : BoolField<"disable-mips-df-forward-search", "true",
  "Disallow MIPS delay filler to search forward">;
defm : BoolField<"disable-mips-df-succbb-search", "true",
  "Disallow MIPS delay filler to search successor basic blocks">;

// MipsISelLowering.cpp
defm : BoolField<"mips-tail-calls", "false",
  "MIPS: permit tail calls">;

// MipsOptimizePICCall.cpp
defm : BoolField<"mips-erase-gp-opnd", "true",
  "Erase GP Operand">;
defm : BoolField<"mips-load-target-from-got", "true",
  "Load target address from GOT">;

// MipsOs16.cpp
defm : ValueField<"mips32-function-mask", "StringRef", "",
  "Force function to be mips32">;

// MipsSEISelLowering.cpp
defm : BoolField<"mno-ldc1-sdc1", "false",
  "Expand double precision loads and stores to their single precision counterparts">;

// MipsSubtarget.cpp
defm : BoolField<"mgpopt", "false",
  "Enable gp-relative addressing of mips small data items">;
defm : EnumField<"mips-compact-branches", "CompactBranchPolicy", "CompactBranchPolicy::Optimal",
  "never,optimal,always", ["CompactBranchPolicy::Never", "CompactBranchPolicy::Optimal", "CompactBranchPolicy::Always"],
  "MIPS Specific: Compact branch policy">;
defm : BoolField<"mips16-constant-islands", "true",
  "Enable mips16 constant islands">;
defm : BoolField<"mips16-hard-float", "false",
  "Enable mips16 hard float">;

// MipsTargetMachine.cpp
defm : BoolField<"mfix4300", "false",
  "Enable the VR4300 mulmul bug fix">;
defm : BoolField<"mips-os16", "false",
  "Compile all functions that don't use floating point as Mips 16">;

// MipsTargetObjectFile.cpp
defm : BoolField<"membedded-data", "false",
  "MIPS: Try to allocate variables in the following sections if possible: .rodata, .sdata, .data">;
defm : BoolField<"mextern-sdata", "true",
  "MIPS: Use gp_rel for data that is not defined by the current object">;
defm : ValueField<"mips-ssection-threshold", "unsigned", "8",
  "Small data and bss section threshold size">;
defm : BoolField<"mlocal-sdata", "true",
  "MIPS: Use gp_rel for object-local data">;
