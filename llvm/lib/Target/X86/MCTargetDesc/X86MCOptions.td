//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def X86MCOptions : OptionsStruct<prefix = "x86-">;

// AsmParser/X86AsmParser.cpp
defm : BoolField<"x86-experimental-lvi-inline-asm-hardening", "false",
  "Harden inline assembly code that may be vulnerable to Load Value Injection (LVI). "
  "This feature is experimental">;

// MCTargetDesc/X86AsmBackend.cpp
defm : ValueField<"x86-align-branch", "std::optional<StringRef>", "std::nullopt",
  "Specify types of branches to align (plus separated list of types):"
  "\njcc      indicates conditional jumps"
  "\nfused    indicates fused conditional jumps"
  "\njmp      indicates direct unconditional jumps"
  "\ncall     indicates direct and indirect calls"
  "\nret      indicates rets"
  "\nindirect indicates indirect unconditional jumps">;
defm : ValueField<"x86-align-branch-boundary", "std::optional<unsigned>", "std::nullopt",
  "Control how the assembler should align branches with NOP. If the boundary's size is not 0, it should be a power of 2 and no less than 32. "
  "Branches will be aligned to prevent from being across or against the boundary of specified size. The default value 0 does not align branches">;
defm : BoolField<"x86-branches-within-32B-boundaries", "false",
  "Align selected instructions to mitigate negative performance impact of Intel's micro code update for errata skx102. "
  "May break assumptions about labels corresponding to particular instructions, and should be used with caution">;
defm : BoolField<"x86-pad-for-align", "false",
  "Pad previous instructions to implement align directives">;
defm : BoolField<"x86-pad-for-branch-align", "true",
  "Pad previous instructions to implement branch alignment">;
defm : ValueField<"x86-pad-max-prefix-size", "std::optional<unsigned>", "std::nullopt",
  "Maximum number of prefixes to use for padding">;

// MCTargetDesc/X86MCAsmInfo.cpp
defm : EnumField<"x86-asm-syntax", "X86::AsmWriterFlavorTy", "X86::ATT",
  "att,intel", ["X86::ATT", "X86::Intel"],
  "Select the assembly style for input">;
defm : BoolField<"mark-data-regions", "true",
  "Mark code section jump table data regions">;
