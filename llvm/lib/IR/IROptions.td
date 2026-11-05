//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def IROptions : OptionsStruct;

// AsmWriter.cpp
// See https://llvm.org/docs/DebuggingLLVM.html for why these flags are useful.
defm : OptionalBoolField<"preserve-ll-uselistorder",
  "Preserve use-list order when writing LLVM assembly">;
defm : BoolField<"print-addrspace-name", "false",
  "Print address space names">;
defm : BoolField<"print-inst-addrs", "false",
  "Print addresses of instructions when dumping">;
defm : BoolField<"print-inst-debug-locs", "false",
  "Pretty print debug locations of instructions when dumping">;
defm : BoolField<"print-prof-data", "false",
  "Pretty print perf data (branch weights, etc) when dumping">;
defm : BoolField<"print-single-flmd-line", "false",
  "Print the full contents of a function-local context on a single line">;

// AutoUpgrade.cpp
defm : BoolField<"auto-upgrade-debug-info", "true",
  "Autoupgrade debug info">;

// Constants.cpp
// As set of temporary options to help migrate how splats are represented.
defm : BoolField<"use-constant-int-for-fixed-length-splat", "false",
  "Use ConstantInt's native fixed-length vector splat support">;
defm : BoolField<"use-constant-int-for-scalable-splat", "false",
  "Use ConstantInt's native scalable vector splat support">;

// DebugInfoMetadata.cpp
// When true, preserves line and column number by picking one of the merged
// location info in a deterministic manner to assist sample based PGO.
defm : BoolField<"pick-merged-source-locations", "false",
  "Preserve line and column number when merging locations">;

// Function.cpp
defm : ValueField<"non-global-value-max-name-size", "int", "1024",
  "Maximum size for the name of non-global values">;

// Instructions.cpp
defm : BoolField<"i2p-p2i-opt", "true",
  "Enable inttoptr/ptrtoint roundtrip optimization">;

// ProfDataUtils.cpp
defm : BoolField<"elide-all-zero-branch-weights", "ElideAllZeroBranchWeightsDefault", "">;

// SafepointIRVerifier.cpp
// This option is used for writing test cases. Instead of crashing the program
// when verification fails, report a message to the console (for FileCheck
// usage) and continue execution as if nothing happened.
defm : BoolField<"safepoint-ir-verifier-print-only", "false", "">;

// Verifier.cpp
defm : BoolField<"verify-noalias-scope-decl-dom", "false",
  "Ensure that llvm.experimental.noalias.scope.decl for identical scopes are not dominating">;

// TODO: Remove the old spellings after 2027-04.
def : AliasOf<"disable-auto-upgrade-debug-info", "auto-upgrade-debug-info=false">;
def : AliasOf<"disable-i2p-p2i-opt", "i2p-p2i-opt=false">;
