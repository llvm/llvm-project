//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def InstCombineCLOptions : OptionsStruct<prefix = "instcombine-">;

defm : BoolField<"instcombine-code-sinking", "1",
  "Enable code sinking">;
defm : ValueField<"instcombine-guard-widening-window", "unsigned", "3",
  "How wide an instruction window to bypass looking for another guard">;
// FIXME: Remove this flag when it is no longer necessary to convert
// llvm.dbg.declare to avoid inaccurate debug info. Setting this to false
// increases variable availability at the cost of accuracy. Variables that
// cannot be promoted by mem2reg or SROA will be described as living in memory
// for their entire lifetime. However, passes like DSE and instcombine can
// delete stores to the alloca, leading to misleading and inaccurate debug
// information. This flag can be removed when those passes are fixed.
defm : BoolField<"instcombine-lower-dbg-declare", "1",
  "">;
defm : ValueField<"instcombine-max-allocsite-removable-users", "unsigned", "2048",
  "Maximum number of users to visit in alloc-site removability analysis">;
defm : ValueField<"instcombine-max-copied-from-constant-users", "unsigned", "300",
  "Maximum users to visit in copy from constant transform">;
defm : ValueField<"instcombine-max-num-phis", "unsigned", "512",
  "Maximum number phis to handle in intptr/ptrint folding">;
defm : ValueField<"instcombine-max-sink-users", "unsigned", "32",
  "Maximum number of undroppable users for instruction sinking">;
defm : ValueField<"instcombine-maxarray-size", "unsigned", "1024",
  "Maximum array size considered when doing a combine">;
defm : BoolField<"instcombine-negator-enabled", "1",
  "Should we attempt to sink negations?">;
// As a default, let's assume that we want to be aggressive, and attempt to
// traverse with no limits in attempt to sink negation.
defm : ValueField<"instcombine-negator-max-depth", "unsigned", "~0U",
  "What is the maximal lookup depth when trying to check for viability of negation sinking">;
defm : ValueField<"instcombine-simplify-vector-elts-depth", "unsigned", "10",
  "Depth limit when simplifying vector instructions and their operands">;
defm : BoolField<"instcombine-verify-known-bits", "0",
  "Verify that computeKnownBits() and SimplifyDemandedBits() are consistent">;
