//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

def CGDataOptions : OptionsStruct;

defm CodeGenDataGenerate : BoolField<"codegen-data-generate", "false",
  "Emit CodeGen Data into custom sections">;
defm CodeGenDataThinLTOTwoRounds : BoolField<"codegen-data-thinlto-two-rounds",
  "false", "Enable two-round ThinLTO code generation. The first round emits "
  "codegen data, while the second round uses the emitted codegen data for "
  "further optimizations.">;
defm CodeGenDataUsePath : ValueField<"codegen-data-use-path", "StringRef",
  "", "File path to where .cgdata file is read">;
defm GlobalMergingCallOverhead : ValueField<"global-merging-call-overhead",
  "double", "1.0", "The overhead cost associated with each function call when "
  "merging functions.">;
defm GlobalMergingExtraThreshold : ValueField<"global-merging-extra-threshold",
  "double", "0.0", "An additional cost threshold that must be exceeded for "
  "merging to be considered beneficial.">;
defm GlobalMergingInstOverhead : ValueField<"global-merging-inst-overhead",
  "double", "1.2", "The overhead cost associated with each instruction when "
  "lowering to machine instruction.">;
defm GlobalMergingMaxParams : ValueField<"global-merging-max-params",
  "unsigned", "std::numeric_limits<unsigned>::max()",
  "The maximum number of parameters allowed when merging functions.">;
defm GlobalMergingMinInstrs : ValueField<"global-merging-min-instrs",
  "unsigned", "1",
  "The minimum instruction count required when merging functions.">;
defm GlobalMergingMinMerges : ValueField<"global-merging-min-merges",
  "unsigned", "2", "Minimum number of similar functions with the same hash "
  "required for merging.">;
defm GlobalMergingParamOverhead : ValueField<"global-merging-param-overhead",
  "double", "2.0", "The overhead cost associated with each parameter when "
  "merging functions.">;
defm GlobalMergingSkipNoParams : BoolField<"global-merging-skip-no-params",
  "true", "Skip merging functions with no parameters.">;
defm IndexedCodeGenDataLazyLoading : BoolField<
  "indexed-codegen-data-lazy-loading", "false", "Lazily load indexed "
  "CodeGenData. Enable to save memory and time for final consumption of the "
  "indexed CodeGenData in production.">;
defm IndexedCodeGenDataReadFunctionMapNames : BoolField<
  "indexed-codegen-data-read-function-map-names", "true", "Read function map "
  "names in indexed CodeGenData. Can be disabled to save memory and time for "
  "final consumption of the indexed CodeGenData in production.">;
