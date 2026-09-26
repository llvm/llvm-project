//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

def CGDataOptions : OptionsStruct;

defm : BoolField<"codegen-data-generate", "0",
  "Emit CodeGen Data into custom sections">;
defm : BoolField<"codegen-data-thinlto-two-rounds", "0",
  "Enable two-round ThinLTO code generation. The first round emits "
  "codegen data, while the second round uses the emitted codegen data for "
  "further optimizations.">;
defm : ValueField<"codegen-data-use-path", "StringRef", "",
  "File path to where .cgdata file is read">;
defm : ValueField<"global-merging-call-overhead", "double", "1.0",
  "The overhead cost associated with each function call when merging "
  "functions.">;
defm : ValueField<"global-merging-extra-threshold", "double", "0.0",
  "An additional cost threshold that must be exceeded for merging to be "
  "considered beneficial.">;
defm : ValueField<"global-merging-inst-overhead", "double", "1.2",
  "The overhead cost associated with each instruction when lowering to "
  "machine instruction.">;
defm : ValueField<"global-merging-max-params", "unsigned",
  "std::numeric_limits<unsigned>::max()",
  "The maximum number of parameters allowed when merging functions.">;
defm : ValueField<"global-merging-min-instrs", "unsigned", "1",
  "The minimum instruction count required when merging functions.">;
defm : ValueField<"global-merging-min-merges", "unsigned", "2",
  "Minimum number of similar functions with the same hash required for "
  "merging.">;
defm : ValueField<"global-merging-param-overhead", "double", "2.0",
  "The overhead cost associated with each parameter when merging "
  "functions.">;
defm : BoolField<"global-merging-skip-no-params", "1",
  "Skip merging functions with no parameters.">;
defm : BoolField<"indexed-codegen-data-lazy-loading", "0",
  "Lazily load indexed CodeGenData. Enable to save memory and time for final "
  "consumption of the indexed CodeGenData in production.">;
defm : BoolField<"indexed-codegen-data-read-function-map-names", "1",
  "Read function map names in indexed CodeGenData. Can be disabled to save "
  "memory and time for final consumption of the indexed CodeGenData in "
  "production.">;
