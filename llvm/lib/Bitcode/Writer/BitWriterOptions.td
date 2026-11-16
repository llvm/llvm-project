//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def BitWriterOptions : OptionsStruct;

defm : ValueField<"bitcode-flush-threshold", "uint32_t", "512",
  "The threshold (unit M) for flushing LLVM bitcode">;
defm : ValueField<"bitcode-mdindex-threshold", "unsigned", "25",
  "Number of metadatas above which we emit an index to enable lazy-loading">;
defm : OptionalBoolField<"preserve-bc-uselistorder",
  "Preserve use-list order when writing LLVM bitcode">;
