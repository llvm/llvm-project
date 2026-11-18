//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def DoublerOptions : OptionsStruct<"llvm", prefix = "doubler-">;

defm : ValueField<"doubler-factor", "unsigned", "2",
  "Multiply the initializer by this factor">;
defm : ValueField<"doubler-pass-name", "StringRef", "\"doubler-pass\"",
  "Name of the pass in pipeline descriptions">;
