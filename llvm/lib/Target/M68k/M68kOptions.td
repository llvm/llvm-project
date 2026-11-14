//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def M68kOptions : OptionsStruct<prefix = "m68k-">;

defm : BoolField<"m68k-use-base-pointer", "true",
  "Enable use of a base pointer for complex stack frames">;
