//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def ByeOptions : OptionsStruct<"llvm">;

defm : BoolField<"wave-goodbye", "false", "wave good bye">;
defm : BoolField<"last-words", "false", "say last words (suppress codegen)">;
defm : ValueField<"bye-greeting", "StringRef", "", "also print this greeting">;
