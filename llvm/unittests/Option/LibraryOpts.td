//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

def TestLibraryOptions : OptionsStruct<"test">;

defm : BoolField<"lib-enable", "0", "A bool">;
let Hidden = 0 in
defm : ValueField<"lib-count", "unsigned", "3", "An unsigned">;
defm : ValueField<"lib-ratio", "double", "0.5", "A double">;
defm Path : ValueField<"lib-path", "StringRef", "\"p\"", "A string">;
