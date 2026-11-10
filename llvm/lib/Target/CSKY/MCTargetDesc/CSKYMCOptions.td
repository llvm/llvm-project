//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def CSKYMCOptions : OptionsStruct<prefix = "csky-">;

// AsmParser/CSKYAsmParser.cpp
defm : BoolField<"csky-add-build-attributes", "true", "">;
defm : BoolField<"csky-asm-compressed-inst", "false",
  "Enable C-SKY asm compressed instruction">;

// MCTargetDesc/CSKYInstPrinter.cpp
defm : BoolField<"csky-arch-reg-names", "false",
  "Print architectural register names rather than the ABI names (such as r14 instead of sp)">;
defm : BoolField<"csky-no-aliases", "false",
  "Disable the emission of assembler pseudo instructions">;
