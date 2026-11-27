//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def PPCMCOptions : OptionsStruct<prefix = "ppc-">;

// MCTargetDesc/PPCInstPrinter.cpp
// FIXME: Once the integrated assembler supports full register names, tie this
// to the verbose-asm setting.
defm : BoolField<"ppc-asm-full-reg-names", "false",
  "Use full register names when printing assembly">;
// Prints full register names with percent symbol.
defm : BoolField<"ppc-reg-with-percent-prefix", "false",
  "Prints full register names with percent">;
// Useful for testing purposes. Prints vs{31-63} as v{0-31} respectively.
defm : BoolField<"ppc-vsr-nums-as-vr", "false",
  "Prints full register names with vs{31-63} as v{0-31}">;
