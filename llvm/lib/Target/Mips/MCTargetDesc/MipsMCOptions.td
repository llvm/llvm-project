//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def MipsMCOptions : OptionsStruct<prefix = "mips-">;

// MCTargetDesc/MipsABIInfo.cpp
defm : BoolField<"mips-jalr-reloc", "true",
  "MIPS: Emit R_{MICRO}MIPS_JALR relocation with jalr">;
defm : BoolField<"mno-check-zero-division", "false",
  "MIPS: Don't trap on integer division by zero">;

// MCTargetDesc/MipsTargetStreamer.cpp
defm : BoolField<"mips-round-section-sizes", "false",
  "Round section sizes up to the section alignment">;
