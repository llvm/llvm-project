//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def MSP430Options : OptionsStruct<prefix = "msp430-">;

// MSP430BranchSelector.cpp
defm : BoolField<"msp430-branch-select", "true",
  "Expand out of range branches">;

// MSP430ISelLowering.cpp
defm : BoolField<"msp430-no-legal-immediate", "false",
  "Enable non legal immediates (for testing purposes only)">;

// MSP430Subtarget.cpp
defm : EnumField<"msp430-hwmult", "MSP430::HWMult", "MSP430::HWMult::None",
  "none,16bit,32bit,f5series", ["MSP430::HWMult::None", "MSP430::HWMult::HW16", "MSP430::HWMult::HW32", "MSP430::HWMult::F5"],
  "Hardware multiplier use mode for MSP430">;

// TODO: Remove the old spellings after 2027-04.
def : AliasOf<"mhwmult", "msp430-hwmult">;
