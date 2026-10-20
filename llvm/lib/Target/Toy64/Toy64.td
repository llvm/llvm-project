//===-- Toy64.td - Describe the Toy64 Target Machine -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Register definitions
//===----------------------------------------------------------------------===//

include "Toy64RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction information
//===----------------------------------------------------------------------===//

// We haven't defined any real instructions yet.
//
// We still need an InstrInfo object because Target requires an instruction
// set description.
def Toy64InstrInfo : InstrInfo {
  let guessInstructionProperties = false;
}

//===----------------------------------------------------------------------===//
// Target
//===----------------------------------------------------------------------===//

def Toy64 : Target {
  let InstructionSet = Toy64InstrInfo;
}