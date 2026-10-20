//===-- Toy64.td - Describe the Toy64 Target Machine -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Registers
//===----------------------------------------------------------------------===//

include "Toy64RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

include "Toy64InstrInfo.td"

// Map the target-independent pseudo-instruction pointer register class
// (ptr_rc) to Toy64's 64-bit general-purpose register class.
//
// Toy64 has 64-bit pointers, and all pointers are held in GPRs.
defm : RemapAllTargetPseudoPointerOperands<GPR>;

//===----------------------------------------------------------------------===//
// Target
//===----------------------------------------------------------------------===//

def Toy64InstrInfo : InstrInfo {
  let guessInstructionProperties = false;
}

def Toy64 : Target {
  let InstructionSet = Toy64InstrInfo;
}