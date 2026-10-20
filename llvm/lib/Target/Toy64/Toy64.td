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

// Map generic LLVM pointer pseudo operands onto Toy64's GPR class.
defm : RemapAllTargetPseudoPointerOperands<GPR>;

//===----------------------------------------------------------------------===//
// Processor definitions
//===----------------------------------------------------------------------===//

// Toy64 currently has one generic processor.
//
// NoSchedModel means that we have not yet described pipeline resources,
// latencies, issue width, etc.
//
// The empty feature list means the generic processor has no optional
// subtarget features yet.
def : ProcessorModel<"generic", NoSchedModel, []>;

//===----------------------------------------------------------------------===//
// Target
//===----------------------------------------------------------------------===//

def Toy64InstrInfo : InstrInfo {
  let guessInstructionProperties = false;
}

def Toy64 : Target {
  let InstructionSet = Toy64InstrInfo;
}
