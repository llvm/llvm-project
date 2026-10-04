//===-- PISA.td - Describe the PISA Target Machine ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

include "PISAFeatures.td"
include "PISARegisterInfo.td"
include "PISAInstrInfo.td"

// Map the pointer-like operands of the target-independent pseudo instructions
// (STACKMAP, PATCHPOINT, ...) onto a concrete PISA register class.
defm : RemapAllTargetPseudoPointerOperands<Reg64b>;

def PISAInstrInfo : InstrInfo;

class Proc<string Name, list<SubtargetFeature> Features>
    : Processor<Name, NoItineraries, Features>;

// Valid -mcpu values.
def : Proc<"100", [Feature100]>;

def PISAInstPrinter : AsmWriter {
  string AsmWriterClassName = "InstPrinter";
  bit isMCAsmWriter = 1;
}

def PISA : Target {
  let InstructionSet = PISAInstrInfo;
  let AssemblyWriters = [PISAInstPrinter];
}
