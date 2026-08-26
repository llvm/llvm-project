//===-- PISA.td - Describe the PISA Target Machine ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"
include "llvm/Target/Target.td"

// Global setting
def Global {
  string NS = "PISA";
}

#define PISA

include "PISAFeatures.td"
include "PISARegisterInfo.td"
include "PISARegisterBanks.td"
include "PISAInstrInfo.td"
include "PISACombine.td"

defm : RemapAllTargetPseudoPointerOperands<Reg64b>;
def PISAInstrInfo : InstrInfo;

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

// valid -mcpu values
def : Proc<"100",   [Feature100]>;

def PISAInstPrinter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}


def PISA : Target {
  let InstructionSet = PISAInstrInfo;
  let AssemblyWriters = [PISAInstPrinter];
}
