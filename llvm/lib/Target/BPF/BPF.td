//===-- BPF.td - Describe the BPF Target Machine -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

include "BPFRegisterInfo.td"
include "BPFCallingConv.td"
include "BPFInstrInfo.td"

def BPFInstrInfo : InstrInfo;

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

def BPFInstPrinter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

def BPF : Target {
  let InstructionSet = BPFInstrInfo;
  let AssemblyWriters = [BPFInstPrinter];
}
