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

def BPFAsmParser : AsmParser {
  bit HasMnemonicFirst = 0;
}

def BPFAsmParserVariant : AsmParserVariant {
  int Variant = 0;
  string Name = "BPF";
  string BreakCharacters = ".";
  string TokenizingCharacters = "#()[]=:.<>!+*";
}

def BPF : Target {
  let InstructionSet = BPFInstrInfo;
  let AssemblyWriters = [BPFInstPrinter];
  let AssemblyParsers = [BPFAsmParser];
  let AssemblyParserVariants = [BPFAsmParserVariant];
}
