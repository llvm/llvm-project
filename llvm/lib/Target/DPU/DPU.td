//===-- DPU.td - Describe the DPU Target Machine -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Target-independent interfaces which we are implementing
include "llvm/Target/Target.td"

// Register File Description
include "DPURegisterInfo.td"

// Instruction Descriptions
include "DPUInstrInfo.td"

// Calling conventions
include "DPUCallingConv.td"

def DPUInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

def DPUAsmParser : AsmParser {
  let ShouldEmitMatchRegisterName = 1;
}

def DPUAsmParserVariant : AsmParserVariant {
  let Variant = 0;
  let RegisterPrefix = "%";
  let CommentDelimiter = "//";
}

// To generate assembler as output.
def DPUInstPrinter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

def DPU : Target {
  let InstructionSet = DPUInstrInfo;
  let AssemblyParsers = [DPUAsmParser];
  let AssemblyParserVariants = [DPUAsmParserVariant];
  let AssemblyWriters = [DPUInstPrinter];
}

def FeatureDPUNoSugar : SubtargetFeature<"nosugar", "noSugar", "false", "Disable the use of Syntactic Sugar Instruction">;
