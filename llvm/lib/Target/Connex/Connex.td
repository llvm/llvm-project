//===-- Connex.td - Describe the Connex Target Machine ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


// The processor system we define is a:
//   - scalar processor, which is basically an almost unmodified version of the
//     extended BPF (Berkeley Packet Filter) 64-bit RISC processor, implemented
//      in LLVM. eBPF is well-described at
//            https://www.kernel.org/doc/Documentation/networking/filter.txt.
//   - a vector/SIMD unit, which is the Connex-S processor, actually an
//              accelerator, with its own separate-space banked vector memory.
// The back end implementation starts from the BPF LLVM back end (the version
//    from Jul 2016), to which we add
//    the vector instruction support inspired mostly from the Mips MSA vector
//    extensions.
// Note: the scalar registers of Connex are R0-31 (e.g., Connex::R10), and the
//       vector registers are Wh0-31 (from wide; e.g., Connex::Wh0).
//
// VERY IMPORTANT: Note that currently our back end targets only our Connex
//   OPINCAA assembler (very easy to learn and use) available at
//   https://gitlab.dcae.pub.ro/research/ConnexRelated/opincaa/ .


include "llvm/Target/Target.td"
include "ConnexRegisterInfo.td"
include "ConnexCallingConv.td"
include "ConnexSchedule.td"
include "ConnexInstrInfo.td"


def ConnexInstrInfo : InstrInfo;

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

def ConnexInstPrinter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

// Inspired from https://github.com/llvm-mirror/llvm/commit/5ef1349
//   (see also https://groups.google.com/forum/#!topic/llvm-dev/_Zr4Oe5MLkE)
def ConnexAsmParser : AsmParser {
  bit HasMnemonicFirst = 0;
}

def ConnexAsmParserVariant : AsmParserVariant {
  int Variant = 0;
  string Name = "Connex";
  string BreakCharacters = ".";
}

def Connex : Target {
  let InstructionSet = ConnexInstrInfo;
  let AssemblyWriters = [ConnexInstPrinter];
  // Inspired from llvm/lib/Target/Hexagon/Hexagon.td
  let AssemblyParsers = [ConnexAsmParser];
  let AssemblyParserVariants = [ConnexAsmParserVariant];
}
