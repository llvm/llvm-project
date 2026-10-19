//===-- AGX.td - Apple AGX GPU target definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top level target description of the Apple AGX GPUs (G13 and later). See
// llvm/docs/AGXUsage.rst.
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

include "AGXFeatures.td"
include "AGXRegisterInfo.td"
include "AGXSchedule.td"
include "AGXProcessors.td"
include "AGXInstrInfo.td"
include "AGXCallingConv.td"

// Standard pseudo instructions that take a pointer use 64-bit register pairs.
defm : RemapAllTargetPseudoPointerOperands<GPR64>;

def AGXInstrInfo : InstrInfo {
  let guessInstructionProperties = 1;
}

def AGXAsmParser : AsmParser {
  let ShouldEmitMatchRegisterName = 0;
}

def AGXAsmParserVariant : AsmParserVariant {
  let Variant = 0;
  let Name = "";
  let TokenizingCharacters = "";
  let SeparatorCharacters = " \t,";
}

def AGXAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def AGX : Target {
  let InstructionSet = AGXInstrInfo;
  let AssemblyParsers = [AGXAsmParser];
  let AssemblyParserVariants = [AGXAsmParserVariant];
  let AssemblyWriters = [AGXAsmWriter];
  let AllowRegisterRenaming = 1;
}
