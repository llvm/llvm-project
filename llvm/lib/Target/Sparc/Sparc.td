//===-- Sparc.td - Describe the Sparc Target Machine -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"
include "llvm/TableGen/SearchableTable.td"

include "SparcFeatures.td"

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "SparcASITags.td"
include "SparcPrefetchTags.td"
include "SparcRegisterInfo.td"
include "SparcCallingConv.td"
include "SparcSchedule.td"
include "SparcInstrInfo.td"
include "SparcSchedLeon.td"

defm : RemapAllTargetPseudoPointerOperands<sparc_ptr_rc>;

def SparcInstrInfo : InstrInfo;

def SparcAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = true;
  let AllowDuplicateRegisterNames = true;
}

def SparcAsmParserVariant : AsmParserVariant {
  let RegisterPrefix = "%";
}

include "SparcProcessors.td"

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def SparcAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  int PassSubtarget = 1;
  int Variant = 0;
}

def Sparc : Target {
  // Pull in Instruction Info:
  let InstructionSet = SparcInstrInfo;
  let AssemblyParsers  = [SparcAsmParser];
  let AssemblyParserVariants = [SparcAsmParserVariant];
  let AssemblyWriters = [SparcAsmWriter];
  let AllowRegisterRenaming = 1;
}
