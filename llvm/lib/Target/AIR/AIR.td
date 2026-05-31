//- AIR.td - Describe the AIR Target Machine -------------*- tablegen -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is a target description file for the AIR (Apple Intermediate Representation) target.
///
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"
include "AIRStubs.td"
include "AIRIntrinsicMappings.td"

defm : RemapAllTargetPseudoPointerOperands<AIRClass>;

def AIRInstrInfo : InstrInfo;

def : ProcessorModel<"generic", NoSchedModel, []>;

def AIRAsmParser : AsmParser {
  let ShouldEmitMatchRegisterName = 0;
}

def AIRAsmWriter : AsmWriter {
  string AsmWriterClassName = "InstPrinter";
  int PassSubtarget = 0;
  int Variant = 0;
  bit isMCAsmWriter = 1;
}

def AIR : Target {
  let InstructionSet = AIRInstrInfo;
  let AssemblyParsers = [AIRAsmParser];
  let AssemblyWriters = [AIRAsmWriter];
}
