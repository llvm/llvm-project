//- Metal.td - Describe the Metal Target Machine -------------*- tablegen -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is a target description file for the Metal (Apple AIR) target.
///
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"
include "MetalStubs.td"
include "MetalAIRIntrinsicMappings.td"

defm : RemapAllTargetPseudoPointerOperands<AIRClass>;

def MetalInstrInfo : InstrInfo;

def : ProcessorModel<"generic", NoSchedModel, []>;

def MetalAsmParser : AsmParser {
  let ShouldEmitMatchRegisterName = 0;
}

def MetalAsmWriter : AsmWriter {
  string AsmWriterClassName = "InstPrinter";
  int PassSubtarget = 0;
  int Variant = 0;
  bit isMCAsmWriter = 1;
}

def Metal : Target {
  let InstructionSet = MetalInstrInfo;
  let AssemblyParsers = [MetalAsmParser];
  let AssemblyWriters = [MetalAsmWriter];
}
