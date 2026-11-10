//===-- SPIRV.td - Describe the SPIR-V Target Machine ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

include "SPIRVRegisterInfo.td"
include "SPIRVRegisterBanks.td"
include "SPIRVInstrInfo.td"
include "SPIRVCombine.td"
include "SPIRVBuiltins.td"

defm : RemapAllTargetPseudoPointerOperands<pID>;

def SPIRVInstrInfo : InstrInfo;

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

defvar IntelIGCATargets = [
  "igca_10", "igca_10r",
  "igca_15", "igca_15r", "igca_15ra",
  "igca_20", "igca_20c", "igca_20ca",
  "igca_30", "igca_30r",
  "igca_35", "igca_35r", "igca_35ra",
  "igca_40", "igca_40r",
  "igca_50", "igca_50r",
  "igca_60", "igca_60r",
];
foreach igca = IntelIGCATargets in
  def : Proc<igca, []>;

def SPIRVInstPrinter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

def SPIRV : Target {
  let InstructionSet = SPIRVInstrInfo;
  let AssemblyWriters = [SPIRVInstPrinter];
}
