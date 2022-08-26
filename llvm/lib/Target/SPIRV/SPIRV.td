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
include "SPIRVBuiltins.td"

def SPIRVInstrInfo : InstrInfo;

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

def SPIRV10 : SubtargetFeature<"spirv1.0", "SPIRVVersion", "10",
                             "Use SPIR-V version 1.0">;
def SPIRV11 : SubtargetFeature<"spirv1.1", "SPIRVVersion", "11",
                             "Use SPIR-V version 1.1">;
def SPIRV12 : SubtargetFeature<"spirv1.2", "SPIRVVersion", "12",
                             "Use SPIR-V version 1.2">;
def SPIRV13 : SubtargetFeature<"spirv1.3", "SPIRVVersion", "13",
                             "Use SPIR-V version 1.3">;
def SPIRV14 : SubtargetFeature<"spirv1.4", "SPIRVVersion", "14",
                             "Use SPIR-V version 1.4">;
def SPIRV15 : SubtargetFeature<"spirv1.5", "SPIRVVersion", "15",
                             "Use SPIR-V version 1.5">;

def SPIRVInstPrinter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

def SPIRV : Target {
  let InstructionSet = SPIRVInstrInfo;
  let AssemblyWriters = [SPIRVInstPrinter];
}
