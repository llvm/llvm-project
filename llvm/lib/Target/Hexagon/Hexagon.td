//===-- Hexagon.td - Describe the Hexagon Target Machine --*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the top level entry point for the Hexagon target.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Hexagon Subtarget features.
//===----------------------------------------------------------------------===//

// Hexagon Archtectures
def ArchV2       : SubtargetFeature<"v2", "HexagonArchVersion", "V2",
                                    "Hexagon v2">;
def ArchV3       : SubtargetFeature<"v3", "HexagonArchVersion", "V3",
                                    "Hexagon v3">;
def ArchV4       : SubtargetFeature<"v4", "HexagonArchVersion", "V4",
                                    "Hexagon v4">;
def ArchV5       : SubtargetFeature<"v5", "HexagonArchVersion", "V5",
                                    "Hexagon v5">;

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//
include "HexagonSchedule.td"
include "HexagonRegisterInfo.td"
include "HexagonCallingConv.td"
include "HexagonInstrInfo.td"
include "HexagonIntrinsics.td"
include "HexagonIntrinsicsDerived.td"

def HexagonInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Hexagon processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, SchedMachineModel Model,
           list<SubtargetFeature> Features>
 : ProcessorModel<Name, Model, Features>;

def : Proc<"hexagonv2", HexagonModel,   [ArchV2]>;
def : Proc<"hexagonv3", HexagonModel,   [ArchV2, ArchV3]>;
def : Proc<"hexagonv4", HexagonModelV4, [ArchV2, ArchV3, ArchV4]>;
def : Proc<"hexagonv5", HexagonModelV4, [ArchV2, ArchV3, ArchV4, ArchV5]>;


// Hexagon Uses the MC printer for assembler output, so make sure the TableGen
// AsmWriter bits get associated with the correct class.
def HexagonAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def Hexagon : Target {
  // Pull in Instruction Info:
  let InstructionSet = HexagonInstrInfo;

  let AssemblyWriters = [HexagonAsmWriter];
}
