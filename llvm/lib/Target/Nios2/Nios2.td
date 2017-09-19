//===-- Nios2.td - Describe the Nios2 Target Machine -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"
include "Nios2RegisterInfo.td"
include "Nios2InstrInfo.td"
include "Nios2Schedule.td"

def Nios2InstrInfo : InstrInfo;

def Nios2 : Target { let InstructionSet = Nios2InstrInfo; }

//===----------------------------------------------------------------------===//
// Nios2 Subtarget features
//===----------------------------------------------------------------------===//
def FeatureNios2r1     : SubtargetFeature<"nios2r1", "Nios2ArchVersion", 
                                "Nios2r1", "Nios2 R1 ISA Support">;
def FeatureNios2r2     : SubtargetFeature<"nios2r2", "Nios2ArchVersion",                      
                               "Nios2r2", "Nios2 R2 ISA Support">;

//===----------------------------------------------------------------------===//
// Nios2 processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, Nios2GenericItineraries, Features>;

def : Proc<"nios2r1", [FeatureNios2r1]>;
def : Proc<"nios2r2", [FeatureNios2r2]>;
