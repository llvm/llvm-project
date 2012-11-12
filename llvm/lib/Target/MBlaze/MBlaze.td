//===-- MBlaze.td - Describe the MBlaze Target Machine -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the MBlaze target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "MBlazeRegisterInfo.td"
include "MBlazeSchedule.td"
include "MBlazeIntrinsics.td"
include "MBlazeInstrInfo.td"
include "MBlazeCallingConv.td"

def MBlazeInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Microblaze Subtarget features                                              //
//===----------------------------------------------------------------------===//

def FeatureBarrel      : SubtargetFeature<"barrel", "HasBarrel", "true",
                                "Implements barrel shifter">;
def FeatureDiv         : SubtargetFeature<"div", "HasDiv", "true",
                                "Implements hardware divider">;
def FeatureMul         : SubtargetFeature<"mul", "HasMul", "true",
                                "Implements hardware multiplier">;
def FeaturePatCmp      : SubtargetFeature<"patcmp", "HasPatCmp", "true",
                                "Implements pattern compare instruction">;
def FeatureFPU         : SubtargetFeature<"fpu", "HasFPU", "true",
                                "Implements floating point unit">;
def FeatureMul64       : SubtargetFeature<"mul64", "HasMul64", "true",
                                "Implements multiplier with 64-bit result">;
def FeatureSqrt        : SubtargetFeature<"sqrt", "HasSqrt", "true",
                                "Implements sqrt and floating point convert">;

//===----------------------------------------------------------------------===//
// MBlaze processors supported.
//===----------------------------------------------------------------------===//

def : Processor<"mblaze",  NoItineraries, []>;
def : Processor<"mblaze3", MBlazePipe3Itineraries, []>;
def : Processor<"mblaze5", MBlazePipe5Itineraries, []>;

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

def MBlazeAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def MBlaze : Target {
  let InstructionSet = MBlazeInstrInfo;
  let AssemblyWriters = [MBlazeAsmWriter];
}
