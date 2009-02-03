//===- Alpha.td - Describe the Alpha Target Machine --------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing...
//
include "llvm/Target/Target.td"

//Alpha is little endian

//===----------------------------------------------------------------------===//
// Subtarget Features
//===----------------------------------------------------------------------===//

def FeatureCIX : SubtargetFeature<"cix", "HasCT", "true",
                                  "Enable CIX extentions">;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "AlphaRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Schedule Description
//===----------------------------------------------------------------------===//

include "AlphaSchedule.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "AlphaInstrInfo.td"

def AlphaInstrInfo : InstrInfo {
  // Define how we want to layout our target-specific information field.
 // let TSFlagsFields = [];
 // let TSFlagsShifts = [];
}

//===----------------------------------------------------------------------===//
// Alpha Processor Definitions
//===----------------------------------------------------------------------===//

def : Processor<"generic", Alpha21264Itineraries, []>;
def : Processor<"ev6"    , Alpha21264Itineraries, []>;
def : Processor<"ev67"   , Alpha21264Itineraries, [FeatureCIX]>;

//===----------------------------------------------------------------------===//
// The Alpha Target
//===----------------------------------------------------------------------===//


def Alpha : Target {
  // Pull in Instruction Info:
  let InstructionSet = AlphaInstrInfo;
}
