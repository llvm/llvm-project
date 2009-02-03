//===- XCore.td - Describe the XCore Target Machine --------*- tablegen -*-===//
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

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Descriptions
//===----------------------------------------------------------------------===//

include "XCoreRegisterInfo.td"
include "XCoreInstrInfo.td"
include "XCoreCallingConv.td"

def XCoreInstrInfo : InstrInfo {
  let TSFlagsFields = [];
  let TSFlagsShifts = [];
}

//===----------------------------------------------------------------------===//
// XCore Subtarget features.
//===----------------------------------------------------------------------===//

def FeatureXS1A
  : SubtargetFeature<"xs1a", "IsXS1A", "true",
                     "Enable XS1A instructions">;

def FeatureXS1B
  : SubtargetFeature<"xs1b", "IsXS1B", "true",
                     "Enable XS1B instructions">;

//===----------------------------------------------------------------------===//
// XCore processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic",      [FeatureXS1A]>;
def : Proc<"xs1a-generic", [FeatureXS1A]>;
def : Proc<"xs1b-generic", [FeatureXS1B]>;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def XCore : Target {
  // Pull in Instruction Info:
  let InstructionSet = XCoreInstrInfo;
}
