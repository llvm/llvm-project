//===- PIC16.td - Describe the PIC16 Target Machine -----------*- tblgen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the PIC16 target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

include "PIC16RegisterInfo.td"
include "PIC16InstrInfo.td"

//===----------------------------------------------------------------------===//
// Subtarget Features. 
//===----------------------------------------------------------------------===//
def FeatureCooper : SubtargetFeature<"cooper", "IsCooper", "true",
                                     "PIC16 Cooper ISA Support">;

//===----------------------------------------------------------------------===//
// PIC16 supported processors.
//===----------------------------------------------------------------------===//

def : Processor<"generic", NoItineraries, []>;
def : Processor<"cooper", NoItineraries, [FeatureCooper]>;


def PIC16InstrInfo : InstrInfo {} 

def PIC16 : Target {
  let InstructionSet = PIC16InstrInfo;
}

