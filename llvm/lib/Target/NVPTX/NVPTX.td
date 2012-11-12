//===- NVPTX.td - Describe the NVPTX Target Machine -----------*- tblgen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the NVPTX target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

include "NVPTXRegisterInfo.td"
include "NVPTXInstrInfo.td"

//===----------------------------------------------------------------------===//
// Subtarget Features.
// - We use the SM version number instead of explicit feature table.
// - Need at least one feature to avoid generating zero sized array by
//   TableGen in NVPTXGenSubtarget.inc.
//===----------------------------------------------------------------------===//

// SM Versions
def SM10 : SubtargetFeature<"sm_10", "SmVersion", "10",
                            "Target SM 1.0">;
def SM11 : SubtargetFeature<"sm_11", "SmVersion", "11",
                            "Target SM 1.1">;
def SM12 : SubtargetFeature<"sm_12", "SmVersion", "12",
                            "Target SM 1.2">;
def SM13 : SubtargetFeature<"sm_13", "SmVersion", "13",
                            "Target SM 1.3">;
def SM20 : SubtargetFeature<"sm_20", "SmVersion", "20",
                            "Target SM 2.0">;
def SM21 : SubtargetFeature<"sm_21", "SmVersion", "21",
                            "Target SM 2.1">;
def SM30 : SubtargetFeature<"sm_30", "SmVersion", "30",
                            "Target SM 3.0">;
def SM35 : SubtargetFeature<"sm_35", "SmVersion", "35",
                            "Target SM 3.5">;

// PTX Versions
def PTX30 : SubtargetFeature<"ptx30", "PTXVersion", "30",
                             "Use PTX version 3.0">;
def PTX31 : SubtargetFeature<"ptx31", "PTXVersion", "31",
                             "Use PTX version 3.1">;

//===----------------------------------------------------------------------===//
// NVPTX supported processors.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"sm_10", [SM10]>;
def : Proc<"sm_11", [SM11]>;
def : Proc<"sm_12", [SM12]>;
def : Proc<"sm_13", [SM13]>;
def : Proc<"sm_20", [SM20]>;
def : Proc<"sm_21", [SM21]>;
def : Proc<"sm_30", [SM30]>;
def : Proc<"sm_35", [SM35]>;


def NVPTXInstrInfo : InstrInfo {
}

def NVPTX : Target {
  let InstructionSet = NVPTXInstrInfo;
}
