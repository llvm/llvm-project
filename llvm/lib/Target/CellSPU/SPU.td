//===-- SPU.td - Describe the STI Cell SPU Target Machine --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the top level entry point for the STI Cell SPU target machine.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing.
//
include "llvm/Target/Target.td"

// Holder of code fragments (you'd think this'd already be in
// a td file somewhere... :-)

class CodeFrag<dag frag> {
  dag Fragment = frag;
}

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "SPURegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction formats, instructions
//===----------------------------------------------------------------------===//

include "SPUNodes.td"
include "SPUOperands.td"
include "SPUSchedule.td"
include "SPUInstrFormats.td"
include "SPUInstrInfo.td"

//===----------------------------------------------------------------------===//
// Subtarget features:
//===----------------------------------------------------------------------===//

def DefaultProc: SubtargetFeature<"", "ProcDirective", "SPU::DEFAULT_PROC", "">;
def LargeMemFeature:
  SubtargetFeature<"large_mem","UseLargeMem", "true",
                   "Use large (>256) LSA memory addressing [default = false]">;

def SPURev0 : Processor<"v0", SPUItineraries, [DefaultProc]>;

//===----------------------------------------------------------------------===//
// Calling convention:
//===----------------------------------------------------------------------===//

include "SPUCallingConv.td"

// Target:

def SPUInstrInfo : InstrInfo {
  let isLittleEndianEncoding = 1;
}

def SPU : Target {
  let InstructionSet = SPUInstrInfo;
}
