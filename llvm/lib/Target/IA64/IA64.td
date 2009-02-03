//===-- IA64.td - Target definition file for Intel IA64 -------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is a target description file for the Intel IA64 architecture,
// also known variously as ia64, IA-64, IPF, "the Itanium architecture" etc.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing...
//
include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "IA64RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "IA64InstrInfo.td"

def IA64InstrInfo : InstrInfo { }

def IA64 : Target {
  // Our instruction set
  let InstructionSet = IA64InstrInfo;

}


