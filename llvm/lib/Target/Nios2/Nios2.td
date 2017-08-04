//===-- Nios2.td - Describe the Nios2 Target Machine -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Target-dependent interfaces
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "Nios2RegisterInfo.td"
include "Nios2InstrInfo.td"

def Nios2InstrInfo : InstrInfo;

def Nios2 : Target { let InstructionSet = Nios2InstrInfo; }
