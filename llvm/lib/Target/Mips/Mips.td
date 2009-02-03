//===- Mips.td - Describe the Mips Target Machine ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the Mips target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "MipsRegisterInfo.td"
include "MipsSchedule.td"
include "MipsInstrInfo.td"
include "MipsCallingConv.td"

def MipsInstrInfo : InstrInfo {
  let TSFlagsFields = [];
  let TSFlagsShifts = [];
}

//===----------------------------------------------------------------------===//
// Mips Subtarget features                                                    //
//===----------------------------------------------------------------------===//

def FeatureGP64Bit     : SubtargetFeature<"gp64", "IsGP64bit", "true",
                                "General Purpose Registers are 64-bit wide.">;
def FeatureFP64Bit     : SubtargetFeature<"fp64", "IsFP64bit", "true",
                                "Support 64-bit FP registers.">;
def FeatureSingleFloat : SubtargetFeature<"single-float", "IsSingleFloat",
                                "true", "Only supports single precision float">;
def FeatureMips2       : SubtargetFeature<"mips2", "MipsArchVersion", "Mips2",
                                "Mips2 ISA Support">;
def FeatureO32         : SubtargetFeature<"o32", "MipsABI", "O32",
                                "Enable o32 ABI">;
def FeatureEABI        : SubtargetFeature<"eabi", "MipsABI", "EABI",
                                "Enable eabi ABI">;
def FeatureVFPU        : SubtargetFeature<"vfpu", "HasVFPU", 
                                "true", "Enable vector FPU instructions.">;
def FeatureSEInReg     : SubtargetFeature<"seinreg", "HasSEInReg", "true", 
                                "Enable 'signext in register' instructions.">;
def FeatureCondMov     : SubtargetFeature<"condmov", "HasCondMov", "true", 
                                "Enable 'conditional move' instructions.">;
def FeatureMulDivAdd   : SubtargetFeature<"muldivadd", "HasMulDivAdd", "true",
                                "Enable 'multiply add/sub' instructions.">;
def FeatureMinMax      : SubtargetFeature<"minmax", "HasMinMax", "true",
                                "Enable 'min/max' instructions.">;
def FeatureSwap        : SubtargetFeature<"swap", "HasSwap", "true",
                                "Enable 'byte/half swap' instructions.">;
def FeatureBitCount    : SubtargetFeature<"bitcount", "HasBitCount", "true",
                                "Enable 'count leading bits' instructions.">;

//===----------------------------------------------------------------------===//
// Mips processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, MipsGenericItineraries, Features>;

def : Proc<"mips1", []>;
def : Proc<"r2000", []>;
def : Proc<"r3000", []>;

def : Proc<"mips2", [FeatureMips2]>;
def : Proc<"r6000", [FeatureMips2]>;

// Allegrex is a 32bit subset of r4000, both for interger and fp registers, 
// but much more similar to Mips2 than Mips3. It also contains some of 
// Mips32/Mips32r2 instructions and a custom vector fpu processor. 
def : Proc<"allegrex", [FeatureMips2, FeatureSingleFloat, FeatureEABI, 
      FeatureVFPU, FeatureSEInReg, FeatureCondMov, FeatureMulDivAdd,
      FeatureMinMax, FeatureSwap, FeatureBitCount]>;

def Mips : Target {
  let InstructionSet = MipsInstrInfo;
}
