//===-- Mips.td - Describe the Mips Target Machine ---------*- tablegen -*-===//
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

def MipsInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Mips Subtarget features                                                    //
//===----------------------------------------------------------------------===//

def FeatureGP64Bit     : SubtargetFeature<"gp64", "IsGP64bit", "true",
                                "General Purpose Registers are 64-bit wide.">;
def FeatureFP64Bit     : SubtargetFeature<"fp64", "IsFP64bit", "true",
                                "Support 64-bit FP registers.">;
def FeatureSingleFloat : SubtargetFeature<"single-float", "IsSingleFloat",
                                "true", "Only supports single precision float">;
def FeatureO32         : SubtargetFeature<"o32", "MipsABI", "O32",
                                "Enable o32 ABI">;
def FeatureN32         : SubtargetFeature<"n32", "MipsABI", "N32",
                                "Enable n32 ABI">;
def FeatureN64         : SubtargetFeature<"n64", "MipsABI", "N64",
                                "Enable n64 ABI">;
def FeatureEABI        : SubtargetFeature<"eabi", "MipsABI", "EABI",
                                "Enable eabi ABI">;
def FeatureAndroid     : SubtargetFeature<"android", "IsAndroid", "true",
                                "Target is android">;
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
def FeatureMips32      : SubtargetFeature<"mips32", "MipsArchVersion", "Mips32",
                                "Mips32 ISA Support",
                                [FeatureCondMov, FeatureBitCount]>;
def FeatureMips32r2    : SubtargetFeature<"mips32r2", "MipsArchVersion",
                                "Mips32r2", "Mips32r2 ISA Support",
                                [FeatureMips32, FeatureSEInReg, FeatureSwap]>;
def FeatureMips64      : SubtargetFeature<"mips64", "MipsArchVersion",
                                "Mips64", "Mips64 ISA Support",
                                [FeatureGP64Bit, FeatureFP64Bit,
                                 FeatureMips32]>;
def FeatureMips64r2    : SubtargetFeature<"mips64r2", "MipsArchVersion",
                                "Mips64r2", "Mips64r2 ISA Support",
                                [FeatureMips64, FeatureMips32r2]>;

def FeatureMips16  : SubtargetFeature<"mips16", "InMips16Mode", "true",
                                      "Mips16 mode">;

def FeatureDSP : SubtargetFeature<"dsp", "HasDSP", "true", "Mips DSP ASE">;
def FeatureDSPR2 : SubtargetFeature<"dspr2", "HasDSPR2", "true",
                                    "Mips DSP-R2 ASE", [FeatureDSP]>;

//===----------------------------------------------------------------------===//
// Mips processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, MipsGenericItineraries, Features>;

def : Proc<"mips32", [FeatureMips32]>;
def : Proc<"mips32r2", [FeatureMips32r2]>;
def : Proc<"mips64", [FeatureMips64]>;
def : Proc<"mips64r2", [FeatureMips64r2]>;
def : Proc<"mips16", [FeatureMips16]>;

def MipsAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

def MipsAsmParser : AsmParser {
  let ShouldEmitMatchRegisterName = 0;
}

def MipsAsmParserVariant : AsmParserVariant {
  int Variant = 0;

  // Recognize hard coded registers.
  string RegisterPrefix = "$";
}

def Mips : Target {
  let InstructionSet = MipsInstrInfo;
  let AssemblyParsers = [MipsAsmParser];
  let AssemblyWriters = [MipsAsmWriter];
  let AssemblyParserVariants = [MipsAsmParserVariant];
}
