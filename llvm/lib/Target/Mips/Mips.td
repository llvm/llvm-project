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

// The overall idea of the PredicateControl class is to chop the Predicates list
// into subsets that are usually overridden independently. This allows
// subclasses to partially override the predicates of their superclasses without
// having to re-add all the existing predicates.
class PredicateControl {
  // Predicates for the encoding scheme in use such as HasStdEnc
  list<Predicate> EncodingPredicates = [];
  // Predicates for the GPR size such as IsGP64bit
  list<Predicate> GPRPredicates = [];
  // Predicates for the FGR size and layout such as IsFP64bit
  list<Predicate> FGRPredicates = [];
  // Predicates for the instruction group membership such as ISA's and ASE's
  list<Predicate> InsnPredicates = [];
  // Predicates for anything else
  list<Predicate> AdditionalPredicates = [];
  list<Predicate> Predicates = !listconcat(EncodingPredicates,
                                           GPRPredicates,
                                           FGRPredicates,
                                           InsnPredicates,
                                           AdditionalPredicates);
}

// Like Requires<> but for the AdditionalPredicates list
class AdditionalRequires<list<Predicate> preds> {
  list<Predicate> AdditionalPredicates = preds;
}

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

def FeatureNoABICalls  : SubtargetFeature<"noabicalls", "NoABICalls", "true",
                                "Disable SVR4-style position-independent code.">;
def FeatureGP64Bit     : SubtargetFeature<"gp64", "IsGP64bit", "true",
                                "General Purpose Registers are 64-bit wide.">;
def FeatureFP64Bit     : SubtargetFeature<"fp64", "IsFP64bit", "true",
                                "Support 64-bit FP registers.">;
def FeatureFPXX        : SubtargetFeature<"fpxx", "IsFPXX", "true",
                                "Support for FPXX.">;
def FeatureNaN2008     : SubtargetFeature<"nan2008", "IsNaN2008bit", "true",
                                "IEEE 754-2008 NaN encoding.">;
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
def FeatureNoOddSPReg  : SubtargetFeature<"nooddspreg", "UseOddSPReg", "false",
                              "Disable odd numbered single-precision "
                              "registers">;
def FeatureVFPU        : SubtargetFeature<"vfpu", "HasVFPU",
                                "true", "Enable vector FPU instructions.">;
def FeatureMips1       : SubtargetFeature<"mips1", "MipsArchVersion", "Mips1",
                                "Mips I ISA Support [highly experimental]">;
def FeatureMips2       : SubtargetFeature<"mips2", "MipsArchVersion", "Mips2",
                                "Mips II ISA Support [highly experimental]",
                                [FeatureMips1]>;
def FeatureMips3_32    : SubtargetFeature<"mips3_32", "HasMips3_32", "true",
                                "Subset of MIPS-III that is also in MIPS32 "
                                "[highly experimental]">;
def FeatureMips3_32r2  : SubtargetFeature<"mips3_32r2", "HasMips3_32r2", "true",
                                "Subset of MIPS-III that is also in MIPS32r2 "
                                "[highly experimental]">;
def FeatureMips3       : SubtargetFeature<"mips3", "MipsArchVersion", "Mips3",
                                "MIPS III ISA Support [highly experimental]",
                                [FeatureMips2, FeatureMips3_32,
                                 FeatureMips3_32r2, FeatureGP64Bit,
                                 FeatureFP64Bit]>;
def FeatureMips4_32    : SubtargetFeature<"mips4_32", "HasMips4_32", "true",
                                "Subset of MIPS-IV that is also in MIPS32 "
                                "[highly experimental]">;
def FeatureMips4_32r2  : SubtargetFeature<"mips4_32r2", "HasMips4_32r2", "true",
                                "Subset of MIPS-IV that is also in MIPS32r2 "
                                "[highly experimental]">;
def FeatureMips4       : SubtargetFeature<"mips4", "MipsArchVersion",
                                "Mips4", "MIPS IV ISA Support",
                                [FeatureMips3, FeatureMips4_32,
                                 FeatureMips4_32r2]>;
def FeatureMips5_32r2  : SubtargetFeature<"mips5_32r2", "HasMips5_32r2", "true",
                                "Subset of MIPS-V that is also in MIPS32r2 "
                                "[highly experimental]">;
def FeatureMips5       : SubtargetFeature<"mips5", "MipsArchVersion", "Mips5",
                                "MIPS V ISA Support [highly experimental]",
                                [FeatureMips4, FeatureMips5_32r2]>;
def FeatureMips32      : SubtargetFeature<"mips32", "MipsArchVersion", "Mips32",
                                "Mips32 ISA Support",
                                [FeatureMips2, FeatureMips3_32,
                                 FeatureMips4_32]>;
def FeatureMips32r2    : SubtargetFeature<"mips32r2", "MipsArchVersion",
                                "Mips32r2", "Mips32r2 ISA Support",
                                [FeatureMips3_32r2, FeatureMips4_32r2,
                                 FeatureMips5_32r2, FeatureMips32]>;
def FeatureMips32r6    : SubtargetFeature<"mips32r6", "MipsArchVersion",
                                "Mips32r6",
                                "Mips32r6 ISA Support [experimental]",
                                [FeatureMips32r2, FeatureFP64Bit,
                                 FeatureNaN2008]>;
def FeatureMips64      : SubtargetFeature<"mips64", "MipsArchVersion",
                                "Mips64", "Mips64 ISA Support",
                                [FeatureMips5, FeatureMips32]>;
def FeatureMips64r2    : SubtargetFeature<"mips64r2", "MipsArchVersion",
                                "Mips64r2", "Mips64r2 ISA Support",
                                [FeatureMips64, FeatureMips32r2]>;
def FeatureMips64r6    : SubtargetFeature<"mips64r6", "MipsArchVersion",
                                "Mips64r6",
                                "Mips64r6 ISA Support [experimental]",
                                [FeatureMips32r6, FeatureMips64r2,
                                 FeatureNaN2008]>;

def FeatureMips16  : SubtargetFeature<"mips16", "InMips16Mode", "true",
                                      "Mips16 mode">;

def FeatureDSP : SubtargetFeature<"dsp", "HasDSP", "true", "Mips DSP ASE">;
def FeatureDSPR2 : SubtargetFeature<"dspr2", "HasDSPR2", "true",
                                    "Mips DSP-R2 ASE", [FeatureDSP]>;

def FeatureMSA : SubtargetFeature<"msa", "HasMSA", "true", "Mips MSA ASE">;

def FeatureMicroMips  : SubtargetFeature<"micromips", "InMicroMipsMode", "true",
                                         "microMips mode">;

def FeatureCnMips     : SubtargetFeature<"cnmips", "HasCnMips",
                                "true", "Octeon cnMIPS Support",
                                [FeatureMips64r2]>;

//===----------------------------------------------------------------------===//
// Mips processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, MipsGenericItineraries, Features>;

def : Proc<"mips1", [FeatureMips1, FeatureO32]>;
def : Proc<"mips2", [FeatureMips2, FeatureO32]>;
def : Proc<"mips32", [FeatureMips32, FeatureO32]>;
def : Proc<"mips32r2", [FeatureMips32r2, FeatureO32]>;
def : Proc<"mips32r6", [FeatureMips32r6, FeatureO32]>;

def : Proc<"mips3", [FeatureMips3, FeatureN64]>;
def : Proc<"mips4", [FeatureMips4, FeatureN64]>;
def : Proc<"mips5", [FeatureMips5, FeatureN64]>;
def : Proc<"mips64", [FeatureMips64, FeatureN64]>;
def : Proc<"mips64r2", [FeatureMips64r2, FeatureN64]>;
def : Proc<"mips64r6", [FeatureMips64r6, FeatureN64]>;
def : Proc<"mips16", [FeatureMips16, FeatureO32]>;
def : Proc<"octeon", [FeatureMips64r2, FeatureN64, FeatureCnMips]>;

def MipsAsmParser : AsmParser {
  let ShouldEmitMatchRegisterName = 0;
  let MnemonicContainsDot = 1;
}

def MipsAsmParserVariant : AsmParserVariant {
  int Variant = 0;

  // Recognize hard coded registers.
  string RegisterPrefix = "$";
}

def Mips : Target {
  let InstructionSet = MipsInstrInfo;
  let AssemblyParsers = [MipsAsmParser];
  let AssemblyParserVariants = [MipsAsmParserVariant];
}
