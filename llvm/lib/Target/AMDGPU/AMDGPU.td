
//===-- AMDGPU.td - AMDGPU Tablegen files --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"
include "llvm/Target/Target.td"
include "AMDGPUFeatures.td"
include "AMDGPUPredicateControl.td"

def p0 : PtrValueType<i64, 0>;
def p1 : PtrValueType<i64, 1>;
def p2 : PtrValueType<i32, 2>;
def p3 : PtrValueType<i32, 3>;
def p4 : PtrValueType<i64, 4>;
def p5 : PtrValueType<i32, 5>;
def p6 : PtrValueType<i32, 6>;
def p15 : PtrValueType<i32, 15>;

include "GCNFeatures.td"

//===----------------------------------------------------------------------===//

def AMDGPUInstrInfo : InstrInfo {
  let guessInstructionProperties = 1;
}

def AMDGPUAsmParser : AsmParser {
  // Some of the R600 registers have the same name, so this crashes.
  // For example T0_XYZW and T0_XY both have the asm name T0.
  let ShouldEmitMatchRegisterName = 0;

  // Call the custom operand parser for all operands.
  let OperandParserMethod = "parseCustomOperand";
  let CallCustomParserForAllOperands = true;
}

def AMDGPUAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def AMDGPUAsmVariants {
  string Default = "Default";
  int Default_ID = 0;
  string VOP3 = "VOP3";
  int VOP3_ID = 1;
  string SDWA = "SDWA";
  int SDWA_ID = 2;
  string SDWA9 = "SDWA9";
  int SDWA9_ID = 3;
  string DPP = "DPP";
  int DPP_ID = 4;
  string VOP3_DPP = "VOP3_DPP";
  int VOP3_DPP_ID = 5;
  string Disable = "Disable";
  int Disable_ID = 6;
}

def DefaultAMDGPUAsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.Default_ID;
  let Name = AMDGPUAsmVariants.Default;
}

def VOP3AsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.VOP3_ID;
  let Name = AMDGPUAsmVariants.VOP3;
}

def SDWAAsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.SDWA_ID;
  let Name = AMDGPUAsmVariants.SDWA;
}

def SDWA9AsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.SDWA9_ID;
  let Name = AMDGPUAsmVariants.SDWA9;
}

def DPPAsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.DPP_ID;
  let Name = AMDGPUAsmVariants.DPP;
}

def VOP3_DPPAsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.VOP3_DPP_ID;
  let Name = AMDGPUAsmVariants.VOP3_DPP;
}

def AMDGPU : Target {
  // Pull in Instruction Info:
  let InstructionSet = AMDGPUInstrInfo;
  let AssemblyParsers = [AMDGPUAsmParser];
  let AssemblyParserVariants = [DefaultAMDGPUAsmParserVariant,
                                VOP3AsmParserVariant,
                                SDWAAsmParserVariant,
                                SDWA9AsmParserVariant,
                                DPPAsmParserVariant,
                                VOP3_DPPAsmParserVariant];
  let AssemblyWriters = [AMDGPUAsmWriter];
  let AllowRegisterRenaming = 1;
}

// Dummy Instruction itineraries for pseudo instructions
def ALU_NULL : FuncUnit;
def NullALU : InstrItinClass;

//===----------------------------------------------------------------------===//
// Predicate helper class
//===----------------------------------------------------------------------===//

def isGFX6 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS">,
  AssemblerPredicate<(all_of FeatureSouthernIslands)>;

def isGFX6GFX7 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), (not FeatureGFX10Insts))>;

def isGFX6GFX7GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), (not FeatureGFX11Insts))>;

def isGFX6GFX7GFX10Plus :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() >= AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding))>;

def isGFX7Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureCIInsts, (not FeatureGFX10Insts))>;

def isGFX7GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureCIInsts, (not FeatureGFX11Insts))>;

def isGFX7GFX10GFX11 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX11">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureCIInsts)>;

def isGFX7GFX8GFX9 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGFX7GFX8GFX9Insts)>;

def isGFX6GFX7GFX8GFX9 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of (not FeatureGFX10Insts))>;

def isGFX6GFX7GFX8GFX9NotGFX90A :
  Predicate<"!Subtarget->hasGFX90AInsts() &&"
            "(Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            " Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            " Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            " Subtarget->getGeneration() == AMDGPUSubtarget::GFX9)">,
  AssemblerPredicate<(all_of (not FeatureGFX10Insts), (not FeatureGFX90AInsts))>;

def isGFX6GFX7GFX8GFX9GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of (not FeatureGFX11Insts))>;

def isNotGFX12Plus :
  Predicate<"Subtarget->getGeneration() <= AMDGPUSubtarget::GFX11">,
  AssemblerPredicate<(all_of (not FeatureGFX12Insts))>;

def isGFX7GFX8GFX9GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of FeatureCIInsts, (not FeatureGFX11Insts))>;

def isGFX8GFX9GFX10GFX11 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX11">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, (not FeatureGFX12Insts))>;

def isGFX8GFX9GFX10GFX11GFX12 :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS &&"
            "Subtarget->getGeneration() < AMDGPUSubtarget::GFX13">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, (not FeatureGFX13Insts))>;

def isGFX7Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::SEA_ISLANDS">,
  AssemblerPredicate<(all_of FeatureCIInsts)>;

def isGFX8Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS">,
  AssemblerPredicate<(all_of FeatureGFX8Insts)>;

def isGFX8Only : Predicate<"Subtarget->getGeneration() =="
                           "AMDGPUSubtarget::VOLCANIC_ISLANDS">,
  AssemblerPredicate <(all_of FeatureVolcanicIslands)>;

def isGFX9Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGFX9Insts)>;

def isNotGFX9Plus :
  Predicate<"Subtarget->getGeneration() < AMDGPUSubtarget::GFX9">;

def isGFX9Only : Predicate <
  "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGCN3Encoding, FeatureGFX9Insts)>;

def isGCN3ExcludingGFX90A :
  Predicate<"Subtarget->isGCN3Encoding() && !Subtarget->hasGFX90AInsts()">,
  AssemblerPredicate<(all_of FeatureGCN3Encoding, (not FeatureGFX90AInsts))>;

def isGFX90APlus :
  Predicate<"Subtarget->hasGFX90AInsts()">,
  AssemblerPredicate<(all_of FeatureGFX90AInsts)>;

def isNotGFX90APlus :
  Predicate<"!Subtarget->hasGFX90AInsts()">,
  AssemblerPredicate<(all_of (not FeatureGFX90AInsts))>;

def isGFX8GFX9NotGFX90A :
  Predicate<"!Subtarget->hasGFX90AInsts() &&"
            "(Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            " Subtarget->getGeneration() == AMDGPUSubtarget::GFX9)">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, FeatureGCN3Encoding, (not FeatureGFX90AInsts))>;

def isGFX90AOnly :
  Predicate<"Subtarget->hasGFX90AInsts() && !Subtarget->hasGFX940Insts()">,
  AssemblerPredicate<(all_of FeatureGFX90AInsts, (not FeatureGFX940Insts))>;

def isGFX908orGFX90A :
  Predicate<"Subtarget->hasMAIInsts() && !Subtarget->hasGFX940Insts()">,
  AssemblerPredicate<(all_of FeatureMAIInsts, (not FeatureGFX940Insts))>;

def isGFX940Plus :
  Predicate<"Subtarget->hasGFX940Insts()">,
  AssemblerPredicate<(all_of FeatureGFX940Insts)>;

def isNotGFX940Plus :
  Predicate<"!Subtarget->hasGFX940Insts()">,
  AssemblerPredicate<(all_of (not FeatureGFX940Insts))>;

def isGFX8GFX9NotGFX940 :
  Predicate<"!Subtarget->hasGFX940Insts() &&"
            "(Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            " Subtarget->getGeneration() == AMDGPUSubtarget::GFX9)">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, FeatureGCN3Encoding, (not FeatureGFX940Insts))>;

def isGFX8GFX9 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, FeatureGCN3Encoding)>;

def isGFX10Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of FeatureGFX10Insts, (not FeatureGFX11Insts))>;

def isGFX10Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of FeatureGFX10Insts)>;

def isGFX10GFX11 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX10 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX11">,
  AssemblerPredicate<(all_of FeatureGFX10Insts, (not FeatureGFX12Insts))>;

def isGFX10Before1030 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX10 &&"
            "!Subtarget->hasGFX10_3Insts()">,
  AssemblerPredicate<(all_of FeatureGFX10Insts,(not FeatureGFX10_3Insts))>;

def isGFX9GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX9 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of FeatureGFX9Insts, (not FeatureGFX11Insts))>;

def isGFX9GFX10GFX11 :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX9 &&"
            "Subtarget->getGeneration() < AMDGPUSubtarget::GFX12">,
  AssemblerPredicate<(all_of FeatureGFX9Insts, (not FeatureGFX12Insts))>;

def isGFX8GFX9GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, (not FeatureGFX11Insts))>;

def isGFX11Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX11">,
  AssemblerPredicate<(all_of FeatureGFX11Insts, (not FeatureGFX12Insts))>;

def isGFX11Not11_70 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX11 &&"
            "!Subtarget->hasGFX11_7Insts()">,
  AssemblerPredicate<(all_of FeatureGFX11Insts, (not FeatureGFX12Insts), (not FeatureGFX11_7Insts))>;

def isGFX1170Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX11 &&"
            "Subtarget->hasGFX11_7Insts()">,
  AssemblerPredicate<(all_of FeatureGFX11_7Insts, (not FeatureGFX12Insts))>;

def isGFX11Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX11">,
  AssemblerPredicate<(all_of FeatureGFX11Insts)>;

def isGFX11GFX12 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX11 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX12">,
  AssemblerPredicate<(all_of FeatureGFX11Insts, (not FeatureGFX13Insts))>;

def isGFX11PlusNot12_50 :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX11 &&"
            "(Subtarget->getGeneration() >= AMDGPUSubtarget::GFX13 || !Subtarget->hasGFX1250Insts())">,
  AssemblerPredicate<(all_of FeatureGFX11Insts, (any_of FeatureGFX13Insts, (not FeatureGFX1250Insts)))>;

def isGFX12Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX12">,
  AssemblerPredicate<(all_of FeatureGFX12Insts, (not FeatureGFX13Insts))>;

def isGFX12Not12_50 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX12 && !Subtarget->hasGFX1250Insts()">,
  AssemblerPredicate<(all_of FeatureGFX12Insts, (not FeatureGFX1250Insts))>;

def isGFX12Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX12">,
  AssemblerPredicate<(all_of FeatureGFX12Insts)>;

def isGFX12PlusNot12_50 :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX12 &&"
            "(Subtarget->getGeneration() >= AMDGPUSubtarget::GFX13 || !Subtarget->hasGFX1250Insts())">,
  AssemblerPredicate<(all_of FeatureGFX12Insts, (any_of FeatureGFX13Insts, (not FeatureGFX1250Insts)))>;

def isGFX125xOnly :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX12 && Subtarget->hasGFX1250Insts()">,
  AssemblerPredicate<(all_of FeatureGFX1250Insts, (not FeatureGFX13Insts))>;

def isGFX1250Plus :
  Predicate<"Subtarget->hasGFX1250Insts()">,
  AssemblerPredicate<(all_of FeatureGFX1250Insts)>;

def isNotGFX1250Plus :
  Predicate<"!Subtarget->hasGFX1250Insts()">,
  AssemblerPredicate<(all_of (not FeatureGFX1250Insts))>;

def isGFX940orGFX1250 :
  Predicate<"Subtarget->hasGFX940Insts() ||"
            "(Subtarget->hasGFX1250Insts() && !Subtarget->hasGFX13Insts())">,
  AssemblerPredicate<(any_of FeatureGFX940Insts,
                     (all_of FeatureGFX1250Insts, (not FeatureGFX13Insts)))>;

def isGFX13Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX13">,
  AssemblerPredicate<(all_of FeatureGFX13Insts)>;

def isGFX13Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX13">,
  AssemblerPredicate<(all_of FeatureGFX13Insts)>;

def HasD16LoadStore : Predicate<"Subtarget->hasD16LoadStore()">,
  AssemblerPredicate<(all_of FeatureGFX9Insts)>;

def HasFlatScratchSTMode : Predicate<"Subtarget->hasFlatScratchSTMode()">,
  AssemblerPredicate<(any_of FeatureGFX10_3Insts, FeatureGFX940Insts)>;
def HasFlatScratchSVSMode : Predicate<"Subtarget->hasFlatScratchSVSMode()">,
  AssemblerPredicate<(any_of FeatureGFX940Insts, FeatureGFX11Insts)>;

def HasGFX10_BEncoding : Predicate<"Subtarget->hasGFX10_BEncoding()">,
  AssemblerPredicate<(all_of FeatureGFX10_BEncoding)>;

def HasPackedD16VMem : Predicate<"!Subtarget->hasUnpackedD16VMem()">,
  AssemblerPredicate<(all_of (not FeatureUnpackedD16VMem))>;

def HasUnrestrictedSOffset : Predicate<"!Subtarget->hasRestrictedSOffset()">,
  AssemblerPredicate<(all_of (not FeatureRestrictedSOffset))>;

def D16PreservesUnusedBits :
  Predicate<"Subtarget->d16PreservesUnusedBits()">,
  AssemblerPredicate<(all_of FeatureGFX9Insts, (not FeatureSupportsSRAMECC))>;

def LDSRequiresM0Init : Predicate<"Subtarget->ldsRequiresM0Init()">;
def NotLDSRequiresM0Init : Predicate<"!Subtarget->ldsRequiresM0Init()">;
def HasUsableDSOffset : Predicate<"Subtarget->hasUsableDSOffset()">;

def HasExportInsts : Predicate<"Subtarget->hasExportInsts()">,
  AssemblerPredicate<(any_of FeatureGFX13Insts, (all_of (not FeatureGFX90AInsts), (not FeatureGFX1250Insts)))>;

def HasVINTERPEncoding : Predicate<"Subtarget->hasVINTERPEncoding()">,
  AssemblerPredicate<(any_of FeatureGFX13Insts, (all_of FeatureGFX11Insts, (not FeatureGFX1250Insts)))>;

def HasDSAddTid : Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGFX9Insts)>;

def HasLDSFPAtomicAddF32 : Predicate<"Subtarget->hasLDSFPAtomicAddF32()">,
  AssemblerPredicate<(all_of FeatureGFX8Insts)>;

def NotHasAddNoCarryInsts : Predicate<"!Subtarget->hasAddNoCarryInsts()">;

def HasXNACKEnabled : Predicate<"Subtarget->isXNACKEnabled()">;

def NotHasTrue16BitInsts : True16PredicateClass<"!Subtarget->hasTrue16BitInsts()">,
  AssemblerPredicate<(all_of (not FeatureTrue16BitInsts))>;

// Control use of True16 instructions. The real True16 instructions are
// True16 instructions as they are defined in the ISA. Fake True16
// instructions have the same encoding as real ones but syntactically
// only allow 32-bit registers in operands and use low halves thereof.
def UseRealTrue16Insts : True16PredicateClass<"Subtarget->useRealTrue16Insts()">,
  AssemblerPredicate<(all_of FeatureTrue16BitInsts, FeatureRealTrue16Insts)>;
def NotUseRealTrue16Insts : True16PredicateClass<"!Subtarget->useRealTrue16Insts()">,
  AssemblerPredicate<(not (all_of FeatureTrue16BitInsts, FeatureRealTrue16Insts))>;
def UseFakeTrue16Insts : True16PredicateClass<"Subtarget->hasTrue16BitInsts() && "
                                              "!Subtarget->useRealTrue16Insts()">,
  AssemblerPredicate<(all_of FeatureTrue16BitInsts, (not FeatureRealTrue16Insts))>;

def UseTrue16WithSramECC : True16PredicateClass<"Subtarget->useRealTrue16Insts() && "
                                                "!Subtarget->d16PreservesUnusedBits()">;

def NotHasD16Writes32BitVgpr: Predicate<"!Subtarget->hasD16Writes32BitVgpr()">,
  AssemblerPredicate<(all_of FeatureTrue16BitInsts, FeatureRealTrue16Insts, (not FeatureD16Writes32BitVgpr))>;

def NotHasMed3_16 : Predicate<"!Subtarget->hasMed3_16()">;
def HasMed3_16 : Predicate<"Subtarget->hasMed3_16()">;

def HasMinMaxDenormModes : Predicate<"Subtarget->supportsMinMaxDenormModes()">;
def NotHasMinMaxDenormModes : Predicate<"!Subtarget->supportsMinMaxDenormModes()">;

def HasFminFmaxLegacy : Predicate<"Subtarget->hasFminFmaxLegacy()">;

def HasSDWA8 : Predicate<"Subtarget->hasSDWA()">,
  AssemblerPredicate<(all_of (not FeatureGFX9Insts), FeatureSDWA)>;

def HasSDWA9 :
  Predicate<"Subtarget->hasSDWA()">,
  AssemblerPredicate<(all_of FeatureGCN3Encoding, FeatureGFX9Insts,FeatureSDWA)>;

def HasSDWA10 :
  Predicate<"Subtarget->hasSDWA()">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureGFX10Insts, FeatureSDWA)>;

def HasDPP : Predicate<"Subtarget->hasDPP()">,
  AssemblerPredicate<(all_of FeatureGCN3Encoding, FeatureDPP)>;

def HasGCN3DPALU_DPP :
  Predicate<"Subtarget->isGCN3Encoding() && Subtarget->hasDPALU_DPP()">,
  AssemblerPredicate<(all_of FeatureGCN3Encoding, FeatureDPALU_DPP)>;

def HasDPP8 : Predicate<"Subtarget->hasDPP8()">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureGFX10Insts, FeatureDPP8)>;

def HasPkMovB32 : Predicate<"Subtarget->hasPkMovB32()">,
  AssemblerPredicate<(all_of FeatureGFX90AInsts)>;

def HasFmaakFmamkF32Insts :
  Predicate<"Subtarget->hasFmaakFmamkF32Insts()">,
  AssemblerPredicate<(any_of FeatureGFX10Insts, FeatureGFX940Insts)>;

def HasFmaakFmamkF64Insts :
  Predicate<"Subtarget->hasFmaakFmamkF64Insts()">,
  AssemblerPredicate<(any_of FeatureGFX1250Insts)>;

def HasPkMinMax3Insts :
  Predicate<"Subtarget->hasPkMinMax3Insts()">,
  AssemblerPredicate<(any_of FeatureGFX1250Insts)>;

def HasSGetShaderCyclesInst :
  Predicate<"Subtarget->hasSGetShaderCyclesInst()">,
  AssemblerPredicate<(any_of FeatureGFX1250Insts)>;

def HasDPP16 : Predicate<"Subtarget->hasDPP()">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureGFX10Insts, FeatureDPP)>;

def HasSdstCMPX : Predicate<"!Subtarget->hasNoSdstCMPX()">,
  AssemblerPredicate<(all_of (not FeatureNoSdstCMPX))>;

def has16BankLDS : Predicate<"Subtarget->getLDSBankCount() == 16">;
def has32BankLDS : Predicate<"Subtarget->getLDSBankCount() == 32">;

def NotHasMAIInsts : Predicate<"!Subtarget->hasMAIInsts()">,
  AssemblerPredicate<(all_of (not FeatureMAIInsts))>;

def NotHasFP8E5M3Insts : Predicate<"!Subtarget->hasFP8E5M3Insts()">,
  AssemblerPredicate<(all_of (not FeatureFP8E5M3Insts))>;

// Byte 0 of the SDWA src_sel form is reachable through the plain VOP1 encoding
// unless that encoding is unreliable.
def HasCvtFP8SDWASrcSelNotVOP1Bug : Predicate<"Subtarget->hasCvtFP8SDWASrcSel() && !Subtarget->hasCvtFP8VOP1Bug()">,
  AssemblerPredicate<(all_of FeatureCvtFP8SDWASrcSel, (not FeatureCvtFP8VOP1Bug))>;

// The byte_sel form of the fp8 to f32 conversion reads the source as e5m3 when
// the clamp bit is set, replacing the plain op_sel form.
def HasCvtFP8ByteSelE5M3 : Predicate<"Subtarget->hasCvtFP8ByteSel() && Subtarget->hasFP8E5M3Insts()">,
  AssemblerPredicate<(all_of FeatureCvtFP8ByteSel, FeatureFP8E5M3Insts)>;

def HasCvtFP8ByteSelNotE5M3 : Predicate<"Subtarget->hasCvtFP8ByteSel() && !Subtarget->hasFP8E5M3Insts()">,
  AssemblerPredicate<(all_of FeatureCvtFP8ByteSel, (not FeatureFP8E5M3Insts))>;

def HasFmacLegacy32 : Predicate<"Subtarget->hasFmaLegacy32Insts() && Subtarget->getGeneration() < AMDGPUSubtarget::GFX12">,
  AssemblerPredicate<(all_of FeatureFmaLegacy32Insts, (not FeatureGFX12Insts))>;

def HasAtomicBufferGlobalPkAddF16NoRtnInsts
  : Predicate<"Subtarget->hasAtomicBufferGlobalPkAddF16NoRtnInsts() || Subtarget->hasAtomicBufferGlobalPkAddF16Insts()">,
  AssemblerPredicate<(any_of FeatureAtomicBufferGlobalPkAddF16NoRtnInsts, FeatureAtomicBufferGlobalPkAddF16Insts)>;

def HasAddPC64Inst : Predicate<"Subtarget->hasAddPC64Inst()">,
  AssemblerPredicate<(any_of FeatureGFX1250Insts)>;

def HasFlatScratchEnabled : Predicate<"Subtarget->hasFlatScratchEnabled()">;

def NotHasFlatScratchEnabled : Predicate<"!Subtarget->hasFlatScratchEnabled()">;

def NotHasMADIntraFwdBug : Predicate<"!Subtarget->hasMADIntraFwdBug()">;

def NotHasSALUFloatInsts : Predicate<"!Subtarget->hasSALUFloatInsts()">,
  AssemblerPredicate<(all_of (not FeatureSALUFloatInsts))>;

def HasSignedDotInsts : Predicate<"Subtarget->hasDot1Insts() || Subtarget->hasDot8Insts()">,
  AssemblerPredicate<(any_of FeatureDot1Insts, FeatureDot8Insts)>;

def NotHasIEEEMinimumMaximumInsts : Predicate<"!Subtarget->hasIEEEMinimumMaximumInsts()">;

def HasAnyPackedFP32Ops : Predicate<"Subtarget->hasAnyPackedFP32Ops()">,
  AssemblerPredicate<(any_of FeaturePackedFP32Ops, FeaturePackedFP32SingleSGPROps)>;

def HasAnyPackedFP64Ops : Predicate<"Subtarget->hasAnyPackedFP64Ops()">,
  AssemblerPredicate<(any_of FeaturePackedFP64SingleSGPROps)>;

def HasAnyPackedU64Ops : Predicate<"Subtarget->hasAnyPackedU64Ops()">,
  AssemblerPredicate<(any_of FeaturePackedU64SingleSGPROps)>;

def NeedsAlignedVGPRs : Predicate<"Subtarget->needsAlignedVGPRs()">,
                      AssemblerPredicate<(all_of FeatureRequiresAlignedVGPRs)>;

def NotNeedsAlignedVGPRs : Predicate<"!Subtarget->needsAlignedVGPRs()">,
                      AssemblerPredicate<(all_of (not FeatureRequiresAlignedVGPRs))>;

def NotNeedsAligned2addrDS : Predicate<"!Subtarget->hasNeedsAligned2addrDS()">;

def isWave32 : Predicate<"Subtarget->isWave32()">,
  AssemblerPredicate <(any_of FeatureWavefrontSize32,
                              FeatureAssemblerPermissiveWavesize)>;
def isWave64 : Predicate<"Subtarget->isWave64()">,
  AssemblerPredicate <(any_of FeatureWavefrontSize64,
                              FeatureAssemblerPermissiveWavesize)>;

def isWave32Strict : Predicate<"Subtarget->isWave32()">,
  AssemblerPredicate <(all_of FeatureWavefrontSize32)>;
def isWave64Strict : Predicate<"Subtarget->isWave64()">,
  AssemblerPredicate <(all_of FeatureWavefrontSize64)>;

//===----------------------------------------------------------------------===//
// HwModes
//===----------------------------------------------------------------------===//

defvar DefaultMode_Wave64 = DefaultMode;
defvar DefaultMode_Wave32 = HwMode<[isWave32, NotNeedsAlignedVGPRs]>;

// gfx90a-gfx950. Has AGPRs, and also the align2 VGPR/AGPR requirement. Implied
// wave64.
def AVAlign2LoadStoreMode : HwMode<[HasMAIInsts, NeedsAlignedVGPRs]>;

// gfx1250, has alignment requirement but no AGPRs.
def AlignedVGPRNoAGPRMode_Wave32 : HwMode<[NotHasMAIInsts, NeedsAlignedVGPRs, isWave32Strict]>;
def AlignedVGPRNoAGPRMode_Wave64 : HwMode<[NotHasMAIInsts, NeedsAlignedVGPRs, isWave64Strict]>;

// FIXME: This should be able to only define a separate hwmode that
// only depends on wavesize for just ValueTypes. These use different
// HwMode namespaces. If we don't define the full set of modes used
// for RegClassByHwMode, tablegen crashes for some reason
def WaveSizeVT : ValueTypeByHwMode<[
  DefaultMode_Wave64,
  AVAlign2LoadStoreMode,
  AlignedVGPRNoAGPRMode_Wave64,
  DefaultMode_Wave32,
  AlignedVGPRNoAGPRMode_Wave32], [i64, i64, i64, i32, i32]>;


// Include AMDGPU TD files
include "SISchedule.td"
include "GCNProcessors.td"
include "AMDGPUInstrInfo.td"
include "SIRegisterInfo.td"
include "AMDGPURegisterBanks.td"
include "AMDGPUInstructions.td"
include "SIInstrInfo.td"
include "AMDGPUCallingConv.td"
include "AMDGPUSearchableTables.td"
