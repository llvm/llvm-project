//===-- Hexagon.td - Describe the Hexagon Target Machine --*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the top level entry point for the Hexagon target.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Hexagon Subtarget features.
//===----------------------------------------------------------------------===//

// Hexagon Architectures
include "HexagonDepArch.td"

// Hexagon ISA Extensions
def ExtensionHVX: SubtargetFeature<"hvx", "HexagonHVXVersion",
      "Hexagon::ArchEnum::V60", "Hexagon HVX instructions">;
def ExtensionHVXV60: SubtargetFeature<"hvxv60", "HexagonHVXVersion",
      "Hexagon::ArchEnum::V60", "Hexagon HVX instructions",
      [ExtensionHVX]>;
def ExtensionHVXV62: SubtargetFeature<"hvxv62", "HexagonHVXVersion",
      "Hexagon::ArchEnum::V62", "Hexagon HVX instructions",
      [ExtensionHVX,ExtensionHVXV60]>;
def ExtensionHVXV65: SubtargetFeature<"hvxv65", "HexagonHVXVersion",
      "Hexagon::ArchEnum::V65", "Hexagon HVX instructions",
      [ExtensionHVX,ExtensionHVXV60, ExtensionHVXV62]>;

def ExtensionHVX64B: SubtargetFeature<"hvx-length64b", "UseHVX64BOps",
      "true", "Hexagon HVX 64B instructions", [ExtensionHVX]>;
def ExtensionHVX128B: SubtargetFeature<"hvx-length128b", "UseHVX128BOps",
      "true", "Hexagon HVX 128B instructions", [ExtensionHVX]>;

def FeaturePackets: SubtargetFeature<"packets", "UsePackets", "true",
      "Support for instruction packets">;
def FeatureLongCalls: SubtargetFeature<"long-calls", "UseLongCalls", "true",
      "Use constant-extended calls">;
def FeatureMemNoShuf: SubtargetFeature<"mem_noshuf", "HasMemNoShuf", "false",
      "Supports mem_noshuf feature">;
def FeatureNVJ: SubtargetFeature<"nvj", "UseNewValueJumps", "true",
      "Support for new-value jumps", [FeaturePackets]>;
def FeatureDuplex: SubtargetFeature<"duplex", "EnableDuplex", "true",
      "Enable generation of duplex instruction">;
def FeatureReservedR19: SubtargetFeature<"reserved-r19", "ReservedR19",
      "true", "Reserve register R19">;

//===----------------------------------------------------------------------===//
// Hexagon Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//

def UseMEMOP           : Predicate<"HST->useMemOps()">;
def IEEERndNearV5T     : Predicate<"HST->modeIEEERndNear()">;
def UseHVX64B          : Predicate<"HST->useHVX64BOps()">,
                         AssemblerPredicate<"ExtensionHVX64B">;
def UseHVX128B         : Predicate<"HST->useHVX128BOps()">,
                         AssemblerPredicate<"ExtensionHVX128B">;
def UseHVX             : Predicate<"HST->useHVXOps()">,
                         AssemblerPredicate<"ExtensionHVXV60">;
def UseHVXV60          : Predicate<"HST->useHVXOps()">,
                         AssemblerPredicate<"ExtensionHVXV60">;
def UseHVXV62          : Predicate<"HST->useHVXOps()">,
                         AssemblerPredicate<"ExtensionHVXV62">;
def UseHVXV65          : Predicate<"HST->useHVXOps()">,
                         AssemblerPredicate<"ExtensionHVXV65">;

def Hvx64:  HwMode<"+hvx-length64b">;
def Hvx128: HwMode<"+hvx-length128b">;

//===----------------------------------------------------------------------===//
// Classes used for relation maps.
//===----------------------------------------------------------------------===//

class ImmRegShl;
// ImmRegRel - Filter class used to relate instructions having reg-reg form
// with their reg-imm counterparts.
class ImmRegRel;
// PredRel - Filter class used to relate non-predicated instructions with their
// predicated forms.
class PredRel;
// PredNewRel - Filter class used to relate predicated instructions with their
// predicate-new forms.
class PredNewRel: PredRel;
// NewValueRel - Filter class used to relate regular store instructions with
// their new-value store form.
class NewValueRel: PredNewRel;
// NewValueRel - Filter class used to relate load/store instructions having
// different addressing modes with each other.
class AddrModeRel: NewValueRel;
class PostInc_BaseImm;
class IntrinsicsRel;

//===----------------------------------------------------------------------===//
// Generate mapping table to relate non-predicate instructions with their
// predicated formats - true and false.
//

def getPredOpcode : InstrMapping {
  let FilterClass = "PredRel";
  // Instructions with the same BaseOpcode and isNVStore values form a row.
  let RowFields = ["BaseOpcode", "isNVStore", "PNewValue", "isBrTaken", "isNT"];
  // Instructions with the same predicate sense form a column.
  let ColFields = ["PredSense"];
  // The key column is the unpredicated instructions.
  let KeyCol = [""];
  // Value columns are PredSense=true and PredSense=false
  let ValueCols = [["true"], ["false"]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate predicate-true instructions with their
// predicate-false forms
//
def getFalsePredOpcode : InstrMapping {
  let FilterClass = "PredRel";
  let RowFields = ["BaseOpcode", "PNewValue", "isNVStore", "isBrTaken", "isNT"];
  let ColFields = ["PredSense"];
  let KeyCol = ["true"];
  let ValueCols = [["false"]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate predicate-false instructions with their
// predicate-true forms
//
def getTruePredOpcode : InstrMapping {
  let FilterClass = "PredRel";
  let RowFields = ["BaseOpcode", "PNewValue", "isNVStore", "isBrTaken", "isNT"];
  let ColFields = ["PredSense"];
  let KeyCol = ["false"];
  let ValueCols = [["true"]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate predicated instructions with their .new
// format.
//
def getPredNewOpcode : InstrMapping {
  let FilterClass = "PredNewRel";
  let RowFields = ["BaseOpcode", "PredSense", "isNVStore", "isBrTaken"];
  let ColFields = ["PNewValue"];
  let KeyCol = [""];
  let ValueCols = [["new"]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate .new predicated instructions with their old
// format.
//
def getPredOldOpcode : InstrMapping {
  let FilterClass = "PredNewRel";
  let RowFields = ["BaseOpcode", "PredSense", "isNVStore", "isBrTaken"];
  let ColFields = ["PNewValue"];
  let KeyCol = ["new"];
  let ValueCols = [[""]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate store instructions with their new-value
// format.
//
def getNewValueOpcode : InstrMapping {
  let FilterClass = "NewValueRel";
  let RowFields = ["BaseOpcode", "PredSense", "PNewValue", "addrMode", "isNT"];
  let ColFields = ["NValueST"];
  let KeyCol = ["false"];
  let ValueCols = [["true"]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate new-value store instructions with their old
// format.
//
def getNonNVStore : InstrMapping {
  let FilterClass = "NewValueRel";
  let RowFields = ["BaseOpcode", "PredSense", "PNewValue", "addrMode", "isNT"];
  let ColFields = ["NValueST"];
  let KeyCol = ["true"];
  let ValueCols = [["false"]];
}

def changeAddrMode_abs_io: InstrMapping {
  let FilterClass = "AddrModeRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore",
                   "isFloat"];
  let ColFields = ["addrMode"];
  let KeyCol = ["Absolute"];
  let ValueCols = [["BaseImmOffset"]];
}

def changeAddrMode_io_abs: InstrMapping {
  let FilterClass = "AddrModeRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore",
                   "isFloat"];
  let ColFields = ["addrMode"];
  let KeyCol = ["BaseImmOffset"];
  let ValueCols = [["Absolute"]];
}

def changeAddrMode_io_rr: InstrMapping {
  let FilterClass = "AddrModeRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
  let ColFields = ["addrMode"];
  let KeyCol = ["BaseImmOffset"];
  let ValueCols = [["BaseRegOffset"]];
}

def changeAddrMode_rr_io: InstrMapping {
  let FilterClass = "AddrModeRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
  let ColFields = ["addrMode"];
  let KeyCol = ["BaseRegOffset"];
  let ValueCols = [["BaseImmOffset"]];
}

def changeAddrMode_pi_io: InstrMapping {
  let FilterClass = "PostInc_BaseImm";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
  let ColFields = ["addrMode"];
  let KeyCol = ["PostInc"];
  let ValueCols = [["BaseImmOffset"]];
}

def changeAddrMode_io_pi: InstrMapping {
  let FilterClass = "PostInc_BaseImm";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
  let ColFields = ["addrMode"];
  let KeyCol = ["BaseImmOffset"];
  let ValueCols = [["PostInc"]];
}

def changeAddrMode_rr_ur: InstrMapping {
  let FilterClass = "ImmRegShl";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
  let ColFields = ["addrMode"];
  let KeyCol = ["BaseRegOffset"];
  let ValueCols = [["BaseLongOffset"]];
}

def changeAddrMode_ur_rr : InstrMapping {
  let FilterClass = "ImmRegShl";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
  let ColFields = ["addrMode"];
  let KeyCol = ["BaseLongOffset"];
  let ValueCols = [["BaseRegOffset"]];
}

def getRegForm : InstrMapping {
  let FilterClass = "ImmRegRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue"];
  let ColFields = ["InputType"];
  let KeyCol = ["imm"];
  let ValueCols = [["reg"]];
}

def notTakenBranchPrediction : InstrMapping {
  let FilterClass = "PredRel";
  let RowFields = ["BaseOpcode", "PNewValue",  "PredSense", "isBranch", "isPredicated"];
  let ColFields = ["isBrTaken"];
  let KeyCol = ["true"];
  let ValueCols = [["false"]];
}

def takenBranchPrediction : InstrMapping {
  let FilterClass = "PredRel";
  let RowFields = ["BaseOpcode", "PNewValue",  "PredSense", "isBranch", "isPredicated"];
  let ColFields = ["isBrTaken"];
  let KeyCol = ["false"];
  let ValueCols = [["true"]];
}

def getRealHWInstr : InstrMapping {
  let FilterClass = "IntrinsicsRel";
  let RowFields = ["BaseOpcode"];
  let ColFields = ["InstrType"];
  let KeyCol = ["Pseudo"];
  let ValueCols = [["Pseudo"], ["Real"]];
}
//===----------------------------------------------------------------------===//
// Register File, Instruction Descriptions
//===----------------------------------------------------------------------===//
include "HexagonSchedule.td"
include "HexagonRegisterInfo.td"
include "HexagonOperands.td"
include "HexagonDepOperands.td"
include "HexagonDepITypes.td"
include "HexagonInstrFormats.td"
include "HexagonDepInstrFormats.td"
include "HexagonDepInstrInfo.td"
include "HexagonCallingConv.td"
include "HexagonPseudo.td"
include "HexagonPatterns.td"
include "HexagonPatternsHVX.td"
include "HexagonPatternsV65.td"
include "HexagonDepMappings.td"
include "HexagonIntrinsics.td"
include "HexagonMapAsm2IntrinV62.gen.td"
include "HexagonMapAsm2IntrinV65.gen.td"

def HexagonInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Hexagon processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, SchedMachineModel Model,
           list<SubtargetFeature> Features>
 : ProcessorModel<Name, Model, Features>;

def : Proc<"hexagonv4",  HexagonModelV4,
           [ArchV4, FeaturePackets, FeatureNVJ, FeatureDuplex]>;
def : Proc<"hexagonv5",  HexagonModelV4,
           [ArchV4, ArchV5, FeaturePackets, FeatureNVJ, FeatureDuplex]>;
def : Proc<"hexagonv55", HexagonModelV55,
           [ArchV4, ArchV5, ArchV55, FeaturePackets, FeatureNVJ,
            FeatureDuplex]>;
def : Proc<"hexagonv60", HexagonModelV60,
           [ArchV4, ArchV5, ArchV55, ArchV60, FeaturePackets, FeatureNVJ,
            FeatureDuplex]>;
def : Proc<"hexagonv62", HexagonModelV62,
           [ArchV4, ArchV5, ArchV55, ArchV60, ArchV62, FeaturePackets,
            FeatureNVJ, FeatureDuplex]>;
def : Proc<"hexagonv65", HexagonModelV65,
           [ArchV4, ArchV5, ArchV55, ArchV60, ArchV62, ArchV65,
            FeatureMemNoShuf, FeaturePackets, FeatureNVJ, FeatureDuplex]>;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def HexagonAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
  bit HasMnemonicFirst = 0;
}

def HexagonAsmParserVariant : AsmParserVariant {
  int Variant = 0;
  string TokenizingCharacters = "#()=:.<>!+*-|^&";
  string BreakCharacters = "";
}

def HexagonAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

def Hexagon : Target {
  let InstructionSet = HexagonInstrInfo;
  let AssemblyParsers = [HexagonAsmParser];
  let AssemblyParserVariants = [HexagonAsmParserVariant];
  let AssemblyWriters = [HexagonAsmWriter];
  let AllowRegisterRenaming = 1;
}
