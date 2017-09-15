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
def ExtensionHVX: SubtargetFeature<"hvx", "UseHVXOps", "true",
      "Hexagon HVX instructions">;
def ExtensionHVXDbl: SubtargetFeature<"hvx-double", "UseHVXDblOps", "true",
      "Hexagon HVX Double instructions">;
def FeatureLongCalls: SubtargetFeature<"long-calls", "UseLongCalls", "true",
      "Use constant-extended calls">;

//===----------------------------------------------------------------------===//
// Hexagon Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//

def UseMEMOP           : Predicate<"HST->useMemOps()">;
def IEEERndNearV5T     : Predicate<"HST->modeIEEERndNear()">;
def UseHVXDbl          : Predicate<"HST->useHVXDblOps()">,
                         AssemblerPredicate<"ExtensionHVXDbl">;
def UseHVXSgl          : Predicate<"HST->useHVXSglOps()">;
def UseHVX             : Predicate<"HST->useHVXSglOps() ||HST->useHVXDblOps()">,
                         AssemblerPredicate<"ExtensionHVX">;

def Hvx64   : HwMode<"+hvx,-hvx-double">;
def Hvx128  : HwMode<"+hvx,+hvx-double">;

//===----------------------------------------------------------------------===//
// Classes used for relation maps.
//===----------------------------------------------------------------------===//

class ImmRegShl;
// PredRel - Filter class used to relate non-predicated instructions with their
// predicated forms.
class PredRel;
// PredNewRel - Filter class used to relate predicated instructions with their
// predicate-new forms.
class PredNewRel: PredRel;
// ImmRegRel - Filter class used to relate instructions having reg-reg form
// with their reg-imm counterparts.
class ImmRegRel;
// NewValueRel - Filter class used to relate regular store instructions with
// their new-value store form.
class NewValueRel: PredNewRel;
// NewValueRel - Filter class used to relate load/store instructions having
// different addressing modes with each other.
class AddrModeRel: NewValueRel;
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

def getBaseWithImmOffset : InstrMapping {
  let FilterClass = "AddrModeRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore",
                   "isFloat"];
  let ColFields = ["addrMode"];
  let KeyCol = ["Absolute"];
  let ValueCols = [["BaseImmOffset"]];
}

def getAbsoluteForm : InstrMapping {
  let FilterClass = "AddrModeRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore",
                   "isFloat"];
  let ColFields = ["addrMode"];
  let KeyCol = ["BaseImmOffset"];
  let ValueCols = [["Absolute"]];
}

def getBaseWithRegOffset : InstrMapping {
  let FilterClass = "AddrModeRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
  let ColFields = ["addrMode"];
  let KeyCol = ["BaseImmOffset"];
  let ValueCols = [["BaseRegOffset"]];
}

def xformRegToImmOffset : InstrMapping {
  let FilterClass = "AddrModeRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
  let ColFields = ["addrMode"];
  let KeyCol = ["BaseRegOffset"];
  let ValueCols = [["BaseImmOffset"]];
}

def getBaseWithLongOffset : InstrMapping {
  let FilterClass = "ImmRegShl";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
  let ColFields = ["addrMode"];
  let KeyCol = ["BaseRegOffset"];
  let ValueCols = [["BaseLongOffset"]];
}

def getRegForm : InstrMapping {
  let FilterClass = "ImmRegRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue"];
  let ColFields = ["InputType"];
  let KeyCol = ["imm"];
  let ValueCols = [["reg"]];
}

def getRegShlForm : InstrMapping {
  let FilterClass = "ImmRegShl";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
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
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//
include "HexagonSchedule.td"
include "HexagonRegisterInfo.td"
include "HexagonOperands.td"
include "HexagonDepOperands.td"
include "HexagonDepITypes.td"
include "HexagonInstrFormats.td"
include "HexagonDepInstrFormats.td"
include "HexagonDepInstrInfo.td"
include "HexagonPseudo.td"
include "HexagonPatterns.td"
include "HexagonDepMappings.td"
include "HexagonIntrinsics.td"
include "HexagonIntrinsicsDerived.td"
include "HexagonMapAsm2IntrinV62.gen.td"

def HexagonInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Hexagon processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, SchedMachineModel Model,
           list<SubtargetFeature> Features>
 : ProcessorModel<Name, Model, Features>;

def : Proc<"hexagonv4",  HexagonModelV4,
           [ArchV4]>;
def : Proc<"hexagonv5",  HexagonModelV4,
           [ArchV4, ArchV5]>;
def : Proc<"hexagonv55", HexagonModelV55,
           [ArchV4, ArchV5, ArchV55]>;
def : Proc<"hexagonv60", HexagonModelV60,
           [ArchV4, ArchV5, ArchV55, ArchV60, ExtensionHVX]>;
def : Proc<"hexagonv62", HexagonModelV62,
           [ArchV4, ArchV5, ArchV55, ArchV60, ArchV62, ExtensionHVX]>;

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
}

def Hexagon : Target {
  // Pull in Instruction Info:
  let InstructionSet = HexagonInstrInfo;
  let AssemblyParsers = [HexagonAsmParser];
  let AssemblyParserVariants = [HexagonAsmParserVariant];
}
