//===-- PPC.td - Describe the PowerPC Target Machine -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level entry point for the PowerPC target.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing.
//
include "llvm/Target/Target.td"

include "PPCFeatures.td"

//===----------------------------------------------------------------------===//
// Classes used for relation maps.
//===----------------------------------------------------------------------===//
// RecFormRel - Filter class used to relate non-record-form instructions with
// their record-form variants.
class RecFormRel;

// AltVSXFMARel - Filter class used to relate the primary addend-killing VSX
// FMA instruction forms with their corresponding factor-killing forms.
class AltVSXFMARel {
  bit IsVSXFMAAlt = 0;
}

//===----------------------------------------------------------------------===//
// Relation Map Definitions.
//===----------------------------------------------------------------------===//

def getRecordFormOpcode : InstrMapping {
  let FilterClass = "RecFormRel";
  // Instructions with the same BaseName and Interpretation64Bit values
  // form a row.
  let RowFields = ["BaseName", "Interpretation64Bit"];
  // Instructions with the same RC value form a column.
  let ColFields = ["RC"];
  // The key column are the non-record-form instructions.
  let KeyCol = ["0"];
  // Value columns RC=1
  let ValueCols = [["1"]];
}

def getNonRecordFormOpcode : InstrMapping {
  let FilterClass = "RecFormRel";
  // Instructions with the same BaseName and Interpretation64Bit values
  // form a row.
  let RowFields = ["BaseName", "Interpretation64Bit"];
  // Instructions with the same RC value form a column.
  let ColFields = ["RC"];
  // The key column are the record-form instructions.
  let KeyCol = ["1"];
  // Value columns are RC=0
  let ValueCols = [["0"]];
}

def getAltVSXFMAOpcode : InstrMapping {
  let FilterClass = "AltVSXFMARel";
  // Instructions with the same BaseName value form a row.
  let RowFields = ["BaseName"];
  // Instructions with the same IsVSXFMAAlt value form a column.
  let ColFields = ["IsVSXFMAAlt"];
  // The key column are the (default) addend-killing instructions.
  let KeyCol = ["0"];
  // Value columns IsVSXFMAAlt=1
  let ValueCols = [["1"]];
}

def get64BitInstrFromSignedExt32BitInstr : InstrMapping {
  let FilterClass = "SExt32To64";
  // Instructions with the same opcode.
  let RowFields = ["Inst"];
  // Instructions with the same Interpretation64Bit value form a column.
  let ColFields = ["Interpretation64Bit"];
  // The key column are not the Interpretation64Bit-form instructions.
  let KeyCol = ["0"];
  // Value columns are the Interpretation64Bit-form instructions.
  let ValueCols = [["1"]];
}

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "PPCRegisterInfo.td"
include "PPCSchedule.td"
include "GISel/PPCRegisterBanks.td"

include "PPCProcessors.td"

//===----------------------------------------------------------------------===//
// Calling Conventions
//===----------------------------------------------------------------------===//

include "PPCCallingConv.td"

def PPCInstrInfo : InstrInfo {
  let isLittleEndianEncoding = 1;
}

def PPCAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  int PassSubtarget = 1;
  int Variant = 0;
  bit isMCAsmWriter = 1;
}

def PPCAsmParser : AsmParser {
  let ShouldEmitMatchRegisterName = 1;
  let AllowDuplicateRegisterNames = 1;
}

def PPCAsmParserVariant : AsmParserVariant {
  int Variant = 0;

  // We do not use hard coded registers in asm strings.  However, some
  // InstAlias definitions use immediate literals.  Set RegisterPrefix
  // so that those are not misinterpreted as registers.
  string RegisterPrefix = "%";
  string BreakCharacters = ".";
}

defm : RemapAllTargetPseudoPointerOperands<ppc_ptr_rc>;

def PPC : Target {
  // Information about the instructions.
  let InstructionSet = PPCInstrInfo;

  let AssemblyWriters = [PPCAsmWriter];
  let AssemblyParsers = [PPCAsmParser];
  let AssemblyParserVariants = [PPCAsmParserVariant];
  let AllowRegisterRenaming = 1;
}

//===----------------------------------------------------------------------===//
// Pfm Counters
//===----------------------------------------------------------------------===//

include "PPCPfmCounters.td"
