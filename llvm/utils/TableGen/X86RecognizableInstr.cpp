//===- X86RecognizableInstr.cpp - Disassembler instruction spec --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the X86 Disassembler Emitter.
// It contains the implementation of a single recognizable instruction.
// Documentation for the disassembler emitter in general can be found in
//  X86DisasemblerEmitter.h.
//
//===----------------------------------------------------------------------===//

#include "X86DisassemblerShared.h"
#include "X86RecognizableInstr.h"
#include "X86ModRMFilters.h"

#include "llvm/Support/ErrorHandling.h"

#include <string>

using namespace llvm;

#define MRM_MAPPING     \
  MAP(C1, 33)           \
  MAP(C2, 34)           \
  MAP(C3, 35)           \
  MAP(C4, 36)           \
  MAP(C8, 37)           \
  MAP(C9, 38)           \
  MAP(E8, 39)           \
  MAP(F0, 40)           \
  MAP(F8, 41)           \
  MAP(F9, 42)           \
  MAP(D0, 45)           \
  MAP(D1, 46)           \
  MAP(D4, 47)           \
  MAP(D5, 48)           \
  MAP(D8, 49)           \
  MAP(D9, 50)           \
  MAP(DA, 51)           \
  MAP(DB, 52)           \
  MAP(DC, 53)           \
  MAP(DD, 54)           \
  MAP(DE, 55)           \
  MAP(DF, 56)

// A clone of X86 since we can't depend on something that is generated.
namespace X86Local {
  enum {
    Pseudo      = 0,
    RawFrm      = 1,
    AddRegFrm   = 2,
    MRMDestReg  = 3,
    MRMDestMem  = 4,
    MRMSrcReg   = 5,
    MRMSrcMem   = 6,
    MRM0r = 16, MRM1r = 17, MRM2r = 18, MRM3r = 19,
    MRM4r = 20, MRM5r = 21, MRM6r = 22, MRM7r = 23,
    MRM0m = 24, MRM1m = 25, MRM2m = 26, MRM3m = 27,
    MRM4m = 28, MRM5m = 29, MRM6m = 30, MRM7m = 31,
    MRMInitReg  = 32,
    RawFrmImm8  = 43,
    RawFrmImm16 = 44,
#define MAP(from, to) MRM_##from = to,
    MRM_MAPPING
#undef MAP
    lastMRM
  };

  enum {
    TB  = 1,
    REP = 2,
    D8 = 3, D9 = 4, DA = 5, DB = 6,
    DC = 7, DD = 8, DE = 9, DF = 10,
    XD = 11,  XS = 12,
    T8 = 13,  P_TA = 14,
    A6 = 15,  A7 = 16, T8XD = 17, T8XS = 18, TAXD = 19
  };
}

// If rows are added to the opcode extension tables, then corresponding entries
// must be added here.
//
// If the row corresponds to a single byte (i.e., 8f), then add an entry for
// that byte to ONE_BYTE_EXTENSION_TABLES.
//
// If the row corresponds to two bytes where the first is 0f, add an entry for
// the second byte to TWO_BYTE_EXTENSION_TABLES.
//
// If the row corresponds to some other set of bytes, you will need to modify
// the code in RecognizableInstr::emitDecodePath() as well, and add new prefixes
// to the X86 TD files, except in two cases: if the first two bytes of such a
// new combination are 0f 38 or 0f 3a, you just have to add maps called
// THREE_BYTE_38_EXTENSION_TABLES and THREE_BYTE_3A_EXTENSION_TABLES and add a
// switch(Opcode) just below the case X86Local::T8: or case X86Local::TA: line
// in RecognizableInstr::emitDecodePath().

#define ONE_BYTE_EXTENSION_TABLES \
  EXTENSION_TABLE(80)             \
  EXTENSION_TABLE(81)             \
  EXTENSION_TABLE(82)             \
  EXTENSION_TABLE(83)             \
  EXTENSION_TABLE(8f)             \
  EXTENSION_TABLE(c0)             \
  EXTENSION_TABLE(c1)             \
  EXTENSION_TABLE(c6)             \
  EXTENSION_TABLE(c7)             \
  EXTENSION_TABLE(d0)             \
  EXTENSION_TABLE(d1)             \
  EXTENSION_TABLE(d2)             \
  EXTENSION_TABLE(d3)             \
  EXTENSION_TABLE(f6)             \
  EXTENSION_TABLE(f7)             \
  EXTENSION_TABLE(fe)             \
  EXTENSION_TABLE(ff)

#define TWO_BYTE_EXTENSION_TABLES \
  EXTENSION_TABLE(00)             \
  EXTENSION_TABLE(01)             \
  EXTENSION_TABLE(18)             \
  EXTENSION_TABLE(71)             \
  EXTENSION_TABLE(72)             \
  EXTENSION_TABLE(73)             \
  EXTENSION_TABLE(ae)             \
  EXTENSION_TABLE(ba)             \
  EXTENSION_TABLE(c7)

#define THREE_BYTE_38_EXTENSION_TABLES \
  EXTENSION_TABLE(F3)

using namespace X86Disassembler;

/// needsModRMForDecode - Indicates whether a particular instruction requires a
///   ModR/M byte for the instruction to be properly decoded.  For example, a
///   MRMDestReg instruction needs the Mod field in the ModR/M byte to be set to
///   0b11.
///
/// @param form - The form of the instruction.
/// @return     - true if the form implies that a ModR/M byte is required, false
///               otherwise.
static bool needsModRMForDecode(uint8_t form) {
  if (form == X86Local::MRMDestReg    ||
     form == X86Local::MRMDestMem    ||
     form == X86Local::MRMSrcReg     ||
     form == X86Local::MRMSrcMem     ||
     (form >= X86Local::MRM0r && form <= X86Local::MRM7r) ||
     (form >= X86Local::MRM0m && form <= X86Local::MRM7m))
    return true;
  else
    return false;
}

/// isRegFormat - Indicates whether a particular form requires the Mod field of
///   the ModR/M byte to be 0b11.
///
/// @param form - The form of the instruction.
/// @return     - true if the form implies that Mod must be 0b11, false
///               otherwise.
static bool isRegFormat(uint8_t form) {
  if (form == X86Local::MRMDestReg ||
     form == X86Local::MRMSrcReg  ||
     (form >= X86Local::MRM0r && form <= X86Local::MRM7r))
    return true;
  else
    return false;
}

/// byteFromBitsInit - Extracts a value at most 8 bits in width from a BitsInit.
///   Useful for switch statements and the like.
///
/// @param init - A reference to the BitsInit to be decoded.
/// @return     - The field, with the first bit in the BitsInit as the lowest
///               order bit.
static uint8_t byteFromBitsInit(BitsInit &init) {
  int width = init.getNumBits();

  assert(width <= 8 && "Field is too large for uint8_t!");

  int     index;
  uint8_t mask = 0x01;

  uint8_t ret = 0;

  for (index = 0; index < width; index++) {
    if (static_cast<BitInit*>(init.getBit(index))->getValue())
      ret |= mask;

    mask <<= 1;
  }

  return ret;
}

/// byteFromRec - Extract a value at most 8 bits in with from a Record given the
///   name of the field.
///
/// @param rec  - The record from which to extract the value.
/// @param name - The name of the field in the record.
/// @return     - The field, as translated by byteFromBitsInit().
static uint8_t byteFromRec(const Record* rec, const std::string &name) {
  BitsInit* bits = rec->getValueAsBitsInit(name);
  return byteFromBitsInit(*bits);
}

RecognizableInstr::RecognizableInstr(DisassemblerTables &tables,
                                     const CodeGenInstruction &insn,
                                     InstrUID uid) {
  UID = uid;

  Rec = insn.TheDef;
  Name = Rec->getName();
  Spec = &tables.specForUID(UID);

  if (!Rec->isSubClassOf("X86Inst")) {
    ShouldBeEmitted = false;
    return;
  }

  Prefix   = byteFromRec(Rec, "Prefix");
  Opcode   = byteFromRec(Rec, "Opcode");
  Form     = byteFromRec(Rec, "FormBits");
  SegOvr   = byteFromRec(Rec, "SegOvrBits");

  HasOpSizePrefix  = Rec->getValueAsBit("hasOpSizePrefix");
  HasAdSizePrefix  = Rec->getValueAsBit("hasAdSizePrefix");
  HasREX_WPrefix   = Rec->getValueAsBit("hasREX_WPrefix");
  HasVEXPrefix     = Rec->getValueAsBit("hasVEXPrefix");
  HasVEX_4VPrefix  = Rec->getValueAsBit("hasVEX_4VPrefix");
  HasVEX_4VOp3Prefix = Rec->getValueAsBit("hasVEX_4VOp3Prefix");
  HasVEX_WPrefix   = Rec->getValueAsBit("hasVEX_WPrefix");
  HasMemOp4Prefix  = Rec->getValueAsBit("hasMemOp4Prefix");
  IgnoresVEX_L     = Rec->getValueAsBit("ignoresVEX_L");
  HasLockPrefix    = Rec->getValueAsBit("hasLockPrefix");
  IsCodeGenOnly    = Rec->getValueAsBit("isCodeGenOnly");

  Name      = Rec->getName();
  AsmString = Rec->getValueAsString("AsmString");

  Operands = &insn.Operands.OperandList;

  IsSSE            = (HasOpSizePrefix && (Name.find("16") == Name.npos)) ||
                     (Name.find("CRC32") != Name.npos);
  HasFROperands    = hasFROperands();
  HasVEX_LPrefix   = Rec->getValueAsBit("hasVEX_L");

  // Check for 64-bit inst which does not require REX
  Is32Bit = false;
  Is64Bit = false;
  // FIXME: Is there some better way to check for In64BitMode?
  std::vector<Record*> Predicates = Rec->getValueAsListOfDefs("Predicates");
  for (unsigned i = 0, e = Predicates.size(); i != e; ++i) {
    if (Predicates[i]->getName().find("32Bit") != Name.npos) {
      Is32Bit = true;
      break;
    }
    if (Predicates[i]->getName().find("64Bit") != Name.npos) {
      Is64Bit = true;
      break;
    }
  }
  // FIXME: These instructions aren't marked as 64-bit in any way
  Is64Bit |= Rec->getName() == "JMP64pcrel32" ||
             Rec->getName() == "MASKMOVDQU64" ||
             Rec->getName() == "POPFS64" ||
             Rec->getName() == "POPGS64" ||
             Rec->getName() == "PUSHFS64" ||
             Rec->getName() == "PUSHGS64" ||
             Rec->getName() == "REX64_PREFIX" ||
             Rec->getName().find("MOV64") != Name.npos ||
             Rec->getName().find("PUSH64") != Name.npos ||
             Rec->getName().find("POP64") != Name.npos;

  ShouldBeEmitted  = true;
}

void RecognizableInstr::processInstr(DisassemblerTables &tables,
                                     const CodeGenInstruction &insn,
                                     InstrUID uid)
{
  // Ignore "asm parser only" instructions.
  if (insn.TheDef->getValueAsBit("isAsmParserOnly"))
    return;

  RecognizableInstr recogInstr(tables, insn, uid);

  recogInstr.emitInstructionSpecifier(tables);

  if (recogInstr.shouldBeEmitted())
    recogInstr.emitDecodePath(tables);
}

InstructionContext RecognizableInstr::insnContext() const {
  InstructionContext insnContext;

  if (HasVEX_4VPrefix || HasVEX_4VOp3Prefix|| HasVEXPrefix) {
    if (HasVEX_LPrefix && HasVEX_WPrefix) {
      if (HasOpSizePrefix)
        insnContext = IC_VEX_L_W_OPSIZE;
      else
        llvm_unreachable("Don't support VEX.L and VEX.W together");
    } else if (HasOpSizePrefix && HasVEX_LPrefix)
      insnContext = IC_VEX_L_OPSIZE;
    else if (HasOpSizePrefix && HasVEX_WPrefix)
      insnContext = IC_VEX_W_OPSIZE;
    else if (HasOpSizePrefix)
      insnContext = IC_VEX_OPSIZE;
    else if (HasVEX_LPrefix &&
             (Prefix == X86Local::XS || Prefix == X86Local::T8XS))
      insnContext = IC_VEX_L_XS;
    else if (HasVEX_LPrefix && (Prefix == X86Local::XD ||
                                Prefix == X86Local::T8XD ||
                                Prefix == X86Local::TAXD))
      insnContext = IC_VEX_L_XD;
    else if (HasVEX_WPrefix &&
             (Prefix == X86Local::XS || Prefix == X86Local::T8XS))
      insnContext = IC_VEX_W_XS;
    else if (HasVEX_WPrefix && (Prefix == X86Local::XD ||
                                Prefix == X86Local::T8XD ||
                                Prefix == X86Local::TAXD))
      insnContext = IC_VEX_W_XD;
    else if (HasVEX_WPrefix)
      insnContext = IC_VEX_W;
    else if (HasVEX_LPrefix)
      insnContext = IC_VEX_L;
    else if (Prefix == X86Local::XD || Prefix == X86Local::T8XD ||
             Prefix == X86Local::TAXD)
      insnContext = IC_VEX_XD;
    else if (Prefix == X86Local::XS || Prefix == X86Local::T8XS)
      insnContext = IC_VEX_XS;
    else
      insnContext = IC_VEX;
  } else if (Is64Bit || HasREX_WPrefix) {
    if (HasREX_WPrefix && HasOpSizePrefix)
      insnContext = IC_64BIT_REXW_OPSIZE;
    else if (HasOpSizePrefix && (Prefix == X86Local::XD ||
                                 Prefix == X86Local::T8XD ||
                                 Prefix == X86Local::TAXD))
      insnContext = IC_64BIT_XD_OPSIZE;
    else if (HasOpSizePrefix &&
             (Prefix == X86Local::XS || Prefix == X86Local::T8XS))
      insnContext = IC_64BIT_XS_OPSIZE;
    else if (HasOpSizePrefix)
      insnContext = IC_64BIT_OPSIZE;
    else if (HasAdSizePrefix)
      insnContext = IC_64BIT_ADSIZE;
    else if (HasREX_WPrefix &&
             (Prefix == X86Local::XS || Prefix == X86Local::T8XS))
      insnContext = IC_64BIT_REXW_XS;
    else if (HasREX_WPrefix && (Prefix == X86Local::XD ||
                                Prefix == X86Local::T8XD ||
                                Prefix == X86Local::TAXD))
      insnContext = IC_64BIT_REXW_XD;
    else if (Prefix == X86Local::XD || Prefix == X86Local::T8XD ||
             Prefix == X86Local::TAXD)
      insnContext = IC_64BIT_XD;
    else if (Prefix == X86Local::XS || Prefix == X86Local::T8XS)
      insnContext = IC_64BIT_XS;
    else if (HasREX_WPrefix)
      insnContext = IC_64BIT_REXW;
    else
      insnContext = IC_64BIT;
  } else {
    if (HasOpSizePrefix && (Prefix == X86Local::XD ||
                            Prefix == X86Local::T8XD ||
                            Prefix == X86Local::TAXD))
      insnContext = IC_XD_OPSIZE;
    else if (HasOpSizePrefix &&
             (Prefix == X86Local::XS || Prefix == X86Local::T8XS))
      insnContext = IC_XS_OPSIZE;
    else if (HasOpSizePrefix)
      insnContext = IC_OPSIZE;
    else if (HasAdSizePrefix)
      insnContext = IC_ADSIZE;
    else if (Prefix == X86Local::XD || Prefix == X86Local::T8XD ||
             Prefix == X86Local::TAXD)
      insnContext = IC_XD;
    else if (Prefix == X86Local::XS || Prefix == X86Local::T8XS ||
             Prefix == X86Local::REP)
      insnContext = IC_XS;
    else
      insnContext = IC;
  }

  return insnContext;
}

RecognizableInstr::filter_ret RecognizableInstr::filter() const {
  ///////////////////
  // FILTER_STRONG
  //

  // Filter out intrinsics

  assert(Rec->isSubClassOf("X86Inst") && "Can only filter X86 instructions");

  if (Form == X86Local::Pseudo ||
      (IsCodeGenOnly && Name.find("_REV") == Name.npos))
    return FILTER_STRONG;


  // Filter out artificial instructions but leave in the LOCK_PREFIX so it is
  // printed as a separate "instruction".

  if (Name.find("_Int") != Name.npos       ||
      Name.find("Int_") != Name.npos)
    return FILTER_STRONG;

  // Filter out instructions with segment override prefixes.
  // They're too messy to handle now and we'll special case them if needed.

  if (SegOvr)
    return FILTER_STRONG;


  /////////////////
  // FILTER_WEAK
  //


  // Filter out instructions with a LOCK prefix;
  //   prefer forms that do not have the prefix
  if (HasLockPrefix)
    return FILTER_WEAK;

  // Filter out alternate forms of AVX instructions
  if (Name.find("_alt") != Name.npos ||
      Name.find("XrYr") != Name.npos ||
      (Name.find("r64r") != Name.npos && Name.find("r64r64") == Name.npos) ||
      Name.find("_64mr") != Name.npos ||
      Name.find("Xrr") != Name.npos ||
      Name.find("rr64") != Name.npos)
    return FILTER_WEAK;

  // Special cases.

  if (Name.find("PCMPISTRI") != Name.npos && Name != "PCMPISTRI")
    return FILTER_WEAK;
  if (Name.find("PCMPESTRI") != Name.npos && Name != "PCMPESTRI")
    return FILTER_WEAK;

  if (Name.find("MOV") != Name.npos && Name.find("r0") != Name.npos)
    return FILTER_WEAK;
  if (Name.find("MOVZ") != Name.npos && Name.find("MOVZX") == Name.npos)
    return FILTER_WEAK;
  if (Name.find("Fs") != Name.npos)
    return FILTER_WEAK;
  if (Name == "PUSH64i16"         ||
      Name == "MOVPQI2QImr"       ||
      Name == "VMOVPQI2QImr"      ||
      Name == "MMX_MOVD64rrv164"  ||
      Name == "MOV64ri64i32"      ||
      Name == "VMASKMOVDQU64"     ||
      Name == "VEXTRACTPSrr64"    ||
      Name == "VMOVQd64rr"        ||
      Name == "VMOVQs64rr")
    return FILTER_WEAK;

  if (HasFROperands && Name.find("MOV") != Name.npos &&
     ((Name.find("2") != Name.npos && Name.find("32") == Name.npos) ||
      (Name.find("to") != Name.npos)))
    return FILTER_STRONG;

  return FILTER_NORMAL;
}

bool RecognizableInstr::hasFROperands() const {
  const std::vector<CGIOperandList::OperandInfo> &OperandList = *Operands;
  unsigned numOperands = OperandList.size();

  for (unsigned operandIndex = 0; operandIndex < numOperands; ++operandIndex) {
    const std::string &recName = OperandList[operandIndex].Rec->getName();

    if (recName.find("FR") != recName.npos)
      return true;
  }
  return false;
}

void RecognizableInstr::handleOperand(bool optional, unsigned &operandIndex,
                                      unsigned &physicalOperandIndex,
                                      unsigned &numPhysicalOperands,
                                      const unsigned *operandMapping,
                                      OperandEncoding (*encodingFromString)
                                        (const std::string&,
                                         bool hasOpSizePrefix)) {
  if (optional) {
    if (physicalOperandIndex >= numPhysicalOperands)
      return;
  } else {
    assert(physicalOperandIndex < numPhysicalOperands);
  }

  while (operandMapping[operandIndex] != operandIndex) {
    Spec->operands[operandIndex].encoding = ENCODING_DUP;
    Spec->operands[operandIndex].type =
      (OperandType)(TYPE_DUP0 + operandMapping[operandIndex]);
    ++operandIndex;
  }

  const std::string &typeName = (*Operands)[operandIndex].Rec->getName();

  Spec->operands[operandIndex].encoding = encodingFromString(typeName,
                                                              HasOpSizePrefix);
  Spec->operands[operandIndex].type = typeFromString(typeName,
                                                     IsSSE,
                                                     HasREX_WPrefix,
                                                     HasOpSizePrefix);

  ++operandIndex;
  ++physicalOperandIndex;
}

void RecognizableInstr::emitInstructionSpecifier(DisassemblerTables &tables) {
  Spec->name       = Name;

  if (!ShouldBeEmitted)
    return;

  switch (filter()) {
  case FILTER_WEAK:
    Spec->filtered = true;
    break;
  case FILTER_STRONG:
    ShouldBeEmitted = false;
    return;
  case FILTER_NORMAL:
    break;
  }

  Spec->insnContext = insnContext();

  const std::vector<CGIOperandList::OperandInfo> &OperandList = *Operands;

  unsigned numOperands = OperandList.size();
  unsigned numPhysicalOperands = 0;

  // operandMapping maps from operands in OperandList to their originals.
  // If operandMapping[i] != i, then the entry is a duplicate.
  unsigned operandMapping[X86_MAX_OPERANDS];
  assert(numOperands <= X86_MAX_OPERANDS && "X86_MAX_OPERANDS is not large enough");

  for (unsigned operandIndex = 0; operandIndex < numOperands; ++operandIndex) {
    if (OperandList[operandIndex].Constraints.size()) {
      const CGIOperandList::ConstraintInfo &Constraint =
        OperandList[operandIndex].Constraints[0];
      if (Constraint.isTied()) {
        operandMapping[operandIndex] = operandIndex;
        operandMapping[Constraint.getTiedOperand()] = operandIndex;
      } else {
        ++numPhysicalOperands;
        operandMapping[operandIndex] = operandIndex;
      }
    } else {
      ++numPhysicalOperands;
      operandMapping[operandIndex] = operandIndex;
    }
  }

#define HANDLE_OPERAND(class)               \
  handleOperand(false,                      \
                operandIndex,               \
                physicalOperandIndex,       \
                numPhysicalOperands,        \
                operandMapping,             \
                class##EncodingFromString);

#define HANDLE_OPTIONAL(class)              \
  handleOperand(true,                       \
                operandIndex,               \
                physicalOperandIndex,       \
                numPhysicalOperands,        \
                operandMapping,             \
                class##EncodingFromString);

  // operandIndex should always be < numOperands
  unsigned operandIndex = 0;
  // physicalOperandIndex should always be < numPhysicalOperands
  unsigned physicalOperandIndex = 0;

  switch (Form) {
  case X86Local::RawFrm:
    // Operand 1 (optional) is an address or immediate.
    // Operand 2 (optional) is an immediate.
    assert(numPhysicalOperands <= 2 &&
           "Unexpected number of operands for RawFrm");
    HANDLE_OPTIONAL(relocation)
    HANDLE_OPTIONAL(immediate)
    break;
  case X86Local::AddRegFrm:
    // Operand 1 is added to the opcode.
    // Operand 2 (optional) is an address.
    assert(numPhysicalOperands >= 1 && numPhysicalOperands <= 2 &&
           "Unexpected number of operands for AddRegFrm");
    HANDLE_OPERAND(opcodeModifier)
    HANDLE_OPTIONAL(relocation)
    break;
  case X86Local::MRMDestReg:
    // Operand 1 is a register operand in the R/M field.
    // Operand 2 is a register operand in the Reg/Opcode field.
    // - In AVX, there is a register operand in the VEX.vvvv field here -
    // Operand 3 (optional) is an immediate.
    if (HasVEX_4VPrefix)
      assert(numPhysicalOperands >= 3 && numPhysicalOperands <= 4 &&
             "Unexpected number of operands for MRMDestRegFrm with VEX_4V");
    else
      assert(numPhysicalOperands >= 2 && numPhysicalOperands <= 3 &&
             "Unexpected number of operands for MRMDestRegFrm");

    HANDLE_OPERAND(rmRegister)

    if (HasVEX_4VPrefix)
      // FIXME: In AVX, the register below becomes the one encoded
      // in ModRMVEX and the one above the one in the VEX.VVVV field
      HANDLE_OPERAND(vvvvRegister)

    HANDLE_OPERAND(roRegister)
    HANDLE_OPTIONAL(immediate)
    break;
  case X86Local::MRMDestMem:
    // Operand 1 is a memory operand (possibly SIB-extended)
    // Operand 2 is a register operand in the Reg/Opcode field.
    // - In AVX, there is a register operand in the VEX.vvvv field here -
    // Operand 3 (optional) is an immediate.
    if (HasVEX_4VPrefix)
      assert(numPhysicalOperands >= 3 && numPhysicalOperands <= 4 &&
             "Unexpected number of operands for MRMDestMemFrm with VEX_4V");
    else
      assert(numPhysicalOperands >= 2 && numPhysicalOperands <= 3 &&
             "Unexpected number of operands for MRMDestMemFrm");
    HANDLE_OPERAND(memory)

    if (HasVEX_4VPrefix)
      // FIXME: In AVX, the register below becomes the one encoded
      // in ModRMVEX and the one above the one in the VEX.VVVV field
      HANDLE_OPERAND(vvvvRegister)

    HANDLE_OPERAND(roRegister)
    HANDLE_OPTIONAL(immediate)
    break;
  case X86Local::MRMSrcReg:
    // Operand 1 is a register operand in the Reg/Opcode field.
    // Operand 2 is a register operand in the R/M field.
    // - In AVX, there is a register operand in the VEX.vvvv field here -
    // Operand 3 (optional) is an immediate.
    // Operand 4 (optional) is an immediate.

    if (HasVEX_4VPrefix || HasVEX_4VOp3Prefix)
      assert(numPhysicalOperands >= 3 && numPhysicalOperands <= 5 &&
             "Unexpected number of operands for MRMSrcRegFrm with VEX_4V");
    else
      assert(numPhysicalOperands >= 2 && numPhysicalOperands <= 4 &&
             "Unexpected number of operands for MRMSrcRegFrm");

    HANDLE_OPERAND(roRegister)

    if (HasVEX_4VPrefix)
      // FIXME: In AVX, the register below becomes the one encoded
      // in ModRMVEX and the one above the one in the VEX.VVVV field
      HANDLE_OPERAND(vvvvRegister)

    if (HasMemOp4Prefix)
      HANDLE_OPERAND(immediate)

    HANDLE_OPERAND(rmRegister)

    if (HasVEX_4VOp3Prefix)
      HANDLE_OPERAND(vvvvRegister)

    if (!HasMemOp4Prefix)
      HANDLE_OPTIONAL(immediate)
    HANDLE_OPTIONAL(immediate) // above might be a register in 7:4
    HANDLE_OPTIONAL(immediate)
    break;
  case X86Local::MRMSrcMem:
    // Operand 1 is a register operand in the Reg/Opcode field.
    // Operand 2 is a memory operand (possibly SIB-extended)
    // - In AVX, there is a register operand in the VEX.vvvv field here -
    // Operand 3 (optional) is an immediate.

    if (HasVEX_4VPrefix || HasVEX_4VOp3Prefix)
      assert(numPhysicalOperands >= 3 && numPhysicalOperands <= 5 &&
             "Unexpected number of operands for MRMSrcMemFrm with VEX_4V");
    else
      assert(numPhysicalOperands >= 2 && numPhysicalOperands <= 3 &&
             "Unexpected number of operands for MRMSrcMemFrm");

    HANDLE_OPERAND(roRegister)

    if (HasVEX_4VPrefix)
      // FIXME: In AVX, the register below becomes the one encoded
      // in ModRMVEX and the one above the one in the VEX.VVVV field
      HANDLE_OPERAND(vvvvRegister)

    if (HasMemOp4Prefix)
      HANDLE_OPERAND(immediate)

    HANDLE_OPERAND(memory)

    if (HasVEX_4VOp3Prefix)
      HANDLE_OPERAND(vvvvRegister)

    if (!HasMemOp4Prefix)
      HANDLE_OPTIONAL(immediate)
    HANDLE_OPTIONAL(immediate) // above might be a register in 7:4
    break;
  case X86Local::MRM0r:
  case X86Local::MRM1r:
  case X86Local::MRM2r:
  case X86Local::MRM3r:
  case X86Local::MRM4r:
  case X86Local::MRM5r:
  case X86Local::MRM6r:
  case X86Local::MRM7r:
    // Operand 1 is a register operand in the R/M field.
    // Operand 2 (optional) is an immediate or relocation.
    // Operand 3 (optional) is an immediate.
    if (HasVEX_4VPrefix)
      assert(numPhysicalOperands <= 3 &&
             "Unexpected number of operands for MRMnRFrm with VEX_4V");
    else
      assert(numPhysicalOperands <= 3 &&
             "Unexpected number of operands for MRMnRFrm");
    if (HasVEX_4VPrefix)
      HANDLE_OPERAND(vvvvRegister)
    HANDLE_OPTIONAL(rmRegister)
    HANDLE_OPTIONAL(relocation)
    HANDLE_OPTIONAL(immediate)
    break;
  case X86Local::MRM0m:
  case X86Local::MRM1m:
  case X86Local::MRM2m:
  case X86Local::MRM3m:
  case X86Local::MRM4m:
  case X86Local::MRM5m:
  case X86Local::MRM6m:
  case X86Local::MRM7m:
    // Operand 1 is a memory operand (possibly SIB-extended)
    // Operand 2 (optional) is an immediate or relocation.
    if (HasVEX_4VPrefix)
      assert(numPhysicalOperands >= 2 && numPhysicalOperands <= 3 &&
             "Unexpected number of operands for MRMnMFrm");
    else
      assert(numPhysicalOperands >= 1 && numPhysicalOperands <= 2 &&
             "Unexpected number of operands for MRMnMFrm");
    if (HasVEX_4VPrefix)
      HANDLE_OPERAND(vvvvRegister)
    HANDLE_OPERAND(memory)
    HANDLE_OPTIONAL(relocation)
    break;
  case X86Local::RawFrmImm8:
    // operand 1 is a 16-bit immediate
    // operand 2 is an 8-bit immediate
    assert(numPhysicalOperands == 2 &&
           "Unexpected number of operands for X86Local::RawFrmImm8");
    HANDLE_OPERAND(immediate)
    HANDLE_OPERAND(immediate)
    break;
  case X86Local::RawFrmImm16:
    // operand 1 is a 16-bit immediate
    // operand 2 is a 16-bit immediate
    HANDLE_OPERAND(immediate)
    HANDLE_OPERAND(immediate)
    break;
  case X86Local::MRMInitReg:
    // Ignored.
    break;
  }

  #undef HANDLE_OPERAND
  #undef HANDLE_OPTIONAL
}

void RecognizableInstr::emitDecodePath(DisassemblerTables &tables) const {
  // Special cases where the LLVM tables are not complete

#define MAP(from, to)                     \
  case X86Local::MRM_##from:              \
    filter = new ExactFilter(0x##from);   \
    break;

  OpcodeType    opcodeType  = (OpcodeType)-1;

  ModRMFilter*  filter      = NULL;
  uint8_t       opcodeToSet = 0;

  switch (Prefix) {
  // Extended two-byte opcodes can start with f2 0f, f3 0f, or 0f
  case X86Local::XD:
  case X86Local::XS:
  case X86Local::TB:
    opcodeType = TWOBYTE;

    switch (Opcode) {
    default:
      if (needsModRMForDecode(Form))
        filter = new ModFilter(isRegFormat(Form));
      else
        filter = new DumbFilter();
      break;
#define EXTENSION_TABLE(n) case 0x##n:
    TWO_BYTE_EXTENSION_TABLES
#undef EXTENSION_TABLE
      switch (Form) {
      default:
        llvm_unreachable("Unhandled two-byte extended opcode");
      case X86Local::MRM0r:
      case X86Local::MRM1r:
      case X86Local::MRM2r:
      case X86Local::MRM3r:
      case X86Local::MRM4r:
      case X86Local::MRM5r:
      case X86Local::MRM6r:
      case X86Local::MRM7r:
        filter = new ExtendedFilter(true, Form - X86Local::MRM0r);
        break;
      case X86Local::MRM0m:
      case X86Local::MRM1m:
      case X86Local::MRM2m:
      case X86Local::MRM3m:
      case X86Local::MRM4m:
      case X86Local::MRM5m:
      case X86Local::MRM6m:
      case X86Local::MRM7m:
        filter = new ExtendedFilter(false, Form - X86Local::MRM0m);
        break;
      MRM_MAPPING
      } // switch (Form)
      break;
    } // switch (Opcode)
    opcodeToSet = Opcode;
    break;
  case X86Local::T8:
  case X86Local::T8XD:
  case X86Local::T8XS:
    opcodeType = THREEBYTE_38;
    switch (Opcode) {
    default:
      if (needsModRMForDecode(Form))
        filter = new ModFilter(isRegFormat(Form));
      else
        filter = new DumbFilter();
      break;
#define EXTENSION_TABLE(n) case 0x##n:
    THREE_BYTE_38_EXTENSION_TABLES
#undef EXTENSION_TABLE
      switch (Form) {
      default:
        llvm_unreachable("Unhandled two-byte extended opcode");
      case X86Local::MRM0r:
      case X86Local::MRM1r:
      case X86Local::MRM2r:
      case X86Local::MRM3r:
      case X86Local::MRM4r:
      case X86Local::MRM5r:
      case X86Local::MRM6r:
      case X86Local::MRM7r:
        filter = new ExtendedFilter(true, Form - X86Local::MRM0r);
        break;
      case X86Local::MRM0m:
      case X86Local::MRM1m:
      case X86Local::MRM2m:
      case X86Local::MRM3m:
      case X86Local::MRM4m:
      case X86Local::MRM5m:
      case X86Local::MRM6m:
      case X86Local::MRM7m:
        filter = new ExtendedFilter(false, Form - X86Local::MRM0m);
        break;
      MRM_MAPPING
      } // switch (Form)
      break;
    } // switch (Opcode)
    opcodeToSet = Opcode;
    break;
  case X86Local::P_TA:
  case X86Local::TAXD:
    opcodeType = THREEBYTE_3A;
    if (needsModRMForDecode(Form))
      filter = new ModFilter(isRegFormat(Form));
    else
      filter = new DumbFilter();
    opcodeToSet = Opcode;
    break;
  case X86Local::A6:
    opcodeType = THREEBYTE_A6;
    if (needsModRMForDecode(Form))
      filter = new ModFilter(isRegFormat(Form));
    else
      filter = new DumbFilter();
    opcodeToSet = Opcode;
    break;
  case X86Local::A7:
    opcodeType = THREEBYTE_A7;
    if (needsModRMForDecode(Form))
      filter = new ModFilter(isRegFormat(Form));
    else
      filter = new DumbFilter();
    opcodeToSet = Opcode;
    break;
  case X86Local::D8:
  case X86Local::D9:
  case X86Local::DA:
  case X86Local::DB:
  case X86Local::DC:
  case X86Local::DD:
  case X86Local::DE:
  case X86Local::DF:
    assert(Opcode >= 0xc0 && "Unexpected opcode for an escape opcode");
    opcodeType = ONEBYTE;
    if (Form == X86Local::AddRegFrm) {
      Spec->modifierType = MODIFIER_MODRM;
      Spec->modifierBase = Opcode;
      filter = new AddRegEscapeFilter(Opcode);
    } else {
      filter = new EscapeFilter(true, Opcode);
    }
    opcodeToSet = 0xd8 + (Prefix - X86Local::D8);
    break;
  case X86Local::REP:
  default:
    opcodeType = ONEBYTE;
    switch (Opcode) {
#define EXTENSION_TABLE(n) case 0x##n:
    ONE_BYTE_EXTENSION_TABLES
#undef EXTENSION_TABLE
      switch (Form) {
      default:
        llvm_unreachable("Fell through the cracks of a single-byte "
                         "extended opcode");
      case X86Local::MRM0r:
      case X86Local::MRM1r:
      case X86Local::MRM2r:
      case X86Local::MRM3r:
      case X86Local::MRM4r:
      case X86Local::MRM5r:
      case X86Local::MRM6r:
      case X86Local::MRM7r:
        filter = new ExtendedFilter(true, Form - X86Local::MRM0r);
        break;
      case X86Local::MRM0m:
      case X86Local::MRM1m:
      case X86Local::MRM2m:
      case X86Local::MRM3m:
      case X86Local::MRM4m:
      case X86Local::MRM5m:
      case X86Local::MRM6m:
      case X86Local::MRM7m:
        filter = new ExtendedFilter(false, Form - X86Local::MRM0m);
        break;
      MRM_MAPPING
      } // switch (Form)
      break;
    case 0xd8:
    case 0xd9:
    case 0xda:
    case 0xdb:
    case 0xdc:
    case 0xdd:
    case 0xde:
    case 0xdf:
      filter = new EscapeFilter(false, Form - X86Local::MRM0m);
      break;
    default:
      if (needsModRMForDecode(Form))
        filter = new ModFilter(isRegFormat(Form));
      else
        filter = new DumbFilter();
      break;
    } // switch (Opcode)
    opcodeToSet = Opcode;
  } // switch (Prefix)

  assert(opcodeType != (OpcodeType)-1 &&
         "Opcode type not set");
  assert(filter && "Filter not set");

  if (Form == X86Local::AddRegFrm) {
    if(Spec->modifierType != MODIFIER_MODRM) {
      assert(opcodeToSet < 0xf9 &&
             "Not enough room for all ADDREG_FRM operands");

      uint8_t currentOpcode;

      for (currentOpcode = opcodeToSet;
           currentOpcode < opcodeToSet + 8;
           ++currentOpcode)
        tables.setTableFields(opcodeType,
                              insnContext(),
                              currentOpcode,
                              *filter,
                              UID, Is32Bit, IgnoresVEX_L);

      Spec->modifierType = MODIFIER_OPCODE;
      Spec->modifierBase = opcodeToSet;
    } else {
      // modifierBase was set where MODIFIER_MODRM was set
      tables.setTableFields(opcodeType,
                            insnContext(),
                            opcodeToSet,
                            *filter,
                            UID, Is32Bit, IgnoresVEX_L);
    }
  } else {
    tables.setTableFields(opcodeType,
                          insnContext(),
                          opcodeToSet,
                          *filter,
                          UID, Is32Bit, IgnoresVEX_L);

    Spec->modifierType = MODIFIER_NONE;
    Spec->modifierBase = opcodeToSet;
  }

  delete filter;

#undef MAP
}

#define TYPE(str, type) if (s == str) return type;
OperandType RecognizableInstr::typeFromString(const std::string &s,
                                              bool isSSE,
                                              bool hasREX_WPrefix,
                                              bool hasOpSizePrefix) {
  if (isSSE) {
    // For SSE instructions, we ignore the OpSize prefix and force operand
    // sizes.
    TYPE("GR16",              TYPE_R16)
    TYPE("GR32",              TYPE_R32)
    TYPE("GR64",              TYPE_R64)
  }
  if(hasREX_WPrefix) {
    // For instructions with a REX_W prefix, a declared 32-bit register encoding
    // is special.
    TYPE("GR32",              TYPE_R32)
  }
  if(!hasOpSizePrefix) {
    // For instructions without an OpSize prefix, a declared 16-bit register or
    // immediate encoding is special.
    TYPE("GR16",              TYPE_R16)
    TYPE("i16imm",            TYPE_IMM16)
  }
  TYPE("i16mem",              TYPE_Mv)
  TYPE("i16imm",              TYPE_IMMv)
  TYPE("i16i8imm",            TYPE_IMMv)
  TYPE("GR16",                TYPE_Rv)
  TYPE("i32mem",              TYPE_Mv)
  TYPE("i32imm",              TYPE_IMMv)
  TYPE("i32i8imm",            TYPE_IMM32)
  TYPE("u32u8imm",            TYPE_IMM32)
  TYPE("GR32",                TYPE_Rv)
  TYPE("i64mem",              TYPE_Mv)
  TYPE("i64i32imm",           TYPE_IMM64)
  TYPE("i64i8imm",            TYPE_IMM64)
  TYPE("GR64",                TYPE_R64)
  TYPE("i8mem",               TYPE_M8)
  TYPE("i8imm",               TYPE_IMM8)
  TYPE("GR8",                 TYPE_R8)
  TYPE("VR128",               TYPE_XMM128)
  TYPE("f128mem",             TYPE_M128)
  TYPE("f256mem",             TYPE_M256)
  TYPE("FR64",                TYPE_XMM64)
  TYPE("f64mem",              TYPE_M64FP)
  TYPE("sdmem",               TYPE_M64FP)
  TYPE("FR32",                TYPE_XMM32)
  TYPE("f32mem",              TYPE_M32FP)
  TYPE("ssmem",               TYPE_M32FP)
  TYPE("RST",                 TYPE_ST)
  TYPE("i128mem",             TYPE_M128)
  TYPE("i256mem",             TYPE_M256)
  TYPE("i64i32imm_pcrel",     TYPE_REL64)
  TYPE("i16imm_pcrel",        TYPE_REL16)
  TYPE("i32imm_pcrel",        TYPE_REL32)
  TYPE("SSECC",               TYPE_IMM3)
  TYPE("AVXCC",               TYPE_IMM5)
  TYPE("brtarget",            TYPE_RELv)
  TYPE("uncondbrtarget",      TYPE_RELv)
  TYPE("brtarget8",           TYPE_REL8)
  TYPE("f80mem",              TYPE_M80FP)
  TYPE("lea32mem",            TYPE_LEA)
  TYPE("lea64_32mem",         TYPE_LEA)
  TYPE("lea64mem",            TYPE_LEA)
  TYPE("VR64",                TYPE_MM64)
  TYPE("i64imm",              TYPE_IMMv)
  TYPE("opaque32mem",         TYPE_M1616)
  TYPE("opaque48mem",         TYPE_M1632)
  TYPE("opaque80mem",         TYPE_M1664)
  TYPE("opaque512mem",        TYPE_M512)
  TYPE("SEGMENT_REG",         TYPE_SEGMENTREG)
  TYPE("DEBUG_REG",           TYPE_DEBUGREG)
  TYPE("CONTROL_REG",         TYPE_CONTROLREG)
  TYPE("offset8",             TYPE_MOFFS8)
  TYPE("offset16",            TYPE_MOFFS16)
  TYPE("offset32",            TYPE_MOFFS32)
  TYPE("offset64",            TYPE_MOFFS64)
  TYPE("VR256",               TYPE_XMM256)
  TYPE("GR16_NOAX",           TYPE_Rv)
  TYPE("GR32_NOAX",           TYPE_Rv)
  TYPE("GR64_NOAX",           TYPE_R64)
  TYPE("vx32mem",             TYPE_M32)
  TYPE("vy32mem",             TYPE_M32)
  TYPE("vx64mem",             TYPE_M64)
  TYPE("vy64mem",             TYPE_M64)
  errs() << "Unhandled type string " << s << "\n";
  llvm_unreachable("Unhandled type string");
}
#undef TYPE

#define ENCODING(str, encoding) if (s == str) return encoding;
OperandEncoding RecognizableInstr::immediateEncodingFromString
  (const std::string &s,
   bool hasOpSizePrefix) {
  if(!hasOpSizePrefix) {
    // For instructions without an OpSize prefix, a declared 16-bit register or
    // immediate encoding is special.
    ENCODING("i16imm",        ENCODING_IW)
  }
  ENCODING("i32i8imm",        ENCODING_IB)
  ENCODING("u32u8imm",        ENCODING_IB)
  ENCODING("SSECC",           ENCODING_IB)
  ENCODING("AVXCC",           ENCODING_IB)
  ENCODING("i16imm",          ENCODING_Iv)
  ENCODING("i16i8imm",        ENCODING_IB)
  ENCODING("i32imm",          ENCODING_Iv)
  ENCODING("i64i32imm",       ENCODING_ID)
  ENCODING("i64i8imm",        ENCODING_IB)
  ENCODING("i8imm",           ENCODING_IB)
  // This is not a typo.  Instructions like BLENDVPD put
  // register IDs in 8-bit immediates nowadays.
  ENCODING("VR256",           ENCODING_IB)
  ENCODING("VR128",           ENCODING_IB)
  ENCODING("FR32",            ENCODING_IB)
  ENCODING("FR64",            ENCODING_IB)
  errs() << "Unhandled immediate encoding " << s << "\n";
  llvm_unreachable("Unhandled immediate encoding");
}

OperandEncoding RecognizableInstr::rmRegisterEncodingFromString
  (const std::string &s,
   bool hasOpSizePrefix) {
  ENCODING("GR16",            ENCODING_RM)
  ENCODING("GR32",            ENCODING_RM)
  ENCODING("GR64",            ENCODING_RM)
  ENCODING("GR8",             ENCODING_RM)
  ENCODING("VR128",           ENCODING_RM)
  ENCODING("FR64",            ENCODING_RM)
  ENCODING("FR32",            ENCODING_RM)
  ENCODING("VR64",            ENCODING_RM)
  ENCODING("VR256",           ENCODING_RM)
  errs() << "Unhandled R/M register encoding " << s << "\n";
  llvm_unreachable("Unhandled R/M register encoding");
}

OperandEncoding RecognizableInstr::roRegisterEncodingFromString
  (const std::string &s,
   bool hasOpSizePrefix) {
  ENCODING("GR16",            ENCODING_REG)
  ENCODING("GR32",            ENCODING_REG)
  ENCODING("GR64",            ENCODING_REG)
  ENCODING("GR8",             ENCODING_REG)
  ENCODING("VR128",           ENCODING_REG)
  ENCODING("FR64",            ENCODING_REG)
  ENCODING("FR32",            ENCODING_REG)
  ENCODING("VR64",            ENCODING_REG)
  ENCODING("SEGMENT_REG",     ENCODING_REG)
  ENCODING("DEBUG_REG",       ENCODING_REG)
  ENCODING("CONTROL_REG",     ENCODING_REG)
  ENCODING("VR256",           ENCODING_REG)
  errs() << "Unhandled reg/opcode register encoding " << s << "\n";
  llvm_unreachable("Unhandled reg/opcode register encoding");
}

OperandEncoding RecognizableInstr::vvvvRegisterEncodingFromString
  (const std::string &s,
   bool hasOpSizePrefix) {
  ENCODING("GR32",            ENCODING_VVVV)
  ENCODING("GR64",            ENCODING_VVVV)
  ENCODING("FR32",            ENCODING_VVVV)
  ENCODING("FR64",            ENCODING_VVVV)
  ENCODING("VR128",           ENCODING_VVVV)
  ENCODING("VR256",           ENCODING_VVVV)
  errs() << "Unhandled VEX.vvvv register encoding " << s << "\n";
  llvm_unreachable("Unhandled VEX.vvvv register encoding");
}

OperandEncoding RecognizableInstr::memoryEncodingFromString
  (const std::string &s,
   bool hasOpSizePrefix) {
  ENCODING("i16mem",          ENCODING_RM)
  ENCODING("i32mem",          ENCODING_RM)
  ENCODING("i64mem",          ENCODING_RM)
  ENCODING("i8mem",           ENCODING_RM)
  ENCODING("ssmem",           ENCODING_RM)
  ENCODING("sdmem",           ENCODING_RM)
  ENCODING("f128mem",         ENCODING_RM)
  ENCODING("f256mem",         ENCODING_RM)
  ENCODING("f64mem",          ENCODING_RM)
  ENCODING("f32mem",          ENCODING_RM)
  ENCODING("i128mem",         ENCODING_RM)
  ENCODING("i256mem",         ENCODING_RM)
  ENCODING("f80mem",          ENCODING_RM)
  ENCODING("lea32mem",        ENCODING_RM)
  ENCODING("lea64_32mem",     ENCODING_RM)
  ENCODING("lea64mem",        ENCODING_RM)
  ENCODING("opaque32mem",     ENCODING_RM)
  ENCODING("opaque48mem",     ENCODING_RM)
  ENCODING("opaque80mem",     ENCODING_RM)
  ENCODING("opaque512mem",    ENCODING_RM)
  ENCODING("vx32mem",         ENCODING_RM)
  ENCODING("vy32mem",         ENCODING_RM)
  ENCODING("vx64mem",         ENCODING_RM)
  ENCODING("vy64mem",         ENCODING_RM)
  errs() << "Unhandled memory encoding " << s << "\n";
  llvm_unreachable("Unhandled memory encoding");
}

OperandEncoding RecognizableInstr::relocationEncodingFromString
  (const std::string &s,
   bool hasOpSizePrefix) {
  if(!hasOpSizePrefix) {
    // For instructions without an OpSize prefix, a declared 16-bit register or
    // immediate encoding is special.
    ENCODING("i16imm",        ENCODING_IW)
  }
  ENCODING("i16imm",          ENCODING_Iv)
  ENCODING("i16i8imm",        ENCODING_IB)
  ENCODING("i32imm",          ENCODING_Iv)
  ENCODING("i32i8imm",        ENCODING_IB)
  ENCODING("i64i32imm",       ENCODING_ID)
  ENCODING("i64i8imm",        ENCODING_IB)
  ENCODING("i8imm",           ENCODING_IB)
  ENCODING("i64i32imm_pcrel", ENCODING_ID)
  ENCODING("i16imm_pcrel",    ENCODING_IW)
  ENCODING("i32imm_pcrel",    ENCODING_ID)
  ENCODING("brtarget",        ENCODING_Iv)
  ENCODING("brtarget8",       ENCODING_IB)
  ENCODING("i64imm",          ENCODING_IO)
  ENCODING("offset8",         ENCODING_Ia)
  ENCODING("offset16",        ENCODING_Ia)
  ENCODING("offset32",        ENCODING_Ia)
  ENCODING("offset64",        ENCODING_Ia)
  errs() << "Unhandled relocation encoding " << s << "\n";
  llvm_unreachable("Unhandled relocation encoding");
}

OperandEncoding RecognizableInstr::opcodeModifierEncodingFromString
  (const std::string &s,
   bool hasOpSizePrefix) {
  ENCODING("RST",             ENCODING_I)
  ENCODING("GR32",            ENCODING_Rv)
  ENCODING("GR64",            ENCODING_RO)
  ENCODING("GR16",            ENCODING_Rv)
  ENCODING("GR8",             ENCODING_RB)
  ENCODING("GR16_NOAX",       ENCODING_Rv)
  ENCODING("GR32_NOAX",       ENCODING_Rv)
  ENCODING("GR64_NOAX",       ENCODING_RO)
  errs() << "Unhandled opcode modifier encoding " << s << "\n";
  llvm_unreachable("Unhandled opcode modifier encoding");
}
#undef ENCODING
