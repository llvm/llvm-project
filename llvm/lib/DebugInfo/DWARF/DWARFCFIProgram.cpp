//===- DWARFDebugFrame.h - Parsing of .debug_frame ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFCFIProgram.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <optional>

using namespace llvm;
using namespace dwarf;

static void printRegister(raw_ostream &OS, DIDumpOptions DumpOpts,
                          unsigned RegNum) {
  if (DumpOpts.GetNameForDWARFReg) {
    auto RegName = DumpOpts.GetNameForDWARFReg(RegNum, DumpOpts.IsEH);
    if (!RegName.empty()) {
      OS << RegName;
      return;
    }
  }
  OS << "reg" << RegNum;
}


// See DWARF standard v3, section 7.23
const uint8_t DWARF_CFI_PRIMARY_OPCODE_MASK = 0xc0;
const uint8_t DWARF_CFI_PRIMARY_OPERAND_MASK = 0x3f;

Error CFIProgram::parse(DWARFDataExtractor Data, uint64_t *Offset,
                        uint64_t EndOffset) {
  DataExtractor::Cursor C(*Offset);
  while (C && C.tell() < EndOffset) {
    uint8_t Opcode = Data.getRelocatedValue(C, 1);
    if (!C)
      break;

    // Some instructions have a primary opcode encoded in the top bits.
    if (uint8_t Primary = Opcode & DWARF_CFI_PRIMARY_OPCODE_MASK) {
      // If it's a primary opcode, the first operand is encoded in the bottom
      // bits of the opcode itself.
      uint64_t Op1 = Opcode & DWARF_CFI_PRIMARY_OPERAND_MASK;
      switch (Primary) {
      case DW_CFA_advance_loc:
      case DW_CFA_restore:
        addInstruction(Primary, Op1);
        break;
      case DW_CFA_offset:
        addInstruction(Primary, Op1, Data.getULEB128(C));
        break;
      default:
        llvm_unreachable("invalid primary CFI opcode");
      }
      continue;
    }

    // Extended opcode - its value is Opcode itself.
    switch (Opcode) {
    default:
      return createStringError(errc::illegal_byte_sequence,
                               "invalid extended CFI opcode 0x%" PRIx8, Opcode);
    case DW_CFA_nop:
    case DW_CFA_remember_state:
    case DW_CFA_restore_state:
    case DW_CFA_GNU_window_save:
    case DW_CFA_AARCH64_negate_ra_state_with_pc:
      // No operands
      addInstruction(Opcode);
      break;
    case DW_CFA_set_loc:
      // Operands: Address
      addInstruction(Opcode, Data.getRelocatedAddress(C));
      break;
    case DW_CFA_advance_loc1:
      // Operands: 1-byte delta
      addInstruction(Opcode, Data.getRelocatedValue(C, 1));
      break;
    case DW_CFA_advance_loc2:
      // Operands: 2-byte delta
      addInstruction(Opcode, Data.getRelocatedValue(C, 2));
      break;
    case DW_CFA_advance_loc4:
      // Operands: 4-byte delta
      addInstruction(Opcode, Data.getRelocatedValue(C, 4));
      break;
    case DW_CFA_restore_extended:
    case DW_CFA_undefined:
    case DW_CFA_same_value:
    case DW_CFA_def_cfa_register:
    case DW_CFA_def_cfa_offset:
    case DW_CFA_GNU_args_size:
      // Operands: ULEB128
      addInstruction(Opcode, Data.getULEB128(C));
      break;
    case DW_CFA_def_cfa_offset_sf:
      // Operands: SLEB128
      addInstruction(Opcode, Data.getSLEB128(C));
      break;
    case DW_CFA_LLVM_def_aspace_cfa:
    case DW_CFA_LLVM_def_aspace_cfa_sf: {
      auto RegNum = Data.getULEB128(C);
      auto CfaOffset = Opcode == DW_CFA_LLVM_def_aspace_cfa
                           ? Data.getULEB128(C)
                           : Data.getSLEB128(C);
      auto AddressSpace = Data.getULEB128(C);
      addInstruction(Opcode, RegNum, CfaOffset, AddressSpace);
      break;
    }
    case DW_CFA_offset_extended:
    case DW_CFA_register:
    case DW_CFA_def_cfa:
    case DW_CFA_val_offset: {
      // Operands: ULEB128, ULEB128
      // Note: We can not embed getULEB128 directly into function
      // argument list. getULEB128 changes Offset and order of evaluation
      // for arguments is unspecified.
      uint64_t op1 = Data.getULEB128(C);
      uint64_t op2 = Data.getULEB128(C);
      addInstruction(Opcode, op1, op2);
      break;
    }
    case DW_CFA_offset_extended_sf:
    case DW_CFA_def_cfa_sf:
    case DW_CFA_val_offset_sf: {
      // Operands: ULEB128, SLEB128
      // Note: see comment for the previous case
      uint64_t op1 = Data.getULEB128(C);
      uint64_t op2 = (uint64_t)Data.getSLEB128(C);
      addInstruction(Opcode, op1, op2);
      break;
    }
    case DW_CFA_def_cfa_expression: {
      uint64_t ExprLength = Data.getULEB128(C);
      addInstruction(Opcode, 0);
      StringRef Expression = Data.getBytes(C, ExprLength);

      DataExtractor Extractor(Expression, Data.isLittleEndian(),
                              Data.getAddressSize());
      // Note. We do not pass the DWARF format to DWARFExpression, because
      // DW_OP_call_ref, the only operation which depends on the format, is
      // prohibited in call frame instructions, see sec. 6.4.2 in DWARFv5.
      Instructions.back().Expression =
          DWARFExpression(Extractor, Data.getAddressSize());
      break;
    }
    case DW_CFA_expression:
    case DW_CFA_val_expression: {
      uint64_t RegNum = Data.getULEB128(C);
      addInstruction(Opcode, RegNum, 0);

      uint64_t BlockLength = Data.getULEB128(C);
      StringRef Expression = Data.getBytes(C, BlockLength);
      DataExtractor Extractor(Expression, Data.isLittleEndian(),
                              Data.getAddressSize());
      // Note. We do not pass the DWARF format to DWARFExpression, because
      // DW_OP_call_ref, the only operation which depends on the format, is
      // prohibited in call frame instructions, see sec. 6.4.2 in DWARFv5.
      Instructions.back().Expression =
          DWARFExpression(Extractor, Data.getAddressSize());
      break;
    }
    }
  }

  *Offset = C.tell();
  return C.takeError();
}

StringRef CFIProgram::callFrameString(unsigned Opcode) const {
  return dwarf::CallFrameString(Opcode, Arch);
}

const char *CFIProgram::operandTypeString(CFIProgram::OperandType OT) {
#define ENUM_TO_CSTR(e)                                                        \
  case e:                                                                      \
    return #e;
  switch (OT) {
    ENUM_TO_CSTR(OT_Unset);
    ENUM_TO_CSTR(OT_None);
    ENUM_TO_CSTR(OT_Address);
    ENUM_TO_CSTR(OT_Offset);
    ENUM_TO_CSTR(OT_FactoredCodeOffset);
    ENUM_TO_CSTR(OT_SignedFactDataOffset);
    ENUM_TO_CSTR(OT_UnsignedFactDataOffset);
    ENUM_TO_CSTR(OT_Register);
    ENUM_TO_CSTR(OT_AddressSpace);
    ENUM_TO_CSTR(OT_Expression);
  }
  return "<unknown CFIProgram::OperandType>";
}

llvm::Expected<uint64_t>
CFIProgram::Instruction::getOperandAsUnsigned(const CFIProgram &CFIP,
                                              uint32_t OperandIdx) const {
  if (OperandIdx >= MaxOperands)
    return createStringError(errc::invalid_argument,
                             "operand index %" PRIu32 " is not valid",
                             OperandIdx);
  OperandType Type = CFIP.getOperandTypes()[Opcode][OperandIdx];
  uint64_t Operand = Ops[OperandIdx];
  switch (Type) {
  case OT_Unset:
  case OT_None:
  case OT_Expression:
    return createStringError(errc::invalid_argument,
                             "op[%" PRIu32 "] has type %s which has no value",
                             OperandIdx, CFIProgram::operandTypeString(Type));

  case OT_Offset:
  case OT_SignedFactDataOffset:
  case OT_UnsignedFactDataOffset:
    return createStringError(
        errc::invalid_argument,
        "op[%" PRIu32 "] has OperandType OT_Offset which produces a signed "
        "result, call getOperandAsSigned instead",
        OperandIdx);

  case OT_Address:
  case OT_Register:
  case OT_AddressSpace:
    return Operand;

  case OT_FactoredCodeOffset: {
    const uint64_t CodeAlignmentFactor = CFIP.codeAlign();
    if (CodeAlignmentFactor == 0)
      return createStringError(
          errc::invalid_argument,
          "op[%" PRIu32 "] has type OT_FactoredCodeOffset but code alignment "
          "is zero",
          OperandIdx);
    return Operand * CodeAlignmentFactor;
  }
  }
  llvm_unreachable("invalid operand type");
}

llvm::Expected<int64_t>
CFIProgram::Instruction::getOperandAsSigned(const CFIProgram &CFIP,
                                            uint32_t OperandIdx) const {
  if (OperandIdx >= MaxOperands)
    return createStringError(errc::invalid_argument,
                             "operand index %" PRIu32 " is not valid",
                             OperandIdx);
  OperandType Type = CFIP.getOperandTypes()[Opcode][OperandIdx];
  uint64_t Operand = Ops[OperandIdx];
  switch (Type) {
  case OT_Unset:
  case OT_None:
  case OT_Expression:
    return createStringError(errc::invalid_argument,
                             "op[%" PRIu32 "] has type %s which has no value",
                             OperandIdx, CFIProgram::operandTypeString(Type));

  case OT_Address:
  case OT_Register:
  case OT_AddressSpace:
    return createStringError(
        errc::invalid_argument,
        "op[%" PRIu32 "] has OperandType %s which produces an unsigned result, "
        "call getOperandAsUnsigned instead",
        OperandIdx, CFIProgram::operandTypeString(Type));

  case OT_Offset:
    return (int64_t)Operand;

  case OT_FactoredCodeOffset:
  case OT_SignedFactDataOffset: {
    const int64_t DataAlignmentFactor = CFIP.dataAlign();
    if (DataAlignmentFactor == 0)
      return createStringError(errc::invalid_argument,
                               "op[%" PRIu32 "] has type %s but data "
                               "alignment is zero",
                               OperandIdx, CFIProgram::operandTypeString(Type));
    return int64_t(Operand) * DataAlignmentFactor;
  }

  case OT_UnsignedFactDataOffset: {
    const int64_t DataAlignmentFactor = CFIP.dataAlign();
    if (DataAlignmentFactor == 0)
      return createStringError(errc::invalid_argument,
                               "op[%" PRIu32
                               "] has type OT_UnsignedFactDataOffset but data "
                               "alignment is zero",
                               OperandIdx);
    return Operand * DataAlignmentFactor;
  }
  }
  llvm_unreachable("invalid operand type");
}

ArrayRef<CFIProgram::OperandType[CFIProgram::MaxOperands]>
CFIProgram::getOperandTypes() {
  static OperandType OpTypes[DW_CFA_restore + 1][MaxOperands];
  static bool Initialized = false;
  if (Initialized) {
    return ArrayRef<OperandType[MaxOperands]>(&OpTypes[0], DW_CFA_restore + 1);
  }
  Initialized = true;

#define DECLARE_OP3(OP, OPTYPE0, OPTYPE1, OPTYPE2)                             \
  do {                                                                         \
    OpTypes[OP][0] = OPTYPE0;                                                  \
    OpTypes[OP][1] = OPTYPE1;                                                  \
    OpTypes[OP][2] = OPTYPE2;                                                  \
  } while (false)
#define DECLARE_OP2(OP, OPTYPE0, OPTYPE1)                                      \
  DECLARE_OP3(OP, OPTYPE0, OPTYPE1, OT_None)
#define DECLARE_OP1(OP, OPTYPE0) DECLARE_OP2(OP, OPTYPE0, OT_None)
#define DECLARE_OP0(OP) DECLARE_OP1(OP, OT_None)

  DECLARE_OP1(DW_CFA_set_loc, OT_Address);
  DECLARE_OP1(DW_CFA_advance_loc, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_advance_loc1, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_advance_loc2, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_advance_loc4, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_MIPS_advance_loc8, OT_FactoredCodeOffset);
  DECLARE_OP2(DW_CFA_def_cfa, OT_Register, OT_Offset);
  DECLARE_OP2(DW_CFA_def_cfa_sf, OT_Register, OT_SignedFactDataOffset);
  DECLARE_OP1(DW_CFA_def_cfa_register, OT_Register);
  DECLARE_OP3(DW_CFA_LLVM_def_aspace_cfa, OT_Register, OT_Offset,
              OT_AddressSpace);
  DECLARE_OP3(DW_CFA_LLVM_def_aspace_cfa_sf, OT_Register,
              OT_SignedFactDataOffset, OT_AddressSpace);
  DECLARE_OP1(DW_CFA_def_cfa_offset, OT_Offset);
  DECLARE_OP1(DW_CFA_def_cfa_offset_sf, OT_SignedFactDataOffset);
  DECLARE_OP1(DW_CFA_def_cfa_expression, OT_Expression);
  DECLARE_OP1(DW_CFA_undefined, OT_Register);
  DECLARE_OP1(DW_CFA_same_value, OT_Register);
  DECLARE_OP2(DW_CFA_offset, OT_Register, OT_UnsignedFactDataOffset);
  DECLARE_OP2(DW_CFA_offset_extended, OT_Register, OT_UnsignedFactDataOffset);
  DECLARE_OP2(DW_CFA_offset_extended_sf, OT_Register, OT_SignedFactDataOffset);
  DECLARE_OP2(DW_CFA_val_offset, OT_Register, OT_UnsignedFactDataOffset);
  DECLARE_OP2(DW_CFA_val_offset_sf, OT_Register, OT_SignedFactDataOffset);
  DECLARE_OP2(DW_CFA_register, OT_Register, OT_Register);
  DECLARE_OP2(DW_CFA_expression, OT_Register, OT_Expression);
  DECLARE_OP2(DW_CFA_val_expression, OT_Register, OT_Expression);
  DECLARE_OP1(DW_CFA_restore, OT_Register);
  DECLARE_OP1(DW_CFA_restore_extended, OT_Register);
  DECLARE_OP0(DW_CFA_remember_state);
  DECLARE_OP0(DW_CFA_restore_state);
  DECLARE_OP0(DW_CFA_GNU_window_save);
  DECLARE_OP0(DW_CFA_AARCH64_negate_ra_state_with_pc);
  DECLARE_OP1(DW_CFA_GNU_args_size, OT_Offset);
  DECLARE_OP0(DW_CFA_nop);

#undef DECLARE_OP0
#undef DECLARE_OP1
#undef DECLARE_OP2

  return ArrayRef<OperandType[MaxOperands]>(&OpTypes[0], DW_CFA_restore + 1);
}

/// Print \p Opcode's operand number \p OperandIdx which has value \p Operand.
void CFIProgram::printOperand(raw_ostream &OS, DIDumpOptions DumpOpts,
                              const Instruction &Instr, unsigned OperandIdx,
                              uint64_t Operand,
                              std::optional<uint64_t> &Address) const {
  assert(OperandIdx < MaxOperands);
  uint8_t Opcode = Instr.Opcode;
  OperandType Type = getOperandTypes()[Opcode][OperandIdx];

  switch (Type) {
  case OT_Unset: {
    OS << " Unsupported " << (OperandIdx ? "second" : "first") << " operand to";
    auto OpcodeName = callFrameString(Opcode);
    if (!OpcodeName.empty())
      OS << " " << OpcodeName;
    else
      OS << format(" Opcode %x",  Opcode);
    break;
  }
  case OT_None:
    break;
  case OT_Address:
    OS << format(" %" PRIx64, Operand);
    Address = Operand;
    break;
  case OT_Offset:
    // The offsets are all encoded in a unsigned form, but in practice
    // consumers use them signed. It's most certainly legacy due to
    // the lack of signed variants in the first Dwarf standards.
    OS << format(" %+" PRId64, int64_t(Operand));
    break;
  case OT_FactoredCodeOffset: // Always Unsigned
    if (CodeAlignmentFactor)
      OS << format(" %" PRId64, Operand * CodeAlignmentFactor);
    else
      OS << format(" %" PRId64 "*code_alignment_factor", Operand);
    if (Address && CodeAlignmentFactor) {
      *Address += Operand * CodeAlignmentFactor;
      OS << format(" to 0x%" PRIx64, *Address);
    }
    break;
  case OT_SignedFactDataOffset:
    if (DataAlignmentFactor)
      OS << format(" %" PRId64, int64_t(Operand) * DataAlignmentFactor);
    else
      OS << format(" %" PRId64 "*data_alignment_factor" , int64_t(Operand));
    break;
  case OT_UnsignedFactDataOffset:
    if (DataAlignmentFactor)
      OS << format(" %" PRId64, Operand * DataAlignmentFactor);
    else
      OS << format(" %" PRId64 "*data_alignment_factor" , Operand);
    break;
  case OT_Register:
    OS << ' ';
    printRegister(OS, DumpOpts, Operand);
    break;
  case OT_AddressSpace:
    OS << format(" in addrspace%" PRId64, Operand);
    break;
  case OT_Expression:
    assert(Instr.Expression && "missing DWARFExpression object");
    OS << " ";
    Instr.Expression->print(OS, DumpOpts, nullptr);
    break;
  }
}

void CFIProgram::dump(raw_ostream &OS, DIDumpOptions DumpOpts,
                      unsigned IndentLevel,
                      std::optional<uint64_t> Address) const {
  for (const auto &Instr : Instructions) {
    uint8_t Opcode = Instr.Opcode;
    OS.indent(2 * IndentLevel);
    OS << callFrameString(Opcode) << ":";
    for (unsigned i = 0; i < Instr.Ops.size(); ++i)
      printOperand(OS, DumpOpts, Instr, i, Instr.Ops[i], Address);
    OS << '\n';
  }
}
