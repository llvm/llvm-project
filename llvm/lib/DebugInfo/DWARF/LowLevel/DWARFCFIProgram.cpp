//===- DWARFCFIProgram.cpp - Parsing the cfi-portions of .debug_frame -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/LowLevel/DWARFCFIProgram.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cinttypes>
#include <cstdint>

using namespace llvm;
using namespace dwarf;

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
