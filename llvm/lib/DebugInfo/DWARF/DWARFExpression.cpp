//===-- DWARFExpression.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Format.h"
#include <cassert>
#include <cstdint>
#include <vector>

using namespace llvm;
using namespace dwarf;

namespace llvm {

typedef std::vector<DWARFExpression::Operation::Description> DescVector;

static DescVector getDescriptions() {
  DescVector Descriptions;
  typedef DWARFExpression::Operation Op;
  typedef Op::Description Desc;

  Descriptions.resize(0xff);
  Descriptions[DW_OP_addr] = Desc(Op::Dwarf2, Op::SizeAddr);
  Descriptions[DW_OP_deref] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_const1u] = Desc(Op::Dwarf2, Op::Size1);
  Descriptions[DW_OP_const1s] = Desc(Op::Dwarf2, Op::SignedSize1);
  Descriptions[DW_OP_const2u] = Desc(Op::Dwarf2, Op::Size2);
  Descriptions[DW_OP_const2s] = Desc(Op::Dwarf2, Op::SignedSize2);
  Descriptions[DW_OP_const4u] = Desc(Op::Dwarf2, Op::Size4);
  Descriptions[DW_OP_const4s] = Desc(Op::Dwarf2, Op::SignedSize4);
  Descriptions[DW_OP_const8u] = Desc(Op::Dwarf2, Op::Size8);
  Descriptions[DW_OP_const8s] = Desc(Op::Dwarf2, Op::SignedSize8);
  Descriptions[DW_OP_constu] = Desc(Op::Dwarf2, Op::SizeLEB);
  Descriptions[DW_OP_consts] = Desc(Op::Dwarf2, Op::SignedSizeLEB);
  Descriptions[DW_OP_dup] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_drop] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_over] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_pick] = Desc(Op::Dwarf2, Op::Size1);
  Descriptions[DW_OP_swap] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_rot] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_xderef] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_abs] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_and] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_div] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_minus] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_mod] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_mul] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_neg] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_not] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_or] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_plus] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_plus_uconst] = Desc(Op::Dwarf2, Op::SizeLEB);
  Descriptions[DW_OP_shl] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_shr] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_shra] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_xor] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_skip] = Desc(Op::Dwarf2, Op::SignedSize2);
  Descriptions[DW_OP_bra] = Desc(Op::Dwarf2, Op::SignedSize2);
  Descriptions[DW_OP_eq] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_ge] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_gt] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_le] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_lt] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_ne] = Desc(Op::Dwarf2);
  for (uint16_t LA = DW_OP_lit0; LA <= DW_OP_lit31; ++LA)
    Descriptions[LA] = Desc(Op::Dwarf2);
  for (uint16_t LA = DW_OP_reg0; LA <= DW_OP_reg31; ++LA)
    Descriptions[LA] = Desc(Op::Dwarf2);
  for (uint16_t LA = DW_OP_breg0; LA <= DW_OP_breg31; ++LA)
    Descriptions[LA] = Desc(Op::Dwarf2, Op::SignedSizeLEB);
  Descriptions[DW_OP_regx] = Desc(Op::Dwarf2, Op::SizeLEB);
  Descriptions[DW_OP_fbreg] = Desc(Op::Dwarf2, Op::SignedSizeLEB);
  Descriptions[DW_OP_bregx] = Desc(Op::Dwarf2, Op::SizeLEB, Op::SignedSizeLEB);
  Descriptions[DW_OP_piece] = Desc(Op::Dwarf2, Op::SizeLEB);
  Descriptions[DW_OP_deref_size] = Desc(Op::Dwarf2, Op::Size1);
  Descriptions[DW_OP_xderef_size] = Desc(Op::Dwarf2, Op::Size1);
  Descriptions[DW_OP_nop] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_push_object_address] = Desc(Op::Dwarf3);
  Descriptions[DW_OP_call2] = Desc(Op::Dwarf3, Op::Size2);
  Descriptions[DW_OP_call4] = Desc(Op::Dwarf3, Op::Size4);
  Descriptions[DW_OP_call_ref] = Desc(Op::Dwarf3, Op::SizeRefAddr);
  Descriptions[DW_OP_form_tls_address] = Desc(Op::Dwarf3);
  Descriptions[DW_OP_call_frame_cfa] = Desc(Op::Dwarf3);
  Descriptions[DW_OP_bit_piece] = Desc(Op::Dwarf3, Op::SizeLEB, Op::SizeLEB);
  Descriptions[DW_OP_implicit_value] =
      Desc(Op::Dwarf3, Op::SizeLEB, Op::SizeBlock);
  Descriptions[DW_OP_stack_value] = Desc(Op::Dwarf3);
  Descriptions[DW_OP_WASM_location] =
      Desc(Op::Dwarf4, Op::SizeLEB, Op::SignedSizeLEB);
  Descriptions[DW_OP_GNU_push_tls_address] = Desc(Op::Dwarf3);
  Descriptions[DW_OP_addrx] = Desc(Op::Dwarf4, Op::SizeLEB);
  Descriptions[DW_OP_GNU_addr_index] = Desc(Op::Dwarf4, Op::SizeLEB);
  Descriptions[DW_OP_GNU_const_index] = Desc(Op::Dwarf4, Op::SizeLEB);
  Descriptions[DW_OP_GNU_entry_value] = Desc(Op::Dwarf4, Op::SizeLEB);

  Descriptions[DW_OP_convert] = Desc(Op::Dwarf5, Op::BaseTypeRef);
  Descriptions[DW_OP_entry_value] = Desc(Op::Dwarf5, Op::SizeLEB);

  return Descriptions;
}

static DWARFExpression::Operation::Description getOpDesc(unsigned OpCode) {
  // FIXME: Make this constexpr once all compilers are smart enough to do it.
  static DescVector Descriptions = getDescriptions();
  // Handle possible corrupted or unsupported operation.
  if (OpCode >= Descriptions.size())
    return {};
  return Descriptions[OpCode];
}

static uint8_t getRefAddrSize(uint8_t AddrSize, uint16_t Version) {
  return (Version == 2) ? AddrSize : 4;
}

bool DWARFExpression::Operation::extract(DataExtractor Data, uint16_t Version,
                                         uint8_t AddressSize, uint64_t Offset) {
  Opcode = Data.getU8(&Offset);

  Desc = getOpDesc(Opcode);
  if (Desc.Version == Operation::DwarfNA) {
    EndOffset = Offset;
    return false;
  }

  for (unsigned Operand = 0; Operand < 2; ++Operand) {
    unsigned Size = Desc.Op[Operand];
    unsigned Signed = Size & Operation::SignBit;

    if (Size == Operation::SizeNA)
      break;

    switch (Size & ~Operation::SignBit) {
    case Operation::Size1:
      Operands[Operand] = Data.getU8(&Offset);
      if (Signed)
        Operands[Operand] = (int8_t)Operands[Operand];
      break;
    case Operation::Size2:
      Operands[Operand] = Data.getU16(&Offset);
      if (Signed)
        Operands[Operand] = (int16_t)Operands[Operand];
      break;
    case Operation::Size4:
      Operands[Operand] = Data.getU32(&Offset);
      if (Signed)
        Operands[Operand] = (int32_t)Operands[Operand];
      break;
    case Operation::Size8:
      Operands[Operand] = Data.getU64(&Offset);
      break;
    case Operation::SizeAddr:
      if (AddressSize == 8) {
        Operands[Operand] = Data.getU64(&Offset);
      } else if (AddressSize == 4) {
        Operands[Operand] = Data.getU32(&Offset);
      } else {
        assert(AddressSize == 2);
        Operands[Operand] = Data.getU16(&Offset);
      }
      break;
    case Operation::SizeRefAddr:
      if (getRefAddrSize(AddressSize, Version) == 8) {
        Operands[Operand] = Data.getU64(&Offset);
      } else if (getRefAddrSize(AddressSize, Version) == 4) {
        Operands[Operand] = Data.getU32(&Offset);
      } else {
        assert(getRefAddrSize(AddressSize, Version) == 2);
        Operands[Operand] = Data.getU16(&Offset);
      }
      break;
    case Operation::SizeLEB:
      if (Signed)
        Operands[Operand] = Data.getSLEB128(&Offset);
      else
        Operands[Operand] = Data.getULEB128(&Offset);
      break;
    case Operation::BaseTypeRef:
      Operands[Operand] = Data.getULEB128(&Offset);
      break;
    case Operation::SizeBlock:
      // We need a size, so this cannot be the first operand
      if (Operand == 0)
        return false;
      // Store the offset of the block as the value.
      Operands[Operand] = Offset;
      Offset += Operands[Operand - 1];
      break;
    default:
      llvm_unreachable("Unknown DWARFExpression Op size");
    }

    OperandEndOffsets[Operand] = Offset;
  }

  EndOffset = Offset;
  return true;
}

static bool prettyPrintRegisterOp(raw_ostream &OS, uint8_t Opcode,
                                  uint64_t Operands[2],
                                  const MCRegisterInfo *MRI, bool isEH) {
  if (!MRI)
    return false;

  uint64_t DwarfRegNum;
  unsigned OpNum = 0;

  if (Opcode == DW_OP_bregx || Opcode == DW_OP_regx)
    DwarfRegNum = Operands[OpNum++];
  else if (Opcode >= DW_OP_breg0 && Opcode < DW_OP_bregx)
    DwarfRegNum = Opcode - DW_OP_breg0;
  else
    DwarfRegNum = Opcode - DW_OP_reg0;

  if (Optional<unsigned> LLVMRegNum = MRI->getLLVMRegNum(DwarfRegNum, isEH)) {
    if (const char *RegName = MRI->getName(*LLVMRegNum)) {
      if ((Opcode >= DW_OP_breg0 && Opcode <= DW_OP_breg31) ||
          Opcode == DW_OP_bregx)
        OS << format(" %s%+" PRId64, RegName, Operands[OpNum]);
      else
        OS << ' ' << RegName;
      return true;
    }
  }

  return false;
}

bool DWARFExpression::Operation::print(raw_ostream &OS,
                                       const DWARFExpression *Expr,
                                       const MCRegisterInfo *RegInfo,
                                       DWARFUnit *U,
                                       bool isEH) {
  if (Error) {
    OS << "<decoding error>";
    return false;
  }

  StringRef Name = OperationEncodingString(Opcode);
  assert(!Name.empty() && "DW_OP has no name!");
  OS << Name;

  if ((Opcode >= DW_OP_breg0 && Opcode <= DW_OP_breg31) ||
      (Opcode >= DW_OP_reg0 && Opcode <= DW_OP_reg31) ||
      Opcode == DW_OP_bregx || Opcode == DW_OP_regx)
    if (prettyPrintRegisterOp(OS, Opcode, Operands, RegInfo, isEH))
      return true;

  for (unsigned Operand = 0; Operand < 2; ++Operand) {
    unsigned Size = Desc.Op[Operand];
    unsigned Signed = Size & Operation::SignBit;

    if (Size == Operation::SizeNA)
      break;

    if (Size == Operation::BaseTypeRef && U) {
      auto Die = U->getDIEForOffset(U->getOffset() + Operands[Operand]);
      if (Die && Die.getTag() == dwarf::DW_TAG_base_type) {
        OS << format(" (0x%08" PRIx64 ")", U->getOffset() + Operands[Operand]);
        if (auto Name = Die.find(dwarf::DW_AT_name))
          OS << " \"" << Name->getAsCString() << "\"";
      } else {
        OS << format(" <invalid base_type ref: 0x%" PRIx64 ">",
                     Operands[Operand]);
      }
    } else if (Size == Operation::SizeBlock) {
      uint64_t Offset = Operands[Operand];
      for (unsigned i = 0; i < Operands[Operand - 1]; ++i)
        OS << format(" 0x%02x", Expr->Data.getU8(&Offset));
    } else {
      if (Signed)
        OS << format(" %+" PRId64, (int64_t)Operands[Operand]);
      else if (Opcode != DW_OP_entry_value &&
               Opcode != DW_OP_GNU_entry_value)
        OS << format(" 0x%" PRIx64, Operands[Operand]);
    }
  }
  return true;
}

void DWARFExpression::print(raw_ostream &OS, const MCRegisterInfo *RegInfo,
                            DWARFUnit *U, bool IsEH) const {
  uint32_t EntryValExprSize = 0;
  for (auto &Op : *this) {
    if (!Op.print(OS, this, RegInfo, U, IsEH)) {
      uint64_t FailOffset = Op.getEndOffset();
      while (FailOffset < Data.getData().size())
        OS << format(" %02x", Data.getU8(&FailOffset));
      return;
    }

    if (Op.getCode() == DW_OP_entry_value ||
        Op.getCode() == DW_OP_GNU_entry_value) {
      OS << "(";
      EntryValExprSize = Op.getRawOperand(0);
      continue;
    }

    if (EntryValExprSize) {
      EntryValExprSize--;
      if (EntryValExprSize == 0)
        OS << ")";
    }

    if (Op.getEndOffset() < Data.getData().size())
      OS << ", ";
  }
}

bool DWARFExpression::Operation::verify(DWARFUnit *U) {

  for (unsigned Operand = 0; Operand < 2; ++Operand) {
    unsigned Size = Desc.Op[Operand];

    if (Size == Operation::SizeNA)
      break;

    if (Size == Operation::BaseTypeRef) {
      auto Die = U->getDIEForOffset(U->getOffset() + Operands[Operand]);
      if (!Die || Die.getTag() != dwarf::DW_TAG_base_type) {
        Error = true;
        return false;
      }
    }
  }

  return true;
}

bool DWARFExpression::verify(DWARFUnit *U) {
  for (auto &Op : *this)
    if (!Op.verify(U))
      return false;

  return true;
}

} // namespace llvm
