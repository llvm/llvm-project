//===-- DWARFExpression.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/LowLevel/DWARFExpression.h"
#include <cassert>
#include <cstdint>
#include <vector>

using namespace llvm;
using namespace dwarf;

namespace llvm {

typedef DWARFExpression::Operation Op;
typedef Op::Description Desc;

static std::vector<Desc> getOpDescriptions() {
  std::vector<Desc> Descriptions;
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
  Descriptions[DW_OP_bra] = Desc(Op::Dwarf2, Op::SignedSize2);
  Descriptions[DW_OP_eq] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_ge] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_gt] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_le] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_lt] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_ne] = Desc(Op::Dwarf2);
  Descriptions[DW_OP_skip] = Desc(Op::Dwarf2, Op::SignedSize2);
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
      Desc(Op::Dwarf4, Op::SizeLEB, Op::SizeBlock);
  Descriptions[DW_OP_stack_value] = Desc(Op::Dwarf4);
  Descriptions[DW_OP_implicit_pointer] =
      Desc(Op::Dwarf5, Op::SizeRefAddr, Op::SignedSizeLEB);
  Descriptions[DW_OP_addrx] = Desc(Op::Dwarf5, Op::SizeLEB);
  Descriptions[DW_OP_constx] = Desc(Op::Dwarf5, Op::SizeLEB);
  Descriptions[DW_OP_entry_value] = Desc(Op::Dwarf5, Op::SizeLEB);
  Descriptions[DW_OP_convert] = Desc(Op::Dwarf5, Op::BaseTypeRef);
  Descriptions[DW_OP_regval_type] =
      Desc(Op::Dwarf5, Op::SizeLEB, Op::BaseTypeRef);
  Descriptions[DW_OP_WASM_location] =
      Desc(Op::Dwarf4, Op::SizeLEB, Op::WasmLocationArg);
  Descriptions[DW_OP_GNU_push_tls_address] = Desc(Op::Dwarf3);
  Descriptions[DW_OP_GNU_addr_index] = Desc(Op::Dwarf4, Op::SizeLEB);
  Descriptions[DW_OP_GNU_const_index] = Desc(Op::Dwarf4, Op::SizeLEB);
  Descriptions[DW_OP_GNU_entry_value] = Desc(Op::Dwarf4, Op::SizeLEB);
  Descriptions[DW_OP_GNU_implicit_pointer] =
      Desc(Op::Dwarf4, Op::SizeRefAddr, Op::SignedSizeLEB);
  // This Description acts as a marker that getSubOpDesc must be called
  // to fetch the final Description for the operation. Each such final
  // Description must share the same first SizeSubOpLEB operand.
  Descriptions[DW_OP_LLVM_user] = Desc(Op::Dwarf5, Op::SizeSubOpLEB);
  return Descriptions;
}

static Desc getDescImpl(ArrayRef<Desc> Descriptions, unsigned Opcode) {
  // Handle possible corrupted or unsupported operation.
  if (Opcode >= Descriptions.size())
    return {};
  return Descriptions[Opcode];
}

static Desc getOpDesc(unsigned Opcode) {
  static std::vector<Desc> Descriptions = getOpDescriptions();
  return getDescImpl(Descriptions, Opcode);
}

static std::vector<Desc> getSubOpDescriptions() {
  static constexpr unsigned LlvmUserDescriptionsSize = 1
#define HANDLE_DW_OP_LLVM_USEROP(ID, NAME) +1
#include "llvm/BinaryFormat/Dwarf.def"
      ;
  std::vector<Desc> Descriptions;
  Descriptions.resize(LlvmUserDescriptionsSize);
  Descriptions[DW_OP_LLVM_nop] = Desc(Op::Dwarf5, Op::SizeSubOpLEB);
  Descriptions[DW_OP_LLVM_form_aspace_address] =
      Desc(Op::Dwarf5, Op::SizeSubOpLEB);
  Descriptions[DW_OP_LLVM_push_lane] = Desc(Op::Dwarf5, Op::SizeSubOpLEB);
  Descriptions[DW_OP_LLVM_offset] = Desc(Op::Dwarf5, Op::SizeSubOpLEB);
  Descriptions[DW_OP_LLVM_offset_uconst] =
      Desc(Op::Dwarf5, Op::SizeSubOpLEB, Op::SizeLEB);
  Descriptions[DW_OP_LLVM_bit_offset] = Desc(Op::Dwarf5, Op::SizeSubOpLEB);
  Descriptions[DW_OP_LLVM_call_frame_entry_reg] =
      Desc(Op::Dwarf5, Op::SizeSubOpLEB, Op::SizeLEB);
  Descriptions[DW_OP_LLVM_undefined] = Desc(Op::Dwarf5, Op::SizeSubOpLEB);
  Descriptions[DW_OP_LLVM_aspace_bregx] =
      Desc(Op::Dwarf5, Op::SizeSubOpLEB, Op::SizeLEB, Op::SizeLEB);
  Descriptions[DW_OP_LLVM_piece_end] = Desc(Op::Dwarf5, Op::SizeSubOpLEB);
  Descriptions[DW_OP_LLVM_extend] =
      Desc(Op::Dwarf5, Op::SizeSubOpLEB, Op::SizeLEB, Op::SizeLEB);
  Descriptions[DW_OP_LLVM_select_bit_piece] =
      Desc(Op::Dwarf5, Op::SizeSubOpLEB, Op::SizeLEB, Op::SizeLEB);
  return Descriptions;
}

static Desc getSubOpDesc(unsigned Opcode, unsigned SubOpcode) {
  assert(Opcode == DW_OP_LLVM_user);
  static std::vector<Desc> Descriptions = getSubOpDescriptions();
  return getDescImpl(Descriptions, SubOpcode);
}

bool DWARFExpression::Operation::extract(DataExtractor Data,
                                         uint8_t AddressSize, uint64_t Offset,
                                         std::optional<DwarfFormat> Format) {
  EndOffset = Offset;
  Opcode = Data.getU8(&Offset);

  Desc = getOpDesc(Opcode);
  if (Desc.Version == Operation::DwarfNA)
    return false;

  Operands.resize(Desc.Op.size());
  OperandEndOffsets.resize(Desc.Op.size());
  for (unsigned Operand = 0; Operand < Desc.Op.size(); ++Operand) {
    unsigned Size = Desc.Op[Operand];
    unsigned Signed = Size & Operation::SignBit;

    switch (Size & ~Operation::SignBit) {
    case Operation::SizeSubOpLEB:
      assert(Operand == 0 && "SubOp operand must be the first operand");
      Operands[Operand] = Data.getULEB128(&Offset);
      Desc = getSubOpDesc(Opcode, Operands[Operand]);
      if (Desc.Version == Operation::DwarfNA)
        return false;
      assert(Desc.Op[Operand] == Operation::SizeSubOpLEB &&
             "SizeSubOpLEB Description must begin with SizeSubOpLEB operand");
      Operands.resize(Desc.Op.size());
      OperandEndOffsets.resize(Desc.Op.size());
      break;
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
      Operands[Operand] = Data.getUnsigned(&Offset, AddressSize);
      break;
    case Operation::SizeRefAddr:
      if (!Format)
        return false;
      Operands[Operand] =
          Data.getUnsigned(&Offset, dwarf::getDwarfOffsetByteSize(*Format));
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
    case Operation::WasmLocationArg:
      assert(Operand == 1);
      switch (Operands[0]) {
      case 0:
      case 1:
      case 2:
      case 4:
        Operands[Operand] = Data.getULEB128(&Offset);
        break;
      case 3: // global as uint32
        Operands[Operand] = Data.getU32(&Offset);
        break;
      default:
        return false; // Unknown Wasm location
      }
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

std::optional<unsigned> DWARFExpression::Operation::getSubCode() const {
  if (!Desc.Op.size() || Desc.Op[0] != Operation::SizeSubOpLEB)
    return std::nullopt;
  return Operands[0];
}

bool DWARFExpression::operator==(const DWARFExpression &RHS) const {
  if (AddressSize != RHS.AddressSize || Format != RHS.Format)
    return false;
  return Data.getData() == RHS.Data.getData();
}

} // namespace llvm
