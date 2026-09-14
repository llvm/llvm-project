//===-- SPIRVNonSemanticDebugInfo200.cpp - NSDI.200 specifics -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVNonSemanticDebugInfo200.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace llvm;

namespace llvm {
namespace SPIRV {
namespace NSDI200 {

std::optional<DebugOp> mapDwarfOpToDebugOp200(uint64_t DwarfOp) {
  if (DwarfOp >= dwarf::DW_OP_lit0 && DwarfOp <= dwarf::DW_OP_lit31)
    return static_cast<DebugOp>(static_cast<uint32_t>(DebugOp::Lit0) +
                                (DwarfOp - dwarf::DW_OP_lit0));
  if (DwarfOp >= dwarf::DW_OP_reg0 && DwarfOp <= dwarf::DW_OP_reg31)
    return static_cast<DebugOp>(static_cast<uint32_t>(DebugOp::Reg0) +
                                (DwarfOp - dwarf::DW_OP_reg0));
  if (DwarfOp >= dwarf::DW_OP_breg0 && DwarfOp <= dwarf::DW_OP_breg31)
    return static_cast<DebugOp>(static_cast<uint32_t>(DebugOp::Breg0) +
                                (DwarfOp - dwarf::DW_OP_breg0));

  switch (DwarfOp) {
  case dwarf::DW_OP_LLVM_convert:
    return DebugOp::Convert;
  case dwarf::DW_OP_addr:
    return DebugOp::Addr;
  case dwarf::DW_OP_const1u:
    return DebugOp::Const1u;
  case dwarf::DW_OP_const1s:
    return DebugOp::Const1s;
  case dwarf::DW_OP_const2u:
    return DebugOp::Const2u;
  case dwarf::DW_OP_const2s:
    return DebugOp::Const2s;
  case dwarf::DW_OP_const4u:
    return DebugOp::Const4u;
  case dwarf::DW_OP_const4s:
    return DebugOp::Const4s;
  case dwarf::DW_OP_const8u:
    return DebugOp::Const8u;
  case dwarf::DW_OP_const8s:
    return DebugOp::Const8s;
  case dwarf::DW_OP_consts:
    return DebugOp::Consts;
  case dwarf::DW_OP_dup:
    return DebugOp::Dup;
  case dwarf::DW_OP_drop:
    return DebugOp::Drop;
  case dwarf::DW_OP_over:
    return DebugOp::Over;
  case dwarf::DW_OP_pick:
    return DebugOp::Pick;
  case dwarf::DW_OP_rot:
    return DebugOp::Rot;
  case dwarf::DW_OP_abs:
    return DebugOp::Abs;
  case dwarf::DW_OP_and:
    return DebugOp::And;
  case dwarf::DW_OP_div:
    return DebugOp::Div;
  case dwarf::DW_OP_mod:
    return DebugOp::Mod;
  case dwarf::DW_OP_mul:
    return DebugOp::Mul;
  case dwarf::DW_OP_neg:
    return DebugOp::Neg;
  case dwarf::DW_OP_not:
    return DebugOp::Not;
  case dwarf::DW_OP_or:
    return DebugOp::Or;
  case dwarf::DW_OP_shl:
    return DebugOp::Shl;
  case dwarf::DW_OP_shr:
    return DebugOp::Shr;
  case dwarf::DW_OP_shra:
    return DebugOp::Shra;
  case dwarf::DW_OP_xor:
    return DebugOp::Xor;
  case dwarf::DW_OP_bra:
    return DebugOp::Bra;
  case dwarf::DW_OP_eq:
    return DebugOp::Eq;
  case dwarf::DW_OP_ge:
    return DebugOp::Ge;
  case dwarf::DW_OP_gt:
    return DebugOp::Gt;
  case dwarf::DW_OP_le:
    return DebugOp::Le;
  case dwarf::DW_OP_lt:
    return DebugOp::Lt;
  case dwarf::DW_OP_ne:
    return DebugOp::Ne;
  case dwarf::DW_OP_skip:
    return DebugOp::Skip;
  case dwarf::DW_OP_regx:
    return DebugOp::Regx;
  case dwarf::DW_OP_bregx:
    return DebugOp::Bregx;
  case dwarf::DW_OP_piece:
    return DebugOp::Piece;
  case dwarf::DW_OP_deref_size:
    return DebugOp::DerefSize;
  case dwarf::DW_OP_xderef_size:
    return DebugOp::XderefSize;
  case dwarf::DW_OP_nop:
    return DebugOp::Nop;
  case dwarf::DW_OP_push_object_address:
    return DebugOp::PushObjectAddress;
  case dwarf::DW_OP_call2:
    return DebugOp::Call2;
  case dwarf::DW_OP_call4:
    return DebugOp::Call4;
  case dwarf::DW_OP_call_ref:
    return DebugOp::CallRef;
  case dwarf::DW_OP_form_tls_address:
    return DebugOp::FormTlsAddress;
  case dwarf::DW_OP_call_frame_cfa:
    return DebugOp::CallFrameCfa;
  case dwarf::DW_OP_implicit_value:
    return DebugOp::ImplicitValue;
  case dwarf::DW_OP_implicit_pointer:
    return DebugOp::ImplicitPointer;
  case dwarf::DW_OP_addrx:
    return DebugOp::Addrx;
  case dwarf::DW_OP_constx:
    return DebugOp::Constx;
  case dwarf::DW_OP_entry_value:
    return DebugOp::EntryValue;
  case dwarf::DW_OP_const_type:
    return DebugOp::ConstTypeOp;
  case dwarf::DW_OP_regval_type:
    return DebugOp::RegvalType;
  case dwarf::DW_OP_deref_type:
    return DebugOp::DerefType;
  case dwarf::DW_OP_xderef_type:
    return DebugOp::XderefType;
  case dwarf::DW_OP_reinterpret:
    return DebugOp::Reinterpret;
  case dwarf::DW_OP_LLVM_arg:
    return DebugOp::LLVMArg;
  case dwarf::DW_OP_LLVM_implicit_pointer:
    return DebugOp::ImplicitPointerTag;
  case dwarf::DW_OP_LLVM_tag_offset:
    return DebugOp::TagOffset;
  default:
    return std::nullopt;
  }
}

} // namespace NSDI200
} // namespace SPIRV
} // namespace llvm
