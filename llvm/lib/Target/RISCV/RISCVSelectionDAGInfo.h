//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_RISCV_RISCVSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "RISCVGenSDNodeInfo.inc"

namespace llvm {
namespace RISCVISD {

enum NodeType : unsigned {
  /// Turn a pair of `i<xlen>`s into an even-odd register pair (`untyped`).
  /// - Output: `untyped` even-odd register pair
  /// - Input 0: `i<xlen>` low-order bits, for even register.
  /// - Input 1: `i<xlen>` high-order bits, for odd register.
  BuildGPRPair = GENERATED_OPCODE_END,

  /// Turn an even-odd register pair (`untyped`) into a pair of `i<xlen>`s.
  /// - Output 0: `i<xlen>` low-order bits, from even register.
  /// - Output 1: `i<xlen>` high-order bits, from odd register.
  /// - Input: `untyped` even-odd register pair
  SplitGPRPair,

  // Splats an 64-bit value that has been split into two i32 parts. This is
  // expanded late to two scalar stores and a stride 0 vector load.
  // The first operand is passthru operand.
  SPLAT_VECTOR_SPLIT_I64_VL,

  // RISC-V vector tuple type version of INSERT_SUBVECTOR/EXTRACT_SUBVECTOR.
  TUPLE_INSERT,
  TUPLE_EXTRACT,
};

enum : unsigned {
  HasPassthruOpMask = 1 << 0,
  HasMaskOpMask = 1 << 1,
};

} // namespace RISCVISD

class RISCVSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  RISCVSelectionDAGInfo();

  ~RISCVSelectionDAGInfo() override;

  const char *getTargetNodeName(unsigned Opcode) const override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;

  bool hasPassthruOp(unsigned Opcode) const {
    return GenNodeInfo.getDesc(Opcode).TSFlags & RISCVISD::HasPassthruOpMask;
  }

  bool hasMaskOp(unsigned Opcode) const {
    return GenNodeInfo.getDesc(Opcode).TSFlags & RISCVISD::HasMaskOpMask;
  }

  unsigned getMAccOpcode(unsigned MulOpcode) const {
    switch (static_cast<RISCVISD::GenNodeType>(MulOpcode)) {
    default:
      llvm_unreachable("Unexpected opcode");
    case RISCVISD::VWMUL_VL:
      return RISCVISD::VWMACC_VL;
    case RISCVISD::VWMULU_VL:
      return RISCVISD::VWMACCU_VL;
    case RISCVISD::VWMULSU_VL:
      return RISCVISD::VWMACCSU_VL;
    }
  }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVSELECTIONDAGINFO_H
