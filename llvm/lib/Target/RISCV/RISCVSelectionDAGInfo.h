//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_RISCV_RISCVSELECTIONDAGINFO_H

#include "llvm/CodeGen/SDNodeInfo.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "RISCVGenSDNodeInfo.inc"

namespace llvm {

namespace RISCVISD {
// RISCVISD Node TSFlags
enum : llvm::SDNodeTSFlags {
  HasPassthruOpMask = 1 << 0,
  HasMaskOpMask = 1 << 1,
};
} // namespace RISCVISD

class RISCVSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  RISCVSelectionDAGInfo();

  ~RISCVSelectionDAGInfo() override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;

  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, Align Alignment,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo) const override;

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
