//===-- AArch64SelectionDAGInfo.h - AArch64 SelectionDAG Info ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AArch64 subclass for SelectionDAGTargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64SELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64SELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/IR/RuntimeLibcalls.h"

#define GET_SDNODE_ENUM
#include "AArch64GenSDNodeInfo.inc"

namespace llvm {
namespace AArch64ISD {

// For predicated nodes where the result is a vector, the operation is
// controlled by a governing predicate and the inactive lanes are explicitly
// defined with a value, please stick the following naming convention:
//
//    _MERGE_OP<n>        The result value is a vector with inactive lanes equal
//                        to source operand OP<n>.
//
//    _MERGE_ZERO         The result value is a vector with inactive lanes
//                        actively zeroed.
//
//    _MERGE_PASSTHRU     The result value is a vector with inactive lanes equal
//                        to the last source operand which only purpose is being
//                        a passthru value.
//
// For other cases where no explicit action is needed to set the inactive lanes,
// or when the result is not a vector and it is needed or helpful to
// distinguish a node from similar unpredicated nodes, use:
//
//    _PRED
//
enum NodeType : unsigned {
  INDEX_VECTOR = GENERATED_OPCODE_END,

  // Structured loads.
  SVE_LD2_MERGE_ZERO,
  SVE_LD3_MERGE_ZERO,
  SVE_LD4_MERGE_ZERO,

  // Unsigned gather loads.
  GLD1Q_INDEX_MERGE_ZERO,

  // Non-temporal gather loads
  GLDNT1_INDEX_MERGE_ZERO,

  // Scatter store
  SST1Q_INDEX_PRED,

  // Non-temporal scatter store
  SSTNT1_INDEX_PRED,

  // 128-bit system register accesses
  // lo64, hi64, chain = MRRS(chain, sysregname)
  MRRS,
  // chain = MSRR(chain, sysregname, lo64, hi64)
  MSRR,

  // NEON Load/Store with post-increment base updates
  FIRST_MEMORY_OPCODE,
  LD2post = FIRST_MEMORY_OPCODE,
  LD3post,
  LD4post,
  ST2post,
  ST3post,
  ST4post,
  LD1x2post,
  LD1x3post,
  LD1x4post,
  ST1x2post,
  ST1x3post,
  ST1x4post,
  LD1DUPpost,
  LD2DUPpost,
  LD3DUPpost,
  LD4DUPpost,
  LD1LANEpost,
  LD2LANEpost,
  LD3LANEpost,
  LD4LANEpost,
  ST2LANEpost,
  ST3LANEpost,
  ST4LANEpost,
  LAST_MEMORY_OPCODE = ST4LANEpost,
};

} // namespace AArch64ISD

class AArch64SelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  AArch64SelectionDAGInfo();

  const char *getTargetNodeName(unsigned Opcode) const override;

  bool isTargetMemoryOpcode(unsigned Opcode) const override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;

  SDValue EmitMOPS(unsigned Opcode, SelectionDAG &DAG, const SDLoc &DL,
                   SDValue Chain, SDValue Dst, SDValue SrcOrValue, SDValue Size,
                   Align Alignment, bool isVolatile,
                   MachinePointerInfo DstPtrInfo,
                   MachinePointerInfo SrcPtrInfo) const;

  SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, Align Alignment,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo,
                                  MachinePointerInfo SrcPtrInfo) const override;
  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, Align Alignment,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo) const override;
  SDValue
  EmitTargetCodeForMemmove(SelectionDAG &DAG, const SDLoc &dl, SDValue Chain,
                           SDValue Dst, SDValue Src, SDValue Size,
                           Align Alignment, bool isVolatile,
                           MachinePointerInfo DstPtrInfo,
                           MachinePointerInfo SrcPtrInfo) const override;

  SDValue EmitTargetCodeForSetTag(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Op1, SDValue Op2,
                                  MachinePointerInfo DstPtrInfo,
                                  bool ZeroData) const override;

  SDValue EmitStreamingCompatibleMemLibCall(SelectionDAG &DAG, const SDLoc &DL,
                                            SDValue Chain, SDValue Dst,
                                            SDValue Src, SDValue Size,
                                            RTLIB::Libcall LC) const;
};
}

#endif
