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
#undef GET_SDNODE_ENUM

namespace llvm {

class AArch64SelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  AArch64SelectionDAGInfo();

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
} // namespace llvm

#endif
