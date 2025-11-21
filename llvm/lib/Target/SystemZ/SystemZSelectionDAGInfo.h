//===-- SystemZSelectionDAGInfo.h - SystemZ SelectionDAG Info ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SystemZ subclass for SelectionDAGTargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "SystemZGenSDNodeInfo.inc"

namespace llvm {
namespace SystemZISD {

enum NodeType : unsigned {
  // Set the condition code from a boolean value in operand 0.
  // Operand 1 is a mask of all condition-code values that may result of this
  // operation, operand 2 is a mask of condition-code values that may result
  // if the boolean is true.
  // Note that this operation is always optimized away, we will never
  // generate any code for it.
  GET_CCMASK = GENERATED_OPCODE_END,
};

// Return true if OPCODE is some kind of PC-relative address.
inline bool isPCREL(unsigned Opcode) {
  return Opcode == PCREL_WRAPPER || Opcode == PCREL_OFFSET;
}

} // namespace SystemZISD

class SystemZSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  SystemZSelectionDAGInfo();

  const char *getTargetNodeName(unsigned Opcode) const override;

  SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, const SDLoc &DL,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, Align Alignment,
                                  bool IsVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo,
                                  MachinePointerInfo SrcPtrInfo) const override;

  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, const SDLoc &DL,
                                  SDValue Chain, SDValue Dst, SDValue Byte,
                                  SDValue Size, Align Alignment,
                                  bool IsVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo) const override;

  std::pair<SDValue, SDValue>
  EmitTargetCodeForMemcmp(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain,
                          SDValue Src1, SDValue Src2, SDValue Size,
                          const CallInst *CI) const override;

  std::pair<SDValue, SDValue>
  EmitTargetCodeForMemchr(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain,
                          SDValue Src, SDValue Char, SDValue Length,
                          MachinePointerInfo SrcPtrInfo) const override;

  std::pair<SDValue, SDValue> EmitTargetCodeForStrcpy(
      SelectionDAG &DAG, const SDLoc &DL, SDValue Chain, SDValue Dest,
      SDValue Src, MachinePointerInfo DestPtrInfo,
      MachinePointerInfo SrcPtrInfo, bool isStpcpy) const override;

  std::pair<SDValue, SDValue>
  EmitTargetCodeForStrcmp(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain,
                          SDValue Src1, SDValue Src2,
                          MachinePointerInfo Op1PtrInfo,
                          MachinePointerInfo Op2PtrInfo) const override;

  std::pair<SDValue, SDValue>
  EmitTargetCodeForStrlen(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain,
                          SDValue Src, const CallInst *CI) const override;

  std::pair<SDValue, SDValue>
  EmitTargetCodeForStrnlen(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain,
                           SDValue Src, SDValue MaxLength,
                           MachinePointerInfo SrcPtrInfo) const override;
};

} // end namespace llvm

#endif
