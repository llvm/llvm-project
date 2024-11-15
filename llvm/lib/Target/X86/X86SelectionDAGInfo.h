//===-- X86SelectionDAGInfo.h - X86 SelectionDAG Info -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86 subclass for SelectionDAGTargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86SELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_X86_X86SELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "X86GenSDNodeInfo.inc"

namespace llvm {
namespace X86ISD {

enum NodeType : unsigned {
  /// The same as ISD::CopyFromReg except that this node makes it explicit
  /// that it may lower to an x87 FPU stack pop. Optimizations should be more
  /// cautious when handling this node than a normal CopyFromReg to avoid
  /// removing a required FPU stack pop. A key requirement is optimizations
  /// should not optimize any users of a chain that contains a
  /// POP_FROM_X87_REG to use a chain from a point earlier than the
  /// POP_FROM_X87_REG (which may remove a required FPU stack pop).
  POP_FROM_X87_REG = X86ISD::GENERATED_OPCODE_END,

  /// On Darwin, this node represents the result of the popl
  /// at function entry, used for PIC code.
  GlobalBaseReg,

  // SSE42 string comparisons.
  // These nodes produce 3 results, index, mask, and flags. X86ISelDAGToDAG
  // will emit one or two instructions based on which results are used. If
  // flags and index/mask this allows us to use a single instruction since
  // we won't have to pick and opcode for flags. Instead we can rely on the
  // DAG to CSE everything and decide at isel.
  PCMPISTR,
  PCMPESTR,

  // Compare and swap.
  FIRST_MEMORY_OPCODE,
  LCMPXCHG16_SAVE_RBX_DAG = FIRST_MEMORY_OPCODE,

  // X86 specific gather and scatter
  MGATHER,
  MSCATTER,

  // Key locker nodes that produce flags.
  AESENCWIDE128KL,
  AESDECWIDE128KL,
  AESENCWIDE256KL,
  AESDECWIDE256KL,
  LAST_MEMORY_OPCODE = AESDECWIDE256KL,
};

} // namespace X86ISD

class X86SelectionDAGInfo : public SelectionDAGGenTargetInfo {
  /// Returns true if it is possible for the base register to conflict with the
  /// given set of clobbers for a memory intrinsic.
  bool isBaseRegConflictPossible(SelectionDAG &DAG,
                                 ArrayRef<MCPhysReg> ClobberSet) const;

public:
  X86SelectionDAGInfo();

  const char *getTargetNodeName(unsigned Opcode) const override;

  bool isTargetMemoryOpcode(unsigned Opcode) const override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;

  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, Align Alignment,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo) const override;

  SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, Align Alignment,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo,
                                  MachinePointerInfo SrcPtrInfo) const override;
};

} // namespace llvm

#endif
