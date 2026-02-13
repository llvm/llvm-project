//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MIPSSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_MIPS_MIPSSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "MipsGenSDNodeInfo.inc"

namespace llvm {
namespace MipsISD {

enum NodeType : unsigned {
  // Floating point Abs
  FAbs = GENERATED_OPCODE_END,

  DynAlloc,

  // Double select nodes for machines without conditional-move.
  DOUBLE_SELECT_I,
  DOUBLE_SELECT_I64,
};

} // namespace MipsISD

class MipsSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  MipsSelectionDAGInfo();

  ~MipsSelectionDAGInfo() override;

  const char *getTargetNodeName(unsigned Opcode) const override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_MIPS_MIPSSELECTIONDAGINFO_H
