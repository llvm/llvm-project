//===- LoongArchSelectionDAGInfo.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_LOONGARCHSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_LOONGARCH_LOONGARCHSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "LoongArchGenSDNodeInfo.inc"

namespace llvm {
namespace LoongArchISD {

enum NodeType {
  // This is skipped by TableGen because it has conflicting SDTypeProfiles.
  VSHUF4I = GENERATED_OPCODE_END,
};

} // namespace LoongArchISD

class LoongArchSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  LoongArchSelectionDAGInfo();

  ~LoongArchSelectionDAGInfo() override;

  const char *getTargetNodeName(unsigned Opcode) const override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_LOONGARCH_LOONGARCHSELECTIONDAGINFO_H
