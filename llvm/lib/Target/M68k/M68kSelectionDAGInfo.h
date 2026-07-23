//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M68K_M68KSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_M68K_M68KSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "M68kGenSDNodeInfo.inc"

namespace llvm {

class M68kSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  M68kSelectionDAGInfo();

  ~M68kSelectionDAGInfo() override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_M68K_M68KSELECTIONDAGINFO_H
