//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_VE_VESELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_VE_VESELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "VEGenSDNodeInfo.inc"

namespace llvm {
namespace VEISD {

enum NodeType : unsigned {
  GLOBAL_BASE_REG = GENERATED_OPCODE_END, // Global base reg for PIC.

  // Annotation as a wrapper. LEGALAVL(VL) means that VL refers to 64bit of
  // data, whereas the raw EVL coming in from VP nodes always refers to number
  // of elements, regardless of their size.
  LEGALAVL,
};

} // namespace VEISD

class VESelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  VESelectionDAGInfo();

  ~VESelectionDAGInfo() override;

  const char *getTargetNodeName(unsigned Opcode) const override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_VE_VESELECTIONDAGINFO_H
