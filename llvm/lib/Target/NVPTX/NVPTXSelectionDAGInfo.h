//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "NVPTXGenSDNodeInfo.inc"

namespace llvm {
namespace NVPTXISD {

enum NodeType : unsigned {
  SETP_F16X2 = GENERATED_OPCODE_END,
  SETP_BF16X2,
  UNPACK_VECTOR,

  FIRST_MEMORY_OPCODE,
  LoadV2 = FIRST_MEMORY_OPCODE,
  LoadV4,
  LoadV8,
  LDUV2, // LDU.v2
  LDUV4, // LDU.v4
  StoreV2,
  StoreV4,
  StoreV8,
  LAST_MEMORY_OPCODE = StoreV8,
};

} // namespace NVPTXISD

class NVPTXSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  NVPTXSelectionDAGInfo();

  ~NVPTXSelectionDAGInfo() override;

  const char *getTargetNodeName(unsigned Opcode) const override;

  bool isTargetMemoryOpcode(unsigned Opcode) const override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_NVPTX_NVPTXSELECTIONDAGINFO_H
