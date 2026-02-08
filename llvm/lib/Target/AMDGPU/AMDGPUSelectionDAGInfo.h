//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "AMDGPUGenSDNodeInfo.inc"

namespace llvm {
namespace AMDGPUISD {

enum NodeType : unsigned {
  // Convert a unswizzled wave uniform stack address to an address compatible
  // with a vector offset for use in stack access.
  WAVE_ADDRESS = GENERATED_OPCODE_END,

  DOT4,
  MAD_U64_U32,
  MAD_I64_I32,
  TEXTURE_FETCH,
  R600_EXPORT,
  CONST_ADDRESS,

  /// This node is for VLIW targets and it is used to represent a vector
  /// that is stored in consecutive registers with the same channel.
  /// For example:
  ///   |X  |Y|Z|W|
  /// T0|v.x| | | |
  /// T1|v.y| | | |
  /// T2|v.z| | | |
  /// T3|v.w| | | |
  BUILD_VERTICAL_VECTOR,

  DUMMY_CHAIN,
};

} // namespace AMDGPUISD

class AMDGPUSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  AMDGPUSelectionDAGInfo();

  ~AMDGPUSelectionDAGInfo() override;

  const char *getTargetNodeName(unsigned Opcode) const override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSELECTIONDAGINFO_H
