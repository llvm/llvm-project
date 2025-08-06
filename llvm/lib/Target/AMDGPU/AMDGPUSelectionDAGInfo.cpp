//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "AMDGPUGenSDNodeInfo.inc"

using namespace llvm;

AMDGPUSelectionDAGInfo::AMDGPUSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(AMDGPUGenSDNodeInfo) {}

AMDGPUSelectionDAGInfo::~AMDGPUSelectionDAGInfo() = default;

const char *AMDGPUSelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
#define NODE_NAME_CASE(node)                                                   \
  case AMDGPUISD::node:                                                        \
    return #node;

  switch (static_cast<AMDGPUISD::NodeType>(Opcode)) {
    // These nodes don't have corresponding entries in *.td files yet.
    NODE_NAME_CASE(WAVE_ADDRESS)
    NODE_NAME_CASE(MAD_I64_I32)
    NODE_NAME_CASE(MAD_U64_U32)
    NODE_NAME_CASE(BUILD_VERTICAL_VECTOR)
    // These do, but only when compiling R600.td,
    // and the enum is generated from AMDGPU.td.
    NODE_NAME_CASE(DOT4)
    NODE_NAME_CASE(TEXTURE_FETCH)
    NODE_NAME_CASE(R600_EXPORT)
    NODE_NAME_CASE(CONST_ADDRESS)
    NODE_NAME_CASE(DUMMY_CHAIN)
  }

#undef NODE_NAME_CASE

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}
