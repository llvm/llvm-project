//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NVPTXSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "NVPTXGenSDNodeInfo.inc"

using namespace llvm;

NVPTXSelectionDAGInfo::NVPTXSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(NVPTXGenSDNodeInfo) {}

NVPTXSelectionDAGInfo::~NVPTXSelectionDAGInfo() = default;

const char *NVPTXSelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
#define MAKE_CASE(V)                                                           \
  case V:                                                                      \
    return #V;

  // These nodes don't have corresponding entries in *.td files yet.
  switch (static_cast<NVPTXISD::NodeType>(Opcode)) {
    MAKE_CASE(NVPTXISD::ATOMIC_CMP_SWAP_B128)
    MAKE_CASE(NVPTXISD::ATOMIC_SWAP_B128)
    MAKE_CASE(NVPTXISD::LoadV2)
    MAKE_CASE(NVPTXISD::LoadV4)
    MAKE_CASE(NVPTXISD::LoadV8)
    MAKE_CASE(NVPTXISD::LDUV2)
    MAKE_CASE(NVPTXISD::LDUV4)
    MAKE_CASE(NVPTXISD::StoreV2)
    MAKE_CASE(NVPTXISD::StoreV4)
    MAKE_CASE(NVPTXISD::StoreV8)
    MAKE_CASE(NVPTXISD::SETP_F16X2)
    MAKE_CASE(NVPTXISD::SETP_BF16X2)
    MAKE_CASE(NVPTXISD::UNPACK_VECTOR)
  }
#undef MAKE_CASE

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}

bool NVPTXSelectionDAGInfo::isTargetMemoryOpcode(unsigned Opcode) const {
  // These nodes don't have corresponding entries in *.td files.
  if (Opcode >= NVPTXISD::FIRST_MEMORY_OPCODE &&
      Opcode <= NVPTXISD::LAST_MEMORY_OPCODE)
    return true;

  return SelectionDAGGenTargetInfo::isTargetMemoryOpcode(Opcode);
}

void NVPTXSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                             const SDNode *N) const {
  switch (N->getOpcode()) {
  default:
    break;
  case NVPTXISD::ProxyReg:
    // invalid number of results; expected 2, got 1
    return;
  }

  return SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}
