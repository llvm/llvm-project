//===- SelectionDAGOperationNameTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class NamedSDNode : public SDNode {
public:
  explicit NamedSDNode(unsigned Opcode)
      : SDNode(Opcode, 0, DebugLoc(), getSDVTList(MVT::Other)) {}
};

TEST(SelectionDAGOperationNameTest, FixedOperationNames) {
#define DAG_NODE_NAME(OPCODE, NAME)                                            \
  EXPECT_EQ(NAME, NamedSDNode(ISD::OPCODE).getOperationName(nullptr));
#include "llvm/CodeGen/SelectionDAGOperationNames.def"
#undef DAG_NODE_NAME
}

} // namespace
