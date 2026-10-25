//===- SelectionDAGUseTest.cpp - SelectionDAG use queries
//------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SelectionDAGTestBase.h"

using namespace llvm;

class SelectionDAGUseTest : public SelectionDAGTestBase {
protected:
  SDLoc DL;

  SDValue reg(unsigned Index) {
    return DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                               Register::index2VirtReg(Index), MVT::i64);
  }

  SDValue one() { return DAG->getConstant(1, DL, MVT::i64); }
};

TEST_F(SelectionDAGUseTest, AddSubUsersInsertedAndDeleted) {
  SDValue Base = reg(1);
  EXPECT_FALSE(Base->hasAnyAddOrSubUser());
  DAG->getNode(ISD::OR, DL, MVT::i64, Base, one());
  EXPECT_FALSE(Base->hasAnyAddOrSubUser());

  for (unsigned Opc : {ISD::ADD, ISD::SUB}) {
    SDValue User = DAG->getNode(Opc, DL, MVT::i64, Base, one());
    EXPECT_TRUE(Base->hasAnyAddOrSubUser());
    DAG->DeleteNode(User.getNode());
    EXPECT_FALSE(Base->hasAnyAddOrSubUser());
  }
}

TEST_F(SelectionDAGUseTest, UpdateOperandsAddsUserToCachedBase) {
  SDValue Before = reg(1);
  SDValue After = reg(2);
  for (bool ArrayForm : {false, true}) {
    SDValue User = DAG->getNode(ISD::ADD, DL, MVT::i64, Before, one());
    EXPECT_FALSE(After->hasAnyAddOrSubUser());
    SDValue Ops[] = {After, one()};
    SDNode *Updated =
        ArrayForm ? DAG->UpdateNodeOperands(User.getNode(), Ops)
                  : DAG->UpdateNodeOperands(User.getNode(), Ops[0], Ops[1]);
    EXPECT_EQ(Updated, User.getNode());
    EXPECT_TRUE(After->hasAnyAddOrSubUser());
    EXPECT_FALSE(Before->hasAnyAddOrSubUser());
    DAG->DeleteNode(Updated);
    EXPECT_FALSE(After->hasAnyAddOrSubUser());
  }
}

TEST_F(SelectionDAGUseTest, ReplaceValueAddsUserToCachedBase) {
  SDValue Before = DAG->getNode(ISD::XOR, DL, MVT::i64, reg(1), one());
  SDValue After = reg(2);
  SDValue User = DAG->getNode(ISD::ADD, DL, MVT::i64, Before, one());
  EXPECT_FALSE(After->hasAnyAddOrSubUser());

  DAG->ReplaceAllUsesWith(Before, After);
  EXPECT_EQ(User.getOperand(0), After);
  EXPECT_TRUE(After->hasAnyAddOrSubUser());
  EXPECT_FALSE(Before->hasAnyAddOrSubUser());
}

TEST_F(SelectionDAGUseTest, ReplaceValueMergesExistingCandidate) {
  SDValue Before = DAG->getNode(ISD::XOR, DL, MVT::i64, reg(1), one());
  SDValue After = DAG->getNode(ISD::XOR, DL, MVT::i64, reg(2), one());
  EXPECT_FALSE(After->hasAnyAddOrSubUser());
  SDValue Existing = DAG->getNode(ISD::ADD, DL, MVT::i64, After, one());
  SDValue Replaced = DAG->getNode(ISD::ADD, DL, MVT::i64, Before, one());
  HandleSDNode Handle(Replaced);

  DAG->ReplaceAllUsesWith(Before, After);
  EXPECT_EQ(Handle.getValue(), Existing);
  EXPECT_FALSE(Before->hasAnyAddOrSubUser());
  EXPECT_TRUE(After->hasAnyAddOrSubUser());
}

TEST_F(SelectionDAGUseTest, ReplaceNodeAddsUserToCachedBase) {
  SDValue Before = reg(1);
  SDValue After = reg(2);
  SDValue User = DAG->getNode(ISD::SUB, DL, MVT::i64, Before, one());
  EXPECT_FALSE(After->hasAnyAddOrSubUser());

  // CopyFromReg has two results, so this exercises SDUse::setNode.
  DAG->ReplaceAllUsesWith(Before.getNode(), After.getNode());
  EXPECT_EQ(User.getOperand(0), After);
  EXPECT_TRUE(After->hasAnyAddOrSubUser());
  EXPECT_FALSE(Before->hasAnyAddOrSubUser());
}

TEST_F(SelectionDAGUseTest, MorphCreatesAndRemovesAddUser) {
  SDValue Base = reg(1);
  SDValue User = DAG->getNode(ISD::OR, DL, MVT::i64, Base, one());
  EXPECT_FALSE(Base->hasAnyAddOrSubUser());
  SDValue Ops[] = {Base, one()};
  SDVTList VTs = DAG->getVTList(MVT::i64);

  EXPECT_EQ(DAG->MorphNodeTo(User.getNode(), ISD::ADD, VTs, Ops),
            User.getNode());
  EXPECT_TRUE(Base->hasAnyAddOrSubUser());
  EXPECT_EQ(DAG->MorphNodeTo(User.getNode(), ISD::OR, VTs, Ops),
            User.getNode());
  EXPECT_FALSE(Base->hasAnyAddOrSubUser());
}

TEST_F(SelectionDAGUseTest, QueryIncludesAllResults) {
  SDValue Pair = DAG->getNode(
      ISD::SMUL_LOHI, DL, DAG->getVTList(MVT::i64, MVT::i64), reg(1), reg(2));
  EXPECT_FALSE(Pair->hasAnyAddOrSubUser());
  DAG->getNode(ISD::ADD, DL, MVT::i64, Pair.getValue(1), one());
  EXPECT_TRUE(Pair->hasAnyAddOrSubUser());
}

TEST_F(SelectionDAGUseTest, CachedAnswerDoesNotAffectMemoryCSE) {
  SDValue Ptr = DAG->CreateStackTemporary(MVT::i64);
  auto Load = [&] {
    return DAG->getLoad(MVT::i64, DL, DAG->getEntryNode(), Ptr,
                        MachinePointerInfo(), Align(8));
  };
  SDValue Original = Load();
  EXPECT_FALSE(Original->hasAnyAddOrSubUser());
  EXPECT_EQ(Load(), Original);
  DAG->DeleteNode(Original.getNode());

  SDValue New = Load();
  EXPECT_FALSE(New->hasAnyAddOrSubUser());
  DAG->getNode(ISD::ADD, DL, MVT::i64, New, one());
  EXPECT_TRUE(New->hasAnyAddOrSubUser());
  EXPECT_EQ(Load(), New);
}
