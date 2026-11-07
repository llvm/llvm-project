//===- PostIndexCandidateCacheTest.cpp - Post-index cache tests
//------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../lib/CodeGen/SelectionDAG/PostIndexCandidateCache.h"
#include "SelectionDAGTestBase.h"

using namespace llvm;

class PostIndexCandidateCacheTest : public SelectionDAGTestBase {
protected:
  SDLoc DL;

  SDValue reg(unsigned Index) {
    return DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                               Register::index2VirtReg(Index), MVT::i64);
  }

  SDValue one() { return DAG->getConstant(1, DL, MVT::i64); }
};

TEST_F(PostIndexCandidateCacheTest, AddSubUsersInsertedAndDeleted) {
  PostIndexCandidateCache Cache(*DAG);
  SDValue Base = reg(1);
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Base.getNode()));
  DAG->getNode(ISD::OR, DL, MVT::i64, Base, one());
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Base.getNode()));

  for (unsigned Opc : {ISD::ADD, ISD::SUB}) {
    SDValue User = DAG->getNode(Opc, DL, MVT::i64, Base, one());
    EXPECT_TRUE(Cache.hasAnyAddOrSubUser(Base.getNode()));
    DAG->DeleteNode(User.getNode());
    EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Base.getNode()));
  }
}

TEST_F(PostIndexCandidateCacheTest, UpdateOperandsAddsUserToCachedBase) {
  PostIndexCandidateCache Cache(*DAG);
  SDValue Before = reg(1);
  SDValue After = reg(2);
  for (bool ArrayForm : {false, true}) {
    SDValue User = DAG->getNode(ISD::ADD, DL, MVT::i64, Before, one());
    EXPECT_FALSE(Cache.hasAnyAddOrSubUser(After.getNode()));
    SDValue Ops[] = {After, one()};
    SDNode *Updated =
        ArrayForm ? DAG->UpdateNodeOperands(User.getNode(), Ops)
                  : DAG->UpdateNodeOperands(User.getNode(), Ops[0], Ops[1]);
    EXPECT_EQ(Updated, User.getNode());
    EXPECT_TRUE(Cache.hasAnyAddOrSubUser(After.getNode()));
    EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Before.getNode()));
    DAG->DeleteNode(Updated);
    EXPECT_FALSE(Cache.hasAnyAddOrSubUser(After.getNode()));
  }
}

TEST_F(PostIndexCandidateCacheTest, ReplaceValueAddsUserToCachedBase) {
  PostIndexCandidateCache Cache(*DAG);
  SDValue Before = DAG->getNode(ISD::XOR, DL, MVT::i64, reg(1), one());
  SDValue After = reg(2);
  SDValue User = DAG->getNode(ISD::ADD, DL, MVT::i64, Before, one());
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(After.getNode()));

  DAG->ReplaceAllUsesWith(Before, After);
  EXPECT_EQ(User.getOperand(0), After);
  EXPECT_TRUE(Cache.hasAnyAddOrSubUser(After.getNode()));
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Before.getNode()));
}

TEST_F(PostIndexCandidateCacheTest, ReplaceValueMergesExistingCandidate) {
  PostIndexCandidateCache Cache(*DAG);
  SDValue Before = DAG->getNode(ISD::XOR, DL, MVT::i64, reg(1), one());
  SDValue After = DAG->getNode(ISD::XOR, DL, MVT::i64, reg(2), one());
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(After.getNode()));
  SDValue Existing = DAG->getNode(ISD::ADD, DL, MVT::i64, After, one());
  SDValue Replaced = DAG->getNode(ISD::ADD, DL, MVT::i64, Before, one());
  HandleSDNode Handle(Replaced);

  DAG->ReplaceAllUsesWith(Before, After);
  EXPECT_EQ(Handle.getValue(), Existing);
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Before.getNode()));
  EXPECT_TRUE(Cache.hasAnyAddOrSubUser(After.getNode()));
}

TEST_F(PostIndexCandidateCacheTest, ReplaceNodeAddsUserToCachedBase) {
  PostIndexCandidateCache Cache(*DAG);
  SDValue Before = reg(1);
  SDValue After = reg(2);
  SDValue User = DAG->getNode(ISD::SUB, DL, MVT::i64, Before, one());
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(After.getNode()));

  // CopyFromReg has two results, so this exercises SDUse::setNode.
  DAG->ReplaceAllUsesWith(Before.getNode(), After.getNode());
  EXPECT_EQ(User.getOperand(0), After);
  EXPECT_TRUE(Cache.hasAnyAddOrSubUser(After.getNode()));
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Before.getNode()));
}

TEST_F(PostIndexCandidateCacheTest, MorphCreatesAndRemovesAddUser) {
  PostIndexCandidateCache Cache(*DAG);
  SDValue Base = reg(1);
  SDValue User = DAG->getNode(ISD::OR, DL, MVT::i64, Base, one());
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Base.getNode()));
  SDValue Ops[] = {Base, one()};
  SDVTList VTs = DAG->getVTList(MVT::i64);

  EXPECT_EQ(DAG->MorphNodeTo(User.getNode(), ISD::ADD, VTs, Ops),
            User.getNode());
  EXPECT_TRUE(Cache.hasAnyAddOrSubUser(Base.getNode()));
  EXPECT_EQ(DAG->MorphNodeTo(User.getNode(), ISD::OR, VTs, Ops),
            User.getNode());
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Base.getNode()));
}

TEST_F(PostIndexCandidateCacheTest, QueryIncludesAllResults) {
  PostIndexCandidateCache Cache(*DAG);
  SDValue Pair = DAG->getNode(
      ISD::SMUL_LOHI, DL, DAG->getVTList(MVT::i64, MVT::i64), reg(1), reg(2));
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Pair.getNode()));
  DAG->getNode(ISD::ADD, DL, MVT::i64, Pair.getValue(1), one());
  EXPECT_TRUE(Cache.hasAnyAddOrSubUser(Pair.getNode()));
}

TEST_F(PostIndexCandidateCacheTest, CachedAnswerDoesNotAffectMemoryCSE) {
  PostIndexCandidateCache Cache(*DAG);
  SDValue Ptr = DAG->CreateStackTemporary(MVT::i64);
  auto Load = [&] {
    return DAG->getLoad(MVT::i64, DL, DAG->getEntryNode(), Ptr,
                        MachinePointerInfo(), Align(8));
  };
  SDValue Original = Load();
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Original.getNode()));
  EXPECT_EQ(Load(), Original);
  DAG->DeleteNode(Original.getNode());

  SDValue New = Load();
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(New.getNode()));
  DAG->getNode(ISD::ADD, DL, MVT::i64, New, one());
  EXPECT_TRUE(Cache.hasAnyAddOrSubUser(New.getNode()));
  EXPECT_EQ(Load(), New);
}

TEST_F(PostIndexCandidateCacheTest, UpdateOperandsCSEDoesNotChangeOriginal) {
  PostIndexCandidateCache Cache(*DAG);
  SDValue Before = reg(1);
  SDValue After = reg(2);
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(Before.getNode()));
  EXPECT_FALSE(Cache.hasAnyAddOrSubUser(After.getNode()));
  SDValue Original = DAG->getNode(ISD::ADD, DL, MVT::i64, Before, one());
  SDValue Existing = DAG->getNode(ISD::ADD, DL, MVT::i64, After, one());
  EXPECT_EQ(DAG->UpdateNodeOperands(Original.getNode(), After, one()),
            Existing.getNode());
  EXPECT_EQ(Original.getOperand(0), Before);
  EXPECT_TRUE(Cache.hasAnyAddOrSubUser(Before.getNode()));
  EXPECT_TRUE(Cache.hasAnyAddOrSubUser(After.getNode()));
}

TEST_F(PostIndexCandidateCacheTest, ExplicitUpdatesHaveTheirOwnNotification) {
  struct Listener : SelectionDAG::DAGUpdateListener {
    unsigned ExplicitUpdates = 0;
    unsigned RAUWUpdates = 0;
    explicit Listener(SelectionDAG &DAG) : DAGUpdateListener(DAG) {}
    void NodeOperandsChanged(SDNode *) override { ++ExplicitUpdates; }
    void NodeUpdated(SDNode *) override { ++RAUWUpdates; }
  } Listener(*DAG);

  SDValue Before = reg(1);
  SDValue After = reg(2);
  SDValue User = DAG->getNode(ISD::BSWAP, DL, MVT::i64, Before);
  EXPECT_EQ(DAG->UpdateNodeOperands(User.getNode(), After), User.getNode());
  EXPECT_EQ(Listener.ExplicitUpdates, 1u);
  EXPECT_EQ(DAG->UpdateNodeOperands(User.getNode(), After), User.getNode());
  EXPECT_EQ(Listener.ExplicitUpdates, 1u);

  SDValue Existing = DAG->getNode(ISD::BSWAP, DL, MVT::i64, Before);
  EXPECT_EQ(DAG->UpdateNodeOperands(User.getNode(), Before),
            Existing.getNode());
  EXPECT_EQ(Listener.ExplicitUpdates, 1u);
  EXPECT_EQ(DAG->MorphNodeTo(User.getNode(), ISD::BITREVERSE,
                             DAG->getVTList(MVT::i64), {After}),
            User.getNode());
  EXPECT_EQ(Listener.ExplicitUpdates, 2u);
  EXPECT_EQ(Listener.RAUWUpdates, 0u);
}
