//===---- llvm/unittest/CodeGen/SelectionDAGPatternMatchTest.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SelectionDAGTestBase.h"

using namespace llvm;

class SelectionDAGNodeConstructionTest : public SelectionDAGTestBase {};

TEST_F(SelectionDAGNodeConstructionTest, ADD) {
  SDLoc DL;
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, MVT::i32, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, MVT::i32, Undef, Op), Undef);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, AND) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::AND, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::AND, DL, MVT::i32, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, MVT::i32, Undef, Op), Zero);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, MUL) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, MVT::i32, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, MVT::i32, Undef, Op), Zero);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, OR) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue AllOnes = DAG->getAllOnesConstant(DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::OR, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::OR, DL, MVT::i32, Op, Undef), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, MVT::i32, Undef, Op), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, SADDSAT) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue AllOnes = DAG->getAllOnesConstant(DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, MVT::i32, Op, Undef), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, MVT::i32, Undef, Op), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, SDIV) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, MVT::i32, Op, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, MVT::i32, Undef, Op), Zero);
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, MVT::i32, Undef, Undef), Poison);
}

TEST_F(SelectionDAGNodeConstructionTest, SMAX) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue MaxInt = DAG->getConstant(APInt::getSignedMaxValue(32), DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, MVT::i32, Op, Undef), MaxInt);
  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, MVT::i32, Undef, Op), MaxInt);
  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, SMIN) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue MinInt = DAG->getConstant(APInt::getSignedMinValue(32), DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, MVT::i32, Op, Undef), MinInt);
  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, MVT::i32, Undef, Op), MinInt);
  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, SREM) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, MVT::i32, Op, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, MVT::i32, Undef, Op), Zero);
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, MVT::i32, Undef, Undef), Poison);
}

TEST_F(SelectionDAGNodeConstructionTest, SSUBSAT) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, MVT::i32, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, MVT::i32, Undef, Op), Zero);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, SUB) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, MVT::i32, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, MVT::i32, Undef, Op), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, UADDSAT) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue AllOnes = DAG->getAllOnesConstant(DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, MVT::i32, Op, Undef), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, MVT::i32, Undef, Op), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, UDIV) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, MVT::i32, Op, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, MVT::i32, Undef, Op), Zero);
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, MVT::i32, Undef, Undef), Poison);
}

TEST_F(SelectionDAGNodeConstructionTest, UMAX) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue AllOnes = DAG->getAllOnesConstant(DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, MVT::i32, Op, Undef), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, MVT::i32, Undef, Op), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, UMIN) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, MVT::i32, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, MVT::i32, Undef, Op), Zero);
  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, UREM) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, MVT::i32, Op, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, MVT::i32, Undef, Op), Zero);
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, MVT::i32, Undef, Undef), Poison);
}

TEST_F(SelectionDAGNodeConstructionTest, USUBSAT) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, MVT::i32, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, MVT::i32, Undef, Op), Zero);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, MVT::i32, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, XOR) {
  SDLoc DL;
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, MVT::i32);
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);

  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, MVT::i32, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, MVT::i32, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, MVT::i32, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, MVT::i32, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, MVT::i32, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, MVT::i32, Undef, Op), Undef);
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, MVT::i32, Undef, Undef), Zero);
}
