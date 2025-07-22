//===---- llvm/unittest/CodeGen/SelectionDAGPatternMatchTest.cpp  ---------===//
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
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Poison, Undef), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Undef, Poison), Undef);

  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Undef, Op), Undef);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, AND) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Poison, Undef), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Undef, Poison), Zero);

  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Undef, Undef), Zero);
}

TEST_F(SelectionDAGNodeConstructionTest, MUL) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Poison, Undef), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Undef, Poison), Zero);

  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Undef, Undef), Zero);
}

TEST_F(SelectionDAGNodeConstructionTest, OR) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue AllOnes = DAG->getAllOnesConstant(DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Poison, Undef), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Undef, Poison), AllOnes);

  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Op, Undef), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Undef, Op), AllOnes);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Undef, Undef), AllOnes);
}

TEST_F(SelectionDAGNodeConstructionTest, SADDSAT) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue AllOnes = DAG->getAllOnesConstant(DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Poison, Undef), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Undef, Poison), AllOnes);

  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Op, Undef), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Undef, Op), AllOnes);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Undef, Undef), AllOnes);
}

TEST_F(SelectionDAGNodeConstructionTest, SDIV) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Op, Poison), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Poison, Op), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Poison, Undef), Undef);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Undef, Poison), Undef);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, SREM) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Op, Poison), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Poison, Op), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Poison, Undef), Undef);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Undef, Poison), Undef);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, SSUBSAT) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Poison, Undef), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Undef, Poison), Zero);

  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Undef, Undef), Zero);
}

TEST_F(SelectionDAGNodeConstructionTest, SUB) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Poison, Undef), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Undef, Poison), Undef);

  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Undef, Op), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, UADDSAT) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue AllOnes = DAG->getAllOnesConstant(DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Poison, Undef), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Undef, Poison), AllOnes);

  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Op, Undef), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Undef, Op), AllOnes);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Undef, Undef), AllOnes);
}

TEST_F(SelectionDAGNodeConstructionTest, UDIV) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Op, Poison), Undef);
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Poison, Op), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Poison, Undef), Undef);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Undef, Poison), Undef);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, UREM) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Op, Poison), Undef);
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Poison, Op), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Poison, Undef), Undef);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Undef, Poison), Undef);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGNodeConstructionTest, USUBSAT) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Poison, Undef), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Undef, Poison), Zero);

  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Undef, Undef), Zero);
}

TEST_F(SelectionDAGNodeConstructionTest, XOR) {
  SDLoc DL;
  EVT Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Poison, Op), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Poison, Undef), Zero);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Undef, Poison), Zero);

  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Undef, Op), Undef);
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Undef, Undef), Zero);
}
