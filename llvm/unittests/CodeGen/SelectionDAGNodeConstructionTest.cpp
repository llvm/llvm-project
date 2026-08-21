//===---- llvm/unittest/CodeGen/SelectionDAGPatternMatchTest.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SelectionDAGTestBase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

class SelectionDAGNodeConstructionTest : public SelectionDAGTestBase {
protected:
  SDValue buildVector(EVT VT, EVT ScalarVT, const SDLoc &DL,
                      ArrayRef<int64_t> Values) {
    SmallVector<SDValue, 8> Elts;
    for (int64_t Value : Values)
      Elts.push_back(DAG->getConstant(
          APInt(ScalarVT.getSizeInBits(), Value, /*isSigned=*/true), DL,
          ScalarVT));
    return DAG->getBuildVector(VT, DL, Elts);
  }

  SDValue buildVector(EVT VT, const SDLoc &DL, ArrayRef<int64_t> Values) {
    return buildVector(VT, VT.getVectorElementType(), DL, Values);
  }

  void checkConstant(SDValue Value, int64_t Expected) {
    auto *C = dyn_cast<ConstantSDNode>(Value);
    ASSERT_NE(C, nullptr);
    EXPECT_EQ(C->getSExtValue(), Expected);
  }

  void checkBuildVector(SDValue Result, ArrayRef<int64_t> Expected) {
    ASSERT_EQ(Result.getOpcode(), ISD::BUILD_VECTOR);
    ASSERT_EQ(Result.getNumOperands(), Expected.size());
    for (unsigned I = 0; I != Expected.size(); ++I)
      checkConstant(Result.getOperand(I), Expected[I]);
  }
};

TEST_F(SelectionDAGNodeConstructionTest, ADD) {
  SDLoc DL;
  SDValue Poison = DAG->getPOISON(MVT::i32);
  SDValue Undef = DAG->getUNDEF(MVT::i32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);

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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                   Register::index2VirtReg(1), MVT::i32);
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

TEST_F(SelectionDAGNodeConstructionTest, CTLS) {
  SDLoc DL;
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);
  SDValue MaxInt = DAG->getConstant(0x7fffffff, DL, MVT::i32);
  SDValue MinInt = DAG->getConstant(0x80000000, DL, MVT::i32);
  SDValue MinShort = DAG->getConstant(0xffff8000, DL, MVT::i32);

  SDValue CtlsZero = DAG->getNode(ISD::CTLS, DL, MVT::i32, Zero);
  SDValue CtlsMinInt = DAG->getNode(ISD::CTLS, DL, MVT::i32, MinInt);
  SDValue CtlsMaxInt = DAG->getNode(ISD::CTLS, DL, MVT::i32, MaxInt);
  SDValue CtlsMinShort = DAG->getNode(ISD::CTLS, DL, MVT::i32, MinShort);
  EXPECT_TRUE(isa<ConstantSDNode>(CtlsZero) &&
              cast<ConstantSDNode>(CtlsZero)->getZExtValue() == 31);
  EXPECT_TRUE(isNullConstant(CtlsMinInt));
  EXPECT_TRUE(isNullConstant(CtlsMaxInt));
  EXPECT_TRUE(isa<ConstantSDNode>(CtlsMinShort) &&
              cast<ConstantSDNode>(CtlsMinShort)->getZExtValue() == 16);

  SDValue i1Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                     Register::index2VirtReg(1), MVT::i1);
  SDValue Ctlsi1 = DAG->getNode(ISD::CTLS, DL, MVT::i32, i1Op);
  EXPECT_TRUE(isNullConstant(Ctlsi1));
}

TEST_F(SelectionDAGNodeConstructionTest,
       FoldConstantPartialReduceMLASignednessAndInputWidth) {
  SDLoc DL;
  SDValue PromotedLHS = buildVector(MVT::v4i8, MVT::i32, DL, {255, 128, 0, 0});
  SDValue PromotedRHS = buildVector(MVT::v4i8, MVT::i32, DL, {255, 255, 0, 0});
  SDValue ZeroAcc = buildVector(MVT::v2i32, DL, {0, 0});

  checkBuildVector(DAG->getNode(ISD::PARTIAL_REDUCE_SMLA, DL, MVT::v2i32,
                                ZeroAcc, PromotedLHS, PromotedRHS),
                   {1, 128});
  checkBuildVector(DAG->getNode(ISD::PARTIAL_REDUCE_UMLA, DL, MVT::v2i32,
                                ZeroAcc, PromotedLHS, PromotedRHS),
                   {65025, 32640});
  checkBuildVector(DAG->getNode(ISD::PARTIAL_REDUCE_SUMLA, DL, MVT::v2i32,
                                ZeroAcc, PromotedLHS, PromotedRHS),
                   {-255, -32640});
}

TEST_F(SelectionDAGNodeConstructionTest,
       FoldConstantPartialReduceMLALegalizedType) {
  SDLoc DL;
  // v4i16 is a legal AArch64 vector type but i16 is not a legal scalar type, so
  // after type legalization the folded constants must be created in the
  // promoted (i32) type. Build the operands with their legalized scalar types
  // before enabling the flag.
  SDValue Acc = buildVector(MVT::v4i16, MVT::i32, DL, {100, -200, 300, -400});
  SDValue LHS = buildVector(MVT::v8i8, MVT::i32, DL, {1, 2, 3, 4, 5, 6, 7, 8});
  SDValue RHS =
      buildVector(MVT::v8i8, MVT::i32, DL, {5, 6, 7, 8, 9, 10, 11, 12});

  DAG->NewNodesMustHaveLegalTypes = true;
  SDValue Result =
      DAG->getNode(ISD::PARTIAL_REDUCE_SMLA, DL, MVT::v4i16, Acc, LHS, RHS);
  ASSERT_EQ(Result.getOpcode(), ISD::BUILD_VECTOR);
  // Signed lane values must survive promotion, so check the negative lanes too.
  checkBuildVector(Result, {/*100 + 1*5 + 5*9=*/150, /*-200 + 2*6 + 6*10=*/-128,
                            /*300 + 3*7 + 7*11=*/398,
                            /*-400 + 4*8 + 8*12=*/-272});
  for (SDValue Op : Result->op_values())
    EXPECT_EQ(Op.getValueType(), MVT::i32);
}

TEST_F(SelectionDAGNodeConstructionTest,
       FoldConstantPartialReduceMLARHSPoison) {
  SDLoc DL;
  SDValue Acc = buildVector(MVT::v2i32, DL, {100, 200});
  SDValue LHS = buildVector(MVT::v4i8, DL, {1, 2, 3, 4});
  SDValue RHS = buildVector(MVT::v4i8, DL, {5, 6, 7, 8});
  SmallVector<SDValue, 4> PoisonRHS(RHS->op_values());
  PoisonRHS[2] = DAG->getPOISON(MVT::i8);
  SDValue PoisonResult =
      DAG->getNode(ISD::PARTIAL_REDUCE_SMLA, DL, MVT::v2i32, Acc, LHS,
                   DAG->getBuildVector(MVT::v4i8, DL, PoisonRHS));
  ASSERT_EQ(PoisonResult.getOpcode(), ISD::BUILD_VECTOR);
  EXPECT_EQ(PoisonResult.getOperand(0).getOpcode(), ISD::POISON);
  checkConstant(PoisonResult.getOperand(1), 244);
}

TEST_F(SelectionDAGNodeConstructionTest, DontFoldPartialReduceMLA) {
  SDLoc DL;
  SDValue Acc = buildVector(MVT::v2i32, DL, {100, 200});
  SDValue LHS = buildVector(MVT::v4i8, DL, {1, 2, 3, 4});
  SDValue RHS = buildVector(MVT::v4i8, DL, {5, 6, 7, 8});

  SmallVector<SDValue, 4> SpecialLHS(LHS->op_values());
  SpecialLHS[2] = DAG->getConstant(APInt(8, 1), DL, MVT::i8,
                                   /*isTarget=*/false, /*isOpaque=*/true);
  SDValue OpaqueResult =
      DAG->getNode(ISD::PARTIAL_REDUCE_SMLA, DL, MVT::v2i32, Acc,
                   DAG->getBuildVector(MVT::v4i8, DL, SpecialLHS), RHS);
  EXPECT_EQ(OpaqueResult.getOpcode(), ISD::PARTIAL_REDUCE_SMLA);

  SpecialLHS[2] = DAG->getUNDEF(MVT::i8);
  SDValue UndefResult =
      DAG->getNode(ISD::PARTIAL_REDUCE_SMLA, DL, MVT::v2i32, Acc,
                   DAG->getBuildVector(MVT::v4i8, DL, SpecialLHS), RHS);
  EXPECT_EQ(UndefResult.getOpcode(), ISD::PARTIAL_REDUCE_SMLA);

  SDValue Variable = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                         Register::index2VirtReg(2), MVT::i32);
  SDValue MixedLHS[] = {Variable, DAG->getConstant(2, DL, MVT::i32)};
  SDValue NonConstantResult =
      DAG->getNode(ISD::PARTIAL_REDUCE_SMLA, DL, MVT::v2i32,
                   buildVector(MVT::v2i32, DL, {1, 2}),
                   DAG->getBuildVector(MVT::v2i32, DL, MixedLHS),
                   buildVector(MVT::v2i32, DL, {3, 4}));
  EXPECT_EQ(NonConstantResult.getOpcode(), ISD::PARTIAL_REDUCE_SMLA);
}

TEST_F(SelectionDAGNodeConstructionTest, ExpandPartialReduceSUMLA) {
  SDLoc DL;
  SDValue Acc = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                    Register::index2VirtReg(1), MVT::v4i32);
  SDValue LHS = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                    Register::index2VirtReg(2), MVT::v16i8);
  SDValue RHS = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                    Register::index2VirtReg(3), MVT::v16i8);
  SDValue PartialReduce =
      DAG->getNode(ISD::PARTIAL_REDUCE_SUMLA, DL, MVT::v4i32, Acc, LHS, RHS);

  SDValue Expanded = DAG->getTargetLoweringInfo().expandPartialReduceMLA(
      PartialReduce.getNode(), *DAG);

  unsigned NumSignExtends = 0;
  unsigned NumZeroExtends = 0;
  for (SDNode &N : DAG->allnodes()) {
    NumSignExtends += N.getOpcode() == ISD::SIGN_EXTEND;
    NumZeroExtends += N.getOpcode() == ISD::ZERO_EXTEND;
  }
  EXPECT_TRUE(Expanded);
  EXPECT_EQ(NumSignExtends, 1u);
  EXPECT_EQ(NumZeroExtends, 1u);
}
