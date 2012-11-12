//===- llvm/unittests/MDBuilderTest.cpp - MDBuilder unit tests ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IRBuilder.h"
#include "llvm/MDBuilder.h"
#include "llvm/Operator.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

class MDBuilderTest : public testing::Test {
protected:
  LLVMContext Context;
};

TEST_F(MDBuilderTest, createString) {
  MDBuilder MDHelper(Context);
  MDString *Str0 = MDHelper.createString("");
  MDString *Str1 = MDHelper.createString("string");
  EXPECT_EQ(Str0->getString(), StringRef(""));
  EXPECT_EQ(Str1->getString(), StringRef("string"));
}
TEST_F(MDBuilderTest, createFPMath) {
  MDBuilder MDHelper(Context);
  MDNode *MD0 = MDHelper.createFPMath(0.0);
  MDNode *MD1 = MDHelper.createFPMath(1.0);
  EXPECT_EQ(MD0, (MDNode *)0);
  EXPECT_NE(MD1, (MDNode *)0);
  EXPECT_EQ(MD1->getNumOperands(), 1U);
  Value *Op = MD1->getOperand(0);
  EXPECT_TRUE(isa<ConstantFP>(Op));
  EXPECT_TRUE(Op->getType()->isFloatingPointTy());
  ConstantFP *Val = cast<ConstantFP>(Op);
  EXPECT_TRUE(Val->isExactlyValue(1.0));
}
TEST_F(MDBuilderTest, createRangeMetadata) {
  MDBuilder MDHelper(Context);
  APInt A(8, 1), B(8, 2);
  MDNode *R0 = MDHelper.createRange(A, A);
  MDNode *R1 = MDHelper.createRange(A, B);
  EXPECT_EQ(R0, (MDNode *)0);
  EXPECT_NE(R1, (MDNode *)0);
  EXPECT_EQ(R1->getNumOperands(), 2U);
  EXPECT_TRUE(isa<ConstantInt>(R1->getOperand(0)));
  EXPECT_TRUE(isa<ConstantInt>(R1->getOperand(1)));
  ConstantInt *C0 = cast<ConstantInt>(R1->getOperand(0));
  ConstantInt *C1 = cast<ConstantInt>(R1->getOperand(1));
  EXPECT_EQ(C0->getValue(), A);
  EXPECT_EQ(C1->getValue(), B);
}
TEST_F(MDBuilderTest, createAnonymousTBAARoot) {
  MDBuilder MDHelper(Context);
  MDNode *R0 = MDHelper.createAnonymousTBAARoot();
  MDNode *R1 = MDHelper.createAnonymousTBAARoot();
  EXPECT_NE(R0, R1);
  EXPECT_GE(R0->getNumOperands(), 1U);
  EXPECT_GE(R1->getNumOperands(), 1U);
  EXPECT_EQ(R0->getOperand(0), R0);
  EXPECT_EQ(R1->getOperand(0), R1);
  EXPECT_TRUE(R0->getNumOperands() == 1 || R0->getOperand(1) == 0);
  EXPECT_TRUE(R1->getNumOperands() == 1 || R1->getOperand(1) == 0);
}
TEST_F(MDBuilderTest, createTBAARoot) {
  MDBuilder MDHelper(Context);
  MDNode *R0 = MDHelper.createTBAARoot("Root");
  MDNode *R1 = MDHelper.createTBAARoot("Root");
  EXPECT_EQ(R0, R1);
  EXPECT_GE(R0->getNumOperands(), 1U);
  EXPECT_TRUE(isa<MDString>(R0->getOperand(0)));
  EXPECT_EQ(cast<MDString>(R0->getOperand(0))->getString(), "Root");
  EXPECT_TRUE(R0->getNumOperands() == 1 || R0->getOperand(1) == 0);
}
TEST_F(MDBuilderTest, createTBAANode) {
  MDBuilder MDHelper(Context);
  MDNode *R = MDHelper.createTBAARoot("Root");
  MDNode *N0 = MDHelper.createTBAANode("Node", R);
  MDNode *N1 = MDHelper.createTBAANode("edoN", R);
  MDNode *N2 = MDHelper.createTBAANode("Node", R, true);
  MDNode *N3 = MDHelper.createTBAANode("Node", R);
  EXPECT_EQ(N0, N3);
  EXPECT_NE(N0, N1);
  EXPECT_NE(N0, N2);
  EXPECT_GE(N0->getNumOperands(), 2U);
  EXPECT_GE(N1->getNumOperands(), 2U);
  EXPECT_GE(N2->getNumOperands(), 3U);
  EXPECT_TRUE(isa<MDString>(N0->getOperand(0)));
  EXPECT_TRUE(isa<MDString>(N1->getOperand(0)));
  EXPECT_TRUE(isa<MDString>(N2->getOperand(0)));
  EXPECT_EQ(cast<MDString>(N0->getOperand(0))->getString(), "Node");
  EXPECT_EQ(cast<MDString>(N1->getOperand(0))->getString(), "edoN");
  EXPECT_EQ(cast<MDString>(N2->getOperand(0))->getString(), "Node");
  EXPECT_EQ(N0->getOperand(1), R);
  EXPECT_EQ(N1->getOperand(1), R);
  EXPECT_EQ(N2->getOperand(1), R);
  EXPECT_TRUE(isa<ConstantInt>(N2->getOperand(2)));
  EXPECT_EQ(cast<ConstantInt>(N2->getOperand(2))->getZExtValue(), 1U);
}
}
