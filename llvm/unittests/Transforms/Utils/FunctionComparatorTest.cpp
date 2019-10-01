//===- FunctionComparator.cpp - Unit tests for FunctionComparator ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Utils/FunctionComparator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

/// Generates a simple test function.
struct TestFunction {
  Function *F;
  BasicBlock *BB;
  Constant *C;
  Instruction *I;
  Type *T;

  TestFunction(LLVMContext &Ctx, Module &M, int addVal) {
    IRBuilder<> B(Ctx);
    T = B.getInt8Ty();
    F = Function::Create(FunctionType::get(T, {B.getInt8PtrTy()}, false),
                         GlobalValue::ExternalLinkage, "F", &M);
    BB = BasicBlock::Create(Ctx, "", F);
    B.SetInsertPoint(BB);
    Argument *PointerArg = &*F->arg_begin();
    LoadInst *LoadInst = B.CreateLoad(T, PointerArg);
    C = B.getInt8(addVal);
    I = cast<Instruction>(B.CreateAdd(LoadInst, C));
    B.CreateRet(I);
  }
};

/// A class for testing the FunctionComparator API.
///
/// The main purpose is to test if the required protected functions are
/// accessible from a derived class of FunctionComparator.
class TestComparator : public FunctionComparator {
public:
  TestComparator(const Function *F1, const Function *F2,
                 GlobalNumberState *GN)
        : FunctionComparator(F1, F2, GN) {
  }

  bool testFunctionAccess(const Function *F1, const Function *F2) {
    // Test if FnL and FnR are accessible.
    return F1 == FnL && F2 == FnR;
  }

  int testCompare() {
    return compare();
  }

  int testCompareSignature() {
    beginCompare();
    return compareSignature();
  }

  int testCmpBasicBlocks(BasicBlock *BBL, BasicBlock *BBR) {
    beginCompare();
    return cmpBasicBlocks(BBL, BBR);
  }

  int testCmpConstants(const Constant *L, const Constant *R) {
    beginCompare();
    return cmpConstants(L, R);
  }

  int testCmpGlobalValues(GlobalValue *L, GlobalValue *R) {
    beginCompare();
    return cmpGlobalValues(L, R);
  }

  int testCmpValues(const Value *L, const Value *R) {
    beginCompare();
    return cmpValues(L, R);
  }

  int testCmpOperations(const Instruction *L, const Instruction *R,
                        bool &needToCmpOperands) {
    beginCompare();
    return cmpOperations(L, R, needToCmpOperands);
  }

  int testCmpTypes(Type *TyL, Type *TyR) {
    beginCompare();
    return cmpTypes(TyL, TyR);
  }

  int testCmpPrimitives() {
    beginCompare();
    return
      cmpNumbers(2, 3) +
      cmpAPInts(APInt(32, 2), APInt(32, 3)) +
      cmpAPFloats(APFloat(2.0), APFloat(3.0)) +
      cmpMem("2", "3");
  }
};

/// A sanity check for the FunctionComparator API.
TEST(FunctionComparatorTest, TestAPI) {
  LLVMContext C;
  Module M("test", C);
  TestFunction F1(C, M, 27);
  TestFunction F2(C, M, 28);

  GlobalNumberState GN;
  TestComparator Cmp(F1.F, F2.F, &GN);

  EXPECT_TRUE(Cmp.testFunctionAccess(F1.F, F2.F));
  EXPECT_EQ(Cmp.testCompare(), -1);
  EXPECT_EQ(Cmp.testCompareSignature(), 0);
  EXPECT_EQ(Cmp.testCmpBasicBlocks(F1.BB, F2.BB), -1);
  EXPECT_EQ(Cmp.testCmpConstants(F1.C, F2.C), -1);
  EXPECT_EQ(Cmp.testCmpGlobalValues(F1.F, F2.F), -1);
  EXPECT_EQ(Cmp.testCmpValues(F1.I, F2.I), 0);
  bool needToCmpOperands = false;
  EXPECT_EQ(Cmp.testCmpOperations(F1.I, F2.I, needToCmpOperands), 0);
  EXPECT_TRUE(needToCmpOperands);
  EXPECT_EQ(Cmp.testCmpTypes(F1.T, F2.T), 0);
  EXPECT_EQ(Cmp.testCmpPrimitives(), -4);
}
