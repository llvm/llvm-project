//===--- MixedTBAATest.cpp - Mixed TBAA unit tests ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Analysis/AliasAnalysisEvaluator.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class MixedTBAATest : public testing::Test {
protected:
  MixedTBAATest() : M("MixedTBAATest", C), MD(C) {}

  LLVMContext C;
  Module M;
  MDBuilder MD;
  legacy::PassManager PM;
};

TEST_F(MixedTBAATest, MixedTBAA) {
  // Setup function.
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C),
                                        std::vector<Type *>(), false);
  auto *F = cast<Function>(M.getOrInsertFunction("f", FTy));
  auto *BB = BasicBlock::Create(C, "entry", F);
  auto IntType = Type::getInt32Ty(C);
  auto PtrType = Type::getInt32PtrTy(C);
  auto *Value  = ConstantInt::get(IntType, 42);
  auto *Addr = ConstantPointerNull::get(PtrType);

  auto *Store1 = new StoreInst(Value, Addr, BB);
  auto *Store2 = new StoreInst(Value, Addr, BB);
  ReturnInst::Create(C, nullptr, BB);

  // New TBAA metadata
  {
    auto RootMD = MD.createTBAARoot("Simple C/C++ TBAA");
    auto MD1 = MD.createTBAAScalarTypeNode("omnipotent char", RootMD);
    auto MD2 = MD.createTBAAScalarTypeNode("int", MD1);
    auto MD3 = MD.createTBAAStructTagNode(MD2, MD2, 0);
    Store2->setMetadata(LLVMContext::MD_tbaa, MD3);
  }

  // Old TBAA metadata
  {
    auto RootMD = MD.createTBAARoot("Simple C/C++ TBAA");
    auto MD1 = MD.createTBAANode("omnipotent char", RootMD);
    auto MD2 = MD.createTBAANode("int", MD1);
    Store1->setMetadata(LLVMContext::MD_tbaa, MD2);
  }

  // Run the TBAA eval pass on a mixture of path-aware and non-path-aware TBAA.
  // The order of the metadata (path-aware vs non-path-aware) is important,
  // because the AA eval pass only runs one test per store-pair.
  const char* args[] = { "MixedTBAATest", "-evaluate-aa-metadata" };
  cl::ParseCommandLineOptions(sizeof(args) / sizeof(const char*), args);
  PM.add(createTypeBasedAAWrapperPass());
  PM.add(createAAEvalPass());
  PM.run(M);
}

} // end anonymous namspace
} // end llvm namespace

