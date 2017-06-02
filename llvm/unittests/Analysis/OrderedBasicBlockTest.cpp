//===- OrderedBasicBlockTest.cpp - OrderedBasicBlock unit tests -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OrderedBasicBlock.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class OrderedBasicBlockTest : public testing::Test {
protected:
  LLVMContext C;

  std::unique_ptr<Module> makeLLVMModule() {
    const char *ModuleString = R"(define i32 @f(i32 %x) {
                                    %add = add i32 %x, 42
                                    ret i32 %add
                                  })";
    SMDiagnostic Err;
    auto foo = parseAssemblyString(ModuleString, Err, C);
    return foo;
  }
};

TEST_F(OrderedBasicBlockTest, Basic) {
  auto M = makeLLVMModule();
  Function *F = M->getFunction("f");
  BasicBlock::iterator I = F->front().begin();
  Instruction *Add = &*I++;
  Instruction *Ret = &*I++;

  OrderedBasicBlock OBB(&F->front());
  // Intentionally duplicated to verify cached and uncached are the same.
  EXPECT_FALSE(OBB.dominates(Add, Add));
  EXPECT_FALSE(OBB.dominates(Add, Add));
  EXPECT_TRUE(OBB.dominates(Add, Ret));
  EXPECT_TRUE(OBB.dominates(Add, Ret));
  EXPECT_FALSE(OBB.dominates(Ret, Add));
  EXPECT_FALSE(OBB.dominates(Ret, Add));
  EXPECT_FALSE(OBB.dominates(Ret, Ret));
  EXPECT_FALSE(OBB.dominates(Ret, Ret));
}

} // end anonymous namespace
} // end namespace llvm
