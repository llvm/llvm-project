//===- MemoryProfileInfoTest.cpp - Memory Profile Info Unit Tests-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemoryProfileInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <cstring>

using namespace llvm;
using namespace llvm::memprof;

extern cl::opt<float> MemProfAccessesPerByteColdThreshold;
extern cl::opt<unsigned> MemProfMinLifetimeColdThreshold;

namespace {

class MemoryProfileInfoTest : public testing::Test {
protected:
  std::unique_ptr<Module> makeLLVMModule(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
    if (!Mod)
      Err.print("MemoryProfileInfoTest", errs());
    return Mod;
  }

  // This looks for a call that has the given value name, which
  // is the name of the value being assigned the call return value.
  CallBase *findCall(Function &F, const char *Name = nullptr) {
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (!Name || CB->getName() == Name)
            return CB;
    return nullptr;
  }
};

// Test getAllocType helper.
// Basic checks on the allocation type for values just above and below
// the thresholds.
TEST_F(MemoryProfileInfoTest, GetAllocType) {
  // Long lived with more accesses per byte than threshold is not cold.
  EXPECT_EQ(
      getAllocType(/*MaxAccessCount=*/MemProfAccessesPerByteColdThreshold + 1,
                   /*MinSize=*/1,
                   /*MinLifetime=*/MemProfMinLifetimeColdThreshold * 1000 + 1),
      AllocationType::NotCold);
  // Long lived with less accesses per byte than threshold is cold.
  EXPECT_EQ(
      getAllocType(/*MaxAccessCount=*/MemProfAccessesPerByteColdThreshold - 1,
                   /*MinSize=*/1,
                   /*MinLifetime=*/MemProfMinLifetimeColdThreshold * 1000 + 1),
      AllocationType::Cold);
  // Short lived with more accesses per byte than threshold is not cold.
  EXPECT_EQ(
      getAllocType(/*MaxAccessCount=*/MemProfAccessesPerByteColdThreshold + 1,
                   /*MinSize=*/1,
                   /*MinLifetime=*/MemProfMinLifetimeColdThreshold * 1000 - 1),
      AllocationType::NotCold);
  // Short lived with less accesses per byte than threshold is not cold.
  EXPECT_EQ(
      getAllocType(/*MaxAccessCount=*/MemProfAccessesPerByteColdThreshold - 1,
                   /*MinSize=*/1,
                   /*MinLifetime=*/MemProfMinLifetimeColdThreshold * 1000 - 1),
      AllocationType::NotCold);
}

// Test buildCallstackMetadata helper.
TEST_F(MemoryProfileInfoTest, BuildCallStackMD) {
  LLVMContext C;
  MDNode *CallStack = buildCallstackMetadata({1, 2, 3}, C);
  ASSERT_EQ(CallStack->getNumOperands(), 3u);
  unsigned ExpectedId = 1;
  for (auto &Op : CallStack->operands()) {
    auto *StackId = mdconst::dyn_extract<ConstantInt>(Op);
    EXPECT_EQ(StackId->getZExtValue(), ExpectedId++);
  }
}

// Test CallStackTrie::addCallStack interface taking allocation type and list of
// call stack ids.
// Check that allocations with a single allocation type along all call stacks
// get an attribute instead of memprof metadata.
TEST_F(MemoryProfileInfoTest, Attribute) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
define i32* @test() {
entry:
  %call1 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40)
  %0 = bitcast i8* %call1 to i32*
  %call2 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40)
  %1 = bitcast i8* %call2 to i32*
  ret i32* %1
}
declare dso_local noalias noundef i8* @malloc(i64 noundef)
)IR");

  Function *Func = M->getFunction("test");

  // First call has all cold contexts.
  CallStackTrie Trie1;
  Trie1.addCallStack(AllocationType::Cold, {1, 2});
  Trie1.addCallStack(AllocationType::Cold, {1, 3, 4});
  CallBase *Call1 = findCall(*Func, "call1");
  Trie1.buildAndAttachMIBMetadata(Call1);

  EXPECT_FALSE(Call1->hasMetadata(LLVMContext::MD_memprof));
  EXPECT_TRUE(Call1->hasFnAttr("memprof"));
  EXPECT_EQ(Call1->getFnAttr("memprof").getValueAsString(), "cold");

  // Second call has all non-cold contexts.
  CallStackTrie Trie2;
  Trie2.addCallStack(AllocationType::NotCold, {5, 6});
  Trie2.addCallStack(AllocationType::NotCold, {5, 7, 8});
  CallBase *Call2 = findCall(*Func, "call2");
  Trie2.buildAndAttachMIBMetadata(Call2);

  EXPECT_FALSE(Call2->hasMetadata(LLVMContext::MD_memprof));
  EXPECT_TRUE(Call2->hasFnAttr("memprof"));
  EXPECT_EQ(Call2->getFnAttr("memprof").getValueAsString(), "notcold");
}

// Test CallStackTrie::addCallStack interface taking allocation type and list of
// call stack ids.
// Test that an allocation call reached by both cold and non cold call stacks
// gets memprof metadata representing the different allocation type contexts.
TEST_F(MemoryProfileInfoTest, ColdAndNotColdMIB) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
define i32* @test() {
entry:
  %call = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40)
  %0 = bitcast i8* %call to i32*
  ret i32* %0
}
declare dso_local noalias noundef i8* @malloc(i64 noundef)
)IR");

  Function *Func = M->getFunction("test");

  CallStackTrie Trie;
  Trie.addCallStack(AllocationType::Cold, {1, 2});
  Trie.addCallStack(AllocationType::NotCold, {1, 3});

  CallBase *Call = findCall(*Func, "call");
  Trie.buildAndAttachMIBMetadata(Call);

  EXPECT_FALSE(Call->hasFnAttr("memprof"));
  EXPECT_TRUE(Call->hasMetadata(LLVMContext::MD_memprof));
  MDNode *MemProfMD = Call->getMetadata(LLVMContext::MD_memprof);
  ASSERT_EQ(MemProfMD->getNumOperands(), 2u);
  for (auto &MIBOp : MemProfMD->operands()) {
    MDNode *MIB = dyn_cast<MDNode>(MIBOp);
    MDNode *StackMD = getMIBStackNode(MIB);
    ASSERT_NE(StackMD, nullptr);
    ASSERT_EQ(StackMD->getNumOperands(), 2u);
    auto *StackId = mdconst::dyn_extract<ConstantInt>(StackMD->getOperand(0));
    ASSERT_EQ(StackId->getZExtValue(), 1u);
    StackId = mdconst::dyn_extract<ConstantInt>(StackMD->getOperand(1));
    if (StackId->getZExtValue() == 2u)
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::Cold);
    else {
      ASSERT_EQ(StackId->getZExtValue(), 3u);
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::NotCold);
    }
  }
}

// Test CallStackTrie::addCallStack interface taking allocation type and list of
// call stack ids.
// Test that an allocation call reached by multiple call stacks has memprof
// metadata with the contexts trimmed to the minimum context required to
// identify the allocation type.
TEST_F(MemoryProfileInfoTest, TrimmedMIBContext) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
define i32* @test() {
entry:
  %call = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40)
  %0 = bitcast i8* %call to i32*
  ret i32* %0
}
declare dso_local noalias noundef i8* @malloc(i64 noundef)
)IR");

  Function *Func = M->getFunction("test");

  CallStackTrie Trie;
  // We should be able to trim the following two and combine into a single MIB
  // with the cold context {1, 2}.
  Trie.addCallStack(AllocationType::Cold, {1, 2, 3});
  Trie.addCallStack(AllocationType::Cold, {1, 2, 4});
  // We should be able to trim the following two and combine into a single MIB
  // with the non-cold context {1, 5}.
  Trie.addCallStack(AllocationType::NotCold, {1, 5, 6});
  Trie.addCallStack(AllocationType::NotCold, {1, 5, 7});

  CallBase *Call = findCall(*Func, "call");
  Trie.buildAndAttachMIBMetadata(Call);

  EXPECT_FALSE(Call->hasFnAttr("memprof"));
  EXPECT_TRUE(Call->hasMetadata(LLVMContext::MD_memprof));
  MDNode *MemProfMD = Call->getMetadata(LLVMContext::MD_memprof);
  ASSERT_EQ(MemProfMD->getNumOperands(), 2u);
  for (auto &MIBOp : MemProfMD->operands()) {
    MDNode *MIB = dyn_cast<MDNode>(MIBOp);
    MDNode *StackMD = getMIBStackNode(MIB);
    ASSERT_NE(StackMD, nullptr);
    ASSERT_EQ(StackMD->getNumOperands(), 2u);
    auto *StackId = mdconst::dyn_extract<ConstantInt>(StackMD->getOperand(0));
    EXPECT_EQ(StackId->getZExtValue(), 1u);
    StackId = mdconst::dyn_extract<ConstantInt>(StackMD->getOperand(1));
    if (StackId->getZExtValue() == 2u)
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::Cold);
    else {
      ASSERT_EQ(StackId->getZExtValue(), 5u);
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::NotCold);
    }
  }
}

// Test CallStackTrie::addCallStack interface taking memprof MIB metadata.
// Check that allocations annotated with memprof metadata with a single
// allocation type get simplified to an attribute.
TEST_F(MemoryProfileInfoTest, SimplifyMIBToAttribute) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
define i32* @test() {
entry:
  %call1 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40), !memprof !0
  %0 = bitcast i8* %call1 to i32*
  %call2 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40), !memprof !3
  %1 = bitcast i8* %call2 to i32*
  ret i32* %1
}
declare dso_local noalias noundef i8* @malloc(i64 noundef)
!0 = !{!1}
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 3}
!3 = !{!4}
!4 = !{!5, !"notcold"}
!5 = !{i64 4, i64 5, i64 6, i64 7}
)IR");

  Function *Func = M->getFunction("test");

  // First call has all cold contexts.
  CallStackTrie Trie1;
  CallBase *Call1 = findCall(*Func, "call1");
  MDNode *MemProfMD1 = Call1->getMetadata(LLVMContext::MD_memprof);
  ASSERT_EQ(MemProfMD1->getNumOperands(), 1u);
  MDNode *MIB1 = dyn_cast<MDNode>(MemProfMD1->getOperand(0));
  Trie1.addCallStack(MIB1);
  Trie1.buildAndAttachMIBMetadata(Call1);

  EXPECT_TRUE(Call1->hasFnAttr("memprof"));
  EXPECT_EQ(Call1->getFnAttr("memprof").getValueAsString(), "cold");

  // Second call has all non-cold contexts.
  CallStackTrie Trie2;
  CallBase *Call2 = findCall(*Func, "call2");
  MDNode *MemProfMD2 = Call2->getMetadata(LLVMContext::MD_memprof);
  ASSERT_EQ(MemProfMD2->getNumOperands(), 1u);
  MDNode *MIB2 = dyn_cast<MDNode>(MemProfMD2->getOperand(0));
  Trie2.addCallStack(MIB2);
  Trie2.buildAndAttachMIBMetadata(Call2);

  EXPECT_TRUE(Call2->hasFnAttr("memprof"));
  EXPECT_EQ(Call2->getFnAttr("memprof").getValueAsString(), "notcold");
}

// Test CallStackTrie::addCallStack interface taking memprof MIB metadata.
// Test that allocations annotated with memprof metadata with multiple call
// stacks gets new memprof metadata with the contexts trimmed to the minimum
// context required to identify the allocation type.
TEST_F(MemoryProfileInfoTest, ReTrimMIBContext) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
define i32* @test() {
entry:
  %call = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40), !memprof !0
  %0 = bitcast i8* %call to i32*
  ret i32* %0
}
declare dso_local noalias noundef i8* @malloc(i64 noundef)
!0 = !{!1, !3, !5, !7}
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 3}
!3 = !{!4, !"cold"}
!4 = !{i64 1, i64 2, i64 4}
!5 = !{!6, !"notcold"}
!6 = !{i64 1, i64 5, i64 6}
!7 = !{!8, !"notcold"}
!8 = !{i64 1, i64 5, i64 7}
)IR");

  Function *Func = M->getFunction("test");

  CallStackTrie Trie;
  ASSERT_TRUE(Trie.empty());
  CallBase *Call = findCall(*Func, "call");
  MDNode *MemProfMD = Call->getMetadata(LLVMContext::MD_memprof);
  for (auto &MIBOp : MemProfMD->operands()) {
    MDNode *MIB = dyn_cast<MDNode>(MIBOp);
    Trie.addCallStack(MIB);
  }
  ASSERT_FALSE(Trie.empty());
  Trie.buildAndAttachMIBMetadata(Call);

  // We should be able to trim the first two and combine into a single MIB
  // with the cold context {1, 2}.
  // We should be able to trim the second two and combine into a single MIB
  // with the non-cold context {1, 5}.

  EXPECT_FALSE(Call->hasFnAttr("memprof"));
  EXPECT_TRUE(Call->hasMetadata(LLVMContext::MD_memprof));
  MemProfMD = Call->getMetadata(LLVMContext::MD_memprof);
  ASSERT_EQ(MemProfMD->getNumOperands(), 2u);
  for (auto &MIBOp : MemProfMD->operands()) {
    MDNode *MIB = dyn_cast<MDNode>(MIBOp);
    MDNode *StackMD = getMIBStackNode(MIB);
    ASSERT_NE(StackMD, nullptr);
    ASSERT_EQ(StackMD->getNumOperands(), 2u);
    auto *StackId = mdconst::dyn_extract<ConstantInt>(StackMD->getOperand(0));
    EXPECT_EQ(StackId->getZExtValue(), 1u);
    StackId = mdconst::dyn_extract<ConstantInt>(StackMD->getOperand(1));
    if (StackId->getZExtValue() == 2u)
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::Cold);
    else {
      ASSERT_EQ(StackId->getZExtValue(), 5u);
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::NotCold);
    }
  }
}

} // end anonymous namespace
