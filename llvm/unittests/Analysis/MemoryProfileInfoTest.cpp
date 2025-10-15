//===- MemoryProfileInfoTest.cpp - Memory Profile Info Unit Tests-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemoryProfileInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstring>
#include <sys/types.h>

using namespace llvm;
using namespace llvm::memprof;

namespace llvm {
LLVM_ABI extern cl::opt<bool> MemProfKeepAllNotColdContexts;
} // end namespace llvm

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

  std::unique_ptr<ModuleSummaryIndex> makeLLVMIndex(const char *Summary) {
    SMDiagnostic Err;
    std::unique_ptr<ModuleSummaryIndex> Index =
        parseSummaryIndexAssemblyString(Summary, Err);
    if (!Index)
      Err.print("MemoryProfileInfoTest", errs());
    return Index;
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

// Test the hasSingleAllocType helper.
TEST_F(MemoryProfileInfoTest, SingleAllocType) {
  uint8_t NotCold = (uint8_t)AllocationType::NotCold;
  uint8_t Cold = (uint8_t)AllocationType::Cold;
  uint8_t Hot = (uint8_t)AllocationType::Hot;
  EXPECT_TRUE(hasSingleAllocType(NotCold));
  EXPECT_TRUE(hasSingleAllocType(Cold));
  EXPECT_TRUE(hasSingleAllocType(Hot));
  EXPECT_FALSE(hasSingleAllocType(NotCold | Cold));
  EXPECT_FALSE(hasSingleAllocType(NotCold | Hot));
  EXPECT_FALSE(hasSingleAllocType(Cold | Hot));
  EXPECT_FALSE(hasSingleAllocType(NotCold | Cold | Hot));
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
  %call3 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40)
  %2 = bitcast i8* %call3 to i32*  
  %call4 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40)
  %3 = bitcast i8* %call4 to i32*
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

  // Third call has all hot contexts.
  CallStackTrie Trie3;
  Trie3.addCallStack(AllocationType::Hot, {9, 10});
  Trie3.addCallStack(AllocationType::Hot, {9, 11, 12});
  CallBase *Call3 = findCall(*Func, "call3");
  Trie3.buildAndAttachMIBMetadata(Call3);

  EXPECT_FALSE(Call3->hasMetadata(LLVMContext::MD_memprof));
  EXPECT_TRUE(Call3->hasFnAttr("memprof"));
  EXPECT_EQ(Call3->getFnAttr("memprof").getValueAsString(), "hot");

  // Fourth call has hot and non-cold contexts. These should be treated as
  // notcold and given a notcold attribute.
  CallStackTrie Trie4;
  Trie4.addCallStack(AllocationType::Hot, {5, 6});
  Trie4.addCallStack(AllocationType::NotCold, {5, 7, 8});
  CallBase *Call4 = findCall(*Func, "call4");
  Trie4.buildAndAttachMIBMetadata(Call4);

  EXPECT_FALSE(Call4->hasMetadata(LLVMContext::MD_memprof));
  EXPECT_TRUE(Call4->hasFnAttr("memprof"));
  EXPECT_EQ(Call4->getFnAttr("memprof").getValueAsString(), "notcold");
}

// TODO: Use this matcher in existing tests.
// ExpectedVals should be a vector of expected MIBs and their allocation type
// and stack id contents in order, of type:
//  std::vector<std::pair<AllocationType, std::vector<unsigned>>>
MATCHER_P(MemprofMetadataEquals, ExpectedVals, "Matching !memprof contents") {
  auto PrintAndFail = [&]() {
    std::string Buffer;
    llvm::raw_string_ostream OS(Buffer);
    OS << "Expected:\n";
    for (auto &[ExpectedAllocType, ExpectedStackIds] : ExpectedVals) {
      OS << "\t" << getAllocTypeAttributeString(ExpectedAllocType) << " { ";
      for (auto Id : ExpectedStackIds)
        OS << Id << " ";
      OS << "}\n";
    }
    OS << "Got:\n";
    arg->printTree(OS);
    *result_listener << "!memprof metadata differs!\n" << Buffer;
    return false;
  };

  if (ExpectedVals.size() != arg->getNumOperands())
    return PrintAndFail();

  for (size_t I = 0; I < ExpectedVals.size(); I++) {
    const auto &[ExpectedAllocType, ExpectedStackIds] = ExpectedVals[I];
    MDNode *MIB = dyn_cast<MDNode>(arg->getOperand(I));
    if (getMIBAllocType(MIB) != ExpectedAllocType)
      return PrintAndFail();
    MDNode *StackMD = getMIBStackNode(MIB);
    EXPECT_NE(StackMD, nullptr);
    if (StackMD->getNumOperands() != ExpectedStackIds.size())
      return PrintAndFail();
    for (size_t J = 0; J < ExpectedStackIds.size(); J++) {
      auto *StackId = mdconst::dyn_extract<ConstantInt>(StackMD->getOperand(J));
      if (StackId->getZExtValue() != ExpectedStackIds[J])
        return PrintAndFail();
    }
  }
  return true;
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

  EXPECT_TRUE(Call->hasFnAttr("memprof"));
  EXPECT_EQ(Call->getFnAttr("memprof").getValueAsString(), "ambiguous");
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
// Test that an allocation call reached by both cold and hot call stacks
// gets memprof metadata representing the different allocation type contexts.
TEST_F(MemoryProfileInfoTest, ColdAndHotMIB) {
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
  Trie.addCallStack(AllocationType::Hot, {1, 3});

  CallBase *Call = findCall(*Func, "call");
  Trie.buildAndAttachMIBMetadata(Call);

  EXPECT_TRUE(Call->hasFnAttr("memprof"));
  EXPECT_EQ(Call->getFnAttr("memprof").getValueAsString(), "ambiguous");
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
      // Hot contexts are converted to NotCold when building the metadata.
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::NotCold);
    }
  }
}

// Test CallStackTrie::addCallStack interface taking allocation type and list of
// call stack ids.
// Test that an allocation call reached by both cold, non cold and hot call
// stacks gets memprof metadata representing the different allocation type
// contexts.
TEST_F(MemoryProfileInfoTest, ColdAndNotColdAndHotMIB) {
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
  // This will be pruned as it is unnecessary to determine how to clone the
  // cold allocation.
  Trie.addCallStack(AllocationType::Hot, {1, 4});

  CallBase *Call = findCall(*Func, "call");
  Trie.buildAndAttachMIBMetadata(Call);

  EXPECT_TRUE(Call->hasFnAttr("memprof"));
  EXPECT_EQ(Call->getFnAttr("memprof").getValueAsString(), "ambiguous");
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
    if (StackId->getZExtValue() == 2u) {
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::Cold);
    } else if (StackId->getZExtValue() == 3u) {
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
  // These will be pruned as they are unnecessary to determine how to clone the
  // cold allocation.
  Trie.addCallStack(AllocationType::Hot, {1, 8, 9});
  Trie.addCallStack(AllocationType::Hot, {1, 8, 10});

  CallBase *Call = findCall(*Func, "call");
  Trie.buildAndAttachMIBMetadata(Call);

  EXPECT_TRUE(Call->hasFnAttr("memprof"));
  EXPECT_EQ(Call->getFnAttr("memprof").getValueAsString(), "ambiguous");
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
    else if (StackId->getZExtValue() == 5u)
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::NotCold);
  }
}

// Test to ensure that we prune NotCold contexts that are unneeded for
// determining where Cold contexts need to be cloned to enable correct hinting.
TEST_F(MemoryProfileInfoTest, PruneUnneededNotColdContexts) {
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

  Trie.addCallStack(AllocationType::Cold, {1, 2, 3, 4});
  Trie.addCallStack(AllocationType::Cold, {1, 2, 3, 5, 6, 7});
  // This NotCold context is needed to know where the above two Cold contexts
  // must be cloned from:
  Trie.addCallStack(AllocationType::NotCold, {1, 2, 3, 5, 6, 13});

  Trie.addCallStack(AllocationType::Cold, {1, 2, 3, 8, 9, 10});
  // This NotCold context is needed to know where the above Cold context must be
  // cloned from:
  Trie.addCallStack(AllocationType::NotCold, {1, 2, 3, 8, 9, 14});
  // This NotCold context is not needed since the above is sufficient (we pick
  // the first in sorted order).
  Trie.addCallStack(AllocationType::NotCold, {1, 2, 3, 8, 9, 15});

  // None of these NotCold contexts are needed as the Cold contexts they
  // overlap with are covered by longer overlapping NotCold contexts.
  Trie.addCallStack(AllocationType::NotCold, {1, 2, 3, 12});
  Trie.addCallStack(AllocationType::NotCold, {1, 2, 11});
  Trie.addCallStack(AllocationType::NotCold, {1, 16});

  std::vector<std::pair<AllocationType, std::vector<unsigned>>> ExpectedVals = {
      {AllocationType::Cold, {1, 2, 3, 4}},
      {AllocationType::Cold, {1, 2, 3, 5, 6, 7}},
      {AllocationType::NotCold, {1, 2, 3, 5, 6, 13}},
      {AllocationType::Cold, {1, 2, 3, 8, 9, 10}},
      {AllocationType::NotCold, {1, 2, 3, 8, 9, 14}}};

  CallBase *Call = findCall(*Func, "call");
  ASSERT_NE(Call, nullptr);
  Trie.buildAndAttachMIBMetadata(Call);

  EXPECT_TRUE(Call->hasFnAttr("memprof"));
  EXPECT_EQ(Call->getFnAttr("memprof").getValueAsString(), "ambiguous");
  EXPECT_TRUE(Call->hasMetadata(LLVMContext::MD_memprof));
  MDNode *MemProfMD = Call->getMetadata(LLVMContext::MD_memprof);
  EXPECT_THAT(MemProfMD, MemprofMetadataEquals(ExpectedVals));
}

// Test to ensure that we keep optionally keep unneeded NotCold contexts.
// Same as PruneUnneededNotColdContexts test but with the
// MemProfKeepAllNotColdContexts set to true.
TEST_F(MemoryProfileInfoTest, KeepUnneededNotColdContexts) {
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

  Trie.addCallStack(AllocationType::Cold, {1, 2, 3, 4});
  Trie.addCallStack(AllocationType::Cold, {1, 2, 3, 5, 6, 7});
  // This NotCold context is needed to know where the above two Cold contexts
  // must be cloned from:
  Trie.addCallStack(AllocationType::NotCold, {1, 2, 3, 5, 6, 13});

  Trie.addCallStack(AllocationType::Cold, {1, 2, 3, 8, 9, 10});
  // This NotCold context is needed to know where the above Cold context must be
  // cloned from:
  Trie.addCallStack(AllocationType::NotCold, {1, 2, 3, 8, 9, 14});
  // This NotCold context is not needed since the above is sufficient (we pick
  // the first in sorted order).
  Trie.addCallStack(AllocationType::NotCold, {1, 2, 3, 8, 9, 15});

  // None of these NotCold contexts are needed as the Cold contexts they
  // overlap with are covered by longer overlapping NotCold contexts.
  Trie.addCallStack(AllocationType::NotCold, {1, 2, 3, 12});
  Trie.addCallStack(AllocationType::NotCold, {1, 2, 11});
  Trie.addCallStack(AllocationType::NotCold, {1, 16});

  // We should keep all of the above contexts, even those that are unneeded by
  // default, because we will set the option to keep them.
  std::vector<std::pair<AllocationType, std::vector<unsigned>>> ExpectedVals = {
      {AllocationType::Cold, {1, 2, 3, 4}},
      {AllocationType::Cold, {1, 2, 3, 5, 6, 7}},
      {AllocationType::NotCold, {1, 2, 3, 5, 6, 13}},
      {AllocationType::Cold, {1, 2, 3, 8, 9, 10}},
      {AllocationType::NotCold, {1, 2, 3, 8, 9, 14}},
      {AllocationType::NotCold, {1, 2, 3, 8, 9, 15}},
      {AllocationType::NotCold, {1, 2, 3, 12}},
      {AllocationType::NotCold, {1, 2, 11}},
      {AllocationType::NotCold, {1, 16}}};

  CallBase *Call = findCall(*Func, "call");
  ASSERT_NE(Call, nullptr);

  // Specify that all non-cold contexts should be kept.
  bool OrigMemProfKeepAllNotColdContexts = MemProfKeepAllNotColdContexts;
  MemProfKeepAllNotColdContexts = true;

  Trie.buildAndAttachMIBMetadata(Call);

  // Restore original option value.
  MemProfKeepAllNotColdContexts = OrigMemProfKeepAllNotColdContexts;

  EXPECT_TRUE(Call->hasFnAttr("memprof"));
  EXPECT_EQ(Call->getFnAttr("memprof").getValueAsString(), "ambiguous");
  EXPECT_TRUE(Call->hasMetadata(LLVMContext::MD_memprof));
  MDNode *MemProfMD = Call->getMetadata(LLVMContext::MD_memprof);
  EXPECT_THAT(MemProfMD, MemprofMetadataEquals(ExpectedVals));
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
  %call3 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40), !memprof !6
  %2 = bitcast i8* %call3 to i32*
  ret i32* %1
}
declare dso_local noalias noundef i8* @malloc(i64 noundef)
!0 = !{!1}
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 3}
!3 = !{!4}
!4 = !{!5, !"notcold"}
!5 = !{i64 4, i64 5, i64 6, i64 7}
!6 = !{!7}
!7 = !{!8, !"hot"}
!8 = !{i64 8, i64 9, i64 10}
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

  // Third call has all hot contexts.
  CallStackTrie Trie3;
  CallBase *Call3 = findCall(*Func, "call3");
  MDNode *MemProfMD3 = Call3->getMetadata(LLVMContext::MD_memprof);
  ASSERT_EQ(MemProfMD2->getNumOperands(), 1u);
  MDNode *MIB3 = dyn_cast<MDNode>(MemProfMD3->getOperand(0));
  Trie3.addCallStack(MIB3);
  Trie3.buildAndAttachMIBMetadata(Call3);

  EXPECT_TRUE(Call3->hasFnAttr("memprof"));
  EXPECT_EQ(Call3->getFnAttr("memprof").getValueAsString(), "hot");
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
!0 = !{!1, !3, !5, !7, !9, !11}
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 3}
!3 = !{!4, !"cold"}
!4 = !{i64 1, i64 2, i64 4}
!5 = !{!6, !"notcold"}
!6 = !{i64 1, i64 5, i64 6}
!7 = !{!8, !"notcold"}
!8 = !{i64 1, i64 5, i64 7}
!9 = !{!10, !"hot"}
!10 = !{i64 1, i64 8, i64 9}
!11 = !{!12, !"hot"}
!12 = !{i64 1, i64 8, i64 10}  
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
  // The hot allocations will be converted to NotCold and pruned as they
  // are unnecessary to determine how to clone the cold allocation.

  EXPECT_TRUE(Call->hasFnAttr("memprof"));
  EXPECT_EQ(Call->getFnAttr("memprof").getValueAsString(), "ambiguous");
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
    else if (StackId->getZExtValue() == 5u)
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::NotCold);
  }
}

TEST_F(MemoryProfileInfoTest, CallStackTestIR) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
define ptr @test() {
entry:
  %call = call noalias noundef nonnull dereferenceable(10) ptr @_Znam(i64 noundef 10), !memprof !1, !callsite !6
  ret ptr %call
}
declare noundef nonnull ptr @_Znam(i64 noundef)
!1 = !{!2, !4, !7}
!2 = !{!3, !"notcold"}
!3 = !{i64 1, i64 2, i64 3, i64 4}
!4 = !{!5, !"cold"}
!5 = !{i64 1, i64 2, i64 3, i64 5}
!6 = !{i64 1}
!7 = !{!8, !"hot"}
!8 = !{i64 1, i64 2, i64 3, i64 6}  
)IR");

  Function *Func = M->getFunction("test");
  CallBase *Call = findCall(*Func, "call");

  CallStack<MDNode, MDNode::op_iterator> InstCallsite(
      Call->getMetadata(LLVMContext::MD_callsite));

  MDNode *MemProfMD = Call->getMetadata(LLVMContext::MD_memprof);
  unsigned Idx = 0;
  for (auto &MIBOp : MemProfMD->operands()) {
    auto *MIBMD = cast<const MDNode>(MIBOp);
    MDNode *StackNode = getMIBStackNode(MIBMD);
    CallStack<MDNode, MDNode::op_iterator> StackContext(StackNode);
    EXPECT_EQ(StackContext.back(), 4 + Idx);
    std::vector<uint64_t> StackIds;
    for (auto ContextIter = StackContext.beginAfterSharedPrefix(InstCallsite);
         ContextIter != StackContext.end(); ++ContextIter)
      StackIds.push_back(*ContextIter);
    if (Idx == 0) {
      std::vector<uint64_t> Expected = {2, 3, 4};
      EXPECT_EQ(ArrayRef(StackIds), ArrayRef(Expected));
    } else if (Idx == 1) {
      std::vector<uint64_t> Expected = {2, 3, 5};
      EXPECT_EQ(ArrayRef(StackIds), ArrayRef(Expected));
    } else {
      std::vector<uint64_t> Expected = {2, 3, 6};
      EXPECT_EQ(ArrayRef(StackIds), ArrayRef(Expected));
    }
    Idx++;
  }
}

TEST_F(MemoryProfileInfoTest, CallStackTestSummary) {
  std::unique_ptr<ModuleSummaryIndex> Index = makeLLVMIndex(R"Summary(
^0 = module: (path: "test.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 23, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), allocs: ((versions: (none), memProf: ((type: notcold, stackIds: (1, 2, 3, 4)), (type: cold, stackIds: (1, 2, 3, 5)), (type: hot, stackIds: (1, 2, 3, 6))))))))
^2 = gv: (guid: 25, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 22, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^1)), callsites: ((callee: ^1, clones: (0), stackIds: (3, 4)), (callee: ^1, clones: (0), stackIds: (3, 5)), (callee: ^1, clones: (0), stackIds: (3, 6))))))
)Summary");

  ASSERT_NE(Index, nullptr);
  auto *CallsiteSummary =
      cast<FunctionSummary>(Index->getGlobalValueSummary(/*guid=*/25));
  unsigned Idx = 0;
  for (auto &CI : CallsiteSummary->callsites()) {
    CallStack<CallsiteInfo, SmallVector<unsigned>::const_iterator> InstCallsite(
        &CI);
    std::vector<uint64_t> StackIds;
    for (auto StackIdIndex : InstCallsite)
      StackIds.push_back(Index->getStackIdAtIndex(StackIdIndex));
    if (Idx == 0) {
      std::vector<uint64_t> Expected = {3, 4};
      EXPECT_EQ(ArrayRef(StackIds), ArrayRef(Expected));
    } else if (Idx == 1) {
      std::vector<uint64_t> Expected = {3, 5};
      EXPECT_EQ(ArrayRef(StackIds), ArrayRef(Expected));
    } else {
      std::vector<uint64_t> Expected = {3, 6};
      EXPECT_EQ(ArrayRef(StackIds), ArrayRef(Expected));
    }
    Idx++;
  }

  auto *AllocSummary =
      cast<FunctionSummary>(Index->getGlobalValueSummary(/*guid=*/23));
  for (auto &AI : AllocSummary->allocs()) {
    unsigned Idx = 0;
    for (auto &MIB : AI.MIBs) {
      CallStack<MIBInfo, SmallVector<unsigned>::const_iterator> StackContext(
          &MIB);
      EXPECT_EQ(Index->getStackIdAtIndex(StackContext.back()), 4 + Idx);
      std::vector<uint64_t> StackIds;
      for (auto StackIdIndex : StackContext)
        StackIds.push_back(Index->getStackIdAtIndex(StackIdIndex));
      if (Idx == 0) {
        std::vector<uint64_t> Expected = {1, 2, 3, 4};
        EXPECT_EQ(ArrayRef(StackIds), ArrayRef(Expected));
      } else if (Idx == 1) {
        std::vector<uint64_t> Expected = {1, 2, 3, 5};
        EXPECT_EQ(ArrayRef(StackIds), ArrayRef(Expected));
      } else {
        std::vector<uint64_t> Expected = {1, 2, 3, 6};
        EXPECT_EQ(ArrayRef(StackIds), ArrayRef(Expected));
      }
      Idx++;
    }
  }
}
} // end anonymous namespace
