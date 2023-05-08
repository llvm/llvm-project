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
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <cstring>
#include <sys/types.h>

using namespace llvm;
using namespace llvm::memprof;

extern cl::opt<float> MemProfLifetimeAccessDensityColdThreshold;
extern cl::opt<unsigned> MemProfAveLifetimeColdThreshold;
extern cl::opt<unsigned> MemProfMinAveLifetimeAccessDensityHotThreshold;

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

// Test getAllocType helper.
// Basic checks on the allocation type for values just above and below
// the thresholds.
TEST_F(MemoryProfileInfoTest, GetAllocType) {
  const uint64_t AllocCount = 2;
  // To be cold we require that
  // ((float)TotalLifetimeAccessDensity) / AllocCount / 100 <
  //    MemProfLifetimeAccessDensityColdThreshold
  // so compute the ColdTotalLifetimeAccessDensityThreshold at the threshold.
  const uint64_t ColdTotalLifetimeAccessDensityThreshold =
      (uint64_t)(MemProfLifetimeAccessDensityColdThreshold * AllocCount * 100);
  // To be cold we require that
  // ((float)TotalLifetime) / AllocCount >=
  //    MemProfAveLifetimeColdThreshold * 1000
  // so compute the TotalLifetime right at the threshold.
  const uint64_t ColdTotalLifetimeThreshold =
      MemProfAveLifetimeColdThreshold * AllocCount * 1000;
  // To be hot we require that
  // ((float)TotalLifetimeAccessDensity) / AllocCount / 100 >
  //    MemProfMinAveLifetimeAccessDensityHotThreshold
  // so compute the HotTotalLifetimeAccessDensityThreshold  at the threshold.
  const uint64_t HotTotalLifetimeAccessDensityThreshold =
      (uint64_t)(MemProfMinAveLifetimeAccessDensityHotThreshold * AllocCount * 100);  
   
  
  // Test Hot
  // More accesses per byte per sec than hot threshold is hot.
  EXPECT_EQ(getAllocType(HotTotalLifetimeAccessDensityThreshold + 1, AllocCount,
                         ColdTotalLifetimeThreshold + 1),
            AllocationType::Hot);  

  // Test Cold
  // Long lived with less accesses per byte per sec than cold threshold is cold.
  EXPECT_EQ(getAllocType(ColdTotalLifetimeAccessDensityThreshold - 1, AllocCount,
                         ColdTotalLifetimeThreshold + 1),
            AllocationType::Cold);
  
  // Test NotCold
  // Long lived with more accesses per byte per sec than cold threshold is not cold.
  EXPECT_EQ(getAllocType(ColdTotalLifetimeAccessDensityThreshold + 1, AllocCount,
                         ColdTotalLifetimeThreshold + 1),
            AllocationType::NotCold);  
  // Short lived with more accesses per byte per sec than cold threshold is not cold.
  EXPECT_EQ(getAllocType(ColdTotalLifetimeAccessDensityThreshold + 1, AllocCount,
                         ColdTotalLifetimeThreshold - 1),
            AllocationType::NotCold);
  // Short lived with less accesses per byte per sec than cold threshold is not cold.
  EXPECT_EQ(getAllocType(ColdTotalLifetimeAccessDensityThreshold - 1, AllocCount,
                         ColdTotalLifetimeThreshold - 1),
            AllocationType::NotCold);
}

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
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::Hot);
    }
  }
}

// Test CallStackTrie::addCallStack interface taking allocation type and list of
// call stack ids.
// Test that an allocation call reached by both non cold and hot call stacks
// gets memprof metadata representing the different allocation type contexts.
TEST_F(MemoryProfileInfoTest, NotColdAndHotMIB) {
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
  Trie.addCallStack(AllocationType::NotCold, {1, 2});
  Trie.addCallStack(AllocationType::Hot, {1, 3});

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
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::NotCold);
    else {
      ASSERT_EQ(StackId->getZExtValue(), 3u);
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::Hot);
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
  Trie.addCallStack(AllocationType::Hot, {1, 4});

  CallBase *Call = findCall(*Func, "call");
  Trie.buildAndAttachMIBMetadata(Call);

  EXPECT_FALSE(Call->hasFnAttr("memprof"));
  EXPECT_TRUE(Call->hasMetadata(LLVMContext::MD_memprof));
  MDNode *MemProfMD = Call->getMetadata(LLVMContext::MD_memprof);
  ASSERT_EQ(MemProfMD->getNumOperands(), 3u);
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
    } else {
      ASSERT_EQ(StackId->getZExtValue(), 4u);
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::Hot);
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
  // We should be able to trim the following two and combine into a single MIB
  // with the hot context {1, 8}.
  Trie.addCallStack(AllocationType::Hot, {1, 8, 9});
  Trie.addCallStack(AllocationType::Hot, {1, 8, 10});

  CallBase *Call = findCall(*Func, "call");
  Trie.buildAndAttachMIBMetadata(Call);

  EXPECT_FALSE(Call->hasFnAttr("memprof"));
  EXPECT_TRUE(Call->hasMetadata(LLVMContext::MD_memprof));
  MDNode *MemProfMD = Call->getMetadata(LLVMContext::MD_memprof);
  ASSERT_EQ(MemProfMD->getNumOperands(), 3u);
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
    else {
      ASSERT_EQ(StackId->getZExtValue(), 8u);
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::Hot);
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

  EXPECT_FALSE(Call->hasFnAttr("memprof"));
  EXPECT_TRUE(Call->hasMetadata(LLVMContext::MD_memprof));
  MemProfMD = Call->getMetadata(LLVMContext::MD_memprof);
  ASSERT_EQ(MemProfMD->getNumOperands(), 3u);
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
    else {
      ASSERT_EQ(StackId->getZExtValue(), 8u);
      EXPECT_EQ(getMIBAllocType(MIB), AllocationType::Hot);
    }
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
