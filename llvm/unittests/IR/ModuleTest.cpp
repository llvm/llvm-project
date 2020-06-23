//===- unittests/IR/ModuleTest.cpp - Module unit tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Module.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Pass.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

#include <random>

using namespace llvm;

namespace {

bool sortByName(const GlobalVariable &L, const GlobalVariable &R) {
  return L.getName() < R.getName();
}

bool sortByNameReverse(const GlobalVariable &L, const GlobalVariable &R) {
  return sortByName(R, L);
}

TEST(ModuleTest, sortGlobalsByName) {
  LLVMContext Context;
  for (auto compare : {&sortByName, &sortByNameReverse}) {
    Module M("M", Context);
    Type *T = Type::getInt8Ty(Context);
    GlobalValue::LinkageTypes L = GlobalValue::ExternalLinkage;
    (void)new GlobalVariable(M, T, false, L, nullptr, "A");
    (void)new GlobalVariable(M, T, false, L, nullptr, "F");
    (void)new GlobalVariable(M, T, false, L, nullptr, "G");
    (void)new GlobalVariable(M, T, false, L, nullptr, "E");
    (void)new GlobalVariable(M, T, false, L, nullptr, "B");
    (void)new GlobalVariable(M, T, false, L, nullptr, "H");
    (void)new GlobalVariable(M, T, false, L, nullptr, "C");
    (void)new GlobalVariable(M, T, false, L, nullptr, "D");

    // Sort the globals by name.
    EXPECT_FALSE(std::is_sorted(M.global_begin(), M.global_end(), compare));
    M.getGlobalList().sort(compare);
    EXPECT_TRUE(std::is_sorted(M.global_begin(), M.global_end(), compare));
  }
}

TEST(ModuleTest, randomNumberGenerator) {
  LLVMContext Context;
  static char ID;
  struct DummyPass : ModulePass {
    DummyPass() : ModulePass(ID) {}
    bool runOnModule(Module &) { return true; }
  } DP;

  Module M("R", Context);

  std::uniform_int_distribution<int> dist;
  const size_t NBCheck = 10;

  std::array<int, NBCheck> RandomStreams[2];
  for (auto &RandomStream : RandomStreams) {
    std::unique_ptr<RandomNumberGenerator> RNG = M.createRNG(DP.getPassName());
    std::generate(RandomStream.begin(), RandomStream.end(),
                  [&]() { return dist(*RNG); });
  }

  EXPECT_TRUE(std::equal(RandomStreams[0].begin(), RandomStreams[0].end(),
                         RandomStreams[1].begin()));
}

TEST(ModuleTest, setModuleFlag) {
  LLVMContext Context;
  Module M("M", Context);
  StringRef Key = "Key";
  Metadata *Val1 = MDString::get(Context, "Val1");
  Metadata *Val2 = MDString::get(Context, "Val2");
  EXPECT_EQ(nullptr, M.getModuleFlag(Key));
  M.setModuleFlag(Module::ModFlagBehavior::Error, Key, Val1);
  EXPECT_EQ(Val1, M.getModuleFlag(Key));
  M.setModuleFlag(Module::ModFlagBehavior::Error, Key, Val2);
  EXPECT_EQ(Val2, M.getModuleFlag(Key));
}

const char *IRString = R"IR(
  !llvm.module.flags = !{!0}

  !0 = !{i32 1, !"ProfileSummary", !1}
  !1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
  !2 = !{!"ProfileFormat", !"SampleProfile"}
  !3 = !{!"TotalCount", i64 10000}
  !4 = !{!"MaxCount", i64 10}
  !5 = !{!"MaxInternalCount", i64 1}
  !6 = !{!"MaxFunctionCount", i64 1000}
  !7 = !{!"NumCounts", i64 200}
  !8 = !{!"NumFunctions", i64 3}
  !9 = !{!"DetailedSummary", !10}
  !10 = !{!11, !12, !13}
  !11 = !{i32 10000, i64 1000, i32 1}
  !12 = !{i32 990000, i64 300, i32 10}
  !13 = !{i32 999999, i64 5, i32 100}
)IR";

TEST(ModuleTest, setProfileSummary) {
  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssemblyString(IRString, Err, Context);
  auto *PS = ProfileSummary::getFromMD(M->getProfileSummary(/*IsCS*/ false));
  EXPECT_NE(nullptr, PS);
  EXPECT_FALSE(PS->isPartialProfile());
  PS->setPartialProfile(true);
  M->setProfileSummary(PS->getMD(Context), ProfileSummary::PSK_Sample);
  delete PS;
  PS = ProfileSummary::getFromMD(M->getProfileSummary(/*IsCS*/ false));
  EXPECT_NE(nullptr, PS);
  EXPECT_EQ(true, PS->isPartialProfile());
  delete PS;
}

TEST(ModuleTest, setPartialSampleProfileRatio) {
  const char *IRString = R"IR(
  !llvm.module.flags = !{!0}

  !0 = !{i32 1, !"ProfileSummary", !1}
  !1 = !{!2, !3, !4, !5, !6, !7, !8, !9, !10, !11}
  !2 = !{!"ProfileFormat", !"SampleProfile"}
  !3 = !{!"TotalCount", i64 10000}
  !4 = !{!"MaxCount", i64 10}
  !5 = !{!"MaxInternalCount", i64 1}
  !6 = !{!"MaxFunctionCount", i64 1000}
  !7 = !{!"NumCounts", i64 200}
  !8 = !{!"NumFunctions", i64 3}
  !9 = !{!"IsPartialProfile", i64 1}
  !10 = !{!"PartialProfileRatio", double 0.0}
  !11 = !{!"DetailedSummary", !12}
  !12 = !{!13, !14, !15}
  !13 = !{i32 10000, i64 1000, i32 1}
  !14 = !{i32 990000, i64 300, i32 10}
  !15 = !{i32 999999, i64 5, i32 100}
  )IR";

  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssemblyString(IRString, Err, Context);
  ModuleSummaryIndex Index(/*HaveGVs*/ false);
  const unsigned BlockCount = 100;
  const unsigned NumCounts = 200;
  Index.setBlockCount(BlockCount);
  M->setPartialSampleProfileRatio(Index);
  double Ratio = (double)BlockCount / NumCounts;
  std::unique_ptr<ProfileSummary> ProfileSummary(
      ProfileSummary::getFromMD(M->getProfileSummary(/*IsCS*/ false)));
  EXPECT_EQ(Ratio, ProfileSummary->getPartialProfileRatio());
}

} // end namespace
