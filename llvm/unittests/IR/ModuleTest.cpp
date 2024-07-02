//===- unittests/IR/ModuleTest.cpp - Module unit tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Module.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
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
  }
}

TEST(ModuleTest, randomNumberGenerator) {
  LLVMContext Context;
  static char ID;
  struct DummyPass : ModulePass {
    DummyPass() : ModulePass(ID) {}
    bool runOnModule(Module &) override { return true; }
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

TEST(ModuleTest, setModuleFlagInt) {
  LLVMContext Context;
  Module M("M", Context);
  StringRef Key = "Key";
  uint32_t Val1 = 1;
  uint32_t Val2 = 2;
  EXPECT_EQ(nullptr, M.getModuleFlag(Key));
  M.setModuleFlag(Module::ModFlagBehavior::Error, Key, Val1);
  auto A1 = mdconst::extract_or_null<ConstantInt>(M.getModuleFlag(Key));
  EXPECT_EQ(Val1, A1->getZExtValue());
  M.setModuleFlag(Module::ModFlagBehavior::Error, Key, Val2);
  auto A2 = mdconst::extract_or_null<ConstantInt>(M.getModuleFlag(Key));
  EXPECT_EQ(Val2, A2->getZExtValue());
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

TEST(ModuleTest, AliasList) {
  // This tests all Module's functions that interact with Module::AliasList.
  LLVMContext C;
  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssemblyString(R"(
declare void @Foo()
@GA = alias void (), ptr @Foo
)",
                                                  Err, Context);
  Function *Foo = M->getFunction("Foo");
  auto *GA = M->getNamedAlias("GA");
  EXPECT_EQ(M->alias_size(), 1u);
  auto *NewGA =
      GlobalAlias::create(Foo->getType(), 0, GlobalValue::ExternalLinkage,
                          "NewGA", Foo, /*Parent=*/nullptr);
  EXPECT_EQ(M->alias_size(), 1u);

  M->insertAlias(NewGA);
  EXPECT_EQ(&*std::prev(M->aliases().end()), NewGA);

  M->removeAlias(NewGA);
  EXPECT_EQ(M->alias_size(), 1u);
  M->insertAlias(NewGA);
  EXPECT_EQ(M->alias_size(), 2u);
  EXPECT_EQ(&*std::prev(M->aliases().end()), NewGA);

  auto Range = M->aliases();
  EXPECT_EQ(&*Range.begin(), GA);
  EXPECT_EQ(&*std::next(Range.begin()), NewGA);
  EXPECT_EQ(std::next(Range.begin(), 2), Range.end());

  M->removeAlias(NewGA);
  EXPECT_EQ(M->alias_size(), 1u);

  M->insertAlias(NewGA);
  M->eraseAlias(NewGA);
  EXPECT_EQ(M->alias_size(), 1u);
}

TEST(ModuleTest, IFuncList) {
  // This tests all Module's functions that interact with Module::IFuncList.
  LLVMContext C;
  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssemblyString(R"(
declare void @Foo()
@GIF = ifunc void (), ptr @Foo
)",
                                                  Err, Context);
  Function *Foo = M->getFunction("Foo");
  auto *GIF = M->getNamedIFunc("GIF");
  EXPECT_EQ(M->ifunc_size(), 1u);
  auto *NewGIF =
      GlobalIFunc::create(Foo->getType(), 0, GlobalValue::ExternalLinkage,
                          "NewGIF", Foo, /*Parent=*/nullptr);
  EXPECT_EQ(M->ifunc_size(), 1u);

  M->insertIFunc(NewGIF);
  EXPECT_EQ(&*std::prev(M->ifuncs().end()), NewGIF);

  M->removeIFunc(NewGIF);
  EXPECT_EQ(M->ifunc_size(), 1u);
  M->insertIFunc(NewGIF);
  EXPECT_EQ(M->ifunc_size(), 2u);
  EXPECT_EQ(&*std::prev(M->ifuncs().end()), NewGIF);

  auto Range = M->ifuncs();
  EXPECT_EQ(&*Range.begin(), GIF);
  EXPECT_EQ(&*std::next(Range.begin()), NewGIF);
  EXPECT_EQ(std::next(Range.begin(), 2), Range.end());

  M->removeIFunc(NewGIF);
  EXPECT_EQ(M->ifunc_size(), 1u);

  M->insertIFunc(NewGIF);
  M->eraseIFunc(NewGIF);
  EXPECT_EQ(M->ifunc_size(), 1u);
}

TEST(ModuleTest, NamedMDList) {
  // This tests all Module's functions that interact with Module::NamedMDList.
  LLVMContext C;
  SMDiagnostic Err;
  LLVMContext Context;
  auto M = std::make_unique<Module>("M", C);
  NamedMDNode *MDN1 = M->getOrInsertNamedMetadata("MDN1");
  EXPECT_EQ(M->named_metadata_size(), 1u);
  NamedMDNode *MDN2 = M->getOrInsertNamedMetadata("MDN2");
  EXPECT_EQ(M->named_metadata_size(), 2u);
  auto *NewMDN = M->getOrInsertNamedMetadata("NewMDN");
  EXPECT_EQ(M->named_metadata_size(), 3u);

  M->removeNamedMDNode(NewMDN);
  EXPECT_EQ(M->named_metadata_size(), 2u);

  M->insertNamedMDNode(NewMDN);
  EXPECT_EQ(&*std::prev(M->named_metadata().end()), NewMDN);

  M->removeNamedMDNode(NewMDN);
  M->insertNamedMDNode(NewMDN);
  EXPECT_EQ(M->named_metadata_size(), 3u);
  EXPECT_EQ(&*std::prev(M->named_metadata().end()), NewMDN);

  auto Range = M->named_metadata();
  EXPECT_EQ(&*Range.begin(), MDN1);
  EXPECT_EQ(&*std::next(Range.begin(), 1), MDN2);
  EXPECT_EQ(&*std::next(Range.begin(), 2), NewMDN);
  EXPECT_EQ(std::next(Range.begin(), 3), Range.end());

  M->eraseNamedMDNode(NewMDN);
  EXPECT_EQ(M->named_metadata_size(), 2u);
}

TEST(ModuleTest, GlobalList) {
  // This tests all Module's functions that interact with Module::GlobalList.
  LLVMContext C;
  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssemblyString(R"(
@GV = external global i32
)",
                                                  Err, Context);
  auto *GV = cast<GlobalVariable>(M->getNamedValue("GV"));
  EXPECT_EQ(M->global_size(), 1u);
  GlobalVariable *NewGV = new GlobalVariable(
      Type::getInt32Ty(C), /*isConstant=*/true, GlobalValue::InternalLinkage,
      /*Initializer=*/nullptr, "NewGV");
  EXPECT_EQ(M->global_size(), 1u);
  // Insert before
  M->insertGlobalVariable(M->globals().begin(), NewGV);
  EXPECT_EQ(M->global_size(), 2u);
  EXPECT_EQ(&*M->globals().begin(), NewGV);
  // Insert at end()
  M->removeGlobalVariable(NewGV);
  EXPECT_EQ(M->global_size(), 1u);
  M->insertGlobalVariable(NewGV);
  EXPECT_EQ(M->global_size(), 2u);
  EXPECT_EQ(&*std::prev(M->globals().end()), NewGV);
  // Check globals()
  auto Range = M->globals();
  EXPECT_EQ(&*Range.begin(), GV);
  EXPECT_EQ(&*std::next(Range.begin()), NewGV);
  EXPECT_EQ(std::next(Range.begin(), 2), Range.end());
  // Check remove
  M->removeGlobalVariable(NewGV);
  EXPECT_EQ(M->global_size(), 1u);
  // Check erase
  M->insertGlobalVariable(NewGV);
  M->eraseGlobalVariable(NewGV);
  EXPECT_EQ(M->global_size(), 1u);
}

} // end namespace
