//===- AssumeBundleQueriesTest.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "gtest/gtest.h"
#include <random>

using namespace llvm;

static void RunTest(
    StringRef Head, StringRef Tail,
    std::vector<std::pair<StringRef, llvm::function_ref<void(Instruction *)>>>
        &Tests) {
  for (auto &Elem : Tests) {
    std::string IR;
    IR.append(Head.begin(), Head.end());
    IR.append(Elem.first.begin(), Elem.first.end());
    IR.append(Tail.begin(), Tail.end());
    LLVMContext C;
    SMDiagnostic Err;
    std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
    if (!Mod)
      Err.print("AssumeQueryAPI", errs());
    Elem.second(&*(Mod->getFunction("test")->begin()->begin()));
  }
}

static bool FindExactlyAttributes(RetainedKnowledgeMap &Map, Value *WasOn,
                                 StringRef AttrToMatch) {
  Regex Reg(AttrToMatch);
  SmallVector<StringRef, 1> Matches;
  for (StringRef Attr : {
#define GET_ATTR_NAMES
#define ATTRIBUTE_ENUM(ENUM_NAME, DISPLAY_NAME) StringRef(#DISPLAY_NAME),
#include "llvm/IR/Attributes.inc"
       }) {
    bool ShouldHaveAttr = Reg.match(Attr, &Matches) && Matches[0] == Attr;

    if (ShouldHaveAttr != (Map.contains(RetainedKnowledgeKey{
                              WasOn, Attribute::getAttrKindFromName(Attr)})))
      return false;
  }
  return true;
}

static bool MapHasRightValue(RetainedKnowledgeMap &Map, AssumeInst *II,
                             RetainedKnowledgeKey Key, MinMax MM) {
  auto LookupIt = Map.find(Key);
  return (LookupIt != Map.end()) && (LookupIt->second[II].Min == MM.Min) &&
         (LookupIt->second[II].Max == MM.Max);
}

TEST(AssumeQueryAPI, fillMapFromAssume) {
  EnableKnowledgeRetention.setValue(true);
  StringRef Head =
      "declare void @llvm.assume(i1)\n"
      "declare void @func(ptr, ptr, ptr)\n"
      "declare void @func1(ptr, ptr, ptr, ptr)\n"
      "declare void @func_many(ptr) \"no-jump-tables\" nounwind "
      "\"less-precise-fpmad\" willreturn norecurse\n"
      "define void @test(ptr %P, ptr %P1, ptr %P2, ptr %P3) {\n";
  StringRef Tail = "ret void\n"
                   "}";
  std::vector<std::pair<StringRef, llvm::function_ref<void(Instruction *)>>>
      Tests;
  Tests.push_back(std::make_pair(
      "call void @func(ptr nonnull align 4 dereferenceable(16) %P, ptr align "
      "8 noalias %P1, ptr align 8 dereferenceable(8) %P2)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I->getIterator());

        RetainedKnowledgeMap Map;
        fillMapFromAssume(*Assume, Map);
        ASSERT_TRUE(FindExactlyAttributes(Map, I->getOperand(0),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_FALSE(FindExactlyAttributes(Map, I->getOperand(1),
                                       "(align)"));
        ASSERT_TRUE(FindExactlyAttributes(Map, I->getOperand(2),
                                       "(align|dereferenceable)"));
        ASSERT_TRUE(MapHasRightValue(
            Map, Assume, {I->getOperand(0), Attribute::Dereferenceable}, {16, 16}));
        ASSERT_TRUE(MapHasRightValue(Map, Assume, {I->getOperand(0), Attribute::Alignment},
                               {4, 4}));
        ASSERT_TRUE(MapHasRightValue(Map, Assume, {I->getOperand(0), Attribute::Alignment},
                               {4, 4}));
      }));
  Tests.push_back(std::make_pair(
      "call void @func1(ptr nonnull align 32 dereferenceable(48) %P, ptr "
      "nonnull "
      "align 8 dereferenceable(28) %P, ptr nonnull align 64 "
      "dereferenceable(4) "
      "%P, ptr nonnull align 16 dereferenceable(12) %P)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I->getIterator());

        RetainedKnowledgeMap Map;
        fillMapFromAssume(*Assume, Map);

        ASSERT_TRUE(FindExactlyAttributes(Map, I->getOperand(0),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(FindExactlyAttributes(Map, I->getOperand(1),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(FindExactlyAttributes(Map, I->getOperand(2),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(FindExactlyAttributes(Map, I->getOperand(3),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(MapHasRightValue(
            Map, Assume, {I->getOperand(0), Attribute::Dereferenceable},
            {48, 48}));
        ASSERT_TRUE(MapHasRightValue(
            Map, Assume, {I->getOperand(0), Attribute::Alignment}, {64, 64}));
      }));
  Tests.push_back(
      std::make_pair("call void @llvm.assume(i1 true)\n", [](Instruction *I) {
        RetainedKnowledgeMap Map;
        fillMapFromAssume(*cast<AssumeInst>(I), Map);

        ASSERT_TRUE(FindExactlyAttributes(Map, nullptr, ""));
        ASSERT_TRUE(Map.empty());
      }));
  Tests.push_back(std::make_pair(
      "call void @func1(ptr readnone align 32 "
      "dereferenceable(48) noalias %P, ptr "
      "align 8 dereferenceable(28) %P1, ptr align 64 "
      "dereferenceable(4) "
      "%P2, ptr nonnull align 16 dereferenceable(12) %P3)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I->getIterator());

        RetainedKnowledgeMap Map;
        fillMapFromAssume(*Assume, Map);

        ASSERT_TRUE(FindExactlyAttributes(Map, I->getOperand(0),
                                    "(align|dereferenceable)"));
        ASSERT_TRUE(FindExactlyAttributes(Map, I->getOperand(1),
                                    "(align|dereferenceable)"));
        ASSERT_TRUE(FindExactlyAttributes(Map, I->getOperand(2),
                                       "(align|dereferenceable)"));
        ASSERT_TRUE(FindExactlyAttributes(Map, I->getOperand(3),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(MapHasRightValue(Map, Assume, {I->getOperand(0), Attribute::Alignment},
                               {32, 32}));
        ASSERT_TRUE(MapHasRightValue(
            Map, Assume, {I->getOperand(0), Attribute::Dereferenceable}, {48, 48}));
        ASSERT_TRUE(MapHasRightValue(
            Map, Assume, {I->getOperand(1), Attribute::Dereferenceable}, {28, 28}));
        ASSERT_TRUE(MapHasRightValue(Map, Assume, {I->getOperand(1), Attribute::Alignment},
                               {8, 8}));
        ASSERT_TRUE(MapHasRightValue(Map, Assume, {I->getOperand(2), Attribute::Alignment},
                               {64, 64}));
        ASSERT_TRUE(MapHasRightValue(
            Map, Assume, {I->getOperand(2), Attribute::Dereferenceable}, {4, 4}));
        ASSERT_TRUE(MapHasRightValue(Map, Assume, {I->getOperand(3), Attribute::Alignment},
                               {16, 16}));
        ASSERT_TRUE(MapHasRightValue(
            Map, Assume, {I->getOperand(3), Attribute::Dereferenceable}, {12, 12}));
      }));

  /// Keep this test last as it modifies the function.
  Tests.push_back(std::make_pair(
      "call void @func(ptr nonnull align 4 dereferenceable(16) %P, ptr align "
      "8 noalias %P1, ptr %P2)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I->getIterator());

        RetainedKnowledgeMap Map;
        fillMapFromAssume(*Assume, Map);

        Value *New = I->getFunction()->getArg(3);
        Value *Old = I->getOperand(0);
        ASSERT_TRUE(FindExactlyAttributes(Map, New, ""));
        ASSERT_TRUE(FindExactlyAttributes(Map, Old,
                                       "(nonnull|align|dereferenceable)"));
        Old->replaceAllUsesWith(New);
        Map.clear();
        fillMapFromAssume(*Assume, Map);
        ASSERT_TRUE(FindExactlyAttributes(Map, New,
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(FindExactlyAttributes(Map, Old, ""));
      }));
  Tests.push_back(std::make_pair(
      "call void @llvm.assume(i1 true) [\"align\"(ptr undef, i32 undef)]",
      [](Instruction *I) {
        // Don't crash but don't learn from undef.
        RetainedKnowledgeMap Map;
        fillMapFromAssume(*cast<AssumeInst>(I), Map);

        ASSERT_TRUE(Map.empty());
      }));
  RunTest(Head, Tail, Tests);
}

TEST(AssumeQueryAPI, AssumptionCache) {
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(
      "declare void @llvm.assume(i1)\n"
      "define void @test(ptr %P, ptr %P1, ptr %P2, ptr %P3, i1 %B) {\n"
      "call void @llvm.assume(i1 true) [\"nonnull\"(ptr %P), \"align\"(ptr "
      "%P2, i32 4), \"align\"(ptr %P, i32 8)]\n"
      "call void @llvm.assume(i1 %B) [\"test\"(ptr %P1), "
      "\"dereferenceable\"(ptr %P, i32 4)]\n"
      "ret void\n}\n",
      Err, C);
  if (!Mod)
    Err.print("AssumeQueryAPI", errs());
  Function *F = Mod->getFunction("test");
  BasicBlock::iterator First = F->begin()->begin();
  BasicBlock::iterator Second = F->begin()->begin();
  Second++;
  AssumptionCache AC(*F);
  auto AR = AC.assumptionsFor(F->getArg(3));
  ASSERT_EQ(AR.size(), 0u);
  AR = AC.assumptionsFor(F->getArg(1));
  ASSERT_EQ(AR.size(), 1u);
  ASSERT_EQ(AR[0].Index, 0u);
  ASSERT_EQ(AR[0].Assume, &*Second);
  AR = AC.assumptionsFor(F->getArg(2));
  ASSERT_EQ(AR.size(), 1u);
  ASSERT_EQ(AR[0].Index, 1u);
  ASSERT_EQ(AR[0].Assume, &*First);
  AR = AC.assumptionsFor(F->getArg(0));
  ASSERT_EQ(AR.size(), 3u);
  llvm::sort(AR,
             [](const auto &L, const auto &R) { return L.Index < R.Index; });
  ASSERT_EQ(AR[0].Assume, &*First);
  ASSERT_EQ(AR[0].Index, 0u);
  ASSERT_EQ(AR[1].Assume, &*Second);
  ASSERT_EQ(AR[1].Index, 1u);
  ASSERT_EQ(AR[2].Assume, &*First);
  ASSERT_EQ(AR[2].Index, 2u);
  AR = AC.assumptionsFor(F->getArg(4));
  ASSERT_EQ(AR.size(), 1u);
  ASSERT_EQ(AR[0].Assume, &*Second);
  ASSERT_EQ(AR[0].Index, AssumptionCache::ExprResultIdx);
  AC.unregisterAssumption(cast<AssumeInst>(&*Second));
  AR = AC.assumptionsFor(F->getArg(1));
  ASSERT_EQ(AR.size(), 0u);
  AR = AC.assumptionsFor(F->getArg(0));
  ASSERT_EQ(AR.size(), 3u);
  llvm::sort(AR,
             [](const auto &L, const auto &R) { return L.Index < R.Index; });
  ASSERT_EQ(AR[0].Assume, &*First);
  ASSERT_EQ(AR[0].Index, 0u);
  ASSERT_EQ(AR[1].Assume, nullptr);
  ASSERT_EQ(AR[1].Index, 1u);
  ASSERT_EQ(AR[2].Assume, &*First);
  ASSERT_EQ(AR[2].Index, 2u);
  AR = AC.assumptionsFor(F->getArg(2));
  ASSERT_EQ(AR.size(), 1u);
  ASSERT_EQ(AR[0].Index, 1u);
  ASSERT_EQ(AR[0].Assume, &*First);
}

TEST(AssumeQueryAPI, Alignment) {
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(
      "declare void @llvm.assume(i1)\n"
      "define void @test(ptr %P, ptr %P1, ptr %P2, i32 %I3, i1 %B) {\n"
      "call void @llvm.assume(i1 true) [\"align\"(ptr %P, i32 8, i32 %I3)]\n"
      "call void @llvm.assume(i1 true) [\"align\"(ptr %P1, i32 %I3, i32 "
      "%I3)]\n"
      "call void @llvm.assume(i1 true) [\"align\"(ptr %P2, i32 16, i32 8)]\n"
      "ret void\n}\n",
      Err, C);
  if (!Mod)
    Err.print("AssumeQueryAPI", errs());

  Function *F = Mod->getFunction("test");
  BasicBlock::iterator Start = F->begin()->begin();
  AssumeInst *II;
  RetainedKnowledge RK;
  II = cast<AssumeInst>(&*Start);
  RK = getKnowledgeFromBundle(*II, II->bundle_op_info_begin()[0]);
  ASSERT_EQ(RK.AttrKind, Attribute::Alignment);
  ASSERT_EQ(RK.WasOn, F->getArg(0));
  ASSERT_EQ(RK.ArgValue, 1u);
  Start++;
  II = cast<AssumeInst>(&*Start);
  RK = getKnowledgeFromBundle(*II, II->bundle_op_info_begin()[0]);
  ASSERT_EQ(RK.AttrKind, Attribute::Alignment);
  ASSERT_EQ(RK.WasOn, F->getArg(1));
  ASSERT_EQ(RK.ArgValue, 1u);
  Start++;
  II = cast<AssumeInst>(&*Start);
  RK = getKnowledgeFromBundle(*II, II->bundle_op_info_begin()[0]);
  ASSERT_EQ(RK.AttrKind, Attribute::Alignment);
  ASSERT_EQ(RK.WasOn, F->getArg(2));
  ASSERT_EQ(RK.ArgValue, 8u);
}
