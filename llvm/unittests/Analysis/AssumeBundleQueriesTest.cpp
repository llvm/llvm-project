//===- AssumeBundleQueriesTest.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "gtest/gtest.h"
#include <random>

using namespace llvm;

namespace llvm {
extern cl::opt<bool> ShouldPreserveAllAttributes;
extern cl::opt<bool> EnableKnowledgeRetention;
} // namespace llvm

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

bool hasMatchesExactlyAttributes(AssumeInst *Assume, Value *WasOn,
                                 StringRef AttrToMatch) {
  Regex Reg(AttrToMatch);
  SmallVector<StringRef, 1> Matches;
  for (StringRef Attr : {
#define GET_ATTR_NAMES
#define ATTRIBUTE_ALL(ENUM_NAME, DISPLAY_NAME) StringRef(#DISPLAY_NAME),
#include "llvm/IR/Attributes.inc"
       }) {
    bool ShouldHaveAttr = Reg.match(Attr, &Matches) && Matches[0] == Attr;
    if (ShouldHaveAttr != hasAttributeInAssume(*Assume, WasOn, Attr))
      return false;
  }
  return true;
}

bool hasTheRightValue(AssumeInst *Assume, Value *WasOn,
                      Attribute::AttrKind Kind, unsigned Value) {
  uint64_t ArgVal = 0;
  if (!hasAttributeInAssume(*Assume, WasOn, Kind, &ArgVal))
    return false;
  if (ArgVal != Value)
    return false;
  return true;
}

TEST(AssumeQueryAPI, hasAttributeInAssume) {
  EnableKnowledgeRetention.setValue(true);
  StringRef Head =
      "declare void @llvm.assume(i1)\n"
      "declare void @func(i32*, i32*, i32*)\n"
      "declare void @func1(i32*, i32*, i32*, i32*)\n"
      "declare void @func_many(i32*) \"no-jump-tables\" nounwind "
      "\"less-precise-fpmad\" willreturn norecurse\n"
      "define void @test(i32* %P, i32* %P1, i32* %P2, i32* %P3) {\n";
  StringRef Tail = "ret void\n"
                   "}";
  std::vector<std::pair<StringRef, llvm::function_ref<void(Instruction *)>>>
      Tests;
  Tests.push_back(std::make_pair(
      "call void @func(i32* nonnull align 4 dereferenceable(16) %P, i32* align "
      "8 noalias %P1, i32* align 8 noundef %P2)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I);
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(0),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(1),
                                       "()"));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(2),
                                       "(align|noundef)"));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(0),
                                     Attribute::AttrKind::Dereferenceable, 16));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(0),
                                     Attribute::AttrKind::Alignment, 4));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(0),
                                     Attribute::AttrKind::Alignment, 4));
      }));
  Tests.push_back(std::make_pair(
      "call void @func1(i32* nonnull align 32 dereferenceable(48) %P, i32* "
      "nonnull "
      "align 8 dereferenceable(28) %P, i32* nonnull align 64 "
      "dereferenceable(4) "
      "%P, i32* nonnull align 16 dereferenceable(12) %P)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I);
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(0),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(1),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(2),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(3),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(0),
                                     Attribute::AttrKind::Dereferenceable, 48));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(0),
                                     Attribute::AttrKind::Alignment, 64));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(1),
                                     Attribute::AttrKind::Alignment, 64));
      }));
  Tests.push_back(std::make_pair(
      "call void @func_many(i32* align 8 noundef %P1) cold\n", [](Instruction *I) {
        ShouldPreserveAllAttributes.setValue(true);
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I);
        ASSERT_TRUE(hasMatchesExactlyAttributes(
            Assume, nullptr,
            "(align|nounwind|norecurse|noundef|willreturn|cold)"));
        ShouldPreserveAllAttributes.setValue(false);
      }));
  Tests.push_back(
      std::make_pair("call void @llvm.assume(i1 true)\n", [](Instruction *I) {
        auto *Assume = cast<AssumeInst>(I);
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, nullptr, ""));
      }));
  Tests.push_back(std::make_pair(
      "call void @func1(i32* readnone align 32 "
      "dereferenceable(48) noalias %P, i32* "
      "align 8 dereferenceable(28) %P1, i32* align 64 "
      "dereferenceable(4) "
      "%P2, i32* nonnull align 16 dereferenceable(12) %P3)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I);
        ASSERT_TRUE(hasMatchesExactlyAttributes(
            Assume, I->getOperand(0),
            "(align|dereferenceable)"));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(1),
                                       "(align|dereferenceable)"));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(2),
                                       "(align|dereferenceable)"));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(3),
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(0),
                                     Attribute::AttrKind::Alignment, 32));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(0),
                                     Attribute::AttrKind::Dereferenceable, 48));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(1),
                                     Attribute::AttrKind::Dereferenceable, 28));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(1),
                                     Attribute::AttrKind::Alignment, 8));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(2),
                                     Attribute::AttrKind::Alignment, 64));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(2),
                                     Attribute::AttrKind::Dereferenceable, 4));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(3),
                                     Attribute::AttrKind::Alignment, 16));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(3),
                                     Attribute::AttrKind::Dereferenceable, 12));
      }));

  Tests.push_back(std::make_pair(
      "call void @func1(i32* readnone align 32 "
      "dereferenceable(48) noalias %P, i32* "
      "align 8 dereferenceable(28) %P1, i32* align 64 "
      "dereferenceable(4) "
      "%P2, i32* nonnull align 16 dereferenceable(12) %P3)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I);
        I->getOperand(1)->dropDroppableUses();
        I->getOperand(2)->dropDroppableUses();
        I->getOperand(3)->dropDroppableUses();
        ASSERT_TRUE(hasMatchesExactlyAttributes(
            Assume, I->getOperand(0),
            "(align|dereferenceable)"));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(1),
                                       ""));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(2),
                                       ""));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, I->getOperand(3),
                                       ""));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(0),
                                     Attribute::AttrKind::Alignment, 32));
        ASSERT_TRUE(hasTheRightValue(Assume, I->getOperand(0),
                                     Attribute::AttrKind::Dereferenceable, 48));
      }));
  Tests.push_back(std::make_pair(
      "call void @func(i32* nonnull align 4 dereferenceable(16) %P, i32* align "
      "8 noalias %P1, i32* %P1)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I);
        Value *New = I->getFunction()->getArg(3);
        Value *Old = I->getOperand(0);
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, New, ""));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, Old,
                                       "(nonnull|align|dereferenceable)"));
        Old->replaceAllUsesWith(New);
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, New,
                                       "(nonnull|align|dereferenceable)"));
        ASSERT_TRUE(hasMatchesExactlyAttributes(Assume, Old, ""));
      }));
  RunTest(Head, Tail, Tests);
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

    if (ShouldHaveAttr != (Map.find(RetainedKnowledgeKey{WasOn, Attribute::getAttrKindFromName(Attr)}) != Map.end()))
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
      "declare void @func(i32*, i32*, i32*)\n"
      "declare void @func1(i32*, i32*, i32*, i32*)\n"
      "declare void @func_many(i32*) \"no-jump-tables\" nounwind "
      "\"less-precise-fpmad\" willreturn norecurse\n"
      "define void @test(i32* %P, i32* %P1, i32* %P2, i32* %P3) {\n";
  StringRef Tail = "ret void\n"
                   "}";
  std::vector<std::pair<StringRef, llvm::function_ref<void(Instruction *)>>>
      Tests;
  Tests.push_back(std::make_pair(
      "call void @func(i32* nonnull align 4 dereferenceable(16) %P, i32* align "
      "8 noalias %P1, i32* align 8 dereferenceable(8) %P2)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I);

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
      "call void @func1(i32* nonnull align 32 dereferenceable(48) %P, i32* "
      "nonnull "
      "align 8 dereferenceable(28) %P, i32* nonnull align 64 "
      "dereferenceable(4) "
      "%P, i32* nonnull align 16 dereferenceable(12) %P)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I);

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
  Tests.push_back(std::make_pair(
      "call void @func_many(i32* align 8 %P1) cold\n", [](Instruction *I) {
        ShouldPreserveAllAttributes.setValue(true);
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I);

        RetainedKnowledgeMap Map;
        fillMapFromAssume(*Assume, Map);

        ASSERT_TRUE(FindExactlyAttributes(
            Map, nullptr, "(nounwind|norecurse|willreturn|cold)"));
        ShouldPreserveAllAttributes.setValue(false);
      }));
  Tests.push_back(
      std::make_pair("call void @llvm.assume(i1 true)\n", [](Instruction *I) {
        RetainedKnowledgeMap Map;
        fillMapFromAssume(*cast<AssumeInst>(I), Map);

        ASSERT_TRUE(FindExactlyAttributes(Map, nullptr, ""));
        ASSERT_TRUE(Map.empty());
      }));
  Tests.push_back(std::make_pair(
      "call void @func1(i32* readnone align 32 "
      "dereferenceable(48) noalias %P, i32* "
      "align 8 dereferenceable(28) %P1, i32* align 64 "
      "dereferenceable(4) "
      "%P2, i32* nonnull align 16 dereferenceable(12) %P3)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I);

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
      "call void @func(i32* nonnull align 4 dereferenceable(16) %P, i32* align "
      "8 noalias %P1, i32* %P2)\n",
      [](Instruction *I) {
        auto *Assume = buildAssumeFromInst(I);
        Assume->insertBefore(I);

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
      "call void @llvm.assume(i1 true) [\"align\"(i8* undef, i32 undef)]",
      [](Instruction *I) {
        // Don't crash but don't learn from undef.
        RetainedKnowledgeMap Map;
        fillMapFromAssume(*cast<AssumeInst>(I), Map);

        ASSERT_TRUE(Map.empty());
      }));
  RunTest(Head, Tail, Tests);
}

static void RunRandTest(uint64_t Seed, int Size, int MinCount, int MaxCount,
                        unsigned MaxValue) {
  LLVMContext C;
  SMDiagnostic Err;

  std::random_device dev;
  std::mt19937 Rng(Seed);
  std::uniform_int_distribution<int> DistCount(MinCount, MaxCount);
  std::uniform_int_distribution<unsigned> DistValue(0, MaxValue);
  std::uniform_int_distribution<unsigned> DistAttr(0,
                                                   Attribute::EndAttrKinds - 1);

  std::unique_ptr<Module> Mod = std::make_unique<Module>("AssumeQueryAPI", C);
  if (!Mod)
    Err.print("AssumeQueryAPI", errs());

  std::vector<Type *> TypeArgs;
  for (int i = 0; i < (Size * 2); i++)
    TypeArgs.push_back(Type::getInt32PtrTy(C));
  FunctionType *FuncType =
      FunctionType::get(Type::getVoidTy(C), TypeArgs, false);

  Function *F =
      Function::Create(FuncType, GlobalValue::ExternalLinkage, "test", &*Mod);
  BasicBlock *BB = BasicBlock::Create(C);
  BB->insertInto(F);
  Instruction *Ret = ReturnInst::Create(C);
  BB->getInstList().insert(BB->begin(), Ret);
  Function *FnAssume = Intrinsic::getDeclaration(Mod.get(), Intrinsic::assume);

  std::vector<Argument *> ShuffledArgs;
  BitVector HasArg;
  for (auto &Arg : F->args()) {
    ShuffledArgs.push_back(&Arg);
    HasArg.push_back(false);
  }

  std::shuffle(ShuffledArgs.begin(), ShuffledArgs.end(), Rng);

  std::vector<OperandBundleDef> OpBundle;
  OpBundle.reserve(Size);
  std::vector<Value *> Args;
  Args.reserve(2);
  for (int i = 0; i < Size; i++) {
    int count = DistCount(Rng);
    int value = DistValue(Rng);
    int attr = DistAttr(Rng);
    std::string str;
    raw_string_ostream ss(str);
    ss << Attribute::getNameFromAttrKind(
        static_cast<Attribute::AttrKind>(attr));
    Args.clear();

    if (count > 0) {
      Args.push_back(ShuffledArgs[i]);
      HasArg[i] = true;
    }
    if (count > 1)
      Args.push_back(ConstantInt::get(Type::getInt32Ty(C), value));

    OpBundle.push_back(OperandBundleDef{ss.str().c_str(), std::move(Args)});
  }

  auto *Assume = cast<AssumeInst>(CallInst::Create(
      FnAssume, ArrayRef<Value *>({ConstantInt::getTrue(C)}), OpBundle));
  Assume->insertBefore(&F->begin()->front());
  RetainedKnowledgeMap Map;
  fillMapFromAssume(*Assume, Map);
  for (int i = 0; i < (Size * 2); i++) {
    if (!HasArg[i])
      continue;
    RetainedKnowledge K =
        getKnowledgeFromUseInAssume(&*ShuffledArgs[i]->use_begin());
    auto LookupIt = Map.find(RetainedKnowledgeKey{K.WasOn, K.AttrKind});
    ASSERT_TRUE(LookupIt != Map.end());
    MinMax MM = LookupIt->second[Assume];
    ASSERT_TRUE(MM.Min == MM.Max);
    ASSERT_TRUE(MM.Min == K.ArgValue);
  }
}

TEST(AssumeQueryAPI, getKnowledgeFromUseInAssume) {
  // // For Fuzzing
  // std::random_device dev;
  // std::mt19937 Rng(dev());
  // while (true) {
  //   unsigned Seed = Rng();
  //   dbgs() << Seed << "\n";
  //   RunRandTest(Seed, 100000, 0, 2, 100);
  // }
  RunRandTest(23456, 4, 0, 2, 100);
  RunRandTest(560987, 25, -3, 2, 100);

  // Large bundles can lead to special cases. this is why this test is soo
  // large.
  RunRandTest(9876789, 100000, -0, 7, 100);
}

TEST(AssumeQueryAPI, AssumptionCache) {
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(
      "declare void @llvm.assume(i1)\n"
      "define void @test(i32* %P, i32* %P1, i32* %P2, i32* %P3, i1 %B) {\n"
      "call void @llvm.assume(i1 true) [\"nonnull\"(i32* %P), \"align\"(i32* "
      "%P2, i32 4), \"align\"(i32* %P, i32 8)]\n"
      "call void @llvm.assume(i1 %B) [\"test\"(i32* %P1), "
      "\"dereferenceable\"(i32* %P, i32 4)]\n"
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
      "define void @test(i32* %P, i32* %P1, i32* %P2, i32 %I3, i1 %B) {\n"
      "call void @llvm.assume(i1 true) [\"align\"(i32* %P, i32 8, i32 %I3)]\n"
      "call void @llvm.assume(i1 true) [\"align\"(i32* %P1, i32 %I3, i32 "
      "%I3)]\n"
      "call void @llvm.assume(i1 true) [\"align\"(i32* %P2, i32 16, i32 8)]\n"
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
