//===- KnowledgeRetention.h - utilities to preserve informations *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/KnowledgeRetention.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"
#include <random>

using namespace llvm;

extern cl::opt<bool> ShouldPreserveAllAttributes;

static void RunTest(
    StringRef Head, StringRef Tail,
    std::vector<std::pair<StringRef, llvm::function_ref<void(Instruction *)>>>
        &Tests) {
  std::string IR;
  IR.append(Head.begin(), Head.end());
  for (auto &Elem : Tests)
    IR.append(Elem.first.begin(), Elem.first.end());
  IR.append(Tail.begin(), Tail.end());
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("AssumeQueryAPI", errs());
  unsigned Idx = 0;
  for (Instruction &I : (*Mod->getFunction("test")->begin())) {
    if (Idx < Tests.size())
      Tests[Idx].second(&I);
    Idx++;
  }
}

static void AssertMatchesExactlyAttributes(CallInst *Assume, Value *WasOn,
                                    StringRef AttrToMatch) {
  Regex Reg(AttrToMatch);
  SmallVector<StringRef, 1> Matches;
  for (StringRef Attr : {
#define GET_ATTR_NAMES
#define ATTRIBUTE_ALL(ENUM_NAME, DISPLAY_NAME) StringRef(#DISPLAY_NAME),
#include "llvm/IR/Attributes.inc"
       }) {
    bool ShouldHaveAttr = Reg.match(Attr, &Matches) && Matches[0] == Attr;
    if (ShouldHaveAttr != hasAttributeInAssume(*Assume, WasOn, Attr)) {
      ASSERT_TRUE(false);
    }
  }
}

static void AssertHasTheRightValue(CallInst *Assume, Value *WasOn,
                            Attribute::AttrKind Kind, unsigned Value, bool Both,
                            AssumeQuery AQ = AssumeQuery::Highest) {
  if (!Both) {
    uint64_t ArgVal = 0;
    ASSERT_TRUE(hasAttributeInAssume(*Assume, WasOn, Kind, &ArgVal, AQ));
    ASSERT_EQ(ArgVal, Value);
    return;
  }
  uint64_t ArgValLow = 0;
  uint64_t ArgValHigh = 0;
  bool ResultLow = hasAttributeInAssume(*Assume, WasOn, Kind, &ArgValLow,
                                        AssumeQuery::Lowest);
  bool ResultHigh = hasAttributeInAssume(*Assume, WasOn, Kind, &ArgValHigh,
                                         AssumeQuery::Highest);
  if (ResultLow != ResultHigh || ResultHigh == false) {
    ASSERT_TRUE(false);
  }
  if (ArgValLow != Value || ArgValLow != ArgValHigh) {
    ASSERT_TRUE(false);
  }
}

TEST(AssumeQueryAPI, hasAttributeInAssume) {
  StringRef Head =
      "declare void @llvm.assume(i1)\n"
      "declare void @func(i32*, i32*)\n"
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
      "8 noalias %P1)\n",
      [](Instruction *I) {
        CallInst *Assume = BuildAssumeFromInst(I);
        Assume->insertBefore(I);
        AssertMatchesExactlyAttributes(Assume, I->getOperand(0),
                                       "(nonnull|align|dereferenceable)");
        AssertMatchesExactlyAttributes(Assume, I->getOperand(1),
                                       "(noalias|align)");
        AssertHasTheRightValue(Assume, I->getOperand(0),
                               Attribute::AttrKind::Dereferenceable, 16, true);
        AssertHasTheRightValue(Assume, I->getOperand(0),
                               Attribute::AttrKind::Alignment, 4, true);
        AssertHasTheRightValue(Assume, I->getOperand(0),
                               Attribute::AttrKind::Alignment, 4, true);
      }));
  Tests.push_back(std::make_pair(
      "call void @func1(i32* nonnull align 32 dereferenceable(48) %P, i32* "
      "nonnull "
      "align 8 dereferenceable(28) %P, i32* nonnull align 64 "
      "dereferenceable(4) "
      "%P, i32* nonnull align 16 dereferenceable(12) %P)\n",
      [](Instruction *I) {
        CallInst *Assume = BuildAssumeFromInst(I);
        Assume->insertBefore(I);
        AssertMatchesExactlyAttributes(Assume, I->getOperand(0),
                                       "(nonnull|align|dereferenceable)");
        AssertMatchesExactlyAttributes(Assume, I->getOperand(1),
                                       "(nonnull|align|dereferenceable)");
        AssertMatchesExactlyAttributes(Assume, I->getOperand(2),
                                       "(nonnull|align|dereferenceable)");
        AssertMatchesExactlyAttributes(Assume, I->getOperand(3),
                                       "(nonnull|align|dereferenceable)");
        AssertHasTheRightValue(Assume, I->getOperand(0),
                               Attribute::AttrKind::Dereferenceable, 48, false,
                               AssumeQuery::Highest);
        AssertHasTheRightValue(Assume, I->getOperand(0),
                               Attribute::AttrKind::Alignment, 64, false,
                               AssumeQuery::Highest);
        AssertHasTheRightValue(Assume, I->getOperand(1),
                               Attribute::AttrKind::Alignment, 64, false,
                               AssumeQuery::Highest);
        AssertHasTheRightValue(Assume, I->getOperand(0),
                               Attribute::AttrKind::Dereferenceable, 4, false,
                               AssumeQuery::Lowest);
        AssertHasTheRightValue(Assume, I->getOperand(0),
                               Attribute::AttrKind::Alignment, 8, false,
                               AssumeQuery::Lowest);
        AssertHasTheRightValue(Assume, I->getOperand(1),
                               Attribute::AttrKind::Alignment, 8, false,
                               AssumeQuery::Lowest);
      }));
  Tests.push_back(std::make_pair(
      "call void @func_many(i32* align 8 %P1) cold\n", [](Instruction *I) {
        ShouldPreserveAllAttributes.setValue(true);
        CallInst *Assume = BuildAssumeFromInst(I);
        Assume->insertBefore(I);
        AssertMatchesExactlyAttributes(
            Assume, nullptr,
            "(align|no-jump-tables|less-precise-fpmad|"
            "nounwind|norecurse|willreturn|cold)");
        ShouldPreserveAllAttributes.setValue(false);
      }));
  Tests.push_back(
      std::make_pair("call void @llvm.assume(i1 true)\n", [](Instruction *I) {
        CallInst *Assume = cast<CallInst>(I);
        AssertMatchesExactlyAttributes(Assume, nullptr, "");
      }));
  Tests.push_back(std::make_pair(
      "call void @func1(i32* readnone align 32 "
      "dereferenceable(48) noalias %P, i32* "
      "align 8 dereferenceable(28) %P1, i32* align 64 "
      "dereferenceable(4) "
      "%P2, i32* nonnull align 16 dereferenceable(12) %P3)\n",
      [](Instruction *I) {
        CallInst *Assume = BuildAssumeFromInst(I);
        Assume->insertBefore(I);
        AssertMatchesExactlyAttributes(
            Assume, I->getOperand(0),
            "(readnone|align|dereferenceable|noalias)");
        AssertMatchesExactlyAttributes(Assume, I->getOperand(1),
                                       "(align|dereferenceable)");
        AssertMatchesExactlyAttributes(Assume, I->getOperand(2),
                                       "(align|dereferenceable)");
        AssertMatchesExactlyAttributes(Assume, I->getOperand(3),
                                       "(nonnull|align|dereferenceable)");
        AssertHasTheRightValue(Assume, I->getOperand(0),
                               Attribute::AttrKind::Alignment, 32, true);
        AssertHasTheRightValue(Assume, I->getOperand(0),
                               Attribute::AttrKind::Dereferenceable, 48, true);
        AssertHasTheRightValue(Assume, I->getOperand(1),
                               Attribute::AttrKind::Dereferenceable, 28, true);
        AssertHasTheRightValue(Assume, I->getOperand(1),
                               Attribute::AttrKind::Alignment, 8, true);
        AssertHasTheRightValue(Assume, I->getOperand(2),
                               Attribute::AttrKind::Alignment, 64, true);
        AssertHasTheRightValue(Assume, I->getOperand(2),
                               Attribute::AttrKind::Dereferenceable, 4, true);
        AssertHasTheRightValue(Assume, I->getOperand(3),
                               Attribute::AttrKind::Alignment, 16, true);
        AssertHasTheRightValue(Assume, I->getOperand(3),
                               Attribute::AttrKind::Dereferenceable, 12, true);
      }));

  /// Keep this test last as it modifies the function.
  Tests.push_back(std::make_pair(
      "call void @func(i32* nonnull align 4 dereferenceable(16) %P, i32* align "
      "8 noalias %P1)\n",
      [](Instruction *I) {
        CallInst *Assume = BuildAssumeFromInst(I);
        Assume->insertBefore(I);
        Value *New = I->getFunction()->getArg(3);
        Value *Old = I->getOperand(0);
        AssertMatchesExactlyAttributes(Assume, New, "");
        AssertMatchesExactlyAttributes(Assume, Old,
                                       "(nonnull|align|dereferenceable)");
        Old->replaceAllUsesWith(New);
        AssertMatchesExactlyAttributes(Assume, New,
                                       "(nonnull|align|dereferenceable)");
        AssertMatchesExactlyAttributes(Assume, Old, "");
      }));
  RunTest(Head, Tail, Tests);
}

static void AssertFindExactlyAttributes(RetainedKnowledgeMap &Map, Value *WasOn,
                                 StringRef AttrToMatch) {
  Regex Reg(AttrToMatch);
  SmallVector<StringRef, 1> Matches;
  for (StringRef Attr : {
#define GET_ATTR_NAMES
#define ATTRIBUTE_ENUM(ENUM_NAME, DISPLAY_NAME) StringRef(#DISPLAY_NAME),
#include "llvm/IR/Attributes.inc"
       }) {
    bool ShouldHaveAttr = Reg.match(Attr, &Matches) && Matches[0] == Attr;

    if (ShouldHaveAttr != (Map.find(RetainedKnowledgeKey{WasOn, Attribute::getAttrKindFromName(Attr)}) != Map.end())) {
      ASSERT_TRUE(false);
    }
  }
}

static void AssertMapHasRightValue(RetainedKnowledgeMap &Map,
                                   RetainedKnowledgeKey Key, MinMax MM) {
  auto LookupIt = Map.find(Key);
  ASSERT_TRUE(LookupIt != Map.end());
  ASSERT_TRUE(LookupIt->second.Min == MM.Min);
  ASSERT_TRUE(LookupIt->second.Max == MM.Max);
}

TEST(AssumeQueryAPI, fillMapFromAssume) {
    StringRef Head =
      "declare void @llvm.assume(i1)\n"
      "declare void @func(i32*, i32*)\n"
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
      "8 noalias %P1)\n",
      [](Instruction *I) {
        CallInst *Assume = BuildAssumeFromInst(I);
        Assume->insertBefore(I);

        RetainedKnowledgeMap Map;
        fillMapFromAssume(*Assume, Map);
        AssertFindExactlyAttributes(Map, I->getOperand(0),
                                       "(nonnull|align|dereferenceable)");
        AssertFindExactlyAttributes(Map, I->getOperand(1),
                                       "(noalias|align)");
        AssertMapHasRightValue(
            Map, {I->getOperand(0), Attribute::Dereferenceable}, {16, 16});
        AssertMapHasRightValue(Map, {I->getOperand(0), Attribute::Alignment},
                               {4, 4});
        AssertMapHasRightValue(Map, {I->getOperand(0), Attribute::Alignment},
                               {4, 4});
      }));
  Tests.push_back(std::make_pair(
      "call void @func1(i32* nonnull align 32 dereferenceable(48) %P, i32* "
      "nonnull "
      "align 8 dereferenceable(28) %P, i32* nonnull align 64 "
      "dereferenceable(4) "
      "%P, i32* nonnull align 16 dereferenceable(12) %P)\n",
      [](Instruction *I) {
        CallInst *Assume = BuildAssumeFromInst(I);
        Assume->insertBefore(I);

        RetainedKnowledgeMap Map;
        fillMapFromAssume(*Assume, Map);

        AssertFindExactlyAttributes(Map, I->getOperand(0),
                                       "(nonnull|align|dereferenceable)");
        AssertFindExactlyAttributes(Map, I->getOperand(1),
                                       "(nonnull|align|dereferenceable)");
        AssertFindExactlyAttributes(Map, I->getOperand(2),
                                       "(nonnull|align|dereferenceable)");
        AssertFindExactlyAttributes(Map, I->getOperand(3),
                                       "(nonnull|align|dereferenceable)");
        AssertMapHasRightValue(
            Map, {I->getOperand(0), Attribute::Dereferenceable}, {4, 48});
        AssertMapHasRightValue(Map, {I->getOperand(0), Attribute::Alignment},
                               {8, 64});
      }));
  Tests.push_back(std::make_pair(
      "call void @func_many(i32* align 8 %P1) cold\n", [](Instruction *I) {
        ShouldPreserveAllAttributes.setValue(true);
        CallInst *Assume = BuildAssumeFromInst(I);
        Assume->insertBefore(I);

        RetainedKnowledgeMap Map;
        fillMapFromAssume(*Assume, Map);

        AssertFindExactlyAttributes(
            Map, nullptr, "(nounwind|norecurse|willreturn|cold)");
        ShouldPreserveAllAttributes.setValue(false);
      }));
  Tests.push_back(
      std::make_pair("call void @llvm.assume(i1 true)\n", [](Instruction *I) {
        RetainedKnowledgeMap Map;
        fillMapFromAssume(*cast<CallInst>(I), Map);

        AssertFindExactlyAttributes(Map, nullptr, "");
        ASSERT_TRUE(Map.empty());
      }));
  Tests.push_back(std::make_pair(
      "call void @func1(i32* readnone align 32 "
      "dereferenceable(48) noalias %P, i32* "
      "align 8 dereferenceable(28) %P1, i32* align 64 "
      "dereferenceable(4) "
      "%P2, i32* nonnull align 16 dereferenceable(12) %P3)\n",
      [](Instruction *I) {
        CallInst *Assume = BuildAssumeFromInst(I);
        Assume->insertBefore(I);

        RetainedKnowledgeMap Map;
        fillMapFromAssume(*Assume, Map);

        AssertFindExactlyAttributes(Map, I->getOperand(0),
                                    "(readnone|align|dereferenceable|noalias)");
        AssertFindExactlyAttributes(Map, I->getOperand(1),
                                    "(align|dereferenceable)");
        AssertFindExactlyAttributes(Map, I->getOperand(2),
                                       "(align|dereferenceable)");
        AssertFindExactlyAttributes(Map, I->getOperand(3),
                                       "(nonnull|align|dereferenceable)");
        AssertMapHasRightValue(Map, {I->getOperand(0), Attribute::Alignment},
                               {32, 32});
        AssertMapHasRightValue(
            Map, {I->getOperand(0), Attribute::Dereferenceable}, {48, 48});
        AssertMapHasRightValue(
            Map, {I->getOperand(0), Attribute::NoAlias}, {0, 0});
        AssertMapHasRightValue(
            Map, {I->getOperand(1), Attribute::Dereferenceable}, {28, 28});
        AssertMapHasRightValue(Map, {I->getOperand(1), Attribute::Alignment},
                               {8, 8});
        AssertMapHasRightValue(Map, {I->getOperand(2), Attribute::Alignment},
                               {64, 64});
        AssertMapHasRightValue(
            Map, {I->getOperand(2), Attribute::Dereferenceable}, {4, 4});
        AssertMapHasRightValue(Map, {I->getOperand(3), Attribute::Alignment},
                               {16, 16});
        AssertMapHasRightValue(
            Map, {I->getOperand(3), Attribute::Dereferenceable}, {12, 12});
      }));

  /// Keep this test last as it modifies the function.
  Tests.push_back(std::make_pair(
      "call void @func(i32* nonnull align 4 dereferenceable(16) %P, i32* align "
      "8 noalias %P1)\n",
      [](Instruction *I) {
        CallInst *Assume = BuildAssumeFromInst(I);
        Assume->insertBefore(I);

        RetainedKnowledgeMap Map;
        fillMapFromAssume(*Assume, Map);

        Value *New = I->getFunction()->getArg(3);
        Value *Old = I->getOperand(0);
        AssertFindExactlyAttributes(Map, New, "");
        AssertFindExactlyAttributes(Map, Old,
                                       "(nonnull|align|dereferenceable)");
        Old->replaceAllUsesWith(New);
        Map.clear();
        fillMapFromAssume(*Assume, Map);
        AssertFindExactlyAttributes(Map, New,
                                       "(nonnull|align|dereferenceable)");
        AssertFindExactlyAttributes(Map, Old, "");
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
  std::vector<bool> HasArg;
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

  Instruction *Assume =
      CallInst::Create(FnAssume, ArrayRef<Value *>({ConstantInt::getTrue(C)}),
                       std::move(OpBundle));
  Assume->insertBefore(&F->begin()->front());
  RetainedKnowledgeMap Map;
  fillMapFromAssume(*cast<CallInst>(Assume), Map);
  for (int i = 0; i < (Size * 2); i++) {
    if (!HasArg[i])
      continue;
    RetainedKnowledge K =
        getKnowledgeFromUseInAssume(&*ShuffledArgs[i]->use_begin());
    auto LookupIt = Map.find(RetainedKnowledgeKey{K.WasOn, K.AttrKind});
    ASSERT_TRUE(LookupIt != Map.end());
    MinMax MM = LookupIt->second;
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
