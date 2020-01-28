//===----------- CoreAPIsTest.cpp - Unit tests for Core ORC APIs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/Orc/Legacy.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

class LegacyAPIsStandardTest : public CoreAPIsBasedStandardTest {};

namespace {

TEST_F(LegacyAPIsStandardTest, TestLambdaSymbolResolver) {
  BarSym.setFlags(BarSym.getFlags() | JITSymbolFlags::Weak);

  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));

  auto Resolver = createSymbolResolver(
      [&](const SymbolNameSet &Symbols) {
        auto FlagsMap = cantFail(JD.lookupFlags(
            LookupKind::Static, JITDylibLookupFlags::MatchExportedSymbolsOnly,
            SymbolLookupSet(Symbols)));
        SymbolNameSet Result;
        for (auto &KV : FlagsMap)
          if (!KV.second.isStrong())
            Result.insert(KV.first);
        return Result;
      },
      [&](std::shared_ptr<AsynchronousSymbolQuery> Q, SymbolNameSet Symbols) {
        return cantFail(JD.legacyLookup(std::move(Q), Symbols));
      });

  auto RS = Resolver->getResponsibilitySet(SymbolNameSet({Bar, Baz}));

  EXPECT_EQ(RS.size(), 1U)
      << "getResponsibilitySet returned the wrong number of results";
  EXPECT_EQ(RS.count(Bar), 1U)
      << "getResponsibilitySet result incorrect. Should be {'bar'}";

  bool OnCompletionRun = false;

  auto OnCompletion = [&](Expected<SymbolMap> Result) {
    OnCompletionRun = true;
    EXPECT_TRUE(!!Result) << "Unexpected error";
    EXPECT_EQ(Result->size(), 2U) << "Unexpected number of resolved symbols";
    EXPECT_EQ(Result->count(Foo), 1U) << "Missing lookup result for foo";
    EXPECT_EQ(Result->count(Bar), 1U) << "Missing lookup result for bar";
    EXPECT_EQ((*Result)[Foo].getAddress(), FooSym.getAddress())
        << "Incorrect address for foo";
    EXPECT_EQ((*Result)[Bar].getAddress(), BarSym.getAddress())
        << "Incorrect address for bar";
  };

  auto Q = std::make_shared<AsynchronousSymbolQuery>(
      SymbolLookupSet({Foo, Bar}), SymbolState::Resolved, OnCompletion);
  auto Unresolved =
      Resolver->lookup(std::move(Q), SymbolNameSet({Foo, Bar, Baz}));

  EXPECT_EQ(Unresolved.size(), 1U) << "Expected one unresolved symbol";
  EXPECT_EQ(Unresolved.count(Baz), 1U) << "Expected baz to not be resolved";
  EXPECT_TRUE(OnCompletionRun) << "OnCompletion was never run";
}

TEST_F(LegacyAPIsStandardTest, LegacyLookupHelpersFn) {
  bool BarMaterialized = false;
  BarSym.setFlags(BarSym.getFlags() | JITSymbolFlags::Weak);

  auto LegacyLookup = [&](StringRef Name) -> JITSymbol {
    if (Name == "foo")
      return FooSym;

    if (Name == "bar") {
      auto BarMaterializer = [&]() -> Expected<JITTargetAddress> {
        BarMaterialized = true;
        return BarAddr;
      };

      return {BarMaterializer, BarSym.getFlags()};
    }

    return nullptr;
  };

  auto RS =
      getResponsibilitySetWithLegacyFn(SymbolNameSet({Bar, Baz}), LegacyLookup);

  EXPECT_TRUE(!!RS) << "Expected getResponsibilitySetWithLegacyFn to succeed";
  EXPECT_EQ(RS->size(), 1U) << "Wrong number of symbols returned";
  EXPECT_EQ(RS->count(Bar), 1U) << "Incorrect responsibility set returned";
  EXPECT_FALSE(BarMaterialized)
      << "lookupFlags should not have materialized bar";

  bool OnCompletionRun = false;
  auto OnCompletion = [&](Expected<SymbolMap> Result) {
    OnCompletionRun = true;
    EXPECT_TRUE(!!Result) << "lookuWithLegacy failed to resolve";

    EXPECT_EQ(Result->size(), 2U) << "Wrong number of symbols resolved";
    EXPECT_EQ(Result->count(Foo), 1U) << "Result for foo missing";
    EXPECT_EQ(Result->count(Bar), 1U) << "Result for bar missing";
    EXPECT_EQ((*Result)[Foo].getAddress(), FooAddr) << "Wrong address for foo";
    EXPECT_EQ((*Result)[Foo].getFlags(), FooSym.getFlags())
        << "Wrong flags for foo";
    EXPECT_EQ((*Result)[Bar].getAddress(), BarAddr) << "Wrong address for bar";
    EXPECT_EQ((*Result)[Bar].getFlags(), BarSym.getFlags())
        << "Wrong flags for bar";
  };

  AsynchronousSymbolQuery Q(SymbolLookupSet({Foo, Bar}), SymbolState::Resolved,
                            OnCompletion);
  auto Unresolved =
      lookupWithLegacyFn(ES, Q, SymbolNameSet({Foo, Bar, Baz}), LegacyLookup);

  EXPECT_TRUE(OnCompletionRun) << "OnCompletion was not run";
  EXPECT_EQ(Unresolved.size(), 1U) << "Expected one unresolved symbol";
  EXPECT_EQ(Unresolved.count(Baz), 1U) << "Expected baz to be unresolved";
}

} // namespace
