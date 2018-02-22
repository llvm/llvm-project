//===----------- CoreAPIsTest.cpp - Unit tests for Core ORC APIs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "gtest/gtest.h"

#include <set>

using namespace llvm;
using namespace llvm::orc;

namespace {

class SimpleSource : public SymbolSource {
public:
  using MaterializeFunction = std::function<Error(VSO &, SymbolNameSet)>;
  using DiscardFunction = std::function<void(VSO &, SymbolStringPtr)>;

  SimpleSource(MaterializeFunction Materialize, DiscardFunction Discard)
      : Materialize(std::move(Materialize)), Discard(std::move(Discard)) {}

  Error materialize(VSO &V, SymbolNameSet Symbols) override {
    return Materialize(V, std::move(Symbols));
  }

  void discard(VSO &V, SymbolStringPtr Name) override {
    Discard(V, std::move(Name));
  }

private:
  MaterializeFunction Materialize;
  DiscardFunction Discard;
};

TEST(CoreAPIsTest, AsynchronousSymbolQuerySuccessfulResolutionOnly) {
  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  constexpr JITTargetAddress FakeAddr = 0xdeadbeef;
  SymbolNameSet Names({Foo});

  bool OnResolutionRun = false;
  bool OnReadyRun = false;
  auto OnResolution = [&](Expected<SymbolMap> Result) {
    EXPECT_TRUE(!!Result) << "Resolution unexpectedly returned error";
    auto I = Result->find(Foo);
    EXPECT_NE(I, Result->end()) << "Could not find symbol definition";
    EXPECT_EQ(I->second.getAddress(), FakeAddr)
        << "Resolution returned incorrect result";
    OnResolutionRun = true;
  };
  auto OnReady = [&](Error Err) {
    cantFail(std::move(Err));
    OnReadyRun = true;
  };

  AsynchronousSymbolQuery Q(Names, OnResolution, OnReady);

  Q.setDefinition(Foo, JITEvaluatedSymbol(FakeAddr, JITSymbolFlags::Exported));

  EXPECT_TRUE(OnResolutionRun) << "OnResolutionCallback was not run";
  EXPECT_FALSE(OnReadyRun) << "OnReady unexpectedly run";
}

TEST(CoreAPIsTest, AsynchronousSymbolQueryResolutionErrorOnly) {
  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  SymbolNameSet Names({Foo});

  bool OnResolutionRun = false;
  bool OnReadyRun = false;

  auto OnResolution = [&](Expected<SymbolMap> Result) {
    EXPECT_FALSE(!!Result) << "Resolution unexpectedly returned success";
    auto Msg = toString(Result.takeError());
    EXPECT_EQ(Msg, "xyz") << "Resolution returned incorrect result";
    OnResolutionRun = true;
  };
  auto OnReady = [&](Error Err) {
    cantFail(std::move(Err));
    OnReadyRun = true;
  };

  AsynchronousSymbolQuery Q(Names, OnResolution, OnReady);

  Q.setFailed(make_error<StringError>("xyz", inconvertibleErrorCode()));

  EXPECT_TRUE(OnResolutionRun) << "OnResolutionCallback was not run";
  EXPECT_FALSE(OnReadyRun) << "OnReady unexpectedly run";
}

TEST(CoreAPIsTest, SimpleAsynchronousSymbolQueryAgainstVSO) {
  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  constexpr JITTargetAddress FakeAddr = 0xdeadbeef;
  SymbolNameSet Names({Foo});

  bool OnResolutionRun = false;
  bool OnReadyRun = false;

  auto OnResolution = [&](Expected<SymbolMap> Result) {
    EXPECT_TRUE(!!Result) << "Query unexpectedly returned error";
    auto I = Result->find(Foo);
    EXPECT_NE(I, Result->end()) << "Could not find symbol definition";
    EXPECT_EQ(I->second.getAddress(), FakeAddr)
        << "Resolution returned incorrect result";
    OnResolutionRun = true;
  };

  auto OnReady = [&](Error Err) {
    cantFail(std::move(Err));
    OnReadyRun = true;
  };

  auto Q =
      std::make_shared<AsynchronousSymbolQuery>(Names, OnResolution, OnReady);
  VSO V;

  SymbolMap Defs;
  Defs[Foo] = JITEvaluatedSymbol(FakeAddr, JITSymbolFlags::Exported);
  cantFail(V.define(std::move(Defs)));
  V.lookup(Q, Names);

  EXPECT_TRUE(OnResolutionRun) << "OnResolutionCallback was not run";
  EXPECT_TRUE(OnReadyRun) << "OnReady was not run";
}

TEST(CoreAPIsTest, LookupFlagsTest) {

  // Test that lookupFlags works on a predefined symbol, and does not trigger
  // materialization of a lazy symbol.

  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  auto Bar = SP.intern("bar");
  auto Baz = SP.intern("baz");

  VSO V;

  auto Source = std::make_shared<SimpleSource>(
      [](VSO &V, SymbolNameSet Symbols) -> Error {
        llvm_unreachable("Symbol materialized on flags lookup");
      },
      [](VSO &V, SymbolStringPtr Name) -> Error {
        llvm_unreachable("Symbol finalized on flags lookup");
      });

  JITSymbolFlags FooFlags = JITSymbolFlags::Exported;
  JITSymbolFlags BarFlags = static_cast<JITSymbolFlags::FlagNames>(
      JITSymbolFlags::Exported | JITSymbolFlags::Weak);

  SymbolMap InitialDefs;
  InitialDefs[Foo] = JITEvaluatedSymbol(0xdeadbeef, FooFlags);
  cantFail(V.define(std::move(InitialDefs)));

  SymbolFlagsMap InitialLazyDefs({{Bar, BarFlags}});
  cantFail(V.defineLazy(InitialLazyDefs, Source));

  SymbolNameSet Names({Foo, Bar, Baz});

  SymbolFlagsMap SymbolFlags;
  auto SymbolsNotFound = V.lookupFlags(SymbolFlags, Names);

  EXPECT_EQ(SymbolsNotFound.size(), 1U) << "Expected one not-found symbol";
  EXPECT_EQ(SymbolsNotFound.count(Baz), 1U) << "Expected Baz to be not-found";
  EXPECT_EQ(SymbolFlags.size(), 2U)
      << "Returned symbol flags contains unexpected results";
  EXPECT_EQ(SymbolFlags.count(Foo), 1U) << "Missing lookupFlags result for Foo";
  EXPECT_EQ(SymbolFlags[Foo], FooFlags) << "Incorrect flags returned for Foo";
  EXPECT_EQ(SymbolFlags.count(Bar), 1U)
      << "Missing  lookupFlags result for Bar";
  EXPECT_EQ(SymbolFlags[Bar], BarFlags) << "Incorrect flags returned for Bar";
}

TEST(CoreAPIsTest, AddAndMaterializeLazySymbol) {

  constexpr JITTargetAddress FakeFooAddr = 0xdeadbeef;
  constexpr JITTargetAddress FakeBarAddr = 0xcafef00d;

  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  auto Bar = SP.intern("bar");

  bool FooMaterialized = false;
  bool BarDiscarded = false;

  VSO V;

  auto Source = std::make_shared<SimpleSource>(
      [&](VSO &V, SymbolNameSet Symbols) {
        EXPECT_EQ(Symbols.size(), 1U)
            << "Expected Symbols set size to be 1 ({ Foo })";
        EXPECT_EQ(*Symbols.begin(), Foo) << "Expected Symbols == { Foo }";

        SymbolMap SymbolsToResolve;
        SymbolsToResolve[Foo] =
            JITEvaluatedSymbol(FakeFooAddr, JITSymbolFlags::Exported);
        V.resolve(std::move(SymbolsToResolve));
        SymbolNameSet SymbolsToFinalize;
        SymbolsToFinalize.insert(Foo);
        V.finalize(SymbolsToFinalize);
        FooMaterialized = true;
        return Error::success();
      },
      [&](VSO &V, SymbolStringPtr Name) {
        EXPECT_EQ(Name, Bar) << "Expected Name to be Bar";
        BarDiscarded = true;
      });

  SymbolFlagsMap InitialSymbols(
      {{Foo, JITSymbolFlags::Exported},
       {Bar, static_cast<JITSymbolFlags::FlagNames>(JITSymbolFlags::Exported |
                                                    JITSymbolFlags::Weak)}});
  cantFail(V.defineLazy(InitialSymbols, Source));

  SymbolMap BarOverride;
  BarOverride[Bar] = JITEvaluatedSymbol(FakeBarAddr, JITSymbolFlags::Exported);
  cantFail(V.define(std::move(BarOverride)));

  SymbolNameSet Names({Foo});

  bool OnResolutionRun = false;
  bool OnReadyRun = false;

  auto OnResolution = [&](Expected<SymbolMap> Result) {
    EXPECT_TRUE(!!Result) << "Resolution unexpectedly returned error";
    auto I = Result->find(Foo);
    EXPECT_NE(I, Result->end()) << "Could not find symbol definition";
    EXPECT_EQ(I->second.getAddress(), FakeFooAddr)
        << "Resolution returned incorrect result";
    OnResolutionRun = true;
  };

  auto OnReady = [&](Error Err) {
    cantFail(std::move(Err));
    OnReadyRun = true;
  };

  auto Q =
      std::make_shared<AsynchronousSymbolQuery>(Names, OnResolution, OnReady);

  auto LR = V.lookup(std::move(Q), Names);

  for (auto &SWKV : LR.MaterializationWork)
    cantFail(SWKV.first->materialize(V, std::move(SWKV.second)));

  EXPECT_TRUE(LR.UnresolvedSymbols.empty()) << "Could not find Foo in dylib";
  EXPECT_TRUE(FooMaterialized) << "Foo was not materialized";
  EXPECT_TRUE(BarDiscarded) << "Bar was not discarded";
  EXPECT_TRUE(OnResolutionRun) << "OnResolutionCallback was not run";
  EXPECT_TRUE(OnReadyRun) << "OnReady was not run";
}

TEST(CoreAPIsTest, TestLambdaSymbolResolver) {
  JITEvaluatedSymbol FooSym(0xdeadbeef, JITSymbolFlags::Exported);
  JITEvaluatedSymbol BarSym(0xcafef00d, JITSymbolFlags::Exported);

  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  auto Bar = SP.intern("bar");
  auto Baz = SP.intern("baz");

  VSO V;
  cantFail(V.define({{Foo, FooSym}, {Bar, BarSym}}));

  auto Resolver = createSymbolResolver(
      [&](SymbolFlagsMap &SymbolFlags, const SymbolNameSet &Symbols) {
        return V.lookupFlags(SymbolFlags, Symbols);
      },
      [&](std::shared_ptr<AsynchronousSymbolQuery> Q, SymbolNameSet Symbols) {
        auto LR = V.lookup(std::move(Q), Symbols);
        assert(LR.MaterializationWork.empty() &&
               "Test generated unexpected materialization "
               "work?");
        return std::move(LR.UnresolvedSymbols);
      });

  SymbolNameSet Symbols({Foo, Bar, Baz});

  SymbolFlagsMap SymbolFlags;
  SymbolNameSet SymbolsNotFound = Resolver->lookupFlags(SymbolFlags, Symbols);

  EXPECT_EQ(SymbolFlags.size(), 2U)
      << "lookupFlags returned the wrong number of results";
  EXPECT_EQ(SymbolFlags.count(Foo), 1U) << "Missing lookupFlags result for foo";
  EXPECT_EQ(SymbolFlags.count(Bar), 1U) << "Missing lookupFlags result for bar";
  EXPECT_EQ(SymbolFlags[Foo], FooSym.getFlags())
      << "Incorrect lookupFlags result for Foo";
  EXPECT_EQ(SymbolFlags[Bar], BarSym.getFlags())
      << "Incorrect lookupFlags result for Bar";
  EXPECT_EQ(SymbolsNotFound.size(), 1U)
      << "Expected one symbol not found in lookupFlags";
  EXPECT_EQ(SymbolsNotFound.count(Baz), 1U)
      << "Expected baz not to be found in lookupFlags";

  bool OnResolvedRun = false;

  auto OnResolved = [&](Expected<SymbolMap> Result) {
    OnResolvedRun = true;
    EXPECT_TRUE(!!Result) << "Unexpected error";
    EXPECT_EQ(Result->size(), 2U) << "Unexpected number of resolved symbols";
    EXPECT_EQ(Result->count(Foo), 1U) << "Missing lookup result for foo";
    EXPECT_EQ(Result->count(Bar), 1U) << "Missing lookup result for bar";
    EXPECT_EQ((*Result)[Foo].getAddress(), FooSym.getAddress())
        << "Incorrect address for foo";
    EXPECT_EQ((*Result)[Bar].getAddress(), BarSym.getAddress())
        << "Incorrect address for bar";
  };
  auto OnReady = [&](Error Err) {
    EXPECT_FALSE(!!Err) << "Finalization should never fail in this test";
  };

  auto Q = std::make_shared<AsynchronousSymbolQuery>(SymbolNameSet({Foo, Bar}),
                                                     OnResolved, OnReady);
  auto Unresolved = Resolver->lookup(std::move(Q), Symbols);

  EXPECT_EQ(Unresolved.size(), 1U) << "Expected one unresolved symbol";
  EXPECT_EQ(Unresolved.count(Baz), 1U) << "Expected baz to not be resolved";
  EXPECT_TRUE(OnResolvedRun) << "OnResolved was never run";
}

} // namespace
