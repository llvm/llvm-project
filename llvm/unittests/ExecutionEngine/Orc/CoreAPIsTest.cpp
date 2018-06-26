//===----------- CoreAPIsTest.cpp - Unit tests for Core ORC APIs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"
#include "gtest/gtest.h"

#include <set>
#include <thread>

using namespace llvm;
using namespace llvm::orc;

namespace {

class SimpleMaterializationUnit : public MaterializationUnit {
public:
  using MaterializeFunction =
      std::function<void(MaterializationResponsibility)>;
  using DiscardFunction = std::function<void(const VSO &, SymbolStringPtr)>;
  using DestructorFunction = std::function<void()>;

  SimpleMaterializationUnit(
      SymbolFlagsMap SymbolFlags, MaterializeFunction Materialize,
      DiscardFunction Discard = DiscardFunction(),
      DestructorFunction Destructor = DestructorFunction())
      : MaterializationUnit(std::move(SymbolFlags)),
        Materialize(std::move(Materialize)), Discard(std::move(Discard)),
        Destructor(std::move(Destructor)) {}

  ~SimpleMaterializationUnit() override {
    if (Destructor)
      Destructor();
  }

  void materialize(MaterializationResponsibility R) override {
    Materialize(std::move(R));
  }

  void discard(const VSO &V, SymbolStringPtr Name) override {
    if (Discard)
      Discard(V, std::move(Name));
    else
      llvm_unreachable("Discard not supported");
  }

private:
  MaterializeFunction Materialize;
  DiscardFunction Discard;
  DestructorFunction Destructor;
};

TEST(CoreAPIsTest, AsynchronousSymbolQuerySuccessfulResolutionOnly) {
  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  constexpr JITTargetAddress FakeAddr = 0xdeadbeef;
  SymbolNameSet Names({Foo});

  bool OnResolutionRun = false;
  bool OnReadyRun = false;
  auto OnResolution =
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> Result) {
        EXPECT_TRUE(!!Result) << "Resolution unexpectedly returned error";
        auto &Resolved = Result->Symbols;
        auto I = Resolved.find(Foo);
        EXPECT_NE(I, Resolved.end()) << "Could not find symbol definition";
        EXPECT_EQ(I->second.getAddress(), FakeAddr)
            << "Resolution returned incorrect result";
        OnResolutionRun = true;
      };
  auto OnReady = [&](Error Err) {
    cantFail(std::move(Err));
    OnReadyRun = true;
  };

  AsynchronousSymbolQuery Q(Names, OnResolution, OnReady);

  Q.resolve(Foo, JITEvaluatedSymbol(FakeAddr, JITSymbolFlags::Exported));

  EXPECT_TRUE(Q.isFullyResolved()) << "Expected query to be fully resolved";

  if (!Q.isFullyResolved())
    return;

  Q.handleFullyResolved();

  EXPECT_TRUE(OnResolutionRun) << "OnResolutionCallback was not run";
  EXPECT_FALSE(OnReadyRun) << "OnReady unexpectedly run";
}

TEST(CoreAPIsTest, ExecutionSessionFailQuery) {
  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");
  SymbolNameSet Names({Foo});

  bool OnResolutionRun = false;
  bool OnReadyRun = false;

  auto OnResolution =
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> Result) {
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

  ES.failQuery(Q, make_error<StringError>("xyz", inconvertibleErrorCode()));

  EXPECT_TRUE(OnResolutionRun) << "OnResolutionCallback was not run";
  EXPECT_FALSE(OnReadyRun) << "OnReady unexpectedly run";
}

TEST(CoreAPIsTest, SimpleAsynchronousSymbolQueryAgainstVSO) {
  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");
  constexpr JITTargetAddress FakeAddr = 0xdeadbeef;
  SymbolNameSet Names({Foo});

  bool OnResolutionRun = false;
  bool OnReadyRun = false;

  auto OnResolution =
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> Result) {
        EXPECT_TRUE(!!Result) << "Query unexpectedly returned error";
        auto &Resolved = Result->Symbols;
        auto I = Resolved.find(Foo);
        EXPECT_NE(I, Resolved.end()) << "Could not find symbol definition";
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
  auto &V = ES.createVSO("V");

  auto Defs = absoluteSymbols(
      {{Foo, JITEvaluatedSymbol(FakeAddr, JITSymbolFlags::Exported)}});
  cantFail(V.define(Defs));
  assert(Defs == nullptr && "Defs should have been accepted");
  V.lookup(Q, Names);

  EXPECT_TRUE(OnResolutionRun) << "OnResolutionCallback was not run";
  EXPECT_TRUE(OnReadyRun) << "OnReady was not run";
}

TEST(CoreAPIsTest, EmptyVSOAndQueryLookup) {
  ExecutionSession ES;
  auto &V = ES.createVSO("V");

  bool OnResolvedRun = false;
  bool OnReadyRun = false;

  auto Q = std::make_shared<AsynchronousSymbolQuery>(
      SymbolNameSet(),
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> RR) {
        cantFail(std::move(RR));
        OnResolvedRun = true;
      },
      [&](Error Err) {
        cantFail(std::move(Err));
        OnReadyRun = true;
      });

  V.lookup(std::move(Q), {});

  EXPECT_TRUE(OnResolvedRun) << "OnResolved was not run for empty query";
  EXPECT_TRUE(OnReadyRun) << "OnReady was not run for empty query";
}

TEST(CoreAPIsTest, ChainedVSOLookup) {
  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto FooSym = JITEvaluatedSymbol(1U, JITSymbolFlags::Exported);

  auto &V1 = ES.createVSO("V1");
  cantFail(V1.define(absoluteSymbols({{Foo, FooSym}})));

  auto &V2 = ES.createVSO("V2");

  bool OnResolvedRun = false;
  bool OnReadyRun = false;

  auto Q = std::make_shared<AsynchronousSymbolQuery>(
      SymbolNameSet({Foo}),
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> RR) {
        cantFail(std::move(RR));
        OnResolvedRun = true;
      },
      [&](Error Err) {
        cantFail(std::move(Err));
        OnReadyRun = true;
      });

  V2.lookup(Q, V1.lookup(Q, {Foo}));

  EXPECT_TRUE(OnResolvedRun) << "OnResolved was not run for empty query";
  EXPECT_TRUE(OnReadyRun) << "OnReady was not run for empty query";
}

TEST(CoreAPIsTest, LookupFlagsTest) {

  // Test that lookupFlags works on a predefined symbol, and does not trigger
  // materialization of a lazy symbol.

  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto Bar = ES.getSymbolStringPool().intern("bar");
  auto Baz = ES.getSymbolStringPool().intern("baz");

  JITSymbolFlags FooFlags = JITSymbolFlags::Exported;
  JITSymbolFlags BarFlags = static_cast<JITSymbolFlags::FlagNames>(
      JITSymbolFlags::Exported | JITSymbolFlags::Weak);

  VSO &V = ES.createVSO("V");

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarFlags}}),
      [](MaterializationResponsibility R) {
        llvm_unreachable("Symbol materialized on flags lookup");
      });

  cantFail(V.define(
      absoluteSymbols({{Foo, JITEvaluatedSymbol(0xdeadbeef, FooFlags)}})));
  cantFail(V.define(std::move(MU)));

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

TEST(CoreAPIsTest, TestAliases) {
  ExecutionSession ES;
  auto &V = ES.createVSO("V");

  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto FooSym = JITEvaluatedSymbol(1U, JITSymbolFlags::Exported);
  auto Bar = ES.getSymbolStringPool().intern("bar");
  auto BarSym = JITEvaluatedSymbol(2U, JITSymbolFlags::Exported);

  auto Baz = ES.getSymbolStringPool().intern("baz");
  auto Qux = ES.getSymbolStringPool().intern("qux");

  auto QuxSym = JITEvaluatedSymbol(3U, JITSymbolFlags::Exported);

  cantFail(V.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));
  cantFail(V.define(symbolAliases({{Baz, {Foo, JITSymbolFlags::Exported}},
                                   {Qux, {Bar, JITSymbolFlags::Weak}}})));
  cantFail(V.define(absoluteSymbols({{Qux, QuxSym}})));

  auto Result = lookup({&V}, {Baz, Qux});
  EXPECT_TRUE(!!Result) << "Unexpected lookup failure";
  EXPECT_EQ(Result->count(Baz), 1U) << "No result for \"baz\"";
  EXPECT_EQ(Result->count(Qux), 1U) << "No result for \"qux\"";
  EXPECT_EQ((*Result)[Baz].getAddress(), FooSym.getAddress())
      << "\"Baz\"'s address should match \"Foo\"'s";
  EXPECT_EQ((*Result)[Qux].getAddress(), QuxSym.getAddress())
      << "The \"Qux\" alias should have been overriden";
}

TEST(CoreAPIsTest, TestTrivialCircularDependency) {
  ExecutionSession ES;

  auto &V = ES.createVSO("V");

  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto FooFlags = JITSymbolFlags::Exported;
  auto FooSym = JITEvaluatedSymbol(1U, FooFlags);

  Optional<MaterializationResponsibility> FooR;
  auto FooMU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooFlags}}),
      [&](MaterializationResponsibility R) { FooR.emplace(std::move(R)); });

  cantFail(V.define(FooMU));

  bool FooReady = false;
  auto Q =
    std::make_shared<AsynchronousSymbolQuery>(
      SymbolNameSet({ Foo }),
      [](Expected<AsynchronousSymbolQuery::ResolutionResult> R) {
        cantFail(std::move(R));
      },
      [&](Error Err) {
        cantFail(std::move(Err));
        FooReady = true;
      });

  V.lookup(std::move(Q), { Foo });

  FooR->addDependencies({{&V, {Foo}}});
  FooR->resolve({{Foo, FooSym}});
  FooR->finalize();

  EXPECT_TRUE(FooReady)
    << "Self-dependency prevented symbol from being marked ready";
}

TEST(CoreAPIsTest, TestCircularDependenceInOneVSO) {

  ExecutionSession ES;

  auto &V = ES.createVSO("V");

  // Create three symbols: Foo, Bar and Baz.
  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto FooFlags = JITSymbolFlags::Exported;
  auto FooSym = JITEvaluatedSymbol(1U, FooFlags);

  auto Bar = ES.getSymbolStringPool().intern("bar");
  auto BarFlags = JITSymbolFlags::Exported;
  auto BarSym = JITEvaluatedSymbol(2U, BarFlags);

  auto Baz = ES.getSymbolStringPool().intern("baz");
  auto BazFlags = JITSymbolFlags::Exported;
  auto BazSym = JITEvaluatedSymbol(3U, BazFlags);

  // Create three MaterializationResponsibility objects: one for each symbol
  // (these are optional because MaterializationResponsibility does not have
  // a default constructor).
  Optional<MaterializationResponsibility> FooR;
  Optional<MaterializationResponsibility> BarR;
  Optional<MaterializationResponsibility> BazR;

  // Create a MaterializationUnit for each symbol that moves the
  // MaterializationResponsibility into one of the locals above.
  auto FooMU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooFlags}}),
      [&](MaterializationResponsibility R) { FooR.emplace(std::move(R)); });

  auto BarMU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarFlags}}),
      [&](MaterializationResponsibility R) { BarR.emplace(std::move(R)); });

  auto BazMU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Baz, BazFlags}}),
      [&](MaterializationResponsibility R) { BazR.emplace(std::move(R)); });

  // Define the symbols.
  cantFail(V.define(FooMU));
  cantFail(V.define(BarMU));
  cantFail(V.define(BazMU));

  // Query each of the symbols to trigger materialization.
  bool FooResolved = false;
  bool FooReady = false;
  auto FooQ = std::make_shared<AsynchronousSymbolQuery>(
      SymbolNameSet({Foo}),
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> RR) {
        cantFail(std::move(RR));
        FooResolved = true;
      },
      [&](Error Err) {
        cantFail(std::move(Err));
        FooReady = true;
      });
  {
    auto Unresolved = V.lookup(FooQ, {Foo});
    EXPECT_TRUE(Unresolved.empty()) << "Failed to resolve \"Foo\"";
  }

  bool BarResolved = false;
  bool BarReady = false;
  auto BarQ = std::make_shared<AsynchronousSymbolQuery>(
      SymbolNameSet({Bar}),
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> RR) {
        cantFail(std::move(RR));
        BarResolved = true;
      },
      [&](Error Err) {
        cantFail(std::move(Err));
        BarReady = true;
      });
  {
    auto Unresolved = V.lookup(BarQ, {Bar});
    EXPECT_TRUE(Unresolved.empty()) << "Failed to resolve \"Bar\"";
  }

  bool BazResolved = false;
  bool BazReady = false;
  auto BazQ = std::make_shared<AsynchronousSymbolQuery>(
      SymbolNameSet({Baz}),
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> RR) {
        cantFail(std::move(RR));
        BazResolved = true;
      },
      [&](Error Err) {
        cantFail(std::move(Err));
        BazReady = true;
      });
  {
    auto Unresolved = V.lookup(BazQ, {Baz});
    EXPECT_TRUE(Unresolved.empty()) << "Failed to resolve \"Baz\"";
  }

  // Add a circular dependency: Foo -> Bar, Bar -> Baz, Baz -> Foo.
  FooR->addDependencies({{&V, SymbolNameSet({Bar})}});
  BarR->addDependencies({{&V, SymbolNameSet({Baz})}});
  BazR->addDependencies({{&V, SymbolNameSet({Foo})}});

  // Add self-dependencies for good measure. This tests that the implementation
  // of addDependencies filters these out.
  FooR->addDependencies({{&V, SymbolNameSet({Foo})}});
  BarR->addDependencies({{&V, SymbolNameSet({Bar})}});
  BazR->addDependencies({{&V, SymbolNameSet({Baz})}});

  EXPECT_FALSE(FooResolved) << "\"Foo\" should not be resolved yet";
  EXPECT_FALSE(BarResolved) << "\"Bar\" should not be resolved yet";
  EXPECT_FALSE(BazResolved) << "\"Baz\" should not be resolved yet";

  FooR->resolve({{Foo, FooSym}});
  BarR->resolve({{Bar, BarSym}});
  BazR->resolve({{Baz, BazSym}});

  EXPECT_TRUE(FooResolved) << "\"Foo\" should be resolved now";
  EXPECT_TRUE(BarResolved) << "\"Bar\" should be resolved now";
  EXPECT_TRUE(BazResolved) << "\"Baz\" should be resolved now";

  EXPECT_FALSE(FooReady) << "\"Foo\" should not be ready yet";
  EXPECT_FALSE(BarReady) << "\"Bar\" should not be ready yet";
  EXPECT_FALSE(BazReady) << "\"Baz\" should not be ready yet";

  FooR->finalize();
  BarR->finalize();

  // Verify that nothing is ready until the circular dependence is resolved.

  EXPECT_FALSE(FooReady) << "\"Foo\" still should not be ready";
  EXPECT_FALSE(BarReady) << "\"Bar\" still should not be ready";
  EXPECT_FALSE(BazReady) << "\"Baz\" still should not be ready";

  BazR->finalize();

  // Verify that everything becomes ready once the circular dependence resolved.
  EXPECT_TRUE(FooReady) << "\"Foo\" should be ready now";
  EXPECT_TRUE(BarReady) << "\"Bar\" should be ready now";
  EXPECT_TRUE(BazReady) << "\"Baz\" should be ready now";
}

TEST(CoreAPIsTest, DropMaterializerWhenEmpty) {
  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto Bar = ES.getSymbolStringPool().intern("bar");

  bool DestructorRun = false;

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap(
          {{Foo, JITSymbolFlags::Weak}, {Bar, JITSymbolFlags::Weak}}),
      [](MaterializationResponsibility R) {
        llvm_unreachable("Unexpected call to materialize");
      },
      [&](const VSO &V, SymbolStringPtr Name) {
        EXPECT_TRUE(Name == Foo || Name == Bar)
            << "Discard of unexpected symbol?";
      },
      [&]() { DestructorRun = true; });

  auto &V = ES.createVSO("V");

  cantFail(V.define(MU));

  auto FooSym = JITEvaluatedSymbol(1, JITSymbolFlags::Exported);
  auto BarSym = JITEvaluatedSymbol(2, JITSymbolFlags::Exported);
  cantFail(V.define(absoluteSymbols({{Foo, FooSym}})));

  EXPECT_FALSE(DestructorRun)
      << "MaterializationUnit should not have been destroyed yet";

  cantFail(V.define(absoluteSymbols({{Bar, BarSym}})));

  EXPECT_TRUE(DestructorRun)
      << "MaterializationUnit should have been destroyed";
}

TEST(CoreAPIsTest, AddAndMaterializeLazySymbol) {

  constexpr JITTargetAddress FakeFooAddr = 0xdeadbeef;
  constexpr JITTargetAddress FakeBarAddr = 0xcafef00d;

  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto Bar = ES.getSymbolStringPool().intern("bar");

  bool FooMaterialized = false;
  bool BarDiscarded = false;

  auto &V = ES.createVSO("V");

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap(
          {{Foo, JITSymbolFlags::Exported},
           {Bar, static_cast<JITSymbolFlags::FlagNames>(
                     JITSymbolFlags::Exported | JITSymbolFlags::Weak)}}),
      [&](MaterializationResponsibility R) {
        assert(BarDiscarded && "Bar should have been discarded by this point");
        SymbolMap SymbolsToResolve;
        SymbolsToResolve[Foo] =
            JITEvaluatedSymbol(FakeFooAddr, JITSymbolFlags::Exported);
        R.resolve(std::move(SymbolsToResolve));
        R.finalize();
        FooMaterialized = true;
      },
      [&](const VSO &V, SymbolStringPtr Name) {
        EXPECT_EQ(Name, Bar) << "Expected Name to be Bar";
        BarDiscarded = true;
      });

  cantFail(V.define(MU));

  ;
  cantFail(V.define(absoluteSymbols(
      {{Bar, JITEvaluatedSymbol(FakeBarAddr, JITSymbolFlags::Exported)}})));

  SymbolNameSet Names({Foo});

  bool OnResolutionRun = false;
  bool OnReadyRun = false;

  auto OnResolution =
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> Result) {
        EXPECT_TRUE(!!Result) << "Resolution unexpectedly returned error";
        auto I = Result->Symbols.find(Foo);
        EXPECT_NE(I, Result->Symbols.end())
            << "Could not find symbol definition";
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

  auto Unresolved = V.lookup(std::move(Q), Names);

  EXPECT_TRUE(Unresolved.empty()) << "Could not find Foo in dylib";
  EXPECT_TRUE(FooMaterialized) << "Foo was not materialized";
  EXPECT_TRUE(BarDiscarded) << "Bar was not discarded";
  EXPECT_TRUE(OnResolutionRun) << "OnResolutionCallback was not run";
  EXPECT_TRUE(OnReadyRun) << "OnReady was not run";
}

TEST(CoreAPIsTest, DefineMaterializingSymbol) {
  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto Bar = ES.getSymbolStringPool().intern("bar");

  auto FooSym = JITEvaluatedSymbol(1, JITSymbolFlags::Exported);
  auto BarSym = JITEvaluatedSymbol(2, JITSymbolFlags::Exported);

  bool ExpectNoMoreMaterialization = false;
  ES.setDispatchMaterialization(
      [&](VSO &V, std::unique_ptr<MaterializationUnit> MU) {
        if (ExpectNoMoreMaterialization)
          ADD_FAILURE() << "Unexpected materialization";
        MU->doMaterialize(V);
      });

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](MaterializationResponsibility R) {
        cantFail(
            R.defineMaterializing(SymbolFlagsMap({{Bar, BarSym.getFlags()}})));
        R.resolve(SymbolMap({{Foo, FooSym}, {Bar, BarSym}}));
        R.finalize();
      });

  auto &V = ES.createVSO("V");
  cantFail(V.define(MU));

  auto OnResolution1 =
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> Result) {
        cantFail(std::move(Result));
      };

  auto OnReady1 = [](Error Err) { cantFail(std::move(Err)); };

  auto Q1 = std::make_shared<AsynchronousSymbolQuery>(SymbolNameSet({Foo}),
                                                      OnResolution1, OnReady1);

  V.lookup(std::move(Q1), {Foo});

  bool BarResolved = false;
  auto OnResolution2 =
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> Result) {
        auto R = cantFail(std::move(Result));
        EXPECT_EQ(R.Symbols.size(), 1U) << "Expected to resolve one symbol";
        EXPECT_EQ(R.Symbols.count(Bar), 1U) << "Expected to resolve 'Bar'";
        EXPECT_EQ(R.Symbols[Bar].getAddress(), BarSym.getAddress())
            << "Expected Bar == BarSym";
        BarResolved = true;
      };

  auto OnReady2 = [](Error Err) { cantFail(std::move(Err)); };

  auto Q2 = std::make_shared<AsynchronousSymbolQuery>(SymbolNameSet({Bar}),
                                                      OnResolution2, OnReady2);

  ExpectNoMoreMaterialization = true;
  V.lookup(std::move(Q2), {Bar});

  EXPECT_TRUE(BarResolved) << "Bar should have been resolved";
}

TEST(CoreAPIsTest, FallbackDefinitionGeneratorTest) {
  constexpr JITTargetAddress FakeFooAddr = 0xdeadbeef;
  constexpr JITTargetAddress FakeBarAddr = 0xcafef00d;

  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto Bar = ES.getSymbolStringPool().intern("bar");

  auto FooSym = JITEvaluatedSymbol(FakeFooAddr, JITSymbolFlags::Exported);
  auto BarSym = JITEvaluatedSymbol(FakeBarAddr, JITSymbolFlags::Exported);

  auto &V = ES.createVSO("V");

  cantFail(V.define(absoluteSymbols({{Foo, FooSym}})));

  V.setFallbackDefinitionGenerator([&](VSO &W, const SymbolNameSet &Names) {
    cantFail(W.define(absoluteSymbols({{Bar, BarSym}})));
    return SymbolNameSet({Bar});
  });

  auto Result = cantFail(lookup({&V}, {Foo, Bar}));

  EXPECT_EQ(Result.count(Bar), 1U) << "Expected to find fallback def for 'bar'";
  EXPECT_EQ(Result[Bar].getAddress(), FakeBarAddr)
      << "Expected address of fallback def for 'bar' to be " << FakeBarAddr;
}

TEST(CoreAPIsTest, FailResolution) {
  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto Bar = ES.getSymbolStringPool().intern("bar");

  SymbolNameSet Names({Foo, Bar});

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap(
          {{Foo, JITSymbolFlags::Weak}, {Bar, JITSymbolFlags::Weak}}),
      [&](MaterializationResponsibility R) { R.failMaterialization(); });

  auto &V = ES.createVSO("V");

  cantFail(V.define(MU));

  auto OnResolution =
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> Result) {
        handleAllErrors(Result.takeError(),
                        [&](FailedToMaterialize &F) {
                          EXPECT_EQ(F.getSymbols(), Names)
                              << "Expected to fail on symbols in Names";
                        },
                        [](ErrorInfoBase &EIB) {
                          std::string ErrMsg;
                          {
                            raw_string_ostream ErrOut(ErrMsg);
                            EIB.log(ErrOut);
                          }
                          ADD_FAILURE()
                              << "Expected a FailedToResolve error. Got:\n"
                              << ErrMsg;
                        });
      };

  auto OnReady = [](Error Err) {
    cantFail(std::move(Err));
    ADD_FAILURE() << "OnReady should never be called";
  };

  auto Q =
      std::make_shared<AsynchronousSymbolQuery>(Names, OnResolution, OnReady);

  V.lookup(std::move(Q), Names);
}

TEST(CoreAPIsTest, TestLambdaSymbolResolver) {
  JITEvaluatedSymbol FooSym(0xdeadbeef, JITSymbolFlags::Exported);
  JITEvaluatedSymbol BarSym(0xcafef00d, JITSymbolFlags::Exported);

  ExecutionSession ES;

  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto Bar = ES.getSymbolStringPool().intern("bar");
  auto Baz = ES.getSymbolStringPool().intern("baz");

  auto &V = ES.createVSO("V");
  cantFail(V.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));

  auto Resolver = createSymbolResolver(
      [&](SymbolFlagsMap &SymbolFlags, const SymbolNameSet &Symbols) {
        return V.lookupFlags(SymbolFlags, Symbols);
      },
      [&](std::shared_ptr<AsynchronousSymbolQuery> Q, SymbolNameSet Symbols) {
        return V.lookup(std::move(Q), Symbols);
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

  auto OnResolved =
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> Result) {
        OnResolvedRun = true;
        EXPECT_TRUE(!!Result) << "Unexpected error";
        EXPECT_EQ(Result->Symbols.size(), 2U)
            << "Unexpected number of resolved symbols";
        EXPECT_EQ(Result->Symbols.count(Foo), 1U)
            << "Missing lookup result for foo";
        EXPECT_EQ(Result->Symbols.count(Bar), 1U)
            << "Missing lookup result for bar";
        EXPECT_EQ(Result->Symbols[Foo].getAddress(), FooSym.getAddress())
            << "Incorrect address for foo";
        EXPECT_EQ(Result->Symbols[Bar].getAddress(), BarSym.getAddress())
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

TEST(CoreAPIsTest, TestLookupWithUnthreadedMaterialization) {
  constexpr JITTargetAddress FakeFooAddr = 0xdeadbeef;
  JITEvaluatedSymbol FooSym(FakeFooAddr, JITSymbolFlags::Exported);

  ExecutionSession ES(std::make_shared<SymbolStringPool>());
  auto Foo = ES.getSymbolStringPool().intern("foo");

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, JITSymbolFlags::Exported}}),
      [&](MaterializationResponsibility R) {
        R.resolve({{Foo, FooSym}});
        R.finalize();
      });

  auto &V = ES.createVSO("V");

  cantFail(V.define(MU));

  auto FooLookupResult = cantFail(lookup({&V}, Foo));

  EXPECT_EQ(FooLookupResult.getAddress(), FooSym.getAddress())
      << "lookup returned an incorrect address";
  EXPECT_EQ(FooLookupResult.getFlags(), FooSym.getFlags())
      << "lookup returned incorrect flags";
}

TEST(CoreAPIsTest, TestLookupWithThreadedMaterialization) {
#if LLVM_ENABLE_THREADS
  constexpr JITTargetAddress FakeFooAddr = 0xdeadbeef;
  JITEvaluatedSymbol FooSym(FakeFooAddr, JITSymbolFlags::Exported);

  ExecutionSession ES(std::make_shared<SymbolStringPool>());

  std::thread MaterializationThread;
  ES.setDispatchMaterialization(
      [&](VSO &V, std::unique_ptr<MaterializationUnit> MU) {
        auto SharedMU = std::shared_ptr<MaterializationUnit>(std::move(MU));
        MaterializationThread =
            std::thread([SharedMU, &V]() { SharedMU->doMaterialize(V); });
      });
  auto Foo = ES.getSymbolStringPool().intern("foo");

  auto &V = ES.createVSO("V");
  cantFail(V.define(absoluteSymbols({{Foo, FooSym}})));

  auto FooLookupResult = cantFail(lookup({&V}, Foo));

  EXPECT_EQ(FooLookupResult.getAddress(), FooSym.getAddress())
      << "lookup returned an incorrect address";
  EXPECT_EQ(FooLookupResult.getFlags(), FooSym.getFlags())
      << "lookup returned incorrect flags";
  MaterializationThread.join();
#endif
}

TEST(CoreAPIsTest, TestGetRequestedSymbolsAndDelegate) {
  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto Bar = ES.getSymbolStringPool().intern("bar");

  JITEvaluatedSymbol FooSym(0xdeadbeef, JITSymbolFlags::Exported);
  JITEvaluatedSymbol BarSym(0xcafef00d, JITSymbolFlags::Exported);

  SymbolNameSet Names({Foo, Bar});

  bool FooMaterialized = false;
  bool BarMaterialized = false;

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}, {Bar, BarSym.getFlags()}}),
      [&](MaterializationResponsibility R) {
        auto Requested = R.getRequestedSymbols();
        EXPECT_EQ(Requested.size(), 1U) << "Expected one symbol requested";
        EXPECT_EQ(*Requested.begin(), Foo) << "Expected \"Foo\" requested";

        auto NewMU = llvm::make_unique<SimpleMaterializationUnit>(
            SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
            [&](MaterializationResponsibility R2) {
              R2.resolve(SymbolMap({{Bar, BarSym}}));
              R2.finalize();
              BarMaterialized = true;
            });

        R.delegate(std::move(NewMU));

        R.resolve(SymbolMap({{Foo, FooSym}}));
        R.finalize();

        FooMaterialized = true;
      });

  auto &V = ES.createVSO("V");

  cantFail(V.define(MU));

  EXPECT_FALSE(FooMaterialized) << "Foo should not be materialized yet";
  EXPECT_FALSE(BarMaterialized) << "Bar should not be materialized yet";

  auto FooSymResult = cantFail(lookup({&V}, Foo));
  EXPECT_EQ(FooSymResult.getAddress(), FooSym.getAddress())
      << "Address mismatch for Foo";

  EXPECT_TRUE(FooMaterialized) << "Foo should be materialized now";
  EXPECT_FALSE(BarMaterialized) << "Bar still should not be materialized";

  auto BarSymResult = cantFail(lookup({&V}, Bar));
  EXPECT_EQ(BarSymResult.getAddress(), BarSym.getAddress())
      << "Address mismatch for Bar";
  EXPECT_TRUE(BarMaterialized) << "Bar should be materialized now";
}

TEST(CoreAPIsTest, TestMaterializeWeakSymbol) {
  // Confirm that once a weak definition is selected for materialization it is
  // treated as strong.

  constexpr JITTargetAddress FakeFooAddr = 0xdeadbeef;
  JITSymbolFlags FooFlags = JITSymbolFlags::Exported;
  FooFlags &= JITSymbolFlags::Weak;
  auto FooSym = JITEvaluatedSymbol(FakeFooAddr, FooFlags);

  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");

  auto &V = ES.createVSO("V");

  std::unique_ptr<MaterializationResponsibility> FooResponsibility;
  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooFlags}}), [&](MaterializationResponsibility R) {
        FooResponsibility =
            llvm::make_unique<MaterializationResponsibility>(std::move(R));
      });

  cantFail(V.define(MU));
  auto Q = std::make_shared<AsynchronousSymbolQuery>(
      SymbolNameSet({Foo}),
      [](Expected<AsynchronousSymbolQuery::ResolutionResult> R) {
        cantFail(std::move(R));
      },
      [](Error Err) { cantFail(std::move(Err)); });
  V.lookup(std::move(Q), SymbolNameSet({Foo}));

  auto MU2 = llvm::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, JITSymbolFlags::Exported}}),
      [](MaterializationResponsibility R) {
        llvm_unreachable("This unit should never be materialized");
      });

  auto Err = V.define(MU2);
  EXPECT_TRUE(!!Err) << "Expected failure value";
  EXPECT_TRUE(Err.isA<DuplicateDefinition>())
      << "Expected a duplicate definition error";
  consumeError(std::move(Err));

  FooResponsibility->resolve(SymbolMap({{Foo, FooSym}}));
  FooResponsibility->finalize();
}

} // namespace
