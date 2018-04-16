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
#include <thread>

using namespace llvm;
using namespace llvm::orc;

namespace {

class SimpleMaterializationUnit : public MaterializationUnit {
public:
  using GetSymbolsFunction = std::function<SymbolFlagsMap()>;
  using MaterializeFunction =
      std::function<void(MaterializationResponsibility)>;
  using DiscardFunction = std::function<void(const VSO &, SymbolStringPtr)>;
  using DestructorFunction = std::function<void()>;

  SimpleMaterializationUnit(
      GetSymbolsFunction GetSymbols, MaterializeFunction Materialize,
      DiscardFunction Discard,
      DestructorFunction Destructor = DestructorFunction())
      : GetSymbols(std::move(GetSymbols)), Materialize(std::move(Materialize)),
        Discard(std::move(Discard)), Destructor(std::move(Destructor)) {}

  ~SimpleMaterializationUnit() override {
    if (Destructor)
      Destructor();
  }

  SymbolFlagsMap getSymbols() override { return GetSymbols(); }

  void materialize(MaterializationResponsibility R) override {
    Materialize(std::move(R));
  }

  void discard(const VSO &V, SymbolStringPtr Name) override {
    Discard(V, std::move(Name));
  }

private:
  GetSymbolsFunction GetSymbols;
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

  Q.resolve(Foo, JITEvaluatedSymbol(FakeAddr, JITSymbolFlags::Exported));

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

  Q.notifyMaterializationFailed(
      make_error<StringError>("xyz", inconvertibleErrorCode()));

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

  JITSymbolFlags FooFlags = JITSymbolFlags::Exported;
  JITSymbolFlags BarFlags = static_cast<JITSymbolFlags::FlagNames>(
      JITSymbolFlags::Exported | JITSymbolFlags::Weak);

  VSO V;

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      [=]() {
        return SymbolFlagsMap({{Bar, BarFlags}});
      },
      [](MaterializationResponsibility R) {
        llvm_unreachable("Symbol materialized on flags lookup");
      },
      [](const VSO &V, SymbolStringPtr Name) {
        llvm_unreachable("Symbol finalized on flags lookup");
      });

  SymbolMap InitialDefs;
  InitialDefs[Foo] = JITEvaluatedSymbol(0xdeadbeef, FooFlags);
  cantFail(V.define(std::move(InitialDefs)));

  cantFail(V.defineLazy(std::move(MU)));

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

TEST(CoreAPIsTest, DropMaterializerWhenEmpty) {
  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  auto Bar = SP.intern("bar");

  bool DestructorRun = false;

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      [=]() {
        return SymbolFlagsMap(
            {{Foo, JITSymbolFlags::Weak}, {Bar, JITSymbolFlags::Weak}});
      },
      [](MaterializationResponsibility R) {
        llvm_unreachable("Unexpected call to materialize");
      },
      [&](const VSO &V, SymbolStringPtr Name) {
        EXPECT_TRUE(Name == Foo || Name == Bar)
            << "Discard of unexpected symbol?";
      },
      [&]() { DestructorRun = true; });

  VSO V;

  cantFail(V.defineLazy(std::move(MU)));

  auto FooSym = JITEvaluatedSymbol(1, JITSymbolFlags::Exported);
  auto BarSym = JITEvaluatedSymbol(2, JITSymbolFlags::Exported);
  cantFail(V.define(SymbolMap({{Foo, FooSym}})));

  EXPECT_FALSE(DestructorRun)
      << "MaterializationUnit should not have been destroyed yet";

  cantFail(V.define(SymbolMap({{Bar, BarSym}})));

  EXPECT_TRUE(DestructorRun)
      << "MaterializationUnit should have been destroyed";
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

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      [=]() {
        return SymbolFlagsMap(
            {{Foo, JITSymbolFlags::Exported},
             {Bar, static_cast<JITSymbolFlags::FlagNames>(
                       JITSymbolFlags::Exported | JITSymbolFlags::Weak)}});
      },
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

  cantFail(V.defineLazy(std::move(MU)));

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

  for (auto &M : LR.Materializers)
    M();

  EXPECT_TRUE(LR.UnresolvedSymbols.empty()) << "Could not find Foo in dylib";
  EXPECT_TRUE(FooMaterialized) << "Foo was not materialized";
  EXPECT_TRUE(BarDiscarded) << "Bar was not discarded";
  EXPECT_TRUE(OnResolutionRun) << "OnResolutionCallback was not run";
  EXPECT_TRUE(OnReadyRun) << "OnReady was not run";
}

TEST(CoreAPIsTest, FailResolution) {
  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  auto Bar = SP.intern("bar");

  SymbolNameSet Names({Foo, Bar});

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      [=]() {
        return SymbolFlagsMap(
            {{Foo, JITSymbolFlags::Weak}, {Bar, JITSymbolFlags::Weak}});
      },
      [&](MaterializationResponsibility R) { R.notifyMaterializationFailed(); },
      [&](const VSO &V, SymbolStringPtr Name) {
        llvm_unreachable("Unexpected call to discard");
      });

  VSO V;

  cantFail(V.defineLazy(std::move(MU)));

  auto OnResolution = [&](Expected<SymbolMap> Result) {
    handleAllErrors(Result.takeError(),
                    [&](FailedToResolve &F) {
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

  auto LR = V.lookup(std::move(Q), Names);
  for (auto &M : LR.Materializers)
    M();
}

TEST(CoreAPIsTest, FailFinalization) {
  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  auto Bar = SP.intern("bar");

  SymbolNameSet Names({Foo, Bar});

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      [=]() {
        return SymbolFlagsMap(
            {{Foo, JITSymbolFlags::Exported}, {Bar, JITSymbolFlags::Exported}});
      },
      [&](MaterializationResponsibility R) {
        constexpr JITTargetAddress FakeFooAddr = 0xdeadbeef;
        constexpr JITTargetAddress FakeBarAddr = 0xcafef00d;

        auto FooSym = JITEvaluatedSymbol(FakeFooAddr, JITSymbolFlags::Exported);
        auto BarSym = JITEvaluatedSymbol(FakeBarAddr, JITSymbolFlags::Exported);
        R.resolve(SymbolMap({{Foo, FooSym}, {Bar, BarSym}}));
        R.notifyMaterializationFailed();
      },
      [&](const VSO &V, SymbolStringPtr Name) {
        llvm_unreachable("Unexpected call to discard");
      });

  VSO V;

  cantFail(V.defineLazy(std::move(MU)));

  auto OnResolution = [](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
  };

  auto OnReady = [&](Error Err) {
    handleAllErrors(std::move(Err),
                    [&](FailedToFinalize &F) {
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
                          << "Expected a FailedToFinalize error. Got:\n"
                          << ErrMsg;
                    });
  };

  auto Q =
      std::make_shared<AsynchronousSymbolQuery>(Names, OnResolution, OnReady);

  auto LR = V.lookup(std::move(Q), Names);
  for (auto &M : LR.Materializers)
    M();
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
        assert(LR.Materializers.empty() &&
               "Test generated unexpected materialization work?");
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

TEST(CoreAPIsTest, TestLookupWithUnthreadedMaterialization) {
  constexpr JITTargetAddress FakeFooAddr = 0xdeadbeef;
  JITEvaluatedSymbol FooSym(FakeFooAddr, JITSymbolFlags::Exported);

  ExecutionSession ES(std::make_shared<SymbolStringPool>());
  auto Foo = ES.getSymbolStringPool().intern("foo");

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      [=]() {
        return SymbolFlagsMap({{Foo, JITSymbolFlags::Exported}});
      },
      [&](MaterializationResponsibility R) {
        R.resolve({{Foo, FooSym}});
        R.finalize();
      },
      [](const VSO &V, SymbolStringPtr Name) {
        llvm_unreachable("Not expecting finalization");
      });

  VSO V;

  cantFail(V.defineLazy(std::move(MU)));

  auto FooLookupResult =
      cantFail(lookup({&V}, Foo, MaterializeOnCurrentThread()));

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
  auto Foo = ES.getSymbolStringPool().intern("foo");

  auto MU = llvm::make_unique<SimpleMaterializationUnit>(
      [=]() {
        return SymbolFlagsMap({{Foo, JITSymbolFlags::Exported}});
      },
      [&](MaterializationResponsibility R) {
        R.resolve({{Foo, FooSym}});
        R.finalize();
      },
      [](const VSO &V, SymbolStringPtr Name) {
        llvm_unreachable("Not expecting finalization");
      });

  VSO V;

  cantFail(V.defineLazy(std::move(MU)));

  std::thread MaterializationThread;
  auto MaterializeOnNewThread = [&](VSO::Materializer M) {
    // FIXME: Use move capture once we move to C++14.
    auto SharedM = std::make_shared<VSO::Materializer>(std::move(M));
    MaterializationThread = std::thread([SharedM]() { (*SharedM)(); });
  };

  auto FooLookupResult =
    cantFail(lookup({&V}, Foo, MaterializeOnNewThread));

  EXPECT_EQ(FooLookupResult.getAddress(), FooSym.getAddress())
      << "lookup returned an incorrect address";
  EXPECT_EQ(FooLookupResult.getFlags(), FooSym.getFlags())
      << "lookup returned incorrect flags";
  MaterializationThread.join();
#endif
}

} // namespace
