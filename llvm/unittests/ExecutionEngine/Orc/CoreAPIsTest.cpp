//===----------- CoreAPIsTest.cpp - Unit tests for Core ORC APIs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcError.h"
#include "llvm/Testing/Support/Error.h"

#include <deque>
#include <set>
#include <thread>

using namespace llvm;
using namespace llvm::orc;

class CoreAPIsStandardTest : public CoreAPIsBasedStandardTest {};

namespace {

class CustomError : public ErrorInfo<CustomError> {
public:
  static char ID;
  void log(raw_ostream &OS) const override { OS << "CustomError"; }
  std::error_code convertToErrorCode() const override { return {}; }
};
char CustomError::ID = 0;

TEST_F(CoreAPIsStandardTest, ErrorReporter) {
  // Check that errors reported via ExecutionSession::reportError are sent to
  // the registered error reporter, and that the error reporter can hold
  // uniquely owned state.

  Error ReportedError = Error::success();

  ES.setErrorReporter(
      // Make sure error reporter can capture uniquely-owned state.
      [&, State = std::make_unique<int>(42)](Error Err) {
        ReportedError = joinErrors(std::move(Err), std::move(ReportedError));
      });

  ES.reportError(make_error<CustomError>());

  EXPECT_THAT_ERROR(std::move(ReportedError), Failed<CustomError>());
}

TEST_F(CoreAPIsStandardTest, JITDylibAddToLinkOrder) {
  // Check that the JITDylib::addToLinkOrder methods behave as expected.
  auto &JD2 = ES.createBareJITDylib("JD2");
  auto &JD3 = ES.createBareJITDylib("JD3");

  JD.addToLinkOrder(JD2);
  JD.withLinkOrderDo([&](const JITDylibSearchOrder &SO) {
    EXPECT_EQ(SO.size(), 2U);
    EXPECT_EQ(SO[0].first, &JD);
    EXPECT_EQ(SO[1].first, &JD2);
  });

  JD.addToLinkOrder(makeJITDylibSearchOrder({&JD2, &JD3}));
  JD.withLinkOrderDo([&](const JITDylibSearchOrder &SO) {
    // JD2 was already in the search order, so we expect just one extra item
    // here.
    EXPECT_EQ(SO.size(), 3U);
    EXPECT_EQ(SO[0].first, &JD);
    EXPECT_EQ(SO[1].first, &JD2);
    EXPECT_EQ(SO[2].first, &JD3);
  });
}

TEST_F(CoreAPIsStandardTest, BasicSuccessfulLookup) {
  bool OnCompletionRun = false;

  auto OnCompletion = [&](Expected<SymbolMap> Result) {
    EXPECT_TRUE(!!Result) << "Resolution unexpectedly returned error";
    auto &Resolved = *Result;
    auto I = Resolved.find(Foo);
    EXPECT_NE(I, Resolved.end()) << "Could not find symbol definition";
    EXPECT_EQ(I->second.getAddress(), FooAddr)
        << "Resolution returned incorrect result";
    OnCompletionRun = true;
  };

  std::unique_ptr<MaterializationResponsibility> FooMR;

  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        FooMR = std::move(R);
      })));

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Foo), SymbolState::Ready, OnCompletion,
            NoDependenciesToRegister);

  EXPECT_FALSE(OnCompletionRun) << "Should not have been resolved yet";

  cantFail(FooMR->notifyResolved({{Foo, FooSym}}));

  EXPECT_FALSE(OnCompletionRun) << "Should not be ready yet";

  cantFail(FooMR->notifyEmitted({}));

  EXPECT_TRUE(OnCompletionRun) << "Should have been marked ready";
}

TEST_F(CoreAPIsStandardTest, EmptyLookup) {
  bool OnCompletionRun = false;

  auto OnCompletion = [&](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
    OnCompletionRun = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(),
            SymbolState::Ready, OnCompletion, NoDependenciesToRegister);

  EXPECT_TRUE(OnCompletionRun) << "OnCompletion was not run for empty query";
}

TEST_F(CoreAPIsStandardTest, ResolveUnrequestedSymbol) {
  // Test that all symbols in a MaterializationUnit materialize corretly when
  // only a subset of symbols is looked up.
  // The aim here is to ensure that we're not relying on the query to set up
  // state needed to materialize the unrequested symbols.

  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}, {Bar, BarSym.getFlags()}}),
      [this](std::unique_ptr<MaterializationResponsibility> R) {
        cantFail(R->notifyResolved({{Foo, FooSym}, {Bar, BarSym}}));
        cantFail(R->notifyEmitted({}));
      })));

  auto Result =
      cantFail(ES.lookup(makeJITDylibSearchOrder(&JD), SymbolLookupSet({Foo})));
  EXPECT_EQ(Result.size(), 1U) << "Unexpected number of results";
  EXPECT_TRUE(Result.count(Foo)) << "Expected result for \"Foo\"";
}

TEST_F(CoreAPIsStandardTest, MaterializationSideEffctsOnlyBasic) {
  // Test that basic materialization-side-effects-only symbols work as expected:
  // that they can be emitted without being resolved, that queries for them
  // don't return until they're emitted, and that they don't appear in query
  // results.

  std::unique_ptr<MaterializationResponsibility> FooR;
  std::optional<SymbolMap> Result;

  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap(
          {{Foo, JITSymbolFlags::Exported |
                     JITSymbolFlags::MaterializationSideEffectsOnly}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        FooR = std::move(R);
      })));

  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD),
      SymbolLookupSet(Foo, SymbolLookupFlags::WeaklyReferencedSymbol),
      SymbolState::Ready,
      [&](Expected<SymbolMap> LookupResult) {
        if (LookupResult)
          Result = std::move(*LookupResult);
        else
          ADD_FAILURE() << "Unexpected lookup error: "
                        << toString(LookupResult.takeError());
      },
      NoDependenciesToRegister);

  EXPECT_FALSE(Result) << "Lookup returned unexpectedly";
  EXPECT_TRUE(FooR) << "Lookup failed to trigger materialization";
  EXPECT_THAT_ERROR(FooR->notifyEmitted({}), Succeeded())
      << "Emission of materialization-side-effects-only symbol failed";

  EXPECT_TRUE(Result) << "Lookup failed to return";
  EXPECT_TRUE(Result->empty()) << "Lookup result contained unexpected value";
}

TEST_F(CoreAPIsStandardTest, MaterializationSideEffectsOnlyFailuresPersist) {
  // Test that when a MaterializationSideEffectsOnly symbol is failed it
  // remains in the failure state rather than vanishing.

  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap(
          {{Foo, JITSymbolFlags::Exported |
                     JITSymbolFlags::MaterializationSideEffectsOnly}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        R->failMaterialization();
      })));

  EXPECT_THAT_EXPECTED(
      ES.lookup(makeJITDylibSearchOrder(&JD), SymbolLookupSet({Foo})),
      Failed());
  EXPECT_THAT_EXPECTED(
      ES.lookup(makeJITDylibSearchOrder(&JD), SymbolLookupSet({Foo})),
      Failed());
}

TEST_F(CoreAPIsStandardTest, RemoveSymbolsTest) {
  // Test that:
  // (1) Missing symbols generate a SymbolsNotFound error.
  // (2) Materializing symbols generate a SymbolCouldNotBeRemoved error.
  // (3) Removal of unmaterialized symbols triggers discard on the
  //     materialization unit.
  // (4) Removal of symbols destroys empty materialization units.
  // (5) Removal of materialized symbols works.

  // Foo will be fully materialized.
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  // Bar will be unmaterialized.
  bool BarDiscarded = false;
  bool BarMaterializerDestructed = false;
  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
      [this](std::unique_ptr<MaterializationResponsibility> R) {
        ADD_FAILURE() << "Unexpected materialization of \"Bar\"";
        cantFail(R->notifyResolved({{Bar, BarSym}}));
        cantFail(R->notifyEmitted({}));
      },
      nullptr,
      [&](const JITDylib &JD, const SymbolStringPtr &Name) {
        EXPECT_EQ(Name, Bar) << "Expected \"Bar\" to be discarded";
        if (Name == Bar)
          BarDiscarded = true;
      },
      [&]() { BarMaterializerDestructed = true; })));

  // Baz will be in the materializing state initially, then
  // materialized for the final removal attempt.
  std::unique_ptr<MaterializationResponsibility> BazR;
  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Baz, BazSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        BazR = std::move(R);
      },
      nullptr,
      [](const JITDylib &JD, const SymbolStringPtr &Name) {
        ADD_FAILURE() << "\"Baz\" discarded unexpectedly";
      })));

  bool OnCompletionRun = false;
  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD),
      SymbolLookupSet({Foo, Baz}), SymbolState::Ready,
      [&](Expected<SymbolMap> Result) {
        cantFail(Result.takeError());
        OnCompletionRun = true;
      },
      NoDependenciesToRegister);

  {
    // Attempt 1: Search for a missing symbol, Qux.
    auto Err = JD.remove({Foo, Bar, Baz, Qux});
    EXPECT_TRUE(!!Err) << "Expected failure";
    EXPECT_TRUE(Err.isA<SymbolsNotFound>())
        << "Expected a SymbolsNotFound error";
    consumeError(std::move(Err));
  }

  {
    // Attempt 2: Search for a symbol that is still materializing, Baz.
    auto Err = JD.remove({Foo, Bar, Baz});
    EXPECT_TRUE(!!Err) << "Expected failure";
    EXPECT_TRUE(Err.isA<SymbolsCouldNotBeRemoved>())
        << "Expected a SymbolsNotFound error";
    consumeError(std::move(Err));
  }

  cantFail(BazR->notifyResolved({{Baz, BazSym}}));
  cantFail(BazR->notifyEmitted({}));
  {
    // Attempt 3: Search now that all symbols are fully materialized
    // (Foo, Baz), or not yet materialized (Bar).
    auto Err = JD.remove({Foo, Bar, Baz});
    EXPECT_FALSE(!!Err) << "Expected success";
  }

  EXPECT_TRUE(BarDiscarded) << "\"Bar\" should have been discarded";
  EXPECT_TRUE(BarMaterializerDestructed)
      << "\"Bar\"'s materializer should have been destructed";
  EXPECT_TRUE(OnCompletionRun) << "OnCompletion should have been run";
}

TEST_F(CoreAPIsStandardTest, DiscardInitSymbol) {
  SymbolStringPtr ForwardedDiscardSym = nullptr;

  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}, {Bar, BarSym.getFlags()}}),
      [](std::unique_ptr<MaterializationResponsibility> R) {
        llvm_unreachable("Materialize called unexpectedly?");
      },
      Foo,
      [&](const JITDylib &, SymbolStringPtr Sym) {
        ForwardedDiscardSym = std::move(Sym);
      });

  MU->doDiscard(JD, Foo);

  EXPECT_EQ(ForwardedDiscardSym, Foo);
  EXPECT_EQ(MU->getSymbols().size(), 1U);
  EXPECT_TRUE(MU->getSymbols().count(Bar));
  EXPECT_EQ(MU->getInitializerSymbol(), nullptr);
}

TEST_F(CoreAPIsStandardTest, LookupWithHiddenSymbols) {
  auto BarHiddenFlags = BarSym.getFlags() & ~JITSymbolFlags::Exported;
  auto BarHiddenSym = ExecutorSymbolDef(BarSym.getAddress(), BarHiddenFlags);

  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarHiddenSym}})));

  auto &JD2 = ES.createBareJITDylib("JD2");
  cantFail(JD2.define(absoluteSymbols({{Bar, QuxSym}})));

  /// Try a blocking lookup.
  auto Result = cantFail(ES.lookup(makeJITDylibSearchOrder({&JD, &JD2}),
                                   SymbolLookupSet({Foo, Bar})));

  EXPECT_EQ(Result.size(), 2U) << "Unexpected number of results";
  EXPECT_EQ(Result.count(Foo), 1U) << "Missing result for \"Foo\"";
  EXPECT_EQ(Result.count(Bar), 1U) << "Missing result for \"Bar\"";
  EXPECT_EQ(Result[Bar].getAddress(), QuxSym.getAddress())
      << "Wrong result for \"Bar\"";
}

TEST_F(CoreAPIsStandardTest, LookupFlagsTest) {
  // Test that lookupFlags works on a predefined symbol, and does not trigger
  // materialization of a lazy symbol. Make the lazy symbol weak to test that
  // the weak flag is propagated correctly.

  BarSym.setFlags(static_cast<JITSymbolFlags::FlagNames>(
      JITSymbolFlags::Exported | JITSymbolFlags::Weak));
  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
      [](std::unique_ptr<MaterializationResponsibility> R) {
        llvm_unreachable("Symbol materialized on flags lookup");
      });

  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));
  cantFail(JD.define(std::move(MU)));

  auto SymbolFlags = cantFail(ES.lookupFlags(
      LookupKind::Static,
      {{&JD, JITDylibLookupFlags::MatchExportedSymbolsOnly}},
      SymbolLookupSet({Foo, Bar, Baz},
                      SymbolLookupFlags::WeaklyReferencedSymbol)));

  EXPECT_EQ(SymbolFlags.size(), 2U)
      << "Returned symbol flags contains unexpected results";
  EXPECT_EQ(SymbolFlags.count(Foo), 1U) << "Missing lookupFlags result for Foo";
  EXPECT_EQ(SymbolFlags[Foo], FooSym.getFlags())
      << "Incorrect flags returned for Foo";
  EXPECT_EQ(SymbolFlags.count(Bar), 1U)
      << "Missing  lookupFlags result for Bar";
  EXPECT_EQ(SymbolFlags[Bar], BarSym.getFlags())
      << "Incorrect flags returned for Bar";
}

TEST_F(CoreAPIsStandardTest, LookupWithGeneratorFailure) {

  class BadGenerator : public DefinitionGenerator {
  public:
    Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &,
                        JITDylibLookupFlags, const SymbolLookupSet &) override {
      return make_error<StringError>("BadGenerator", inconvertibleErrorCode());
    }
  };

  JD.addGenerator(std::make_unique<BadGenerator>());

  EXPECT_THAT_ERROR(
      ES.lookupFlags(LookupKind::Static,
                     {{&JD, JITDylibLookupFlags::MatchExportedSymbolsOnly}},
                     SymbolLookupSet(Foo))
          .takeError(),
      Failed<StringError>())
      << "Generator failure did not propagate through lookupFlags";

  EXPECT_THAT_ERROR(
      ES.lookup(makeJITDylibSearchOrder(&JD), SymbolLookupSet(Foo)).takeError(),
      Failed<StringError>())
      << "Generator failure did not propagate through lookup";
}

TEST_F(CoreAPIsStandardTest, TestBasicAliases) {
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));
  cantFail(JD.define(symbolAliases({{Baz, {Foo, JITSymbolFlags::Exported}},
                                    {Qux, {Bar, JITSymbolFlags::Weak}}})));
  cantFail(JD.define(absoluteSymbols({{Qux, QuxSym}})));

  auto Result =
      ES.lookup(makeJITDylibSearchOrder(&JD), SymbolLookupSet({Baz, Qux}));
  EXPECT_TRUE(!!Result) << "Unexpected lookup failure";
  EXPECT_EQ(Result->count(Baz), 1U) << "No result for \"baz\"";
  EXPECT_EQ(Result->count(Qux), 1U) << "No result for \"qux\"";
  EXPECT_EQ((*Result)[Baz].getAddress(), FooSym.getAddress())
      << "\"Baz\"'s address should match \"Foo\"'s";
  EXPECT_EQ((*Result)[Qux].getAddress(), QuxSym.getAddress())
      << "The \"Qux\" alias should have been overriden";
}

TEST_F(CoreAPIsStandardTest, TestChainedAliases) {
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));
  cantFail(JD.define(symbolAliases(
      {{Baz, {Bar, BazSym.getFlags()}}, {Bar, {Foo, BarSym.getFlags()}}})));

  auto Result =
      ES.lookup(makeJITDylibSearchOrder(&JD), SymbolLookupSet({Bar, Baz}));
  EXPECT_TRUE(!!Result) << "Unexpected lookup failure";
  EXPECT_EQ(Result->count(Bar), 1U) << "No result for \"bar\"";
  EXPECT_EQ(Result->count(Baz), 1U) << "No result for \"baz\"";
  EXPECT_EQ((*Result)[Bar].getAddress(), FooSym.getAddress())
      << "\"Bar\"'s address should match \"Foo\"'s";
  EXPECT_EQ((*Result)[Baz].getAddress(), FooSym.getAddress())
      << "\"Baz\"'s address should match \"Foo\"'s";
}

TEST_F(CoreAPIsStandardTest, TestBasicReExports) {
  // Test that the basic use case of re-exporting a single symbol from another
  // JITDylib works.
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  auto &JD2 = ES.createBareJITDylib("JD2");

  cantFail(JD2.define(reexports(JD, {{Bar, {Foo, BarSym.getFlags()}}})));

  auto Result = cantFail(ES.lookup(makeJITDylibSearchOrder(&JD2), Bar));
  EXPECT_EQ(Result.getAddress(), FooSym.getAddress())
      << "Re-export Bar for symbol Foo should match FooSym's address";
}

TEST_F(CoreAPIsStandardTest, TestThatReExportsDontUnnecessarilyMaterialize) {
  // Test that re-exports do not materialize symbols that have not been queried
  // for.
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  bool BarMaterialized = false;
  auto BarMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        BarMaterialized = true;
        cantFail(R->notifyResolved({{Bar, BarSym}}));
        cantFail(R->notifyEmitted({}));
      });

  cantFail(JD.define(BarMU));

  auto &JD2 = ES.createBareJITDylib("JD2");

  cantFail(JD2.define(reexports(
      JD, {{Baz, {Foo, BazSym.getFlags()}}, {Qux, {Bar, QuxSym.getFlags()}}})));

  auto Result = cantFail(ES.lookup(makeJITDylibSearchOrder(&JD2), Baz));
  EXPECT_EQ(Result.getAddress(), FooSym.getAddress())
      << "Re-export Baz for symbol Foo should match FooSym's address";

  EXPECT_FALSE(BarMaterialized) << "Bar should not have been materialized";
}

TEST_F(CoreAPIsStandardTest, TestReexportsGenerator) {
  // Test that a re-exports generator can dynamically generate reexports.

  auto &JD2 = ES.createBareJITDylib("JD2");
  cantFail(JD2.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));

  auto Filter = [this](SymbolStringPtr Name) { return Name != Bar; };

  JD.addGenerator(std::make_unique<ReexportsGenerator>(
      JD2, JITDylibLookupFlags::MatchExportedSymbolsOnly, Filter));

  auto Flags = cantFail(ES.lookupFlags(
      LookupKind::Static,
      {{&JD, JITDylibLookupFlags::MatchExportedSymbolsOnly}},
      SymbolLookupSet({Foo, Bar, Baz},
                      SymbolLookupFlags::WeaklyReferencedSymbol)));
  EXPECT_EQ(Flags.size(), 1U) << "Unexpected number of results";
  EXPECT_EQ(Flags[Foo], FooSym.getFlags()) << "Unexpected flags for Foo";

  auto Result = cantFail(ES.lookup(makeJITDylibSearchOrder(&JD), Foo));

  EXPECT_EQ(Result.getAddress(), FooSym.getAddress())
      << "Incorrect reexported symbol address";
}

TEST_F(CoreAPIsStandardTest, TestTrivialCircularDependency) {
  std::unique_ptr<MaterializationResponsibility> FooR;
  auto FooMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        FooR = std::move(R);
      });

  cantFail(JD.define(FooMU));

  bool FooReady = false;
  auto OnCompletion = [&](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
    FooReady = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet({Foo}), SymbolState::Ready, OnCompletion,
            NoDependenciesToRegister);

  EXPECT_THAT_ERROR(FooR->notifyResolved({{Foo, FooSym}}), Succeeded())
      << "No symbols marked failed, but Foo failed to resolve";
  SymbolDependenceGroup SDG({{Foo}, {{&JD, SymbolNameSet({Foo})}}});
  EXPECT_THAT_ERROR(FooR->notifyEmitted(SDG), Succeeded())
      << "No symbols marked failed, but Foo failed to emit";

  EXPECT_TRUE(FooReady)
    << "Self-dependency prevented symbol from being marked ready";
}

TEST_F(CoreAPIsStandardTest, TestCircularDependenceInOneJITDylib) {
  // Test that a circular symbol dependency between three symbols in a JITDylib
  // does not prevent any symbol from becoming 'ready' once all symbols are
  // emitted.

  std::unique_ptr<MaterializationResponsibility> FooR;
  std::unique_ptr<MaterializationResponsibility> BarR;
  std::unique_ptr<MaterializationResponsibility> BazR;

  // Create a MaterializationUnit for each symbol that moves the
  // MaterializationResponsibility into one of the locals above.
  auto FooMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        FooR = std::move(R);
      });

  auto BarMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        BarR = std::move(R);
      });

  auto BazMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Baz, BazSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        BazR = std::move(R);
      });

  // Define the symbols.
  cantFail(JD.define(FooMU));
  cantFail(JD.define(BarMU));
  cantFail(JD.define(BazMU));

  // Query each of the symbols to trigger materialization.
  bool FooResolved = false;
  bool FooReady = false;

  auto OnFooResolution = [&](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
    FooResolved = true;
  };

  auto OnFooReady = [&](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
    FooReady = true;
  };

  // Issue lookups for Foo. Use NoDependenciesToRegister: We're going to add
  // the dependencies manually below.
  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Foo), SymbolState::Resolved,
            std::move(OnFooResolution), NoDependenciesToRegister);

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Foo), SymbolState::Ready, std::move(OnFooReady),
            NoDependenciesToRegister);

  bool BarResolved = false;
  bool BarReady = false;
  auto OnBarResolution = [&](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
    BarResolved = true;
  };

  auto OnBarReady = [&](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
    BarReady = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Bar), SymbolState::Resolved,
            std::move(OnBarResolution), NoDependenciesToRegister);

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Bar), SymbolState::Ready, std::move(OnBarReady),
            NoDependenciesToRegister);

  bool BazResolved = false;
  bool BazReady = false;

  auto OnBazResolution = [&](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
    BazResolved = true;
  };

  auto OnBazReady = [&](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
    BazReady = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Baz), SymbolState::Resolved,
            std::move(OnBazResolution), NoDependenciesToRegister);

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Baz), SymbolState::Ready, std::move(OnBazReady),
            NoDependenciesToRegister);

  // Check that nothing has been resolved yet.
  EXPECT_FALSE(FooResolved) << "\"Foo\" should not be resolved yet";
  EXPECT_FALSE(BarResolved) << "\"Bar\" should not be resolved yet";
  EXPECT_FALSE(BazResolved) << "\"Baz\" should not be resolved yet";

  // Resolve the symbols (but do not emit them).
  EXPECT_THAT_ERROR(FooR->notifyResolved({{Foo, FooSym}}), Succeeded())
      << "No symbols failed, but Foo failed to resolve";
  EXPECT_THAT_ERROR(BarR->notifyResolved({{Bar, BarSym}}), Succeeded())
      << "No symbols failed, but Bar failed to resolve";
  EXPECT_THAT_ERROR(BazR->notifyResolved({{Baz, BazSym}}), Succeeded())
      << "No symbols failed, but Baz failed to resolve";

  // Verify that the symbols have been resolved, but are not ready yet.
  EXPECT_TRUE(FooResolved) << "\"Foo\" should be resolved now";
  EXPECT_TRUE(BarResolved) << "\"Bar\" should be resolved now";
  EXPECT_TRUE(BazResolved) << "\"Baz\" should be resolved now";

  EXPECT_FALSE(FooReady) << "\"Foo\" should not be ready yet";
  EXPECT_FALSE(BarReady) << "\"Bar\" should not be ready yet";
  EXPECT_FALSE(BazReady) << "\"Baz\" should not be ready yet";

  // Emit two of the symbols.
  {
    SymbolDependenceGroup FooDeps({{Foo}, {{&JD, {Foo, Bar}}}});
    EXPECT_THAT_ERROR(FooR->notifyEmitted(FooDeps), Succeeded())
        << "No symbols failed, but Foo failed to emit";

    SymbolDependenceGroup BarDeps({{Bar}, {{&JD, {Bar, Baz}}}});
    EXPECT_THAT_ERROR(BarR->notifyEmitted(BarDeps), Succeeded())
        << "No symbols failed, but Bar failed to emit";
  }

  // Verify that nothing is ready until the circular dependence is resolved.
  EXPECT_FALSE(FooReady) << "\"Foo\" still should not be ready";
  EXPECT_FALSE(BarReady) << "\"Bar\" still should not be ready";
  EXPECT_FALSE(BazReady) << "\"Baz\" still should not be ready";

  // Emit the last symbol.
  {
    SymbolDependenceGroup BazDeps({{Baz}, {{&JD, {Baz, Foo}}}});
    EXPECT_THAT_ERROR(BazR->notifyEmitted(BazDeps), Succeeded())
        << "No symbols failed, but Baz failed to emit";
  }

  // Verify that everything becomes ready once the circular dependence resolved.
  EXPECT_TRUE(FooReady) << "\"Foo\" should be ready now";
  EXPECT_TRUE(BarReady) << "\"Bar\" should be ready now";
  EXPECT_TRUE(BazReady) << "\"Baz\" should be ready now";
}

TEST_F(CoreAPIsStandardTest, FailureInDependency) {
  std::unique_ptr<MaterializationResponsibility> FooR;
  std::unique_ptr<MaterializationResponsibility> BarR;

  // Create a MaterializationUnit for each symbol that moves the
  // MaterializationResponsibility into one of the locals above.
  auto FooMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        FooR = std::move(R);
      });

  auto BarMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        BarR = std::move(R);
      });

  // Define the symbols.
  cantFail(JD.define(FooMU));
  cantFail(JD.define(BarMU));

  bool OnFooReadyRun = false;
  auto OnFooReady = [&](Expected<SymbolMap> Result) {
    EXPECT_THAT_EXPECTED(std::move(Result), Failed());
    OnFooReadyRun = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Foo), SymbolState::Ready, std::move(OnFooReady),
            NoDependenciesToRegister);

  bool OnBarReadyRun = false;
  auto OnBarReady = [&](Expected<SymbolMap> Result) {
    EXPECT_THAT_EXPECTED(std::move(Result), Failed());
    OnBarReadyRun = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Bar), SymbolState::Ready, std::move(OnBarReady),
            NoDependenciesToRegister);

  // Fail bar.
  BarR->failMaterialization();

  // Verify that queries on Bar failed, but queries on Foo have not yet.
  EXPECT_TRUE(OnBarReadyRun) << "Query for \"Bar\" was not run";
  EXPECT_FALSE(OnFooReadyRun) << "Query for \"Foo\" was run unexpectedly";

  // Check that we can still resolve Foo (even though it has been failed).
  EXPECT_THAT_ERROR(FooR->notifyResolved({{Foo, FooSym}}), Succeeded())
      << "Expected resolution for \"Foo\" to succeed despite error state.";

  FooR->failMaterialization();

  // Verify that queries on Foo have now failed.
  EXPECT_TRUE(OnFooReadyRun) << "Query for \"Foo\" was not run";

  // Verify that subsequent lookups on Bar and Foo fail.
  EXPECT_THAT_EXPECTED(ES.lookup({&JD}, {Bar}), Failed())
      << "Lookup on failed symbol should fail";

  EXPECT_THAT_EXPECTED(ES.lookup({&JD}, {Foo}), Failed())
      << "Lookup on failed symbol should fail";
}

TEST_F(CoreAPIsStandardTest, AddDependencyOnFailedSymbol) {
  std::unique_ptr<MaterializationResponsibility> FooR;
  std::unique_ptr<MaterializationResponsibility> BarR;

  // Create a MaterializationUnit for each symbol that moves the
  // MaterializationResponsibility into one of the locals above.
  auto FooMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        FooR = std::move(R);
      });

  auto BarMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        BarR = std::move(R);
      });

  // Define the symbols.
  cantFail(JD.define(FooMU));
  cantFail(JD.define(BarMU));

  bool OnFooReadyRun = false;
  auto OnFooReady = [&](Expected<SymbolMap> Result) {
    EXPECT_THAT_EXPECTED(std::move(Result), Failed());
    OnFooReadyRun = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Foo), SymbolState::Ready, std::move(OnFooReady),
            NoDependenciesToRegister);

  bool OnBarReadyRun = false;
  auto OnBarReady = [&](Expected<SymbolMap> Result) {
    EXPECT_THAT_EXPECTED(std::move(Result), Failed());
    OnBarReadyRun = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Bar), SymbolState::Ready, std::move(OnBarReady),
            NoDependenciesToRegister);

  // Fail bar.
  BarR->failMaterialization();

  // We expect Bar's query to fail immediately, but Foo's query not to have run
  // yet.
  EXPECT_TRUE(OnBarReadyRun) << "Query for \"Bar\" was not run";
  EXPECT_FALSE(OnFooReadyRun) << "Query for \"Foo\" should not have run yet";

  EXPECT_THAT_ERROR(FooR->notifyResolved({{Foo, FooSym}}), Succeeded())
      << "Expected resolution for \"Foo\" to succeed.";

  // Check that emission of Foo fails.
  {
    SymbolDependenceGroup FooDeps({{Foo}, {{&JD, {Bar}}}});
    EXPECT_THAT_ERROR(FooR->notifyEmitted(FooDeps), Failed());
  }

  FooR->failMaterialization();

  // Foo's query should have failed before we return from addDependencies.
  EXPECT_TRUE(OnFooReadyRun) << "Query for \"Foo\" was not run";

  // Verify that subsequent lookups on Bar and Foo fail.
  EXPECT_THAT_EXPECTED(ES.lookup({&JD}, {Bar}), Failed())
      << "Lookup on failed symbol should fail";

  EXPECT_THAT_EXPECTED(ES.lookup({&JD}, {Foo}), Failed())
      << "Lookup on failed symbol should fail";
}

TEST_F(CoreAPIsStandardTest, FailAfterMaterialization) {
  std::unique_ptr<MaterializationResponsibility> FooR;
  std::unique_ptr<MaterializationResponsibility> BarR;

  // Create a MaterializationUnit for each symbol that moves the
  // MaterializationResponsibility into one of the locals above.
  auto FooMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        FooR = std::move(R);
      });

  auto BarMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        BarR = std::move(R);
      });

  // Define the symbols.
  cantFail(JD.define(FooMU));
  cantFail(JD.define(BarMU));

  bool OnFooReadyRun = false;
  auto OnFooReady = [&](Expected<SymbolMap> Result) {
    EXPECT_THAT_EXPECTED(std::move(Result), Failed());
    OnFooReadyRun = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Foo), SymbolState::Ready, std::move(OnFooReady),
            NoDependenciesToRegister);

  bool OnBarReadyRun = false;
  auto OnBarReady = [&](Expected<SymbolMap> Result) {
    EXPECT_THAT_EXPECTED(std::move(Result), Failed());
    OnBarReadyRun = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Bar), SymbolState::Ready, std::move(OnBarReady),
            NoDependenciesToRegister);

  // Materialize Foo.
  EXPECT_THAT_ERROR(FooR->notifyResolved({{Foo, FooSym}}), Succeeded())
      << "Expected resolution for \"Foo\" to succeed.";
  {
    SymbolDependenceGroup FooDeps({{Foo}, {{&JD, {Bar}}}});
    EXPECT_THAT_ERROR(FooR->notifyEmitted(FooDeps), Succeeded())
        << "Expected emission for \"Foo\" to succeed.";
  }

  // Fail bar.
  BarR->failMaterialization();

  // Verify that both queries failed.
  EXPECT_TRUE(OnFooReadyRun) << "Query for Foo did not run";
  EXPECT_TRUE(OnBarReadyRun) << "Query for Bar did not run";
}

TEST_F(CoreAPIsStandardTest, FailMaterializerWithUnqueriedSymbols) {
  // Make sure that symbols with no queries aganist them still
  // fail correctly.

  bool MaterializerRun = false;
  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap(
          {{Foo, JITSymbolFlags::Exported}, {Bar, JITSymbolFlags::Exported}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        MaterializerRun = true;
        R->failMaterialization();
      });

  cantFail(JD.define(std::move(MU)));

  // Issue a query for Foo, but not bar.
  EXPECT_THAT_EXPECTED(ES.lookup({&JD}, {Foo}), Failed())
      << "Expected lookup to fail.";

  // Check that the materializer (and therefore failMaterialization) ran.
  EXPECT_TRUE(MaterializerRun) << "Expected materializer to have run by now";

  // Check that subsequent queries against both symbols fail.
  EXPECT_THAT_EXPECTED(ES.lookup({&JD}, {Foo}), Failed())
      << "Expected lookup for Foo to fail.";
  EXPECT_THAT_EXPECTED(ES.lookup({&JD}, {Bar}), Failed())
      << "Expected lookup for Bar to fail.";
}

TEST_F(CoreAPIsStandardTest, DropMaterializerWhenEmpty) {
  bool DestructorRun = false;

  JITSymbolFlags WeakExported(JITSymbolFlags::Exported);
  WeakExported |= JITSymbolFlags::Weak;

  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, WeakExported}, {Bar, WeakExported}}),
      [](std::unique_ptr<MaterializationResponsibility> R) {
        llvm_unreachable("Unexpected call to materialize");
      },
      nullptr,
      [&](const JITDylib &JD, SymbolStringPtr Name) {
        EXPECT_TRUE(Name == Foo || Name == Bar)
            << "Discard of unexpected symbol?";
      },
      [&]() { DestructorRun = true; });

  cantFail(JD.define(MU));

  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  EXPECT_FALSE(DestructorRun)
      << "MaterializationUnit should not have been destroyed yet";

  cantFail(JD.define(absoluteSymbols({{Bar, BarSym}})));

  EXPECT_TRUE(DestructorRun)
      << "MaterializationUnit should have been destroyed";
}

TEST_F(CoreAPIsStandardTest, AddAndMaterializeLazySymbol) {
  bool FooMaterialized = false;
  bool BarDiscarded = false;

  JITSymbolFlags WeakExported(JITSymbolFlags::Exported);
  WeakExported |= JITSymbolFlags::Weak;

  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, JITSymbolFlags::Exported}, {Bar, WeakExported}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        assert(BarDiscarded && "Bar should have been discarded by this point");
        cantFail(R->notifyResolved(SymbolMap({{Foo, FooSym}})));
        cantFail(R->notifyEmitted({}));
        FooMaterialized = true;
      },
      nullptr,
      [&](const JITDylib &JD, SymbolStringPtr Name) {
        EXPECT_EQ(Name, Bar) << "Expected Name to be Bar";
        BarDiscarded = true;
      });

  cantFail(JD.define(MU));
  cantFail(JD.define(absoluteSymbols({{Bar, BarSym}})));

  bool OnCompletionRun = false;

  auto OnCompletion = [&](Expected<SymbolMap> Result) {
    EXPECT_TRUE(!!Result) << "Resolution unexpectedly returned error";
    auto I = Result->find(Foo);
    EXPECT_NE(I, Result->end()) << "Could not find symbol definition";
    EXPECT_EQ(I->second.getAddress(), FooSym.getAddress())
        << "Resolution returned incorrect result";
    OnCompletionRun = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Foo), SymbolState::Ready, std::move(OnCompletion),
            NoDependenciesToRegister);

  EXPECT_TRUE(FooMaterialized) << "Foo was not materialized";
  EXPECT_TRUE(BarDiscarded) << "Bar was not discarded";
  EXPECT_TRUE(OnCompletionRun) << "OnResolutionCallback was not run";
}

TEST_F(CoreAPIsStandardTest, TestBasicWeakSymbolMaterialization) {
  // Test that weak symbols are materialized correctly when we look them up.
  BarSym.setFlags(BarSym.getFlags() | JITSymbolFlags::Weak);

  bool BarMaterialized = false;
  auto MU1 = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}, {Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        cantFail(R->notifyResolved(SymbolMap({{Foo, FooSym}, {Bar, BarSym}})));
        cantFail(R->notifyEmitted({}));
        BarMaterialized = true;
      });

  bool DuplicateBarDiscarded = false;
  auto MU2 = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        ADD_FAILURE() << "Attempt to materialize Bar from the wrong unit";
        R->failMaterialization();
      },
      nullptr,
      [&](const JITDylib &JD, SymbolStringPtr Name) {
        EXPECT_EQ(Name, Bar) << "Expected \"Bar\" to be discarded";
        DuplicateBarDiscarded = true;
      });

  cantFail(JD.define(MU1));
  cantFail(JD.define(MU2));

  bool OnCompletionRun = false;

  auto OnCompletion = [&](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
    OnCompletionRun = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Bar), SymbolState::Ready, std::move(OnCompletion),
            NoDependenciesToRegister);

  EXPECT_TRUE(OnCompletionRun) << "OnCompletion not run";
  EXPECT_TRUE(BarMaterialized) << "Bar was not materialized at all";
  EXPECT_TRUE(DuplicateBarDiscarded)
      << "Duplicate bar definition not discarded";
}

TEST_F(CoreAPIsStandardTest, RedefineBoundWeakSymbol) {
  // Check that redefinition of a bound weak symbol fails.

  JITSymbolFlags WeakExported(JITSymbolFlags::Exported);
  WeakExported |= JITSymbolFlags::Weak;

  // Define "Foo" as weak, force materialization.
  cantFail(JD.define(absoluteSymbols({{Foo, {FooAddr, WeakExported}}})));
  cantFail(ES.lookup({&JD}, Foo));

  // Attempt to redefine "Foo". Expect failure, despite "Foo" being weak,
  // since it has already been bound.
  EXPECT_THAT_ERROR(JD.define(absoluteSymbols({{Foo, FooSym}})), Failed());
}

TEST_F(CoreAPIsStandardTest, DefineMaterializingSymbol) {
  bool ExpectNoMoreMaterialization = false;
  DispatchOverride = [&](std::unique_ptr<Task> T) {
    if (ExpectNoMoreMaterialization && isa<MaterializationTask>(*T))
      ADD_FAILURE() << "Unexpected materialization";
    T->run();
  };

  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        cantFail(
            R->defineMaterializing(SymbolFlagsMap({{Bar, BarSym.getFlags()}})));
        cantFail(R->notifyResolved(SymbolMap({{Foo, FooSym}, {Bar, BarSym}})));
        cantFail(R->notifyEmitted({}));
      });

  cantFail(JD.define(MU));
  cantFail(ES.lookup(makeJITDylibSearchOrder(&JD), Foo));

  // Assert that materialization is complete by now.
  ExpectNoMoreMaterialization = true;

  // Look up bar to verify that no further materialization happens.
  auto BarResult = cantFail(ES.lookup(makeJITDylibSearchOrder(&JD), Bar));
  EXPECT_EQ(BarResult.getAddress(), BarSym.getAddress())
      << "Expected Bar == BarSym";
}

TEST_F(CoreAPIsStandardTest, GeneratorTest) {
  ExecutorSymbolDef BazHiddenSym(BazSym.getAddress(),
                                 BazSym.getFlags() & ~JITSymbolFlags::Exported);
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}, {Baz, BazHiddenSym}})));

  class TestGenerator : public DefinitionGenerator {
  public:
    TestGenerator(SymbolMap Symbols) : Symbols(std::move(Symbols)) {}
    Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                        JITDylibLookupFlags JDLookupFlags,
                        const SymbolLookupSet &Names) override {
      SymbolMap NewDefs;

      for (const auto &KV : Names) {
        const auto &Name = KV.first;
        if (Symbols.count(Name))
          NewDefs[Name] = Symbols[Name];
      }

      cantFail(JD.define(absoluteSymbols(std::move(NewDefs))));
      return Error::success();
    };

  private:
    SymbolMap Symbols;
  };

  JD.addGenerator(std::make_unique<TestGenerator>(
      SymbolMap({{Bar, BarSym}, {Baz, BazSym}})));

  auto Result = cantFail(
      ES.lookup(makeJITDylibSearchOrder(&JD),
                SymbolLookupSet({Foo, Bar})
                    .add(Baz, SymbolLookupFlags::WeaklyReferencedSymbol)));

  EXPECT_EQ(Result.count(Bar), 1U) << "Expected to find fallback def for 'bar'";
  EXPECT_EQ(Result[Bar].getAddress(), BarSym.getAddress())
      << "Expected fallback def for Bar to be equal to BarSym";
}

/// By default appends LookupStates to a queue.
/// Behavior can be overridden by setting TryToGenerateOverride.
class SimpleAsyncGenerator : public DefinitionGenerator {
public:
  struct SuspendedLookupInfo {
    LookupState LS;
    LookupKind K;
    JITDylibSP JD;
    JITDylibLookupFlags JDLookupFlags;
    SymbolLookupSet Names;
  };

  unique_function<Error(LookupState &, LookupKind, JITDylib &,
                        JITDylibLookupFlags, const SymbolLookupSet &)>
      TryToGenerateOverride;

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Names) override {
    if (TryToGenerateOverride)
      return TryToGenerateOverride(LS, K, JD, JDLookupFlags, Names);
    Lookup = SuspendedLookupInfo{std::move(LS), K, &JD, JDLookupFlags, Names};
    return Error::success();
  }

  SuspendedLookupInfo takeLookup() {
    std::optional<SuspendedLookupInfo> Tmp;
    std::swap(Tmp, Lookup);
    return std::move(*Tmp);
  }

  std::optional<SuspendedLookupInfo> Lookup;
};

TEST_F(CoreAPIsStandardTest, SimpleAsynchronousGeneratorTest) {

  auto &G = JD.addGenerator(std::make_unique<SimpleAsyncGenerator>());

  bool LookupCompleted = false;

  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Foo),
      SymbolState::Ready,
      [&](Expected<SymbolMap> Result) {
        LookupCompleted = true;
        EXPECT_THAT_EXPECTED(Result, Succeeded());
        if (Result) {
          EXPECT_EQ(*Result, SymbolMap({{Foo, FooSym}}));
        }
      },
      NoDependenciesToRegister);

  EXPECT_FALSE(LookupCompleted);

  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));
  G.takeLookup().LS.continueLookup(Error::success());

  EXPECT_TRUE(LookupCompleted);
}

TEST_F(CoreAPIsStandardTest, ErrorFromSuspendedAsynchronousGeneratorTest) {

  auto &G = JD.addGenerator(std::make_unique<SimpleAsyncGenerator>());

  bool LookupCompleted = false;

  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Foo),
      SymbolState::Ready,
      [&](Expected<SymbolMap> Result) {
        LookupCompleted = true;
        EXPECT_THAT_EXPECTED(Result, Failed());
      },
      NoDependenciesToRegister);

  EXPECT_FALSE(LookupCompleted);

  G.takeLookup().LS.continueLookup(
      make_error<StringError>("boom", inconvertibleErrorCode()));

  EXPECT_TRUE(LookupCompleted);
}

TEST_F(CoreAPIsStandardTest, ErrorFromAutoSuspendedAsynchronousGeneratorTest) {

  auto &G = JD.addGenerator(std::make_unique<SimpleAsyncGenerator>());

  std::atomic_size_t LookupsCompleted = 0;

  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Foo),
      SymbolState::Ready,
      [&](Expected<SymbolMap> Result) {
        ++LookupsCompleted;
        EXPECT_THAT_EXPECTED(Result, Failed());
      },
      NoDependenciesToRegister);

  EXPECT_EQ(LookupsCompleted, 0U);

  // Suspend the first lookup.
  auto LS1 = std::move(G.takeLookup().LS);

  // Start a second lookup that should be auto-suspended.
  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Foo),
      SymbolState::Ready,
      [&](Expected<SymbolMap> Result) {
        ++LookupsCompleted;
        EXPECT_THAT_EXPECTED(Result, Failed());
      },
      NoDependenciesToRegister);

  EXPECT_EQ(LookupsCompleted, 0U);

  // Unsuspend the first lookup.
  LS1.continueLookup(make_error<StringError>("boom", inconvertibleErrorCode()));

  // Unsuspend the second.
  G.takeLookup().LS.continueLookup(
      make_error<StringError>("boom", inconvertibleErrorCode()));

  EXPECT_EQ(LookupsCompleted, 2U);
}

TEST_F(CoreAPIsStandardTest, BlockedGeneratorAutoSuspensionTest) {
  // Test that repeated lookups while a generator is in use cause automatic
  // lookup suspension / resumption.

  auto &G = JD.addGenerator(std::make_unique<SimpleAsyncGenerator>());

  bool Lookup1Completed = false;
  bool Lookup2Completed = false;
  bool Lookup3Completed = false;
  bool Lookup4Completed = false;

  // Add lookup 1.
  //
  // Tests that tryToGenerate-suspended lookups resume auto-suspended lookups
  // when the tryToGenerate-suspended lookup continues (i.e. the call to
  // OL_resumeLookupAfterGeneration at the top of OL_applyQueryPhase1).
  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Foo),
      SymbolState::Ready,
      [&](Expected<SymbolMap> Result) {
        Lookup1Completed = true;
        EXPECT_THAT_EXPECTED(Result, Succeeded());
        if (Result) {
          EXPECT_EQ(*Result, SymbolMap({{Foo, FooSym}}));
        }
      },
      NoDependenciesToRegister);

  // The generator should immediately see the first lookup.
  EXPECT_NE(G.Lookup, std::nullopt);

  // Add lookup 2.
  //
  // Tests that lookups that pass through tryToGenerate without being captured
  // resume auto-suspended lookups. We set a one-shot TryToGenerateOverride to
  // prevent capture of lookup 2 by tryToGenerate. This tests the call to
  // OL_resumeLookupAfterGeneration inside the generator loop.
  G.TryToGenerateOverride = [&](LookupState &LS, LookupKind K, JITDylib &JD,
                                JITDylibLookupFlags JDLookupFlags,
                                const SymbolLookupSet &Names) -> Error {
    cantFail(JD.define(absoluteSymbols({{Bar, BarSym}})));
    G.TryToGenerateOverride = nullptr;
    return Error::success();
  };

  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Bar),
      SymbolState::Ready,
      [&](Expected<SymbolMap> Result) {
        Lookup2Completed = true;
        EXPECT_THAT_EXPECTED(Result, Succeeded());
        if (Result) {
          EXPECT_EQ(*Result, SymbolMap({{Bar, BarSym}}));
        }
      },
      NoDependenciesToRegister);

  // Add lookup 3.
  //
  // Test that if a lookup's symbols have already been generated (and it
  // consequently skips the generator loop entirely) it still resumes the next
  // suspended lookup. This tests the call to OL_resumeLookupAfterGeneration
  // just above the generator loop.
  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Bar),
      SymbolState::Ready,
      [&](Expected<SymbolMap> Result) {
        Lookup3Completed = true;
        EXPECT_THAT_EXPECTED(Result, Succeeded());
        if (Result) {
          EXPECT_EQ(*Result, SymbolMap({{Bar, BarSym}}));
        }
      },
      NoDependenciesToRegister);

  // Add lookup 4.
  //
  // This is just used to verify that lookup 3 triggered resumption of the next
  // lookup as expected.
  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Baz),
      SymbolState::Ready,
      [&](Expected<SymbolMap> Result) {
        Lookup4Completed = true;
        EXPECT_THAT_EXPECTED(Result, Succeeded());
        if (Result) {
          EXPECT_EQ(*Result, SymbolMap({{Baz, BazSym}}));
        }
      },
      NoDependenciesToRegister);

  // All lookups have been started, but none should have been completed yet.
  EXPECT_FALSE(Lookup1Completed);
  EXPECT_FALSE(Lookup2Completed);
  EXPECT_FALSE(Lookup3Completed);
  EXPECT_FALSE(Lookup4Completed);

  // Start continuing lookups.

  // First Define foo and continue lookup 1. We expect this to complete lookups
  // 1, 2 and 3: the TryToGenerateOverride set above will define bar, which will
  // allow both 2 and 3 to complete.
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));
  G.takeLookup().LS.continueLookup(Error::success());

  EXPECT_TRUE(Lookup1Completed);
  EXPECT_TRUE(Lookup2Completed);
  EXPECT_TRUE(Lookup3Completed);
  EXPECT_FALSE(Lookup4Completed);
  EXPECT_NE(G.Lookup, std::nullopt);

  // Check that the most recently captured lookup is lookup 4 (for baz).
  if (G.Lookup) {
    EXPECT_EQ(G.Lookup->Names.begin()->first, Baz);
  }

  cantFail(JD.define(absoluteSymbols({{Baz, BazSym}})));
  G.takeLookup().LS.continueLookup(Error::success());

  EXPECT_TRUE(Lookup4Completed);
}

TEST_F(CoreAPIsStandardTest, FailResolution) {
  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, JITSymbolFlags::Exported | JITSymbolFlags::Weak},
                      {Bar, JITSymbolFlags::Exported | JITSymbolFlags::Weak}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        R->failMaterialization();
      });

  cantFail(JD.define(MU));

  SymbolNameSet Names({Foo, Bar});
  auto Result = ES.lookup(makeJITDylibSearchOrder(&JD), SymbolLookupSet(Names));

  EXPECT_FALSE(!!Result) << "Expected failure";
  if (!Result) {
    handleAllErrors(
        Result.takeError(),
        [&](FailedToMaterialize &F) {
          EXPECT_TRUE(F.getSymbols().count(&JD))
              << "Expected to fail on JITDylib JD";
          EXPECT_EQ(F.getSymbols().find(&JD)->second, Names)
              << "Expected to fail on symbols in Names";
        },
        [](ErrorInfoBase &EIB) {
          std::string ErrMsg;
          {
            raw_string_ostream ErrOut(ErrMsg);
            EIB.log(ErrOut);
          }
          ADD_FAILURE() << "Expected a FailedToResolve error. Got:\n" << ErrMsg;
        });
  }
}

TEST_F(CoreAPIsStandardTest, FailEmissionAfterResolution) {

  cantFail(JD.define(absoluteSymbols({{Baz, BazSym}})));

  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}, {Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        cantFail(R->notifyResolved(SymbolMap({{Foo, FooSym}, {Bar, BarSym}})));
        ES.lookup(
            LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet({Baz}), SymbolState::Resolved,
            [&](Expected<SymbolMap> Result) {
              // Called when "baz" is resolved. We don't actually depend
              // on or care about baz, but use it to trigger failure of
              // this materialization before Baz has been finalized in
              // order to test that error propagation is correct in this
              // scenario.
              cantFail(std::move(Result));
              R->failMaterialization();
            },
            NoDependenciesToRegister);
      });

  cantFail(JD.define(MU));

  auto Result =
      ES.lookup(makeJITDylibSearchOrder(&JD), SymbolLookupSet({Foo, Bar}));

  EXPECT_THAT_EXPECTED(std::move(Result), Failed())
      << "Unexpected success while trying to test error propagation";
}

TEST_F(CoreAPIsStandardTest, FailAfterPartialResolution) {

  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  // Fail materialization of bar.
  auto BarMU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        R->failMaterialization();
      });

  cantFail(JD.define(std::move(BarMU)));

  bool QueryHandlerRun = false;
  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD),
      SymbolLookupSet({Foo, Bar}), SymbolState::Resolved,
      [&](Expected<SymbolMap> Result) {
        EXPECT_THAT_EXPECTED(std::move(Result), Failed())
            << "Expected query to fail";
        QueryHandlerRun = true;
      },
      NoDependenciesToRegister);
  EXPECT_TRUE(QueryHandlerRun) << "Query handler never ran";
}

TEST_F(CoreAPIsStandardTest, FailDefineMaterializingDueToDefunctTracker) {
  // Check that a defunct resource tracker causes defineMaterializing to error
  // immediately.

  std::unique_ptr<MaterializationResponsibility> FooMR;
  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        FooMR = std::move(R);
      });

  auto RT = JD.createResourceTracker();
  cantFail(JD.define(std::move(MU), RT));

  bool OnCompletionRan = false;
  auto OnCompletion = [&](Expected<SymbolMap> Result) {
    EXPECT_THAT_EXPECTED(Result, Failed());
    OnCompletionRan = true;
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Foo), SymbolState::Ready, OnCompletion,
            NoDependenciesToRegister);

  cantFail(RT->remove());

  EXPECT_THAT_ERROR(FooMR->defineMaterializing(SymbolFlagsMap()), Failed())
      << "defineMaterializing should have failed due to a defunct tracker";

  FooMR->failMaterialization();

  EXPECT_TRUE(OnCompletionRan) << "OnCompletion handler did not run.";
}

TEST_F(CoreAPIsStandardTest, TestLookupWithUnthreadedMaterialization) {
  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, JITSymbolFlags::Exported}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        cantFail(R->notifyResolved({{Foo, FooSym}}));
        cantFail(R->notifyEmitted({}));
      });

  cantFail(JD.define(MU));

  auto FooLookupResult = cantFail(ES.lookup(makeJITDylibSearchOrder(&JD), Foo));

  EXPECT_EQ(FooLookupResult.getAddress(), FooSym.getAddress())
      << "lookup returned an incorrect address";
  EXPECT_EQ(FooLookupResult.getFlags(), FooSym.getFlags())
      << "lookup returned incorrect flags";
}

TEST_F(CoreAPIsStandardTest, TestLookupWithThreadedMaterialization) {
#if LLVM_ENABLE_THREADS

  std::mutex WorkThreadsMutex;
  std::vector<std::thread> WorkThreads;
  DispatchOverride = [&](std::unique_ptr<Task> T) {
    std::promise<void> WaitP;
    std::lock_guard<std::mutex> Lock(WorkThreadsMutex);
    WorkThreads.push_back(
        std::thread([T = std::move(T), WaitF = WaitP.get_future()]() mutable {
          WaitF.get();
          T->run();
        }));
    WaitP.set_value();
  };

  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  auto FooLookupResult = cantFail(ES.lookup(makeJITDylibSearchOrder(&JD), Foo));

  EXPECT_EQ(FooLookupResult.getAddress(), FooSym.getAddress())
      << "lookup returned an incorrect address";
  EXPECT_EQ(FooLookupResult.getFlags(), FooSym.getFlags())
      << "lookup returned incorrect flags";

  for (auto &WT : WorkThreads)
    WT.join();
#endif
}

TEST_F(CoreAPIsStandardTest, TestGetRequestedSymbolsAndReplace) {
  // Test that GetRequestedSymbols returns the set of symbols that currently
  // have pending queries, and test that MaterializationResponsibility's
  // replace method can be used to return definitions to the JITDylib in a new
  // MaterializationUnit.
  SymbolNameSet Names({Foo, Bar});

  bool FooMaterialized = false;
  bool BarMaterialized = false;

  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}, {Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        auto Requested = R->getRequestedSymbols();
        EXPECT_EQ(Requested.size(), 1U) << "Expected one symbol requested";
        EXPECT_EQ(*Requested.begin(), Foo) << "Expected \"Foo\" requested";

        auto NewMU = std::make_unique<SimpleMaterializationUnit>(
            SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
            [&](std::unique_ptr<MaterializationResponsibility> R2) {
              cantFail(R2->notifyResolved(SymbolMap({{Bar, BarSym}})));
              cantFail(R2->notifyEmitted({}));
              BarMaterialized = true;
            });

        cantFail(R->replace(std::move(NewMU)));

        cantFail(R->notifyResolved(SymbolMap({{Foo, FooSym}})));
        cantFail(R->notifyEmitted({}));

        FooMaterialized = true;
      });

  cantFail(JD.define(MU));

  EXPECT_FALSE(FooMaterialized) << "Foo should not be materialized yet";
  EXPECT_FALSE(BarMaterialized) << "Bar should not be materialized yet";

  auto FooSymResult = cantFail(ES.lookup(makeJITDylibSearchOrder(&JD), Foo));
  EXPECT_EQ(FooSymResult.getAddress(), FooSym.getAddress())
      << "Address mismatch for Foo";

  EXPECT_TRUE(FooMaterialized) << "Foo should be materialized now";
  EXPECT_FALSE(BarMaterialized) << "Bar still should not be materialized";

  auto BarSymResult = cantFail(ES.lookup(makeJITDylibSearchOrder(&JD), Bar));
  EXPECT_EQ(BarSymResult.getAddress(), BarSym.getAddress())
      << "Address mismatch for Bar";
  EXPECT_TRUE(BarMaterialized) << "Bar should be materialized now";
}

TEST_F(CoreAPIsStandardTest, TestMaterializationResponsibilityDelegation) {
  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}, {Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        auto R2 = cantFail(R->delegate({Bar}));

        cantFail(R->notifyResolved({{Foo, FooSym}}));
        cantFail(R->notifyEmitted({}));
        cantFail(R2->notifyResolved({{Bar, BarSym}}));
        cantFail(R2->notifyEmitted({}));
      });

  cantFail(JD.define(MU));

  auto Result =
      ES.lookup(makeJITDylibSearchOrder(&JD), SymbolLookupSet({Foo, Bar}));

  EXPECT_TRUE(!!Result) << "Result should be a success value";
  EXPECT_EQ(Result->count(Foo), 1U) << "\"Foo\" entry missing";
  EXPECT_EQ(Result->count(Bar), 1U) << "\"Bar\" entry missing";
  EXPECT_EQ((*Result)[Foo].getAddress(), FooSym.getAddress())
      << "Address mismatch for \"Foo\"";
  EXPECT_EQ((*Result)[Bar].getAddress(), BarSym.getAddress())
      << "Address mismatch for \"Bar\"";
}

TEST_F(CoreAPIsStandardTest, TestMaterializeWeakSymbol) {
  // Confirm that once a weak definition is selected for materialization it is
  // treated as strong.
  JITSymbolFlags WeakExported = JITSymbolFlags::Exported;
  WeakExported &= JITSymbolFlags::Weak;

  std::unique_ptr<MaterializationResponsibility> FooR;
  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        FooR = std::move(R);
      });

  cantFail(JD.define(MU));
  auto OnCompletion = [](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
  };

  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet({Foo}), SymbolState::Ready, std::move(OnCompletion),
            NoDependenciesToRegister);

  auto MU2 = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, JITSymbolFlags::Exported}}),
      [](std::unique_ptr<MaterializationResponsibility> R) {
        llvm_unreachable("This unit should never be materialized");
      });

  auto Err = JD.define(MU2);
  EXPECT_TRUE(!!Err) << "Expected failure value";
  EXPECT_TRUE(Err.isA<DuplicateDefinition>())
      << "Expected a duplicate definition error";
  consumeError(std::move(Err));

  // No dependencies registered, can't fail:
  cantFail(FooR->notifyResolved(SymbolMap({{Foo, FooSym}})));
  cantFail(FooR->notifyEmitted({}));
}

static bool linkOrdersEqual(const std::vector<JITDylibSP> &LHS,
                            ArrayRef<JITDylib *> RHS) {
  if (LHS.size() != RHS.size())
    return false;
  auto *RHSE = RHS.begin();
  for (auto &LHSE : LHS)
    if (LHSE.get() != *RHSE)
      return false;
    else
      ++RHSE;
  return true;
}

TEST(JITDylibTest, GetDFSLinkOrderTree) {
  // Test that DFS ordering behaves as expected when the linkage relationships
  // form a tree.

  ExecutionSession ES{std::make_unique<UnsupportedExecutorProcessControl>()};
  auto _ = make_scope_exit([&]() { cantFail(ES.endSession()); });

  auto &LibA = ES.createBareJITDylib("A");
  auto &LibB = ES.createBareJITDylib("B");
  auto &LibC = ES.createBareJITDylib("C");
  auto &LibD = ES.createBareJITDylib("D");
  auto &LibE = ES.createBareJITDylib("E");
  auto &LibF = ES.createBareJITDylib("F");

  // Linkage relationships:
  // A --- B -- D
  //  \      \- E
  //    \- C -- F
  LibA.setLinkOrder(makeJITDylibSearchOrder({&LibB, &LibC}));
  LibB.setLinkOrder(makeJITDylibSearchOrder({&LibD, &LibE}));
  LibC.setLinkOrder(makeJITDylibSearchOrder({&LibF}));

  auto DFSOrderFromB = cantFail(JITDylib::getDFSLinkOrder({&LibB}));
  EXPECT_TRUE(linkOrdersEqual(DFSOrderFromB, {&LibB, &LibD, &LibE}))
      << "Incorrect DFS link order for LibB";

  auto DFSOrderFromA = cantFail(JITDylib::getDFSLinkOrder({&LibA}));
  EXPECT_TRUE(linkOrdersEqual(DFSOrderFromA,
                              {&LibA, &LibB, &LibD, &LibE, &LibC, &LibF}))
      << "Incorrect DFS link order for libA";

  auto DFSOrderFromAB = cantFail(JITDylib::getDFSLinkOrder({&LibA, &LibB}));
  EXPECT_TRUE(linkOrdersEqual(DFSOrderFromAB,
                              {&LibA, &LibB, &LibD, &LibE, &LibC, &LibF}))
      << "Incorrect DFS link order for { libA, libB }";

  auto DFSOrderFromBA = cantFail(JITDylib::getDFSLinkOrder({&LibB, &LibA}));
  EXPECT_TRUE(linkOrdersEqual(DFSOrderFromBA,
                              {&LibB, &LibD, &LibE, &LibA, &LibC, &LibF}))
      << "Incorrect DFS link order for { libB, libA }";
}

TEST(JITDylibTest, GetDFSLinkOrderDiamond) {
  // Test that DFS ordering behaves as expected when the linkage relationships
  // contain a diamond.

  ExecutionSession ES{std::make_unique<UnsupportedExecutorProcessControl>()};
  auto _ = make_scope_exit([&]() { cantFail(ES.endSession()); });

  auto &LibA = ES.createBareJITDylib("A");
  auto &LibB = ES.createBareJITDylib("B");
  auto &LibC = ES.createBareJITDylib("C");
  auto &LibD = ES.createBareJITDylib("D");

  // Linkage relationships:
  // A -- B --- D
  //  \-- C --/
  LibA.setLinkOrder(makeJITDylibSearchOrder({&LibB, &LibC}));
  LibB.setLinkOrder(makeJITDylibSearchOrder({&LibD}));
  LibC.setLinkOrder(makeJITDylibSearchOrder({&LibD}));

  auto DFSOrderFromA = cantFail(JITDylib::getDFSLinkOrder({&LibA}));
  EXPECT_TRUE(linkOrdersEqual(DFSOrderFromA, {&LibA, &LibB, &LibD, &LibC}))
      << "Incorrect DFS link order for libA";
}

TEST(JITDylibTest, GetDFSLinkOrderCycle) {
  // Test that DFS ordering behaves as expected when the linkage relationships
  // contain a cycle.

  ExecutionSession ES{std::make_unique<UnsupportedExecutorProcessControl>()};
  auto _ = make_scope_exit([&]() { cantFail(ES.endSession()); });

  auto &LibA = ES.createBareJITDylib("A");
  auto &LibB = ES.createBareJITDylib("B");
  auto &LibC = ES.createBareJITDylib("C");

  // Linkage relationships:
  // A -- B --- C -- A
  LibA.setLinkOrder(makeJITDylibSearchOrder({&LibB}));
  LibB.setLinkOrder(makeJITDylibSearchOrder({&LibC}));
  LibC.setLinkOrder(makeJITDylibSearchOrder({&LibA}));

  auto DFSOrderFromA = cantFail(JITDylib::getDFSLinkOrder({&LibA}));
  EXPECT_TRUE(linkOrdersEqual(DFSOrderFromA, {&LibA, &LibB, &LibC}))
      << "Incorrect DFS link order for libA";

  auto DFSOrderFromB = cantFail(JITDylib::getDFSLinkOrder({&LibB}));
  EXPECT_TRUE(linkOrdersEqual(DFSOrderFromB, {&LibB, &LibC, &LibA}))
      << "Incorrect DFS link order for libB";

  auto DFSOrderFromC = cantFail(JITDylib::getDFSLinkOrder({&LibC}));
  EXPECT_TRUE(linkOrdersEqual(DFSOrderFromC, {&LibC, &LibA, &LibB}))
      << "Incorrect DFS link order for libC";
}

TEST_F(CoreAPIsStandardTest, RemoveJITDylibs) {
  // Foo will be fully materialized.
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  // Bar should not be materialized at all.
  bool BarMaterializerDestroyed = false;
  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> MR) {
        llvm_unreachable("Unexpected call to materialize");
      },
      nullptr,
      [](const JITDylib &, SymbolStringPtr Name) {
        llvm_unreachable("Unexpected call to discard");
      },
      [&]() { BarMaterializerDestroyed = true; })));

  // Baz will be in the materializing state.
  std::unique_ptr<MaterializationResponsibility> BazMR;
  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Baz, BazSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> MR) {
        BazMR = std::move(MR);
      })));

  // Lookup to force materialization of Foo.
  cantFail(ES.lookup(makeJITDylibSearchOrder(&JD), SymbolLookupSet({Foo})));

  // Start a lookup to force materialization of Baz.
  bool BazLookupFailed = false;
  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet({Baz}),
      SymbolState::Ready,
      [&](Expected<SymbolMap> Result) {
        if (!Result) {
          BazLookupFailed = true;
          consumeError(Result.takeError());
        }
      },
      NoDependenciesToRegister);

  // Remove the JITDylib.
  auto Err = ES.removeJITDylib(JD);
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());

  EXPECT_TRUE(BarMaterializerDestroyed);
  EXPECT_TRUE(BazLookupFailed);

  EXPECT_THAT_ERROR(BazMR->notifyResolved({{Baz, BazSym}}), Failed());

  EXPECT_THAT_EXPECTED(JD.getDFSLinkOrder(), Failed());

  BazMR->failMaterialization();
}

TEST(CoreAPIsExtraTest, SessionTeardownByFailedToMaterialize) {

  auto RunTestCase = []() -> Error {
    ExecutionSession ES{std::make_unique<UnsupportedExecutorProcessControl>(
        std::make_shared<SymbolStringPool>())};
    auto Foo = ES.intern("foo");
    auto FooFlags = JITSymbolFlags::Exported;

    auto &JD = ES.createBareJITDylib("Foo");
    cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
        SymbolFlagsMap({{Foo, FooFlags}}),
        [&](std::unique_ptr<MaterializationResponsibility> R) {
          R->failMaterialization();
        })));

    auto Sym = ES.lookup({&JD}, Foo);
    assert(!Sym && "Query should have failed");
    cantFail(ES.endSession());
    return Sym.takeError();
  };

  auto Err = RunTestCase();
  EXPECT_TRUE(!!Err); // Expect that error occurred.
  EXPECT_TRUE(
      Err.isA<FailedToMaterialize>()); // Expect FailedToMaterialize error.

  // Make sure that we can log errors, even though the session has been
  // destroyed.
  logAllUnhandledErrors(std::move(Err), nulls(), "");
}

} // namespace
