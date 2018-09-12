//===----------- CoreAPIsTest.cpp - Unit tests for Core ORC APIs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  cantFail(V.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));

  auto Resolver = createSymbolResolver(
      [&](const SymbolNameSet &Symbols) { return V.lookupFlags(Symbols); },
      [&](std::shared_ptr<AsynchronousSymbolQuery> Q, SymbolNameSet Symbols) {
        return V.legacyLookup(std::move(Q), Symbols);
      });

  SymbolNameSet Symbols({Foo, Bar, Baz});

  SymbolFlagsMap SymbolFlags = Resolver->lookupFlags(Symbols);

  EXPECT_EQ(SymbolFlags.size(), 2U)
      << "lookupFlags returned the wrong number of results";
  EXPECT_EQ(SymbolFlags.count(Foo), 1U) << "Missing lookupFlags result for foo";
  EXPECT_EQ(SymbolFlags.count(Bar), 1U) << "Missing lookupFlags result for bar";
  EXPECT_EQ(SymbolFlags[Foo], FooSym.getFlags())
      << "Incorrect lookupFlags result for Foo";
  EXPECT_EQ(SymbolFlags[Bar], BarSym.getFlags())
      << "Incorrect lookupFlags result for Bar";

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

TEST(LegacyAPIInteropTest, QueryAgainstVSO) {

  ExecutionSession ES(std::make_shared<SymbolStringPool>());
  auto Foo = ES.getSymbolStringPool().intern("foo");

  auto &V = ES.createVSO("V");
  JITEvaluatedSymbol FooSym(0xdeadbeef, JITSymbolFlags::Exported);
  cantFail(V.define(absoluteSymbols({{Foo, FooSym}})));

  auto LookupFlags = [&](const SymbolNameSet &Names) {
    return V.lookupFlags(Names);
  };

  auto Lookup = [&](std::shared_ptr<AsynchronousSymbolQuery> Query,
                    SymbolNameSet Symbols) {
    return V.legacyLookup(std::move(Query), Symbols);
  };

  auto UnderlyingResolver =
      createSymbolResolver(std::move(LookupFlags), std::move(Lookup));
  JITSymbolResolverAdapter Resolver(ES, *UnderlyingResolver, nullptr);

  JITSymbolResolver::LookupSet Names{StringRef("foo")};

  auto LFR = Resolver.lookupFlags(Names);
  EXPECT_TRUE(!!LFR) << "lookupFlags failed";
  EXPECT_EQ(LFR->size(), 1U)
      << "lookupFlags returned the wrong number of results";
  EXPECT_EQ(LFR->count(*Foo), 1U)
      << "lookupFlags did not contain a result for 'foo'";
  EXPECT_EQ((*LFR)[*Foo], FooSym.getFlags())
      << "lookupFlags contained the wrong result for 'foo'";

  auto LR = Resolver.lookup(Names);
  EXPECT_TRUE(!!LR) << "lookup failed";
  EXPECT_EQ(LR->size(), 1U) << "lookup returned the wrong number of results";
  EXPECT_EQ(LR->count(*Foo), 1U) << "lookup did not contain a result for 'foo'";
  EXPECT_EQ((*LR)[*Foo].getFlags(), FooSym.getFlags())
      << "lookup returned the wrong result for flags of 'foo'";
  EXPECT_EQ((*LR)[*Foo].getAddress(), FooSym.getAddress())
      << "lookup returned the wrong result for address of 'foo'";
}

TEST(LegacyAPIInteropTset, LegacyLookupHelpersFn) {
  constexpr JITTargetAddress FooAddr = 0xdeadbeef;
  JITSymbolFlags FooFlags = JITSymbolFlags::Exported;

  bool BarMaterialized = false;
  constexpr JITTargetAddress BarAddr = 0xcafef00d;
  JITSymbolFlags BarFlags = static_cast<JITSymbolFlags::FlagNames>(
      JITSymbolFlags::Exported | JITSymbolFlags::Weak);

  auto LegacyLookup = [&](const std::string &Name) -> JITSymbol {
    if (Name == "foo")
      return {FooAddr, FooFlags};

    if (Name == "bar") {
      auto BarMaterializer = [&]() -> Expected<JITTargetAddress> {
        BarMaterialized = true;
        return BarAddr;
      };

      return {BarMaterializer, BarFlags};
    }

    return nullptr;
  };

  ExecutionSession ES;
  auto Foo = ES.getSymbolStringPool().intern("foo");
  auto Bar = ES.getSymbolStringPool().intern("bar");
  auto Baz = ES.getSymbolStringPool().intern("baz");

  SymbolNameSet Symbols({Foo, Bar, Baz});

  auto SymbolFlags = lookupFlagsWithLegacyFn(Symbols, LegacyLookup);

  EXPECT_TRUE(!!SymbolFlags) << "Expected lookupFlagsWithLegacyFn to succeed";
  EXPECT_EQ(SymbolFlags->size(), 2U) << "Wrong number of flags returned";
  EXPECT_EQ(SymbolFlags->count(Foo), 1U) << "Flags for foo missing";
  EXPECT_EQ(SymbolFlags->count(Bar), 1U) << "Flags for foo missing";
  EXPECT_EQ((*SymbolFlags)[Foo], FooFlags) << "Wrong flags for foo";
  EXPECT_EQ((*SymbolFlags)[Bar], BarFlags) << "Wrong flags for foo";
  EXPECT_FALSE(BarMaterialized)
      << "lookupFlags should not have materialized bar";

  bool OnResolvedRun = false;
  bool OnReadyRun = false;
  auto OnResolved = [&](Expected<SymbolMap> Result) {
    OnResolvedRun = true;
    EXPECT_TRUE(!!Result) << "lookuWithLegacy failed to resolve";

    EXPECT_EQ(Result->size(), 2U) << "Wrong number of symbols resolved";
    EXPECT_EQ(Result->count(Foo), 1U) << "Result for foo missing";
    EXPECT_EQ(Result->count(Bar), 1U) << "Result for bar missing";
    EXPECT_EQ((*Result)[Foo].getAddress(), FooAddr) << "Wrong address for foo";
    EXPECT_EQ((*Result)[Foo].getFlags(), FooFlags) << "Wrong flags for foo";
    EXPECT_EQ((*Result)[Bar].getAddress(), BarAddr) << "Wrong address for bar";
    EXPECT_EQ((*Result)[Bar].getFlags(), BarFlags) << "Wrong flags for bar";
  };
  auto OnReady = [&](Error Err) {
    EXPECT_FALSE(!!Err) << "Finalization unexpectedly failed";
    OnReadyRun = true;
  };

  AsynchronousSymbolQuery Q({Foo, Bar}, OnResolved, OnReady);
  auto Unresolved = lookupWithLegacyFn(ES, Q, Symbols, LegacyLookup);

  EXPECT_TRUE(OnResolvedRun) << "OnResolved was not run";
  EXPECT_TRUE(OnReadyRun) << "OnReady was not run";
  EXPECT_EQ(Unresolved.size(), 1U) << "Expected one unresolved symbol";
  EXPECT_EQ(Unresolved.count(Baz), 1U) << "Expected baz to be unresolved";
}

} // namespace
