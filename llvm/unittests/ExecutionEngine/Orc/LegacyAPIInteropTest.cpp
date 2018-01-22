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

class SimpleORCResolver : public SymbolResolver {
public:
  using LookupFlagsFn = std::function<LookupFlagsResult(const SymbolNameSet &)>;
  using LookupFn = std::function<SymbolNameSet(AsynchronousSymbolQuery &Q,
                                               SymbolNameSet Symbols)>;

  SimpleORCResolver(LookupFlagsFn LookupFlags, LookupFn Lookup)
      : LookupFlags(std::move(LookupFlags)), Lookup(std::move(Lookup)) {}

  LookupFlagsResult lookupFlags(const SymbolNameSet &Symbols) override {
    return LookupFlags(Symbols);
  }

  SymbolNameSet lookup(AsynchronousSymbolQuery &Query,
                       SymbolNameSet Symbols) override {
    return Lookup(Query, std::move(Symbols));
  };

private:
  LookupFlagsFn LookupFlags;
  LookupFn Lookup;
};

namespace {

TEST(LegacyAPIInteropTest, QueryAgainstVSO) {

  SymbolStringPool SP;
  ExecutionSession ES(SP);
  auto Foo = SP.intern("foo");

  VSO V;
  SymbolMap Defs;
  JITEvaluatedSymbol FooSym(0xdeadbeef, JITSymbolFlags::Exported);
  Defs[Foo] = FooSym;
  cantFail(V.define(std::move(Defs)));

  auto LookupFlags = [&](const SymbolNameSet &Names) {
    return V.lookupFlags(Names);
  };

  auto Lookup = [&](AsynchronousSymbolQuery &Query, SymbolNameSet Symbols) {
    auto R = V.lookup(Query, Symbols);
    EXPECT_TRUE(R.MaterializationWork.empty())
        << "Query resulted in unexpected materialization work";
    return std::move(R.UnresolvedSymbols);
  };

  SimpleORCResolver UnderlyingResolver(std::move(LookupFlags),
                                       std::move(Lookup));
  JITSymbolResolverAdapter Resolver(ES, UnderlyingResolver);

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

} // namespace
