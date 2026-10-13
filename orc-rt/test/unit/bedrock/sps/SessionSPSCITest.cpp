//===- SessionSPSCITest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for Session's SPS Controller Interface.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/SessionSPSCI.h"
#include "orc-rt/bedrock/Session.h"
#include "orc-rt/support/sps/SPSSymbolLookupSet.h"
#include "orc-rt/support/sps/SPSWrapperFunction.h"

#include "BedrockTestUtils.h"
#include "CommonTestUtils.h"
#include "DirectCaller.h"
#include "ErrorMatchers.h"
#include "gtest/gtest.h"

#include <future>

using namespace orc_rt;
using namespace orc_rt::test;

namespace {
// Local aliases for brevity in test bodies.
constexpr auto Req = SymbolLookupFlags::RequiredSymbol;
constexpr auto Weak = SymbolLookupFlags::WeaklyReferencedSymbol;
} // namespace

class SessionSPSCITest : public ::testing::Test {
protected:
  void SetUp() override {
    S = std::make_unique<Session>(mockExecutorProcessInfo(), noDispatch,
                                  noErrors);
    ASSERT_THAT_ERROR(sps_ci::addSession(CI), Succeeded());

    // Instance symbols are keyed by linker-level name, so register them with
    // SymbolNameSpec::linker to keep the lookup names platform-independent.
    std::pair<SymbolNameSpec, const void *> Entries[] = {
        {SymbolNameSpec::linker("orc_rt_test_Foo"), &Foo},
        {SymbolNameSpec::linker("orc_rt_test_Bar"), &Bar}};
    ASSERT_THAT_ERROR(S->addInstanceSymbols(Entries), Succeeded());
  }

  Expected<SymbolLookupResult> spsLookup(SymbolLookupSet Symbols) {
    using SPSSig = SPSSymbolLookupResult(SPSSymbolLookupSet);
    std::future<Expected<SymbolLookupResult>> Result;
    SPSWrapperFunction<SPSSig>::call(
        caller(orc_rt_ci_sps_Session_lookupInstanceSymbols), waitFor(Result),
        std::move(Symbols));
    return Result.get();
  }

  DirectCaller caller(orc_rt_WrapperFunction Fn) { return {wrap(S.get()), Fn}; }

  int Foo = 0;
  int Bar = 0;
  SimpleSymbolTable CI;
  std::unique_ptr<Session> S;
};

TEST_F(SessionSPSCITest, Registration) {
  EXPECT_TRUE(CI.count(
      SymbolNameSpec::c("orc_rt_ci_sps_Session_lookupInstanceSymbols")));
}

TEST_F(SessionSPSCITest, LookupRegisteredSymbol) {
  auto Addrs = spsLookup({{"orc_rt_test_Foo", Req}});
  ASSERT_THAT_EXPECTED(Addrs, Succeeded());
  ASSERT_EQ(Addrs->size(), 1U);
  ASSERT_TRUE((*Addrs)[0].has_value());
  EXPECT_EQ(*(*Addrs)[0], &Foo);
}

TEST_F(SessionSPSCITest, LookupMissingRequiredSymbol) {
  auto Addrs = spsLookup({{"orc_rt_test_Missing", Req}});
  ASSERT_THAT_EXPECTED(Addrs, Succeeded());
  ASSERT_EQ(Addrs->size(), 1U);
  EXPECT_FALSE((*Addrs)[0].has_value())
      << "required-missing symbol should be reported as an empty optional";
}

TEST_F(SessionSPSCITest, LookupMissingWeakSymbol) {
  auto Addrs = spsLookup({{"orc_rt_test_Missing", Weak}});
  ASSERT_THAT_EXPECTED(Addrs, Succeeded());
  ASSERT_EQ(Addrs->size(), 1U);
  ASSERT_TRUE((*Addrs)[0].has_value())
      << "weak-missing symbol should be reported as a present optional";
  EXPECT_EQ(*(*Addrs)[0], nullptr);
}

TEST_F(SessionSPSCITest, LookupMixedSetPreservesOrder) {
  auto Addrs = spsLookup({{"orc_rt_test_Bar", Req},
                          {"orc_rt_test_Missing", Req},
                          {"orc_rt_test_Foo", Weak},
                          {"orc_rt_test_Missing", Weak}});
  ASSERT_THAT_EXPECTED(Addrs, Succeeded());
  ASSERT_EQ(Addrs->size(), 4U);
  ASSERT_TRUE((*Addrs)[0].has_value());
  EXPECT_EQ(*(*Addrs)[0], &Bar);
  EXPECT_FALSE((*Addrs)[1].has_value());
  ASSERT_TRUE((*Addrs)[2].has_value());
  EXPECT_EQ(*(*Addrs)[2], &Foo);
  ASSERT_TRUE((*Addrs)[3].has_value());
  EXPECT_EQ(*(*Addrs)[3], nullptr);
}

TEST_F(SessionSPSCITest, LookupEmptySet) {
  auto Addrs = spsLookup({});
  ASSERT_THAT_EXPECTED(Addrs, Succeeded());
  EXPECT_TRUE(Addrs->empty());
}
