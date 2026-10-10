//===- NativeDylibManagerSPSCITest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for NativeDylibManager's SPS Controller Interface.
//
//===----------------------------------------------------------------------===//
#ifndef _WIN32
#include "orc-rt/bedrock/sps/NativeDylibManagerSPSCI.h"
#include "orc-rt/bedrock/NativeDylibManager.h"
#include "orc-rt/bedrock/Session.h"
#include "orc-rt/support/sps/SPSSymbolLookupSet.h"
#include "orc-rt/support/sps/SPSWrapperFunction.h"

#include "BedrockTestUtils.h"
#include "CommonTestUtils.h"
#include "DirectCaller.h"
#include "ErrorMatchers.h"
#include "gtest/gtest.h"

using namespace orc_rt;
using namespace orc_rt::test;

using ::testing::Ne;

namespace {
// Local aliases for brevity in test bodies.
constexpr auto Req = SymbolLookupFlags::RequiredSymbol;
constexpr auto Weak = SymbolLookupFlags::WeaklyReferencedSymbol;
} // namespace

// Wrap a symbol-name string literal in the platform's linker-mangling.
// NativeDylibManager::lookup takes linker-mangled names; on Darwin the
// linker prefixes C names with '_'.
#if defined(__APPLE__)
#define MANGLED(name) "_" name
#else
#define MANGLED(name) name
#endif

#ifndef NDM_TEST_LIB_PATH
#error                                                                         \
    "NDM_TEST_LIB_PATH must be defined to the path of the test shared library"
#endif

static DirectCaller caller(orc_rt_WrapperFunction Fn) { return {nullptr, Fn}; }

class NativeDylibManagerSPSCITest : public ::testing::Test {
protected:
  void SetUp() override {
    S = std::make_unique<Session>(mockExecutorProcessInfo(), noDispatch,
                                  noErrors);
    auto NDMOrErr = NativeDylibManager::Create(*S, CI);
    ASSERT_THAT_EXPECTED(NDMOrErr, Succeeded());
    NDM = std::move(*NDMOrErr);
  }

  template <typename OnCompleteFn>
  void spsLoad(OnCompleteFn &&OnComplete, std::string Path) {
    using SPSSig = SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSString);
    SPSWrapperFunction<SPSSig>::call(
        caller(orc_rt_ci_sps_NativeDylibManager_load),
        std::forward<OnCompleteFn>(OnComplete), NDM.get(), std::move(Path));
  }

  template <typename OnCompleteFn>
  void spsLookup(OnCompleteFn &&OnComplete, void *Handle,
                 SymbolLookupSet Symbols) {
    using SPSSig = SPSExpected<SPSSequence<SPSOptional<SPSExecutorAddr>>>(
        SPSExecutorAddr, SPSExecutorAddr,
        SPSSequence<SPSTuple<SPSString, bool>>);
    SPSWrapperFunction<SPSSig>::call(
        caller(orc_rt_ci_sps_NativeDylibManager_lookup),
        std::forward<OnCompleteFn>(OnComplete), NDM.get(), Handle,
        std::move(Symbols));
  }

  SimpleSymbolTable CI;
  std::unique_ptr<Session> S;
  std::unique_ptr<NativeDylibManager> NDM;
};

TEST_F(NativeDylibManagerSPSCITest, Registration) {
  EXPECT_TRUE(
      CI.count(SymbolNameSpec::c("orc_rt_ci_sps_NativeDylibManager_load")));
  EXPECT_TRUE(
      CI.count(SymbolNameSpec::c("orc_rt_ci_sps_NativeDylibManager_lookup")));
}

TEST_F(NativeDylibManagerSPSCITest, Load) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  auto HandleOrErr = LoadResult.get();
  ASSERT_THAT_EXPECTED(HandleOrErr, Succeeded());
  EXPECT_THAT_EXPECTED(*HandleOrErr, HasValue(Ne(nullptr)));
}

TEST_F(NativeDylibManagerSPSCITest, LoadNonExistent) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), "/no/such/library.dylib");
  auto HandleOrErr = LoadResult.get();
  ASSERT_THAT_EXPECTED(HandleOrErr, Succeeded());
  EXPECT_THAT_EXPECTED(*HandleOrErr, Failed());
}

TEST_F(NativeDylibManagerSPSCITest, LoadEmptyPathReturnsGlobalHandle) {
  // The global handle's value is implementation-defined, so verify by looking
  // up through it.
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), "");
  auto Handle = LoadResult.get();
  ASSERT_THAT_EXPECTED(Handle, Succeeded());
  ASSERT_THAT_EXPECTED(*Handle, Succeeded());

  std::future<Expected<Expected<SymbolLookupResult>>> LookupResult;
  spsLookup(waitFor(LookupResult), **Handle, {{MANGLED("malloc"), Req}});
  auto AddrsOrErr = LookupResult.get();
  ASSERT_THAT_EXPECTED(AddrsOrErr, Succeeded());
  ASSERT_THAT_EXPECTED(*AddrsOrErr, Succeeded());
  auto &Addrs = **AddrsOrErr;
  ASSERT_EQ(Addrs.size(), 1U);
  ASSERT_TRUE(Addrs[0].has_value())
      << "malloc should be findable via the process's global lookup handle";
  EXPECT_NE(*Addrs[0], nullptr);
}

TEST_F(NativeDylibManagerSPSCITest, LookupSingleSymbol) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  auto Handle = LoadResult.get();
  ASSERT_THAT_EXPECTED(Handle, Succeeded());
  ASSERT_THAT_EXPECTED(*Handle, Succeeded());

  std::future<Expected<Expected<SymbolLookupResult>>> LookupResult;
  spsLookup(waitFor(LookupResult), **Handle,
            {{MANGLED("NativeDylibManagerTestFunc"), Req}});
  auto AddrsOrErr = LookupResult.get();
  ASSERT_THAT_EXPECTED(AddrsOrErr, Succeeded());
  ASSERT_THAT_EXPECTED(*AddrsOrErr, Succeeded());
  auto &Addrs = **AddrsOrErr;
  ASSERT_EQ(Addrs.size(), 1U);
  ASSERT_TRUE(Addrs[0].has_value());
  EXPECT_NE(*Addrs[0], nullptr);

  auto *Func = reinterpret_cast<int (*)()>(const_cast<void *>(*Addrs[0]));
  EXPECT_EQ(Func(), 42);
}

TEST_F(NativeDylibManagerSPSCITest, LookupMultipleSymbols) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  auto Handle = LoadResult.get();
  ASSERT_THAT_EXPECTED(Handle, Succeeded());
  ASSERT_THAT_EXPECTED(*Handle, Succeeded());

  std::future<Expected<Expected<SymbolLookupResult>>> LookupResult;
  spsLookup(waitFor(LookupResult), **Handle,
            {{MANGLED("NativeDylibManagerTestFunc"), Req},
             {MANGLED("NativeDylibManagerTestFunc2"), Req}});
  auto AddrsOrErr = LookupResult.get();
  ASSERT_THAT_EXPECTED(AddrsOrErr, Succeeded());
  ASSERT_THAT_EXPECTED(*AddrsOrErr, Succeeded());
  auto &Addrs = **AddrsOrErr;
  ASSERT_EQ(Addrs.size(), 2U);
  ASSERT_TRUE(Addrs[0].has_value());
  ASSERT_TRUE(Addrs[1].has_value());
  EXPECT_NE(*Addrs[0], nullptr);
  EXPECT_NE(*Addrs[1], nullptr);

  auto *Func1 = reinterpret_cast<int (*)()>(const_cast<void *>(*Addrs[0]));
  auto *Func2 = reinterpret_cast<int (*)()>(const_cast<void *>(*Addrs[1]));
  EXPECT_EQ(Func1(), 42);
  EXPECT_EQ(Func2(), 7);
}

TEST_F(NativeDylibManagerSPSCITest, LookupWeakMissingSymbol) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  auto Handle = LoadResult.get();
  ASSERT_THAT_EXPECTED(Handle, Succeeded());
  ASSERT_THAT_EXPECTED(*Handle, Succeeded());

  std::future<Expected<Expected<SymbolLookupResult>>> LookupResult;
  spsLookup(waitFor(LookupResult), **Handle,
            {{MANGLED("no_such_symbol"), Weak}});
  auto AddrsOrErr = LookupResult.get();
  ASSERT_THAT_EXPECTED(AddrsOrErr, Succeeded());
  ASSERT_THAT_EXPECTED(*AddrsOrErr, Succeeded());
  auto &Addrs = **AddrsOrErr;
  ASSERT_EQ(Addrs.size(), 1U);
  ASSERT_TRUE(Addrs[0].has_value())
      << "weak-missing symbol should be reported as a present optional";
  EXPECT_EQ(*Addrs[0], nullptr);
}

TEST_F(NativeDylibManagerSPSCITest, LookupRequiredMissingSymbol) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  auto Handle = LoadResult.get();
  ASSERT_THAT_EXPECTED(Handle, Succeeded());
  ASSERT_THAT_EXPECTED(*Handle, Succeeded());

  std::future<Expected<Expected<SymbolLookupResult>>> LookupResult;
  spsLookup(waitFor(LookupResult), **Handle,
            {{MANGLED("no_such_symbol"), Req}});
  auto AddrsOrErr = LookupResult.get();
  ASSERT_THAT_EXPECTED(AddrsOrErr, Succeeded());
  ASSERT_THAT_EXPECTED(*AddrsOrErr, Succeeded());
  auto &Addrs = **AddrsOrErr;
  ASSERT_EQ(Addrs.size(), 1U);
  EXPECT_FALSE(Addrs[0].has_value())
      << "required-missing symbol should be reported as an empty optional";
}

TEST_F(NativeDylibManagerSPSCITest, LookupMixedRequiredAndWeak) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  auto Handle = LoadResult.get();
  ASSERT_THAT_EXPECTED(Handle, Succeeded());
  ASSERT_THAT_EXPECTED(*Handle, Succeeded());

  std::future<Expected<Expected<SymbolLookupResult>>> LookupResult;
  spsLookup(waitFor(LookupResult), **Handle,
            {{MANGLED("NativeDylibManagerTestFunc"), Req},
             {MANGLED("no_such_symbol"), Weak}});
  auto AddrsOrErr = LookupResult.get();
  ASSERT_THAT_EXPECTED(AddrsOrErr, Succeeded());
  ASSERT_THAT_EXPECTED(*AddrsOrErr, Succeeded());
  auto &Addrs = **AddrsOrErr;
  ASSERT_EQ(Addrs.size(), 2U);
  ASSERT_TRUE(Addrs[0].has_value());
  EXPECT_NE(*Addrs[0], nullptr);
  ASSERT_TRUE(Addrs[1].has_value())
      << "weak-missing symbol should be reported as a present optional";
  EXPECT_EQ(*Addrs[1], nullptr);
}
#endif
