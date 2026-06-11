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

#include "orc-rt/sps-ci/NativeDylibManagerSPSCI.h"
#include "orc-rt/NativeDylibManager.h"
#include "orc-rt/SPSWrapperFunction.h"
#include "orc-rt/Session.h"

#include "CommonTestUtils.h"
#include "DirectCaller.h"
#include "gtest/gtest.h"

using namespace orc_rt;

namespace orc_rt {

/// SPS serialization for NativeDylibManager::LookupFlags as a bool.
///
/// Duplicated from NativeDylibManagerSPSCI.cpp so the test can serialize
/// SymbolLookupSet values when invoking the SPS wrapper via
/// SPSWrapperFunction<...>::call.
template <>
class SPSSerializationTraits<bool, NativeDylibManager::LookupFlags> {
public:
  static size_t size(NativeDylibManager::LookupFlags) { return sizeof(bool); }

  static bool serialize(SPSOutputBuffer &OB,
                        NativeDylibManager::LookupFlags L) {
    return SPSSerializationTraits<bool, bool>::serialize(
        OB, L == NativeDylibManager::RequiredSymbol);
  }

  static bool deserialize(SPSInputBuffer &IB,
                          NativeDylibManager::LookupFlags &L) {
    bool Required;
    if (!SPSSerializationTraits<bool, bool>::deserialize(IB, Required))
      return false;
    L = Required ? NativeDylibManager::RequiredSymbol
                 : NativeDylibManager::WeaklyReferencedSymbol;
    return true;
  }
};

} // namespace orc_rt

namespace {
// Local aliases for brevity in test bodies.
constexpr auto Req = NativeDylibManager::RequiredSymbol;
constexpr auto Weak = NativeDylibManager::WeaklyReferencedSymbol;
} // namespace

#ifndef NDM_TEST_LIB_PATH
#error                                                                         \
    "NDM_TEST_LIB_PATH must be defined to the path of the test shared library"
#endif

class NativeDylibManagerSPSCITest : public ::testing::Test {
protected:
  void SetUp() override {
    S = std::make_unique<Session>(mockExecutorProcessInfo(),
                                  std::make_unique<NoDispatcher>(), noErrors);
    NDM = cantFail(NativeDylibManager::Create(*S, CI));
  }

  DirectCaller caller(const char *Name) {
    return DirectCaller(nullptr, reinterpret_cast<orc_rt_WrapperFunction>(
                                     const_cast<void *>(CI.at(Name))));
  }

  template <typename OnCompleteFn>
  void spsLoad(OnCompleteFn &&OnComplete, std::string Path) {
    using SPSSig = SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSString);
    SPSWrapperFunction<SPSSig>::call(
        caller("orc_rt_ci_sps_NativeDylibManager_load"),
        std::forward<OnCompleteFn>(OnComplete), NDM.get(), std::move(Path));
  }

  template <typename OnCompleteFn>
  void spsLookup(OnCompleteFn &&OnComplete, void *Handle,
                 NativeDylibManager::SymbolLookupSet Symbols) {
    using SPSSig = SPSExpected<SPSSequence<SPSOptional<SPSExecutorAddr>>>(
        SPSExecutorAddr, SPSExecutorAddr,
        SPSSequence<SPSTuple<SPSString, bool>>);
    SPSWrapperFunction<SPSSig>::call(
        caller("orc_rt_ci_sps_NativeDylibManager_lookup"),
        std::forward<OnCompleteFn>(OnComplete), NDM.get(), Handle,
        std::move(Symbols));
  }

  SimpleSymbolTable CI;
  std::unique_ptr<Session> S;
  std::unique_ptr<NativeDylibManager> NDM;
};

TEST_F(NativeDylibManagerSPSCITest, Registration) {
  EXPECT_TRUE(CI.count("orc_rt_ci_sps_NativeDylibManager_load"));
  EXPECT_TRUE(CI.count("orc_rt_ci_sps_NativeDylibManager_lookup"));
}

TEST_F(NativeDylibManagerSPSCITest, Load) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  void *Handle = cantFail(cantFail(LoadResult.get()));
  EXPECT_NE(Handle, nullptr);
}

TEST_F(NativeDylibManagerSPSCITest, LoadNonExistent) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), "/no/such/library.dylib");
  auto Handle = cantFail(LoadResult.get());
  EXPECT_FALSE(!!Handle);
  consumeError(Handle.takeError());
}

TEST_F(NativeDylibManagerSPSCITest, LoadEmptyPathReturnsGlobalHandle) {
  // The global handle's value is implementation-defined, so verify by looking
  // up through it.
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), "");
  void *Handle = cantFail(cantFail(LoadResult.get()));

  std::future<Expected<Expected<std::vector<std::optional<void *>>>>>
      LookupResult;
  spsLookup(waitFor(LookupResult), Handle, {{"malloc", Req}});
  auto Addrs = cantFail(cantFail(LookupResult.get()));
  ASSERT_EQ(Addrs.size(), 1U);
  ASSERT_TRUE(Addrs[0].has_value())
      << "malloc should be findable via the process's global lookup handle";
  EXPECT_NE(*Addrs[0], nullptr);
}

TEST_F(NativeDylibManagerSPSCITest, LookupSingleSymbol) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  void *Handle = cantFail(cantFail(LoadResult.get()));

  std::future<Expected<Expected<std::vector<std::optional<void *>>>>>
      LookupResult;
  spsLookup(waitFor(LookupResult), Handle,
            {{"NativeDylibManagerTestFunc", Req}});
  auto Addrs = cantFail(cantFail(LookupResult.get()));
  ASSERT_EQ(Addrs.size(), 1U);
  ASSERT_TRUE(Addrs[0].has_value());
  EXPECT_NE(*Addrs[0], nullptr);

  auto *Func = reinterpret_cast<int (*)()>(*Addrs[0]);
  EXPECT_EQ(Func(), 42);
}

TEST_F(NativeDylibManagerSPSCITest, LookupMultipleSymbols) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  void *Handle = cantFail(cantFail(LoadResult.get()));

  std::future<Expected<Expected<std::vector<std::optional<void *>>>>>
      LookupResult;
  spsLookup(waitFor(LookupResult), Handle,
            {{"NativeDylibManagerTestFunc", Req},
             {"NativeDylibManagerTestFunc2", Req}});
  auto Addrs = cantFail(cantFail(LookupResult.get()));
  ASSERT_EQ(Addrs.size(), 2U);
  ASSERT_TRUE(Addrs[0].has_value());
  ASSERT_TRUE(Addrs[1].has_value());
  EXPECT_NE(*Addrs[0], nullptr);
  EXPECT_NE(*Addrs[1], nullptr);

  auto *Func1 = reinterpret_cast<int (*)()>(*Addrs[0]);
  auto *Func2 = reinterpret_cast<int (*)()>(*Addrs[1]);
  EXPECT_EQ(Func1(), 42);
  EXPECT_EQ(Func2(), 7);
}

TEST_F(NativeDylibManagerSPSCITest, LookupWeakMissingSymbol) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  void *Handle = cantFail(cantFail(LoadResult.get()));

  std::future<Expected<Expected<std::vector<std::optional<void *>>>>>
      LookupResult;
  spsLookup(waitFor(LookupResult), Handle, {{"no_such_symbol", Weak}});
  auto Addrs = cantFail(cantFail(LookupResult.get()));
  ASSERT_EQ(Addrs.size(), 1U);
  ASSERT_TRUE(Addrs[0].has_value())
      << "weak-missing symbol should be reported as a present optional";
  EXPECT_EQ(*Addrs[0], nullptr);
}

TEST_F(NativeDylibManagerSPSCITest, LookupRequiredMissingSymbol) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  void *Handle = cantFail(cantFail(LoadResult.get()));

  std::future<Expected<Expected<std::vector<std::optional<void *>>>>>
      LookupResult;
  spsLookup(waitFor(LookupResult), Handle, {{"no_such_symbol", Req}});
  auto Addrs = cantFail(cantFail(LookupResult.get()));
  ASSERT_EQ(Addrs.size(), 1U);
  EXPECT_FALSE(Addrs[0].has_value())
      << "required-missing symbol should be reported as an empty optional";
}

TEST_F(NativeDylibManagerSPSCITest, LookupMixedRequiredAndWeak) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  void *Handle = cantFail(cantFail(LoadResult.get()));

  std::future<Expected<Expected<std::vector<std::optional<void *>>>>>
      LookupResult;
  spsLookup(waitFor(LookupResult), Handle,
            {{"NativeDylibManagerTestFunc", Req}, {"no_such_symbol", Weak}});
  auto Addrs = cantFail(cantFail(LookupResult.get()));
  ASSERT_EQ(Addrs.size(), 2U);
  ASSERT_TRUE(Addrs[0].has_value());
  EXPECT_NE(*Addrs[0], nullptr);
  ASSERT_TRUE(Addrs[1].has_value())
      << "weak-missing symbol should be reported as a present optional";
  EXPECT_EQ(*Addrs[1], nullptr);
}
