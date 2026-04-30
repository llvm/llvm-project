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

  void TearDown() override {
    if (NDM) {
      std::future<void> F;
      NDM->onShutdown(waitFor(F));
      F.get();
    }
  }

  DirectCaller caller(const char *Name) {
    return DirectCaller(nullptr, reinterpret_cast<orc_rt_WrapperFunction>(
                                     const_cast<void *>(CI.at(Name))));
  }

  template <typename OnCompleteFn>
  void spsLoad(OnCompleteFn &&OnComplete, std::string Path) {
    using SPSSig = SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSString);
    SPSWrapperFunction<SPSSig>::call(
        caller("orc_rt_sps_ci_NativeDylibManager_load_sps_wrapper"),
        std::forward<OnCompleteFn>(OnComplete), NDM.get(), std::move(Path));
  }

  template <typename OnCompleteFn>
  void spsUnload(OnCompleteFn &&OnComplete, void *Handle) {
    using SPSSig = SPSError(SPSExecutorAddr, SPSExecutorAddr);
    SPSWrapperFunction<SPSSig>::call(
        caller("orc_rt_sps_ci_NativeDylibManager_unload_sps_wrapper"),
        std::forward<OnCompleteFn>(OnComplete), NDM.get(), Handle);
  }

  template <typename OnCompleteFn>
  void spsLookup(OnCompleteFn &&OnComplete, void *Handle,
                 std::vector<std::string> Names) {
    using SPSSig = SPSExpected<SPSSequence<SPSExecutorAddr>>(
        SPSExecutorAddr, SPSExecutorAddr, SPSSequence<SPSString>);
    SPSWrapperFunction<SPSSig>::call(
        caller("orc_rt_sps_ci_NativeDylibManager_lookup_sps_wrapper"),
        std::forward<OnCompleteFn>(OnComplete), NDM.get(), Handle,
        std::move(Names));
  }

  SimpleSymbolTable CI;
  std::unique_ptr<Session> S;
  std::unique_ptr<NativeDylibManager> NDM;
};

TEST_F(NativeDylibManagerSPSCITest, Registration) {
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_NativeDylibManager_load_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_NativeDylibManager_unload_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_NativeDylibManager_lookup_sps_wrapper"));
}

TEST_F(NativeDylibManagerSPSCITest, LoadAndUnload) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  void *Handle = cantFail(cantFail(LoadResult.get()));
  EXPECT_NE(Handle, nullptr);

  std::future<Expected<Error>> UnloadResult;
  spsUnload(waitFor(UnloadResult), Handle);
  cantFail(cantFail(UnloadResult.get()));
}

TEST_F(NativeDylibManagerSPSCITest, LoadNonExistent) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), "/no/such/library.dylib");
  auto Handle = cantFail(LoadResult.get());
  EXPECT_FALSE(!!Handle);
  consumeError(Handle.takeError());
}

TEST_F(NativeDylibManagerSPSCITest, UnloadUnrecognizedHandle) {
  // Use Session object address as bogus handle.
  void *BadHandle = reinterpret_cast<void *>(&S);
  std::future<Expected<Error>> UnloadResult;
  spsUnload(waitFor(UnloadResult), BadHandle);
  auto Handle = cantFail(UnloadResult.get());
  EXPECT_TRUE(!!Handle);
  consumeError(std::move(Handle));
}

TEST_F(NativeDylibManagerSPSCITest, LookupSingleSymbol) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  void *Handle = cantFail(cantFail(LoadResult.get()));

  std::future<Expected<Expected<std::vector<void *>>>> LookupResult;
  spsLookup(waitFor(LookupResult), Handle, {"NativeDylibManagerTestFunc"});
  auto Addrs = cantFail(cantFail(LookupResult.get()));
  ASSERT_EQ(Addrs.size(), 1U);
  EXPECT_NE(Addrs[0], nullptr);

  auto *Func = reinterpret_cast<int (*)()>(Addrs[0]);
  EXPECT_EQ(Func(), 42);

  std::future<Expected<Error>> UnloadResult;
  spsUnload(waitFor(UnloadResult), Handle);
  cantFail(cantFail(UnloadResult.get()));
}

TEST_F(NativeDylibManagerSPSCITest, LookupMultipleSymbols) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  void *Handle = cantFail(cantFail(LoadResult.get()));

  std::future<Expected<Expected<std::vector<void *>>>> LookupResult;
  spsLookup(waitFor(LookupResult), Handle,
            {"NativeDylibManagerTestFunc", "NativeDylibManagerTestFunc2"});
  auto Addrs = cantFail(cantFail(LookupResult.get()));
  ASSERT_EQ(Addrs.size(), 2U);
  EXPECT_NE(Addrs[0], nullptr);
  EXPECT_NE(Addrs[1], nullptr);

  auto *Func1 = reinterpret_cast<int (*)()>(Addrs[0]);
  auto *Func2 = reinterpret_cast<int (*)()>(Addrs[1]);
  EXPECT_EQ(Func1(), 42);
  EXPECT_EQ(Func2(), 7);

  std::future<Expected<Error>> UnloadResult;
  spsUnload(waitFor(UnloadResult), Handle);
  cantFail(cantFail(UnloadResult.get()));
}

TEST_F(NativeDylibManagerSPSCITest, LookupNonExistentSymbol) {
  std::future<Expected<Expected<void *>>> LoadResult;
  spsLoad(waitFor(LoadResult), NDM_TEST_LIB_PATH);
  void *Handle = cantFail(cantFail(LoadResult.get()));

  std::future<Expected<Expected<std::vector<void *>>>> LookupResult;
  spsLookup(waitFor(LookupResult), Handle, {"no_such_symbol"});
  auto Addrs = cantFail(cantFail(LookupResult.get()));
  ASSERT_EQ(Addrs.size(), 1U);
  EXPECT_EQ(Addrs[0], nullptr);

  std::future<Expected<Error>> UnloadResult;
  spsUnload(waitFor(UnloadResult), Handle);
  cantFail(cantFail(UnloadResult.get()));
}

TEST_F(NativeDylibManagerSPSCITest, LookupOnUnrecognizedHandle) {
  // Use Session object address as bogus handle.
  void *BadHandle = reinterpret_cast<void *>(&S);
  std::future<Expected<Expected<std::vector<void *>>>> LookupResult;
  spsLookup(waitFor(LookupResult), BadHandle, {"NativeDylibManagerTestFunc"});
  auto Addrs = cantFail(LookupResult.get());
  EXPECT_FALSE(!!Addrs);
  consumeError(Addrs.takeError());
}
