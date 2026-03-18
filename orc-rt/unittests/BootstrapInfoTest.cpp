//===- BootstrapInfoTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's BootstrapInfo.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/BootstrapInfo.h"
#include "orc-rt/Session.h"
#include "orc-rt/TaskDispatcher.h"
#include "gtest/gtest.h"

#include "CommonTestUtils.h"

using namespace orc_rt;

TEST(BootstrapInfoTest, ExplicitConstruction) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  BootstrapInfo BI(S);
  EXPECT_EQ(&BI.session(), &S);
  EXPECT_TRUE(BI.symbols().empty());
  EXPECT_TRUE(BI.values().empty());
}

TEST(BootstrapInfoTest, ExplicitConstructionWithSymbolsAndValues) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  int X = 0;
  SimpleSymbolTable Symbols;
  std::pair<const char *, void *> Syms[] = {{"orc_rt_X", &X}};
  cantFail(Symbols.addUnique(Syms));

  BootstrapInfo::ValueMap Values;
  Values["key"] = "value";

  BootstrapInfo BI(S, std::move(Symbols), std::move(Values));
  EXPECT_EQ(BI.symbols().size(), 1U);
  EXPECT_TRUE(BI.symbols().count("orc_rt_X"));
  EXPECT_EQ(BI.symbols().at("orc_rt_X"), &X);
  EXPECT_EQ(BI.values().size(), 1U);
  EXPECT_EQ(BI.values().at("key"), "value");
}

TEST(BootstrapInfoTest, ProcessInfoDelegates) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  BootstrapInfo BI(S);
  EXPECT_EQ(&BI.processInfo(), &S.processInfo());
}

TEST(BootstrapInfoTest, CreateDefaultSucceeds) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  auto BI = cantFail(BootstrapInfo::CreateDefault(S));
  EXPECT_EQ(&BI.session(), &S);
}

TEST(BootstrapInfoTest, CreateDefaultContainsSessionSymbol) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  auto BI = cantFail(BootstrapInfo::CreateDefault(S));
  ASSERT_TRUE(BI.symbols().count("orc_rt_Session_Instance"));
  EXPECT_EQ(BI.symbols().at("orc_rt_Session_Instance"),
            static_cast<const void *>(&S));
}

TEST(BootstrapInfoTest, CreateDefaultContainsSPSCISymbols) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  auto BI = cantFail(BootstrapInfo::CreateDefault(S));
  // The default addAll should have registered SPS CI symbols.
  EXPECT_TRUE(BI.symbols().count(
      "orc_rt_sps_ci_SimpleNativeMemoryMap_reserve_sps_wrapper"));
}

TEST(BootstrapInfoTest, CreateDefaultWithNoSymbolsBuilder) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  auto BI = cantFail(BootstrapInfo::CreateDefault(S, /*AddInitialSymbols=*/{},
                                                  /*AddInitialValues=*/{}));
  // Should still contain the session symbol (added unconditionally).
  ASSERT_TRUE(BI.symbols().count("orc_rt_Session_Instance"));
  // But no SPS CI symbols.
  EXPECT_FALSE(BI.symbols().count(
      "orc_rt_sps_ci_SimpleNativeMemoryMap_reserve_sps_wrapper"));
}

TEST(BootstrapInfoTest, CreateDefaultWithCustomValuesBuilder) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  auto BI = cantFail(BootstrapInfo::CreateDefault(
      S, sps_ci::addAll, [](BootstrapInfo::ValueMap &Values) -> Error {
        Values["test_key"] = "test_value";
        return Error::success();
      }));
  EXPECT_EQ(BI.values().at("test_key"), "test_value");
}

TEST(BootstrapInfoTest, CreateDefaultSymbolsBuilderError) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  auto BI = BootstrapInfo::CreateDefault(S, [](SimpleSymbolTable &) -> Error {
    return make_error<StringError>("symbols builder failed");
  });
  EXPECT_FALSE(!!BI);
  auto ErrMsg = toString(BI.takeError());
  EXPECT_NE(ErrMsg.find("symbols builder failed"), std::string::npos);
}

TEST(BootstrapInfoTest, CreateDefaultValuesBuilderError) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  auto BI = BootstrapInfo::CreateDefault(
      S, sps_ci::addAll, [](BootstrapInfo::ValueMap &) -> Error {
        return make_error<StringError>("values builder failed");
      });
  EXPECT_FALSE(!!BI);
  auto ErrMsg = toString(BI.takeError());
  EXPECT_NE(ErrMsg.find("values builder failed"), std::string::npos);
}

TEST(BootstrapInfoTest, MutableSymbolsAndValues) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  BootstrapInfo BI(S);

  int X = 0;
  std::pair<const char *, void *> Syms[] = {{"orc_rt_X", &X}};
  cantFail(BI.symbols().addUnique(Syms));
  BI.values()["key"] = "value";

  EXPECT_EQ(BI.symbols().size(), 1U);
  EXPECT_EQ(BI.values().size(), 1U);
}
