//===- HandleLookupInJITDylibTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/HandleLookupInJITDylib.h"
#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/BindCallControllerHandler.h"
#include "llvm/ExecutionEngine/Orc/HandleLookupInJITDylibSPS.h"
#include "llvm/ExecutionEngine/Orc/Shared/Mangler.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/ExecutionEngine/Orc/SymbolLookupSet.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <future>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

class HandleLookupInJITDylibTest : public CoreAPIsBasedStandardTest {
protected:
  // The executor-side key for JD. Arbitrary: the handler only passes it to its
  // GetJITDylibFn.
  static constexpr ExecutorAddr JDKey{0x1000};
  static constexpr ExecutorAddr TagAddr{0x2000};

  void SetUp() override {
    auto TagName =
        Mangler(ES.getTargetTriple())
            .withMangledNameDo([&](StringRef N) { return ES.intern(N); },
                               sps::HandleLookupInJITDylibSpec::Name);
    cantFail(JD.define(
        absoluteSymbols({{TagName, {TagAddr, JITSymbolFlags::Exported}}})));
    cantFail(ES.registerCallControllerHandlers(
        JD, bindCallControllerHandler<sps::HandleLookupInJITDylibSpec>(
                createLookupInJITDylibHandler(ES, [this](ExecutorAddr Key) {
                  return Key == JDKey ? JITDylibSP(&JD) : JITDylibSP();
                }))));
  }

  // Call the handler as the executor would.
  Expected<SymbolLookupResult>
  lookup(ExecutorAddr Key,
         const std::vector<std::pair<std::string, bool>> &Symbols,
         JITDylibLookupFlags JDLookupFlags =
             JITDylibLookupFlags::MatchExportedSymbolsOnly) {
    std::promise<MSVCPExpected<SymbolLookupResult>> P;
    auto F = P.get_future();
    WrapperFunction<sps::HandleLookupInJITDylibSpec::SPSSig>::callAsync(
        [&](auto SendResult, const char *ArgData, size_t ArgSize) {
          ES.runCallControllerHandler(
              std::move(SendResult), TagAddr,
              WrapperFunctionBuffer::copyFrom(ArgData, ArgSize));
        },
        [&](Error Err, Expected<SymbolLookupResult> Result) {
          if (Err) {
            consumeError(Result.takeError());
            P.set_value(std::move(Err));
          } else
            P.set_value(std::move(Result));
        },
        Key, JDLookupFlags, Symbols);
    return F.get();
  }
};

} // namespace

TEST_F(HandleLookupInJITDylibTest, EmptyLookup) {
  auto Result = lookup(JDKey, {});
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  EXPECT_TRUE(Result->empty());
}

TEST_F(HandleLookupInJITDylibTest, ResultsInRequestOrder) {
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));

  auto Result = lookup(JDKey, {{"bar", true}, {"foo", true}});
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  ASSERT_EQ(Result->size(), 2U);
  EXPECT_EQ((*Result)[0], std::optional<ExecutorAddr>(BarAddr));
  EXPECT_EQ((*Result)[1], std::optional<ExecutorAddr>(FooAddr));
}

TEST_F(HandleLookupInJITDylibTest, MissingWeakSymbolIsNullopt) {
  // A missing weakly referenced symbol is reported as nullopt, and doesn't
  // affect the other results.
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  auto Result = lookup(JDKey, {{"foo", true}, {"baz", false}});
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  ASSERT_EQ(Result->size(), 2U);
  EXPECT_EQ((*Result)[0], std::optional<ExecutorAddr>(FooAddr));
  EXPECT_EQ((*Result)[1], std::nullopt);
}

TEST_F(HandleLookupInJITDylibTest, SymbolAtAddressZeroIsNull) {
  // A symbol defined at address zero is reported as a null address, distinct
  // from a missing weakly referenced symbol.
  auto Zero = ES.intern("zero");
  cantFail(JD.define(
      absoluteSymbols({{Zero, {ExecutorAddr(), JITSymbolFlags::Exported}}})));

  auto Result = lookup(JDKey, {{"zero", false}, {"baz", false}});
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  ASSERT_EQ(Result->size(), 2U);
  EXPECT_EQ((*Result)[0], std::optional<ExecutorAddr>(ExecutorAddr()));
  EXPECT_EQ((*Result)[1], std::nullopt);
}

TEST_F(HandleLookupInJITDylibTest, MissingRequiredSymbolFails) {
  // A missing required symbol fails the lookup before anything is
  // materialized, so the definition of foo must not be materialized.
  bool FooMaterialized = false;
  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        FooMaterialized = true;
        cantFail(R->notifyResolved({{Foo, FooSym}}));
        cantFail(R->notifyEmitted({}));
      })));

  EXPECT_THAT_EXPECTED(lookup(JDKey, {{"foo", true}, {"baz", true}}), Failed());
  EXPECT_FALSE(FooMaterialized);
}

TEST_F(HandleLookupInJITDylibTest, MatchExportedSymbolsOnlySkipsHidden) {
  // With MatchExportedSymbolsOnly, a hidden symbol is not found.
  cantFail(
      JD.define(absoluteSymbols({{Foo, {FooAddr, JITSymbolFlags::None}}})));

  auto Result = lookup(JDKey, {{"foo", false}},
                       JITDylibLookupFlags::MatchExportedSymbolsOnly);
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  ASSERT_EQ(Result->size(), 1U);
  EXPECT_EQ((*Result)[0], std::nullopt);
}

TEST_F(HandleLookupInJITDylibTest, MatchAllSymbolsFindsHidden) {
  // With MatchAllSymbols, a hidden symbol is found.
  cantFail(
      JD.define(absoluteSymbols({{Foo, {FooAddr, JITSymbolFlags::None}}})));

  auto Result =
      lookup(JDKey, {{"foo", true}}, JITDylibLookupFlags::MatchAllSymbols);
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  ASSERT_EQ(Result->size(), 1U);
  EXPECT_EQ((*Result)[0], std::optional<ExecutorAddr>(FooAddr));
}

TEST_F(HandleLookupInJITDylibTest, DuplicateNames) {
  // A name may appear more than once in a request. Each occurrence gets its
  // own entry in the result.
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));

  auto Result = lookup(JDKey, {{"foo", false}, {"bar", true}, {"foo", true}});
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  ASSERT_EQ(Result->size(), 3U);
  EXPECT_EQ((*Result)[0], std::optional<ExecutorAddr>(FooAddr));
  EXPECT_EQ((*Result)[1], std::optional<ExecutorAddr>(BarAddr));
  EXPECT_EQ((*Result)[2], std::optional<ExecutorAddr>(FooAddr));
}

TEST_F(HandleLookupInJITDylibTest, UnknownKeyFails) {
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  EXPECT_THAT_EXPECTED(
      lookup(ExecutorAddr(0x3000), {{"foo", true}}),
      FailedWithMessage(testing::HasSubstr("No JITDylib associated with key")));
}
