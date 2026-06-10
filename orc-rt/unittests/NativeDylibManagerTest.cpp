//===- NativeDylibManagerTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test NativeDylibManager APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/NativeDylibManager.h"
#include "orc-rt/Session.h"

#include "gtest/gtest.h"

#include "CommonTestUtils.h"

#include <optional>

using namespace orc_rt;

namespace {
// Local aliases for brevity in test bodies.
constexpr auto Req = NativeDylibManager::RequiredSymbol;
constexpr auto Weak = NativeDylibManager::WeaklyReferencedSymbol;
} // namespace

#ifndef NDM_TEST_LIB_PATH
#error                                                                         \
    "NDM_TEST_LIB_PATH must be defined to the path of the test shared library"
#endif

// Helper: synchronously run load and return result.
static Expected<void *> syncLoad(NativeDylibManager &NDM, std::string Path) {
  std::optional<Expected<void *>> Result;
  NDM.load([&](Expected<void *> R) { Result = std::move(R); }, std::move(Path));
  return std::move(*Result);
}

// Helper: synchronously run lookup and return results.
static Expected<std::vector<std::optional<void *>>>
syncLookup(NativeDylibManager &NDM, void *Handle,
           NativeDylibManager::SymbolLookupSet Symbols) {
  std::optional<Expected<std::vector<std::optional<void *>>>> Result;
  NDM.lookup(
      [&](Expected<std::vector<std::optional<void *>>> R) {
        Result = std::move(R);
      },
      Handle, std::move(Symbols));
  return std::move(*Result);
}

TEST(NativeDylibManagerTest, Create) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = NativeDylibManager::Create(S, ST);
  ASSERT_TRUE(!!NDM) << toString(NDM.takeError());
}

TEST(NativeDylibManagerTest, Load) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  auto LoadResult = syncLoad(*NDM, NDM_TEST_LIB_PATH);
  ASSERT_TRUE(!!LoadResult) << toString(LoadResult.takeError());
  EXPECT_NE(*LoadResult, nullptr);
}

TEST(NativeDylibManagerTest, LoadNonExistent) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  auto LoadResult = syncLoad(*NDM, "/no/such/library.dylib");
  EXPECT_FALSE(!!LoadResult);
  consumeError(LoadResult.takeError());
}

TEST(NativeDylibManagerTest, LoadEmptyPathReturnsGlobalHandle) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  // The global handle's value is implementation-defined, so verify by looking
  // up through it.
  auto LoadResult = syncLoad(*NDM, "");
  ASSERT_TRUE(!!LoadResult) << toString(LoadResult.takeError());
  void *Handle = *LoadResult;

  auto Result = syncLookup(*NDM, Handle, {{"malloc", Req}});
  ASSERT_TRUE(!!Result) << toString(Result.takeError());
  ASSERT_EQ(Result->size(), 1U);
  ASSERT_TRUE((*Result)[0].has_value())
      << "malloc should be findable via the process's global lookup handle";
  EXPECT_NE(*(*Result)[0], nullptr);
}

TEST(NativeDylibManagerTest, LookupSingleSymbol) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  void *Handle = cantFail(syncLoad(*NDM, NDM_TEST_LIB_PATH));

  auto Result = syncLookup(*NDM, Handle, {{"NativeDylibManagerTestFunc", Req}});
  ASSERT_TRUE(!!Result) << toString(Result.takeError());
  ASSERT_EQ(Result->size(), 1U);
  ASSERT_TRUE((*Result)[0].has_value());
  EXPECT_NE(*(*Result)[0], nullptr);

  // Verify the symbol points to the right function.
  auto *Func = reinterpret_cast<int (*)()>(*(*Result)[0]);
  EXPECT_EQ(Func(), 42);
}

TEST(NativeDylibManagerTest, LookupMultipleSymbols) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  void *Handle = cantFail(syncLoad(*NDM, NDM_TEST_LIB_PATH));

  auto Result = syncLookup(*NDM, Handle,
                           {{"NativeDylibManagerTestFunc", Req},
                            {"NativeDylibManagerTestFunc2", Req}});
  ASSERT_TRUE(!!Result) << toString(Result.takeError());
  ASSERT_EQ(Result->size(), 2U);
  ASSERT_TRUE((*Result)[0].has_value());
  ASSERT_TRUE((*Result)[1].has_value());
  EXPECT_NE(*(*Result)[0], nullptr);
  EXPECT_NE(*(*Result)[1], nullptr);

  auto *Func1 = reinterpret_cast<int (*)()>(*(*Result)[0]);
  auto *Func2 = reinterpret_cast<int (*)()>(*(*Result)[1]);
  EXPECT_EQ(Func1(), 42);
  EXPECT_EQ(Func2(), 7);
}

TEST(NativeDylibManagerTest, LookupWeakMissingSymbol) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  void *Handle = cantFail(syncLoad(*NDM, NDM_TEST_LIB_PATH));

  auto Result = syncLookup(*NDM, Handle, {{"no_such_symbol", Weak}});
  ASSERT_TRUE(!!Result) << toString(Result.takeError());
  ASSERT_EQ(Result->size(), 1U);
  ASSERT_TRUE((*Result)[0].has_value())
      << "weak-missing symbol should be reported as a present optional";
  EXPECT_EQ(*(*Result)[0], nullptr);
}

TEST(NativeDylibManagerTest, LookupRequiredMissingSymbol) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  void *Handle = cantFail(syncLoad(*NDM, NDM_TEST_LIB_PATH));

  auto Result = syncLookup(*NDM, Handle, {{"no_such_symbol", Req}});
  ASSERT_TRUE(!!Result) << toString(Result.takeError());
  ASSERT_EQ(Result->size(), 1U);
  EXPECT_FALSE((*Result)[0].has_value())
      << "required-missing symbol should be reported as an empty optional";
}

TEST(NativeDylibManagerTest, LookupMixedRequiredAndWeak) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  void *Handle = cantFail(syncLoad(*NDM, NDM_TEST_LIB_PATH));

  auto Result = syncLookup(
      *NDM, Handle,
      {{"NativeDylibManagerTestFunc", Req}, {"no_such_symbol", Weak}});
  ASSERT_TRUE(!!Result) << toString(Result.takeError());
  ASSERT_EQ(Result->size(), 2U);
  ASSERT_TRUE((*Result)[0].has_value());
  EXPECT_NE(*(*Result)[0], nullptr);
  ASSERT_TRUE((*Result)[1].has_value())
      << "weak-missing symbol should be reported as a present optional";
  EXPECT_EQ(*(*Result)[1], nullptr);
}
