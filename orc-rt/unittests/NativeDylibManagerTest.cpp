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

// Helper: synchronously run unload and return result.
static Error syncUnload(NativeDylibManager &NDM, void *Handle) {
  std::optional<Error> Result;
  NDM.unload([&](Error R) { Result = std::move(R); }, Handle);
  return std::move(*Result);
}

// Helper: synchronously run lookup and return results.
static Expected<std::vector<void *>>
syncLookup(NativeDylibManager &NDM, void *Handle,
           std::vector<std::string> Names) {
  std::optional<Expected<std::vector<void *>>> Result;
  NDM.lookup([&](Expected<std::vector<void *>> R) { Result = std::move(R); },
             Handle, std::move(Names));
  return std::move(*Result);
}

TEST(NativeDylibManagerTest, Create) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = NativeDylibManager::Create(S, ST);
  ASSERT_TRUE(!!NDM) << toString(NDM.takeError());
}

TEST(NativeDylibManagerTest, LoadAndUnload) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  // Load the test library.
  auto LoadResult = syncLoad(*NDM, NDM_TEST_LIB_PATH);
  ASSERT_TRUE(!!LoadResult) << toString(LoadResult.takeError());
  void *Handle = *LoadResult;
  EXPECT_NE(Handle, nullptr);

  // Unload it.
  auto UnloadResult = syncUnload(*NDM, Handle);
  EXPECT_FALSE(!!UnloadResult) << toString(std::move(UnloadResult));
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

TEST(NativeDylibManagerTest, UnloadUnrecognizedHandle) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  void *Bogus = reinterpret_cast<void *>(0xDEADBEEF);
  auto UnloadResult = syncUnload(*NDM, Bogus);
  EXPECT_TRUE(!!UnloadResult);
  consumeError(std::move(UnloadResult));
}

TEST(NativeDylibManagerTest, LoadSameLibraryTwice) {
  // Loading the same library twice should succeed both times and return
  // the same handle (since dlopen refcounts).
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  auto R1 = syncLoad(*NDM, NDM_TEST_LIB_PATH);
  ASSERT_TRUE(!!R1) << toString(R1.takeError());
  void *H1 = *R1;

  auto R2 = syncLoad(*NDM, NDM_TEST_LIB_PATH);
  ASSERT_TRUE(!!R2) << toString(R2.takeError());
  void *H2 = *R2;

  EXPECT_EQ(H1, H2);

  // Unload both references.
  auto UR1 = syncUnload(*NDM, H1);
  EXPECT_FALSE(!!UR1) << toString(std::move(UR1));
  auto UR2 = syncUnload(*NDM, H2);
  EXPECT_FALSE(!!UR2) << toString(std::move(UR2));
}

TEST(NativeDylibManagerTest, LookupSingleSymbol) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  void *Handle = cantFail(syncLoad(*NDM, NDM_TEST_LIB_PATH));

  auto Result = syncLookup(*NDM, Handle, {"NativeDylibManagerTestFunc"});
  ASSERT_TRUE(!!Result) << toString(Result.takeError());
  ASSERT_EQ(Result->size(), 1U);
  EXPECT_NE((*Result)[0], nullptr);

  // Verify the symbol points to the right function.
  auto *Func = reinterpret_cast<int (*)()>((*Result)[0]);
  EXPECT_EQ(Func(), 42);

  auto UR = syncUnload(*NDM, Handle);
  EXPECT_FALSE(!!UR) << toString(std::move(UR));
}

TEST(NativeDylibManagerTest, LookupMultipleSymbols) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  void *Handle = cantFail(syncLoad(*NDM, NDM_TEST_LIB_PATH));

  auto Result =
      syncLookup(*NDM, Handle,
                 {"NativeDylibManagerTestFunc", "NativeDylibManagerTestFunc2"});
  ASSERT_TRUE(!!Result) << toString(Result.takeError());
  ASSERT_EQ(Result->size(), 2U);
  EXPECT_NE((*Result)[0], nullptr);
  EXPECT_NE((*Result)[1], nullptr);

  auto *Func1 = reinterpret_cast<int (*)()>((*Result)[0]);
  auto *Func2 = reinterpret_cast<int (*)()>((*Result)[1]);
  EXPECT_EQ(Func1(), 42);
  EXPECT_EQ(Func2(), 7);

  auto UR = syncUnload(*NDM, Handle);
  EXPECT_FALSE(!!UR) << toString(std::move(UR));
}

TEST(NativeDylibManagerTest, LookupNonExistentSymbol) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  void *Handle = cantFail(syncLoad(*NDM, NDM_TEST_LIB_PATH));

  auto Result = syncLookup(*NDM, Handle, {"no_such_symbol"});
  ASSERT_TRUE(!!Result) << toString(Result.takeError());
  ASSERT_EQ(Result->size(), 1U);
  EXPECT_EQ((*Result)[0], nullptr);

  auto UR = syncUnload(*NDM, Handle);
  EXPECT_FALSE(!!UR) << toString(std::move(UR));
}

TEST(NativeDylibManagerTest, LookupOnUnrecognizedHandle) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ST;
  auto NDM = cantFail(NativeDylibManager::Create(S, ST));

  void *Bogus = reinterpret_cast<void *>(0xDEADBEEF);
  auto Result = syncLookup(*NDM, Bogus, {"NativeDylibManagerTestFunc"});
  EXPECT_FALSE(!!Result);
  consumeError(Result.takeError());
}
