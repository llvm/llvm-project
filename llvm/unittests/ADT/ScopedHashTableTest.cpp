//===- ScopedHashTableTest.cpp - ScopedHashTable unit tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <memory>
#include <stack>

using ::llvm::ScopedHashTable;
using ::llvm::ScopedHashTableScope;
using ::llvm::StringLiteral;
using ::llvm::StringRef;

using ::testing::Test;

class ScopedHashTableTest : public Test {
protected:
  ScopedHashTableTest() { symbolTable.insert(kGlobalName, kGlobalValue); }

  ScopedHashTable<StringRef, StringRef> symbolTable{};
  ScopedHashTableScope<StringRef, StringRef> globalScope{symbolTable};

  static constexpr StringLiteral kGlobalName = "global";
  static constexpr StringLiteral kGlobalValue = "gvalue";
  static constexpr StringLiteral kLocalName = "local";
  static constexpr StringLiteral kLocalValue = "lvalue";
  static constexpr StringLiteral kLocalValue2 = "lvalue2";
};

TEST_F(ScopedHashTableTest, AccessWithNoActiveScope) {
  EXPECT_EQ(symbolTable.count(kGlobalName), 1U);
}

TEST_F(ScopedHashTableTest, AccessWithAScope) {
  [[maybe_unused]] ScopedHashTableScope<StringRef, StringRef> varScope(
      symbolTable);
  EXPECT_EQ(symbolTable.count(kGlobalName), 1U);
}

TEST_F(ScopedHashTableTest, InsertInScope) {
  [[maybe_unused]] ScopedHashTableScope<StringRef, StringRef> varScope(
      symbolTable);
  symbolTable.insert(kLocalName, kLocalValue);
  EXPECT_EQ(symbolTable.count(kLocalName), 1U);
}

TEST_F(ScopedHashTableTest, InsertInLinearSortedScope) {
  [[maybe_unused]] ScopedHashTableScope<StringRef, StringRef> varScope(
      symbolTable);
  [[maybe_unused]] ScopedHashTableScope<StringRef, StringRef> varScope2(
      symbolTable);
  [[maybe_unused]] ScopedHashTableScope<StringRef, StringRef> varScope3(
      symbolTable);
  symbolTable.insert(kLocalName, kLocalValue);
  EXPECT_EQ(symbolTable.count(kLocalName), 1U);
}

TEST_F(ScopedHashTableTest, InsertInOutedScope) {
  {
    [[maybe_unused]] ScopedHashTableScope<StringRef, StringRef> varScope(
        symbolTable);
    symbolTable.insert(kLocalName, kLocalValue);
  }
  EXPECT_EQ(symbolTable.count(kLocalName), 0U);
}

TEST_F(ScopedHashTableTest, OverrideInScope) {
  [[maybe_unused]] ScopedHashTableScope<StringRef, StringRef> funScope(
      symbolTable);
  symbolTable.insert(kLocalName, kLocalValue);
  {
    [[maybe_unused]] ScopedHashTableScope<StringRef, StringRef> varScope(
        symbolTable);
    symbolTable.insert(kLocalName, kLocalValue2);
    EXPECT_EQ(symbolTable.lookup(kLocalName), kLocalValue2);
  }
  EXPECT_EQ(symbolTable.lookup(kLocalName), kLocalValue);
}

TEST_F(ScopedHashTableTest, GetCurScope) {
  EXPECT_EQ(symbolTable.getCurScope(), &globalScope);
  {
    ScopedHashTableScope<StringRef, StringRef> funScope(symbolTable);
    ScopedHashTableScope<StringRef, StringRef> funScope2(symbolTable);
    EXPECT_EQ(symbolTable.getCurScope(), &funScope2);
    {
      ScopedHashTableScope<StringRef, StringRef> blockScope(symbolTable);
      EXPECT_EQ(symbolTable.getCurScope(), &blockScope);
    }
    EXPECT_EQ(symbolTable.getCurScope(), &funScope2);
  }
  EXPECT_EQ(symbolTable.getCurScope(), &globalScope);
}

TEST_F(ScopedHashTableTest, PopScope) {
  using SymbolTableScopeTy = ScopedHashTable<StringRef, StringRef>::ScopeTy;

  std::stack<StringRef> ExpectedValues;
  std::stack<std::unique_ptr<SymbolTableScopeTy>> Scopes;

  Scopes.emplace(std::make_unique<SymbolTableScopeTy>(symbolTable));
  ExpectedValues.emplace(kLocalValue);
  symbolTable.insert(kGlobalName, kLocalValue);

  Scopes.emplace(std::make_unique<SymbolTableScopeTy>(symbolTable));
  ExpectedValues.emplace(kLocalValue2);
  symbolTable.insert(kGlobalName, kLocalValue2);

  while (symbolTable.getCurScope() != &globalScope) {
    EXPECT_EQ(symbolTable.getCurScope(), Scopes.top().get());
    EXPECT_EQ(symbolTable.lookup(kGlobalName), ExpectedValues.top());
    ExpectedValues.pop();
    Scopes.pop(); // destructs the SymbolTableScopeTy instance implicitly
                  // calling Scopes.top()->~SymbolTableScopeTy();
    EXPECT_NE(symbolTable.getCurScope(), nullptr);
  }
  ASSERT_TRUE(ExpectedValues.empty());
  ASSERT_TRUE(Scopes.empty());
  EXPECT_EQ(symbolTable.lookup(kGlobalName), kGlobalValue);
}

TEST_F(ScopedHashTableTest, DISABLED_PopScopeOnStack) {
  using SymbolTableScopeTy = ScopedHashTable<StringRef, StringRef>::ScopeTy;
  SymbolTableScopeTy funScope(symbolTable);
  symbolTable.insert(kGlobalName, kLocalValue);
  SymbolTableScopeTy funScope2(symbolTable);
  symbolTable.insert(kGlobalName, kLocalValue2);

  std::stack<StringRef> expectedValues{{kLocalValue, kLocalValue2}};
  std::stack<SymbolTableScopeTy *> expectedScopes{{&funScope, &funScope2}};

  while (symbolTable.getCurScope() != &globalScope) {
    EXPECT_EQ(symbolTable.getCurScope(), expectedScopes.top());
    expectedScopes.pop();
    EXPECT_EQ(symbolTable.lookup(kGlobalName), expectedValues.top());
    expectedValues.pop();
    symbolTable.getCurScope()->~SymbolTableScopeTy();
    EXPECT_NE(symbolTable.getCurScope(), nullptr);
  }

  // We have imbalanced scopes here:
  // Assertion `HT.CurScope == this && "Scope imbalance!"' failed
  // HT.CurScope is a pointer to the `globalScope` while
  // `SymbolTableScopeTy.this` is still a pointer to `funScope2`.
  // There is no way to write an assert on an assert in googletest so that we
  // mark the test case as DISABLED.
}
