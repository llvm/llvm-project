//===----- SymbolStringPoolTest.cpp - Unit tests for SymbolStringPool -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace llvm::orc {

class SymbolStringPoolTest : public testing::Test {
public:
  size_t getRefCount(const SymbolStringPtrBase &S) const {
    return SP.getRefCount(S);
  }

protected:
  SymbolStringPool SP;
};
} // namespace llvm::orc

namespace {

TEST_F(SymbolStringPoolTest, UniquingAndComparisons) {
  auto P1 = SP.intern("hello");

  std::string S("hel");
  S += "lo";
  auto P2 = SP.intern(S);

  auto P3 = SP.intern("goodbye");

  EXPECT_EQ(P1, P2) << "Failed to unique entries";
  EXPECT_NE(P1, P3) << "Unequal pooled symbol strings comparing equal";

  // We want to test that less-than comparison of SymbolStringPtrs compiles,
  // however we can't test the actual result as this is a pointer comparison and
  // SymbolStringPtr doesn't expose the underlying address of the string.
  (void)(P1 < P3);
}

TEST_F(SymbolStringPoolTest, Dereference) {
  auto Foo = SP.intern("foo");
  EXPECT_EQ(*Foo, "foo") << "Equality on dereferenced string failed";
}

TEST_F(SymbolStringPoolTest, ClearDeadEntries) {
  {
    auto P1 = SP.intern("s1");
    SP.clearDeadEntries();
    EXPECT_FALSE(SP.empty()) << "\"s1\" entry in pool should still be retained";
  }
  SP.clearDeadEntries();
  EXPECT_TRUE(SP.empty()) << "pool should be empty";
}

TEST_F(SymbolStringPoolTest, DebugDump) {
  auto A1 = SP.intern("a");
  auto A2 = A1;
  auto B = SP.intern("b");

  std::string DumpString;
  raw_string_ostream(DumpString) << SP;

  EXPECT_EQ(DumpString, "a: 2\nb: 1\n");
}

TEST_F(SymbolStringPoolTest, NonOwningPointerBasics) {
  auto A = SP.intern("a");
  auto B = SP.intern("b");

  NonOwningSymbolStringPtr ANP1(A);    // Constuct from SymbolStringPtr.
  NonOwningSymbolStringPtr ANP2(ANP1); // Copy-construct.
  NonOwningSymbolStringPtr BNP(B);

  // Equality comparisons.
  EXPECT_EQ(A, ANP1);
  EXPECT_EQ(ANP1, ANP2);
  EXPECT_NE(ANP1, BNP);

  EXPECT_EQ(*ANP1, "a"); // Dereference.

  // Assignment.
  ANP2 = ANP1;
  ANP2 = A;

  SymbolStringPtr S(ANP1); // Construct SymbolStringPtr from non-owning.
  EXPECT_EQ(S, A);

  DenseMap<SymbolStringPtr, int> M;
  M[A] = 42;
  EXPECT_EQ(M.find_as(ANP1)->second, 42);
  EXPECT_EQ(M.find_as(BNP), M.end());
}

TEST_F(SymbolStringPoolTest, NonOwningPointerRefCounts) {
  // Check that creating and destroying non-owning pointers doesn't affect
  // ref-counts.
  auto A = SP.intern("a");
  EXPECT_EQ(getRefCount(A), 1U);

  NonOwningSymbolStringPtr ANP(A);
  EXPECT_EQ(getRefCount(ANP), 1U)
      << "Construction of NonOwningSymbolStringPtr from SymbolStringPtr "
         "changed ref-count";

  {
    NonOwningSymbolStringPtr ANP2(ANP);
    EXPECT_EQ(getRefCount(ANP2), 1U)
        << "Copy-construction of NonOwningSymbolStringPtr changed ref-count";
  }

  EXPECT_EQ(getRefCount(ANP), 1U)
      << "Destruction of NonOwningSymbolStringPtr changed ref-count";

  {
    NonOwningSymbolStringPtr ANP2;
    ANP2 = ANP;
    EXPECT_EQ(getRefCount(ANP2), 1U)
        << "Copy-assignment of NonOwningSymbolStringPtr changed ref-count";
  }

  {
    NonOwningSymbolStringPtr ANP2(ANP);
    NonOwningSymbolStringPtr ANP3(std::move(ANP2));
    EXPECT_EQ(getRefCount(ANP3), 1U)
        << "Move-construction of NonOwningSymbolStringPtr changed ref-count";
  }

  {
    NonOwningSymbolStringPtr ANP2(ANP);
    NonOwningSymbolStringPtr ANP3;
    ANP3 = std::move(ANP2);
    EXPECT_EQ(getRefCount(ANP3), 1U)
        << "Copy-assignment of NonOwningSymbolStringPtr changed ref-count";
  }
}

TEST_F(SymbolStringPoolTest, SymbolStringPoolEntryUnsafe) {

  auto A = SP.intern("a");
  EXPECT_EQ(getRefCount(A), 1U);

  {
    // Try creating an unsafe pool entry ref from the given SymbolStringPtr.
    // This should not affect the ref-count.
    auto AUnsafe = SymbolStringPoolEntryUnsafe::from(A);
    EXPECT_EQ(getRefCount(A), 1U);

    // Create a new SymbolStringPtr from the unsafe ref. This should increment
    // the ref-count.
    auto ACopy = AUnsafe.copyToSymbolStringPtr();
    EXPECT_EQ(getRefCount(A), 2U);
  }

  {
    // Create a copy of the original string. Move it into an unsafe ref, and
    // then move it back. None of these operations should affect the ref-count.
    auto ACopy = A;
    EXPECT_EQ(getRefCount(A), 2U);
    auto AUnsafe = SymbolStringPoolEntryUnsafe::take(std::move(ACopy));
    EXPECT_EQ(getRefCount(A), 2U);
    ACopy = AUnsafe.moveToSymbolStringPtr();
    EXPECT_EQ(getRefCount(A), 2U);
  }

  // Test manual retain / release.
  auto AUnsafe = SymbolStringPoolEntryUnsafe::from(A);
  EXPECT_EQ(getRefCount(A), 1U);
  AUnsafe.retain();
  EXPECT_EQ(getRefCount(A), 2U);
  AUnsafe.release();
  EXPECT_EQ(getRefCount(A), 1U);
}

TEST_F(SymbolStringPoolTest, Hashing) {
  auto A = SP.intern("a");
  auto B = NonOwningSymbolStringPtr(A);

  hash_code AHash = hash_value(A);
  hash_code BHash = hash_value(B);

  EXPECT_EQ(AHash, BHash);
}

} // namespace
