//===- llvm/unittest/Support/AnyTest.cpp - Any tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Any.h"
#include "gtest/gtest.h"
#include <cstdlib>

using namespace llvm;

namespace {

// Make sure we can construct, copy-construct, move-construct, and assign Any's.
TEST(AnyTest, ConstructionAndAssignment) {
  llvm::Any A;
  llvm::Any B{7};
  llvm::Any C{8};
  llvm::Any D{"hello"};
  llvm::Any E{3.7};

  // An empty Any is not anything.
  EXPECT_FALSE(A.has_value());
  EXPECT_FALSE(any_cast<int>(&A));

  // An int is an int but not something else.
  EXPECT_TRUE(B.has_value());
  EXPECT_TRUE(any_cast<int>(&B));
  EXPECT_FALSE(any_cast<float>(&B));

  EXPECT_TRUE(C.has_value());
  EXPECT_TRUE(any_cast<int>(&C));

  // A const char * is a const char * but not an int.
  EXPECT_TRUE(D.has_value());
  EXPECT_TRUE(any_cast<const char *>(&D));
  EXPECT_FALSE(any_cast<int>(&D));

  // A double is a double but not a float.
  EXPECT_TRUE(E.has_value());
  EXPECT_TRUE(any_cast<double>(&E));
  EXPECT_FALSE(any_cast<float>(&E));

  // After copy constructing from an int, the new item and old item are both
  // ints.
  llvm::Any F(B);
  EXPECT_TRUE(B.has_value());
  EXPECT_TRUE(F.has_value());
  EXPECT_TRUE(any_cast<int>(&F));
  EXPECT_TRUE(any_cast<int>(&B));

  // After move constructing from an int, the new item is an int and the old one
  // isn't.
  llvm::Any G(std::move(C));
  EXPECT_FALSE(C.has_value());
  EXPECT_TRUE(G.has_value());
  EXPECT_TRUE(any_cast<int>(&G));
  EXPECT_FALSE(any_cast<int>(&C));

  // After copy-assigning from an int, the new item and old item are both ints.
  A = F;
  EXPECT_TRUE(A.has_value());
  EXPECT_TRUE(F.has_value());
  EXPECT_TRUE(any_cast<int>(&A));
  EXPECT_TRUE(any_cast<int>(&F));

  // After move-assigning from an int, the new item and old item are both ints.
  B = std::move(G);
  EXPECT_TRUE(B.has_value());
  EXPECT_FALSE(G.has_value());
  EXPECT_TRUE(any_cast<int>(&B));
  EXPECT_FALSE(any_cast<int>(&G));
}

TEST(AnyTest, GoodAnyCast) {
  llvm::Any A;
  llvm::Any B{7};
  llvm::Any C{8};
  llvm::Any D{"hello"};
  llvm::Any E{'x'};

  // Check each value twice to make sure it isn't damaged by the cast.
  EXPECT_EQ(7, llvm::any_cast<int>(B));
  EXPECT_EQ(7, llvm::any_cast<int>(B));

  EXPECT_STREQ("hello", llvm::any_cast<const char *>(D));
  EXPECT_STREQ("hello", llvm::any_cast<const char *>(D));

  EXPECT_EQ('x', llvm::any_cast<char>(E));
  EXPECT_EQ('x', llvm::any_cast<char>(E));

  llvm::Any F(B);
  EXPECT_EQ(7, llvm::any_cast<int>(F));
  EXPECT_EQ(7, llvm::any_cast<int>(F));

  llvm::Any G(std::move(C));
  EXPECT_EQ(8, llvm::any_cast<int>(G));
  EXPECT_EQ(8, llvm::any_cast<int>(G));

  A = F;
  EXPECT_EQ(7, llvm::any_cast<int>(A));
  EXPECT_EQ(7, llvm::any_cast<int>(A));

  E = std::move(G);
  EXPECT_EQ(8, llvm::any_cast<int>(E));
  EXPECT_EQ(8, llvm::any_cast<int>(E));

  // Make sure we can any_cast from an rvalue and that it's properly destroyed
  // in the process.
  EXPECT_EQ(8, llvm::any_cast<int>(std::move(E)));
  EXPECT_TRUE(E.has_value());

  // Make sure moving from pointers gives back pointers, and that we can modify
  // the underlying value through those pointers.
  EXPECT_EQ(7, *llvm::any_cast<int>(&A));
  int *N = llvm::any_cast<int>(&A);
  *N = 42;
  EXPECT_EQ(42, llvm::any_cast<int>(A));

  // Make sure that we can any_cast to a reference and this is considered a good
  // cast, resulting in an lvalue which can be modified.
  llvm::any_cast<int &>(A) = 43;
  EXPECT_EQ(43, llvm::any_cast<int>(A));
}

TEST(AnyTest, CopiesAndMoves) {
  struct TestType {
    TestType() = default;
    TestType(const TestType &Other)
        : Copies(Other.Copies + 1), Moves(Other.Moves) {}
    TestType(TestType &&Other) : Copies(Other.Copies), Moves(Other.Moves + 1) {}
    int Copies = 0;
    int Moves = 0;
  };

  // One move to get TestType into the Any, and one move on the cast.
  TestType T1 = llvm::any_cast<TestType>(Any{TestType()});
  EXPECT_EQ(0, T1.Copies);
  EXPECT_EQ(2, T1.Moves);

  // One move to get TestType into the Any, and one copy on the cast.
  Any A{TestType()};
  TestType T2 = llvm::any_cast<TestType>(A);
  EXPECT_EQ(1, T2.Copies);
  EXPECT_EQ(1, T2.Moves);

  // One move to get TestType into the Any, and one move on the cast.
  TestType T3 = llvm::any_cast<TestType>(std::move(A));
  EXPECT_EQ(0, T3.Copies);
  EXPECT_EQ(2, T3.Moves);
}

TEST(AnyTest, BadAnyCast) {
  llvm::Any A;
  llvm::Any B{7};
  llvm::Any C{"hello"};
  llvm::Any D{'x'};

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(llvm::any_cast<int>(A), "");

  EXPECT_DEATH(llvm::any_cast<float>(B), "");
  EXPECT_DEATH(llvm::any_cast<int *>(B), "");

  EXPECT_DEATH(llvm::any_cast<std::string>(C), "");

  EXPECT_DEATH(llvm::any_cast<unsigned char>(D), "");
#endif
}

} // anonymous namespace
