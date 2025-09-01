//===- move_only_function-test.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt/move_only_function.h"
#include "gtest/gtest.h"

using namespace orc_rt;

TEST(MoveOnlyFunctionTest, Basic) {
  move_only_function<int(int, int)> Sum = [](int A, int B) { return A + B; };
  EXPECT_EQ(Sum(1, 2), 3);

  move_only_function<int(int, int)> Sum2 = std::move(Sum);
  EXPECT_EQ(Sum2(1, 2), 3);

  move_only_function<int(int, int)> Sum3 = [](int A, int B) { return A + B; };
  Sum2 = std::move(Sum3);
  EXPECT_EQ(Sum2(1, 2), 3);

  Sum2 = move_only_function<int(int, int)>([](int A, int B) { return A + B; });
  EXPECT_EQ(Sum2(1, 2), 3);

  // Explicit self-move test.
  *&Sum2 = std::move(Sum2);
  EXPECT_EQ(Sum2(1, 2), 3);

  Sum2 = move_only_function<int(int, int)>();
  EXPECT_FALSE(Sum2);

  // Make sure we can forward through l-value reference parameters.
  move_only_function<void(int &)> Inc = [](int &X) { ++X; };
  int X = 42;
  Inc(X);
  EXPECT_EQ(X, 43);

  // Make sure we can forward through r-value reference parameters with
  // move-only types.
  move_only_function<int(std::unique_ptr<int> &&)> ReadAndDeallocByRef =
      [](std::unique_ptr<int> &&Ptr) {
        int V = *Ptr;
        Ptr.reset();
        return V;
      };
  std::unique_ptr<int> Ptr{new int(13)};
  EXPECT_EQ(ReadAndDeallocByRef(std::move(Ptr)), 13);
  EXPECT_FALSE((bool)Ptr);

  // Make sure we can pass a move-only temporary as opposed to a local variable.
  EXPECT_EQ(ReadAndDeallocByRef(std::unique_ptr<int>(new int(42))), 42);

  // Make sure we can pass a move-only type by-value.
  move_only_function<int(std::unique_ptr<int>)> ReadAndDeallocByVal =
      [](std::unique_ptr<int> Ptr) {
        int V = *Ptr;
        Ptr.reset();
        return V;
      };
  Ptr.reset(new int(13));
  EXPECT_EQ(ReadAndDeallocByVal(std::move(Ptr)), 13);
  EXPECT_FALSE((bool)Ptr);

  EXPECT_EQ(ReadAndDeallocByVal(std::unique_ptr<int>(new int(42))), 42);
}

TEST(MoveOnlyFunctionTest, Captures) {
  long A = 1, B = 2, C = 3, D = 4, E = 5;

  move_only_function<long()> Tmp;

  move_only_function<long()> C1 = [A]() { return A; };
  EXPECT_EQ(C1(), 1);
  Tmp = std::move(C1);
  EXPECT_EQ(Tmp(), 1);

  move_only_function<long()> C2 = [A, B]() { return A + B; };
  EXPECT_EQ(C2(), 3);
  Tmp = std::move(C2);
  EXPECT_EQ(Tmp(), 3);

  move_only_function<long()> C3 = [A, B, C]() { return A + B + C; };
  EXPECT_EQ(C3(), 6);
  Tmp = std::move(C3);
  EXPECT_EQ(Tmp(), 6);

  move_only_function<long()> C4 = [A, B, C, D]() { return A + B + C + D; };
  EXPECT_EQ(C4(), 10);
  Tmp = std::move(C4);
  EXPECT_EQ(Tmp(), 10);

  move_only_function<long()> C5 = [A, B, C, D, E]() {
    return A + B + C + D + E;
  };
  EXPECT_EQ(C5(), 15);
  Tmp = std::move(C5);
  EXPECT_EQ(Tmp(), 15);

  // Test capture via lvalue.
  auto Inc = [](int N) { return N + 1; };
  move_only_function<int(int)> C6(Inc);
  EXPECT_EQ(C6(1), 2);
}

TEST(MoveOnlyFunctionTest, MoveOnly) {
  struct SmallCallable {
    std::unique_ptr<int> A = std::make_unique<int>(1);
    int operator()(int B) { return *A + B; }
  };

  move_only_function<int(int)> Small = SmallCallable();
  EXPECT_EQ(Small(2), 3);
  move_only_function<int(int)> Small2 = std::move(Small);
  EXPECT_EQ(Small2(2), 3);
}

TEST(MoveOnlyFunctionTest, CountForwardingCopies) {
  struct CopyCounter {
    int &CopyCount;

    CopyCounter(int &CopyCount) : CopyCount(CopyCount) {}
    CopyCounter(const CopyCounter &Arg) : CopyCount(Arg.CopyCount) {
      ++CopyCount;
    }
  };

  move_only_function<void(CopyCounter)> ByValF = [](CopyCounter) {};
  int CopyCount = 0;
  ByValF(CopyCounter(CopyCount));
  EXPECT_EQ(1, CopyCount);

  CopyCount = 0;
  {
    CopyCounter Counter{CopyCount};
    ByValF(Counter);
  }
  EXPECT_EQ(2, CopyCount);

  // Check that we don't generate a copy at all when we can bind a reference all
  // the way down, even if that reference could *in theory* allow copies.
  move_only_function<void(const CopyCounter &)> ByRefF =
      [](const CopyCounter &) {};
  CopyCount = 0;
  ByRefF(CopyCounter(CopyCount));
  EXPECT_EQ(0, CopyCount);

  CopyCount = 0;
  {
    CopyCounter Counter{CopyCount};
    ByRefF(Counter);
  }
  EXPECT_EQ(0, CopyCount);

  // If we use a reference, we can make a stronger guarantee that *no* copy
  // occurs.
  struct Uncopyable {
    Uncopyable() = default;
    Uncopyable(const Uncopyable &) = delete;
  };
  move_only_function<void(const Uncopyable &)> UncopyableF =
      [](const Uncopyable &) {};
  UncopyableF(Uncopyable());
  Uncopyable X;
  UncopyableF(X);
}

TEST(MoveOnlyFunctionTest, BooleanConversion) {
  move_only_function<void()> D;
  EXPECT_FALSE(D);

  move_only_function<void()> F = []() {};
  EXPECT_TRUE(F);
}
