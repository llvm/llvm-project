//===- FunctionExtrasTest.cpp - Unit tests for function type erasure ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/FunctionExtras.h"
#include "gtest/gtest.h"

#include <memory>

using namespace llvm;

namespace {

TEST(UniqueFunctionTest, Basic) {
  unique_function<int(int, int)> Sum = [](int A, int B) { return A + B; };
  EXPECT_EQ(Sum(1, 2), 3);

  unique_function<int(int, int)> Sum2 = std::move(Sum);
  EXPECT_EQ(Sum2(1, 2), 3);

  unique_function<int(int, int)> Sum3 = [](int A, int B) { return A + B; };
  Sum2 = std::move(Sum3);
  EXPECT_EQ(Sum2(1, 2), 3);

  Sum2 = unique_function<int(int, int)>([](int A, int B) { return A + B; });
  EXPECT_EQ(Sum2(1, 2), 3);

  // Explicit self-move test.
  *&Sum2 = std::move(Sum2);
  EXPECT_EQ(Sum2(1, 2), 3);

  Sum2 = unique_function<int(int, int)>();
  EXPECT_FALSE(Sum2);

  // Make sure we can forward through l-value reference parameters.
  unique_function<void(int &)> Inc = [](int &X) { ++X; };
  int X = 42;
  Inc(X);
  EXPECT_EQ(X, 43);

  // Make sure we can forward through r-value reference parameters with
  // move-only types.
  unique_function<int(std::unique_ptr<int> &&)> ReadAndDeallocByRef =
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
  unique_function<int(std::unique_ptr<int>)> ReadAndDeallocByVal =
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

TEST(UniqueFunctionTest, Captures) {
  long A = 1, B = 2, C = 3, D = 4, E = 5;

  unique_function<long()> Tmp;

  unique_function<long()> C1 = [A]() { return A; };
  EXPECT_EQ(C1(), 1);
  Tmp = std::move(C1);
  EXPECT_EQ(Tmp(), 1);

  unique_function<long()> C2 = [A, B]() { return A + B; };
  EXPECT_EQ(C2(), 3);
  Tmp = std::move(C2);
  EXPECT_EQ(Tmp(), 3);

  unique_function<long()> C3 = [A, B, C]() { return A + B + C; };
  EXPECT_EQ(C3(), 6);
  Tmp = std::move(C3);
  EXPECT_EQ(Tmp(), 6);

  unique_function<long()> C4 = [A, B, C, D]() { return A + B + C + D; };
  EXPECT_EQ(C4(), 10);
  Tmp = std::move(C4);
  EXPECT_EQ(Tmp(), 10);

  unique_function<long()> C5 = [A, B, C, D, E]() { return A + B + C + D + E; };
  EXPECT_EQ(C5(), 15);
  Tmp = std::move(C5);
  EXPECT_EQ(Tmp(), 15);
}

TEST(UniqueFunctionTest, MoveOnly) {
  struct SmallCallable {
    std::unique_ptr<int> A{new int(1)};

    int operator()(int B) { return *A + B; }
  };
  unique_function<int(int)> Small = SmallCallable();
  EXPECT_EQ(Small(2), 3);
  unique_function<int(int)> Small2 = std::move(Small);
  EXPECT_EQ(Small2(2), 3);

  struct LargeCallable {
    std::unique_ptr<int> A{new int(1)};
    std::unique_ptr<int> B{new int(2)};
    std::unique_ptr<int> C{new int(3)};
    std::unique_ptr<int> D{new int(4)};
    std::unique_ptr<int> E{new int(5)};

    int operator()() { return *A + *B + *C + *D + *E; }
  };
  unique_function<int()> Large = LargeCallable();
  EXPECT_EQ(Large(), 15);
  unique_function<int()> Large2 = std::move(Large);
  EXPECT_EQ(Large2(), 15);
}

TEST(UniqueFunctionTest, CountForwardingCopies) {
  struct CopyCounter {
    int &CopyCount;

    CopyCounter(int &CopyCount) : CopyCount(CopyCount) {}
    CopyCounter(const CopyCounter &Arg) : CopyCount(Arg.CopyCount) {
      ++CopyCount;
    }
  };

  unique_function<void(CopyCounter)> ByValF = [](CopyCounter) {};
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
  unique_function<void(const CopyCounter &)> ByRefF = [](const CopyCounter &) {
  };
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
  unique_function<void(const Uncopyable &)> UncopyableF =
      [](const Uncopyable &) {};
  UncopyableF(Uncopyable());
  Uncopyable X;
  UncopyableF(X);
}

TEST(UniqueFunctionTest, CountForwardingMoves) {
  struct MoveCounter {
    int &MoveCount;

    MoveCounter(int &MoveCount) : MoveCount(MoveCount) {}
    MoveCounter(MoveCounter &&Arg) : MoveCount(Arg.MoveCount) { ++MoveCount; }
  };

  unique_function<void(MoveCounter)> ByValF = [](MoveCounter) {};
  int MoveCount = 0;
  ByValF(MoveCounter(MoveCount));
  EXPECT_EQ(1, MoveCount);

  MoveCount = 0;
  {
    MoveCounter Counter{MoveCount};
    ByValF(std::move(Counter));
  }
  EXPECT_EQ(2, MoveCount);

  // Check that when we use an r-value reference we get no spurious copies.
  unique_function<void(MoveCounter &&)> ByRefF = [](MoveCounter &&) {};
  MoveCount = 0;
  ByRefF(MoveCounter(MoveCount));
  EXPECT_EQ(0, MoveCount);

  MoveCount = 0;
  {
    MoveCounter Counter{MoveCount};
    ByRefF(std::move(Counter));
  }
  EXPECT_EQ(0, MoveCount);

  // If we use an r-value reference we can in fact make a stronger guarantee
  // with an unmovable type.
  struct Unmovable {
    Unmovable() = default;
    Unmovable(Unmovable &&) = delete;
  };
  unique_function<void(const Unmovable &)> UnmovableF = [](const Unmovable &) {
  };
  UnmovableF(Unmovable());
  Unmovable X;
  UnmovableF(X);
}

} // anonymous namespace
