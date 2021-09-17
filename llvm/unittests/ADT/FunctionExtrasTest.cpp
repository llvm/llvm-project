//===- FunctionExtrasTest.cpp - Unit tests for function type erasure ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/FunctionExtras.h"
#include "gtest/gtest.h"

#include <memory>
#include <type_traits>

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

TEST(UniqueFunctionTest, Const) {
  // Can assign from const lambda.
  unique_function<int(int) const> Plus2 = [X(std::make_unique<int>(2))](int Y) {
    return *X + Y;
  };
  EXPECT_EQ(5, Plus2(3));

  // Can call through a const ref.
  const auto &Plus2Ref = Plus2;
  EXPECT_EQ(5, Plus2Ref(3));

  // Can move-construct and assign.
  unique_function<int(int) const> Plus2A = std::move(Plus2);
  EXPECT_EQ(5, Plus2A(3));
  unique_function<int(int) const> Plus2B;
  Plus2B = std::move(Plus2A);
  EXPECT_EQ(5, Plus2B(3));

  // Can convert to non-const function type, but not back.
  unique_function<int(int)> Plus2C = std::move(Plus2B);
  EXPECT_EQ(5, Plus2C(3));

  // Overloaded call operator correctly resolved.
  struct ChooseCorrectOverload {
    StringRef operator()() { return "non-const"; }
    StringRef operator()() const { return "const"; }
  };
  unique_function<StringRef()> ChooseMutable = ChooseCorrectOverload();
  ChooseCorrectOverload A;
  EXPECT_EQ("non-const", ChooseMutable());
  EXPECT_EQ("non-const", A());
  unique_function<StringRef() const> ChooseConst = ChooseCorrectOverload();
  const ChooseCorrectOverload &X = A;
  EXPECT_EQ("const", ChooseConst());
  EXPECT_EQ("const", X());
}

// Test that overloads on unique_functions are resolved as expected.
std::string returns(StringRef) { return "not a function"; }
std::string returns(unique_function<double()> F) { return "number"; }
std::string returns(unique_function<StringRef()> F) { return "string"; }

TEST(UniqueFunctionTest, SFINAE) {
  EXPECT_EQ("not a function", returns("boo!"));
  EXPECT_EQ("number", returns([] { return 42; }));
  EXPECT_EQ("string", returns([] { return "hello"; }));
}

// A forward declared type, and a templated type.
class Incomplete;
template <typename T> class Templated { T A; };

// Check that we can define unique_function that have references to
// incomplete types, even if those types are templated over an
// incomplete type.
TEST(UniqueFunctionTest, IncompleteTypes) {
  unique_function<void(Templated<Incomplete> &&)>
      IncompleteArgumentRValueReference;
  unique_function<void(Templated<Incomplete> &)>
      IncompleteArgumentLValueReference;
  unique_function<void(Templated<Incomplete> *)> IncompleteArgumentPointer;
  unique_function<Templated<Incomplete> &()> IncompleteResultLValueReference;
  unique_function<Templated<Incomplete> && ()> IncompleteResultRValueReference2;
  unique_function<Templated<Incomplete> *()> IncompleteResultPointer;
}

// Incomplete function returning an incomplete type
Incomplete incompleteFunction();
const Incomplete incompleteFunctionConst();

// Check that we can assign a callable to a unique_function when the
// callable return value is incomplete.
TEST(UniqueFunctionTest, IncompleteCallableType) {
  unique_function<Incomplete()> IncompleteReturnInCallable{incompleteFunction};
  unique_function<const Incomplete()> IncompleteReturnInCallableConst{
      incompleteFunctionConst};
  unique_function<const Incomplete()> IncompleteReturnInCallableConstConversion{
      incompleteFunction};
}

// Define the incomplete function
class Incomplete {};
Incomplete incompleteFunction() { return {}; }
const Incomplete incompleteFunctionConst() { return {}; }

} // anonymous namespace
