//===- bind-test.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's bind.h APIs.
//
//===----------------------------------------------------------------------===//

#include "CommonTestUtils.h"
#include "orc-rt-c/config.h"
#include "orc-rt/support/bind.h"
#include "orc-rt/support/move_only_function.h"
#include "gtest/gtest.h"

#include <memory>
#include <type_traits>
#include <utility>

using namespace orc_rt;
using namespace orc_rt::test;

static int addInts(int X, int Y) { return X + Y; }
static int subtract(int X, int Y) { return X - Y; }
static int increment(int N) { return N + 1; }
static void doNothing() {}
static int noexceptAdd(int X, int Y) noexcept { return X + Y; }

namespace {

/// The cv qualification and value category of a call's object or argument.
enum class Category { LValue, ConstLValue, RValue, ConstRValue };

} // namespace

template <typename T> static constexpr Category categoryOf() {
  constexpr bool IsConst = std::is_const_v<std::remove_reference_t<T>>;
  if constexpr (std::is_lvalue_reference_v<T>)
    return IsConst ? Category::ConstLValue : Category::LValue;
  else
    return IsConst ? Category::ConstRValue : Category::RValue;
}

namespace {

/// Reports which of its call operators was selected.
struct CategoryOfCall {
  Category operator()() & { return Category::LValue; }
  Category operator()() const & { return Category::ConstLValue; }
  Category operator()() && { return Category::RValue; }
  Category operator()() const && { return Category::ConstRValue; }
};

/// Reports how its argument was passed.
struct CategoryOfArg {
  template <typename T> Category operator()(T &&) const {
    return categoryOf<T &&>();
  }
};

struct Counter {
  int Value = 0;
  int add(int N) { return Value += N; }
};

/// Copyable, but copying is not noexcept.
struct MayThrowOnCopy {
  MayThrowOnCopy() = default;
  MayThrowOnCopy(const MayThrowOnCopy &) {}
};

} // namespace

// Basic binding.

TEST(BindTest, BindsLeadingArguments) {
  auto SubtractFrom10 = bind_front(subtract, 10);
  EXPECT_EQ(SubtractFrom10(3), 7);
}

TEST(BindTest, NoBoundArguments) {
  auto Add = bind_front(addInts);
  EXPECT_EQ(Add(1, 2), 3);
}

TEST(BindTest, NoFreeArguments) {
  auto Add1And2 = bind_front(addInts, 1, 2);
  EXPECT_EQ(Add1And2(), 3);
}

TEST(BindTest, VoidReturn) {
  auto B = bind_front(doNothing);
  B();
  EXPECT_TRUE((std::is_void_v<decltype(B())>));
}

// Kinds of callable.

TEST(BindTest, BindsLambda) {
  auto Add1 = bind_front([](int X, int Y) { return X + Y; }, 1);
  EXPECT_EQ(Add1(2), 3);
}

TEST(BindTest, BindsMoveOnlyCallable) {
  move_only_function<int(int, int)> Add = [](int X, int Y) { return X + Y; };
  auto Add1 = bind_front(std::move(Add), 1);
  EXPECT_EQ(Add1(2), 3);
}

TEST(BindTest, BindsMemberFunctionPointer) {
  Counter C;
  auto AddToC = bind_front(&Counter::add, &C);
  AddToC(2);
  AddToC(3);
  EXPECT_EQ(C.Value, 5);
}

TEST(BindTest, BindsDataMemberPointer) {
  Counter C;
  C.Value = 7;
  auto ValueOf = bind_front(&Counter::Value);
  EXPECT_EQ(ValueOf(C), 7);
}

TEST(BindTest, BindsFunctionAsBoundArgument) {
  auto Apply =
      bind_front([](int Op(int), int Arg) { return Op(Arg); }, increment);
  EXPECT_EQ(Apply(1), 2);
}

// Storage of the callable and bound arguments.

TEST(BindTest, BoundRValueIsMovedIn) {
  OpCounter<>::reset();
  {
    auto B = bind_front([](OpCounter<> &, int) {}, OpCounter<>());
    B(0);
  }
  EXPECT_EQ(OpCounter<>::defaultConstructions(), 1U);
  EXPECT_EQ(OpCounter<>::copies(), 0U);
  EXPECT_EQ(OpCounter<>::moves(), 1U);
  EXPECT_EQ(OpCounter<>::destructions(), 2U);
}

TEST(BindTest, BoundLValueIsCopiedIn) {
  OpCounter<>::reset();
  {
    OpCounter<> O;
    auto B = bind_front([](OpCounter<> &, int) {}, O);
    B(0);
  }
  EXPECT_EQ(OpCounter<>::defaultConstructions(), 1U);
  EXPECT_EQ(OpCounter<>::copies(), 1U);
  EXPECT_EQ(OpCounter<>::moves(), 0U);
  EXPECT_EQ(OpCounter<>::destructions(), 2U);
}

TEST(BindTest, BoundArgumentsPersistAcrossCalls) {
  auto Count = bind_front([](int &N) { return ++N; }, 0);
  EXPECT_EQ(Count(), 1);
  EXPECT_EQ(Count(), 2);
}

TEST(BindTest, CopiesAreIndependent) {
  auto Count = bind_front([](int &N) { return ++N; }, 0);
  EXPECT_EQ(Count(), 1);

  auto Copy = Count;
  EXPECT_EQ(Copy(), 2);
  EXPECT_EQ(Count(), 2);
}

TEST(BindTest, WrapperCanBeBound) {
  auto Digits = [](int X, int Y, int Z) { return X * 100 + Y * 10 + Z; };
  auto Inner = bind_front(Digits, 1);

  auto Outer = bind_front(Inner, 2);
  EXPECT_EQ(Outer(3), 123);

  auto Rewrapped = bind_front(Inner);
  EXPECT_EQ(Rewrapped(2, 3), 123);
}

TEST(BindTest, MoveOnlyBoundArgumentMakesWrapperMoveOnly) {
  auto B = bind_front([](std::unique_ptr<int> &P) { return *P; },
                      std::make_unique<int>(42));
  using WrapperT = decltype(B);
  EXPECT_FALSE((std::is_copy_constructible_v<WrapperT>));
  EXPECT_FALSE((std::is_constructible_v<WrapperT, WrapperT &>));
  EXPECT_TRUE((std::is_move_constructible_v<WrapperT>));
  EXPECT_EQ(B(), 42);
}

// Forwarding into the call.

TEST(BindTest, FreeArgumentsAreForwarded) {
  auto B = bind_front(CategoryOfArg());
  int N = 0;
  const int CN = 0;
  EXPECT_EQ(B(N), Category::LValue);
  EXPECT_EQ(B(CN), Category::ConstLValue);
  EXPECT_EQ(B(std::move(N)), Category::RValue);
  EXPECT_EQ(B(std::move(CN)), Category::ConstRValue);
}

TEST(BindTest, MoveOnlyFreeArgument) {
  auto Deref = bind_front([](std::unique_ptr<int> P) { return *P; });
  EXPECT_EQ(Deref(std::make_unique<int>(42)), 42);
}

TEST(BindTest, ReferenceReturnIsPreserved) {
  auto Ref = bind_front([](int &N) -> int & { return N; }, 5);
  EXPECT_TRUE((std::is_same_v<decltype(Ref()), int &>));

  Ref() = 7;
  EXPECT_EQ(Ref(), 7);
}

TEST(BindTest, CallableSeesWrapperCategory) {
  auto B = bind_front(CategoryOfCall());
  EXPECT_EQ(B(), Category::LValue);
  EXPECT_EQ(std::as_const(B)(), Category::ConstLValue);
  EXPECT_EQ(std::move(B)(), Category::RValue);
  EXPECT_EQ(std::move(std::as_const(B))(), Category::ConstRValue);
}

TEST(BindTest, BoundArgumentsSeeWrapperCategory) {
  auto B = bind_front(CategoryOfArg(), 0);
  EXPECT_EQ(B(), Category::LValue);
  EXPECT_EQ(std::as_const(B)(), Category::ConstLValue);
  EXPECT_EQ(std::move(B)(), Category::RValue);
  EXPECT_EQ(std::move(std::as_const(B))(), Category::ConstRValue);
}

TEST(BindTest, RValueCallCanConsumeBoundArguments) {
  auto Take = bind_front([](std::unique_ptr<int> P) { return *P; },
                         std::make_unique<int>(42));
  EXPECT_FALSE((std::is_invocable_v<decltype(Take) &>));
  EXPECT_EQ(std::move(Take)(), 42);
}

TEST(BindTest, ConstCallRequiresConstCallable) {
  auto Count = bind_front([N = 0]() mutable { return ++N; });
  EXPECT_TRUE((std::is_invocable_v<decltype(Count) &>));
  EXPECT_FALSE((std::is_invocable_v<const decltype(Count) &>));
}

// A call the callable rejects for the wrapper's category must not fall back to
// another category's overload, even one that would accept it.

TEST(BindTest, NonConstCallDoesNotFallBackToConst) {
  struct ConstOnly {
    void operator()() & = delete;
    void operator()() const & {}
  };
  auto B = bind_front(ConstOnly());
  EXPECT_FALSE((std::is_invocable_v<decltype(B) &>));
  EXPECT_TRUE((std::is_invocable_v<const decltype(B) &>));
}

TEST(BindTest, RValueCallDoesNotFallBackToConst) {
  struct LValueOnly {
    void operator()() && = delete;
    void operator()() const & {}
  };
  auto B = bind_front(LValueOnly());
  EXPECT_FALSE((std::is_invocable_v<decltype(B) &&>));
  EXPECT_TRUE((std::is_invocable_v<decltype(B) &>));
}

// noexcept.

TEST(BindTest, NoexceptIsPropagated) {
  auto Nothrow = bind_front(noexceptAdd, 41);
  EXPECT_TRUE((std::is_nothrow_invocable_v<decltype(Nothrow), int>));

  auto MayThrow = bind_front(addInts, 41);
  EXPECT_FALSE((std::is_nothrow_invocable_v<decltype(MayThrow), int>));
}

TEST(BindTest, NoexceptFollowsTheSelectedOverload) {
  struct NoexceptWhenConst {
    void operator()() & {}
    void operator()() const & noexcept {}
  };
  auto B = bind_front(NoexceptWhenConst());
  EXPECT_FALSE((std::is_nothrow_invocable_v<decltype(B) &>));
  EXPECT_TRUE((std::is_nothrow_invocable_v<const decltype(B) &>));
}

TEST(BindTest, CopyAndMoveNoexceptFollowState) {
  using IntB = decltype(bind_front(addInts, 1));
  EXPECT_TRUE((std::is_nothrow_copy_constructible_v<IntB>));
  EXPECT_TRUE((std::is_nothrow_move_constructible_v<IntB>));

  auto Ignore = [](const MayThrowOnCopy &) {};
  using MayThrowB = decltype(bind_front(Ignore, MayThrowOnCopy()));
  EXPECT_FALSE((std::is_nothrow_copy_constructible_v<MayThrowB>));
  EXPECT_FALSE((std::is_nothrow_move_constructible_v<MayThrowB>));
}

#if ORC_RT_ENABLE_EXCEPTIONS

// A throwing copy of a bound argument must reach the caller rather than
// terminating the program.
TEST(BindTest, ExceptionFromBoundArgumentCopyPropagates) {
  struct ThrowsOnCopy {
    ThrowsOnCopy() = default;
    ThrowsOnCopy(const ThrowsOnCopy &) { throw 42; }
  };
  ThrowsOnCopy T;
  EXPECT_THROW((void)bind_front([](ThrowsOnCopy &) {}, T), int);
}

#endif // ORC_RT_ENABLE_EXCEPTIONS
