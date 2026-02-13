//===- STLForwardCompatTest.cpp - Unit tests for STLForwardCompat ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLForwardCompat.h"
#include "CountCopyAndMove.h"
#include "gtest/gtest.h"

#include <optional>
#include <type_traits>
#include <utility>

namespace llvm {
namespace {

template <typename T>
class STLForwardCompatRemoveCVRefTest : public ::testing::Test {};

using STLForwardCompatRemoveCVRefTestTypes = ::testing::Types<
    // clang-format off
    std::pair<int, int>,
    std::pair<int &, int>,
    std::pair<const int, int>,
    std::pair<volatile int, int>,
    std::pair<const volatile int &, int>,
    std::pair<int *, int *>,
    std::pair<int *const, int *>,
    std::pair<const int *, const int *>,
    std::pair<int *&, int *>
    // clang-format on
    >;

TYPED_TEST_SUITE(STLForwardCompatRemoveCVRefTest,
                 STLForwardCompatRemoveCVRefTestTypes, );

TYPED_TEST(STLForwardCompatRemoveCVRefTest, RemoveCVRef) {
  using From = typename TypeParam::first_type;
  using To = typename TypeParam::second_type;
  EXPECT_TRUE(
      (std::is_same<typename llvm::remove_cvref<From>::type, To>::value));
}

TYPED_TEST(STLForwardCompatRemoveCVRefTest, RemoveCVRefT) {
  using From = typename TypeParam::first_type;
  EXPECT_TRUE((std::is_same<typename llvm::remove_cvref<From>::type,
                            llvm::remove_cvref_t<From>>::value));
}

template <typename T> class TypeIdentityTest : public ::testing::Test {
public:
  using TypeIdentity = llvm::type_identity<T>;
};

struct A {
  struct B {};
};
using TypeIdentityTestTypes =
    ::testing::Types<int, volatile int, A, const A::B>;

TYPED_TEST_SUITE(TypeIdentityTest, TypeIdentityTestTypes, /*NameGenerator*/);

TYPED_TEST(TypeIdentityTest, Identity) {
  // TestFixture is the instantiated TypeIdentityTest.
  EXPECT_TRUE(
      (std::is_same_v<TypeParam, typename TestFixture::TypeIdentity::type>));
}

TEST(TransformTest, TransformStd) {
  std::optional<int> A;

  std::optional<int> B = llvm::transformOptional(A, [&](int N) { return N + 1; });
  EXPECT_FALSE(B.has_value());

  A = 3;
  std::optional<int> C = llvm::transformOptional(A, [&](int N) { return N + 1; });
  EXPECT_TRUE(C.has_value());
  EXPECT_EQ(4, *C);
}

TEST(TransformTest, MoveTransformStd) {
  using llvm::CountCopyAndMove;

  std::optional<CountCopyAndMove> A;

  CountCopyAndMove::ResetCounts();
  std::optional<int> B = llvm::transformOptional(
      std::move(A), [&](const CountCopyAndMove &M) { return M.val + 2; });
  EXPECT_FALSE(B.has_value());
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(0, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(0, CountCopyAndMove::MoveAssignments);
  EXPECT_EQ(0, CountCopyAndMove::Destructions);

  A = CountCopyAndMove(5);
  CountCopyAndMove::ResetCounts();
  std::optional<int> C = llvm::transformOptional(
      std::move(A), [&](const CountCopyAndMove &M) { return M.val + 2; });
  EXPECT_TRUE(C.has_value());
  EXPECT_EQ(7, *C);
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(0, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(0, CountCopyAndMove::MoveAssignments);
  EXPECT_EQ(0, CountCopyAndMove::Destructions);
}

TEST(TransformTest, TransformLlvm) {
  std::optional<int> A;

  std::optional<int> B =
      llvm::transformOptional(A, [&](int N) { return N + 1; });
  EXPECT_FALSE(B.has_value());

  A = 3;
  std::optional<int> C =
      llvm::transformOptional(A, [&](int N) { return N + 1; });
  EXPECT_TRUE(C.has_value());
  EXPECT_EQ(4, *C);
}

TEST(TransformTest, MoveTransformLlvm) {
  using llvm::CountCopyAndMove;

  std::optional<CountCopyAndMove> A;

  CountCopyAndMove::ResetCounts();
  std::optional<int> B = llvm::transformOptional(
      std::move(A), [&](const CountCopyAndMove &M) { return M.val + 2; });
  EXPECT_FALSE(B.has_value());
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(0, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(0, CountCopyAndMove::MoveAssignments);
  EXPECT_EQ(0, CountCopyAndMove::Destructions);

  A = CountCopyAndMove(5);
  CountCopyAndMove::ResetCounts();
  std::optional<int> C = llvm::transformOptional(
      std::move(A), [&](const CountCopyAndMove &M) { return M.val + 2; });
  EXPECT_TRUE(C.has_value());
  EXPECT_EQ(7, *C);
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(0, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(0, CountCopyAndMove::MoveAssignments);
  EXPECT_EQ(0, CountCopyAndMove::Destructions);
}

TEST(TransformTest, TransformCategory) {
  struct StructA {
    int x;
  };
  struct StructB : StructA {
    StructB(StructA &&A) : StructA(std::move(A)) {}
  };

  std::optional<StructA> A{StructA{}};
  llvm::transformOptional(A, [](auto &&s) {
    EXPECT_FALSE(std::is_rvalue_reference_v<decltype(s)>);
    return StructB{std::move(s)};
  });

  llvm::transformOptional(std::move(A), [](auto &&s) {
    EXPECT_TRUE(std::is_rvalue_reference_v<decltype(s)>);
    return StructB{std::move(s)};
  });
}

TEST(TransformTest, ToUnderlying) {
  enum E { A1 = 0, B1 = -1 };
  static_assert(llvm::to_underlying(A1) == 0);
  static_assert(llvm::to_underlying(B1) == -1);

  enum E2 : unsigned char { A2 = 0, B2 };
  static_assert(
      std::is_same_v<unsigned char, decltype(llvm::to_underlying(A2))>);
  static_assert(llvm::to_underlying(A2) == 0);
  static_assert(llvm::to_underlying(B2) == 1);

  enum class E3 { A3 = -1, B3 };
  static_assert(std::is_same_v<int, decltype(llvm::to_underlying(E3::A3))>);
  static_assert(llvm::to_underlying(E3::A3) == -1);
  static_assert(llvm::to_underlying(E3::B3) == 0);
}

TEST(STLForwardCompatTest, IdentityCxx20) {
  llvm::identity identity;

  // Test with an lvalue.
  int X = 42;
  int &Y = identity(X);
  EXPECT_EQ(&X, &Y);

  // Test with a const lvalue.
  const int CX = 10;
  const int &CY = identity(CX);
  EXPECT_EQ(&CX, &CY);

  // Test with an rvalue.
  EXPECT_EQ(identity(123), 123);

  // Test perfect forwarding.
  static_assert(std::is_same_v<int &, decltype(identity(X))>);
  static_assert(std::is_same_v<const int &, decltype(identity(CX))>);
  static_assert(std::is_same_v<int &&, decltype(identity(int(5)))>);
}

//===----------------------------------------------------------------------===//
// llvm::invoke tests
//===----------------------------------------------------------------------===//

TEST(STLForwardCompatTest, InvokePerfectForwarding) {
  auto CheckArgs = [](auto &&A, auto &&B, auto &&C) {
    static_assert(std::is_same_v<decltype(A), int &>);
    static_assert(std::is_same_v<decltype(B), const int &>);
    static_assert(std::is_same_v<decltype(C), int &&>);
    return A + B + C;
  };

  int X = 1;
  const int Y = 2;
  EXPECT_EQ(llvm::invoke(CheckArgs, X, Y, 3), 6);
  // Make sure the result produced by std::invoke is identical --
  // the only meaningful difference is that std::invoke became
  // constexpr in C++20.
  EXPECT_EQ(std::invoke(CheckArgs, X, Y, 3), 6);
}

namespace {
struct InvokeTest {
  int Value;
  constexpr int scale(int Factor) const { return Value * Factor; }
  constexpr int &getRef() { return Value; }
};
} // namespace

TEST(STLForwardCompatTest, InvokeMemberPointers) {
  InvokeTest Obj{10};

  // Member function pointer.
  EXPECT_EQ(llvm::invoke(&InvokeTest::scale, Obj, 3), 30);
  EXPECT_EQ(llvm::invoke(&InvokeTest::scale, &Obj, 3), 30);

  // Member data pointer.
  EXPECT_EQ(llvm::invoke(&InvokeTest::Value, Obj), 10);
  EXPECT_EQ(llvm::invoke(&InvokeTest::Value, &Obj), 10);

  // Member function returning reference - args are forwarded, not copied.
  llvm::invoke(&InvokeTest::getRef, Obj) = 20;
  EXPECT_EQ(Obj.Value, 20); // Lvalue forwarded by reference.
  llvm::invoke(&InvokeTest::getRef, &Obj) = 30;
  EXPECT_EQ(Obj.Value, 30); // Pointer also works.
}

TEST(STLForwardCompatTest, InvokeConstexpr) {
  // Regular function.
  static constexpr int A =
      llvm::invoke([](int X, int Y) { return X + Y; }, 1, 2);
  static_assert(A == 3);

  // Member data pointer.
  static constexpr int B = llvm::invoke(&InvokeTest::Value, InvokeTest{42});
  static_assert(B == 42);

  // Member function pointer.
  static constexpr int C = llvm::invoke(&InvokeTest::scale, InvokeTest{5}, 3);
  static_assert(C == 15);
}

TEST(STLForwardCompatTest, BindFrontReferences) {
  // All bound arguments are forwarded (for ints, this is a copy) into the
  // wrapper. Call arguments are forwarded with their original value category.
  int A = 1;
  const int B = 2;
  int C = 3;
  int D = 4;
  const int E = 5;
  int F = 6;

  auto TestTypes = [](auto &&AArg, auto &&BArg, auto &&CArg, auto &&DArg,
                      auto &&EArg, auto &&FArg) {
    // Bound args: all stored as values, passed as lvalue refs.
    EXPECT_EQ(AArg, 1);
    static_assert(std::is_same_v<decltype(AArg), int &>);
    EXPECT_EQ(BArg, 2);
    static_assert(std::is_same_v<decltype(BArg), int &>); // Const decayed away.
    EXPECT_EQ(CArg, 3);
    static_assert(std::is_same_v<decltype(CArg), int &>);
    // Call args: forwarded with original value category.
    EXPECT_EQ(DArg, 4);
    static_assert(std::is_same_v<decltype(DArg), int &>);
    EXPECT_EQ(EArg, 5);
    static_assert(std::is_same_v<decltype(EArg), const int &>);
    EXPECT_EQ(FArg, 6);
    static_assert(std::is_same_v<decltype(FArg), int &&>);

    ++AArg;
    ++DArg;
  };

  llvm::bind_front(TestTypes, A, B, std::move(C))(D, E, std::move(F));
  EXPECT_EQ(A, 1); // A was copied, original unchanged.
  EXPECT_EQ(D, 5); // D was passed by reference and incremented.
}

TEST(STLForwardCompatTest, BindBackReferences) {
  // With std::decay_t, all bound arguments are copied into the wrapper.
  // Call arguments are forwarded with their original value category.
  int A = 1;
  const int B = 2;
  int C = 3;
  int D = 4;
  const int E = 5;
  int F = 6;

  auto TestTypes = [](auto &&AArg, auto &&BArg, auto &&CArg, auto &&DArg,
                      auto &&EArg, auto &&FArg) {
    // Call args: forwarded with original value category.
    EXPECT_EQ(AArg, 1);
    static_assert(std::is_same_v<decltype(AArg), int &>);
    EXPECT_EQ(BArg, 2);
    static_assert(std::is_same_v<decltype(BArg), const int &>);
    EXPECT_EQ(CArg, 3);
    static_assert(std::is_same_v<decltype(CArg), int &&>);
    // Bound args: all stored as values, passed as lvalue refs.
    EXPECT_EQ(DArg, 4);
    static_assert(std::is_same_v<decltype(DArg), int &>);
    EXPECT_EQ(EArg, 5);
    static_assert(std::is_same_v<decltype(EArg), int &>); // Const decayed away.
    EXPECT_EQ(FArg, 6);
    static_assert(std::is_same_v<decltype(FArg), int &>);

    ++AArg;
    ++DArg;
  };

  llvm::bind_back(TestTypes, D, E, std::move(F))(A, B, std::move(C));
  EXPECT_EQ(A, 2); // A was passed by reference and incremented.
  EXPECT_EQ(D, 4); // D was copied, original unchanged.
}

// Check that bound args are copied once during bind, then passed by reference.
TEST(STLForwardCompatTest, BindBoundArgsForwarding) {
  auto Fn = [](CountCopyAndMove &A) -> int { return A.val; };

  CountCopyAndMove::ResetCounts();
  CountCopyAndMove Arg(42);
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 0);
  EXPECT_EQ(CountCopyAndMove::TotalMoves(), 0);

  // Creating the wrapper should copy the bound args once.
  auto Bound = llvm::bind_front(Fn, Arg);
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 1);
  EXPECT_EQ(CountCopyAndMove::TotalMoves(), 0);

  // Calling should not copy -- bound args are passed by reference.
  EXPECT_EQ(Bound(), 42);
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 1);
  EXPECT_EQ(CountCopyAndMove::TotalMoves(), 0);
}

// Check that call args are forwarded without copies.
TEST(STLForwardCompatTest, BindCallArgsForwarding) {
  auto Fn = [](int, CountCopyAndMove &&A) -> int { return A.val; };

  CountCopyAndMove::ResetCounts();
  auto Bound = llvm::bind_front(Fn, 1);
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 0);
  EXPECT_EQ(CountCopyAndMove::TotalMoves(), 0);

  // Call arg should be forwarded as rvalue, no copies.
  EXPECT_EQ(Bound(CountCopyAndMove(42)), 42);
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 0);
  EXPECT_EQ(CountCopyAndMove::TotalMoves(), 0);
}

// Check that the callable itself is moved, not copied excessively.
TEST(STLForwardCompatTest, BindCallableForwarding) {
  CountCopyAndMove::ResetCounts();
  CountCopyAndMove Capture(42);
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 0);

  // Lambda captures by value -- this copy is outside bind's control.
  auto Fn = [Capture]() { return Capture.val; };
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 1);
  EXPECT_EQ(CountCopyAndMove::TotalMoves(), 0);

  // Moving lambda into bind should move, not copy.
  auto Bound = llvm::bind_front(std::move(Fn));
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 1);
  EXPECT_EQ(CountCopyAndMove::TotalMoves(), 1);

  // Calling should not copy the callable.
  EXPECT_EQ(Bound(), 42);
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 1);
  EXPECT_EQ(CountCopyAndMove::TotalMoves(), 1);

  // The bind object should be copyable.
  auto BoundCopy = Bound;
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 2);

  // The bind object should be movable.
  auto BoundMove = std::move(BoundCopy);
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 2);
  EXPECT_EQ(CountCopyAndMove::TotalMoves(), 2);
}

// Check that moving bound args works correctly.
TEST(STLForwardCompatTest, BindMoveBoundArgs) {
  auto Fn = [](CountCopyAndMove &A) -> int { return A.val; };

  CountCopyAndMove::ResetCounts();
  CountCopyAndMove Arg(42);

  // Moving into bind should move, not copy.
  auto Bound = llvm::bind_front(Fn, std::move(Arg));
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 0);
  EXPECT_EQ(CountCopyAndMove::TotalMoves(), 1);

  EXPECT_EQ(Bound(), 42);
  EXPECT_EQ(CountCopyAndMove::TotalCopies(), 0);
  EXPECT_EQ(CountCopyAndMove::TotalMoves(), 1);
}

TEST(STLForwardCompatTest, BindFrontMutableStorage) {
  // With std::decay_t, A is copied into the wrapper. The stored copy can be
  // mutated across calls, but the original A is unchanged.
  int A = 1;

  auto TestMutation = [](int &AArg, int &BArg, auto ExtraCheckFn) {
    ++AArg;
    ++BArg;
    ExtraCheckFn(AArg, BArg);
  };

  auto BoundA = llvm::bind_front(TestMutation, A, 42);
  BoundA([](int AVal, int BVal) {
    EXPECT_EQ(AVal, 2); // Stored copy incremented from 1.
    EXPECT_EQ(BVal, 43);
  });
  EXPECT_EQ(A, 1); // Original unchanged.

  BoundA([](int AVal, int BVal) {
    EXPECT_EQ(AVal, 3); // Stored copy incremented again.
    EXPECT_EQ(BVal, 44);
  });
  EXPECT_EQ(A, 1); // Original still unchanged.
}

TEST(STLForwardCompatTest, BindBackMutableStorage) {
  // With std::decay_t, A is copied into the wrapper. The stored copy can be
  // mutated across calls, but the original A is unchanged.
  int A = 1;

  auto TestMutation = [](auto ExtraCheckFn, int &AArg, int &BArg) {
    ++AArg;
    ++BArg;
    ExtraCheckFn(AArg, BArg);
  };

  auto BoundA = llvm::bind_back(TestMutation, A, 42);
  BoundA([](int AVal, int BVal) {
    EXPECT_EQ(AVal, 2); // Stored copy incremented from 1.
    EXPECT_EQ(BVal, 43);
  });
  EXPECT_EQ(A, 1); // Original unchanged.

  BoundA([](int AVal, int BVal) {
    EXPECT_EQ(AVal, 3); // Stored copy incremented again.
    EXPECT_EQ(BVal, 44);
  });
  EXPECT_EQ(A, 1); // Original still unchanged.
}

// Free function for compile-time bind tests.
static int subtract(int A, int B) { return A - B; }

TEST(STLForwardCompatTest, BindFrontConstexprCallable) {
  // Test compile-time callable with llvm::bind_front.
  auto TimesFive = llvm::bind_front<subtract>(5);
  EXPECT_EQ(TimesFive(3), 2);

  // Test compile-time callable with llvm::bind_back.
  auto FiveTimesX = llvm::bind_back<subtract>(5);
  EXPECT_EQ(FiveTimesX(3), -2);
}

TEST(STLForwardCompatTest, BindFrontBackNoBoundArgs) {
  auto Fn1 = llvm::bind_front([](int A, int B) { return A + B; });
  EXPECT_EQ(Fn1(3, 4), 7);
  auto Fn2 = llvm::bind_back([](int A, int B) { return A + B; });
  EXPECT_EQ(Fn2(3, 4), 7);
}

TEST(STLForwardCompatTest, BindFrontBindBackConstexpr) {
  static constexpr auto Fn1 =
      llvm::bind_front([](int A, int B) { return A + B; }, 1);
  static_assert(Fn1(3) == 4);
  static constexpr auto Fn2 =
      llvm::bind_back([](int A, int B) { return A + B; }, 1);
  static_assert(Fn2(3) == 4);
}

// Test that reference return types are preserved (with `decltype(auto)`).
TEST(STLForwardCompatTest, BindPreservesReferenceReturn) {
  int X = 10;
  auto GetRef = [&X]() -> int & { return X; };

  auto BoundFront = llvm::bind_front(GetRef);
  static_assert(std::is_same_v<decltype(BoundFront()), int &>);
  BoundFront() = 20;
  EXPECT_EQ(X, 20);

  const auto BoundFrontConst = llvm::bind_front(GetRef);
  static_assert(std::is_same_v<decltype(BoundFrontConst()), int &>);

  auto BoundBack = llvm::bind_back(GetRef);
  static_assert(std::is_same_v<decltype(BoundBack()), int &>);
  BoundBack() = 30;
  EXPECT_EQ(X, 30);

  const auto BoundBackConst = llvm::bind_back(GetRef);
  static_assert(std::is_same_v<decltype(BoundBackConst()), int &>);
}

// Use std::ref/std::cref to bind references (bound args are decay-copied).
TEST(STLForwardCompatTest, BindWithReferenceWrapper) {
  int X = 1;
  auto Increment = llvm::bind_front([](int &Val) { ++Val; }, std::ref(X));
  Increment();
  EXPECT_EQ(X, 2);
  Increment();
  EXPECT_EQ(X, 3);
}

// The callable itself can have mutable state.
TEST(STLForwardCompatTest, BindMutableCallable) {
  auto Counter = llvm::bind_front([N = 0]() mutable { return ++N; });
  EXPECT_EQ(Counter(), 1);
  EXPECT_EQ(Counter(), 2);
  EXPECT_EQ(Counter(), 3);
}

namespace {
struct MemberTest {
  int Value;
  int scale(int Factor) const { return Value * Factor; }
};
} // namespace

TEST(STLForwardCompatTest, BindMembers) {
  // Member function pointer support via std::apply (with std::invoke used
  // internally).
  MemberTest Obj{10};
  auto ScaleObj = llvm::bind_front(&MemberTest::scale, Obj);
  EXPECT_EQ(ScaleObj(3), 30);
  auto ScaleBy5 = llvm::bind_back(&MemberTest::scale, 5);
  EXPECT_EQ(ScaleBy5(Obj), 50);

  // Member data pointer support via std::apply (with std::invoke used
  // internally).
  auto GetValue = llvm::bind_front(&MemberTest::Value);
  EXPECT_EQ(GetValue(Obj), 10);

  // Make sure we can use member data pointers for constexpr callables.
  static constexpr int MemberVal =
      llvm::bind_front(&MemberTest::Value)(MemberTest{10});
  EXPECT_EQ(MemberVal, 10);
}

TEST(STLForwardCompatTest, BindFrontBindBack) {
  std::vector<int> V;
  auto MulAdd = [](int A, int B, int C) { return A * (B + C) == 12; };
  auto MulAdd1 = [](const int &A, const int &B, const int &C) {
    return A * (B + C) == 12;
  };
  auto Mul0 = llvm::bind_back(MulAdd, 4, 2);
  auto MulL = llvm::bind_front(MulAdd1, 2, 4);
  auto Mul20 = llvm::bind_back(MulAdd, 4);
  auto Mul21 = llvm::bind_front(MulAdd1, 2);
  EXPECT_TRUE(all_of(V, Mul0));
  EXPECT_TRUE(all_of(V, MulL));

  V.push_back(2);
  EXPECT_TRUE(all_of(V, Mul0));
  EXPECT_TRUE(all_of(V, MulL));

  V.push_back(2);
  V.push_back(2);
  EXPECT_TRUE(all_of(V, Mul0));
  EXPECT_TRUE(all_of(V, MulL));

  auto Spec0 = llvm::bind_front(Mul20, 2);
  auto Spec1 = llvm::bind_back(Mul21, 4);
  EXPECT_TRUE(all_of(V, Spec0));
  EXPECT_TRUE(all_of(V, Spec1));

  V.push_back(3);
  EXPECT_FALSE(all_of(V, Mul0));
  EXPECT_FALSE(all_of(V, MulL));
  EXPECT_FALSE(all_of(V, Spec0));
  EXPECT_FALSE(all_of(V, Spec1));
  EXPECT_TRUE(any_of(V, Spec0));
  EXPECT_TRUE(any_of(V, Spec1));
}

} // namespace
} // namespace llvm
