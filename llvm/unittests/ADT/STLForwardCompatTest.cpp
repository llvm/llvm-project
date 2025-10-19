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
#include <tuple>
#include <type_traits>
#include <utility>

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
  llvm::identity_cxx20 identity;

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

} // namespace
