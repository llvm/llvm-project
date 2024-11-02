//===------ MappedIteratorTest.cpp - Unit tests for mapped_iterator -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <typename T> class MappedIteratorTestBasic : public testing::Test {};

struct Plus1Lambda {
  auto operator()() const {
    return [](int X) { return X + 1; };
  }
};

struct Plus1LambdaWithCapture {
  const int One = 1;

  auto operator()() const {
    return [=](int X) { return X + One; };
  }
};

struct Plus1FunctionRef {
  static int plus1(int X) { return X + 1; }

  using FuncT = int (&)(int);

  FuncT operator()() const { return (FuncT)*plus1; }
};

struct Plus1FunctionPtr {
  static int plus1(int X) { return X + 1; }

  using FuncT = int (*)(int);

  FuncT operator()() const { return plus1; }
};

struct Plus1Functor {
  struct Plus1 {
    int operator()(int X) const { return X + 1; }
  };

  auto operator()() const { return Plus1(); }
};

struct Plus1FunctorNotDefaultConstructible {
  class PlusN {
    const int N;

  public:
    PlusN(int NArg) : N(NArg) {}

    int operator()(int X) const { return X + N; }
  };

  auto operator()() const { return PlusN(1); }
};

// clang-format off
using FunctionTypes =
  ::testing::Types<
    Plus1Lambda,
    Plus1LambdaWithCapture,
    Plus1FunctionRef,
    Plus1FunctionPtr,
    Plus1Functor,
    Plus1FunctorNotDefaultConstructible
  >;
// clang-format on

TYPED_TEST_SUITE(MappedIteratorTestBasic, FunctionTypes, );

template <typename T> using GetFuncT = decltype(std::declval<T>().operator()());

TYPED_TEST(MappedIteratorTestBasic, DefaultConstruct) {
  using FuncT = GetFuncT<TypeParam>;
  using IterT = mapped_iterator<typename std::vector<int>::iterator, FuncT>;
  TypeParam GetCallable;

  auto Func = GetCallable();
  (void)Func;
  constexpr bool DefaultConstruct =
      std::is_default_constructible_v<callable_detail::Callable<FuncT>>;
  EXPECT_TRUE(DefaultConstruct);
  EXPECT_TRUE(std::is_default_constructible_v<IterT>);

  if constexpr (std::is_default_constructible_v<IterT>) {
    IterT I;
    (void)I;
  }
}

TYPED_TEST(MappedIteratorTestBasic, CopyConstruct) {
  std::vector<int> V({0});

  using FuncT = GetFuncT<TypeParam>;
  using IterT = mapped_iterator<decltype(V)::iterator, FuncT>;

  EXPECT_TRUE(std::is_copy_constructible_v<IterT>);

  if constexpr (std::is_copy_constructible_v<IterT>) {
    TypeParam GetCallable;

    IterT I1(V.begin(), GetCallable());
    IterT I2(I1);

    EXPECT_EQ(I2, I1) << "copy constructed iterator is a different position";
  }
}

TYPED_TEST(MappedIteratorTestBasic, MoveConstruct) {
  std::vector<int> V({0});

  using FuncT = GetFuncT<TypeParam>;
  using IterT = mapped_iterator<decltype(V)::iterator, FuncT>;

  EXPECT_TRUE(std::is_move_constructible_v<IterT>);

  if constexpr (std::is_move_constructible_v<IterT>) {
    TypeParam GetCallable;

    IterT I1(V.begin(), GetCallable());
    IterT I2(V.begin(), GetCallable());
    IterT I3(std::move(I2));

    EXPECT_EQ(I3, I1) << "move constructed iterator is a different position";
  }
}

TYPED_TEST(MappedIteratorTestBasic, CopyAssign) {
  std::vector<int> V({0});

  using FuncT = GetFuncT<TypeParam>;
  using IterT = mapped_iterator<decltype(V)::iterator, FuncT>;

  EXPECT_TRUE(std::is_copy_assignable_v<IterT>);

  if constexpr (std::is_copy_assignable_v<IterT>) {
    TypeParam GetCallable;

    IterT I1(V.begin(), GetCallable());
    IterT I2(V.end(), GetCallable());

    I2 = I1;

    EXPECT_EQ(I2, I1) << "copy assigned iterator is a different position";
  }
}

TYPED_TEST(MappedIteratorTestBasic, MoveAssign) {
  std::vector<int> V({0});

  using FuncT = GetFuncT<TypeParam>;
  using IterT = mapped_iterator<decltype(V)::iterator, FuncT>;

  EXPECT_TRUE(std::is_move_assignable_v<IterT>);

  if constexpr (std::is_move_assignable_v<IterT>) {
    TypeParam GetCallable;

    IterT I1(V.begin(), GetCallable());
    IterT I2(V.begin(), GetCallable());
    IterT I3(V.end(), GetCallable());

    I3 = std::move(I2);

    EXPECT_EQ(I2, I1) << "move assigned iterator is a different position";
  }
}

TYPED_TEST(MappedIteratorTestBasic, GetFunction) {
  std::vector<int> V({0});

  using FuncT = GetFuncT<TypeParam>;
  using IterT = mapped_iterator<decltype(V)::iterator, FuncT>;

  TypeParam GetCallable;
  IterT I(V.begin(), GetCallable());

  EXPECT_EQ(I.getFunction()(200), 201);
}

TYPED_TEST(MappedIteratorTestBasic, GetCurrent) {
  std::vector<int> V({0});

  using FuncT = GetFuncT<TypeParam>;
  using IterT = mapped_iterator<decltype(V)::iterator, FuncT>;

  TypeParam GetCallable;
  IterT I(V.begin(), GetCallable());

  EXPECT_EQ(I.getCurrent(), V.begin());
  EXPECT_EQ(std::next(I).getCurrent(), V.end());
}

TYPED_TEST(MappedIteratorTestBasic, ApplyFunctionOnDereference) {
  std::vector<int> V({0});
  TypeParam GetCallable;

  auto I = map_iterator(V.begin(), GetCallable());

  EXPECT_EQ(*I, 1) << "should have applied function in dereference";
}

TEST(MappedIteratorTest, ApplyFunctionOnArrow) {
  struct S {
    int Z = 0;
  };

  std::vector<int> V({0});
  S Y;
  S *P = &Y;

  auto I = map_iterator(V.begin(), [&](int X) -> S & { return *(P + X); });

  I->Z = 42;

  EXPECT_EQ(Y.Z, 42) << "should have applied function during arrow";
}

TEST(MappedIteratorTest, FunctionPreservesReferences) {
  std::vector<int> V({1});
  std::map<int, int> M({{1, 1}});

  auto I = map_iterator(V.begin(), [&](int X) -> int & { return M[X]; });
  *I = 42;

  EXPECT_EQ(M[1], 42) << "assignment should have modified M";
}

TEST(MappedIteratorTest, CustomIteratorApplyFunctionOnDereference) {
  struct CustomMapIterator
      : public llvm::mapped_iterator_base<CustomMapIterator,
                                          std::vector<int>::iterator, int> {
    using BaseT::BaseT;

    /// Map the element to the iterator result type.
    int mapElement(int X) const { return X + 1; }
  };

  std::vector<int> V({0});

  CustomMapIterator I(V.begin());

  EXPECT_EQ(*I, 1) << "should have applied function in dereference";
}

TEST(MappedIteratorTest, CustomIteratorApplyFunctionOnArrow) {
  struct S {
    int Z = 0;
  };
  struct CustomMapIterator
      : public llvm::mapped_iterator_base<CustomMapIterator,
                                          std::vector<int>::iterator, S &> {
    CustomMapIterator(std::vector<int>::iterator it, S *P) : BaseT(it), P(P) {}

    /// Map the element to the iterator result type.
    S &mapElement(int X) const { return *(P + X); }

    S *P;
  };

  std::vector<int> V({0});
  S Y;

  CustomMapIterator I(V.begin(), &Y);

  I->Z = 42;

  EXPECT_EQ(Y.Z, 42) << "should have applied function during arrow";
}

TEST(MappedIteratorTest, CustomIteratorFunctionPreservesReferences) {
  struct CustomMapIterator
      : public llvm::mapped_iterator_base<CustomMapIterator,
                                          std::vector<int>::iterator, int &> {
    CustomMapIterator(std::vector<int>::iterator it, std::map<int, int> &M)
        : BaseT(it), M(M) {}

    /// Map the element to the iterator result type.
    int &mapElement(int X) const { return M[X]; }

    std::map<int, int> &M;
  };
  std::vector<int> V({1});
  std::map<int, int> M({{1, 1}});

  auto I = CustomMapIterator(V.begin(), M);
  *I = 42;

  EXPECT_EQ(M[1], 42) << "assignment should have modified M";
}

} // anonymous namespace
