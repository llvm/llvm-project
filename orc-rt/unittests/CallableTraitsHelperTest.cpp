//===- CallableTraitsHelperTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's CallableTraitsHelper.h APIs.
//
// NOTE: All tests in this file are testing compile-time functionality, so the
//       tests at runtime all end up being noops. That's fine -- those are
//       cheap.
//===----------------------------------------------------------------------===//

#include "orc-rt/CallableTraitsHelper.h"
#include "gtest/gtest.h"

using namespace orc_rt;

static void freeVoidVoid() {}

TEST(CallableTraitsHelperTest, FreeVoidVoid) {
  (void)freeVoidVoid;
  typedef CallableArgInfo<decltype(freeVoidVoid)> CAI;
  static_assert(std::is_void_v<CAI::return_type>);
  static_assert(std::is_same_v<CAI::args_tuple_type, std::tuple<>>);
}

static int freeBinaryOp(int, float) { return 0; }

TEST(CallableTraitsHelperTest, FreeBinaryOp) {
  (void)freeBinaryOp;
  typedef CallableArgInfo<decltype(freeBinaryOp)> CAI;
  static_assert(std::is_same_v<CAI::return_type, int>);
  static_assert(std::is_same_v<CAI::args_tuple_type, std::tuple<int, float>>);
}

TEST(CallableTraitsHelperTest, VoidVoidObj) {
  auto VoidVoid = []() {};
  typedef CallableArgInfo<decltype(VoidVoid)> CAI;
  static_assert(std::is_void_v<CAI::return_type>);
  static_assert(std::is_same_v<CAI::args_tuple_type, std::tuple<>>);
}

TEST(CallableTraitsHelperTest, BinaryOpObj) {
  auto BinaryOp = [](int X, float Y) -> int { return X + Y; };
  typedef CallableArgInfo<decltype(BinaryOp)> CAI;
  static_assert(std::is_same_v<CAI::return_type, int>);
  static_assert(std::is_same_v<CAI::args_tuple_type, std::tuple<int, float>>);
}

TEST(CallableTraitsHelperTest, PreservesLValueRef) {
  auto RefOp = [](int &) {};
  typedef CallableArgInfo<decltype(RefOp)> CAI;
  static_assert(std::is_same_v<CAI::args_tuple_type, std::tuple<int &>>);
}

TEST(CallableTraitsHelperTest, PreservesLValueRefConstness) {
  auto RefOp = [](const int &) {};
  typedef CallableArgInfo<decltype(RefOp)> CAI;
  static_assert(std::is_same_v<CAI::args_tuple_type, std::tuple<const int &>>);
}

TEST(CallableTraitsHelperTest, PreservesRValueRef) {
  auto RefOp = [](int &&) {};
  typedef CallableArgInfo<decltype(RefOp)> CAI;
  static_assert(std::is_same_v<CAI::args_tuple_type, std::tuple<int &&>>);
}

// Free functions cannot be const-qualified.
TEST(CallableTraitsHelperTest, FreeFunctionIsNonConst) {
  static_assert(!CallableArgInfo<decltype(freeVoidVoid)>::is_const);
  static_assert(!CallableArgInfo<decltype(freeBinaryOp)>::is_const);
}

TEST(CallableTraitsHelperTest, FunctionPointerIsNonConst) {
  static_assert(!CallableArgInfo<int (*)(int, float)>::is_const);
}

TEST(CallableTraitsHelperTest, FunctionReferenceIsNonConst) {
  static_assert(!CallableArgInfo<int (&)(int, float)>::is_const);
}

// A non-mutable lambda's call operator is const-qualified.
TEST(CallableTraitsHelperTest, NonMutableLambdaIsConst) {
  auto L = []() {};
  static_assert(CallableArgInfo<decltype(L)>::is_const);
}

// A mutable lambda's call operator is not const-qualified.
TEST(CallableTraitsHelperTest, MutableLambdaIsNonConst) {
  auto L = []() mutable {};
  static_assert(!CallableArgInfo<decltype(L)>::is_const);
}

TEST(CallableTraitsHelperTest, ConstFunctorIsConst) {
  struct F {
    void operator()() const {}
  };
  static_assert(CallableArgInfo<F>::is_const);
}

TEST(CallableTraitsHelperTest, NonConstFunctorIsNonConst) {
  struct F {
    void operator()() {}
  };
  static_assert(!CallableArgInfo<F>::is_const);
}

TEST(CallableTraitsHelperTest, NonConstMemberFnPtrIsNonConst) {
  struct C {
    void m(int) {}
  };
  static_assert(!CallableArgInfo<decltype(&C::m)>::is_const);
}

TEST(CallableTraitsHelperTest, ConstMemberFnPtrIsConst) {
  struct C {
    void m(int) const {}
  };
  static_assert(CallableArgInfo<decltype(&C::m)>::is_const);
}

TEST(CallableTraitsHelperTest, FunctionTypeIsNonConst) {
  static_assert(!CallableArgInfo<int(int, float)>::is_const);
}

// Abominable function type: const cv-qualifier on the function type itself.
TEST(CallableTraitsHelperTest, AbominableFunctionTypeIsConst) {
  static_assert(CallableArgInfo<int(int, float) const>::is_const);
}

// noexcept coverage — mirrors the const-qualifier tests above.

static void freeVoidVoidNoexcept() noexcept {}

TEST(CallableTraitsHelperTest, FreeFunctionNoexcept) {
  (void)freeVoidVoidNoexcept;
  static_assert(!CallableArgInfo<decltype(freeVoidVoid)>::is_noexcept);
  static_assert(CallableArgInfo<decltype(freeVoidVoidNoexcept)>::is_noexcept);
}

TEST(CallableTraitsHelperTest, FunctionPointerNoexcept) {
  static_assert(!CallableArgInfo<int (*)(int, float)>::is_noexcept);
  static_assert(CallableArgInfo<int (*)(int, float) noexcept>::is_noexcept);
}

TEST(CallableTraitsHelperTest, FunctionReferenceNoexcept) {
  static_assert(!CallableArgInfo<int (&)(int, float)>::is_noexcept);
  static_assert(CallableArgInfo<int (&)(int, float) noexcept>::is_noexcept);
}

TEST(CallableTraitsHelperTest, NonNoexceptLambdaIsNotNoexcept) {
  auto L = []() {};
  static_assert(!CallableArgInfo<decltype(L)>::is_noexcept);
}

TEST(CallableTraitsHelperTest, NoexceptLambdaIsNoexcept) {
  auto L = []() noexcept {};
  static_assert(CallableArgInfo<decltype(L)>::is_noexcept);
}

TEST(CallableTraitsHelperTest, NoexceptFunctor) {
  struct F {
    void operator()() const noexcept {}
  };
  static_assert(CallableArgInfo<F>::is_noexcept);
  static_assert(CallableArgInfo<F>::is_const);
}

TEST(CallableTraitsHelperTest, NonNoexceptFunctor) {
  struct F {
    void operator()() const {}
  };
  static_assert(!CallableArgInfo<F>::is_noexcept);
}

TEST(CallableTraitsHelperTest, NoexceptMemberFnPtr) {
  struct C {
    void m(int) noexcept {}
    void mConstNoexcept(int) const noexcept {}
    void mConst(int) const {}
    void mPlain(int) {}
  };
  static_assert(CallableArgInfo<decltype(&C::m)>::is_noexcept);
  static_assert(!CallableArgInfo<decltype(&C::m)>::is_const);
  static_assert(CallableArgInfo<decltype(&C::mConstNoexcept)>::is_noexcept);
  static_assert(CallableArgInfo<decltype(&C::mConstNoexcept)>::is_const);
  static_assert(!CallableArgInfo<decltype(&C::mConst)>::is_noexcept);
  static_assert(CallableArgInfo<decltype(&C::mConst)>::is_const);
  static_assert(!CallableArgInfo<decltype(&C::mPlain)>::is_noexcept);
  static_assert(!CallableArgInfo<decltype(&C::mPlain)>::is_const);
}

TEST(CallableTraitsHelperTest, FunctionTypeNoexcept) {
  static_assert(!CallableArgInfo<int(int, float)>::is_noexcept);
  static_assert(CallableArgInfo<int(int, float) noexcept>::is_noexcept);
}

// Abominable function type with both const and noexcept cv-qualifiers.
TEST(CallableTraitsHelperTest, AbominableFunctionTypeConstNoexcept) {
  using F = int(int, float) const noexcept;
  static_assert(CallableArgInfo<F>::is_const);
  static_assert(CallableArgInfo<F>::is_noexcept);
}
