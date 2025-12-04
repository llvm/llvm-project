//===- CallableTraitsHelperTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for llvm::orc::CallableTraitsHelper APIs.
//
// NOTE: All tests in this file are testing compile-time functionality, so the
//       tests at runtime all end up being noops. That's fine -- those are
//       cheap.
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/CallableTraitsHelper.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

static void freeVoidVoid() {}

TEST(CallableTraitsHelperTest, FreeVoidVoid) {
  (void)freeVoidVoid;
  typedef CallableArgInfo<decltype(freeVoidVoid)> CAI;
  static_assert(std::is_void_v<CAI::ReturnType>);
  static_assert(std::is_same_v<CAI::ArgsTupleType, std::tuple<>>);
}

static int freeBinaryOp(int, float) { return 0; }

TEST(CallableTraitsHelperTest, FreeBinaryOp) {
  (void)freeBinaryOp;
  typedef CallableArgInfo<decltype(freeBinaryOp)> CAI;
  static_assert(std::is_same_v<CAI::ReturnType, int>);
  static_assert(std::is_same_v<CAI::ArgsTupleType, std::tuple<int, float>>);
}

TEST(CallableTraitsHelperTest, VoidVoidObj) {
  auto VoidVoid = []() {};
  typedef CallableArgInfo<decltype(VoidVoid)> CAI;
  static_assert(std::is_void_v<CAI::ReturnType>);
  static_assert(std::is_same_v<CAI::ArgsTupleType, std::tuple<>>);
}

TEST(CallableTraitsHelperTest, BinaryOpObj) {
  auto BinaryOp = [](int X, float Y) -> int { return X + Y; };
  typedef CallableArgInfo<decltype(BinaryOp)> CAI;
  static_assert(std::is_same_v<CAI::ReturnType, int>);
  static_assert(std::is_same_v<CAI::ArgsTupleType, std::tuple<int, float>>);
}

TEST(CallableTraitsHelperTest, PreservesLValueRef) {
  auto RefOp = [](int &) {};
  typedef CallableArgInfo<decltype(RefOp)> CAI;
  static_assert(std::is_same_v<CAI::ArgsTupleType, std::tuple<int &>>);
}

TEST(CallableTraitsHelperTest, PreservesLValueRefConstness) {
  auto RefOp = [](const int &) {};
  typedef CallableArgInfo<decltype(RefOp)> CAI;
  static_assert(std::is_same_v<CAI::ArgsTupleType, std::tuple<const int &>>);
}

TEST(CallableTraitsHelperTest, PreservesRValueRef) {
  auto RefOp = [](int &&) {};
  typedef CallableArgInfo<decltype(RefOp)> CAI;
  static_assert(std::is_same_v<CAI::ArgsTupleType, std::tuple<int &&>>);
}
