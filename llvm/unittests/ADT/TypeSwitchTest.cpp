//===- TypeSwitchTest.cpp - TypeSwitch unit tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
/// Utility classes to setup casting functionality.
struct Base {
  enum Kind { DerivedA, DerivedB, DerivedC, DerivedD, DerivedE };
  Kind kind;
};
template <Base::Kind DerivedKind> struct DerivedImpl : Base {
  DerivedImpl() : Base{DerivedKind} {}
  static bool classof(const Base *base) { return base->kind == DerivedKind; }
};
struct DerivedA : public DerivedImpl<Base::DerivedA> {};
struct DerivedB : public DerivedImpl<Base::DerivedB> {};
struct DerivedC : public DerivedImpl<Base::DerivedC> {};
struct DerivedD : public DerivedImpl<Base::DerivedD> {};
struct DerivedE : public DerivedImpl<Base::DerivedE> {};
} // end anonymous namespace

TEST(TypeSwitchTest, CaseResult) {
  auto translate = [](auto value) {
    return TypeSwitch<Base *, int>(&value)
        .Case<DerivedA>([](DerivedA *) { return 0; })
        .Case([](DerivedB *) { return 1; })
        .Case([](DerivedC *) { return 2; })
        .Default([](Base *) { return -1; });
  };
  EXPECT_EQ(0, translate(DerivedA()));
  EXPECT_EQ(1, translate(DerivedB()));
  EXPECT_EQ(2, translate(DerivedC()));
  EXPECT_EQ(-1, translate(DerivedD()));
}

TEST(TypeSwitchTest, CasesResult) {
  auto translate = [](auto value) {
    return TypeSwitch<Base *, int>(&value)
        .Case<DerivedA, DerivedB, DerivedD>([](auto *) { return 0; })
        .Case([](DerivedC *) { return 1; })
        .Default(-1);
  };
  EXPECT_EQ(0, translate(DerivedA()));
  EXPECT_EQ(0, translate(DerivedB()));
  EXPECT_EQ(1, translate(DerivedC()));
  EXPECT_EQ(0, translate(DerivedD()));
  EXPECT_EQ(-1, translate(DerivedE()));
}

TEST(TypeSwitchTest, CaseVoid) {
  auto translate = [](auto value) {
    int result = -2;
    TypeSwitch<Base *>(&value)
        .Case([&](DerivedA *) { result = 0; })
        .Case([&](DerivedB *) { result = 1; })
        .Case([&](DerivedC *) { result = 2; })
        .Default([&](Base *) { result = -1; });
    return result;
  };
  EXPECT_EQ(0, translate(DerivedA()));
  EXPECT_EQ(1, translate(DerivedB()));
  EXPECT_EQ(2, translate(DerivedC()));
  EXPECT_EQ(-1, translate(DerivedD()));
}

TEST(TypeSwitchTest, CasesVoid) {
  auto translate = [](auto value) {
    int result = -1;
    TypeSwitch<Base *>(&value)
        .Case<DerivedA, DerivedB, DerivedD>([&](auto *) { result = 0; })
        .Case([&](DerivedC *) { result = 1; });
    return result;
  };
  EXPECT_EQ(0, translate(DerivedA()));
  EXPECT_EQ(0, translate(DerivedB()));
  EXPECT_EQ(1, translate(DerivedC()));
  EXPECT_EQ(0, translate(DerivedD()));
  EXPECT_EQ(-1, translate(DerivedE()));
}
