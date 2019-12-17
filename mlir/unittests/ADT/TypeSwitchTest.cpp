//===- TypeSwitchTest.cpp - TypeSwitch unit tests -------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/ADT/TypeSwitch.h"
#include "gtest/gtest.h"

using namespace mlir;

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

TEST(StringSwitchTest, CaseResult) {
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

TEST(StringSwitchTest, CasesResult) {
  auto translate = [](auto value) {
    return TypeSwitch<Base *, int>(&value)
        .Case<DerivedA, DerivedB, DerivedD>([](auto *) { return 0; })
        .Case([](DerivedC *) { return 1; })
        .Default([](Base *) { return -1; });
  };
  EXPECT_EQ(0, translate(DerivedA()));
  EXPECT_EQ(0, translate(DerivedB()));
  EXPECT_EQ(1, translate(DerivedC()));
  EXPECT_EQ(0, translate(DerivedD()));
  EXPECT_EQ(-1, translate(DerivedE()));
}

TEST(StringSwitchTest, CaseVoid) {
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

TEST(StringSwitchTest, CasesVoid) {
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
