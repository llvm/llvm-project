//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/TypeSize.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

constexpr ElementCount CEElementCount = ElementCount();

static_assert(!CEElementCount.isScalar());
static_assert(!CEElementCount.isVector());

constexpr ElementCount CEElementCountFixed1 = ElementCount::getFixed(1);
static_assert(CEElementCountFixed1.isScalar());
static_assert(!CEElementCountFixed1.isVector());
static_assert(!CEElementCountFixed1.isScalable());

constexpr ElementCount CEElementCountFixed3 = ElementCount::getFixed(3);
constexpr ElementCount CEElementCountFixed4 = ElementCount::getFixed(4);

static_assert(!CEElementCountFixed4.isScalar());
static_assert(CEElementCountFixed4.isVector());
static_assert(CEElementCountFixed4.isKnownEven());
static_assert(!CEElementCountFixed3.isKnownEven());
static_assert(!CEElementCountFixed4.isScalable());
static_assert(!CEElementCountFixed3.isScalable());

constexpr ElementCount CEElementCountScalable4 = ElementCount::getScalable(4);

static_assert(CEElementCountScalable4.isScalable());
static_assert(!ElementCount().isScalable());
static_assert(
    CEElementCountScalable4.hasKnownScalarFactor(ElementCount::getScalable(2)));
static_assert(ElementCount::getScalable(8).getKnownScalarFactor(
                  ElementCount::getScalable(2)) == 4);

static_assert(CEElementCountScalable4 == ElementCount::get(4, true));
static_assert(CEElementCountFixed4 == ElementCount::get(4, false));
static_assert(ElementCount::isKnownLT(CEElementCountFixed3,
                                      CEElementCountFixed4));
static_assert(ElementCount::isKnownLE(CEElementCountFixed3,
                                      CEElementCountFixed4));
static_assert(ElementCount::isKnownGT(CEElementCountFixed4,
                                      CEElementCountFixed3));
static_assert(ElementCount::isKnownGE(CEElementCountFixed4,
                                      CEElementCountFixed3));
static_assert(CEElementCountFixed3.coefficientNextPowerOf2() ==
              CEElementCountFixed4);
static_assert(ElementCount::getFixed(8).divideCoefficientBy(2) ==
              ElementCount::getFixed(4));
static_assert(ElementCount::getFixed(8).multiplyCoefficientBy(3) ==
              ElementCount::getFixed(24));
static_assert(ElementCount::getFixed(8).isKnownMultipleOf(2));

constexpr TypeSize TSFixed0 = TypeSize::getFixed(0);
constexpr TypeSize TSFixed1 = TypeSize::getFixed(1);
constexpr TypeSize TSFixed32 = TypeSize::getFixed(32);

static_assert(TSFixed0.getFixedValue() == 0);
static_assert(TSFixed1.getFixedValue() == 1);
static_assert(TSFixed32.getFixedValue() == 32);
static_assert(TSFixed32.getKnownMinValue() == 32);

static_assert(TypeSize::getScalable(32).getKnownMinValue() == 32);

static_assert(TSFixed32 * 2 == TypeSize::getFixed(64));
static_assert(TSFixed32 * 2u == TypeSize::getFixed(64));
static_assert(TSFixed32 * INT64_C(2) == TypeSize::getFixed(64));
static_assert(TSFixed32 * UINT64_C(2) == TypeSize::getFixed(64));

static_assert(2 * TSFixed32 == TypeSize::getFixed(64));
static_assert(2u * TSFixed32 == TypeSize::getFixed(64));
static_assert(INT64_C(2) * TSFixed32 == TypeSize::getFixed(64));
static_assert(UINT64_C(2) * TSFixed32 == TypeSize::getFixed(64));
static_assert(alignTo(TypeSize::getFixed(7), 8) == TypeSize::getFixed(8));

static_assert(TypeSize::getZero() == TypeSize::getFixed(0));
static_assert(TypeSize::getZero() != TypeSize::getScalable(0));
static_assert(TypeSize::getFixed(0) != TypeSize::getScalable(0));
static_assert(TypeSize::getFixed(0).isZero());
static_assert(TypeSize::getScalable(0).isZero());
static_assert(TypeSize::getZero().isZero());
static_assert(TypeSize::getFixed(0) ==
              (TypeSize::getFixed(4) - TypeSize::getFixed(4)));
static_assert(TypeSize::getScalable(0) ==
              (TypeSize::getScalable(4) - TypeSize::getScalable(4)));
static_assert(TypeSize::getFixed(0) + TypeSize::getScalable(8) ==
              TypeSize::getScalable(8));
static_assert(TypeSize::getScalable(8) + TypeSize::getFixed(0) ==
              TypeSize::getScalable(8));
static_assert(TypeSize::getFixed(8) + TypeSize::getScalable(0) ==
              TypeSize::getFixed(8));
static_assert(TypeSize::getScalable(0) + TypeSize::getFixed(8) ==
              TypeSize::getFixed(8));
static_assert(TypeSize::getScalable(8) - TypeSize::getFixed(0) ==
              TypeSize::getScalable(8));
static_assert(TypeSize::getFixed(8) - TypeSize::getScalable(0) ==
              TypeSize::getFixed(8));

TEST(TypeSize, FailIncompatibleTypes) {
  EXPECT_DEBUG_DEATH(TypeSize::getFixed(8) + TypeSize::getScalable(8),
                     "Incompatible types");
}

} // namespace
