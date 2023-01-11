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

constexpr TypeSize TSFixed0 = TypeSize::Fixed(0);
constexpr TypeSize TSFixed1 = TypeSize::Fixed(1);
constexpr TypeSize TSFixed32 = TypeSize::Fixed(32);

static_assert(TSFixed0.getFixedSize() == 0);
static_assert(TSFixed1.getFixedSize() == 1);
static_assert(TSFixed32.getFixedSize() == 32);
static_assert(TSFixed32.getKnownMinValue() == 32);

static_assert(TypeSize::Scalable(32).getKnownMinValue() == 32);

static_assert(TSFixed32 * 2 == TypeSize::Fixed(64));
static_assert(TSFixed32 * 2u == TypeSize::Fixed(64));
static_assert(TSFixed32 * INT64_C(2) == TypeSize::Fixed(64));
static_assert(TSFixed32 * UINT64_C(2) == TypeSize::Fixed(64));

static_assert(2 * TSFixed32 == TypeSize::Fixed(64));
static_assert(2u * TSFixed32 == TypeSize::Fixed(64));
static_assert(INT64_C(2) * TSFixed32 == TypeSize::Fixed(64));
static_assert(UINT64_C(2) * TSFixed32 == TypeSize::Fixed(64));
static_assert(alignTo(TypeSize::Fixed(7), 8) == TypeSize::Fixed(8));

} // namespace
