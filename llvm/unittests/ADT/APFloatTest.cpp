//===- llvm/unittest/ADT/APFloat.cpp - APFloat unit tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"
#include <cmath>
#include <ostream>
#include <string>
#include <tuple>

using namespace llvm;

static std::string convertToErrorFromString(StringRef Str) {
  llvm::APFloat F(0.0);
  auto StatusOrErr =
      F.convertFromString(Str, llvm::APFloat::rmNearestTiesToEven);
  EXPECT_TRUE(!StatusOrErr);
  return toString(StatusOrErr.takeError());
}

static double convertToDoubleFromString(StringRef Str) {
  llvm::APFloat F(0.0);
  auto StatusOrErr =
      F.convertFromString(Str, llvm::APFloat::rmNearestTiesToEven);
  EXPECT_FALSE(!StatusOrErr);
  consumeError(StatusOrErr.takeError());
  return F.convertToDouble();
}

static std::string convertToString(double d, unsigned Prec, unsigned Pad,
                                   bool Tr = true) {
  llvm::SmallVector<char, 100> Buffer;
  llvm::APFloat F(d);
  F.toString(Buffer, Prec, Pad, Tr);
  return std::string(Buffer.data(), Buffer.size());
}

namespace {

TEST(APFloatTest, isSignaling) {
  // We test qNaN, -qNaN, +sNaN, -sNaN with and without payloads. *NOTE* The
  // positive/negative distinction is included only since the getQNaN/getSNaN
  // API provides the option.
  APInt payload = APInt::getOneBitSet(4, 2);
  APFloat QNan = APFloat::getQNaN(APFloat::IEEEsingle(), false);
  EXPECT_FALSE(QNan.isSignaling());
  EXPECT_EQ(fcQNan, QNan.classify());

  EXPECT_FALSE(APFloat::getQNaN(APFloat::IEEEsingle(), true).isSignaling());
  EXPECT_FALSE(APFloat::getQNaN(APFloat::IEEEsingle(), false, &payload).isSignaling());
  EXPECT_FALSE(APFloat::getQNaN(APFloat::IEEEsingle(), true, &payload).isSignaling());

  APFloat SNan = APFloat::getSNaN(APFloat::IEEEsingle(), false);
  EXPECT_TRUE(SNan.isSignaling());
  EXPECT_EQ(fcSNan, SNan.classify());

  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle(), true).isSignaling());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle(), false, &payload).isSignaling());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle(), true, &payload).isSignaling());
}

TEST(APFloatTest, next) {

  APFloat test(APFloat::IEEEquad(), APFloat::uninitialized);
  APFloat expected(APFloat::IEEEquad(), APFloat::uninitialized);

  // 1. Test Special Cases Values.
  //
  // Test all special values for nextUp and nextDown perscribed by IEEE-754R
  // 2008. These are:
  //   1.  +inf
  //   2.  -inf
  //   3.  getLargest()
  //   4.  -getLargest()
  //   5.  getSmallest()
  //   6.  -getSmallest()
  //   7.  qNaN
  //   8.  sNaN
  //   9.  +0
  //   10. -0

  // nextUp(+inf) = +inf.
  test = APFloat::getInf(APFloat::IEEEquad(), false);
  expected = APFloat::getInf(APFloat::IEEEquad(), false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+inf) = -nextUp(-inf) = -(-getLargest()) = getLargest()
  test = APFloat::getInf(APFloat::IEEEquad(), false);
  expected = APFloat::getLargest(APFloat::IEEEquad(), false);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-inf) = -getLargest()
  test = APFloat::getInf(APFloat::IEEEquad(), true);
  expected = APFloat::getLargest(APFloat::IEEEquad(), true);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-inf) = -nextUp(+inf) = -(+inf) = -inf.
  test = APFloat::getInf(APFloat::IEEEquad(), true);
  expected = APFloat::getInf(APFloat::IEEEquad(), true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity() && test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(getLargest()) = +inf
  test = APFloat::getLargest(APFloat::IEEEquad(), false);
  expected = APFloat::getInf(APFloat::IEEEquad(), false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity() && !test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(getLargest()) = -nextUp(-getLargest())
  //                        = -(-getLargest() + inc)
  //                        = getLargest() - inc.
  test = APFloat::getLargest(APFloat::IEEEquad(), false);
  expected = APFloat(APFloat::IEEEquad(),
                     "0x1.fffffffffffffffffffffffffffep+16383");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isInfinity() && !test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-getLargest()) = -getLargest() + inc.
  test = APFloat::getLargest(APFloat::IEEEquad(), true);
  expected = APFloat(APFloat::IEEEquad(),
                     "-0x1.fffffffffffffffffffffffffffep+16383");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-getLargest()) = -nextUp(getLargest()) = -(inf) = -inf.
  test = APFloat::getLargest(APFloat::IEEEquad(), true);
  expected = APFloat::getInf(APFloat::IEEEquad(), true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity() && test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(getSmallest()) = getSmallest() + inc.
  test = APFloat(APFloat::IEEEquad(), "0x0.0000000000000000000000000001p-16382");
  expected = APFloat(APFloat::IEEEquad(),
                     "0x0.0000000000000000000000000002p-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(getSmallest()) = -nextUp(-getSmallest()) = -(-0) = +0.
  test = APFloat(APFloat::IEEEquad(), "0x0.0000000000000000000000000001p-16382");
  expected = APFloat::getZero(APFloat::IEEEquad(), false);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isPosZero());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-getSmallest()) = -0.
  test = APFloat(APFloat::IEEEquad(), "-0x0.0000000000000000000000000001p-16382");
  expected = APFloat::getZero(APFloat::IEEEquad(), true);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isNegZero());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-getSmallest()) = -nextUp(getSmallest()) = -getSmallest() - inc.
  test = APFloat(APFloat::IEEEquad(), "-0x0.0000000000000000000000000001p-16382");
  expected = APFloat(APFloat::IEEEquad(),
                     "-0x0.0000000000000000000000000002p-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(qNaN) = qNaN
  test = APFloat::getQNaN(APFloat::IEEEquad(), false);
  expected = APFloat::getQNaN(APFloat::IEEEquad(), false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(qNaN) = qNaN
  test = APFloat::getQNaN(APFloat::IEEEquad(), false);
  expected = APFloat::getQNaN(APFloat::IEEEquad(), false);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(sNaN) = qNaN
  test = APFloat::getSNaN(APFloat::IEEEquad(), false);
  expected = APFloat::getQNaN(APFloat::IEEEquad(), false);
  EXPECT_EQ(test.next(false), APFloat::opInvalidOp);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(sNaN) = qNaN
  test = APFloat::getSNaN(APFloat::IEEEquad(), false);
  expected = APFloat::getQNaN(APFloat::IEEEquad(), false);
  EXPECT_EQ(test.next(true), APFloat::opInvalidOp);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+0) = +getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad(), false);
  expected = APFloat::getSmallest(APFloat::IEEEquad(), false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+0) = -nextUp(-0) = -getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad(), false);
  expected = APFloat::getSmallest(APFloat::IEEEquad(), true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-0) = +getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad(), true);
  expected = APFloat::getSmallest(APFloat::IEEEquad(), false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-0) = -nextUp(0) = -getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad(), true);
  expected = APFloat::getSmallest(APFloat::IEEEquad(), true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 2. Binade Boundary Tests.

  // 2a. Test denormal <-> normal binade boundaries.
  //     * nextUp(+Largest Denormal) -> +Smallest Normal.
  //     * nextDown(-Largest Denormal) -> -Smallest Normal.
  //     * nextUp(-Smallest Normal) -> -Largest Denormal.
  //     * nextDown(+Smallest Normal) -> +Largest Denormal.

  // nextUp(+Largest Denormal) -> +Smallest Normal.
  test = APFloat(APFloat::IEEEquad(), "0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad(),
                     "0x1.0000000000000000000000000000p-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Largest Denormal) -> -Smallest Normal.
  test = APFloat(APFloat::IEEEquad(),
                 "-0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad(),
                     "-0x1.0000000000000000000000000000p-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-Smallest Normal) -> -LargestDenormal.
  test = APFloat(APFloat::IEEEquad(),
                 "-0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad(),
                     "-0x0.ffffffffffffffffffffffffffffp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Smallest Normal) -> +Largest Denormal.
  test = APFloat(APFloat::IEEEquad(),
                 "+0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad(),
                     "+0x0.ffffffffffffffffffffffffffffp-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 2b. Test normal <-> normal binade boundaries.
  //     * nextUp(-Normal Binade Boundary) -> -Normal Binade Boundary + 1.
  //     * nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
  //     * nextUp(+Normal Binade Boundary - 1) -> +Normal Binade Boundary.
  //     * nextDown(-Normal Binade Boundary + 1) -> -Normal Binade Boundary.

  // nextUp(-Normal Binade Boundary) -> -Normal Binade Boundary + 1.
  test = APFloat(APFloat::IEEEquad(), "-0x1p+1");
  expected = APFloat(APFloat::IEEEquad(),
                     "-0x1.ffffffffffffffffffffffffffffp+0");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
  test = APFloat(APFloat::IEEEquad(), "0x1p+1");
  expected = APFloat(APFloat::IEEEquad(), "0x1.ffffffffffffffffffffffffffffp+0");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+Normal Binade Boundary - 1) -> +Normal Binade Boundary.
  test = APFloat(APFloat::IEEEquad(), "0x1.ffffffffffffffffffffffffffffp+0");
  expected = APFloat(APFloat::IEEEquad(), "0x1p+1");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Normal Binade Boundary + 1) -> -Normal Binade Boundary.
  test = APFloat(APFloat::IEEEquad(), "-0x1.ffffffffffffffffffffffffffffp+0");
  expected = APFloat(APFloat::IEEEquad(), "-0x1p+1");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 2c. Test using next at binade boundaries with a direction away from the
  // binade boundary. Away from denormal <-> normal boundaries.
  //
  // This is to make sure that even though we are at a binade boundary, since
  // we are rounding away, we do not trigger the binade boundary code. Thus we
  // test:
  //   * nextUp(-Largest Denormal) -> -Largest Denormal + inc.
  //   * nextDown(+Largest Denormal) -> +Largest Denormal - inc.
  //   * nextUp(+Smallest Normal) -> +Smallest Normal + inc.
  //   * nextDown(-Smallest Normal) -> -Smallest Normal - inc.

  // nextUp(-Largest Denormal) -> -Largest Denormal + inc.
  test = APFloat(APFloat::IEEEquad(), "-0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad(),
                     "-0x0.fffffffffffffffffffffffffffep-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Largest Denormal) -> +Largest Denormal - inc.
  test = APFloat(APFloat::IEEEquad(), "0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad(),
                     "0x0.fffffffffffffffffffffffffffep-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+Smallest Normal) -> +Smallest Normal + inc.
  test = APFloat(APFloat::IEEEquad(), "0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad(),
                     "0x1.0000000000000000000000000001p-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Smallest Normal) -> -Smallest Normal - inc.
  test = APFloat(APFloat::IEEEquad(), "-0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad(),
                     "-0x1.0000000000000000000000000001p-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 2d. Test values which cause our exponent to go to min exponent. This
  // is to ensure that guards in the code to check for min exponent
  // trigger properly.
  //     * nextUp(-0x1p-16381) -> -0x1.ffffffffffffffffffffffffffffp-16382
  //     * nextDown(-0x1.ffffffffffffffffffffffffffffp-16382) ->
  //         -0x1p-16381
  //     * nextUp(0x1.ffffffffffffffffffffffffffffp-16382) -> 0x1p-16382
  //     * nextDown(0x1p-16382) -> 0x1.ffffffffffffffffffffffffffffp-16382

  // nextUp(-0x1p-16381) -> -0x1.ffffffffffffffffffffffffffffp-16382
  test = APFloat(APFloat::IEEEquad(), "-0x1p-16381");
  expected = APFloat(APFloat::IEEEquad(),
                     "-0x1.ffffffffffffffffffffffffffffp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-0x1.ffffffffffffffffffffffffffffp-16382) ->
  //         -0x1p-16381
  test = APFloat(APFloat::IEEEquad(), "-0x1.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad(), "-0x1p-16381");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(0x1.ffffffffffffffffffffffffffffp-16382) -> 0x1p-16381
  test = APFloat(APFloat::IEEEquad(), "0x1.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad(), "0x1p-16381");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(0x1p-16381) -> 0x1.ffffffffffffffffffffffffffffp-16382
  test = APFloat(APFloat::IEEEquad(), "0x1p-16381");
  expected = APFloat(APFloat::IEEEquad(),
                     "0x1.ffffffffffffffffffffffffffffp-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 3. Now we test both denormal/normal computation which will not cause us
  // to go across binade boundaries. Specifically we test:
  //   * nextUp(+Denormal) -> +Denormal.
  //   * nextDown(+Denormal) -> +Denormal.
  //   * nextUp(-Denormal) -> -Denormal.
  //   * nextDown(-Denormal) -> -Denormal.
  //   * nextUp(+Normal) -> +Normal.
  //   * nextDown(+Normal) -> +Normal.
  //   * nextUp(-Normal) -> -Normal.
  //   * nextDown(-Normal) -> -Normal.

  // nextUp(+Denormal) -> +Denormal.
  test = APFloat(APFloat::IEEEquad(),
                 "0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad(),
                 "0x0.ffffffffffffffffffffffff000dp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Denormal) -> +Denormal.
  test = APFloat(APFloat::IEEEquad(),
                 "0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad(),
                 "0x0.ffffffffffffffffffffffff000bp-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-Denormal) -> -Denormal.
  test = APFloat(APFloat::IEEEquad(),
                 "-0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad(),
                 "-0x0.ffffffffffffffffffffffff000bp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Denormal) -> -Denormal
  test = APFloat(APFloat::IEEEquad(),
                 "-0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad(),
                 "-0x0.ffffffffffffffffffffffff000dp-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+Normal) -> +Normal.
  test = APFloat(APFloat::IEEEquad(),
                 "0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad(),
                 "0x1.ffffffffffffffffffffffff000dp-16000");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Normal) -> +Normal.
  test = APFloat(APFloat::IEEEquad(),
                 "0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad(),
                 "0x1.ffffffffffffffffffffffff000bp-16000");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-Normal) -> -Normal.
  test = APFloat(APFloat::IEEEquad(),
                 "-0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad(),
                 "-0x1.ffffffffffffffffffffffff000bp-16000");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Normal) -> -Normal.
  test = APFloat(APFloat::IEEEquad(),
                 "-0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad(),
                 "-0x1.ffffffffffffffffffffffff000dp-16000");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
}

TEST(APFloatTest, FMA) {
  APFloat::roundingMode rdmd = APFloat::rmNearestTiesToEven;

  {
    APFloat f1(14.5f);
    APFloat f2(-14.5f);
    APFloat f3(225.0f);
    f1.fusedMultiplyAdd(f2, f3, APFloat::rmNearestTiesToEven);
    EXPECT_EQ(14.75f, f1.convertToFloat());
  }

  {
    APFloat Val2(2.0f);
    APFloat f1((float)1.17549435e-38F);
    APFloat f2((float)1.17549435e-38F);
    f1.divide(Val2, rdmd);
    f2.divide(Val2, rdmd);
    APFloat f3(12.0f);
    f1.fusedMultiplyAdd(f2, f3, APFloat::rmNearestTiesToEven);
    EXPECT_EQ(12.0f, f1.convertToFloat());
  }

  // Test for correct zero sign when answer is exactly zero.
  // fma(1.0, -1.0, 1.0) -> +ve 0.
  {
    APFloat f1(1.0);
    APFloat f2(-1.0);
    APFloat f3(1.0);
    f1.fusedMultiplyAdd(f2, f3, APFloat::rmNearestTiesToEven);
    EXPECT_TRUE(!f1.isNegative() && f1.isZero());
  }

  // Test for correct zero sign when answer is exactly zero and rounding towards
  // negative.
  // fma(1.0, -1.0, 1.0) -> +ve 0.
  {
    APFloat f1(1.0);
    APFloat f2(-1.0);
    APFloat f3(1.0);
    f1.fusedMultiplyAdd(f2, f3, APFloat::rmTowardNegative);
    EXPECT_TRUE(f1.isNegative() && f1.isZero());
  }

  // Test for correct (in this case -ve) sign when adding like signed zeros.
  // Test fma(0.0, -0.0, -0.0) -> -ve 0.
  {
    APFloat f1(0.0);
    APFloat f2(-0.0);
    APFloat f3(-0.0);
    f1.fusedMultiplyAdd(f2, f3, APFloat::rmNearestTiesToEven);
    EXPECT_TRUE(f1.isNegative() && f1.isZero());
  }

  // Test -ve sign preservation when small negative results underflow.
  {
    APFloat f1(APFloat::IEEEdouble(),  "-0x1p-1074");
    APFloat f2(APFloat::IEEEdouble(), "+0x1p-1074");
    APFloat f3(0.0);
    f1.fusedMultiplyAdd(f2, f3, APFloat::rmNearestTiesToEven);
    EXPECT_TRUE(f1.isNegative() && f1.isZero());
  }

  // Test x87 extended precision case from http://llvm.org/PR20728.
  {
    APFloat M1(APFloat::x87DoubleExtended(), 1);
    APFloat M2(APFloat::x87DoubleExtended(), 1);
    APFloat A(APFloat::x87DoubleExtended(), 3);

    bool losesInfo = false;
    M1.fusedMultiplyAdd(M1, A, APFloat::rmNearestTiesToEven);
    M1.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
    EXPECT_FALSE(losesInfo);
    EXPECT_EQ(4.0f, M1.convertToFloat());
  }

  // Regression test that failed an assertion.
  {
    APFloat f1(-8.85242279E-41f);
    APFloat f2(2.0f);
    APFloat f3(8.85242279E-41f);
    f1.fusedMultiplyAdd(f2, f3, APFloat::rmNearestTiesToEven);
    EXPECT_EQ(-8.85242279E-41f, f1.convertToFloat());
  }

  // Test using only a single instance of APFloat.
  {
    APFloat F(1.5);

    F.fusedMultiplyAdd(F, F, APFloat::rmNearestTiesToEven);
    EXPECT_EQ(3.75, F.convertToDouble());
  }
}

TEST(APFloatTest, MinNum) {
  APFloat f1(1.0);
  APFloat f2(2.0);
  APFloat nan = APFloat::getNaN(APFloat::IEEEdouble());

  EXPECT_EQ(1.0, minnum(f1, f2).convertToDouble());
  EXPECT_EQ(1.0, minnum(f2, f1).convertToDouble());
  EXPECT_EQ(1.0, minnum(f1, nan).convertToDouble());
  EXPECT_EQ(1.0, minnum(nan, f1).convertToDouble());
}

TEST(APFloatTest, MaxNum) {
  APFloat f1(1.0);
  APFloat f2(2.0);
  APFloat nan = APFloat::getNaN(APFloat::IEEEdouble());

  EXPECT_EQ(2.0, maxnum(f1, f2).convertToDouble());
  EXPECT_EQ(2.0, maxnum(f2, f1).convertToDouble());
  EXPECT_EQ(1.0, maxnum(f1, nan).convertToDouble());
  EXPECT_EQ(1.0, maxnum(nan, f1).convertToDouble());
}

TEST(APFloatTest, Minimum) {
  APFloat f1(1.0);
  APFloat f2(2.0);
  APFloat zp(0.0);
  APFloat zn(-0.0);
  APFloat nan = APFloat::getNaN(APFloat::IEEEdouble());

  EXPECT_EQ(1.0, minimum(f1, f2).convertToDouble());
  EXPECT_EQ(1.0, minimum(f2, f1).convertToDouble());
  EXPECT_EQ(-0.0, minimum(zp, zn).convertToDouble());
  EXPECT_EQ(-0.0, minimum(zn, zp).convertToDouble());
  EXPECT_TRUE(std::isnan(minimum(f1, nan).convertToDouble()));
  EXPECT_TRUE(std::isnan(minimum(nan, f1).convertToDouble()));
}

TEST(APFloatTest, Maximum) {
  APFloat f1(1.0);
  APFloat f2(2.0);
  APFloat zp(0.0);
  APFloat zn(-0.0);
  APFloat nan = APFloat::getNaN(APFloat::IEEEdouble());

  EXPECT_EQ(2.0, maximum(f1, f2).convertToDouble());
  EXPECT_EQ(2.0, maximum(f2, f1).convertToDouble());
  EXPECT_EQ(0.0, maximum(zp, zn).convertToDouble());
  EXPECT_EQ(0.0, maximum(zn, zp).convertToDouble());
  EXPECT_TRUE(std::isnan(maximum(f1, nan).convertToDouble()));
  EXPECT_TRUE(std::isnan(maximum(nan, f1).convertToDouble()));
}

TEST(APFloatTest, Denormal) {
  APFloat::roundingMode rdmd = APFloat::rmNearestTiesToEven;

  // Test single precision
  {
    const char *MinNormalStr = "1.17549435082228750797e-38";
    EXPECT_FALSE(APFloat(APFloat::IEEEsingle(), MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::IEEEsingle(), 0).isDenormal());

    APFloat Val2(APFloat::IEEEsingle(), 2);
    APFloat T(APFloat::IEEEsingle(), MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
    EXPECT_EQ(fcPosSubnormal, T.classify());


    const char *NegMinNormalStr = "-1.17549435082228750797e-38";
    EXPECT_FALSE(APFloat(APFloat::IEEEsingle(), NegMinNormalStr).isDenormal());
    APFloat NegT(APFloat::IEEEsingle(), NegMinNormalStr);
    NegT.divide(Val2, rdmd);
    EXPECT_TRUE(NegT.isDenormal());
    EXPECT_EQ(fcNegSubnormal, NegT.classify());
  }

  // Test double precision
  {
    const char *MinNormalStr = "2.22507385850720138309e-308";
    EXPECT_FALSE(APFloat(APFloat::IEEEdouble(), MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::IEEEdouble(), 0).isDenormal());

    APFloat Val2(APFloat::IEEEdouble(), 2);
    APFloat T(APFloat::IEEEdouble(), MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
    EXPECT_EQ(fcPosSubnormal, T.classify());
  }

  // Test Intel double-ext
  {
    const char *MinNormalStr = "3.36210314311209350626e-4932";
    EXPECT_FALSE(APFloat(APFloat::x87DoubleExtended(), MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::x87DoubleExtended(), 0).isDenormal());

    APFloat Val2(APFloat::x87DoubleExtended(), 2);
    APFloat T(APFloat::x87DoubleExtended(), MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
    EXPECT_EQ(fcPosSubnormal, T.classify());
  }

  // Test quadruple precision
  {
    const char *MinNormalStr = "3.36210314311209350626267781732175260e-4932";
    EXPECT_FALSE(APFloat(APFloat::IEEEquad(), MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::IEEEquad(), 0).isDenormal());

    APFloat Val2(APFloat::IEEEquad(), 2);
    APFloat T(APFloat::IEEEquad(), MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
    EXPECT_EQ(fcPosSubnormal, T.classify());
  }

  // Test TF32
  {
    const char *MinNormalStr = "1.17549435082228750797e-38";
    EXPECT_FALSE(APFloat(APFloat::FloatTF32(), MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::FloatTF32(), 0).isDenormal());

    APFloat Val2(APFloat::FloatTF32(), 2);
    APFloat T(APFloat::FloatTF32(), MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
    EXPECT_EQ(fcPosSubnormal, T.classify());

    const char *NegMinNormalStr = "-1.17549435082228750797e-38";
    EXPECT_FALSE(APFloat(APFloat::FloatTF32(), NegMinNormalStr).isDenormal());
    APFloat NegT(APFloat::FloatTF32(), NegMinNormalStr);
    NegT.divide(Val2, rdmd);
    EXPECT_TRUE(NegT.isDenormal());
    EXPECT_EQ(fcNegSubnormal, NegT.classify());
  }
}

TEST(APFloatTest, IsSmallestNormalized) {
  for (unsigned I = 0; I != APFloat::S_MaxSemantics + 1; ++I) {
    const fltSemantics &Semantics =
        APFloat::EnumToSemantics(static_cast<APFloat::Semantics>(I));

    EXPECT_FALSE(APFloat::getZero(Semantics, false).isSmallestNormalized());
    EXPECT_FALSE(APFloat::getZero(Semantics, true).isSmallestNormalized());

    EXPECT_FALSE(APFloat::getInf(Semantics, false).isSmallestNormalized());
    EXPECT_FALSE(APFloat::getInf(Semantics, true).isSmallestNormalized());

    EXPECT_FALSE(APFloat::getQNaN(Semantics).isSmallestNormalized());
    EXPECT_FALSE(APFloat::getSNaN(Semantics).isSmallestNormalized());

    EXPECT_FALSE(APFloat::getLargest(Semantics).isSmallestNormalized());
    EXPECT_FALSE(APFloat::getLargest(Semantics, true).isSmallestNormalized());

    EXPECT_FALSE(APFloat::getSmallest(Semantics).isSmallestNormalized());
    EXPECT_FALSE(APFloat::getSmallest(Semantics, true).isSmallestNormalized());

    EXPECT_FALSE(APFloat::getAllOnesValue(Semantics).isSmallestNormalized());

    APFloat PosSmallestNormalized =
        APFloat::getSmallestNormalized(Semantics, false);
    APFloat NegSmallestNormalized =
        APFloat::getSmallestNormalized(Semantics, true);
    EXPECT_TRUE(PosSmallestNormalized.isSmallestNormalized());
    EXPECT_TRUE(NegSmallestNormalized.isSmallestNormalized());
    EXPECT_EQ(fcPosNormal, PosSmallestNormalized.classify());
    EXPECT_EQ(fcNegNormal, NegSmallestNormalized.classify());

    for (APFloat *Val : {&PosSmallestNormalized, &NegSmallestNormalized}) {
      bool OldSign = Val->isNegative();

      // Step down, make sure it's still not smallest normalized.
      EXPECT_EQ(APFloat::opOK, Val->next(false));
      EXPECT_EQ(OldSign, Val->isNegative());
      EXPECT_FALSE(Val->isSmallestNormalized());
      EXPECT_EQ(OldSign, Val->isNegative());

      // Step back up should restore it to being smallest normalized.
      EXPECT_EQ(APFloat::opOK, Val->next(true));
      EXPECT_TRUE(Val->isSmallestNormalized());
      EXPECT_EQ(OldSign, Val->isNegative());

      // Step beyond should no longer smallest normalized.
      EXPECT_EQ(APFloat::opOK, Val->next(true));
      EXPECT_FALSE(Val->isSmallestNormalized());
      EXPECT_EQ(OldSign, Val->isNegative());
    }
  }
}

TEST(APFloatTest, Zero) {
  EXPECT_EQ(0.0f,  APFloat(0.0f).convertToFloat());
  EXPECT_EQ(-0.0f, APFloat(-0.0f).convertToFloat());
  EXPECT_TRUE(APFloat(-0.0f).isNegative());

  EXPECT_EQ(0.0,  APFloat(0.0).convertToDouble());
  EXPECT_EQ(-0.0, APFloat(-0.0).convertToDouble());
  EXPECT_TRUE(APFloat(-0.0).isNegative());

  EXPECT_EQ(fcPosZero, APFloat(0.0).classify());
  EXPECT_EQ(fcNegZero, APFloat(-0.0).classify());
}

TEST(APFloatTest, DecimalStringsWithoutNullTerminators) {
  // Make sure that we can parse strings without null terminators.
  // rdar://14323230.
  EXPECT_EQ(convertToDoubleFromString(StringRef("0.00", 3)), 0.0);
  EXPECT_EQ(convertToDoubleFromString(StringRef("0.01", 3)), 0.0);
  EXPECT_EQ(convertToDoubleFromString(StringRef("0.09", 3)), 0.0);
  EXPECT_EQ(convertToDoubleFromString(StringRef("0.095", 4)), 0.09);
  EXPECT_EQ(convertToDoubleFromString(StringRef("0.00e+3", 7)), 0.00);
  EXPECT_EQ(convertToDoubleFromString(StringRef("0e+3", 4)), 0.00);
}

TEST(APFloatTest, fromZeroDecimalString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0.").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0.").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0.").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  ".0").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+.0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-.0").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0.0").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0.0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0.0").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "00000.").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+00000.").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-00000.").convertToDouble());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble(), ".00000").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+.00000").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-.00000").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0000.00000").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0000.00000").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0000.00000").convertToDouble());
}

TEST(APFloatTest, fromZeroDecimalSingleExponentString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),   "0e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(),  "+0e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(),  "-0e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0e-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),   "0.e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(),  "+0.e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(),  "-0.e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0.e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0.e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0.e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0.e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0.e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0.e-1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),   ".0e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(),  "+.0e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(),  "-.0e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  ".0e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+.0e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-.0e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  ".0e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+.0e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-.0e-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),   "0.0e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(),  "+0.0e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(),  "-0.0e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0.0e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0.0e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0.0e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0.0e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0.0e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0.0e-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "000.0000e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+000.0000e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-000.0000e+1").convertToDouble());
}

TEST(APFloatTest, fromZeroDecimalLargeExponentString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0e1234").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0e1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0e1234").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0e+1234").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0e+1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0e+1234").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0e-1234").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0e-1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0e-1234").convertToDouble());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble(), "000.0000e1234").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble(), "000.0000e-1234").convertToDouble());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble(), StringRef("0e1234" "\0" "2", 6)).convertToDouble());
}

TEST(APFloatTest, fromZeroHexadecimalString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x0p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x0p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x0p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x0p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x0p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x0p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x0p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x0p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x0p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x0.p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x0.p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x0.p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x0.p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x0.p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x0.p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x0.p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x0.p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x0.p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x.0p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x.0p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x.0p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x.0p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x.0p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x.0p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x.0p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x.0p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x.0p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x0.0p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x0.0p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x0.0p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x0.0p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x0.0p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x0.0p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(),  "0x0.0p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble(), "+0x0.0p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x0.0p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(), "0x00000.p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(), "0x0000.00000p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(), "0x.00000p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(), "0x0.p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(), "0x0p1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble(), "-0x0p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(), "0x00000.p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(), "0x0000.00000p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(), "0x.00000p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble(), "0x0.p1234").convertToDouble());
}

TEST(APFloatTest, fromDecimalString) {
  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble(), "1").convertToDouble());
  EXPECT_EQ(2.0,      APFloat(APFloat::IEEEdouble(), "2.").convertToDouble());
  EXPECT_EQ(0.5,      APFloat(APFloat::IEEEdouble(), ".5").convertToDouble());
  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble(), "1.0").convertToDouble());
  EXPECT_EQ(-2.0,     APFloat(APFloat::IEEEdouble(), "-2").convertToDouble());
  EXPECT_EQ(-4.0,     APFloat(APFloat::IEEEdouble(), "-4.").convertToDouble());
  EXPECT_EQ(-0.5,     APFloat(APFloat::IEEEdouble(), "-.5").convertToDouble());
  EXPECT_EQ(-1.5,     APFloat(APFloat::IEEEdouble(), "-1.5").convertToDouble());
  EXPECT_EQ(1.25e12,  APFloat(APFloat::IEEEdouble(), "1.25e12").convertToDouble());
  EXPECT_EQ(1.25e+12, APFloat(APFloat::IEEEdouble(), "1.25e+12").convertToDouble());
  EXPECT_EQ(1.25e-12, APFloat(APFloat::IEEEdouble(), "1.25e-12").convertToDouble());
  EXPECT_EQ(1024.0,   APFloat(APFloat::IEEEdouble(), "1024.").convertToDouble());
  EXPECT_EQ(1024.05,  APFloat(APFloat::IEEEdouble(), "1024.05000").convertToDouble());
  EXPECT_EQ(0.05,     APFloat(APFloat::IEEEdouble(), ".05000").convertToDouble());
  EXPECT_EQ(2.0,      APFloat(APFloat::IEEEdouble(), "2.").convertToDouble());
  EXPECT_EQ(2.0e2,    APFloat(APFloat::IEEEdouble(), "2.e2").convertToDouble());
  EXPECT_EQ(2.0e+2,   APFloat(APFloat::IEEEdouble(), "2.e+2").convertToDouble());
  EXPECT_EQ(2.0e-2,   APFloat(APFloat::IEEEdouble(), "2.e-2").convertToDouble());
  EXPECT_EQ(2.05e2,    APFloat(APFloat::IEEEdouble(), "002.05000e2").convertToDouble());
  EXPECT_EQ(2.05e+2,   APFloat(APFloat::IEEEdouble(), "002.05000e+2").convertToDouble());
  EXPECT_EQ(2.05e-2,   APFloat(APFloat::IEEEdouble(), "002.05000e-2").convertToDouble());
  EXPECT_EQ(2.05e12,   APFloat(APFloat::IEEEdouble(), "002.05000e12").convertToDouble());
  EXPECT_EQ(2.05e+12,  APFloat(APFloat::IEEEdouble(), "002.05000e+12").convertToDouble());
  EXPECT_EQ(2.05e-12,  APFloat(APFloat::IEEEdouble(), "002.05000e-12").convertToDouble());

  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble(), "1e").convertToDouble());
  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble(), "+1e").convertToDouble());
  EXPECT_EQ(-1.0,      APFloat(APFloat::IEEEdouble(), "-1e").convertToDouble());

  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble(), "1.e").convertToDouble());
  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble(), "+1.e").convertToDouble());
  EXPECT_EQ(-1.0,      APFloat(APFloat::IEEEdouble(), "-1.e").convertToDouble());

  EXPECT_EQ(0.1,      APFloat(APFloat::IEEEdouble(), ".1e").convertToDouble());
  EXPECT_EQ(0.1,      APFloat(APFloat::IEEEdouble(), "+.1e").convertToDouble());
  EXPECT_EQ(-0.1,      APFloat(APFloat::IEEEdouble(), "-.1e").convertToDouble());

  EXPECT_EQ(1.1,      APFloat(APFloat::IEEEdouble(), "1.1e").convertToDouble());
  EXPECT_EQ(1.1,      APFloat(APFloat::IEEEdouble(), "+1.1e").convertToDouble());
  EXPECT_EQ(-1.1,      APFloat(APFloat::IEEEdouble(), "-1.1e").convertToDouble());

  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble(), "1e+").convertToDouble());
  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble(), "1e-").convertToDouble());

  EXPECT_EQ(0.1,      APFloat(APFloat::IEEEdouble(), ".1e").convertToDouble());
  EXPECT_EQ(0.1,      APFloat(APFloat::IEEEdouble(), ".1e+").convertToDouble());
  EXPECT_EQ(0.1,      APFloat(APFloat::IEEEdouble(), ".1e-").convertToDouble());

  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble(), "1.0e").convertToDouble());
  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble(), "1.0e+").convertToDouble());
  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble(), "1.0e-").convertToDouble());

  // These are "carefully selected" to overflow the fast log-base
  // calculations in APFloat.cpp
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "99e99999").isInfinity());
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "-99e99999").isInfinity());
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "1e-99999").isPosZero());
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "-1e-99999").isNegZero());

  EXPECT_EQ(2.71828, convertToDoubleFromString("2.71828"));
}

TEST(APFloatTest, fromStringSpecials) {
  const fltSemantics &Sem = APFloat::IEEEdouble();
  const unsigned Precision = 53;
  const unsigned PayloadBits = Precision - 2;
  uint64_t PayloadMask = (uint64_t(1) << PayloadBits) - uint64_t(1);

  uint64_t NaNPayloads[] = {
      0,
      1,
      123,
      0xDEADBEEF,
      uint64_t(-2),
      uint64_t(1) << PayloadBits,       // overflow bit
      uint64_t(1) << (PayloadBits - 1), // signaling bit
      uint64_t(1) << (PayloadBits - 2)  // highest possible bit
  };

  // Convert payload integer to decimal string representation.
  std::string NaNPayloadDecStrings[std::size(NaNPayloads)];
  for (size_t I = 0; I < std::size(NaNPayloads); ++I)
    NaNPayloadDecStrings[I] = utostr(NaNPayloads[I]);

  // Convert payload integer to hexadecimal string representation.
  std::string NaNPayloadHexStrings[std::size(NaNPayloads)];
  for (size_t I = 0; I < std::size(NaNPayloads); ++I)
    NaNPayloadHexStrings[I] = "0x" + utohexstr(NaNPayloads[I]);

  // Fix payloads to expected result.
  for (uint64_t &Payload : NaNPayloads)
    Payload &= PayloadMask;

  // Signaling NaN must have a non-zero payload. In case a zero payload is
  // requested, a default arbitrary payload is set instead. Save this payload
  // for testing.
  const uint64_t SNaNDefaultPayload =
      APFloat::getSNaN(Sem).bitcastToAPInt().getZExtValue() & PayloadMask;

  // Negative sign prefix (or none - for positive).
  const char Signs[] = {0, '-'};

  // "Signaling" prefix (or none - for "Quiet").
  const char NaNTypes[] = {0, 's', 'S'};

  const StringRef NaNStrings[] = {"nan", "NaN"};
  for (StringRef NaNStr : NaNStrings)
    for (char TypeChar : NaNTypes) {
      bool Signaling = (TypeChar == 's' || TypeChar == 'S');

      for (size_t J = 0; J < std::size(NaNPayloads); ++J) {
        uint64_t Payload = (Signaling && !NaNPayloads[J]) ? SNaNDefaultPayload
                                                          : NaNPayloads[J];
        std::string &PayloadDec = NaNPayloadDecStrings[J];
        std::string &PayloadHex = NaNPayloadHexStrings[J];

        for (char SignChar : Signs) {
          bool Negative = (SignChar == '-');

          std::string TestStrings[5];
          size_t NumTestStrings = 0;

          std::string Prefix;
          if (SignChar)
            Prefix += SignChar;
          if (TypeChar)
            Prefix += TypeChar;
          Prefix += NaNStr;

          // Test without any paylod.
          if (!Payload)
            TestStrings[NumTestStrings++] = Prefix;

          // Test with the payload as a suffix.
          TestStrings[NumTestStrings++] = Prefix + PayloadDec;
          TestStrings[NumTestStrings++] = Prefix + PayloadHex;

          // Test with the payload inside parentheses.
          TestStrings[NumTestStrings++] = Prefix + '(' + PayloadDec + ')';
          TestStrings[NumTestStrings++] = Prefix + '(' + PayloadHex + ')';

          for (size_t K = 0; K < NumTestStrings; ++K) {
            StringRef TestStr = TestStrings[K];

            APFloat F(Sem);
            bool HasError = !F.convertFromString(
                TestStr, llvm::APFloat::rmNearestTiesToEven);
            EXPECT_FALSE(HasError);
            EXPECT_TRUE(F.isNaN());
            EXPECT_EQ(Signaling, F.isSignaling());
            EXPECT_EQ(Negative, F.isNegative());
            uint64_t PayloadResult =
                F.bitcastToAPInt().getZExtValue() & PayloadMask;
            EXPECT_EQ(Payload, PayloadResult);
          }
        }
      }
    }

  const StringRef InfStrings[] = {"inf",  "INFINITY",  "+Inf",
                                  "-inf", "-INFINITY", "-Inf"};
  for (StringRef InfStr : InfStrings) {
    bool Negative = InfStr.front() == '-';

    APFloat F(Sem);
    bool HasError =
        !F.convertFromString(InfStr, llvm::APFloat::rmNearestTiesToEven);
    EXPECT_FALSE(HasError);
    EXPECT_TRUE(F.isInfinity());
    EXPECT_EQ(Negative, F.isNegative());
    uint64_t PayloadResult = F.bitcastToAPInt().getZExtValue() & PayloadMask;
    EXPECT_EQ(UINT64_C(0), PayloadResult);
  }
}

TEST(APFloatTest, fromToStringSpecials) {
  auto expects = [] (const char *first, const char *second) {
    std::string roundtrip = convertToString(convertToDoubleFromString(second), 0, 3);
    EXPECT_STREQ(first, roundtrip.c_str());
  };
  expects("+Inf", "+Inf");
  expects("+Inf", "INFINITY");
  expects("+Inf", "inf");
  expects("-Inf", "-Inf");
  expects("-Inf", "-INFINITY");
  expects("-Inf", "-inf");
  expects("NaN", "NaN");
  expects("NaN", "nan");
  expects("NaN", "-NaN");
  expects("NaN", "-nan");
}

TEST(APFloatTest, fromHexadecimalString) {
  EXPECT_EQ( 1.0, APFloat(APFloat::IEEEdouble(),  "0x1p0").convertToDouble());
  EXPECT_EQ(+1.0, APFloat(APFloat::IEEEdouble(), "+0x1p0").convertToDouble());
  EXPECT_EQ(-1.0, APFloat(APFloat::IEEEdouble(), "-0x1p0").convertToDouble());

  EXPECT_EQ( 1.0, APFloat(APFloat::IEEEdouble(),  "0x1p+0").convertToDouble());
  EXPECT_EQ(+1.0, APFloat(APFloat::IEEEdouble(), "+0x1p+0").convertToDouble());
  EXPECT_EQ(-1.0, APFloat(APFloat::IEEEdouble(), "-0x1p+0").convertToDouble());

  EXPECT_EQ( 1.0, APFloat(APFloat::IEEEdouble(),  "0x1p-0").convertToDouble());
  EXPECT_EQ(+1.0, APFloat(APFloat::IEEEdouble(), "+0x1p-0").convertToDouble());
  EXPECT_EQ(-1.0, APFloat(APFloat::IEEEdouble(), "-0x1p-0").convertToDouble());


  EXPECT_EQ( 2.0, APFloat(APFloat::IEEEdouble(),  "0x1p1").convertToDouble());
  EXPECT_EQ(+2.0, APFloat(APFloat::IEEEdouble(), "+0x1p1").convertToDouble());
  EXPECT_EQ(-2.0, APFloat(APFloat::IEEEdouble(), "-0x1p1").convertToDouble());

  EXPECT_EQ( 2.0, APFloat(APFloat::IEEEdouble(),  "0x1p+1").convertToDouble());
  EXPECT_EQ(+2.0, APFloat(APFloat::IEEEdouble(), "+0x1p+1").convertToDouble());
  EXPECT_EQ(-2.0, APFloat(APFloat::IEEEdouble(), "-0x1p+1").convertToDouble());

  EXPECT_EQ( 0.5, APFloat(APFloat::IEEEdouble(),  "0x1p-1").convertToDouble());
  EXPECT_EQ(+0.5, APFloat(APFloat::IEEEdouble(), "+0x1p-1").convertToDouble());
  EXPECT_EQ(-0.5, APFloat(APFloat::IEEEdouble(), "-0x1p-1").convertToDouble());


  EXPECT_EQ( 3.0, APFloat(APFloat::IEEEdouble(),  "0x1.8p1").convertToDouble());
  EXPECT_EQ(+3.0, APFloat(APFloat::IEEEdouble(), "+0x1.8p1").convertToDouble());
  EXPECT_EQ(-3.0, APFloat(APFloat::IEEEdouble(), "-0x1.8p1").convertToDouble());

  EXPECT_EQ( 3.0, APFloat(APFloat::IEEEdouble(),  "0x1.8p+1").convertToDouble());
  EXPECT_EQ(+3.0, APFloat(APFloat::IEEEdouble(), "+0x1.8p+1").convertToDouble());
  EXPECT_EQ(-3.0, APFloat(APFloat::IEEEdouble(), "-0x1.8p+1").convertToDouble());

  EXPECT_EQ( 0.75, APFloat(APFloat::IEEEdouble(),  "0x1.8p-1").convertToDouble());
  EXPECT_EQ(+0.75, APFloat(APFloat::IEEEdouble(), "+0x1.8p-1").convertToDouble());
  EXPECT_EQ(-0.75, APFloat(APFloat::IEEEdouble(), "-0x1.8p-1").convertToDouble());


  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble(),  "0x1000.000p1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble(), "+0x1000.000p1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble(), "-0x1000.000p1").convertToDouble());

  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble(),  "0x1000.000p+1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble(), "+0x1000.000p+1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble(), "-0x1000.000p+1").convertToDouble());

  EXPECT_EQ( 2048.0, APFloat(APFloat::IEEEdouble(),  "0x1000.000p-1").convertToDouble());
  EXPECT_EQ(+2048.0, APFloat(APFloat::IEEEdouble(), "+0x1000.000p-1").convertToDouble());
  EXPECT_EQ(-2048.0, APFloat(APFloat::IEEEdouble(), "-0x1000.000p-1").convertToDouble());


  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble(),  "0x1000p1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble(), "+0x1000p1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble(), "-0x1000p1").convertToDouble());

  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble(),  "0x1000p+1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble(), "+0x1000p+1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble(), "-0x1000p+1").convertToDouble());

  EXPECT_EQ( 2048.0, APFloat(APFloat::IEEEdouble(),  "0x1000p-1").convertToDouble());
  EXPECT_EQ(+2048.0, APFloat(APFloat::IEEEdouble(), "+0x1000p-1").convertToDouble());
  EXPECT_EQ(-2048.0, APFloat(APFloat::IEEEdouble(), "-0x1000p-1").convertToDouble());


  EXPECT_EQ( 16384.0, APFloat(APFloat::IEEEdouble(),  "0x10p10").convertToDouble());
  EXPECT_EQ(+16384.0, APFloat(APFloat::IEEEdouble(), "+0x10p10").convertToDouble());
  EXPECT_EQ(-16384.0, APFloat(APFloat::IEEEdouble(), "-0x10p10").convertToDouble());

  EXPECT_EQ( 16384.0, APFloat(APFloat::IEEEdouble(),  "0x10p+10").convertToDouble());
  EXPECT_EQ(+16384.0, APFloat(APFloat::IEEEdouble(), "+0x10p+10").convertToDouble());
  EXPECT_EQ(-16384.0, APFloat(APFloat::IEEEdouble(), "-0x10p+10").convertToDouble());

  EXPECT_EQ( 0.015625, APFloat(APFloat::IEEEdouble(),  "0x10p-10").convertToDouble());
  EXPECT_EQ(+0.015625, APFloat(APFloat::IEEEdouble(), "+0x10p-10").convertToDouble());
  EXPECT_EQ(-0.015625, APFloat(APFloat::IEEEdouble(), "-0x10p-10").convertToDouble());

  EXPECT_EQ(1.0625, APFloat(APFloat::IEEEdouble(), "0x1.1p0").convertToDouble());
  EXPECT_EQ(1.0, APFloat(APFloat::IEEEdouble(), "0x1p0").convertToDouble());

  EXPECT_EQ(convertToDoubleFromString("0x1p-150"),
            convertToDoubleFromString("+0x800000000000000001.p-221"));
  EXPECT_EQ(2251799813685248.5,
            convertToDoubleFromString("0x80000000000004000000.010p-28"));
}

TEST(APFloatTest, toString) {
  ASSERT_EQ("10", convertToString(10.0, 6, 3));
  ASSERT_EQ("1.0E+1", convertToString(10.0, 6, 0));
  ASSERT_EQ("10100", convertToString(1.01E+4, 5, 2));
  ASSERT_EQ("1.01E+4", convertToString(1.01E+4, 4, 2));
  ASSERT_EQ("1.01E+4", convertToString(1.01E+4, 5, 1));
  ASSERT_EQ("0.0101", convertToString(1.01E-2, 5, 2));
  ASSERT_EQ("0.0101", convertToString(1.01E-2, 4, 2));
  ASSERT_EQ("1.01E-2", convertToString(1.01E-2, 5, 1));
  ASSERT_EQ("0.78539816339744828", convertToString(0.78539816339744830961, 0, 3));
  ASSERT_EQ("4.9406564584124654E-324", convertToString(4.9406564584124654e-324, 0, 3));
  ASSERT_EQ("873.18340000000001", convertToString(873.1834, 0, 1));
  ASSERT_EQ("8.7318340000000001E+2", convertToString(873.1834, 0, 0));
  ASSERT_EQ("1.7976931348623157E+308", convertToString(1.7976931348623157E+308, 0, 0));
  ASSERT_EQ("10", convertToString(10.0, 6, 3, false));
  ASSERT_EQ("1.000000e+01", convertToString(10.0, 6, 0, false));
  ASSERT_EQ("10100", convertToString(1.01E+4, 5, 2, false));
  ASSERT_EQ("1.0100e+04", convertToString(1.01E+4, 4, 2, false));
  ASSERT_EQ("1.01000e+04", convertToString(1.01E+4, 5, 1, false));
  ASSERT_EQ("0.0101", convertToString(1.01E-2, 5, 2, false));
  ASSERT_EQ("0.0101", convertToString(1.01E-2, 4, 2, false));
  ASSERT_EQ("1.01000e-02", convertToString(1.01E-2, 5, 1, false));
  ASSERT_EQ("0.78539816339744828",
            convertToString(0.78539816339744830961, 0, 3, false));
  ASSERT_EQ("4.94065645841246540e-324",
            convertToString(4.9406564584124654e-324, 0, 3, false));
  ASSERT_EQ("873.18340000000001", convertToString(873.1834, 0, 1, false));
  ASSERT_EQ("8.73183400000000010e+02", convertToString(873.1834, 0, 0, false));
  ASSERT_EQ("1.79769313486231570e+308",
            convertToString(1.7976931348623157E+308, 0, 0, false));

  {
    SmallString<64> Str;
    APFloat UnnormalZero(APFloat::x87DoubleExtended(), APInt(80, {0, 1}));
    UnnormalZero.toString(Str);
    ASSERT_EQ("NaN", Str);
  }
}

TEST(APFloatTest, toInteger) {
  bool isExact = false;
  APSInt result(5, /*isUnsigned=*/true);

  EXPECT_EQ(APFloat::opOK,
            APFloat(APFloat::IEEEdouble(), "10")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_TRUE(isExact);
  EXPECT_EQ(APSInt(APInt(5, 10), true), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble(), "-10")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMinValue(5, true), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble(), "32")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMaxValue(5, true), result);

  EXPECT_EQ(APFloat::opInexact,
            APFloat(APFloat::IEEEdouble(), "7.9")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt(APInt(5, 7), true), result);

  result.setIsUnsigned(false);
  EXPECT_EQ(APFloat::opOK,
            APFloat(APFloat::IEEEdouble(), "-10")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_TRUE(isExact);
  EXPECT_EQ(APSInt(APInt(5, -10, true), false), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble(), "-17")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMinValue(5, false), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble(), "16")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMaxValue(5, false), result);
}

static APInt nanbitsFromAPInt(const fltSemantics &Sem, bool SNaN, bool Negative,
                              uint64_t payload) {
  APInt appayload(64, payload);
  if (SNaN)
    return APFloat::getSNaN(Sem, Negative, &appayload).bitcastToAPInt();
  else
    return APFloat::getQNaN(Sem, Negative, &appayload).bitcastToAPInt();
}

TEST(APFloatTest, makeNaN) {
  const struct {
    uint64_t expected;
    const fltSemantics &semantics;
    bool SNaN;
    bool Negative;
    uint64_t payload;
  } tests[] = {
      // clang-format off
    /*             expected              semantics   SNaN    Neg                payload */
    {         0x7fc00000ULL, APFloat::IEEEsingle(), false, false,         0x00000000ULL },
    {         0xffc00000ULL, APFloat::IEEEsingle(), false,  true,         0x00000000ULL },
    {         0x7fc0ae72ULL, APFloat::IEEEsingle(), false, false,         0x0000ae72ULL },
    {         0x7fffae72ULL, APFloat::IEEEsingle(), false, false,         0xffffae72ULL },
    {         0x7fdaae72ULL, APFloat::IEEEsingle(), false, false,         0x00daae72ULL },
    {         0x7fa00000ULL, APFloat::IEEEsingle(),  true, false,         0x00000000ULL },
    {         0xffa00000ULL, APFloat::IEEEsingle(),  true,  true,         0x00000000ULL },
    {         0x7f80ae72ULL, APFloat::IEEEsingle(),  true, false,         0x0000ae72ULL },
    {         0x7fbfae72ULL, APFloat::IEEEsingle(),  true, false,         0xffffae72ULL },
    {         0x7f9aae72ULL, APFloat::IEEEsingle(),  true, false,         0x001aae72ULL },
    { 0x7ff8000000000000ULL, APFloat::IEEEdouble(), false, false, 0x0000000000000000ULL },
    { 0xfff8000000000000ULL, APFloat::IEEEdouble(), false,  true, 0x0000000000000000ULL },
    { 0x7ff800000000ae72ULL, APFloat::IEEEdouble(), false, false, 0x000000000000ae72ULL },
    { 0x7fffffffffffae72ULL, APFloat::IEEEdouble(), false, false, 0xffffffffffffae72ULL },
    { 0x7ffdaaaaaaaaae72ULL, APFloat::IEEEdouble(), false, false, 0x000daaaaaaaaae72ULL },
    { 0x7ff4000000000000ULL, APFloat::IEEEdouble(),  true, false, 0x0000000000000000ULL },
    { 0xfff4000000000000ULL, APFloat::IEEEdouble(),  true,  true, 0x0000000000000000ULL },
    { 0x7ff000000000ae72ULL, APFloat::IEEEdouble(),  true, false, 0x000000000000ae72ULL },
    { 0x7ff7ffffffffae72ULL, APFloat::IEEEdouble(),  true, false, 0xffffffffffffae72ULL },
    { 0x7ff1aaaaaaaaae72ULL, APFloat::IEEEdouble(),  true, false, 0x0001aaaaaaaaae72ULL },
    {               0x80ULL, APFloat::Float8E5M2FNUZ(), false, false,           0xaaULL },
    {               0x80ULL, APFloat::Float8E5M2FNUZ(), false, true,            0xaaULL },
    {               0x80ULL, APFloat::Float8E5M2FNUZ(), true, false,            0xaaULL },
    {               0x80ULL, APFloat::Float8E5M2FNUZ(), true, true,             0xaaULL },
    {               0x80ULL, APFloat::Float8E4M3FNUZ(), false, false,           0xaaULL },
    {               0x80ULL, APFloat::Float8E4M3FNUZ(), false, true,            0xaaULL },
    {               0x80ULL, APFloat::Float8E4M3FNUZ(), true, false,            0xaaULL },
    {               0x80ULL, APFloat::Float8E4M3FNUZ(), true, true,             0xaaULL },
    {               0x80ULL, APFloat::Float8E4M3B11FNUZ(), false, false,        0xaaULL },
    {               0x80ULL, APFloat::Float8E4M3B11FNUZ(), false, true,         0xaaULL },
    {               0x80ULL, APFloat::Float8E4M3B11FNUZ(), true, false,         0xaaULL },
    {               0x80ULL, APFloat::Float8E4M3B11FNUZ(), true, true,          0xaaULL },
    {            0x3fe00ULL, APFloat::FloatTF32(), false, false,          0x00000000ULL },
    {            0x7fe00ULL, APFloat::FloatTF32(), false,  true,          0x00000000ULL },
    {            0x3feaaULL, APFloat::FloatTF32(), false, false,                0xaaULL },
    {            0x3ffaaULL, APFloat::FloatTF32(), false, false,               0xdaaULL },
    {            0x3ffaaULL, APFloat::FloatTF32(), false, false,              0xfdaaULL },
    {            0x3fd00ULL, APFloat::FloatTF32(),  true, false,          0x00000000ULL },
    {            0x7fd00ULL, APFloat::FloatTF32(),  true,  true,          0x00000000ULL },
    {            0x3fcaaULL, APFloat::FloatTF32(),  true, false,                0xaaULL },
    {            0x3fdaaULL, APFloat::FloatTF32(),  true, false,               0xfaaULL },
    {            0x3fdaaULL, APFloat::FloatTF32(),  true, false,               0x1aaULL },
      // clang-format on
  };

  for (const auto &t : tests) {
    ASSERT_EQ(t.expected, nanbitsFromAPInt(t.semantics, t.SNaN, t.Negative, t.payload));
  }
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(APFloatTest, SemanticsDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEquad(), 0).convertToDouble(),
               "Float semantics is not representable by IEEEdouble");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble(), 0).convertToFloat(),
               "Float semantics is not representable by IEEEsingle");
}
#endif
#endif

TEST(APFloatTest, StringDecimalError) {
  EXPECT_EQ("Invalid string length", convertToErrorFromString(""));
  EXPECT_EQ("String has no digits", convertToErrorFromString("+"));
  EXPECT_EQ("String has no digits", convertToErrorFromString("-"));

  EXPECT_EQ("Invalid character in significand", convertToErrorFromString(StringRef("\0", 1)));
  EXPECT_EQ("Invalid character in significand", convertToErrorFromString(StringRef("1\0", 2)));
  EXPECT_EQ("Invalid character in significand", convertToErrorFromString(StringRef("1" "\0" "2", 3)));
  EXPECT_EQ("Invalid character in significand", convertToErrorFromString(StringRef("1" "\0" "2e1", 5)));
  EXPECT_EQ("Invalid character in exponent", convertToErrorFromString(StringRef("1e\0", 3)));
  EXPECT_EQ("Invalid character in exponent", convertToErrorFromString(StringRef("1e1\0", 4)));
  EXPECT_EQ("Invalid character in exponent", convertToErrorFromString(StringRef("1e1" "\0" "2", 5)));

  EXPECT_EQ("Invalid character in significand", convertToErrorFromString("1.0f"));

  EXPECT_EQ("String contains multiple dots", convertToErrorFromString(".."));
  EXPECT_EQ("String contains multiple dots", convertToErrorFromString("..0"));
  EXPECT_EQ("String contains multiple dots", convertToErrorFromString("1.0.0"));
}

TEST(APFloatTest, StringDecimalSignificandError) {
  EXPECT_EQ("Significand has no digits", convertToErrorFromString( "."));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+."));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-."));


  EXPECT_EQ("Significand has no digits", convertToErrorFromString( "e"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+e"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-e"));

  EXPECT_EQ("Significand has no digits", convertToErrorFromString( "e1"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+e1"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-e1"));

  EXPECT_EQ("Significand has no digits", convertToErrorFromString( ".e1"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+.e1"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-.e1"));


  EXPECT_EQ("Significand has no digits", convertToErrorFromString( ".e"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+.e"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-.e"));
}

TEST(APFloatTest, StringHexadecimalError) {
  EXPECT_EQ("Invalid string", convertToErrorFromString( "0x"));
  EXPECT_EQ("Invalid string", convertToErrorFromString("+0x"));
  EXPECT_EQ("Invalid string", convertToErrorFromString("-0x"));

  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString( "0x0"));
  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString("+0x0"));
  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString("-0x0"));

  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString( "0x0."));
  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString("+0x0."));
  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString("-0x0."));

  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString( "0x.0"));
  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString("+0x.0"));
  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString("-0x.0"));

  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString( "0x0.0"));
  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString("+0x0.0"));
  EXPECT_EQ("Hex strings require an exponent", convertToErrorFromString("-0x0.0"));

  EXPECT_EQ("Invalid character in significand", convertToErrorFromString(StringRef("0x\0", 3)));
  EXPECT_EQ("Invalid character in significand", convertToErrorFromString(StringRef("0x1\0", 4)));
  EXPECT_EQ("Invalid character in significand", convertToErrorFromString(StringRef("0x1" "\0" "2", 5)));
  EXPECT_EQ("Invalid character in significand", convertToErrorFromString(StringRef("0x1" "\0" "2p1", 7)));
  EXPECT_EQ("Invalid character in exponent", convertToErrorFromString(StringRef("0x1p\0", 5)));
  EXPECT_EQ("Invalid character in exponent", convertToErrorFromString(StringRef("0x1p1\0", 6)));
  EXPECT_EQ("Invalid character in exponent", convertToErrorFromString(StringRef("0x1p1" "\0" "2", 7)));

  EXPECT_EQ("Invalid character in exponent", convertToErrorFromString("0x1p0f"));

  EXPECT_EQ("String contains multiple dots", convertToErrorFromString("0x..p1"));
  EXPECT_EQ("String contains multiple dots", convertToErrorFromString("0x..0p1"));
  EXPECT_EQ("String contains multiple dots", convertToErrorFromString("0x1.0.0p1"));
}

TEST(APFloatTest, StringHexadecimalSignificandError) {
  EXPECT_EQ("Significand has no digits", convertToErrorFromString( "0x."));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+0x."));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-0x."));

  EXPECT_EQ("Significand has no digits", convertToErrorFromString( "0xp"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+0xp"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-0xp"));

  EXPECT_EQ("Significand has no digits", convertToErrorFromString( "0xp+"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+0xp+"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-0xp+"));

  EXPECT_EQ("Significand has no digits", convertToErrorFromString( "0xp-"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+0xp-"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-0xp-"));


  EXPECT_EQ("Significand has no digits", convertToErrorFromString( "0x.p"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+0x.p"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-0x.p"));

  EXPECT_EQ("Significand has no digits", convertToErrorFromString( "0x.p+"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+0x.p+"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-0x.p+"));

  EXPECT_EQ("Significand has no digits", convertToErrorFromString( "0x.p-"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("+0x.p-"));
  EXPECT_EQ("Significand has no digits", convertToErrorFromString("-0x.p-"));
}

TEST(APFloatTest, StringHexadecimalExponentError) {
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x1p"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x1p"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x1p"));

  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x1p+"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x1p+"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x1p+"));

  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x1p-"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x1p-"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x1p-"));


  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x1.p"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x1.p"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x1.p"));

  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x1.p+"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x1.p+"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x1.p+"));

  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x1.p-"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x1.p-"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x1.p-"));


  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x.1p"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x.1p"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x.1p"));

  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x.1p+"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x.1p+"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x.1p+"));

  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x.1p-"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x.1p-"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x.1p-"));


  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x1.1p"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x1.1p"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x1.1p"));

  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x1.1p+"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x1.1p+"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x1.1p+"));

  EXPECT_EQ("Exponent has no digits", convertToErrorFromString( "0x1.1p-"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("+0x1.1p-"));
  EXPECT_EQ("Exponent has no digits", convertToErrorFromString("-0x1.1p-"));
}

TEST(APFloatTest, exactInverse) {
  APFloat inv(0.0f);

  // Trivial operation.
  EXPECT_TRUE(APFloat(2.0).getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(0.5)));
  EXPECT_TRUE(APFloat(2.0f).getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(0.5f)));
  EXPECT_TRUE(APFloat(APFloat::IEEEquad(), "2.0").getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(APFloat::IEEEquad(), "0.5")));
  EXPECT_TRUE(APFloat(APFloat::PPCDoubleDouble(), "2.0").getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(APFloat::PPCDoubleDouble(), "0.5")));
  EXPECT_TRUE(APFloat(APFloat::x87DoubleExtended(), "2.0").getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(APFloat::x87DoubleExtended(), "0.5")));

  // FLT_MIN
  EXPECT_TRUE(APFloat(1.17549435e-38f).getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(8.5070592e+37f)));

  // Large float, inverse is a denormal.
  EXPECT_FALSE(APFloat(1.7014118e38f).getExactInverse(nullptr));
  // Zero
  EXPECT_FALSE(APFloat(0.0).getExactInverse(nullptr));
  // Denormalized float
  EXPECT_FALSE(APFloat(1.40129846e-45f).getExactInverse(nullptr));
}

TEST(APFloatTest, roundToIntegral) {
  APFloat T(-0.5), S(3.14), R(APFloat::getLargest(APFloat::IEEEdouble())), P(0.0);

  P = T;
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(-0.0, P.convertToDouble());
  P = T;
  P.roundToIntegral(APFloat::rmTowardNegative);
  EXPECT_EQ(-1.0, P.convertToDouble());
  P = T;
  P.roundToIntegral(APFloat::rmTowardPositive);
  EXPECT_EQ(-0.0, P.convertToDouble());
  P = T;
  P.roundToIntegral(APFloat::rmNearestTiesToEven);
  EXPECT_EQ(-0.0, P.convertToDouble());

  P = S;
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(3.0, P.convertToDouble());
  P = S;
  P.roundToIntegral(APFloat::rmTowardNegative);
  EXPECT_EQ(3.0, P.convertToDouble());
  P = S;
  P.roundToIntegral(APFloat::rmTowardPositive);
  EXPECT_EQ(4.0, P.convertToDouble());
  P = S;
  P.roundToIntegral(APFloat::rmNearestTiesToEven);
  EXPECT_EQ(3.0, P.convertToDouble());

  P = R;
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(R.convertToDouble(), P.convertToDouble());
  P = R;
  P.roundToIntegral(APFloat::rmTowardNegative);
  EXPECT_EQ(R.convertToDouble(), P.convertToDouble());
  P = R;
  P.roundToIntegral(APFloat::rmTowardPositive);
  EXPECT_EQ(R.convertToDouble(), P.convertToDouble());
  P = R;
  P.roundToIntegral(APFloat::rmNearestTiesToEven);
  EXPECT_EQ(R.convertToDouble(), P.convertToDouble());

  P = APFloat::getZero(APFloat::IEEEdouble());
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(0.0, P.convertToDouble());
  P = APFloat::getZero(APFloat::IEEEdouble(), true);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(-0.0, P.convertToDouble());
  P = APFloat::getNaN(APFloat::IEEEdouble());
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(std::isnan(P.convertToDouble()));
  P = APFloat::getInf(APFloat::IEEEdouble());
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(std::isinf(P.convertToDouble()) && P.convertToDouble() > 0.0);
  P = APFloat::getInf(APFloat::IEEEdouble(), true);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(std::isinf(P.convertToDouble()) && P.convertToDouble() < 0.0);

  APFloat::opStatus St;

  P = APFloat::getNaN(APFloat::IEEEdouble());
  St = P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(P.isNaN());
  EXPECT_FALSE(P.isNegative());
  EXPECT_EQ(APFloat::opOK, St);

  P = APFloat::getNaN(APFloat::IEEEdouble(), true);
  St = P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(P.isNaN());
  EXPECT_TRUE(P.isNegative());
  EXPECT_EQ(APFloat::opOK, St);

  P = APFloat::getSNaN(APFloat::IEEEdouble());
  St = P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(P.isNaN());
  EXPECT_FALSE(P.isSignaling());
  EXPECT_FALSE(P.isNegative());
  EXPECT_EQ(APFloat::opInvalidOp, St);

  P = APFloat::getSNaN(APFloat::IEEEdouble(), true);
  St = P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(P.isNaN());
  EXPECT_FALSE(P.isSignaling());
  EXPECT_TRUE(P.isNegative());
  EXPECT_EQ(APFloat::opInvalidOp, St);

  P = APFloat::getInf(APFloat::IEEEdouble());
  St = P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(P.isInfinity());
  EXPECT_FALSE(P.isNegative());
  EXPECT_EQ(APFloat::opOK, St);

  P = APFloat::getInf(APFloat::IEEEdouble(), true);
  St = P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(P.isInfinity());
  EXPECT_TRUE(P.isNegative());
  EXPECT_EQ(APFloat::opOK, St);

  P = APFloat::getZero(APFloat::IEEEdouble(), false);
  St = P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(P.isZero());
  EXPECT_FALSE(P.isNegative());
  EXPECT_EQ(APFloat::opOK, St);

  P = APFloat::getZero(APFloat::IEEEdouble(), false);
  St = P.roundToIntegral(APFloat::rmTowardNegative);
  EXPECT_TRUE(P.isZero());
  EXPECT_FALSE(P.isNegative());
  EXPECT_EQ(APFloat::opOK, St);

  P = APFloat::getZero(APFloat::IEEEdouble(), true);
  St = P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(P.isZero());
  EXPECT_TRUE(P.isNegative());
  EXPECT_EQ(APFloat::opOK, St);

  P = APFloat::getZero(APFloat::IEEEdouble(), true);
  St = P.roundToIntegral(APFloat::rmTowardNegative);
  EXPECT_TRUE(P.isZero());
  EXPECT_TRUE(P.isNegative());
  EXPECT_EQ(APFloat::opOK, St);

  P = APFloat(1E-100);
  St = P.roundToIntegral(APFloat::rmTowardNegative);
  EXPECT_TRUE(P.isZero());
  EXPECT_FALSE(P.isNegative());
  EXPECT_EQ(APFloat::opInexact, St);

  P = APFloat(1E-100);
  St = P.roundToIntegral(APFloat::rmTowardPositive);
  EXPECT_EQ(1.0, P.convertToDouble());
  EXPECT_FALSE(P.isNegative());
  EXPECT_EQ(APFloat::opInexact, St);

  P = APFloat(-1E-100);
  St = P.roundToIntegral(APFloat::rmTowardNegative);
  EXPECT_TRUE(P.isNegative());
  EXPECT_EQ(-1.0, P.convertToDouble());
  EXPECT_EQ(APFloat::opInexact, St);

  P = APFloat(-1E-100);
  St = P.roundToIntegral(APFloat::rmTowardPositive);
  EXPECT_TRUE(P.isZero());
  EXPECT_TRUE(P.isNegative());
  EXPECT_EQ(APFloat::opInexact, St);

  P = APFloat(10.0);
  St = P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(10.0, P.convertToDouble());
  EXPECT_EQ(APFloat::opOK, St);

  P = APFloat(10.5);
  St = P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(10.0, P.convertToDouble());
  EXPECT_EQ(APFloat::opInexact, St);

  P = APFloat(10.5);
  St = P.roundToIntegral(APFloat::rmTowardPositive);
  EXPECT_EQ(11.0, P.convertToDouble());
  EXPECT_EQ(APFloat::opInexact, St);

  P = APFloat(10.5);
  St = P.roundToIntegral(APFloat::rmTowardNegative);
  EXPECT_EQ(10.0, P.convertToDouble());
  EXPECT_EQ(APFloat::opInexact, St);

  P = APFloat(10.5);
  St = P.roundToIntegral(APFloat::rmNearestTiesToAway);
  EXPECT_EQ(11.0, P.convertToDouble());
  EXPECT_EQ(APFloat::opInexact, St);

  P = APFloat(10.5);
  St = P.roundToIntegral(APFloat::rmNearestTiesToEven);
  EXPECT_EQ(10.0, P.convertToDouble());
  EXPECT_EQ(APFloat::opInexact, St);
}

TEST(APFloatTest, isInteger) {
  APFloat T(-0.0);
  EXPECT_TRUE(T.isInteger());
  T = APFloat(3.14159);
  EXPECT_FALSE(T.isInteger());
  T = APFloat::getNaN(APFloat::IEEEdouble());
  EXPECT_FALSE(T.isInteger());
  T = APFloat::getInf(APFloat::IEEEdouble());
  EXPECT_FALSE(T.isInteger());
  T = APFloat::getInf(APFloat::IEEEdouble(), true);
  EXPECT_FALSE(T.isInteger());
  T = APFloat::getLargest(APFloat::IEEEdouble());
  EXPECT_TRUE(T.isInteger());
}

TEST(DoubleAPFloatTest, isInteger) {
  APFloat F1(-0.0);
  APFloat F2(-0.0);
  llvm::detail::DoubleAPFloat T(APFloat::PPCDoubleDouble(), std::move(F1),
                                std::move(F2));
  EXPECT_TRUE(T.isInteger());
  APFloat F3(3.14159);
  APFloat F4(-0.0);
  llvm::detail::DoubleAPFloat T2(APFloat::PPCDoubleDouble(), std::move(F3),
                                std::move(F4));
  EXPECT_FALSE(T2.isInteger());
  APFloat F5(-0.0);
  APFloat F6(3.14159);
  llvm::detail::DoubleAPFloat T3(APFloat::PPCDoubleDouble(), std::move(F5),
                                std::move(F6));
  EXPECT_FALSE(T3.isInteger());
}

TEST(APFloatTest, getLargest) {
  EXPECT_EQ(3.402823466e+38f, APFloat::getLargest(APFloat::IEEEsingle()).convertToFloat());
  EXPECT_EQ(1.7976931348623158e+308, APFloat::getLargest(APFloat::IEEEdouble()).convertToDouble());
  EXPECT_EQ(448, APFloat::getLargest(APFloat::Float8E4M3FN()).convertToDouble());
  EXPECT_EQ(240,
            APFloat::getLargest(APFloat::Float8E4M3FNUZ()).convertToDouble());
  EXPECT_EQ(57344,
            APFloat::getLargest(APFloat::Float8E5M2FNUZ()).convertToDouble());
  EXPECT_EQ(
      30, APFloat::getLargest(APFloat::Float8E4M3B11FNUZ()).convertToDouble());
  EXPECT_EQ(3.40116213421e+38f,
            APFloat::getLargest(APFloat::FloatTF32()).convertToFloat());
}

TEST(APFloatTest, getSmallest) {
  APFloat test = APFloat::getSmallest(APFloat::IEEEsingle(), false);
  APFloat expected = APFloat(APFloat::IEEEsingle(), "0x0.000002p-126");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::IEEEsingle(), true);
  expected = APFloat(APFloat::IEEEsingle(), "-0x0.000002p-126");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::IEEEquad(), false);
  expected = APFloat(APFloat::IEEEquad(), "0x0.0000000000000000000000000001p-16382");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::IEEEquad(), true);
  expected = APFloat(APFloat::IEEEquad(), "-0x0.0000000000000000000000000001p-16382");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::Float8E5M2FNUZ(), false);
  expected = APFloat(APFloat::Float8E5M2FNUZ(), "0x0.4p-15");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::Float8E4M3FNUZ(), false);
  expected = APFloat(APFloat::Float8E4M3FNUZ(), "0x0.2p-7");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::Float8E4M3B11FNUZ(), false);
  expected = APFloat(APFloat::Float8E4M3B11FNUZ(), "0x0.2p-10");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::FloatTF32(), true);
  expected = APFloat(APFloat::FloatTF32(), "-0x0.004p-126");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
}

TEST(APFloatTest, getSmallestNormalized) {
  APFloat test = APFloat::getSmallestNormalized(APFloat::IEEEsingle(), false);
  APFloat expected = APFloat(APFloat::IEEEsingle(), "0x1p-126");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
  EXPECT_TRUE(test.isSmallestNormalized());

  test = APFloat::getSmallestNormalized(APFloat::IEEEsingle(), true);
  expected = APFloat(APFloat::IEEEsingle(), "-0x1p-126");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
  EXPECT_TRUE(test.isSmallestNormalized());

  test = APFloat::getSmallestNormalized(APFloat::IEEEdouble(), false);
  expected = APFloat(APFloat::IEEEdouble(), "0x1p-1022");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
  EXPECT_TRUE(test.isSmallestNormalized());

  test = APFloat::getSmallestNormalized(APFloat::IEEEdouble(), true);
  expected = APFloat(APFloat::IEEEdouble(), "-0x1p-1022");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
  EXPECT_TRUE(test.isSmallestNormalized());

  test = APFloat::getSmallestNormalized(APFloat::IEEEquad(), false);
  expected = APFloat(APFloat::IEEEquad(), "0x1p-16382");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
  EXPECT_TRUE(test.isSmallestNormalized());

  test = APFloat::getSmallestNormalized(APFloat::IEEEquad(), true);
  expected = APFloat(APFloat::IEEEquad(), "-0x1p-16382");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
  EXPECT_TRUE(test.isSmallestNormalized());

  test = APFloat::getSmallestNormalized(APFloat::Float8E5M2FNUZ(), false);
  expected = APFloat(APFloat::Float8E5M2FNUZ(), "0x1.0p-15");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
  EXPECT_TRUE(test.isSmallestNormalized());

  test = APFloat::getSmallestNormalized(APFloat::Float8E4M3FNUZ(), false);
  expected = APFloat(APFloat::Float8E4M3FNUZ(), "0x1.0p-7");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
  EXPECT_TRUE(test.isSmallestNormalized());

  test = APFloat::getSmallestNormalized(APFloat::Float8E4M3B11FNUZ(), false);
  expected = APFloat(APFloat::Float8E4M3B11FNUZ(), "0x1.0p-10");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
  EXPECT_TRUE(test.isSmallestNormalized());

  test = APFloat::getSmallestNormalized(APFloat::FloatTF32(), false);
  expected = APFloat(APFloat::FloatTF32(), "0x1p-126");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
  EXPECT_TRUE(test.isSmallestNormalized());
}

TEST(APFloatTest, getZero) {
  struct {
    const fltSemantics *semantics;
    const bool sign;
    const bool signedZero;
    const unsigned long long bitPattern[2];
    const unsigned bitPatternLength;
  } const GetZeroTest[] = {
      {&APFloat::IEEEhalf(), false, true, {0, 0}, 1},
      {&APFloat::IEEEhalf(), true, true, {0x8000ULL, 0}, 1},
      {&APFloat::IEEEsingle(), false, true, {0, 0}, 1},
      {&APFloat::IEEEsingle(), true, true, {0x80000000ULL, 0}, 1},
      {&APFloat::IEEEdouble(), false, true, {0, 0}, 1},
      {&APFloat::IEEEdouble(), true, true, {0x8000000000000000ULL, 0}, 1},
      {&APFloat::IEEEquad(), false, true, {0, 0}, 2},
      {&APFloat::IEEEquad(), true, true, {0, 0x8000000000000000ULL}, 2},
      {&APFloat::PPCDoubleDouble(), false, true, {0, 0}, 2},
      {&APFloat::PPCDoubleDouble(), true, true, {0x8000000000000000ULL, 0}, 2},
      {&APFloat::x87DoubleExtended(), false, true, {0, 0}, 2},
      {&APFloat::x87DoubleExtended(), true, true, {0, 0x8000ULL}, 2},
      {&APFloat::Float8E5M2(), false, true, {0, 0}, 1},
      {&APFloat::Float8E5M2(), true, true, {0x80ULL, 0}, 1},
      {&APFloat::Float8E5M2FNUZ(), false, false, {0, 0}, 1},
      {&APFloat::Float8E5M2FNUZ(), true, false, {0, 0}, 1},
      {&APFloat::Float8E4M3FN(), false, true, {0, 0}, 1},
      {&APFloat::Float8E4M3FN(), true, true, {0x80ULL, 0}, 1},
      {&APFloat::Float8E4M3FNUZ(), false, false, {0, 0}, 1},
      {&APFloat::Float8E4M3FNUZ(), true, false, {0, 0}, 1},
      {&APFloat::Float8E4M3B11FNUZ(), false, false, {0, 0}, 1},
      {&APFloat::Float8E4M3B11FNUZ(), true, false, {0, 0}, 1},
      {&APFloat::FloatTF32(), false, true, {0, 0}, 1},
      {&APFloat::FloatTF32(), true, true, {0x40000ULL, 0}, 1}};
  const unsigned NumGetZeroTests = std::size(GetZeroTest);
  for (unsigned i = 0; i < NumGetZeroTests; ++i) {
    APFloat test = APFloat::getZero(*GetZeroTest[i].semantics,
                                    GetZeroTest[i].sign);
    const char *pattern = GetZeroTest[i].sign? "-0x0p+0" : "0x0p+0";
    APFloat expected = APFloat(*GetZeroTest[i].semantics,
                               pattern);
    EXPECT_TRUE(test.isZero());
    if (GetZeroTest[i].signedZero)
      EXPECT_TRUE(GetZeroTest[i].sign ? test.isNegative() : !test.isNegative());
    else
      EXPECT_TRUE(!test.isNegative());
    EXPECT_TRUE(test.bitwiseIsEqual(expected));
    for (unsigned j = 0, je = GetZeroTest[i].bitPatternLength; j < je; ++j) {
      EXPECT_EQ(GetZeroTest[i].bitPattern[j],
                test.bitcastToAPInt().getRawData()[j]);
    }
  }
}

TEST(APFloatTest, copySign) {
  EXPECT_TRUE(APFloat(-42.0).bitwiseIsEqual(
      APFloat::copySign(APFloat(42.0), APFloat(-1.0))));
  EXPECT_TRUE(APFloat(42.0).bitwiseIsEqual(
      APFloat::copySign(APFloat(-42.0), APFloat(1.0))));
  EXPECT_TRUE(APFloat(-42.0).bitwiseIsEqual(
      APFloat::copySign(APFloat(-42.0), APFloat(-1.0))));
  EXPECT_TRUE(APFloat(42.0).bitwiseIsEqual(
      APFloat::copySign(APFloat(42.0), APFloat(1.0))));
  // For floating-point formats with unsigned 0, copySign() to a zero is a noop
  for (APFloat::Semantics S :
       {APFloat::S_Float8E4M3FNUZ, APFloat::S_Float8E4M3B11FNUZ}) {
    const llvm::fltSemantics &Sem = APFloat::EnumToSemantics(S);
    EXPECT_TRUE(APFloat::getZero(Sem).bitwiseIsEqual(
        APFloat::copySign(APFloat::getZero(Sem), APFloat(-1.0))));
    EXPECT_TRUE(APFloat::getNaN(Sem, true).bitwiseIsEqual(
        APFloat::copySign(APFloat::getNaN(Sem, true), APFloat(1.0))));
  }
}

TEST(APFloatTest, convert) {
  bool losesInfo;
  APFloat test(APFloat::IEEEdouble(), "1.0");
  test.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0f, test.convertToFloat());
  EXPECT_FALSE(losesInfo);

  test = APFloat(APFloat::x87DoubleExtended(), "0x1p-53");
  test.add(APFloat(APFloat::x87DoubleExtended(), "1.0"), APFloat::rmNearestTiesToEven);
  test.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0, test.convertToDouble());
  EXPECT_TRUE(losesInfo);

  test = APFloat(APFloat::IEEEquad(), "0x1p-53");
  test.add(APFloat(APFloat::IEEEquad(), "1.0"), APFloat::rmNearestTiesToEven);
  test.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0, test.convertToDouble());
  EXPECT_TRUE(losesInfo);

  test = APFloat(APFloat::x87DoubleExtended(), "0xf.fffffffp+28");
  test.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(4294967295.0, test.convertToDouble());
  EXPECT_FALSE(losesInfo);

  test = APFloat::getSNaN(APFloat::IEEEsingle());
  APFloat::opStatus status = test.convert(APFloat::x87DoubleExtended(), APFloat::rmNearestTiesToEven, &losesInfo);
  // Conversion quiets the SNAN, so now 2 bits of the 64-bit significand should be set.
  APInt topTwoBits(64, 0x6000000000000000);
  EXPECT_TRUE(test.bitwiseIsEqual(APFloat::getQNaN(APFloat::x87DoubleExtended(), false, &topTwoBits)));
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opInvalidOp);

  test = APFloat::getQNaN(APFloat::IEEEsingle());
  APFloat X87QNaN = APFloat::getQNaN(APFloat::x87DoubleExtended());
  test.convert(APFloat::x87DoubleExtended(), APFloat::rmNearestTiesToEven,
               &losesInfo);
  EXPECT_TRUE(test.bitwiseIsEqual(X87QNaN));
  EXPECT_FALSE(losesInfo);

  test = APFloat::getSNaN(APFloat::x87DoubleExtended());
  test.convert(APFloat::x87DoubleExtended(), APFloat::rmNearestTiesToEven,
               &losesInfo);
  APFloat X87SNaN = APFloat::getSNaN(APFloat::x87DoubleExtended());
  EXPECT_TRUE(test.bitwiseIsEqual(X87SNaN));
  EXPECT_FALSE(losesInfo);

  test = APFloat::getQNaN(APFloat::x87DoubleExtended());
  test.convert(APFloat::x87DoubleExtended(), APFloat::rmNearestTiesToEven,
               &losesInfo);
  EXPECT_TRUE(test.bitwiseIsEqual(X87QNaN));
  EXPECT_FALSE(losesInfo);

  // The payload is lost in truncation, but we retain NaN by setting the quiet bit.
  APInt payload(52, 1);
  test = APFloat::getSNaN(APFloat::IEEEdouble(), false, &payload);
  status = test.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(0x7fc00000, test.bitcastToAPInt());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opInvalidOp);

  // The payload is lost in truncation. QNaN remains QNaN.
  test = APFloat::getQNaN(APFloat::IEEEdouble(), false, &payload);
  status = test.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(0x7fc00000, test.bitcastToAPInt());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  // Test that subnormals are handled correctly in double to float conversion
  test = APFloat(APFloat::IEEEdouble(), "0x0.0000010000000p-1022");
  test.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(0.0f, test.convertToFloat());
  EXPECT_TRUE(losesInfo);

  test = APFloat(APFloat::IEEEdouble(), "0x0.0000010000001p-1022");
  test.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(0.0f, test.convertToFloat());
  EXPECT_TRUE(losesInfo);

  test = APFloat(APFloat::IEEEdouble(), "-0x0.0000010000001p-1022");
  test.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(0.0f, test.convertToFloat());
  EXPECT_TRUE(losesInfo);

  test = APFloat(APFloat::IEEEdouble(), "0x0.0000020000000p-1022");
  test.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(0.0f, test.convertToFloat());
  EXPECT_TRUE(losesInfo);

  test = APFloat(APFloat::IEEEdouble(), "0x0.0000020000001p-1022");
  test.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(0.0f, test.convertToFloat());
  EXPECT_TRUE(losesInfo);

  // Test subnormal conversion to bfloat
  test = APFloat(APFloat::IEEEsingle(), "0x0.01p-126");
  test.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(0.0f, test.convertToFloat());
  EXPECT_TRUE(losesInfo);

  test = APFloat(APFloat::IEEEsingle(), "0x0.02p-126");
  test.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(0x01, test.bitcastToAPInt());
  EXPECT_FALSE(losesInfo);

  test = APFloat(APFloat::IEEEsingle(), "0x0.01p-126");
  test.convert(APFloat::BFloat(), APFloat::rmNearestTiesToAway, &losesInfo);
  EXPECT_EQ(0x01, test.bitcastToAPInt());
  EXPECT_TRUE(losesInfo);
}

TEST(APFloatTest, Float8UZConvert) {
  bool losesInfo = false;
  std::pair<APFloat, APFloat::opStatus> toNaNTests[] = {
      {APFloat::getQNaN(APFloat::IEEEsingle(), false), APFloat::opOK},
      {APFloat::getQNaN(APFloat::IEEEsingle(), true), APFloat::opOK},
      {APFloat::getSNaN(APFloat::IEEEsingle(), false), APFloat::opInvalidOp},
      {APFloat::getSNaN(APFloat::IEEEsingle(), true), APFloat::opInvalidOp},
      {APFloat::getInf(APFloat::IEEEsingle(), false), APFloat::opInexact},
      {APFloat::getInf(APFloat::IEEEsingle(), true), APFloat::opInexact}};
  for (APFloat::Semantics S :
       {APFloat::S_Float8E5M2FNUZ, APFloat::S_Float8E4M3FNUZ,
        APFloat::S_Float8E4M3B11FNUZ}) {
    const llvm::fltSemantics &Sem = APFloat::EnumToSemantics(S);
    SCOPED_TRACE("Semantics = " + std::to_string(S));
    for (auto [toTest, expectedRes] : toNaNTests) {
      llvm::SmallString<16> value;
      toTest.toString(value);
      SCOPED_TRACE("toTest = " + value);
      losesInfo = false;
      APFloat test = toTest;
      EXPECT_EQ(test.convert(Sem, APFloat::rmNearestTiesToAway, &losesInfo),
                expectedRes);
      EXPECT_TRUE(test.isNaN());
      EXPECT_TRUE(test.isNegative());
      EXPECT_FALSE(test.isSignaling());
      EXPECT_FALSE(test.isInfinity());
      EXPECT_EQ(0x80, test.bitcastToAPInt());
      EXPECT_TRUE(losesInfo);
    }

    // Negative zero conversions are information losing.
    losesInfo = false;
    APFloat test = APFloat::getZero(APFloat::IEEEsingle(), true);
    EXPECT_EQ(test.convert(Sem, APFloat::rmNearestTiesToAway, &losesInfo),
              APFloat::opInexact);
    EXPECT_TRUE(test.isZero());
    EXPECT_FALSE(test.isNegative());
    EXPECT_TRUE(losesInfo);
    EXPECT_EQ(0x0, test.bitcastToAPInt());

    losesInfo = true;
    test = APFloat::getZero(APFloat::IEEEsingle(), false);
    EXPECT_EQ(test.convert(Sem, APFloat::rmNearestTiesToAway, &losesInfo),
              APFloat::opOK);
    EXPECT_TRUE(test.isZero());
    EXPECT_FALSE(test.isNegative());
    EXPECT_FALSE(losesInfo);
    EXPECT_EQ(0x0, test.bitcastToAPInt());

    // Except in casts between ourselves.
    losesInfo = true;
    test = APFloat::getZero(Sem);
    EXPECT_EQ(test.convert(Sem, APFloat::rmNearestTiesToAway, &losesInfo),
              APFloat::opOK);
    EXPECT_FALSE(losesInfo);
    EXPECT_EQ(0x0, test.bitcastToAPInt());
  }
}

TEST(APFloatTest, PPCDoubleDouble) {
  APFloat test(APFloat::PPCDoubleDouble(), "1.0");
  EXPECT_EQ(0x3ff0000000000000ull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x0000000000000000ull, test.bitcastToAPInt().getRawData()[1]);

  // LDBL_MAX
  test = APFloat(APFloat::PPCDoubleDouble(), "1.79769313486231580793728971405301e+308");
  EXPECT_EQ(0x7fefffffffffffffull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x7c8ffffffffffffeull, test.bitcastToAPInt().getRawData()[1]);

  // LDBL_MIN
  test = APFloat(APFloat::PPCDoubleDouble(), "2.00416836000897277799610805135016e-292");
  EXPECT_EQ(0x0360000000000000ull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x0000000000000000ull, test.bitcastToAPInt().getRawData()[1]);

  // PR30869
  {
    auto Result = APFloat(APFloat::PPCDoubleDouble(), "1.0") +
                  APFloat(APFloat::PPCDoubleDouble(), "1.0");
    EXPECT_EQ(&APFloat::PPCDoubleDouble(), &Result.getSemantics());

    Result = APFloat(APFloat::PPCDoubleDouble(), "1.0") -
             APFloat(APFloat::PPCDoubleDouble(), "1.0");
    EXPECT_EQ(&APFloat::PPCDoubleDouble(), &Result.getSemantics());

    Result = APFloat(APFloat::PPCDoubleDouble(), "1.0") *
             APFloat(APFloat::PPCDoubleDouble(), "1.0");
    EXPECT_EQ(&APFloat::PPCDoubleDouble(), &Result.getSemantics());

    Result = APFloat(APFloat::PPCDoubleDouble(), "1.0") /
             APFloat(APFloat::PPCDoubleDouble(), "1.0");
    EXPECT_EQ(&APFloat::PPCDoubleDouble(), &Result.getSemantics());

    int Exp;
    Result = frexp(APFloat(APFloat::PPCDoubleDouble(), "1.0"), Exp,
                   APFloat::rmNearestTiesToEven);
    EXPECT_EQ(&APFloat::PPCDoubleDouble(), &Result.getSemantics());

    Result = scalbn(APFloat(APFloat::PPCDoubleDouble(), "1.0"), 1,
                    APFloat::rmNearestTiesToEven);
    EXPECT_EQ(&APFloat::PPCDoubleDouble(), &Result.getSemantics());
  }
}

TEST(APFloatTest, isNegative) {
  APFloat t(APFloat::IEEEsingle(), "0x1p+0");
  EXPECT_FALSE(t.isNegative());
  t = APFloat(APFloat::IEEEsingle(), "-0x1p+0");
  EXPECT_TRUE(t.isNegative());

  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle(), false).isNegative());
  EXPECT_TRUE(APFloat::getInf(APFloat::IEEEsingle(), true).isNegative());

  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle(), false).isNegative());
  EXPECT_TRUE(APFloat::getZero(APFloat::IEEEsingle(), true).isNegative());

  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle(), false).isNegative());
  EXPECT_TRUE(APFloat::getNaN(APFloat::IEEEsingle(), true).isNegative());

  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle(), false).isNegative());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle(), true).isNegative());
}

TEST(APFloatTest, isNormal) {
  APFloat t(APFloat::IEEEsingle(), "0x1p+0");
  EXPECT_TRUE(t.isNormal());

  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle(), false).isNormal());
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle(), false).isNormal());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle(), false).isNormal());
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle(), false).isNormal());
  EXPECT_FALSE(APFloat(APFloat::IEEEsingle(), "0x1p-149").isNormal());
}

TEST(APFloatTest, isFinite) {
  APFloat t(APFloat::IEEEsingle(), "0x1p+0");
  EXPECT_TRUE(t.isFinite());
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle(), false).isFinite());
  EXPECT_TRUE(APFloat::getZero(APFloat::IEEEsingle(), false).isFinite());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle(), false).isFinite());
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle(), false).isFinite());
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle(), "0x1p-149").isFinite());
}

TEST(APFloatTest, isInfinity) {
  APFloat t(APFloat::IEEEsingle(), "0x1p+0");
  EXPECT_FALSE(t.isInfinity());

  APFloat PosInf = APFloat::getInf(APFloat::IEEEsingle(), false);
  APFloat NegInf = APFloat::getInf(APFloat::IEEEsingle(), true);

  EXPECT_TRUE(PosInf.isInfinity());
  EXPECT_TRUE(PosInf.isPosInfinity());
  EXPECT_FALSE(PosInf.isNegInfinity());
  EXPECT_EQ(fcPosInf, PosInf.classify());

  EXPECT_TRUE(NegInf.isInfinity());
  EXPECT_FALSE(NegInf.isPosInfinity());
  EXPECT_TRUE(NegInf.isNegInfinity());
  EXPECT_EQ(fcNegInf, NegInf.classify());

  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle(), false).isInfinity());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle(), false).isInfinity());
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle(), false).isInfinity());
  EXPECT_FALSE(APFloat(APFloat::IEEEsingle(), "0x1p-149").isInfinity());
}

TEST(APFloatTest, isNaN) {
  APFloat t(APFloat::IEEEsingle(), "0x1p+0");
  EXPECT_FALSE(t.isNaN());
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle(), false).isNaN());
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle(), false).isNaN());
  EXPECT_TRUE(APFloat::getNaN(APFloat::IEEEsingle(), false).isNaN());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle(), false).isNaN());
  EXPECT_FALSE(APFloat(APFloat::IEEEsingle(), "0x1p-149").isNaN());
}

TEST(APFloatTest, isFiniteNonZero) {
  // Test positive/negative normal value.
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle(), "0x1p+0").isFiniteNonZero());
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle(), "-0x1p+0").isFiniteNonZero());

  // Test positive/negative denormal value.
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle(), "0x1p-149").isFiniteNonZero());
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle(), "-0x1p-149").isFiniteNonZero());

  // Test +/- Infinity.
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle(), false).isFiniteNonZero());
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle(), true).isFiniteNonZero());

  // Test +/- Zero.
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle(), false).isFiniteNonZero());
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle(), true).isFiniteNonZero());

  // Test +/- qNaN. +/- dont mean anything with qNaN but paranoia can't hurt in
  // this instance.
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle(), false).isFiniteNonZero());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle(), true).isFiniteNonZero());

  // Test +/- sNaN. +/- dont mean anything with sNaN but paranoia can't hurt in
  // this instance.
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle(), false).isFiniteNonZero());
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle(), true).isFiniteNonZero());
}

TEST(APFloatTest, add) {
  // Test Special Cases against each other and normal values.

  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle(), false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle(), true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle(), false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle(), true);
  APFloat QNaN = APFloat::getNaN(APFloat::IEEEsingle(), false);
  APFloat SNaN = APFloat(APFloat::IEEEsingle(), "snan123");
  APFloat PNormalValue = APFloat(APFloat::IEEEsingle(), "0x1p+0");
  APFloat MNormalValue = APFloat(APFloat::IEEEsingle(), "-0x1p+0");
  APFloat PLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), false);
  APFloat MLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), true);
  APFloat PSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), false);
  APFloat MSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), true);
  APFloat PSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle(), false);
  APFloat MSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle(), true);

  const int OverflowStatus = APFloat::opOverflow | APFloat::opInexact;

  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
  } SpecialCaseTests[] = {
    { PInf, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PZero, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MZero, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PInf, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PNormalValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MNormalValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PLargestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MLargestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PSmallestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MSmallestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PSmallestNormalized, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MSmallestNormalized, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PZero, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MZero, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MInf, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PNormalValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MNormalValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PLargestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MLargestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PSmallestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MSmallestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PSmallestNormalized, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MSmallestNormalized, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PZero, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PZero, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PZero, PZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PZero, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, PNormalValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PZero, MNormalValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PZero, PLargestValue, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PZero, MLargestValue, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PZero, PSmallestValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PZero, MSmallestValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PZero, PSmallestNormalized, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PZero, MSmallestNormalized, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MZero, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MZero, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MZero, PZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MZero, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MZero, PNormalValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MZero, MNormalValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MZero, PLargestValue, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MZero, MLargestValue, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MZero, PSmallestValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MZero, MSmallestValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MZero, PSmallestNormalized, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MZero, MSmallestNormalized, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { QNaN, PInf, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MInf, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PZero, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MZero, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { QNaN, PNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { SNaN, PInf, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MInf, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PZero, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MZero, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, QNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PNormalValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MNormalValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PLargestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MLargestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestNormalized, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestNormalized, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, PZero, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MZero, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PNormalValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, PNormalValue, "0x1p+1", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, PLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PNormalValue, MLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PNormalValue, PSmallestValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PNormalValue, MSmallestValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PNormalValue, PSmallestNormalized, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PNormalValue, MSmallestNormalized, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MNormalValue, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MNormalValue, PZero, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, MZero, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MNormalValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MNormalValue, PNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, MNormalValue, "-0x1p+1", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, PLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, MLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, PSmallestValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, MSmallestValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, PSmallestNormalized, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, MSmallestNormalized, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PLargestValue, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PLargestValue, PZero, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, MZero, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PLargestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PLargestValue, PNormalValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, MNormalValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, PLargestValue, "inf", OverflowStatus, APFloat::fcInfinity },
    { PLargestValue, MLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, PSmallestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, MSmallestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, PSmallestNormalized, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, MSmallestNormalized, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MLargestValue, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MLargestValue, PZero, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, MZero, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MLargestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MLargestValue, PNormalValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, MNormalValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, PLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, MLargestValue, "-inf", OverflowStatus, APFloat::fcInfinity },
    { MLargestValue, PSmallestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, MSmallestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, PSmallestNormalized, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, MSmallestNormalized, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestValue, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestValue, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestValue, PZero, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MZero, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PSmallestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestValue, PNormalValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestValue, MNormalValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestValue, PLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestValue, MLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestValue, PSmallestValue, "0x1p-148", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestValue, PSmallestNormalized, "0x1.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MSmallestNormalized, "-0x1.fffffcp-127", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestValue, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestValue, PZero, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MZero, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MSmallestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestValue, PNormalValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestValue, MNormalValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestValue, PLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestValue, MLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestValue, PSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestValue, MSmallestValue, "-0x1p-148", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PSmallestNormalized, "0x1.fffffcp-127", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MSmallestNormalized, "-0x1.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestNormalized, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestNormalized, PZero, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MZero, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PSmallestNormalized, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestNormalized, PNormalValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestNormalized, MNormalValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestNormalized, PLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestNormalized, MLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestNormalized, PSmallestValue, "0x1.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MSmallestValue, "0x1.fffffcp-127", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PSmallestNormalized, "0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestNormalized, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestNormalized, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestNormalized, PZero, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MZero, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MSmallestNormalized, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestNormalized, PNormalValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, MNormalValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, PLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, MLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestValue, "-0x1.fffffcp-127", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MSmallestValue, "-0x1.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestNormalized, MSmallestNormalized, "-0x1p-125", APFloat::opOK, APFloat::fcNormal }
  };

  for (size_t i = 0; i < std::size(SpecialCaseTests); ++i) {
    APFloat x(SpecialCaseTests[i].x);
    APFloat y(SpecialCaseTests[i].y);
    APFloat::opStatus status = x.add(y, APFloat::rmNearestTiesToEven);

    APFloat result(APFloat::IEEEsingle(), SpecialCaseTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_EQ(SpecialCaseTests[i].status, (int)status);
    EXPECT_EQ(SpecialCaseTests[i].category, (int)x.getCategory());
  }
}

TEST(APFloatTest, subtract) {
  // Test Special Cases against each other and normal values.

  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle(), false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle(), true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle(), false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle(), true);
  APFloat QNaN = APFloat::getNaN(APFloat::IEEEsingle(), false);
  APFloat SNaN = APFloat(APFloat::IEEEsingle(), "snan123");
  APFloat PNormalValue = APFloat(APFloat::IEEEsingle(), "0x1p+0");
  APFloat MNormalValue = APFloat(APFloat::IEEEsingle(), "-0x1p+0");
  APFloat PLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), false);
  APFloat MLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), true);
  APFloat PSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), false);
  APFloat MSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), true);
  APFloat PSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle(), false);
  APFloat MSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle(), true);

  const int OverflowStatus = APFloat::opOverflow | APFloat::opInexact;

  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
  } SpecialCaseTests[] = {
    { PInf, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PZero, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MZero, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PInf, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PNormalValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MNormalValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PLargestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MLargestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PSmallestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MSmallestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PSmallestNormalized, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MSmallestNormalized, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PZero, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MZero, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MInf, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PNormalValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MNormalValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PLargestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MLargestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PSmallestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MSmallestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PSmallestNormalized, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MSmallestNormalized, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PZero, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PZero, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PZero, PZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PZero, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, PNormalValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PZero, MNormalValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PZero, PLargestValue, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PZero, MLargestValue, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PZero, PSmallestValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PZero, MSmallestValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PZero, PSmallestNormalized, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PZero, MSmallestNormalized, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MZero, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MZero, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MZero, PZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MZero, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MZero, PNormalValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MZero, MNormalValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MZero, PLargestValue, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MZero, MLargestValue, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MZero, PSmallestValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MZero, MSmallestValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MZero, PSmallestNormalized, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MZero, MSmallestNormalized, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { QNaN, PInf, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MInf, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PZero, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MZero, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { QNaN, PNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { SNaN, PInf, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MInf, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PZero, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MZero, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, QNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PNormalValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MNormalValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PLargestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MLargestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestNormalized, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestNormalized, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, PZero, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MZero, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PNormalValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, PNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, MNormalValue, "0x1p+1", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, PLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PNormalValue, MLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PNormalValue, PSmallestValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PNormalValue, MSmallestValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PNormalValue, PSmallestNormalized, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PNormalValue, MSmallestNormalized, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MNormalValue, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MNormalValue, PZero, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, MZero, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MNormalValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MNormalValue, PNormalValue, "-0x1p+1", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, MNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, PLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, MLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, PSmallestValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, MSmallestValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, PSmallestNormalized, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MNormalValue, MSmallestNormalized, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PLargestValue, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PLargestValue, PZero, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, MZero, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PLargestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PLargestValue, PNormalValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, MNormalValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, PLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, MLargestValue, "inf", OverflowStatus, APFloat::fcInfinity },
    { PLargestValue, PSmallestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, MSmallestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, PSmallestNormalized, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PLargestValue, MSmallestNormalized, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MLargestValue, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MLargestValue, PZero, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, MZero, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MLargestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MLargestValue, PNormalValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, MNormalValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, PLargestValue, "-inf", OverflowStatus, APFloat::fcInfinity },
    { MLargestValue, MLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, PSmallestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, MSmallestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, PSmallestNormalized, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MLargestValue, MSmallestNormalized, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestValue, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestValue, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestValue, PZero, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MZero, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PSmallestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestValue, PNormalValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestValue, MNormalValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestValue, PLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestValue, MLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestValue, PSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestValue, MSmallestValue, "0x1p-148", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, PSmallestNormalized, "-0x1.fffffcp-127", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MSmallestNormalized, "0x1.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestValue, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestValue, PZero, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MZero, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MSmallestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestValue, PNormalValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestValue, MNormalValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestValue, PLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestValue, MLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestValue, PSmallestValue, "-0x1p-148", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestValue, PSmallestNormalized, "-0x1.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MSmallestNormalized, "0x1.fffffcp-127", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestNormalized, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestNormalized, PZero, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MZero, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PSmallestNormalized, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestNormalized, PNormalValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestNormalized, MNormalValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestNormalized, PLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestNormalized, MLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { PSmallestNormalized, PSmallestValue, "0x1.fffffcp-127", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MSmallestValue, "0x1.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestNormalized, MSmallestNormalized, "0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestNormalized, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestNormalized, PZero, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MZero, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MSmallestNormalized, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestNormalized, PNormalValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, MNormalValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, PLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, MLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestValue, "-0x1.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MSmallestValue, "-0x1.fffffcp-127", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestNormalized, "-0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero }
  };

  for (size_t i = 0; i < std::size(SpecialCaseTests); ++i) {
    APFloat x(SpecialCaseTests[i].x);
    APFloat y(SpecialCaseTests[i].y);
    APFloat::opStatus status = x.subtract(y, APFloat::rmNearestTiesToEven);

    APFloat result(APFloat::IEEEsingle(), SpecialCaseTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_EQ(SpecialCaseTests[i].status, (int)status);
    EXPECT_EQ(SpecialCaseTests[i].category, (int)x.getCategory());
  }
}

TEST(APFloatTest, multiply) {
  // Test Special Cases against each other and normal values.

  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle(), false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle(), true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle(), false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle(), true);
  APFloat QNaN = APFloat::getNaN(APFloat::IEEEsingle(), false);
  APFloat SNaN = APFloat(APFloat::IEEEsingle(), "snan123");
  APFloat PNormalValue = APFloat(APFloat::IEEEsingle(), "0x1p+0");
  APFloat MNormalValue = APFloat(APFloat::IEEEsingle(), "-0x1p+0");
  APFloat PLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), false);
  APFloat MLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), true);
  APFloat PSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), false);
  APFloat MSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), true);
  APFloat PSmallestNormalized =
      APFloat::getSmallestNormalized(APFloat::IEEEsingle(), false);
  APFloat MSmallestNormalized =
      APFloat::getSmallestNormalized(APFloat::IEEEsingle(), true);

  APFloat MaxQuad(APFloat::IEEEquad(),
                  "0x1.ffffffffffffffffffffffffffffp+16383");
  APFloat MinQuad(APFloat::IEEEquad(),
                  "0x0.0000000000000000000000000001p-16382");
  APFloat NMinQuad(APFloat::IEEEquad(),
                   "-0x0.0000000000000000000000000001p-16382");

  const int OverflowStatus = APFloat::opOverflow | APFloat::opInexact;
  const int UnderflowStatus = APFloat::opUnderflow | APFloat::opInexact;

  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
    APFloat::roundingMode roundingMode = APFloat::rmNearestTiesToEven;
  } SpecialCaseTests[] = {
    { PInf, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PInf, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PNormalValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MNormalValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PLargestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MLargestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PSmallestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MSmallestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PSmallestNormalized, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MSmallestNormalized, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MInf, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PNormalValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MNormalValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PLargestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MLargestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PSmallestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MSmallestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PSmallestNormalized, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MSmallestNormalized, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PZero, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, PZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PZero, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, PNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MNormalValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, PLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MLargestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, PSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, PSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MZero, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MZero, PZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MZero, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MZero, PNormalValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PLargestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { QNaN, PInf, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MInf, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PZero, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MZero, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { QNaN, PNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { SNaN, PInf, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MInf, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PZero, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MZero, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, QNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PNormalValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MNormalValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PLargestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MLargestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestNormalized, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestNormalized, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, PZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, MZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PNormalValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, PNormalValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MNormalValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, PLargestValue, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MLargestValue, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, PSmallestValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MSmallestValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, PSmallestNormalized, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MSmallestNormalized, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MNormalValue, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MNormalValue, PZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, MZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MNormalValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MNormalValue, PNormalValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, MNormalValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, PLargestValue, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, MLargestValue, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, PSmallestValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, MSmallestValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, PSmallestNormalized, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, MSmallestNormalized, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PLargestValue, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PLargestValue, PZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, MZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PLargestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PLargestValue, PNormalValue, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, MNormalValue, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, PLargestValue, "inf", OverflowStatus, APFloat::fcInfinity },
    { PLargestValue, MLargestValue, "-inf", OverflowStatus, APFloat::fcInfinity },
    { PLargestValue, PSmallestValue, "0x1.fffffep-22", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, MSmallestValue, "-0x1.fffffep-22", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, PSmallestNormalized, "0x1.fffffep+1", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, MSmallestNormalized, "-0x1.fffffep+1", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MLargestValue, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MLargestValue, PZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, MZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MLargestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MLargestValue, PNormalValue, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, MNormalValue, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, PLargestValue, "-inf", OverflowStatus, APFloat::fcInfinity },
    { MLargestValue, MLargestValue, "inf", OverflowStatus, APFloat::fcInfinity },
    { MLargestValue, PSmallestValue, "-0x1.fffffep-22", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, MSmallestValue, "0x1.fffffep-22", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, PSmallestNormalized, "-0x1.fffffep+1", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, MSmallestNormalized, "0x1.fffffep+1", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestValue, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestValue, PZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestValue, MZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PSmallestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestValue, PNormalValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MNormalValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, PLargestValue, "0x1.fffffep-22", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MLargestValue, "-0x1.fffffep-22", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, PSmallestValue, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { PSmallestValue, MSmallestValue, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { PSmallestValue, PSmallestNormalized, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { PSmallestValue, MSmallestNormalized, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestValue, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestValue, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestValue, PZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestValue, MZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MSmallestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestValue, PNormalValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MNormalValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PLargestValue, "-0x1.fffffep-22", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MLargestValue, "0x1.fffffep-22", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PSmallestValue, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestValue, MSmallestValue, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestValue, PSmallestNormalized, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestValue, MSmallestNormalized, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { PSmallestNormalized, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestNormalized, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PSmallestNormalized, PZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestNormalized, MZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestNormalized, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PSmallestNormalized, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestNormalized, PNormalValue, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MNormalValue, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PLargestValue, "0x1.fffffep+1", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MLargestValue, "-0x1.fffffep+1", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PSmallestValue, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { PSmallestNormalized, MSmallestValue, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { PSmallestNormalized, PSmallestNormalized, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { PSmallestNormalized, MSmallestNormalized, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestNormalized, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestNormalized, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MSmallestNormalized, PZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestNormalized, MZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestNormalized, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MSmallestNormalized, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestNormalized, PNormalValue, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MNormalValue, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PLargestValue, "-0x1.fffffep+1", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MLargestValue, "0x1.fffffep+1", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestValue, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestNormalized, MSmallestValue, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestNormalized, PSmallestNormalized, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestNormalized, MSmallestNormalized, "0x0p+0", UnderflowStatus, APFloat::fcZero },

    {MaxQuad, MinQuad, "0x1.ffffffffffffffffffffffffffffp-111", APFloat::opOK,
     APFloat::fcNormal, APFloat::rmNearestTiesToEven},
    {MaxQuad, MinQuad, "0x1.ffffffffffffffffffffffffffffp-111", APFloat::opOK,
     APFloat::fcNormal, APFloat::rmTowardPositive},
    {MaxQuad, MinQuad, "0x1.ffffffffffffffffffffffffffffp-111", APFloat::opOK,
     APFloat::fcNormal, APFloat::rmTowardNegative},
    {MaxQuad, MinQuad, "0x1.ffffffffffffffffffffffffffffp-111", APFloat::opOK,
     APFloat::fcNormal, APFloat::rmTowardZero},
    {MaxQuad, MinQuad, "0x1.ffffffffffffffffffffffffffffp-111", APFloat::opOK,
     APFloat::fcNormal, APFloat::rmNearestTiesToAway},

    {MaxQuad, NMinQuad, "-0x1.ffffffffffffffffffffffffffffp-111", APFloat::opOK,
     APFloat::fcNormal, APFloat::rmNearestTiesToEven},
    {MaxQuad, NMinQuad, "-0x1.ffffffffffffffffffffffffffffp-111", APFloat::opOK,
     APFloat::fcNormal, APFloat::rmTowardPositive},
    {MaxQuad, NMinQuad, "-0x1.ffffffffffffffffffffffffffffp-111", APFloat::opOK,
     APFloat::fcNormal, APFloat::rmTowardNegative},
    {MaxQuad, NMinQuad, "-0x1.ffffffffffffffffffffffffffffp-111", APFloat::opOK,
     APFloat::fcNormal, APFloat::rmTowardZero},
    {MaxQuad, NMinQuad, "-0x1.ffffffffffffffffffffffffffffp-111", APFloat::opOK,
     APFloat::fcNormal, APFloat::rmNearestTiesToAway},

    {MaxQuad, MaxQuad, "inf", OverflowStatus, APFloat::fcInfinity,
     APFloat::rmNearestTiesToEven},
    {MaxQuad, MaxQuad, "inf", OverflowStatus, APFloat::fcInfinity,
     APFloat::rmTowardPositive},
    {MaxQuad, MaxQuad, "0x1.ffffffffffffffffffffffffffffp+16383",
     APFloat::opInexact, APFloat::fcNormal, APFloat::rmTowardNegative},
    {MaxQuad, MaxQuad, "0x1.ffffffffffffffffffffffffffffp+16383",
     APFloat::opInexact, APFloat::fcNormal, APFloat::rmTowardZero},
    {MaxQuad, MaxQuad, "inf", OverflowStatus, APFloat::fcInfinity,
     APFloat::rmNearestTiesToAway},

    {MinQuad, MinQuad, "0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmNearestTiesToEven},
    {MinQuad, MinQuad, "0x0.0000000000000000000000000001p-16382",
     UnderflowStatus, APFloat::fcNormal, APFloat::rmTowardPositive},
    {MinQuad, MinQuad, "0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmTowardNegative},
    {MinQuad, MinQuad, "0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmTowardZero},
    {MinQuad, MinQuad, "0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmNearestTiesToAway},

    {MinQuad, NMinQuad, "-0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmNearestTiesToEven},
    {MinQuad, NMinQuad, "-0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmTowardPositive},
    {MinQuad, NMinQuad, "-0x0.0000000000000000000000000001p-16382",
     UnderflowStatus, APFloat::fcNormal, APFloat::rmTowardNegative},
    {MinQuad, NMinQuad, "-0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmTowardZero},
    {MinQuad, NMinQuad, "-0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmNearestTiesToAway},
  };

  for (size_t i = 0; i < std::size(SpecialCaseTests); ++i) {
    APFloat x(SpecialCaseTests[i].x);
    APFloat y(SpecialCaseTests[i].y);
    APFloat::opStatus status = x.multiply(y, SpecialCaseTests[i].roundingMode);

    APFloat result(x.getSemantics(), SpecialCaseTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_EQ(SpecialCaseTests[i].status, (int)status);
    EXPECT_EQ(SpecialCaseTests[i].category, (int)x.getCategory());
  }
}

TEST(APFloatTest, divide) {
  // Test Special Cases against each other and normal values.

  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle(), false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle(), true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle(), false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle(), true);
  APFloat QNaN = APFloat::getNaN(APFloat::IEEEsingle(), false);
  APFloat SNaN = APFloat(APFloat::IEEEsingle(), "snan123");
  APFloat PNormalValue = APFloat(APFloat::IEEEsingle(), "0x1p+0");
  APFloat MNormalValue = APFloat(APFloat::IEEEsingle(), "-0x1p+0");
  APFloat PLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), false);
  APFloat MLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), true);
  APFloat PSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), false);
  APFloat MSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), true);
  APFloat PSmallestNormalized =
      APFloat::getSmallestNormalized(APFloat::IEEEsingle(), false);
  APFloat MSmallestNormalized =
      APFloat::getSmallestNormalized(APFloat::IEEEsingle(), true);

  APFloat MaxQuad(APFloat::IEEEquad(),
                  "0x1.ffffffffffffffffffffffffffffp+16383");
  APFloat MinQuad(APFloat::IEEEquad(),
                  "0x0.0000000000000000000000000001p-16382");
  APFloat NMinQuad(APFloat::IEEEquad(),
                   "-0x0.0000000000000000000000000001p-16382");

  const int OverflowStatus = APFloat::opOverflow | APFloat::opInexact;
  const int UnderflowStatus = APFloat::opUnderflow | APFloat::opInexact;

  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
    APFloat::roundingMode roundingMode = APFloat::rmNearestTiesToEven;
  } SpecialCaseTests[] = {
    { PInf, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PZero, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MZero, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PInf, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PNormalValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MNormalValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PLargestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MLargestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PSmallestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MSmallestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PSmallestNormalized, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MSmallestNormalized, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PZero, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MZero, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MInf, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PNormalValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MNormalValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PLargestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MLargestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PSmallestValue, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MSmallestValue, "inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, PSmallestNormalized, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { MInf, MSmallestNormalized, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PZero, PInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PZero, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, PNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MNormalValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, PLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MLargestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, PSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, PSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MZero, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MZero, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MZero, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MZero, PNormalValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PLargestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { QNaN, PInf, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MInf, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PZero, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MZero, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { QNaN, PNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { SNaN, PInf, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MInf, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PZero, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MZero, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, QNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PNormalValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MNormalValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PLargestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MLargestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestNormalized, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestNormalized, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, PInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, MInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, PZero, "inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { PNormalValue, MZero, "-inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { PNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PNormalValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, PNormalValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MNormalValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, PLargestValue, "0x1p-128", UnderflowStatus, APFloat::fcNormal },
    { PNormalValue, MLargestValue, "-0x1p-128", UnderflowStatus, APFloat::fcNormal },
    { PNormalValue, PSmallestValue, "inf", OverflowStatus, APFloat::fcInfinity },
    { PNormalValue, MSmallestValue, "-inf", OverflowStatus, APFloat::fcInfinity },
    { PNormalValue, PSmallestNormalized, "0x1p+126", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MSmallestNormalized, "-0x1p+126", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, PInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, MInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, PZero, "-inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { MNormalValue, MZero, "inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { MNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MNormalValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MNormalValue, PNormalValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, MNormalValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, PLargestValue, "-0x1p-128", UnderflowStatus, APFloat::fcNormal },
    { MNormalValue, MLargestValue, "0x1p-128", UnderflowStatus, APFloat::fcNormal },
    { MNormalValue, PSmallestValue, "-inf", OverflowStatus, APFloat::fcInfinity },
    { MNormalValue, MSmallestValue, "inf", OverflowStatus, APFloat::fcInfinity },
    { MNormalValue, PSmallestNormalized, "-0x1p+126", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, MSmallestNormalized, "0x1p+126", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, PInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, MInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, PZero, "inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { PLargestValue, MZero, "-inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { PLargestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PLargestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PLargestValue, PNormalValue, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, MNormalValue, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, PLargestValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, MLargestValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, PSmallestValue, "inf", OverflowStatus, APFloat::fcInfinity },
    { PLargestValue, MSmallestValue, "-inf", OverflowStatus, APFloat::fcInfinity },
    { PLargestValue, PSmallestNormalized, "inf", OverflowStatus, APFloat::fcInfinity },
    { PLargestValue, MSmallestNormalized, "-inf", OverflowStatus, APFloat::fcInfinity },
    { MLargestValue, PInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, MInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, PZero, "-inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { MLargestValue, MZero, "inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { MLargestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MLargestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MLargestValue, PNormalValue, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, MNormalValue, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, PLargestValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, MLargestValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, PSmallestValue, "-inf", OverflowStatus, APFloat::fcInfinity },
    { MLargestValue, MSmallestValue, "inf", OverflowStatus, APFloat::fcInfinity },
    { MLargestValue, PSmallestNormalized, "-inf", OverflowStatus, APFloat::fcInfinity },
    { MLargestValue, MSmallestNormalized, "inf", OverflowStatus, APFloat::fcInfinity },
    { PSmallestValue, PInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestValue, MInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestValue, PZero, "inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { PSmallestValue, MZero, "-inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { PSmallestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PSmallestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestValue, PNormalValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MNormalValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, PLargestValue, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { PSmallestValue, MLargestValue, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { PSmallestValue, PSmallestValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MSmallestValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, PSmallestNormalized, "0x1p-23", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MSmallestNormalized, "-0x1p-23", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestValue, MInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestValue, PZero, "-inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { MSmallestValue, MZero, "inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { MSmallestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MSmallestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestValue, PNormalValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MNormalValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PLargestValue, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestValue, MLargestValue, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestValue, PSmallestValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MSmallestValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PSmallestNormalized, "-0x1p-23", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MSmallestNormalized, "0x1p-23", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestNormalized, MInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestNormalized, PZero, "inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { PSmallestNormalized, MZero, "-inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { PSmallestNormalized, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PSmallestNormalized, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestNormalized, PNormalValue, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MNormalValue, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PLargestValue, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { PSmallestNormalized, MLargestValue, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { PSmallestNormalized, PSmallestValue, "0x1p+23", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MSmallestValue, "-0x1p+23", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PSmallestNormalized, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MSmallestNormalized, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestNormalized, MInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestNormalized, PZero, "-inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { MSmallestNormalized, MZero, "inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { MSmallestNormalized, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MSmallestNormalized, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestNormalized, PNormalValue, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MNormalValue, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PLargestValue, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestNormalized, MLargestValue, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestNormalized, PSmallestValue, "-0x1p+23", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MSmallestValue, "0x1p+23", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestNormalized, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MSmallestNormalized, "0x1p+0", APFloat::opOK, APFloat::fcNormal },

    {MaxQuad, NMinQuad, "-inf", OverflowStatus, APFloat::fcInfinity,
     APFloat::rmNearestTiesToEven},
    {MaxQuad, NMinQuad, "-0x1.ffffffffffffffffffffffffffffp+16383",
     APFloat::opInexact, APFloat::fcNormal, APFloat::rmTowardPositive},
    {MaxQuad, NMinQuad, "-inf", OverflowStatus, APFloat::fcInfinity,
     APFloat::rmTowardNegative},
    {MaxQuad, NMinQuad, "-0x1.ffffffffffffffffffffffffffffp+16383",
     APFloat::opInexact, APFloat::fcNormal, APFloat::rmTowardZero},
    {MaxQuad, NMinQuad, "-inf", OverflowStatus, APFloat::fcInfinity,
     APFloat::rmNearestTiesToAway},

    {MinQuad, MaxQuad, "0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmNearestTiesToEven},
    {MinQuad, MaxQuad, "0x0.0000000000000000000000000001p-16382",
     UnderflowStatus, APFloat::fcNormal, APFloat::rmTowardPositive},
    {MinQuad, MaxQuad, "0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmTowardNegative},
    {MinQuad, MaxQuad, "0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmTowardZero},
    {MinQuad, MaxQuad, "0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmNearestTiesToAway},

    {NMinQuad, MaxQuad, "-0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmNearestTiesToEven},
    {NMinQuad, MaxQuad, "-0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmTowardPositive},
    {NMinQuad, MaxQuad, "-0x0.0000000000000000000000000001p-16382",
     UnderflowStatus, APFloat::fcNormal, APFloat::rmTowardNegative},
    {NMinQuad, MaxQuad, "-0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmTowardZero},
    {NMinQuad, MaxQuad, "-0", UnderflowStatus, APFloat::fcZero,
     APFloat::rmNearestTiesToAway},
  };

  for (size_t i = 0; i < std::size(SpecialCaseTests); ++i) {
    APFloat x(SpecialCaseTests[i].x);
    APFloat y(SpecialCaseTests[i].y);
    APFloat::opStatus status = x.divide(y, SpecialCaseTests[i].roundingMode);

    APFloat result(x.getSemantics(), SpecialCaseTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_EQ(SpecialCaseTests[i].status, (int)status);
    EXPECT_EQ(SpecialCaseTests[i].category, (int)x.getCategory());
  }
}

TEST(APFloatTest, operatorOverloads) {
  // This is mostly testing that these operator overloads compile.
  APFloat One = APFloat(APFloat::IEEEsingle(), "0x1p+0");
  APFloat Two = APFloat(APFloat::IEEEsingle(), "0x2p+0");
  EXPECT_TRUE(Two.bitwiseIsEqual(One + One));
  EXPECT_TRUE(One.bitwiseIsEqual(Two - One));
  EXPECT_TRUE(Two.bitwiseIsEqual(One * Two));
  EXPECT_TRUE(One.bitwiseIsEqual(Two / Two));
}

TEST(APFloatTest, Comparisons) {
  enum {MNan, MInf, MBig, MOne, MZer, PZer, POne, PBig, PInf, PNan, NumVals};
  APFloat Vals[NumVals] = {
    APFloat::getNaN(APFloat::IEEEsingle(), true),
    APFloat::getInf(APFloat::IEEEsingle(), true),
    APFloat::getLargest(APFloat::IEEEsingle(), true),
    APFloat(APFloat::IEEEsingle(), "-0x1p+0"),
    APFloat::getZero(APFloat::IEEEsingle(), true),
    APFloat::getZero(APFloat::IEEEsingle(), false),
    APFloat(APFloat::IEEEsingle(), "0x1p+0"),
    APFloat::getLargest(APFloat::IEEEsingle(), false),
    APFloat::getInf(APFloat::IEEEsingle(), false),
    APFloat::getNaN(APFloat::IEEEsingle(), false),
  };
  using Relation = void (*)(const APFloat &, const APFloat &);
  Relation LT = [](const APFloat &LHS, const APFloat &RHS) {
    EXPECT_FALSE(LHS == RHS);
    EXPECT_TRUE(LHS != RHS);
    EXPECT_TRUE(LHS < RHS);
    EXPECT_FALSE(LHS > RHS);
    EXPECT_TRUE(LHS <= RHS);
    EXPECT_FALSE(LHS >= RHS);
  };
  Relation EQ = [](const APFloat &LHS, const APFloat &RHS) {
    EXPECT_TRUE(LHS == RHS);
    EXPECT_FALSE(LHS != RHS);
    EXPECT_FALSE(LHS < RHS);
    EXPECT_FALSE(LHS > RHS);
    EXPECT_TRUE(LHS <= RHS);
    EXPECT_TRUE(LHS >= RHS);
  };
  Relation GT = [](const APFloat &LHS, const APFloat &RHS) {
    EXPECT_FALSE(LHS == RHS);
    EXPECT_TRUE(LHS != RHS);
    EXPECT_FALSE(LHS < RHS);
    EXPECT_TRUE(LHS > RHS);
    EXPECT_FALSE(LHS <= RHS);
    EXPECT_TRUE(LHS >= RHS);
  };
  Relation UN = [](const APFloat &LHS, const APFloat &RHS) {
    EXPECT_FALSE(LHS == RHS);
    EXPECT_TRUE(LHS != RHS);
    EXPECT_FALSE(LHS < RHS);
    EXPECT_FALSE(LHS > RHS);
    EXPECT_FALSE(LHS <= RHS);
    EXPECT_FALSE(LHS >= RHS);
  };
  Relation Relations[NumVals][NumVals] = {
    //          -N  -I  -B  -1  -0  +0  +1  +B  +I  +N
    /* MNan */ {UN, UN, UN, UN, UN, UN, UN, UN, UN, UN},
    /* MInf */ {UN, EQ, LT, LT, LT, LT, LT, LT, LT, UN},
    /* MBig */ {UN, GT, EQ, LT, LT, LT, LT, LT, LT, UN},
    /* MOne */ {UN, GT, GT, EQ, LT, LT, LT, LT, LT, UN},
    /* MZer */ {UN, GT, GT, GT, EQ, EQ, LT, LT, LT, UN},
    /* PZer */ {UN, GT, GT, GT, EQ, EQ, LT, LT, LT, UN},
    /* POne */ {UN, GT, GT, GT, GT, GT, EQ, LT, LT, UN},
    /* PBig */ {UN, GT, GT, GT, GT, GT, GT, EQ, LT, UN},
    /* PInf */ {UN, GT, GT, GT, GT, GT, GT, GT, EQ, UN},
    /* PNan */ {UN, UN, UN, UN, UN, UN, UN, UN, UN, UN},
  };
  for (unsigned I = 0; I < NumVals; ++I)
    for (unsigned J = 0; J < NumVals; ++J)
      Relations[I][J](Vals[I], Vals[J]);
}

TEST(APFloatTest, abs) {
  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle(), false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle(), true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle(), false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle(), true);
  APFloat PQNaN = APFloat::getNaN(APFloat::IEEEsingle(), false);
  APFloat MQNaN = APFloat::getNaN(APFloat::IEEEsingle(), true);
  APFloat PSNaN = APFloat::getSNaN(APFloat::IEEEsingle(), false);
  APFloat MSNaN = APFloat::getSNaN(APFloat::IEEEsingle(), true);
  APFloat PNormalValue = APFloat(APFloat::IEEEsingle(), "0x1p+0");
  APFloat MNormalValue = APFloat(APFloat::IEEEsingle(), "-0x1p+0");
  APFloat PLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), false);
  APFloat MLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), true);
  APFloat PSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), false);
  APFloat MSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), true);
  APFloat PSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle(), false);
  APFloat MSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle(), true);

  EXPECT_TRUE(PInf.bitwiseIsEqual(abs(PInf)));
  EXPECT_TRUE(PInf.bitwiseIsEqual(abs(MInf)));
  EXPECT_TRUE(PZero.bitwiseIsEqual(abs(PZero)));
  EXPECT_TRUE(PZero.bitwiseIsEqual(abs(MZero)));
  EXPECT_TRUE(PQNaN.bitwiseIsEqual(abs(PQNaN)));
  EXPECT_TRUE(PQNaN.bitwiseIsEqual(abs(MQNaN)));
  EXPECT_TRUE(PSNaN.bitwiseIsEqual(abs(PSNaN)));
  EXPECT_TRUE(PSNaN.bitwiseIsEqual(abs(MSNaN)));
  EXPECT_TRUE(PNormalValue.bitwiseIsEqual(abs(PNormalValue)));
  EXPECT_TRUE(PNormalValue.bitwiseIsEqual(abs(MNormalValue)));
  EXPECT_TRUE(PLargestValue.bitwiseIsEqual(abs(PLargestValue)));
  EXPECT_TRUE(PLargestValue.bitwiseIsEqual(abs(MLargestValue)));
  EXPECT_TRUE(PSmallestValue.bitwiseIsEqual(abs(PSmallestValue)));
  EXPECT_TRUE(PSmallestValue.bitwiseIsEqual(abs(MSmallestValue)));
  EXPECT_TRUE(PSmallestNormalized.bitwiseIsEqual(abs(PSmallestNormalized)));
  EXPECT_TRUE(PSmallestNormalized.bitwiseIsEqual(abs(MSmallestNormalized)));
}

TEST(APFloatTest, neg) {
  APFloat One = APFloat(APFloat::IEEEsingle(), "1.0");
  APFloat NegOne = APFloat(APFloat::IEEEsingle(), "-1.0");
  APFloat Zero = APFloat::getZero(APFloat::IEEEsingle(), false);
  APFloat NegZero = APFloat::getZero(APFloat::IEEEsingle(), true);
  APFloat Inf = APFloat::getInf(APFloat::IEEEsingle(), false);
  APFloat NegInf = APFloat::getInf(APFloat::IEEEsingle(), true);
  APFloat QNaN = APFloat::getNaN(APFloat::IEEEsingle(), false);
  APFloat NegQNaN = APFloat::getNaN(APFloat::IEEEsingle(), true);

  EXPECT_TRUE(NegOne.bitwiseIsEqual(neg(One)));
  EXPECT_TRUE(One.bitwiseIsEqual(neg(NegOne)));
  EXPECT_TRUE(NegZero.bitwiseIsEqual(neg(Zero)));
  EXPECT_TRUE(Zero.bitwiseIsEqual(neg(NegZero)));
  EXPECT_TRUE(NegInf.bitwiseIsEqual(neg(Inf)));
  EXPECT_TRUE(Inf.bitwiseIsEqual(neg(NegInf)));
  EXPECT_TRUE(NegInf.bitwiseIsEqual(neg(Inf)));
  EXPECT_TRUE(Inf.bitwiseIsEqual(neg(NegInf)));
  EXPECT_TRUE(NegQNaN.bitwiseIsEqual(neg(QNaN)));
  EXPECT_TRUE(QNaN.bitwiseIsEqual(neg(NegQNaN)));

  EXPECT_TRUE(NegOne.bitwiseIsEqual(-One));
  EXPECT_TRUE(One.bitwiseIsEqual(-NegOne));
  EXPECT_TRUE(NegZero.bitwiseIsEqual(-Zero));
  EXPECT_TRUE(Zero.bitwiseIsEqual(-NegZero));
  EXPECT_TRUE(NegInf.bitwiseIsEqual(-Inf));
  EXPECT_TRUE(Inf.bitwiseIsEqual(-NegInf));
  EXPECT_TRUE(NegInf.bitwiseIsEqual(-Inf));
  EXPECT_TRUE(Inf.bitwiseIsEqual(-NegInf));
  EXPECT_TRUE(NegQNaN.bitwiseIsEqual(-QNaN));
  EXPECT_TRUE(QNaN.bitwiseIsEqual(-NegQNaN));
}

TEST(APFloatTest, ilogb) {
  EXPECT_EQ(-1074, ilogb(APFloat::getSmallest(APFloat::IEEEdouble(), false)));
  EXPECT_EQ(-1074, ilogb(APFloat::getSmallest(APFloat::IEEEdouble(), true)));
  EXPECT_EQ(-1023, ilogb(APFloat(APFloat::IEEEdouble(), "0x1.ffffffffffffep-1024")));
  EXPECT_EQ(-1023, ilogb(APFloat(APFloat::IEEEdouble(), "0x1.ffffffffffffep-1023")));
  EXPECT_EQ(-1023, ilogb(APFloat(APFloat::IEEEdouble(), "-0x1.ffffffffffffep-1023")));
  EXPECT_EQ(-51, ilogb(APFloat(APFloat::IEEEdouble(), "0x1p-51")));
  EXPECT_EQ(-1023, ilogb(APFloat(APFloat::IEEEdouble(), "0x1.c60f120d9f87cp-1023")));
  EXPECT_EQ(-2, ilogb(APFloat(APFloat::IEEEdouble(), "0x0.ffffp-1")));
  EXPECT_EQ(-1023, ilogb(APFloat(APFloat::IEEEdouble(), "0x1.fffep-1023")));
  EXPECT_EQ(1023, ilogb(APFloat::getLargest(APFloat::IEEEdouble(), false)));
  EXPECT_EQ(1023, ilogb(APFloat::getLargest(APFloat::IEEEdouble(), true)));


  EXPECT_EQ(0, ilogb(APFloat(APFloat::IEEEsingle(), "0x1p+0")));
  EXPECT_EQ(0, ilogb(APFloat(APFloat::IEEEsingle(), "-0x1p+0")));
  EXPECT_EQ(42, ilogb(APFloat(APFloat::IEEEsingle(), "0x1p+42")));
  EXPECT_EQ(-42, ilogb(APFloat(APFloat::IEEEsingle(), "0x1p-42")));

  EXPECT_EQ(APFloat::IEK_Inf,
            ilogb(APFloat::getInf(APFloat::IEEEsingle(), false)));
  EXPECT_EQ(APFloat::IEK_Inf,
            ilogb(APFloat::getInf(APFloat::IEEEsingle(), true)));
  EXPECT_EQ(APFloat::IEK_Zero,
            ilogb(APFloat::getZero(APFloat::IEEEsingle(), false)));
  EXPECT_EQ(APFloat::IEK_Zero,
            ilogb(APFloat::getZero(APFloat::IEEEsingle(), true)));
  EXPECT_EQ(APFloat::IEK_NaN,
            ilogb(APFloat::getNaN(APFloat::IEEEsingle(), false)));
  EXPECT_EQ(APFloat::IEK_NaN,
            ilogb(APFloat::getSNaN(APFloat::IEEEsingle(), false)));

  EXPECT_EQ(127, ilogb(APFloat::getLargest(APFloat::IEEEsingle(), false)));
  EXPECT_EQ(127, ilogb(APFloat::getLargest(APFloat::IEEEsingle(), true)));

  EXPECT_EQ(-149, ilogb(APFloat::getSmallest(APFloat::IEEEsingle(), false)));
  EXPECT_EQ(-149, ilogb(APFloat::getSmallest(APFloat::IEEEsingle(), true)));
  EXPECT_EQ(-126,
            ilogb(APFloat::getSmallestNormalized(APFloat::IEEEsingle(), false)));
  EXPECT_EQ(-126,
            ilogb(APFloat::getSmallestNormalized(APFloat::IEEEsingle(), true)));
}

TEST(APFloatTest, scalbn) {

  const APFloat::roundingMode RM = APFloat::rmNearestTiesToEven;
  EXPECT_TRUE(
      APFloat(APFloat::IEEEsingle(), "0x1p+0")
      .bitwiseIsEqual(scalbn(APFloat(APFloat::IEEEsingle(), "0x1p+0"), 0, RM)));
  EXPECT_TRUE(
      APFloat(APFloat::IEEEsingle(), "0x1p+42")
      .bitwiseIsEqual(scalbn(APFloat(APFloat::IEEEsingle(), "0x1p+0"), 42, RM)));
  EXPECT_TRUE(
      APFloat(APFloat::IEEEsingle(), "0x1p-42")
      .bitwiseIsEqual(scalbn(APFloat(APFloat::IEEEsingle(), "0x1p+0"), -42, RM)));

  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle(), false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle(), true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle(), false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle(), true);
  APFloat QPNaN = APFloat::getNaN(APFloat::IEEEsingle(), false);
  APFloat QMNaN = APFloat::getNaN(APFloat::IEEEsingle(), true);
  APFloat SNaN = APFloat::getSNaN(APFloat::IEEEsingle(), false);

  EXPECT_TRUE(PInf.bitwiseIsEqual(scalbn(PInf, 0, RM)));
  EXPECT_TRUE(MInf.bitwiseIsEqual(scalbn(MInf, 0, RM)));
  EXPECT_TRUE(PZero.bitwiseIsEqual(scalbn(PZero, 0, RM)));
  EXPECT_TRUE(MZero.bitwiseIsEqual(scalbn(MZero, 0, RM)));
  EXPECT_TRUE(QPNaN.bitwiseIsEqual(scalbn(QPNaN, 0, RM)));
  EXPECT_TRUE(QMNaN.bitwiseIsEqual(scalbn(QMNaN, 0, RM)));
  EXPECT_FALSE(scalbn(SNaN, 0, RM).isSignaling());

  APFloat ScalbnSNaN = scalbn(SNaN, 1, RM);
  EXPECT_TRUE(ScalbnSNaN.isNaN() && !ScalbnSNaN.isSignaling());

  // Make sure highest bit of payload is preserved.
  const APInt Payload(64, (UINT64_C(1) << 50) |
                      (UINT64_C(1) << 49) |
                      (UINT64_C(1234) << 32) |
                      1);

  APFloat SNaNWithPayload = APFloat::getSNaN(APFloat::IEEEdouble(), false,
                                             &Payload);
  APFloat QuietPayload = scalbn(SNaNWithPayload, 1, RM);
  EXPECT_TRUE(QuietPayload.isNaN() && !QuietPayload.isSignaling());
  EXPECT_EQ(Payload, QuietPayload.bitcastToAPInt().getLoBits(51));

  EXPECT_TRUE(PInf.bitwiseIsEqual(
                scalbn(APFloat(APFloat::IEEEsingle(), "0x1p+0"), 128, RM)));
  EXPECT_TRUE(MInf.bitwiseIsEqual(
                scalbn(APFloat(APFloat::IEEEsingle(), "-0x1p+0"), 128, RM)));
  EXPECT_TRUE(PInf.bitwiseIsEqual(
                scalbn(APFloat(APFloat::IEEEsingle(), "0x1p+127"), 1, RM)));
  EXPECT_TRUE(PZero.bitwiseIsEqual(
                scalbn(APFloat(APFloat::IEEEsingle(), "0x1p-127"), -127, RM)));
  EXPECT_TRUE(MZero.bitwiseIsEqual(
                scalbn(APFloat(APFloat::IEEEsingle(), "-0x1p-127"), -127, RM)));
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle(), "-0x1p-149").bitwiseIsEqual(
                scalbn(APFloat(APFloat::IEEEsingle(), "-0x1p-127"), -22, RM)));
  EXPECT_TRUE(PZero.bitwiseIsEqual(
                scalbn(APFloat(APFloat::IEEEsingle(), "0x1p-126"), -24, RM)));


  APFloat SmallestF64 = APFloat::getSmallest(APFloat::IEEEdouble(), false);
  APFloat NegSmallestF64 = APFloat::getSmallest(APFloat::IEEEdouble(), true);

  APFloat LargestF64 = APFloat::getLargest(APFloat::IEEEdouble(), false);
  APFloat NegLargestF64 = APFloat::getLargest(APFloat::IEEEdouble(), true);

  APFloat SmallestNormalizedF64
    = APFloat::getSmallestNormalized(APFloat::IEEEdouble(), false);
  APFloat NegSmallestNormalizedF64
    = APFloat::getSmallestNormalized(APFloat::IEEEdouble(), true);

  APFloat LargestDenormalF64(APFloat::IEEEdouble(), "0x1.ffffffffffffep-1023");
  APFloat NegLargestDenormalF64(APFloat::IEEEdouble(), "-0x1.ffffffffffffep-1023");


  EXPECT_TRUE(SmallestF64.bitwiseIsEqual(
                scalbn(APFloat(APFloat::IEEEdouble(), "0x1p-1074"), 0, RM)));
  EXPECT_TRUE(NegSmallestF64.bitwiseIsEqual(
                scalbn(APFloat(APFloat::IEEEdouble(), "-0x1p-1074"), 0, RM)));

  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1p+1023")
              .bitwiseIsEqual(scalbn(SmallestF64, 2097, RM)));

  EXPECT_TRUE(scalbn(SmallestF64, -2097, RM).isPosZero());
  EXPECT_TRUE(scalbn(SmallestF64, -2098, RM).isPosZero());
  EXPECT_TRUE(scalbn(SmallestF64, -2099, RM).isPosZero());
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1p+1022")
              .bitwiseIsEqual(scalbn(SmallestF64, 2096, RM)));
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1p+1023")
              .bitwiseIsEqual(scalbn(SmallestF64, 2097, RM)));
  EXPECT_TRUE(scalbn(SmallestF64, 2098, RM).isInfinity());
  EXPECT_TRUE(scalbn(SmallestF64, 2099, RM).isInfinity());

  // Test for integer overflows when adding to exponent.
  EXPECT_TRUE(scalbn(SmallestF64, -INT_MAX, RM).isPosZero());
  EXPECT_TRUE(scalbn(LargestF64, INT_MAX, RM).isInfinity());

  EXPECT_TRUE(LargestDenormalF64
              .bitwiseIsEqual(scalbn(LargestDenormalF64, 0, RM)));
  EXPECT_TRUE(NegLargestDenormalF64
              .bitwiseIsEqual(scalbn(NegLargestDenormalF64, 0, RM)));

  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.ffffffffffffep-1022")
              .bitwiseIsEqual(scalbn(LargestDenormalF64, 1, RM)));
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "-0x1.ffffffffffffep-1021")
              .bitwiseIsEqual(scalbn(NegLargestDenormalF64, 2, RM)));

  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.ffffffffffffep+1")
              .bitwiseIsEqual(scalbn(LargestDenormalF64, 1024, RM)));
  EXPECT_TRUE(scalbn(LargestDenormalF64, -1023, RM).isPosZero());
  EXPECT_TRUE(scalbn(LargestDenormalF64, -1024, RM).isPosZero());
  EXPECT_TRUE(scalbn(LargestDenormalF64, -2048, RM).isPosZero());
  EXPECT_TRUE(scalbn(LargestDenormalF64, 2047, RM).isInfinity());
  EXPECT_TRUE(scalbn(LargestDenormalF64, 2098, RM).isInfinity());
  EXPECT_TRUE(scalbn(LargestDenormalF64, 2099, RM).isInfinity());

  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.ffffffffffffep-2")
              .bitwiseIsEqual(scalbn(LargestDenormalF64, 1021, RM)));
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.ffffffffffffep-1")
              .bitwiseIsEqual(scalbn(LargestDenormalF64, 1022, RM)));
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.ffffffffffffep+0")
              .bitwiseIsEqual(scalbn(LargestDenormalF64, 1023, RM)));
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.ffffffffffffep+1023")
              .bitwiseIsEqual(scalbn(LargestDenormalF64, 2046, RM)));
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1p+974")
              .bitwiseIsEqual(scalbn(SmallestF64, 2048, RM)));

  APFloat RandomDenormalF64(APFloat::IEEEdouble(), "0x1.c60f120d9f87cp+51");
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.c60f120d9f87cp-972")
              .bitwiseIsEqual(scalbn(RandomDenormalF64, -1023, RM)));
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.c60f120d9f87cp-1")
              .bitwiseIsEqual(scalbn(RandomDenormalF64, -52, RM)));
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.c60f120d9f87cp-2")
              .bitwiseIsEqual(scalbn(RandomDenormalF64, -53, RM)));
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.c60f120d9f87cp+0")
              .bitwiseIsEqual(scalbn(RandomDenormalF64, -51, RM)));

  EXPECT_TRUE(scalbn(RandomDenormalF64, -2097, RM).isPosZero());
  EXPECT_TRUE(scalbn(RandomDenormalF64, -2090, RM).isPosZero());


  EXPECT_TRUE(
    APFloat(APFloat::IEEEdouble(), "-0x1p-1073")
    .bitwiseIsEqual(scalbn(NegLargestF64, -2097, RM)));

  EXPECT_TRUE(
    APFloat(APFloat::IEEEdouble(), "-0x1p-1024")
    .bitwiseIsEqual(scalbn(NegLargestF64, -2048, RM)));

  EXPECT_TRUE(
    APFloat(APFloat::IEEEdouble(), "0x1p-1073")
    .bitwiseIsEqual(scalbn(LargestF64, -2097, RM)));

  EXPECT_TRUE(
    APFloat(APFloat::IEEEdouble(), "0x1p-1074")
    .bitwiseIsEqual(scalbn(LargestF64, -2098, RM)));
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "-0x1p-1074")
              .bitwiseIsEqual(scalbn(NegLargestF64, -2098, RM)));
  EXPECT_TRUE(scalbn(NegLargestF64, -2099, RM).isNegZero());
  EXPECT_TRUE(scalbn(LargestF64, 1, RM).isInfinity());


  EXPECT_TRUE(
    APFloat(APFloat::IEEEdouble(), "0x1p+0")
    .bitwiseIsEqual(scalbn(APFloat(APFloat::IEEEdouble(), "0x1p+52"), -52, RM)));

  EXPECT_TRUE(
    APFloat(APFloat::IEEEdouble(), "0x1p-103")
    .bitwiseIsEqual(scalbn(APFloat(APFloat::IEEEdouble(), "0x1p-51"), -52, RM)));
}

TEST(APFloatTest, frexp) {
  const APFloat::roundingMode RM = APFloat::rmNearestTiesToEven;

  APFloat PZero = APFloat::getZero(APFloat::IEEEdouble(), false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEdouble(), true);
  APFloat One(1.0);
  APFloat MOne(-1.0);
  APFloat Two(2.0);
  APFloat MTwo(-2.0);

  APFloat LargestDenormal(APFloat::IEEEdouble(), "0x1.ffffffffffffep-1023");
  APFloat NegLargestDenormal(APFloat::IEEEdouble(), "-0x1.ffffffffffffep-1023");

  APFloat Smallest = APFloat::getSmallest(APFloat::IEEEdouble(), false);
  APFloat NegSmallest = APFloat::getSmallest(APFloat::IEEEdouble(), true);

  APFloat Largest = APFloat::getLargest(APFloat::IEEEdouble(), false);
  APFloat NegLargest = APFloat::getLargest(APFloat::IEEEdouble(), true);

  APFloat PInf = APFloat::getInf(APFloat::IEEEdouble(), false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEdouble(), true);

  APFloat QPNaN = APFloat::getNaN(APFloat::IEEEdouble(), false);
  APFloat QMNaN = APFloat::getNaN(APFloat::IEEEdouble(), true);
  APFloat SNaN = APFloat::getSNaN(APFloat::IEEEdouble(), false);

  // Make sure highest bit of payload is preserved.
  const APInt Payload(64, (UINT64_C(1) << 50) |
                      (UINT64_C(1) << 49) |
                      (UINT64_C(1234) << 32) |
                      1);

  APFloat SNaNWithPayload = APFloat::getSNaN(APFloat::IEEEdouble(), false,
                                             &Payload);

  APFloat SmallestNormalized
    = APFloat::getSmallestNormalized(APFloat::IEEEdouble(), false);
  APFloat NegSmallestNormalized
    = APFloat::getSmallestNormalized(APFloat::IEEEdouble(), true);

  int Exp;
  APFloat Frac(APFloat::IEEEdouble());


  Frac = frexp(PZero, Exp, RM);
  EXPECT_EQ(0, Exp);
  EXPECT_TRUE(Frac.isPosZero());

  Frac = frexp(MZero, Exp, RM);
  EXPECT_EQ(0, Exp);
  EXPECT_TRUE(Frac.isNegZero());


  Frac = frexp(One, Exp, RM);
  EXPECT_EQ(1, Exp);
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1p-1").bitwiseIsEqual(Frac));

  Frac = frexp(MOne, Exp, RM);
  EXPECT_EQ(1, Exp);
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "-0x1p-1").bitwiseIsEqual(Frac));

  Frac = frexp(LargestDenormal, Exp, RM);
  EXPECT_EQ(-1022, Exp);
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.ffffffffffffep-1").bitwiseIsEqual(Frac));

  Frac = frexp(NegLargestDenormal, Exp, RM);
  EXPECT_EQ(-1022, Exp);
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "-0x1.ffffffffffffep-1").bitwiseIsEqual(Frac));


  Frac = frexp(Smallest, Exp, RM);
  EXPECT_EQ(-1073, Exp);
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1p-1").bitwiseIsEqual(Frac));

  Frac = frexp(NegSmallest, Exp, RM);
  EXPECT_EQ(-1073, Exp);
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "-0x1p-1").bitwiseIsEqual(Frac));


  Frac = frexp(Largest, Exp, RM);
  EXPECT_EQ(1024, Exp);
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.fffffffffffffp-1").bitwiseIsEqual(Frac));

  Frac = frexp(NegLargest, Exp, RM);
  EXPECT_EQ(1024, Exp);
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "-0x1.fffffffffffffp-1").bitwiseIsEqual(Frac));


  Frac = frexp(PInf, Exp, RM);
  EXPECT_EQ(INT_MAX, Exp);
  EXPECT_TRUE(Frac.isInfinity() && !Frac.isNegative());

  Frac = frexp(MInf, Exp, RM);
  EXPECT_EQ(INT_MAX, Exp);
  EXPECT_TRUE(Frac.isInfinity() && Frac.isNegative());

  Frac = frexp(QPNaN, Exp, RM);
  EXPECT_EQ(INT_MIN, Exp);
  EXPECT_TRUE(Frac.isNaN());

  Frac = frexp(QMNaN, Exp, RM);
  EXPECT_EQ(INT_MIN, Exp);
  EXPECT_TRUE(Frac.isNaN());

  Frac = frexp(SNaN, Exp, RM);
  EXPECT_EQ(INT_MIN, Exp);
  EXPECT_TRUE(Frac.isNaN() && !Frac.isSignaling());

  Frac = frexp(SNaNWithPayload, Exp, RM);
  EXPECT_EQ(INT_MIN, Exp);
  EXPECT_TRUE(Frac.isNaN() && !Frac.isSignaling());
  EXPECT_EQ(Payload, Frac.bitcastToAPInt().getLoBits(51));

  Frac = frexp(APFloat(APFloat::IEEEdouble(), "0x0.ffffp-1"), Exp, RM);
  EXPECT_EQ(-1, Exp);
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.fffep-1").bitwiseIsEqual(Frac));

  Frac = frexp(APFloat(APFloat::IEEEdouble(), "0x1p-51"), Exp, RM);
  EXPECT_EQ(-50, Exp);
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1p-1").bitwiseIsEqual(Frac));

  Frac = frexp(APFloat(APFloat::IEEEdouble(), "0x1.c60f120d9f87cp+51"), Exp, RM);
  EXPECT_EQ(52, Exp);
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble(), "0x1.c60f120d9f87cp-1").bitwiseIsEqual(Frac));
}

TEST(APFloatTest, mod) {
  {
    APFloat f1(APFloat::IEEEdouble(), "1.5");
    APFloat f2(APFloat::IEEEdouble(), "1.0");
    APFloat expected(APFloat::IEEEdouble(), "0.5");
    EXPECT_EQ(f1.mod(f2), APFloat::opOK);
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "0.5");
    APFloat f2(APFloat::IEEEdouble(), "1.0");
    APFloat expected(APFloat::IEEEdouble(), "0.5");
    EXPECT_EQ(f1.mod(f2), APFloat::opOK);
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "0x1.3333333333333p-2"); // 0.3
    APFloat f2(APFloat::IEEEdouble(), "0x1.47ae147ae147bp-7"); // 0.01
    APFloat expected(APFloat::IEEEdouble(),
                     "0x1.47ae147ae1471p-7"); // 0.009999999999999983
    EXPECT_EQ(f1.mod(f2), APFloat::opOK);
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "0x1p64"); // 1.8446744073709552e19
    APFloat f2(APFloat::IEEEdouble(), "1.5");
    APFloat expected(APFloat::IEEEdouble(), "1.0");
    EXPECT_EQ(f1.mod(f2), APFloat::opOK);
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "0x1p1000");
    APFloat f2(APFloat::IEEEdouble(), "0x1p-1000");
    APFloat expected(APFloat::IEEEdouble(), "0.0");
    EXPECT_EQ(f1.mod(f2), APFloat::opOK);
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "0.0");
    APFloat f2(APFloat::IEEEdouble(), "1.0");
    APFloat expected(APFloat::IEEEdouble(), "0.0");
    EXPECT_EQ(f1.mod(f2), APFloat::opOK);
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "1.0");
    APFloat f2(APFloat::IEEEdouble(), "0.0");
    EXPECT_EQ(f1.mod(f2), APFloat::opInvalidOp);
    EXPECT_TRUE(f1.isNaN());
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "0.0");
    APFloat f2(APFloat::IEEEdouble(), "0.0");
    EXPECT_EQ(f1.mod(f2), APFloat::opInvalidOp);
    EXPECT_TRUE(f1.isNaN());
  }
  {
    APFloat f1 = APFloat::getInf(APFloat::IEEEdouble(), false);
    APFloat f2(APFloat::IEEEdouble(), "1.0");
    EXPECT_EQ(f1.mod(f2), APFloat::opInvalidOp);
    EXPECT_TRUE(f1.isNaN());
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "-4.0");
    APFloat f2(APFloat::IEEEdouble(), "-2.0");
    APFloat expected(APFloat::IEEEdouble(), "-0.0");
    EXPECT_EQ(f1.mod(f2), APFloat::opOK);
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "-4.0");
    APFloat f2(APFloat::IEEEdouble(), "2.0");
    APFloat expected(APFloat::IEEEdouble(), "-0.0");
    EXPECT_EQ(f1.mod(f2), APFloat::opOK);
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    // Test E4M3FN mod where the LHS exponent is maxExponent (8) and the RHS is
    // the max value whose exponent is minExponent (-6). This requires special
    // logic in the mod implementation to prevent overflow to NaN.
    APFloat f1(APFloat::Float8E4M3FN(), "0x1p8");        // 256
    APFloat f2(APFloat::Float8E4M3FN(), "0x1.ep-6");     // 0.029296875
    APFloat expected(APFloat::Float8E4M3FN(), "0x1p-8"); // 0.00390625
    EXPECT_EQ(f1.mod(f2), APFloat::opOK);
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
}

TEST(APFloatTest, remainder) {
  // Test Special Cases against each other and normal values.

  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle(), false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle(), true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle(), false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle(), true);
  APFloat QNaN = APFloat::getNaN(APFloat::IEEEsingle(), false);
  APFloat SNaN = APFloat(APFloat::IEEEsingle(), "snan123");
  APFloat PNormalValue = APFloat(APFloat::IEEEsingle(), "0x1p+0");
  APFloat MNormalValue = APFloat(APFloat::IEEEsingle(), "-0x1p+0");
  APFloat PLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), false);
  APFloat MLargestValue = APFloat::getLargest(APFloat::IEEEsingle(), true);
  APFloat PSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), false);
  APFloat MSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle(), true);
  APFloat PSmallestNormalized =
      APFloat::getSmallestNormalized(APFloat::IEEEsingle(), false);
  APFloat MSmallestNormalized =
      APFloat::getSmallestNormalized(APFloat::IEEEsingle(), true);

  APFloat PVal1(APFloat::IEEEsingle(), "0x1.fffffep+126");
  APFloat MVal1(APFloat::IEEEsingle(), "-0x1.fffffep+126");
  APFloat PVal2(APFloat::IEEEsingle(), "0x1.fffffep-126");
  APFloat MVal2(APFloat::IEEEsingle(), "-0x1.fffffep-126");
  APFloat PVal3(APFloat::IEEEsingle(), "0x1p-125");
  APFloat MVal3(APFloat::IEEEsingle(), "-0x1p-125");
  APFloat PVal4(APFloat::IEEEsingle(), "0x1p+127");
  APFloat MVal4(APFloat::IEEEsingle(), "-0x1p+127");
  APFloat PVal5(APFloat::IEEEsingle(), "1.5");
  APFloat MVal5(APFloat::IEEEsingle(), "-1.5");
  APFloat PVal6(APFloat::IEEEsingle(), "1");
  APFloat MVal6(APFloat::IEEEsingle(), "-1");

  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
  } SpecialCaseTests[] = {
    { PInf, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PInf, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MInf, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, MNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, MLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, MSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, PSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MInf, MSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, PInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PZero, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PZero, PNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, PLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, PSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, PSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PZero, MSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MZero, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MZero, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MZero, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MZero, PNormalValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MNormalValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PLargestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MLargestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, PSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MZero, MSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { QNaN, PInf, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MInf, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PZero, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MZero, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { QNaN, PNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { SNaN, PInf, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MInf, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PZero, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MZero, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, QNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PNormalValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MNormalValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PLargestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MLargestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestValue, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestNormalized, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestNormalized, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, PInf, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MInf, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PNormalValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PNormalValue, PNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, MNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, PLargestValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MLargestValue, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, PSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, MSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, PSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, MSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, PInf, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, MInf, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MNormalValue, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MNormalValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MNormalValue, PNormalValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, MNormalValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, PLargestValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, MLargestValue, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MNormalValue, PSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, MSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, PSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MNormalValue, MSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, PInf, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, MInf, "0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { PLargestValue, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PLargestValue, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PLargestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PLargestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PLargestValue, PNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, MNormalValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, PLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, MLargestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, PSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, MSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, PSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PLargestValue, MSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, PInf, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, MInf, "-0x1.fffffep+127", APFloat::opOK, APFloat::fcNormal },
    { MLargestValue, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MLargestValue, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MLargestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MLargestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MLargestValue, PNormalValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, MNormalValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, PLargestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, MLargestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, PSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, MSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, PSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MLargestValue, MSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestValue, PInf, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MInf, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestValue, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PSmallestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestValue, PNormalValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MNormalValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, PLargestValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MLargestValue, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, PSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestValue, MSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestValue, PSmallestNormalized, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestValue, MSmallestNormalized, "0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PInf, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MInf, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestValue, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MSmallestValue, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestValue, PNormalValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MNormalValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PLargestValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MLargestValue, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, PSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestValue, MSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestValue, PSmallestNormalized, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { MSmallestValue, MSmallestNormalized, "-0x1p-149", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PInf, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MInf, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestNormalized, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestNormalized, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { PSmallestNormalized, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { PSmallestNormalized, PNormalValue, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MNormalValue, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PLargestValue, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, MLargestValue, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { PSmallestNormalized, PSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestNormalized, MSmallestValue, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestNormalized, PSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PSmallestNormalized, MSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestNormalized, PInf, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MInf, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestNormalized, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestNormalized, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
    { MSmallestNormalized, SNaN, "nan123", APFloat::opInvalidOp, APFloat::fcNaN },
    { MSmallestNormalized, PNormalValue, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MNormalValue, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PLargestValue, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MLargestValue, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestNormalized, MSmallestValue, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestNormalized, PSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestNormalized, MSmallestNormalized, "-0x0p+0", APFloat::opOK, APFloat::fcZero },

    { PVal1, PVal1, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal1, MVal1, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal1, PVal2, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal1, MVal2, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal1, PVal3, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal1, MVal3, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal1, PVal4, "-0x1p+103", APFloat::opOK, APFloat::fcNormal },
    { PVal1, MVal4, "-0x1p+103", APFloat::opOK, APFloat::fcNormal },
    { PVal1, PVal5, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal1, MVal5, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal1, PVal6, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal1, MVal6, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal1, PVal1, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal1, MVal1, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal1, PVal2, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal1, MVal2, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal1, PVal3, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal1, MVal3, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal1, PVal4, "0x1p+103", APFloat::opOK, APFloat::fcNormal },
    { MVal1, MVal4, "0x1p+103", APFloat::opOK, APFloat::fcNormal },
    { MVal1, PVal5, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal1, MVal5, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal1, PVal6, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal1, MVal6, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal2, PVal1, "0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { PVal2, MVal1, "0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { PVal2, PVal2, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal2, MVal2, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal2, PVal3, "-0x0.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { PVal2, MVal3, "-0x0.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { PVal2, PVal4, "0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { PVal2, MVal4, "0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { PVal2, PVal5, "0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { PVal2, MVal5, "0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { PVal2, PVal6, "0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { PVal2, MVal6, "0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { MVal2, PVal1, "-0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { MVal2, MVal1, "-0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { MVal2, PVal2, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal2, MVal2, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal2, PVal3, "0x0.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { MVal2, MVal3, "0x0.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { MVal2, PVal4, "-0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { MVal2, MVal4, "-0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { MVal2, PVal5, "-0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { MVal2, MVal5, "-0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { MVal2, PVal6, "-0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { MVal2, MVal6, "-0x1.fffffep-126", APFloat::opOK, APFloat::fcNormal },
    { PVal3, PVal1, "0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { PVal3, MVal1, "0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { PVal3, PVal2, "0x0.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { PVal3, MVal2, "0x0.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { PVal3, PVal3, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal3, MVal3, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal3, PVal4, "0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { PVal3, MVal4, "0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { PVal3, PVal5, "0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { PVal3, MVal5, "0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { PVal3, PVal6, "0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { PVal3, MVal6, "0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { MVal3, PVal1, "-0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { MVal3, MVal1, "-0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { MVal3, PVal2, "-0x0.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { MVal3, MVal2, "-0x0.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { MVal3, PVal3, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal3, MVal3, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal3, PVal4, "-0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { MVal3, MVal4, "-0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { MVal3, PVal5, "-0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { MVal3, MVal5, "-0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { MVal3, PVal6, "-0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { MVal3, MVal6, "-0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { PVal4, PVal1, "0x1p+103", APFloat::opOK, APFloat::fcNormal },
    { PVal4, MVal1, "0x1p+103", APFloat::opOK, APFloat::fcNormal },
    { PVal4, PVal2, "0x0.002p-126", APFloat::opOK, APFloat::fcNormal },
    { PVal4, MVal2, "0x0.002p-126", APFloat::opOK, APFloat::fcNormal },
    { PVal4, PVal3, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal4, MVal3, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal4, PVal4, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal4, MVal4, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal4, PVal5, "0.5", APFloat::opOK, APFloat::fcNormal },
    { PVal4, MVal5, "0.5", APFloat::opOK, APFloat::fcNormal },
    { PVal4, PVal6, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal4, MVal6, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal4, PVal1, "-0x1p+103", APFloat::opOK, APFloat::fcNormal },
    { MVal4, MVal1, "-0x1p+103", APFloat::opOK, APFloat::fcNormal },
    { MVal4, PVal2, "-0x0.002p-126", APFloat::opOK, APFloat::fcNormal },
    { MVal4, MVal2, "-0x0.002p-126", APFloat::opOK, APFloat::fcNormal },
    { MVal4, PVal3, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal4, MVal3, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal4, PVal4, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal4, MVal4, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal4, PVal5, "-0.5", APFloat::opOK, APFloat::fcNormal },
    { MVal4, MVal5, "-0.5", APFloat::opOK, APFloat::fcNormal },
    { MVal4, PVal6, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal4, MVal6, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal5, PVal1, "1.5", APFloat::opOK, APFloat::fcNormal },
    { PVal5, MVal1, "1.5", APFloat::opOK, APFloat::fcNormal },
    { PVal5, PVal2, "0x0.00006p-126", APFloat::opOK, APFloat::fcNormal },
    { PVal5, MVal2, "0x0.00006p-126", APFloat::opOK, APFloat::fcNormal },
    { PVal5, PVal3, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal5, MVal3, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal5, PVal4, "1.5", APFloat::opOK, APFloat::fcNormal },
    { PVal5, MVal4, "1.5", APFloat::opOK, APFloat::fcNormal },
    { PVal5, PVal5, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal5, MVal5, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal5, PVal6, "-0.5", APFloat::opOK, APFloat::fcNormal },
    { PVal5, MVal6, "-0.5", APFloat::opOK, APFloat::fcNormal },
    { MVal5, PVal1, "-1.5", APFloat::opOK, APFloat::fcNormal },
    { MVal5, MVal1, "-1.5", APFloat::opOK, APFloat::fcNormal },
    { MVal5, PVal2, "-0x0.00006p-126", APFloat::opOK, APFloat::fcNormal },
    { MVal5, MVal2, "-0x0.00006p-126", APFloat::opOK, APFloat::fcNormal },
    { MVal5, PVal3, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal5, MVal3, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal5, PVal4, "-1.5", APFloat::opOK, APFloat::fcNormal },
    { MVal5, MVal4, "-1.5", APFloat::opOK, APFloat::fcNormal },
    { MVal5, PVal5, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal5, MVal5, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal5, PVal6, "0.5", APFloat::opOK, APFloat::fcNormal },
    { MVal5, MVal6, "0.5", APFloat::opOK, APFloat::fcNormal },
    { PVal6, PVal1, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PVal6, MVal1, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PVal6, PVal2, "0x0.00004p-126", APFloat::opOK, APFloat::fcNormal },
    { PVal6, MVal2, "0x0.00004p-126", APFloat::opOK, APFloat::fcNormal },
    { PVal6, PVal3, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal6, MVal3, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal6, PVal4, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PVal6, MVal4, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PVal6, PVal5, "-0.5", APFloat::opOK, APFloat::fcNormal },
    { PVal6, MVal5, "-0.5", APFloat::opOK, APFloat::fcNormal },
    { PVal6, PVal6, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PVal6, MVal6, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal6, PVal1, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MVal6, MVal1, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MVal6, PVal2, "-0x0.00004p-126", APFloat::opOK, APFloat::fcNormal },
    { MVal6, MVal2, "-0x0.00004p-126", APFloat::opOK, APFloat::fcNormal },
    { MVal6, PVal3, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal6, MVal3, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal6, PVal4, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MVal6, MVal4, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MVal6, PVal5, "0.5", APFloat::opOK, APFloat::fcNormal },
    { MVal6, MVal5, "0.5", APFloat::opOK, APFloat::fcNormal },
    { MVal6, PVal6, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MVal6, MVal6, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
  };

  for (size_t i = 0; i < std::size(SpecialCaseTests); ++i) {
    APFloat x(SpecialCaseTests[i].x);
    APFloat y(SpecialCaseTests[i].y);
    APFloat::opStatus status = x.remainder(y);

    APFloat result(x.getSemantics(), SpecialCaseTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_EQ(SpecialCaseTests[i].status, (int)status);
    EXPECT_EQ(SpecialCaseTests[i].category, (int)x.getCategory());
  }

  {
    APFloat f1(APFloat::IEEEdouble(), "0x1.3333333333333p-2"); // 0.3
    APFloat f2(APFloat::IEEEdouble(), "0x1.47ae147ae147bp-7"); // 0.01
    APFloat expected(APFloat::IEEEdouble(), "-0x1.4p-56");
    EXPECT_EQ(APFloat::opOK, f1.remainder(f2));
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "0x1p64"); // 1.8446744073709552e19
    APFloat f2(APFloat::IEEEdouble(), "1.5");
    APFloat expected(APFloat::IEEEdouble(), "-0.5");
    EXPECT_EQ(APFloat::opOK, f1.remainder(f2));
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "0x1p1000");
    APFloat f2(APFloat::IEEEdouble(), "0x1p-1000");
    APFloat expected(APFloat::IEEEdouble(), "0.0");
    EXPECT_EQ(APFloat::opOK, f1.remainder(f2));
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    APFloat f1 = APFloat::getInf(APFloat::IEEEdouble(), false);
    APFloat f2(APFloat::IEEEdouble(), "1.0");
    EXPECT_EQ(f1.remainder(f2), APFloat::opInvalidOp);
    EXPECT_TRUE(f1.isNaN());
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "-4.0");
    APFloat f2(APFloat::IEEEdouble(), "-2.0");
    APFloat expected(APFloat::IEEEdouble(), "-0.0");
    EXPECT_EQ(APFloat::opOK, f1.remainder(f2));
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
  {
    APFloat f1(APFloat::IEEEdouble(), "-4.0");
    APFloat f2(APFloat::IEEEdouble(), "2.0");
    APFloat expected(APFloat::IEEEdouble(), "-0.0");
    EXPECT_EQ(APFloat::opOK, f1.remainder(f2));
    EXPECT_TRUE(f1.bitwiseIsEqual(expected));
  }
}

TEST(APFloatTest, PPCDoubleDoubleAddSpecial) {
  using DataType = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t,
                              APFloat::fltCategory, APFloat::roundingMode>;
  DataType Data[] = {
      // (1 + 0) + (-1 + 0) = fcZero
      std::make_tuple(0x3ff0000000000000ull, 0, 0xbff0000000000000ull, 0,
                      APFloat::fcZero, APFloat::rmNearestTiesToEven),
      // LDBL_MAX + (1.1 >> (1023 - 106) + 0)) = fcInfinity
      std::make_tuple(0x7fefffffffffffffull, 0x7c8ffffffffffffeull,
                      0x7948000000000000ull, 0ull, APFloat::fcInfinity,
                      APFloat::rmNearestTiesToEven),
      // TODO: change the 4th 0x75effffffffffffe to 0x75efffffffffffff when
      // semPPCDoubleDoubleLegacy is gone.
      // LDBL_MAX + (1.011111... >> (1023 - 106) + (1.1111111...0 >> (1023 -
      // 160))) = fcNormal
      std::make_tuple(0x7fefffffffffffffull, 0x7c8ffffffffffffeull,
                      0x7947ffffffffffffull, 0x75effffffffffffeull,
                      APFloat::fcNormal, APFloat::rmNearestTiesToEven),
      // LDBL_MAX + (1.1 >> (1023 - 106) + 0)) = fcInfinity
      std::make_tuple(0x7fefffffffffffffull, 0x7c8ffffffffffffeull,
                      0x7fefffffffffffffull, 0x7c8ffffffffffffeull,
                      APFloat::fcInfinity, APFloat::rmNearestTiesToEven),
      // NaN + (1 + 0) = fcNaN
      std::make_tuple(0x7ff8000000000000ull, 0, 0x3ff0000000000000ull, 0,
                      APFloat::fcNaN, APFloat::rmNearestTiesToEven),
  };

  for (auto Tp : Data) {
    uint64_t Op1[2], Op2[2];
    APFloat::fltCategory Expected;
    APFloat::roundingMode RM;
    std::tie(Op1[0], Op1[1], Op2[0], Op2[1], Expected, RM) = Tp;

    {
      APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
      APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
      A1.add(A2, RM);

      EXPECT_EQ(Expected, A1.getCategory())
          << formatv("({0:x} + {1:x}) + ({2:x} + {3:x})", Op1[0], Op1[1],
                     Op2[0], Op2[1])
                 .str();
    }
    {
      APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
      APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
      A2.add(A1, RM);

      EXPECT_EQ(Expected, A2.getCategory())
          << formatv("({0:x} + {1:x}) + ({2:x} + {3:x})", Op2[0], Op2[1],
                     Op1[0], Op1[1])
                 .str();
    }
  }
}

TEST(APFloatTest, PPCDoubleDoubleAdd) {
  using DataType = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                              uint64_t, APFloat::roundingMode>;
  DataType Data[] = {
      // (1 + 0) + (1e-105 + 0) = (1 + 1e-105)
      std::make_tuple(0x3ff0000000000000ull, 0, 0x3960000000000000ull, 0,
                      0x3ff0000000000000ull, 0x3960000000000000ull,
                      APFloat::rmNearestTiesToEven),
      // (1 + 0) + (1e-106 + 0) = (1 + 1e-106)
      std::make_tuple(0x3ff0000000000000ull, 0, 0x3950000000000000ull, 0,
                      0x3ff0000000000000ull, 0x3950000000000000ull,
                      APFloat::rmNearestTiesToEven),
      // (1 + 1e-106) + (1e-106 + 0) = (1 + 1e-105)
      std::make_tuple(0x3ff0000000000000ull, 0x3950000000000000ull,
                      0x3950000000000000ull, 0, 0x3ff0000000000000ull,
                      0x3960000000000000ull, APFloat::rmNearestTiesToEven),
      // (1 + 0) + (epsilon + 0) = (1 + epsilon)
      std::make_tuple(0x3ff0000000000000ull, 0, 0x0000000000000001ull, 0,
                      0x3ff0000000000000ull, 0x0000000000000001ull,
                      APFloat::rmNearestTiesToEven),
      // TODO: change 0xf950000000000000 to 0xf940000000000000, when
      // semPPCDoubleDoubleLegacy is gone.
      // (DBL_MAX - 1 << (1023 - 105)) + (1 << (1023 - 53) + 0) = DBL_MAX +
      // 1.11111... << (1023 - 52)
      std::make_tuple(0x7fefffffffffffffull, 0xf950000000000000ull,
                      0x7c90000000000000ull, 0, 0x7fefffffffffffffull,
                      0x7c8ffffffffffffeull, APFloat::rmNearestTiesToEven),
      // TODO: change 0xf950000000000000 to 0xf940000000000000, when
      // semPPCDoubleDoubleLegacy is gone.
      // (1 << (1023 - 53) + 0) + (DBL_MAX - 1 << (1023 - 105)) = DBL_MAX +
      // 1.11111... << (1023 - 52)
      std::make_tuple(0x7c90000000000000ull, 0, 0x7fefffffffffffffull,
                      0xf950000000000000ull, 0x7fefffffffffffffull,
                      0x7c8ffffffffffffeull, APFloat::rmNearestTiesToEven),
  };

  for (auto Tp : Data) {
    uint64_t Op1[2], Op2[2], Expected[2];
    APFloat::roundingMode RM;
    std::tie(Op1[0], Op1[1], Op2[0], Op2[1], Expected[0], Expected[1], RM) = Tp;

    {
      APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
      APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
      A1.add(A2, RM);

      EXPECT_EQ(Expected[0], A1.bitcastToAPInt().getRawData()[0])
          << formatv("({0:x} + {1:x}) + ({2:x} + {3:x})", Op1[0], Op1[1],
                     Op2[0], Op2[1])
                 .str();
      EXPECT_EQ(Expected[1], A1.bitcastToAPInt().getRawData()[1])
          << formatv("({0:x} + {1:x}) + ({2:x} + {3:x})", Op1[0], Op1[1],
                     Op2[0], Op2[1])
                 .str();
    }
    {
      APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
      APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
      A2.add(A1, RM);

      EXPECT_EQ(Expected[0], A2.bitcastToAPInt().getRawData()[0])
          << formatv("({0:x} + {1:x}) + ({2:x} + {3:x})", Op2[0], Op2[1],
                     Op1[0], Op1[1])
                 .str();
      EXPECT_EQ(Expected[1], A2.bitcastToAPInt().getRawData()[1])
          << formatv("({0:x} + {1:x}) + ({2:x} + {3:x})", Op2[0], Op2[1],
                     Op1[0], Op1[1])
                 .str();
    }
  }
}

TEST(APFloatTest, PPCDoubleDoubleSubtract) {
  using DataType = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                              uint64_t, APFloat::roundingMode>;
  DataType Data[] = {
      // (1 + 0) - (-1e-105 + 0) = (1 + 1e-105)
      std::make_tuple(0x3ff0000000000000ull, 0, 0xb960000000000000ull, 0,
                      0x3ff0000000000000ull, 0x3960000000000000ull,
                      APFloat::rmNearestTiesToEven),
      // (1 + 0) - (-1e-106 + 0) = (1 + 1e-106)
      std::make_tuple(0x3ff0000000000000ull, 0, 0xb950000000000000ull, 0,
                      0x3ff0000000000000ull, 0x3950000000000000ull,
                      APFloat::rmNearestTiesToEven),
  };

  for (auto Tp : Data) {
    uint64_t Op1[2], Op2[2], Expected[2];
    APFloat::roundingMode RM;
    std::tie(Op1[0], Op1[1], Op2[0], Op2[1], Expected[0], Expected[1], RM) = Tp;

    APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
    APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
    A1.subtract(A2, RM);

    EXPECT_EQ(Expected[0], A1.bitcastToAPInt().getRawData()[0])
        << formatv("({0:x} + {1:x}) - ({2:x} + {3:x})", Op1[0], Op1[1], Op2[0],
                   Op2[1])
               .str();
    EXPECT_EQ(Expected[1], A1.bitcastToAPInt().getRawData()[1])
        << formatv("({0:x} + {1:x}) - ({2:x} + {3:x})", Op1[0], Op1[1], Op2[0],
                   Op2[1])
               .str();
  }
}

TEST(APFloatTest, PPCDoubleDoubleMultiplySpecial) {
  using DataType = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t,
                              APFloat::fltCategory, APFloat::roundingMode>;
  DataType Data[] = {
      // fcNaN * fcNaN = fcNaN
      std::make_tuple(0x7ff8000000000000ull, 0, 0x7ff8000000000000ull, 0,
                      APFloat::fcNaN, APFloat::rmNearestTiesToEven),
      // fcNaN * fcZero = fcNaN
      std::make_tuple(0x7ff8000000000000ull, 0, 0, 0, APFloat::fcNaN,
                      APFloat::rmNearestTiesToEven),
      // fcNaN * fcInfinity = fcNaN
      std::make_tuple(0x7ff8000000000000ull, 0, 0x7ff0000000000000ull, 0,
                      APFloat::fcNaN, APFloat::rmNearestTiesToEven),
      // fcNaN * fcNormal = fcNaN
      std::make_tuple(0x7ff8000000000000ull, 0, 0x3ff0000000000000ull, 0,
                      APFloat::fcNaN, APFloat::rmNearestTiesToEven),
      // fcInfinity * fcInfinity = fcInfinity
      std::make_tuple(0x7ff0000000000000ull, 0, 0x7ff0000000000000ull, 0,
                      APFloat::fcInfinity, APFloat::rmNearestTiesToEven),
      // fcInfinity * fcZero = fcNaN
      std::make_tuple(0x7ff0000000000000ull, 0, 0, 0, APFloat::fcNaN,
                      APFloat::rmNearestTiesToEven),
      // fcInfinity * fcNormal = fcInfinity
      std::make_tuple(0x7ff0000000000000ull, 0, 0x3ff0000000000000ull, 0,
                      APFloat::fcInfinity, APFloat::rmNearestTiesToEven),
      // fcZero * fcZero = fcZero
      std::make_tuple(0, 0, 0, 0, APFloat::fcZero,
                      APFloat::rmNearestTiesToEven),
      // fcZero * fcNormal = fcZero
      std::make_tuple(0, 0, 0x3ff0000000000000ull, 0, APFloat::fcZero,
                      APFloat::rmNearestTiesToEven),
  };

  for (auto Tp : Data) {
    uint64_t Op1[2], Op2[2];
    APFloat::fltCategory Expected;
    APFloat::roundingMode RM;
    std::tie(Op1[0], Op1[1], Op2[0], Op2[1], Expected, RM) = Tp;

    {
      APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
      APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
      A1.multiply(A2, RM);

      EXPECT_EQ(Expected, A1.getCategory())
          << formatv("({0:x} + {1:x}) * ({2:x} + {3:x})", Op1[0], Op1[1],
                     Op2[0], Op2[1])
                 .str();
    }
    {
      APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
      APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
      A2.multiply(A1, RM);

      EXPECT_EQ(Expected, A2.getCategory())
          << formatv("({0:x} + {1:x}) * ({2:x} + {3:x})", Op2[0], Op2[1],
                     Op1[0], Op1[1])
                 .str();
    }
  }
}

TEST(APFloatTest, PPCDoubleDoubleMultiply) {
  using DataType = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                              uint64_t, APFloat::roundingMode>;
  DataType Data[] = {
      // 1/3 * 3 = 1.0
      std::make_tuple(0x3fd5555555555555ull, 0x3c75555555555556ull,
                      0x4008000000000000ull, 0, 0x3ff0000000000000ull, 0,
                      APFloat::rmNearestTiesToEven),
      // (1 + epsilon) * (1 + 0) = fcZero
      std::make_tuple(0x3ff0000000000000ull, 0x0000000000000001ull,
                      0x3ff0000000000000ull, 0, 0x3ff0000000000000ull,
                      0x0000000000000001ull, APFloat::rmNearestTiesToEven),
      // (1 + epsilon) * (1 + epsilon) = 1 + 2 * epsilon
      std::make_tuple(0x3ff0000000000000ull, 0x0000000000000001ull,
                      0x3ff0000000000000ull, 0x0000000000000001ull,
                      0x3ff0000000000000ull, 0x0000000000000002ull,
                      APFloat::rmNearestTiesToEven),
      // -(1 + epsilon) * (1 + epsilon) = -1
      std::make_tuple(0xbff0000000000000ull, 0x0000000000000001ull,
                      0x3ff0000000000000ull, 0x0000000000000001ull,
                      0xbff0000000000000ull, 0, APFloat::rmNearestTiesToEven),
      // (0.5 + 0) * (1 + 2 * epsilon) = 0.5 + epsilon
      std::make_tuple(0x3fe0000000000000ull, 0, 0x3ff0000000000000ull,
                      0x0000000000000002ull, 0x3fe0000000000000ull,
                      0x0000000000000001ull, APFloat::rmNearestTiesToEven),
      // (0.5 + 0) * (1 + epsilon) = 0.5
      std::make_tuple(0x3fe0000000000000ull, 0, 0x3ff0000000000000ull,
                      0x0000000000000001ull, 0x3fe0000000000000ull, 0,
                      APFloat::rmNearestTiesToEven),
      // __LDBL_MAX__ * (1 + 1 << 106) = inf
      std::make_tuple(0x7fefffffffffffffull, 0x7c8ffffffffffffeull,
                      0x3ff0000000000000ull, 0x3950000000000000ull,
                      0x7ff0000000000000ull, 0, APFloat::rmNearestTiesToEven),
      // __LDBL_MAX__ * (1 + 1 << 107) > __LDBL_MAX__, but not inf, yes =_=|||
      std::make_tuple(0x7fefffffffffffffull, 0x7c8ffffffffffffeull,
                      0x3ff0000000000000ull, 0x3940000000000000ull,
                      0x7fefffffffffffffull, 0x7c8fffffffffffffull,
                      APFloat::rmNearestTiesToEven),
      // __LDBL_MAX__ * (1 + 1 << 108) = __LDBL_MAX__
      std::make_tuple(0x7fefffffffffffffull, 0x7c8ffffffffffffeull,
                      0x3ff0000000000000ull, 0x3930000000000000ull,
                      0x7fefffffffffffffull, 0x7c8ffffffffffffeull,
                      APFloat::rmNearestTiesToEven),
  };

  for (auto Tp : Data) {
    uint64_t Op1[2], Op2[2], Expected[2];
    APFloat::roundingMode RM;
    std::tie(Op1[0], Op1[1], Op2[0], Op2[1], Expected[0], Expected[1], RM) = Tp;

    {
      APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
      APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
      A1.multiply(A2, RM);

      EXPECT_EQ(Expected[0], A1.bitcastToAPInt().getRawData()[0])
          << formatv("({0:x} + {1:x}) * ({2:x} + {3:x})", Op1[0], Op1[1],
                     Op2[0], Op2[1])
                 .str();
      EXPECT_EQ(Expected[1], A1.bitcastToAPInt().getRawData()[1])
          << formatv("({0:x} + {1:x}) * ({2:x} + {3:x})", Op1[0], Op1[1],
                     Op2[0], Op2[1])
                 .str();
    }
    {
      APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
      APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
      A2.multiply(A1, RM);

      EXPECT_EQ(Expected[0], A2.bitcastToAPInt().getRawData()[0])
          << formatv("({0:x} + {1:x}) * ({2:x} + {3:x})", Op2[0], Op2[1],
                     Op1[0], Op1[1])
                 .str();
      EXPECT_EQ(Expected[1], A2.bitcastToAPInt().getRawData()[1])
          << formatv("({0:x} + {1:x}) * ({2:x} + {3:x})", Op2[0], Op2[1],
                     Op1[0], Op1[1])
                 .str();
    }
  }
}

TEST(APFloatTest, PPCDoubleDoubleDivide) {
  using DataType = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                              uint64_t, APFloat::roundingMode>;
  // TODO: Only a sanity check for now. Add more edge cases when the
  // double-double algorithm is implemented.
  DataType Data[] = {
      // 1 / 3 = 1/3
      std::make_tuple(0x3ff0000000000000ull, 0, 0x4008000000000000ull, 0,
                      0x3fd5555555555555ull, 0x3c75555555555556ull,
                      APFloat::rmNearestTiesToEven),
  };

  for (auto Tp : Data) {
    uint64_t Op1[2], Op2[2], Expected[2];
    APFloat::roundingMode RM;
    std::tie(Op1[0], Op1[1], Op2[0], Op2[1], Expected[0], Expected[1], RM) = Tp;

    APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
    APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
    A1.divide(A2, RM);

    EXPECT_EQ(Expected[0], A1.bitcastToAPInt().getRawData()[0])
        << formatv("({0:x} + {1:x}) / ({2:x} + {3:x})", Op1[0], Op1[1], Op2[0],
                   Op2[1])
               .str();
    EXPECT_EQ(Expected[1], A1.bitcastToAPInt().getRawData()[1])
        << formatv("({0:x} + {1:x}) / ({2:x} + {3:x})", Op1[0], Op1[1], Op2[0],
                   Op2[1])
               .str();
  }
}

TEST(APFloatTest, PPCDoubleDoubleRemainder) {
  using DataType =
      std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>;
  DataType Data[] = {
      // remainder(3.0 + 3.0 << 53, 1.25 + 1.25 << 53) = (0.5 + 0.5 << 53)
      std::make_tuple(0x4008000000000000ull, 0x3cb8000000000000ull,
                      0x3ff4000000000000ull, 0x3ca4000000000000ull,
                      0x3fe0000000000000ull, 0x3c90000000000000ull),
      // remainder(3.0 + 3.0 << 53, 1.75 + 1.75 << 53) = (-0.5 - 0.5 << 53)
      std::make_tuple(0x4008000000000000ull, 0x3cb8000000000000ull,
                      0x3ffc000000000000ull, 0x3cac000000000000ull,
                      0xbfe0000000000000ull, 0xbc90000000000000ull),
  };

  for (auto Tp : Data) {
    uint64_t Op1[2], Op2[2], Expected[2];
    std::tie(Op1[0], Op1[1], Op2[0], Op2[1], Expected[0], Expected[1]) = Tp;

    APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
    APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
    A1.remainder(A2);

    EXPECT_EQ(Expected[0], A1.bitcastToAPInt().getRawData()[0])
        << formatv("remainder({0:x} + {1:x}), ({2:x} + {3:x}))", Op1[0], Op1[1],
                   Op2[0], Op2[1])
               .str();
    EXPECT_EQ(Expected[1], A1.bitcastToAPInt().getRawData()[1])
        << formatv("remainder(({0:x} + {1:x}), ({2:x} + {3:x}))", Op1[0],
                   Op1[1], Op2[0], Op2[1])
               .str();
  }
}

TEST(APFloatTest, PPCDoubleDoubleMod) {
  using DataType =
      std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>;
  DataType Data[] = {
      // mod(3.0 + 3.0 << 53, 1.25 + 1.25 << 53) = (0.5 + 0.5 << 53)
      std::make_tuple(0x4008000000000000ull, 0x3cb8000000000000ull,
                      0x3ff4000000000000ull, 0x3ca4000000000000ull,
                      0x3fe0000000000000ull, 0x3c90000000000000ull),
      // mod(3.0 + 3.0 << 53, 1.75 + 1.75 << 53) = (1.25 + 1.25 << 53)
      // 0xbc98000000000000 doesn't seem right, but it's what we currently have.
      // TODO: investigate
      std::make_tuple(0x4008000000000000ull, 0x3cb8000000000000ull,
                      0x3ffc000000000000ull, 0x3cac000000000000ull,
                      0x3ff4000000000001ull, 0xbc98000000000000ull),
  };

  for (auto Tp : Data) {
    uint64_t Op1[2], Op2[2], Expected[2];
    std::tie(Op1[0], Op1[1], Op2[0], Op2[1], Expected[0], Expected[1]) = Tp;

    APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
    APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
    A1.mod(A2);

    EXPECT_EQ(Expected[0], A1.bitcastToAPInt().getRawData()[0])
        << formatv("fmod(({0:x} + {1:x}),  ({2:x} + {3:x}))", Op1[0], Op1[1],
                   Op2[0], Op2[1])
               .str();
    EXPECT_EQ(Expected[1], A1.bitcastToAPInt().getRawData()[1])
        << formatv("fmod(({0:x} + {1:x}), ({2:x} + {3:x}))", Op1[0], Op1[1],
                   Op2[0], Op2[1])
               .str();
  }
}

TEST(APFloatTest, PPCDoubleDoubleFMA) {
  // Sanity check for now.
  APFloat A(APFloat::PPCDoubleDouble(), "2");
  A.fusedMultiplyAdd(APFloat(APFloat::PPCDoubleDouble(), "3"),
                     APFloat(APFloat::PPCDoubleDouble(), "4"),
                     APFloat::rmNearestTiesToEven);
  EXPECT_EQ(APFloat::cmpEqual,
            APFloat(APFloat::PPCDoubleDouble(), "10").compare(A));
}

TEST(APFloatTest, PPCDoubleDoubleRoundToIntegral) {
  {
    APFloat A(APFloat::PPCDoubleDouble(), "1.5");
    A.roundToIntegral(APFloat::rmNearestTiesToEven);
    EXPECT_EQ(APFloat::cmpEqual,
              APFloat(APFloat::PPCDoubleDouble(), "2").compare(A));
  }
  {
    APFloat A(APFloat::PPCDoubleDouble(), "2.5");
    A.roundToIntegral(APFloat::rmNearestTiesToEven);
    EXPECT_EQ(APFloat::cmpEqual,
              APFloat(APFloat::PPCDoubleDouble(), "2").compare(A));
  }
}

TEST(APFloatTest, PPCDoubleDoubleCompare) {
  using DataType =
      std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, APFloat::cmpResult>;

  DataType Data[] = {
      // (1 + 0) = (1 + 0)
      std::make_tuple(0x3ff0000000000000ull, 0, 0x3ff0000000000000ull, 0,
                      APFloat::cmpEqual),
      // (1 + 0) < (1.00...1 + 0)
      std::make_tuple(0x3ff0000000000000ull, 0, 0x3ff0000000000001ull, 0,
                      APFloat::cmpLessThan),
      // (1.00...1 + 0) > (1 + 0)
      std::make_tuple(0x3ff0000000000001ull, 0, 0x3ff0000000000000ull, 0,
                      APFloat::cmpGreaterThan),
      // (1 + 0) < (1 + epsilon)
      std::make_tuple(0x3ff0000000000000ull, 0, 0x3ff0000000000001ull,
                      0x0000000000000001ull, APFloat::cmpLessThan),
      // NaN != NaN
      std::make_tuple(0x7ff8000000000000ull, 0, 0x7ff8000000000000ull, 0,
                      APFloat::cmpUnordered),
      // (1 + 0) != NaN
      std::make_tuple(0x3ff0000000000000ull, 0, 0x7ff8000000000000ull, 0,
                      APFloat::cmpUnordered),
      // Inf = Inf
      std::make_tuple(0x7ff0000000000000ull, 0, 0x7ff0000000000000ull, 0,
                      APFloat::cmpEqual),
  };

  for (auto Tp : Data) {
    uint64_t Op1[2], Op2[2];
    APFloat::cmpResult Expected;
    std::tie(Op1[0], Op1[1], Op2[0], Op2[1], Expected) = Tp;

    APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
    APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
    EXPECT_EQ(Expected, A1.compare(A2))
        << formatv("compare(({0:x} + {1:x}), ({2:x} + {3:x}))", Op1[0], Op1[1],
                   Op2[0], Op2[1])
               .str();
  }
}

TEST(APFloatTest, PPCDoubleDoubleBitwiseIsEqual) {
  using DataType = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, bool>;

  DataType Data[] = {
      // (1 + 0) = (1 + 0)
      std::make_tuple(0x3ff0000000000000ull, 0, 0x3ff0000000000000ull, 0, true),
      // (1 + 0) != (1.00...1 + 0)
      std::make_tuple(0x3ff0000000000000ull, 0, 0x3ff0000000000001ull, 0,
                      false),
      // NaN = NaN
      std::make_tuple(0x7ff8000000000000ull, 0, 0x7ff8000000000000ull, 0, true),
      // NaN != NaN with a different bit pattern
      std::make_tuple(0x7ff8000000000000ull, 0, 0x7ff8000000000000ull,
                      0x3ff0000000000000ull, false),
      // Inf = Inf
      std::make_tuple(0x7ff0000000000000ull, 0, 0x7ff0000000000000ull, 0, true),
  };

  for (auto Tp : Data) {
    uint64_t Op1[2], Op2[2];
    bool Expected;
    std::tie(Op1[0], Op1[1], Op2[0], Op2[1], Expected) = Tp;

    APFloat A1(APFloat::PPCDoubleDouble(), APInt(128, 2, Op1));
    APFloat A2(APFloat::PPCDoubleDouble(), APInt(128, 2, Op2));
    EXPECT_EQ(Expected, A1.bitwiseIsEqual(A2))
        << formatv("({0:x} + {1:x}) = ({2:x} + {3:x})", Op1[0], Op1[1], Op2[0],
                   Op2[1])
               .str();
  }
}

TEST(APFloatTest, PPCDoubleDoubleHashValue) {
  uint64_t Data1[] = {0x3ff0000000000001ull, 0x0000000000000001ull};
  uint64_t Data2[] = {0x3ff0000000000001ull, 0};
  // The hash values are *hopefully* different.
  EXPECT_NE(
      hash_value(APFloat(APFloat::PPCDoubleDouble(), APInt(128, 2, Data1))),
      hash_value(APFloat(APFloat::PPCDoubleDouble(), APInt(128, 2, Data2))));
}

TEST(APFloatTest, PPCDoubleDoubleChangeSign) {
  uint64_t Data[] = {
      0x400f000000000000ull, 0xbcb0000000000000ull,
  };
  APFloat Float(APFloat::PPCDoubleDouble(), APInt(128, 2, Data));
  {
    APFloat Actual =
        APFloat::copySign(Float, APFloat(APFloat::IEEEdouble(), "1"));
    EXPECT_EQ(0x400f000000000000ull, Actual.bitcastToAPInt().getRawData()[0]);
    EXPECT_EQ(0xbcb0000000000000ull, Actual.bitcastToAPInt().getRawData()[1]);
  }
  {
    APFloat Actual =
        APFloat::copySign(Float, APFloat(APFloat::IEEEdouble(), "-1"));
    EXPECT_EQ(0xc00f000000000000ull, Actual.bitcastToAPInt().getRawData()[0]);
    EXPECT_EQ(0x3cb0000000000000ull, Actual.bitcastToAPInt().getRawData()[1]);
  }
}

TEST(APFloatTest, PPCDoubleDoubleFactories) {
  {
    uint64_t Data[] = {
        0, 0,
    };
    EXPECT_EQ(APInt(128, 2, Data),
              APFloat::getZero(APFloat::PPCDoubleDouble()).bitcastToAPInt());
  }
  {
    uint64_t Data[] = {
        0x7fefffffffffffffull, 0x7c8ffffffffffffeull,
    };
    EXPECT_EQ(APInt(128, 2, Data),
              APFloat::getLargest(APFloat::PPCDoubleDouble()).bitcastToAPInt());
  }
  {
    uint64_t Data[] = {
        0x0000000000000001ull, 0,
    };
    EXPECT_EQ(
        APInt(128, 2, Data),
        APFloat::getSmallest(APFloat::PPCDoubleDouble()).bitcastToAPInt());
  }
  {
    uint64_t Data[] = {0x0360000000000000ull, 0};
    EXPECT_EQ(APInt(128, 2, Data),
              APFloat::getSmallestNormalized(APFloat::PPCDoubleDouble())
                  .bitcastToAPInt());
  }
  {
    uint64_t Data[] = {
        0x8000000000000000ull, 0x0000000000000000ull,
    };
    EXPECT_EQ(
        APInt(128, 2, Data),
        APFloat::getZero(APFloat::PPCDoubleDouble(), true).bitcastToAPInt());
  }
  {
    uint64_t Data[] = {
        0xffefffffffffffffull, 0xfc8ffffffffffffeull,
    };
    EXPECT_EQ(
        APInt(128, 2, Data),
        APFloat::getLargest(APFloat::PPCDoubleDouble(), true).bitcastToAPInt());
  }
  {
    uint64_t Data[] = {
        0x8000000000000001ull, 0x0000000000000000ull,
    };
    EXPECT_EQ(APInt(128, 2, Data),
              APFloat::getSmallest(APFloat::PPCDoubleDouble(), true)
                  .bitcastToAPInt());
  }
  {
    uint64_t Data[] = {
        0x8360000000000000ull, 0x0000000000000000ull,
    };
    EXPECT_EQ(APInt(128, 2, Data),
              APFloat::getSmallestNormalized(APFloat::PPCDoubleDouble(), true)
                  .bitcastToAPInt());
  }
  EXPECT_TRUE(APFloat::getSmallest(APFloat::PPCDoubleDouble()).isSmallest());
  EXPECT_TRUE(APFloat::getLargest(APFloat::PPCDoubleDouble()).isLargest());
}

TEST(APFloatTest, PPCDoubleDoubleIsDenormal) {
  EXPECT_TRUE(APFloat::getSmallest(APFloat::PPCDoubleDouble()).isDenormal());
  EXPECT_FALSE(APFloat::getLargest(APFloat::PPCDoubleDouble()).isDenormal());
  EXPECT_FALSE(
      APFloat::getSmallestNormalized(APFloat::PPCDoubleDouble()).isDenormal());
  {
    // (4 + 3) is not normalized
    uint64_t Data[] = {
        0x4010000000000000ull, 0x4008000000000000ull,
    };
    EXPECT_TRUE(
        APFloat(APFloat::PPCDoubleDouble(), APInt(128, 2, Data)).isDenormal());
  }
}

TEST(APFloatTest, PPCDoubleDoubleScalbn) {
  // 3.0 + 3.0 << 53
  uint64_t Input[] = {
      0x4008000000000000ull, 0x3cb8000000000000ull,
  };
  APFloat Result =
      scalbn(APFloat(APFloat::PPCDoubleDouble(), APInt(128, 2, Input)), 1,
             APFloat::rmNearestTiesToEven);
  // 6.0 + 6.0 << 53
  EXPECT_EQ(0x4018000000000000ull, Result.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x3cc8000000000000ull, Result.bitcastToAPInt().getRawData()[1]);
}

TEST(APFloatTest, PPCDoubleDoubleFrexp) {
  // 3.0 + 3.0 << 53
  uint64_t Input[] = {
      0x4008000000000000ull, 0x3cb8000000000000ull,
  };
  int Exp;
  // 0.75 + 0.75 << 53
  APFloat Result =
      frexp(APFloat(APFloat::PPCDoubleDouble(), APInt(128, 2, Input)), Exp,
            APFloat::rmNearestTiesToEven);
  EXPECT_EQ(2, Exp);
  EXPECT_EQ(0x3fe8000000000000ull, Result.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x3c98000000000000ull, Result.bitcastToAPInt().getRawData()[1]);
}

TEST(APFloatTest, x87Largest) {
  APFloat MaxX87Val = APFloat::getLargest(APFloat::x87DoubleExtended());
  EXPECT_TRUE(MaxX87Val.isLargest());
}

TEST(APFloatTest, x87Next) {
  APFloat F(APFloat::x87DoubleExtended(), "-1.0");
  F.next(false);
  EXPECT_TRUE(ilogb(F) == -1);
}

TEST(APFloatTest, Float8ExhaustivePair) {
  // Test each pair of 8-bit floats with non-standard semantics
  for (APFloat::Semantics Sem :
       {APFloat::S_Float8E4M3FN, APFloat::S_Float8E5M2FNUZ,
        APFloat::S_Float8E4M3FNUZ, APFloat::S_Float8E4M3B11FNUZ}) {
    const llvm::fltSemantics &S = APFloat::EnumToSemantics(Sem);
    for (int i = 0; i < 256; i++) {
      for (int j = 0; j < 256; j++) {
        SCOPED_TRACE("sem=" + std::to_string(Sem) + ",i=" + std::to_string(i) +
                     ",j=" + std::to_string(j));
        APFloat x(S, APInt(8, i));
        APFloat y(S, APInt(8, j));

        bool losesInfo;
        APFloat x16 = x;
        x16.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven,
                    &losesInfo);
        EXPECT_FALSE(losesInfo);
        APFloat y16 = y;
        y16.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven,
                    &losesInfo);
        EXPECT_FALSE(losesInfo);

        // Add
        APFloat z = x;
        z.add(y, APFloat::rmNearestTiesToEven);
        APFloat z16 = x16;
        z16.add(y16, APFloat::rmNearestTiesToEven);
        z16.convert(S, APFloat::rmNearestTiesToEven, &losesInfo);
        EXPECT_TRUE(z.bitwiseIsEqual(z16))
            << "sem=" << Sem << ", i=" << i << ", j=" << j;

        // Subtract
        z = x;
        z.subtract(y, APFloat::rmNearestTiesToEven);
        z16 = x16;
        z16.subtract(y16, APFloat::rmNearestTiesToEven);
        z16.convert(S, APFloat::rmNearestTiesToEven, &losesInfo);
        EXPECT_TRUE(z.bitwiseIsEqual(z16))
            << "sem=" << Sem << ", i=" << i << ", j=" << j;

        // Multiply
        z = x;
        z.multiply(y, APFloat::rmNearestTiesToEven);
        z16 = x16;
        z16.multiply(y16, APFloat::rmNearestTiesToEven);
        z16.convert(S, APFloat::rmNearestTiesToEven, &losesInfo);
        EXPECT_TRUE(z.bitwiseIsEqual(z16))
            << "sem=" << Sem << ", i=" << i << ", j=" << j;

        // Divide
        z = x;
        z.divide(y, APFloat::rmNearestTiesToEven);
        z16 = x16;
        z16.divide(y16, APFloat::rmNearestTiesToEven);
        z16.convert(S, APFloat::rmNearestTiesToEven, &losesInfo);
        EXPECT_TRUE(z.bitwiseIsEqual(z16))
            << "sem=" << Sem << ", i=" << i << ", j=" << j;

        // Mod
        z = x;
        z.mod(y);
        z16 = x16;
        z16.mod(y16);
        z16.convert(S, APFloat::rmNearestTiesToEven, &losesInfo);
        EXPECT_TRUE(z.bitwiseIsEqual(z16))
            << "sem=" << Sem << ", i=" << i << ", j=" << j;

        // Remainder
        z = x;
        z.remainder(y);
        z16 = x16;
        z16.remainder(y16);
        z16.convert(S, APFloat::rmNearestTiesToEven, &losesInfo);
        EXPECT_TRUE(z.bitwiseIsEqual(z16))
            << "sem=" << Sem << ", i=" << i << ", j=" << j;
      }
    }
  }
}

TEST(APFloatTest, ConvertE4M3FNToE5M2) {
  bool losesInfo;
  APFloat test(APFloat::Float8E4M3FN(), "1.0");
  APFloat::opStatus status = test.convert(
      APFloat::Float8E5M2(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0f, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  test = APFloat(APFloat::Float8E4M3FN(), "0.0");
  status = test.convert(APFloat::Float8E5M2(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0.0f, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  test = APFloat(APFloat::Float8E4M3FN(), "0x1.2p0"); // 1.125
  status = test.convert(APFloat::Float8E5M2(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.0p0 /* 1.0 */, test.convertToFloat());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opInexact);

  test = APFloat(APFloat::Float8E4M3FN(), "0x1.6p0"); // 1.375
  status = test.convert(APFloat::Float8E5M2(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.8p0 /* 1.5 */, test.convertToFloat());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opInexact);

  // Convert E4M3 denormal to E5M2 normal. Should not be truncated, despite the
  // destination format having one fewer significand bit
  test = APFloat(APFloat::Float8E4M3FN(), "0x1.Cp-7");
  status = test.convert(APFloat::Float8E5M2(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.Cp-7, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  // Test convert from NaN
  test = APFloat(APFloat::Float8E4M3FN(), "nan");
  status = test.convert(APFloat::Float8E5M2(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_TRUE(std::isnan(test.convertToFloat()));
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);
}

TEST(APFloatTest, ConvertE5M2ToE4M3FN) {
  bool losesInfo;
  APFloat test(APFloat::Float8E5M2(), "1.0");
  APFloat::opStatus status = test.convert(
      APFloat::Float8E4M3FN(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0f, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  test = APFloat(APFloat::Float8E5M2(), "0.0");
  status = test.convert(APFloat::Float8E4M3FN(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0.0f, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  test = APFloat(APFloat::Float8E5M2(), "0x1.Cp8"); // 448
  status = test.convert(APFloat::Float8E4M3FN(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.Cp8 /* 448 */, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  // Test overflow
  test = APFloat(APFloat::Float8E5M2(), "0x1.0p9"); // 512
  status = test.convert(APFloat::Float8E4M3FN(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_TRUE(std::isnan(test.convertToFloat()));
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opOverflow | APFloat::opInexact);

  // Test underflow
  test = APFloat(APFloat::Float8E5M2(), "0x1.0p-10");
  status = test.convert(APFloat::Float8E4M3FN(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0., test.convertToFloat());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opUnderflow | APFloat::opInexact);

  // Test rounding up to smallest denormal number
  test = APFloat(APFloat::Float8E5M2(), "0x1.8p-10");
  status = test.convert(APFloat::Float8E4M3FN(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.0p-9, test.convertToFloat());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opUnderflow | APFloat::opInexact);

  // Testing inexact rounding to denormal number
  test = APFloat(APFloat::Float8E5M2(), "0x1.8p-9");
  status = test.convert(APFloat::Float8E4M3FN(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.0p-8, test.convertToFloat());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opUnderflow | APFloat::opInexact);

  APFloat nan = APFloat(APFloat::Float8E4M3FN(), "nan");

  // Testing convert from Inf
  test = APFloat(APFloat::Float8E5M2(), "inf");
  status = test.convert(APFloat::Float8E4M3FN(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_TRUE(std::isnan(test.convertToFloat()));
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opInexact);
  EXPECT_TRUE(test.bitwiseIsEqual(nan));

  // Testing convert from quiet NaN
  test = APFloat(APFloat::Float8E5M2(), "nan");
  status = test.convert(APFloat::Float8E4M3FN(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_TRUE(std::isnan(test.convertToFloat()));
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(nan));

  // Testing convert from signaling NaN
  test = APFloat(APFloat::Float8E5M2(), "snan");
  status = test.convert(APFloat::Float8E4M3FN(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_TRUE(std::isnan(test.convertToFloat()));
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opInvalidOp);
  EXPECT_TRUE(test.bitwiseIsEqual(nan));
}

TEST(APFloatTest, Float8E4M3FNGetInf) {
  APFloat t = APFloat::getInf(APFloat::Float8E4M3FN());
  EXPECT_TRUE(t.isNaN());
  EXPECT_FALSE(t.isInfinity());
}

TEST(APFloatTest, Float8E4M3FNFromString) {
  // Exactly representable
  EXPECT_EQ(448, APFloat(APFloat::Float8E4M3FN(), "448").convertToDouble());
  // Round down to maximum value
  EXPECT_EQ(448, APFloat(APFloat::Float8E4M3FN(), "464").convertToDouble());
  // Round up, causing overflow to NaN
  EXPECT_TRUE(APFloat(APFloat::Float8E4M3FN(), "465").isNaN());
  // Overflow without rounding
  EXPECT_TRUE(APFloat(APFloat::Float8E4M3FN(), "480").isNaN());
  // Inf converted to NaN
  EXPECT_TRUE(APFloat(APFloat::Float8E4M3FN(), "inf").isNaN());
  // NaN converted to NaN
  EXPECT_TRUE(APFloat(APFloat::Float8E4M3FN(), "nan").isNaN());
}

TEST(APFloatTest, Float8E4M3FNAdd) {
  APFloat QNaN = APFloat::getNaN(APFloat::Float8E4M3FN(), false);

  auto FromStr = [](StringRef S) {
    return APFloat(APFloat::Float8E4M3FN(), S);
  };

  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
    APFloat::roundingMode roundingMode = APFloat::rmNearestTiesToEven;
  } AdditionTests[] = {
      // Test addition operations involving NaN, overflow, and the max E4M3
      // value (448) because E4M3 differs from IEEE-754 types in these regards
      {FromStr("448"), FromStr("16"), "448", APFloat::opInexact,
       APFloat::fcNormal},
      {FromStr("448"), FromStr("18"), "NaN",
       APFloat::opOverflow | APFloat::opInexact, APFloat::fcNaN},
      {FromStr("448"), FromStr("32"), "NaN",
       APFloat::opOverflow | APFloat::opInexact, APFloat::fcNaN},
      {FromStr("-448"), FromStr("-32"), "-NaN",
       APFloat::opOverflow | APFloat::opInexact, APFloat::fcNaN},
      {QNaN, FromStr("-448"), "NaN", APFloat::opOK, APFloat::fcNaN},
      {FromStr("448"), FromStr("-32"), "416", APFloat::opOK, APFloat::fcNormal},
      {FromStr("448"), FromStr("0"), "448", APFloat::opOK, APFloat::fcNormal},
      {FromStr("448"), FromStr("32"), "448", APFloat::opInexact,
       APFloat::fcNormal, APFloat::rmTowardZero},
      {FromStr("448"), FromStr("448"), "448", APFloat::opInexact,
       APFloat::fcNormal, APFloat::rmTowardZero},
  };

  for (size_t i = 0; i < std::size(AdditionTests); ++i) {
    APFloat x(AdditionTests[i].x);
    APFloat y(AdditionTests[i].y);
    APFloat::opStatus status = x.add(y, AdditionTests[i].roundingMode);

    APFloat result(APFloat::Float8E4M3FN(), AdditionTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_EQ(AdditionTests[i].status, (int)status);
    EXPECT_EQ(AdditionTests[i].category, (int)x.getCategory());
  }
}

TEST(APFloatTest, Float8E4M3FNDivideByZero) {
  APFloat x(APFloat::Float8E4M3FN(), "1");
  APFloat zero(APFloat::Float8E4M3FN(), "0");
  EXPECT_EQ(x.divide(zero, APFloat::rmNearestTiesToEven), APFloat::opDivByZero);
  EXPECT_TRUE(x.isNaN());
}

TEST(APFloatTest, Float8E4M3FNNext) {
  APFloat test(APFloat::Float8E4M3FN(), APFloat::uninitialized);
  APFloat expected(APFloat::Float8E4M3FN(), APFloat::uninitialized);

  // nextUp on positive numbers
  for (int i = 0; i < 127; i++) {
    test = APFloat(APFloat::Float8E4M3FN(), APInt(8, i));
    expected = APFloat(APFloat::Float8E4M3FN(), APInt(8, i + 1));
    EXPECT_EQ(test.next(false), APFloat::opOK);
    EXPECT_TRUE(test.bitwiseIsEqual(expected));
  }

  // nextUp on negative zero
  test = APFloat::getZero(APFloat::Float8E4M3FN(), true);
  expected = APFloat::getSmallest(APFloat::Float8E4M3FN(), false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp on negative nonzero numbers
  for (int i = 129; i < 255; i++) {
    test = APFloat(APFloat::Float8E4M3FN(), APInt(8, i));
    expected = APFloat(APFloat::Float8E4M3FN(), APInt(8, i - 1));
    EXPECT_EQ(test.next(false), APFloat::opOK);
    EXPECT_TRUE(test.bitwiseIsEqual(expected));
  }

  // nextUp on NaN
  test = APFloat::getQNaN(APFloat::Float8E4M3FN(), false);
  expected = APFloat::getQNaN(APFloat::Float8E4M3FN(), false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown on positive nonzero finite numbers
  for (int i = 1; i < 127; i++) {
    test = APFloat(APFloat::Float8E4M3FN(), APInt(8, i));
    expected = APFloat(APFloat::Float8E4M3FN(), APInt(8, i - 1));
    EXPECT_EQ(test.next(true), APFloat::opOK);
    EXPECT_TRUE(test.bitwiseIsEqual(expected));
  }

  // nextDown on positive zero
  test = APFloat::getZero(APFloat::Float8E4M3FN(), true);
  expected = APFloat::getSmallest(APFloat::Float8E4M3FN(), true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown on negative finite numbers
  for (int i = 128; i < 255; i++) {
    test = APFloat(APFloat::Float8E4M3FN(), APInt(8, i));
    expected = APFloat(APFloat::Float8E4M3FN(), APInt(8, i + 1));
    EXPECT_EQ(test.next(true), APFloat::opOK);
    EXPECT_TRUE(test.bitwiseIsEqual(expected));
  }

  // nextDown on NaN
  test = APFloat::getQNaN(APFloat::Float8E4M3FN(), false);
  expected = APFloat::getQNaN(APFloat::Float8E4M3FN(), false);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
}

TEST(APFloatTest, Float8E4M3FNExhaustive) {
  // Test each of the 256 Float8E4M3FN values.
  for (int i = 0; i < 256; i++) {
    APFloat test(APFloat::Float8E4M3FN(), APInt(8, i));
    SCOPED_TRACE("i=" + std::to_string(i));

    // isLargest
    if (i == 126 || i == 254) {
      EXPECT_TRUE(test.isLargest());
      EXPECT_EQ(abs(test).convertToDouble(), 448.);
    } else {
      EXPECT_FALSE(test.isLargest());
    }

    // isSmallest
    if (i == 1 || i == 129) {
      EXPECT_TRUE(test.isSmallest());
      EXPECT_EQ(abs(test).convertToDouble(), 0x1p-9);
    } else {
      EXPECT_FALSE(test.isSmallest());
    }

    // convert to BFloat
    APFloat test2 = test;
    bool losesInfo;
    APFloat::opStatus status = test2.convert(
        APFloat::BFloat(), APFloat::rmNearestTiesToEven, &losesInfo);
    EXPECT_EQ(status, APFloat::opOK);
    EXPECT_FALSE(losesInfo);
    if (i == 127 || i == 255)
      EXPECT_TRUE(test2.isNaN());
    else
      EXPECT_EQ(test.convertToFloat(), test2.convertToFloat());

    // bitcastToAPInt
    EXPECT_EQ(i, test.bitcastToAPInt());
  }
}

TEST(APFloatTest, Float8E5M2FNUZNext) {
  APFloat test(APFloat::Float8E5M2FNUZ(), APFloat::uninitialized);
  APFloat expected(APFloat::Float8E5M2FNUZ(), APFloat::uninitialized);

  // 1. NextUp of largest bit pattern is nan
  test = APFloat::getLargest(APFloat::Float8E5M2FNUZ());
  expected = APFloat::getNaN(APFloat::Float8E5M2FNUZ());
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_FALSE(test.isInfinity());
  EXPECT_FALSE(test.isZero());
  EXPECT_TRUE(test.isNaN());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 2. NextUp of smallest negative denormal is +0
  test = APFloat::getSmallest(APFloat::Float8E5M2FNUZ(), true);
  expected = APFloat::getZero(APFloat::Float8E5M2FNUZ(), false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_FALSE(test.isNegZero());
  EXPECT_TRUE(test.isPosZero());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 3. nextDown of negative of largest value is NaN
  test = APFloat::getLargest(APFloat::Float8E5M2FNUZ(), true);
  expected = APFloat::getNaN(APFloat::Float8E5M2FNUZ());
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_FALSE(test.isInfinity());
  EXPECT_FALSE(test.isZero());
  EXPECT_TRUE(test.isNaN());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 4. nextDown of +0 is smallest negative denormal
  test = APFloat::getZero(APFloat::Float8E5M2FNUZ(), false);
  expected = APFloat::getSmallest(APFloat::Float8E5M2FNUZ(), true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_FALSE(test.isZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 5. nextUp of NaN is NaN
  test = APFloat::getNaN(APFloat::Float8E5M2FNUZ(), false);
  expected = APFloat::getNaN(APFloat::Float8E5M2FNUZ(), true);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isNaN());

  // 6. nextDown of NaN is NaN
  test = APFloat::getNaN(APFloat::Float8E5M2FNUZ(), false);
  expected = APFloat::getNaN(APFloat::Float8E5M2FNUZ(), true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isNaN());
}

TEST(APFloatTest, Float8E5M2FNUZChangeSign) {
  APFloat test = APFloat(APFloat::Float8E5M2FNUZ(), "1.0");
  APFloat expected = APFloat(APFloat::Float8E5M2FNUZ(), "-1.0");
  test.changeSign();
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getZero(APFloat::Float8E5M2FNUZ());
  expected = test;
  test.changeSign();
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getNaN(APFloat::Float8E5M2FNUZ());
  expected = test;
  test.changeSign();
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
}

TEST(APFloatTest, Float8E5M2FNUZFromString) {
  // Exactly representable
  EXPECT_EQ(57344,
            APFloat(APFloat::Float8E5M2FNUZ(), "57344").convertToDouble());
  // Round down to maximum value
  EXPECT_EQ(57344,
            APFloat(APFloat::Float8E5M2FNUZ(), "59392").convertToDouble());
  // Round up, causing overflow to NaN
  EXPECT_TRUE(APFloat(APFloat::Float8E5M2FNUZ(), "61440").isNaN());
  // Overflow without rounding
  EXPECT_TRUE(APFloat(APFloat::Float8E5M2FNUZ(), "131072").isNaN());
  // Inf converted to NaN
  EXPECT_TRUE(APFloat(APFloat::Float8E5M2FNUZ(), "inf").isNaN());
  // NaN converted to NaN
  EXPECT_TRUE(APFloat(APFloat::Float8E5M2FNUZ(), "nan").isNaN());
  // Negative zero converted to positive zero
  EXPECT_TRUE(APFloat(APFloat::Float8E5M2FNUZ(), "-0").isPosZero());
}

TEST(APFloatTest, UnsignedZeroArithmeticSpecial) {
  // Float semantics with only unsigned zero (ex. Float8E4M3FNUZ) violate the
  // IEEE rules about signs in arithmetic operations when producing zeros,
  // because they only have one zero. Most of the rest of the complexities of
  // arithmetic on these values are covered by the other Float8 types' test
  // cases and so are not repeated here.

  // The IEEE round towards negative rule doesn't apply
  for (APFloat::Semantics S :
       {APFloat::S_Float8E4M3FNUZ, APFloat::S_Float8E4M3B11FNUZ}) {
    const llvm::fltSemantics &Sem = APFloat::EnumToSemantics(S);
    APFloat test = APFloat::getSmallest(Sem);
    APFloat rhs = test;
    EXPECT_EQ(test.subtract(rhs, APFloat::rmTowardNegative), APFloat::opOK);
    EXPECT_TRUE(test.isZero());
    EXPECT_FALSE(test.isNegative());

    // Multiplication of (small) * (-small) is +0
    test = APFloat::getSmallestNormalized(Sem);
    rhs = -test;
    EXPECT_EQ(test.multiply(rhs, APFloat::rmNearestTiesToAway),
              APFloat::opInexact | APFloat::opUnderflow);
    EXPECT_TRUE(test.isZero());
    EXPECT_FALSE(test.isNegative());

    // Dividing the negatize float_min by anything gives +0
    test = APFloat::getSmallest(Sem, true);
    rhs = APFloat(Sem, "2.0");
    EXPECT_EQ(test.divide(rhs, APFloat::rmNearestTiesToEven),
              APFloat::opInexact | APFloat::opUnderflow);
    EXPECT_TRUE(test.isZero());
    EXPECT_FALSE(test.isNegative());

    // Remainder can't copy sign because there's only one zero
    test = APFloat(Sem, "-4.0");
    rhs = APFloat(Sem, "2.0");
    EXPECT_EQ(test.remainder(rhs), APFloat::opOK);
    EXPECT_TRUE(test.isZero());
    EXPECT_FALSE(test.isNegative());

    // And same for mod
    test = APFloat(Sem, "-4.0");
    rhs = APFloat(Sem, "2.0");
    EXPECT_EQ(test.mod(rhs), APFloat::opOK);
    EXPECT_TRUE(test.isZero());
    EXPECT_FALSE(test.isNegative());

    // FMA correctly handles both the multiply and add parts of all this
    test = APFloat(Sem, "2.0");
    rhs = test;
    APFloat addend = APFloat(Sem, "-4.0");
    EXPECT_EQ(test.fusedMultiplyAdd(rhs, addend, APFloat::rmTowardNegative),
              APFloat::opOK);
    EXPECT_TRUE(test.isZero());
    EXPECT_FALSE(test.isNegative());
  }
}

TEST(APFloatTest, Float8E5M2FNUZAdd) {
  APFloat QNaN = APFloat::getNaN(APFloat::Float8E5M2FNUZ(), false);

  auto FromStr = [](StringRef S) {
    return APFloat(APFloat::Float8E5M2FNUZ(), S);
  };

  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
    APFloat::roundingMode roundingMode = APFloat::rmNearestTiesToEven;
  } AdditionTests[] = {
      // Test addition operations involving NaN, overflow, and the max E5M2FNUZ
      // value (57344) because E5M2FNUZ differs from IEEE-754 types in these
      // regards
      {FromStr("57344"), FromStr("2048"), "57344", APFloat::opInexact,
       APFloat::fcNormal},
      {FromStr("57344"), FromStr("4096"), "NaN",
       APFloat::opOverflow | APFloat::opInexact, APFloat::fcNaN},
      {FromStr("-57344"), FromStr("-4096"), "NaN",
       APFloat::opOverflow | APFloat::opInexact, APFloat::fcNaN},
      {QNaN, FromStr("-57344"), "NaN", APFloat::opOK, APFloat::fcNaN},
      {FromStr("57344"), FromStr("-8192"), "49152", APFloat::opOK,
       APFloat::fcNormal},
      {FromStr("57344"), FromStr("0"), "57344", APFloat::opOK,
       APFloat::fcNormal},
      {FromStr("57344"), FromStr("4096"), "57344", APFloat::opInexact,
       APFloat::fcNormal, APFloat::rmTowardZero},
      {FromStr("57344"), FromStr("57344"), "57344", APFloat::opInexact,
       APFloat::fcNormal, APFloat::rmTowardZero},
  };

  for (size_t i = 0; i < std::size(AdditionTests); ++i) {
    APFloat x(AdditionTests[i].x);
    APFloat y(AdditionTests[i].y);
    APFloat::opStatus status = x.add(y, AdditionTests[i].roundingMode);

    APFloat result(APFloat::Float8E5M2FNUZ(), AdditionTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_EQ(AdditionTests[i].status, (int)status);
    EXPECT_EQ(AdditionTests[i].category, (int)x.getCategory());
  }
}

TEST(APFloatTest, Float8E5M2FNUZDivideByZero) {
  APFloat x(APFloat::Float8E5M2FNUZ(), "1");
  APFloat zero(APFloat::Float8E5M2FNUZ(), "0");
  EXPECT_EQ(x.divide(zero, APFloat::rmNearestTiesToEven), APFloat::opDivByZero);
  EXPECT_TRUE(x.isNaN());
}

TEST(APFloatTest, Float8UnsignedZeroExhaustive) {
  struct {
    const fltSemantics *semantics;
    const double largest;
    const double smallest;
  } const exhaustiveTests[] = {{&APFloat::Float8E5M2FNUZ(), 57344., 0x1.0p-17},
                               {&APFloat::Float8E4M3FNUZ(), 240., 0x1.0p-10},
                               {&APFloat::Float8E4M3B11FNUZ(), 30., 0x1.0p-13}};
  for (const auto &testInfo : exhaustiveTests) {
    const fltSemantics &sem = *testInfo.semantics;
    SCOPED_TRACE("Semantics=" + std::to_string(APFloat::SemanticsToEnum(sem)));
    // Test each of the 256 values.
    for (int i = 0; i < 256; i++) {
      SCOPED_TRACE("i=" + std::to_string(i));
      APFloat test(sem, APInt(8, i));

      // isLargest
      if (i == 127 || i == 255) {
        EXPECT_TRUE(test.isLargest());
        EXPECT_EQ(abs(test).convertToDouble(), testInfo.largest);
      } else {
        EXPECT_FALSE(test.isLargest());
      }

      // isSmallest
      if (i == 1 || i == 129) {
        EXPECT_TRUE(test.isSmallest());
        EXPECT_EQ(abs(test).convertToDouble(), testInfo.smallest);
      } else {
        EXPECT_FALSE(test.isSmallest());
      }

      // convert to BFloat
      APFloat test2 = test;
      bool losesInfo;
      APFloat::opStatus status = test2.convert(
          APFloat::BFloat(), APFloat::rmNearestTiesToEven, &losesInfo);
      EXPECT_EQ(status, APFloat::opOK);
      EXPECT_FALSE(losesInfo);
      if (i == 128)
        EXPECT_TRUE(test2.isNaN());
      else
        EXPECT_EQ(test.convertToFloat(), test2.convertToFloat());

      // bitcastToAPInt
      EXPECT_EQ(i, test.bitcastToAPInt());
    }
  }
}

TEST(APFloatTest, Float8E4M3FNUZNext) {
  for (APFloat::Semantics S :
       {APFloat::S_Float8E4M3FNUZ, APFloat::S_Float8E4M3B11FNUZ}) {
    const llvm::fltSemantics &Sem = APFloat::EnumToSemantics(S);
    APFloat test(Sem, APFloat::uninitialized);
    APFloat expected(Sem, APFloat::uninitialized);

    // 1. NextUp of largest bit pattern is nan
    test = APFloat::getLargest(Sem);
    expected = APFloat::getNaN(Sem);
    EXPECT_EQ(test.next(false), APFloat::opOK);
    EXPECT_FALSE(test.isInfinity());
    EXPECT_FALSE(test.isZero());
    EXPECT_TRUE(test.isNaN());
    EXPECT_TRUE(test.bitwiseIsEqual(expected));

    // 2. NextUp of smallest negative denormal is +0
    test = APFloat::getSmallest(Sem, true);
    expected = APFloat::getZero(Sem, false);
    EXPECT_EQ(test.next(false), APFloat::opOK);
    EXPECT_FALSE(test.isNegZero());
    EXPECT_TRUE(test.isPosZero());
    EXPECT_TRUE(test.bitwiseIsEqual(expected));

    // 3. nextDown of negative of largest value is NaN
    test = APFloat::getLargest(Sem, true);
    expected = APFloat::getNaN(Sem);
    EXPECT_EQ(test.next(true), APFloat::opOK);
    EXPECT_FALSE(test.isInfinity());
    EXPECT_FALSE(test.isZero());
    EXPECT_TRUE(test.isNaN());
    EXPECT_TRUE(test.bitwiseIsEqual(expected));

    // 4. nextDown of +0 is smallest negative denormal
    test = APFloat::getZero(Sem, false);
    expected = APFloat::getSmallest(Sem, true);
    EXPECT_EQ(test.next(true), APFloat::opOK);
    EXPECT_FALSE(test.isZero());
    EXPECT_TRUE(test.isDenormal());
    EXPECT_TRUE(test.bitwiseIsEqual(expected));

    // 5. nextUp of NaN is NaN
    test = APFloat::getNaN(Sem, false);
    expected = APFloat::getNaN(Sem, true);
    EXPECT_EQ(test.next(false), APFloat::opOK);
    EXPECT_TRUE(test.isNaN());

    // 6. nextDown of NaN is NaN
    test = APFloat::getNaN(Sem, false);
    expected = APFloat::getNaN(Sem, true);
    EXPECT_EQ(test.next(true), APFloat::opOK);
    EXPECT_TRUE(test.isNaN());
  }
}

TEST(APFloatTest, Float8E4M3FNUZChangeSign) {
  for (APFloat::Semantics S :
       {APFloat::S_Float8E4M3FNUZ, APFloat::S_Float8E4M3B11FNUZ}) {
    const llvm::fltSemantics &Sem = APFloat::EnumToSemantics(S);
    APFloat test = APFloat(Sem, "1.0");
    APFloat expected = APFloat(Sem, "-1.0");
    test.changeSign();
    EXPECT_TRUE(test.bitwiseIsEqual(expected));

    test = APFloat::getZero(Sem);
    expected = test;
    test.changeSign();
    EXPECT_TRUE(test.bitwiseIsEqual(expected));

    test = APFloat::getNaN(Sem);
    expected = test;
    test.changeSign();
    EXPECT_TRUE(test.bitwiseIsEqual(expected));
  }
}

TEST(APFloatTest, Float8E4M3FNUZFromString) {
  // Exactly representable
  EXPECT_EQ(240, APFloat(APFloat::Float8E4M3FNUZ(), "240").convertToDouble());
  // Round down to maximum value
  EXPECT_EQ(240, APFloat(APFloat::Float8E4M3FNUZ(), "247").convertToDouble());
  // Round up, causing overflow to NaN
  EXPECT_TRUE(APFloat(APFloat::Float8E4M3FNUZ(), "248").isNaN());
  // Overflow without rounding
  EXPECT_TRUE(APFloat(APFloat::Float8E4M3FNUZ(), "480").isNaN());
  // Inf converted to NaN
  EXPECT_TRUE(APFloat(APFloat::Float8E4M3FNUZ(), "inf").isNaN());
  // NaN converted to NaN
  EXPECT_TRUE(APFloat(APFloat::Float8E4M3FNUZ(), "nan").isNaN());
  // Negative zero converted to positive zero
  EXPECT_TRUE(APFloat(APFloat::Float8E4M3FNUZ(), "-0").isPosZero());
}

TEST(APFloatTest, Float8E4M3FNUZAdd) {
  APFloat QNaN = APFloat::getNaN(APFloat::Float8E4M3FNUZ(), false);

  auto FromStr = [](StringRef S) {
    return APFloat(APFloat::Float8E4M3FNUZ(), S);
  };

  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
    APFloat::roundingMode roundingMode = APFloat::rmNearestTiesToEven;
  } AdditionTests[] = {
      // Test addition operations involving NaN, overflow, and the max E4M3FNUZ
      // value (240) because E4M3FNUZ differs from IEEE-754 types in these
      // regards
      {FromStr("240"), FromStr("4"), "240", APFloat::opInexact,
       APFloat::fcNormal},
      {FromStr("240"), FromStr("8"), "NaN",
       APFloat::opOverflow | APFloat::opInexact, APFloat::fcNaN},
      {FromStr("240"), FromStr("16"), "NaN",
       APFloat::opOverflow | APFloat::opInexact, APFloat::fcNaN},
      {FromStr("-240"), FromStr("-16"), "NaN",
       APFloat::opOverflow | APFloat::opInexact, APFloat::fcNaN},
      {QNaN, FromStr("-240"), "NaN", APFloat::opOK, APFloat::fcNaN},
      {FromStr("240"), FromStr("-16"), "224", APFloat::opOK, APFloat::fcNormal},
      {FromStr("240"), FromStr("0"), "240", APFloat::opOK, APFloat::fcNormal},
      {FromStr("240"), FromStr("32"), "240", APFloat::opInexact,
       APFloat::fcNormal, APFloat::rmTowardZero},
      {FromStr("240"), FromStr("240"), "240", APFloat::opInexact,
       APFloat::fcNormal, APFloat::rmTowardZero},
  };

  for (size_t i = 0; i < std::size(AdditionTests); ++i) {
    APFloat x(AdditionTests[i].x);
    APFloat y(AdditionTests[i].y);
    APFloat::opStatus status = x.add(y, AdditionTests[i].roundingMode);

    APFloat result(APFloat::Float8E4M3FNUZ(), AdditionTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_EQ(AdditionTests[i].status, (int)status);
    EXPECT_EQ(AdditionTests[i].category, (int)x.getCategory());
  }
}

TEST(APFloatTest, Float8E4M3FNUZDivideByZero) {
  APFloat x(APFloat::Float8E4M3FNUZ(), "1");
  APFloat zero(APFloat::Float8E4M3FNUZ(), "0");
  EXPECT_EQ(x.divide(zero, APFloat::rmNearestTiesToEven), APFloat::opDivByZero);
  EXPECT_TRUE(x.isNaN());
}

TEST(APFloatTest, ConvertE5M2FNUZToE4M3FNUZ) {
  bool losesInfo;
  APFloat test(APFloat::Float8E5M2FNUZ(), "1.0");
  APFloat::opStatus status = test.convert(
      APFloat::Float8E4M3FNUZ(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0f, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  losesInfo = true;
  test = APFloat(APFloat::Float8E5M2FNUZ(), "0.0");
  status = test.convert(APFloat::Float8E4M3FNUZ(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0.0f, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  losesInfo = true;
  test = APFloat(APFloat::Float8E5M2FNUZ(), "0x1.Cp7"); // 224
  status = test.convert(APFloat::Float8E4M3FNUZ(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.Cp7 /* 224 */, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  // Test overflow
  losesInfo = false;
  test = APFloat(APFloat::Float8E5M2FNUZ(), "0x1.0p8"); // 256
  status = test.convert(APFloat::Float8E4M3FNUZ(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_TRUE(std::isnan(test.convertToFloat()));
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opOverflow | APFloat::opInexact);

  // Test underflow
  test = APFloat(APFloat::Float8E5M2FNUZ(), "0x1.0p-11");
  status = test.convert(APFloat::Float8E4M3FNUZ(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0., test.convertToFloat());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opUnderflow | APFloat::opInexact);

  // Test rounding up to smallest denormal number
  losesInfo = false;
  test = APFloat(APFloat::Float8E5M2FNUZ(), "0x1.8p-11");
  status = test.convert(APFloat::Float8E4M3FNUZ(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.0p-10, test.convertToFloat());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opUnderflow | APFloat::opInexact);

  // Testing inexact rounding to denormal number
  losesInfo = false;
  test = APFloat(APFloat::Float8E5M2FNUZ(), "0x1.8p-10");
  status = test.convert(APFloat::Float8E4M3FNUZ(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.0p-9, test.convertToFloat());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opUnderflow | APFloat::opInexact);
}

TEST(APFloatTest, ConvertE4M3FNUZToE5M2FNUZ) {
  bool losesInfo;
  APFloat test(APFloat::Float8E4M3FNUZ(), "1.0");
  APFloat::opStatus status = test.convert(
      APFloat::Float8E5M2FNUZ(), APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0f, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  losesInfo = true;
  test = APFloat(APFloat::Float8E4M3FNUZ(), "0.0");
  status = test.convert(APFloat::Float8E5M2FNUZ(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0.0f, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);

  losesInfo = false;
  test = APFloat(APFloat::Float8E4M3FNUZ(), "0x1.2p0"); // 1.125
  status = test.convert(APFloat::Float8E5M2FNUZ(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.0p0 /* 1.0 */, test.convertToFloat());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opInexact);

  losesInfo = false;
  test = APFloat(APFloat::Float8E4M3FNUZ(), "0x1.6p0"); // 1.375
  status = test.convert(APFloat::Float8E5M2FNUZ(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.8p0 /* 1.5 */, test.convertToFloat());
  EXPECT_TRUE(losesInfo);
  EXPECT_EQ(status, APFloat::opInexact);

  // Convert E4M3 denormal to E5M2 normal. Should not be truncated, despite the
  // destination format having one fewer significand bit
  losesInfo = true;
  test = APFloat(APFloat::Float8E4M3FNUZ(), "0x1.Cp-8");
  status = test.convert(APFloat::Float8E5M2FNUZ(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
  EXPECT_EQ(0x1.Cp-8, test.convertToFloat());
  EXPECT_FALSE(losesInfo);
  EXPECT_EQ(status, APFloat::opOK);
}

TEST(APFloatTest, F8ToString) {
  for (APFloat::Semantics S :
       {APFloat::S_Float8E5M2, APFloat::S_Float8E4M3FN,
        APFloat::S_Float8E5M2FNUZ, APFloat::S_Float8E4M3FNUZ,
        APFloat::S_Float8E4M3B11FNUZ}) {
    SCOPED_TRACE("Semantics=" + std::to_string(S));
    for (int i = 0; i < 256; i++) {
      SCOPED_TRACE("i=" + std::to_string(i));
      APFloat test(APFloat::EnumToSemantics(S), APInt(8, i));
      llvm::SmallString<128> str;
      test.toString(str);

      if (test.isNaN()) {
        EXPECT_EQ(str, "NaN");
      } else {
        APFloat test2(APFloat::EnumToSemantics(S), str);
        EXPECT_TRUE(test.bitwiseIsEqual(test2));
      }
    }
  }
}

TEST(APFloatTest, BitsToF8ToBits) {
  for (APFloat::Semantics S :
       {APFloat::S_Float8E5M2, APFloat::S_Float8E4M3FN,
        APFloat::S_Float8E5M2FNUZ, APFloat::S_Float8E4M3FNUZ,
        APFloat::S_Float8E4M3B11FNUZ}) {
    SCOPED_TRACE("Semantics=" + std::to_string(S));
    for (int i = 0; i < 256; i++) {
      SCOPED_TRACE("i=" + std::to_string(i));
      APInt bits_in = APInt(8, i);
      APFloat test(APFloat::EnumToSemantics(S), bits_in);
      APInt bits_out = test.bitcastToAPInt();
      EXPECT_EQ(bits_in, bits_out);
    }
  }
}

TEST(APFloatTest, F8ToBitsToF8) {
  for (APFloat::Semantics S :
       {APFloat::S_Float8E5M2, APFloat::S_Float8E4M3FN,
        APFloat::S_Float8E5M2FNUZ, APFloat::S_Float8E4M3FNUZ,
        APFloat::S_Float8E4M3B11FNUZ}) {
    SCOPED_TRACE("Semantics=" + std::to_string(S));
    auto &Sem = APFloat::EnumToSemantics(S);
    for (bool negative : {false, true}) {
      SCOPED_TRACE("negative=" + std::to_string(negative));
      APFloat test = APFloat::getZero(Sem, /*Negative=*/negative);
      for (int i = 0; i < 128; i++, test.next(/*nextDown=*/negative)) {
        SCOPED_TRACE("i=" + std::to_string(i));
        APInt bits = test.bitcastToAPInt();
        APFloat test2 = APFloat(Sem, bits);
        if (test.isNaN()) {
          EXPECT_TRUE(test2.isNaN());
        } else {
          EXPECT_TRUE(test.bitwiseIsEqual(test2));
        }
      }
    }
  }
}

TEST(APFloatTest, IEEEdoubleToDouble) {
  APFloat DPosZero(0.0);
  APFloat DPosZeroToDouble(DPosZero.convertToDouble());
  EXPECT_TRUE(DPosZeroToDouble.isPosZero());
  APFloat DNegZero(-0.0);
  APFloat DNegZeroToDouble(DNegZero.convertToDouble());
  EXPECT_TRUE(DNegZeroToDouble.isNegZero());

  APFloat DOne(1.0);
  EXPECT_EQ(1.0, DOne.convertToDouble());
  APFloat DPosLargest = APFloat::getLargest(APFloat::IEEEdouble(), false);
  EXPECT_EQ(std::numeric_limits<double>::max(), DPosLargest.convertToDouble());
  APFloat DNegLargest = APFloat::getLargest(APFloat::IEEEdouble(), true);
  EXPECT_EQ(-std::numeric_limits<double>::max(), DNegLargest.convertToDouble());
  APFloat DPosSmallest =
      APFloat::getSmallestNormalized(APFloat::IEEEdouble(), false);
  EXPECT_EQ(std::numeric_limits<double>::min(), DPosSmallest.convertToDouble());
  APFloat DNegSmallest =
      APFloat::getSmallestNormalized(APFloat::IEEEdouble(), true);
  EXPECT_EQ(-std::numeric_limits<double>::min(),
            DNegSmallest.convertToDouble());

  APFloat DSmallestDenorm = APFloat::getSmallest(APFloat::IEEEdouble(), false);
  EXPECT_EQ(std::numeric_limits<double>::denorm_min(),
            DSmallestDenorm.convertToDouble());
  APFloat DLargestDenorm(APFloat::IEEEdouble(), "0x0.FFFFFFFFFFFFFp-1022");
  EXPECT_EQ(/*0x0.FFFFFFFFFFFFFp-1022*/ 2.225073858507201e-308,
            DLargestDenorm.convertToDouble());

  APFloat DPosInf = APFloat::getInf(APFloat::IEEEdouble());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), DPosInf.convertToDouble());
  APFloat DNegInf = APFloat::getInf(APFloat::IEEEdouble(), true);
  EXPECT_EQ(-std::numeric_limits<double>::infinity(),
            DNegInf.convertToDouble());
  APFloat DQNaN = APFloat::getQNaN(APFloat::IEEEdouble());
  EXPECT_TRUE(std::isnan(DQNaN.convertToDouble()));
}

TEST(APFloatTest, IEEEsingleToDouble) {
  APFloat FPosZero(0.0F);
  APFloat FPosZeroToDouble(FPosZero.convertToDouble());
  EXPECT_TRUE(FPosZeroToDouble.isPosZero());
  APFloat FNegZero(-0.0F);
  APFloat FNegZeroToDouble(FNegZero.convertToDouble());
  EXPECT_TRUE(FNegZeroToDouble.isNegZero());

  APFloat FOne(1.0F);
  EXPECT_EQ(1.0, FOne.convertToDouble());
  APFloat FPosLargest = APFloat::getLargest(APFloat::IEEEsingle(), false);
  EXPECT_EQ(std::numeric_limits<float>::max(), FPosLargest.convertToDouble());
  APFloat FNegLargest = APFloat::getLargest(APFloat::IEEEsingle(), true);
  EXPECT_EQ(-std::numeric_limits<float>::max(), FNegLargest.convertToDouble());
  APFloat FPosSmallest =
      APFloat::getSmallestNormalized(APFloat::IEEEsingle(), false);
  EXPECT_EQ(std::numeric_limits<float>::min(), FPosSmallest.convertToDouble());
  APFloat FNegSmallest =
      APFloat::getSmallestNormalized(APFloat::IEEEsingle(), true);
  EXPECT_EQ(-std::numeric_limits<float>::min(), FNegSmallest.convertToDouble());

  APFloat FSmallestDenorm = APFloat::getSmallest(APFloat::IEEEsingle(), false);
  EXPECT_EQ(std::numeric_limits<float>::denorm_min(),
            FSmallestDenorm.convertToDouble());
  APFloat FLargestDenorm(APFloat::IEEEdouble(), "0x0.FFFFFEp-126");
  EXPECT_EQ(/*0x0.FFFFFEp-126*/ 1.1754942106924411e-38,
            FLargestDenorm.convertToDouble());

  APFloat FPosInf = APFloat::getInf(APFloat::IEEEsingle());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), FPosInf.convertToDouble());
  APFloat FNegInf = APFloat::getInf(APFloat::IEEEsingle(), true);
  EXPECT_EQ(-std::numeric_limits<double>::infinity(),
            FNegInf.convertToDouble());
  APFloat FQNaN = APFloat::getQNaN(APFloat::IEEEsingle());
  EXPECT_TRUE(std::isnan(FQNaN.convertToDouble()));
}

TEST(APFloatTest, IEEEhalfToDouble) {
  APFloat HPosZero = APFloat::getZero(APFloat::IEEEhalf());
  APFloat HPosZeroToDouble(HPosZero.convertToDouble());
  EXPECT_TRUE(HPosZeroToDouble.isPosZero());
  APFloat HNegZero = APFloat::getZero(APFloat::IEEEhalf(), true);
  APFloat HNegZeroToDouble(HNegZero.convertToDouble());
  EXPECT_TRUE(HNegZeroToDouble.isNegZero());

  APFloat HOne(APFloat::IEEEhalf(), "1.0");
  EXPECT_EQ(1.0, HOne.convertToDouble());
  APFloat HPosLargest = APFloat::getLargest(APFloat::IEEEhalf(), false);
  EXPECT_EQ(65504.0, HPosLargest.convertToDouble());
  APFloat HNegLargest = APFloat::getLargest(APFloat::IEEEhalf(), true);
  EXPECT_EQ(-65504.0, HNegLargest.convertToDouble());
  APFloat HPosSmallest =
      APFloat::getSmallestNormalized(APFloat::IEEEhalf(), false);
  EXPECT_EQ(/*0x1.p-14*/ 6.103515625e-05, HPosSmallest.convertToDouble());
  APFloat HNegSmallest =
      APFloat::getSmallestNormalized(APFloat::IEEEhalf(), true);
  EXPECT_EQ(/*-0x1.p-14*/ -6.103515625e-05, HNegSmallest.convertToDouble());

  APFloat HSmallestDenorm = APFloat::getSmallest(APFloat::IEEEhalf(), false);
  EXPECT_EQ(/*0x1.p-24*/ 5.960464477539063e-08,
            HSmallestDenorm.convertToDouble());
  APFloat HLargestDenorm(APFloat::IEEEhalf(), "0x1.FFCp-14");
  EXPECT_EQ(/*0x1.FFCp-14*/ 0.00012201070785522461,
            HLargestDenorm.convertToDouble());

  APFloat HPosInf = APFloat::getInf(APFloat::IEEEhalf());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), HPosInf.convertToDouble());
  APFloat HNegInf = APFloat::getInf(APFloat::IEEEhalf(), true);
  EXPECT_EQ(-std::numeric_limits<double>::infinity(),
            HNegInf.convertToDouble());
  APFloat HQNaN = APFloat::getQNaN(APFloat::IEEEhalf());
  EXPECT_TRUE(std::isnan(HQNaN.convertToDouble()));

  APFloat BPosZero = APFloat::getZero(APFloat::IEEEhalf());
  APFloat BPosZeroToDouble(BPosZero.convertToDouble());
  EXPECT_TRUE(BPosZeroToDouble.isPosZero());
  APFloat BNegZero = APFloat::getZero(APFloat::IEEEhalf(), true);
  APFloat BNegZeroToDouble(BNegZero.convertToDouble());
  EXPECT_TRUE(BNegZeroToDouble.isNegZero());
}

TEST(APFloatTest, BFloatToDouble) {
  APFloat BOne(APFloat::BFloat(), "1.0");
  EXPECT_EQ(1.0, BOne.convertToDouble());
  APFloat BPosLargest = APFloat::getLargest(APFloat::BFloat(), false);
  EXPECT_EQ(/*0x1.FEp127*/ 3.3895313892515355e+38,
            BPosLargest.convertToDouble());
  APFloat BNegLargest = APFloat::getLargest(APFloat::BFloat(), true);
  EXPECT_EQ(/*-0x1.FEp127*/ -3.3895313892515355e+38,
            BNegLargest.convertToDouble());
  APFloat BPosSmallest =
      APFloat::getSmallestNormalized(APFloat::BFloat(), false);
  EXPECT_EQ(/*0x1.p-126*/ 1.1754943508222875e-38,
            BPosSmallest.convertToDouble());
  APFloat BNegSmallest =
      APFloat::getSmallestNormalized(APFloat::BFloat(), true);
  EXPECT_EQ(/*-0x1.p-126*/ -1.1754943508222875e-38,
            BNegSmallest.convertToDouble());

  APFloat BSmallestDenorm = APFloat::getSmallest(APFloat::BFloat(), false);
  EXPECT_EQ(/*0x1.p-133*/ 9.183549615799121e-41,
            BSmallestDenorm.convertToDouble());
  APFloat BLargestDenorm(APFloat::BFloat(), "0x1.FCp-127");
  EXPECT_EQ(/*0x1.FCp-127*/ 1.1663108012064884e-38,
            BLargestDenorm.convertToDouble());

  APFloat BPosInf = APFloat::getInf(APFloat::BFloat());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), BPosInf.convertToDouble());
  APFloat BNegInf = APFloat::getInf(APFloat::BFloat(), true);
  EXPECT_EQ(-std::numeric_limits<double>::infinity(),
            BNegInf.convertToDouble());
  APFloat BQNaN = APFloat::getQNaN(APFloat::BFloat());
  EXPECT_TRUE(std::isnan(BQNaN.convertToDouble()));
}

TEST(APFloatTest, Float8E5M2ToDouble) {
  APFloat One(APFloat::Float8E5M2(), "1.0");
  EXPECT_EQ(1.0, One.convertToDouble());
  APFloat Two(APFloat::Float8E5M2(), "2.0");
  EXPECT_EQ(2.0, Two.convertToDouble());
  APFloat PosLargest = APFloat::getLargest(APFloat::Float8E5M2(), false);
  EXPECT_EQ(5.734400e+04, PosLargest.convertToDouble());
  APFloat NegLargest = APFloat::getLargest(APFloat::Float8E5M2(), true);
  EXPECT_EQ(-5.734400e+04, NegLargest.convertToDouble());
  APFloat PosSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E5M2(), false);
  EXPECT_EQ(0x1.p-14, PosSmallest.convertToDouble());
  APFloat NegSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E5M2(), true);
  EXPECT_EQ(-0x1.p-14, NegSmallest.convertToDouble());

  APFloat SmallestDenorm = APFloat::getSmallest(APFloat::Float8E5M2(), false);
  EXPECT_TRUE(SmallestDenorm.isDenormal());
  EXPECT_EQ(0x1p-16, SmallestDenorm.convertToDouble());

  APFloat PosInf = APFloat::getInf(APFloat::Float8E5M2());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), PosInf.convertToDouble());
  APFloat NegInf = APFloat::getInf(APFloat::Float8E5M2(), true);
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), NegInf.convertToDouble());
  APFloat QNaN = APFloat::getQNaN(APFloat::Float8E5M2());
  EXPECT_TRUE(std::isnan(QNaN.convertToDouble()));
}

TEST(APFloatTest, Float8E4M3FNToDouble) {
  APFloat One(APFloat::Float8E4M3FN(), "1.0");
  EXPECT_EQ(1.0, One.convertToDouble());
  APFloat Two(APFloat::Float8E4M3FN(), "2.0");
  EXPECT_EQ(2.0, Two.convertToDouble());
  APFloat PosLargest = APFloat::getLargest(APFloat::Float8E4M3FN(), false);
  EXPECT_EQ(448., PosLargest.convertToDouble());
  APFloat NegLargest = APFloat::getLargest(APFloat::Float8E4M3FN(), true);
  EXPECT_EQ(-448., NegLargest.convertToDouble());
  APFloat PosSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E4M3FN(), false);
  EXPECT_EQ(0x1.p-6, PosSmallest.convertToDouble());
  APFloat NegSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E4M3FN(), true);
  EXPECT_EQ(-0x1.p-6, NegSmallest.convertToDouble());

  APFloat SmallestDenorm = APFloat::getSmallest(APFloat::Float8E4M3FN(), false);
  EXPECT_TRUE(SmallestDenorm.isDenormal());
  EXPECT_EQ(0x1p-9, SmallestDenorm.convertToDouble());

  APFloat QNaN = APFloat::getQNaN(APFloat::Float8E4M3FN());
  EXPECT_TRUE(std::isnan(QNaN.convertToDouble()));
}

TEST(APFloatTest, Float8E5M2FNUZToDouble) {
  APFloat One(APFloat::Float8E5M2FNUZ(), "1.0");
  EXPECT_EQ(1.0, One.convertToDouble());
  APFloat Two(APFloat::Float8E5M2FNUZ(), "2.0");
  EXPECT_EQ(2.0, Two.convertToDouble());
  APFloat PosLargest = APFloat::getLargest(APFloat::Float8E5M2FNUZ(), false);
  EXPECT_EQ(57344., PosLargest.convertToDouble());
  APFloat NegLargest = APFloat::getLargest(APFloat::Float8E5M2FNUZ(), true);
  EXPECT_EQ(-57344., NegLargest.convertToDouble());
  APFloat PosSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E5M2FNUZ(), false);
  EXPECT_EQ(0x1.p-15, PosSmallest.convertToDouble());
  APFloat NegSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E5M2FNUZ(), true);
  EXPECT_EQ(-0x1.p-15, NegSmallest.convertToDouble());

  APFloat SmallestDenorm =
      APFloat::getSmallest(APFloat::Float8E5M2FNUZ(), false);
  EXPECT_TRUE(SmallestDenorm.isDenormal());
  EXPECT_EQ(0x1p-17, SmallestDenorm.convertToDouble());

  APFloat QNaN = APFloat::getQNaN(APFloat::Float8E5M2FNUZ());
  EXPECT_TRUE(std::isnan(QNaN.convertToDouble()));
}

TEST(APFloatTest, Float8E4M3FNUZToDouble) {
  APFloat One(APFloat::Float8E4M3FNUZ(), "1.0");
  EXPECT_EQ(1.0, One.convertToDouble());
  APFloat Two(APFloat::Float8E4M3FNUZ(), "2.0");
  EXPECT_EQ(2.0, Two.convertToDouble());
  APFloat PosLargest = APFloat::getLargest(APFloat::Float8E4M3FNUZ(), false);
  EXPECT_EQ(240., PosLargest.convertToDouble());
  APFloat NegLargest = APFloat::getLargest(APFloat::Float8E4M3FNUZ(), true);
  EXPECT_EQ(-240., NegLargest.convertToDouble());
  APFloat PosSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E4M3FNUZ(), false);
  EXPECT_EQ(0x1.p-7, PosSmallest.convertToDouble());
  APFloat NegSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E4M3FNUZ(), true);
  EXPECT_EQ(-0x1.p-7, NegSmallest.convertToDouble());

  APFloat SmallestDenorm =
      APFloat::getSmallest(APFloat::Float8E4M3FNUZ(), false);
  EXPECT_TRUE(SmallestDenorm.isDenormal());
  EXPECT_EQ(0x1p-10, SmallestDenorm.convertToDouble());

  APFloat QNaN = APFloat::getQNaN(APFloat::Float8E4M3FNUZ());
  EXPECT_TRUE(std::isnan(QNaN.convertToDouble()));
}

TEST(APFloatTest, FloatTF32ToDouble) {
  APFloat One(APFloat::FloatTF32(), "1.0");
  EXPECT_EQ(1.0, One.convertToDouble());
  APFloat PosLargest = APFloat::getLargest(APFloat::FloatTF32(), false);
  EXPECT_EQ(3.401162134214653489792616e+38, PosLargest.convertToDouble());
  APFloat NegLargest = APFloat::getLargest(APFloat::FloatTF32(), true);
  EXPECT_EQ(-3.401162134214653489792616e+38, NegLargest.convertToDouble());
  APFloat PosSmallest =
      APFloat::getSmallestNormalized(APFloat::FloatTF32(), false);
  EXPECT_EQ(1.1754943508222875079687e-38, PosSmallest.convertToDouble());
  APFloat NegSmallest =
      APFloat::getSmallestNormalized(APFloat::FloatTF32(), true);
  EXPECT_EQ(-1.1754943508222875079687e-38, NegSmallest.convertToDouble());

  APFloat SmallestDenorm = APFloat::getSmallest(APFloat::FloatTF32(), false);
  EXPECT_EQ(1.1479437019748901445007e-41, SmallestDenorm.convertToDouble());
  APFloat LargestDenorm(APFloat::FloatTF32(), "0x1.FF8p-127");
  EXPECT_EQ(/*0x1.FF8p-127*/ 1.1743464071203126178242e-38,
            LargestDenorm.convertToDouble());

  APFloat PosInf = APFloat::getInf(APFloat::FloatTF32());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), PosInf.convertToDouble());
  APFloat NegInf = APFloat::getInf(APFloat::FloatTF32(), true);
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), NegInf.convertToDouble());
  APFloat QNaN = APFloat::getQNaN(APFloat::FloatTF32());
  EXPECT_TRUE(std::isnan(QNaN.convertToDouble()));
}

TEST(APFloatTest, Float8E5M2FNUZToFloat) {
  APFloat PosZero = APFloat::getZero(APFloat::Float8E5M2FNUZ());
  APFloat PosZeroToFloat(PosZero.convertToFloat());
  EXPECT_TRUE(PosZeroToFloat.isPosZero());
  // Negative zero is not supported
  APFloat NegZero = APFloat::getZero(APFloat::Float8E5M2FNUZ(), true);
  APFloat NegZeroToFloat(NegZero.convertToFloat());
  EXPECT_TRUE(NegZeroToFloat.isPosZero());
  APFloat One(APFloat::Float8E5M2FNUZ(), "1.0");
  EXPECT_EQ(1.0F, One.convertToFloat());
  APFloat Two(APFloat::Float8E5M2FNUZ(), "2.0");
  EXPECT_EQ(2.0F, Two.convertToFloat());
  APFloat PosLargest = APFloat::getLargest(APFloat::Float8E5M2FNUZ(), false);
  EXPECT_EQ(57344.F, PosLargest.convertToFloat());
  APFloat NegLargest = APFloat::getLargest(APFloat::Float8E5M2FNUZ(), true);
  EXPECT_EQ(-57344.F, NegLargest.convertToFloat());
  APFloat PosSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E5M2FNUZ(), false);
  EXPECT_EQ(0x1.p-15F, PosSmallest.convertToFloat());
  APFloat NegSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E5M2FNUZ(), true);
  EXPECT_EQ(-0x1.p-15F, NegSmallest.convertToFloat());

  APFloat SmallestDenorm =
      APFloat::getSmallest(APFloat::Float8E5M2FNUZ(), false);
  EXPECT_TRUE(SmallestDenorm.isDenormal());
  EXPECT_EQ(0x1p-17F, SmallestDenorm.convertToFloat());

  APFloat QNaN = APFloat::getQNaN(APFloat::Float8E5M2FNUZ());
  EXPECT_TRUE(std::isnan(QNaN.convertToFloat()));
}

TEST(APFloatTest, Float8E4M3FNUZToFloat) {
  APFloat PosZero = APFloat::getZero(APFloat::Float8E4M3FNUZ());
  APFloat PosZeroToFloat(PosZero.convertToFloat());
  EXPECT_TRUE(PosZeroToFloat.isPosZero());
  // Negative zero is not supported
  APFloat NegZero = APFloat::getZero(APFloat::Float8E4M3FNUZ(), true);
  APFloat NegZeroToFloat(NegZero.convertToFloat());
  EXPECT_TRUE(NegZeroToFloat.isPosZero());
  APFloat One(APFloat::Float8E4M3FNUZ(), "1.0");
  EXPECT_EQ(1.0F, One.convertToFloat());
  APFloat Two(APFloat::Float8E4M3FNUZ(), "2.0");
  EXPECT_EQ(2.0F, Two.convertToFloat());
  APFloat PosLargest = APFloat::getLargest(APFloat::Float8E4M3FNUZ(), false);
  EXPECT_EQ(240.F, PosLargest.convertToFloat());
  APFloat NegLargest = APFloat::getLargest(APFloat::Float8E4M3FNUZ(), true);
  EXPECT_EQ(-240.F, NegLargest.convertToFloat());
  APFloat PosSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E4M3FNUZ(), false);
  EXPECT_EQ(0x1.p-7F, PosSmallest.convertToFloat());
  APFloat NegSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E4M3FNUZ(), true);
  EXPECT_EQ(-0x1.p-7F, NegSmallest.convertToFloat());

  APFloat SmallestDenorm =
      APFloat::getSmallest(APFloat::Float8E4M3FNUZ(), false);
  EXPECT_TRUE(SmallestDenorm.isDenormal());
  EXPECT_EQ(0x1p-10F, SmallestDenorm.convertToFloat());

  APFloat QNaN = APFloat::getQNaN(APFloat::Float8E4M3FNUZ());
  EXPECT_TRUE(std::isnan(QNaN.convertToFloat()));
}

TEST(APFloatTest, IEEEsingleToFloat) {
  APFloat FPosZero(0.0F);
  APFloat FPosZeroToFloat(FPosZero.convertToFloat());
  EXPECT_TRUE(FPosZeroToFloat.isPosZero());
  APFloat FNegZero(-0.0F);
  APFloat FNegZeroToFloat(FNegZero.convertToFloat());
  EXPECT_TRUE(FNegZeroToFloat.isNegZero());

  APFloat FOne(1.0F);
  EXPECT_EQ(1.0F, FOne.convertToFloat());
  APFloat FPosLargest = APFloat::getLargest(APFloat::IEEEsingle(), false);
  EXPECT_EQ(std::numeric_limits<float>::max(), FPosLargest.convertToFloat());
  APFloat FNegLargest = APFloat::getLargest(APFloat::IEEEsingle(), true);
  EXPECT_EQ(-std::numeric_limits<float>::max(), FNegLargest.convertToFloat());
  APFloat FPosSmallest =
      APFloat::getSmallestNormalized(APFloat::IEEEsingle(), false);
  EXPECT_EQ(std::numeric_limits<float>::min(), FPosSmallest.convertToFloat());
  APFloat FNegSmallest =
      APFloat::getSmallestNormalized(APFloat::IEEEsingle(), true);
  EXPECT_EQ(-std::numeric_limits<float>::min(), FNegSmallest.convertToFloat());

  APFloat FSmallestDenorm = APFloat::getSmallest(APFloat::IEEEsingle(), false);
  EXPECT_EQ(std::numeric_limits<float>::denorm_min(),
            FSmallestDenorm.convertToFloat());
  APFloat FLargestDenorm(APFloat::IEEEsingle(), "0x1.FFFFFEp-126");
  EXPECT_EQ(/*0x1.FFFFFEp-126*/ 2.3509885615147286e-38F,
            FLargestDenorm.convertToFloat());

  APFloat FPosInf = APFloat::getInf(APFloat::IEEEsingle());
  EXPECT_EQ(std::numeric_limits<float>::infinity(), FPosInf.convertToFloat());
  APFloat FNegInf = APFloat::getInf(APFloat::IEEEsingle(), true);
  EXPECT_EQ(-std::numeric_limits<float>::infinity(), FNegInf.convertToFloat());
  APFloat FQNaN = APFloat::getQNaN(APFloat::IEEEsingle());
  EXPECT_TRUE(std::isnan(FQNaN.convertToFloat()));
}

TEST(APFloatTest, IEEEhalfToFloat) {
  APFloat HPosZero = APFloat::getZero(APFloat::IEEEhalf());
  APFloat HPosZeroToFloat(HPosZero.convertToFloat());
  EXPECT_TRUE(HPosZeroToFloat.isPosZero());
  APFloat HNegZero = APFloat::getZero(APFloat::IEEEhalf(), true);
  APFloat HNegZeroToFloat(HNegZero.convertToFloat());
  EXPECT_TRUE(HNegZeroToFloat.isNegZero());

  APFloat HOne(APFloat::IEEEhalf(), "1.0");
  EXPECT_EQ(1.0F, HOne.convertToFloat());
  APFloat HPosLargest = APFloat::getLargest(APFloat::IEEEhalf(), false);
  EXPECT_EQ(/*0x1.FFCp15*/ 65504.0F, HPosLargest.convertToFloat());
  APFloat HNegLargest = APFloat::getLargest(APFloat::IEEEhalf(), true);
  EXPECT_EQ(/*-0x1.FFCp15*/ -65504.0F, HNegLargest.convertToFloat());
  APFloat HPosSmallest =
      APFloat::getSmallestNormalized(APFloat::IEEEhalf(), false);
  EXPECT_EQ(/*0x1.p-14*/ 6.103515625e-05F, HPosSmallest.convertToFloat());
  APFloat HNegSmallest =
      APFloat::getSmallestNormalized(APFloat::IEEEhalf(), true);
  EXPECT_EQ(/*-0x1.p-14*/ -6.103515625e-05F, HNegSmallest.convertToFloat());

  APFloat HSmallestDenorm = APFloat::getSmallest(APFloat::IEEEhalf(), false);
  EXPECT_EQ(/*0x1.p-24*/ 5.960464477539063e-08F,
            HSmallestDenorm.convertToFloat());
  APFloat HLargestDenorm(APFloat::IEEEhalf(), "0x1.FFCp-14");
  EXPECT_EQ(/*0x1.FFCp-14*/ 0.00012201070785522461F,
            HLargestDenorm.convertToFloat());

  APFloat HPosInf = APFloat::getInf(APFloat::IEEEhalf());
  EXPECT_EQ(std::numeric_limits<float>::infinity(), HPosInf.convertToFloat());
  APFloat HNegInf = APFloat::getInf(APFloat::IEEEhalf(), true);
  EXPECT_EQ(-std::numeric_limits<float>::infinity(), HNegInf.convertToFloat());
  APFloat HQNaN = APFloat::getQNaN(APFloat::IEEEhalf());
  EXPECT_TRUE(std::isnan(HQNaN.convertToFloat()));
}

TEST(APFloatTest, BFloatToFloat) {
  APFloat BPosZero = APFloat::getZero(APFloat::BFloat());
  APFloat BPosZeroToDouble(BPosZero.convertToFloat());
  EXPECT_TRUE(BPosZeroToDouble.isPosZero());
  APFloat BNegZero = APFloat::getZero(APFloat::BFloat(), true);
  APFloat BNegZeroToDouble(BNegZero.convertToFloat());
  EXPECT_TRUE(BNegZeroToDouble.isNegZero());

  APFloat BOne(APFloat::BFloat(), "1.0");
  EXPECT_EQ(1.0F, BOne.convertToFloat());
  APFloat BPosLargest = APFloat::getLargest(APFloat::BFloat(), false);
  EXPECT_EQ(/*0x1.FEp127*/ 3.3895313892515355e+38F,
            BPosLargest.convertToFloat());
  APFloat BNegLargest = APFloat::getLargest(APFloat::BFloat(), true);
  EXPECT_EQ(/*-0x1.FEp127*/ -3.3895313892515355e+38F,
            BNegLargest.convertToFloat());
  APFloat BPosSmallest =
      APFloat::getSmallestNormalized(APFloat::BFloat(), false);
  EXPECT_EQ(/*0x1.p-126*/ 1.1754943508222875e-38F,
            BPosSmallest.convertToFloat());
  APFloat BNegSmallest =
      APFloat::getSmallestNormalized(APFloat::BFloat(), true);
  EXPECT_EQ(/*-0x1.p-126*/ -1.1754943508222875e-38F,
            BNegSmallest.convertToFloat());

  APFloat BSmallestDenorm = APFloat::getSmallest(APFloat::BFloat(), false);
  EXPECT_EQ(/*0x1.p-133*/ 9.183549615799121e-41F,
            BSmallestDenorm.convertToFloat());
  APFloat BLargestDenorm(APFloat::BFloat(), "0x1.FCp-127");
  EXPECT_EQ(/*0x1.FCp-127*/ 1.1663108012064884e-38F,
            BLargestDenorm.convertToFloat());

  APFloat BPosInf = APFloat::getInf(APFloat::BFloat());
  EXPECT_EQ(std::numeric_limits<float>::infinity(), BPosInf.convertToFloat());
  APFloat BNegInf = APFloat::getInf(APFloat::BFloat(), true);
  EXPECT_EQ(-std::numeric_limits<float>::infinity(), BNegInf.convertToFloat());
  APFloat BQNaN = APFloat::getQNaN(APFloat::BFloat());
  EXPECT_TRUE(std::isnan(BQNaN.convertToFloat()));
}

TEST(APFloatTest, Float8E5M2ToFloat) {
  APFloat PosZero = APFloat::getZero(APFloat::Float8E5M2());
  APFloat PosZeroToFloat(PosZero.convertToFloat());
  EXPECT_TRUE(PosZeroToFloat.isPosZero());
  APFloat NegZero = APFloat::getZero(APFloat::Float8E5M2(), true);
  APFloat NegZeroToFloat(NegZero.convertToFloat());
  EXPECT_TRUE(NegZeroToFloat.isNegZero());

  APFloat One(APFloat::Float8E5M2(), "1.0");
  EXPECT_EQ(1.0F, One.convertToFloat());
  APFloat Two(APFloat::Float8E5M2(), "2.0");
  EXPECT_EQ(2.0F, Two.convertToFloat());

  APFloat PosLargest = APFloat::getLargest(APFloat::Float8E5M2(), false);
  EXPECT_EQ(5.734400e+04, PosLargest.convertToFloat());
  APFloat NegLargest = APFloat::getLargest(APFloat::Float8E5M2(), true);
  EXPECT_EQ(-5.734400e+04, NegLargest.convertToFloat());
  APFloat PosSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E5M2(), false);
  EXPECT_EQ(0x1.p-14, PosSmallest.convertToFloat());
  APFloat NegSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E5M2(), true);
  EXPECT_EQ(-0x1.p-14, NegSmallest.convertToFloat());

  APFloat SmallestDenorm = APFloat::getSmallest(APFloat::Float8E5M2(), false);
  EXPECT_TRUE(SmallestDenorm.isDenormal());
  EXPECT_EQ(0x1.p-16, SmallestDenorm.convertToFloat());

  APFloat PosInf = APFloat::getInf(APFloat::Float8E5M2());
  EXPECT_EQ(std::numeric_limits<float>::infinity(), PosInf.convertToFloat());
  APFloat NegInf = APFloat::getInf(APFloat::Float8E5M2(), true);
  EXPECT_EQ(-std::numeric_limits<float>::infinity(), NegInf.convertToFloat());
  APFloat QNaN = APFloat::getQNaN(APFloat::Float8E5M2());
  EXPECT_TRUE(std::isnan(QNaN.convertToFloat()));
}

TEST(APFloatTest, Float8E4M3FNToFloat) {
  APFloat PosZero = APFloat::getZero(APFloat::Float8E4M3FN());
  APFloat PosZeroToFloat(PosZero.convertToFloat());
  EXPECT_TRUE(PosZeroToFloat.isPosZero());
  APFloat NegZero = APFloat::getZero(APFloat::Float8E4M3FN(), true);
  APFloat NegZeroToFloat(NegZero.convertToFloat());
  EXPECT_TRUE(NegZeroToFloat.isNegZero());

  APFloat One(APFloat::Float8E4M3FN(), "1.0");
  EXPECT_EQ(1.0F, One.convertToFloat());
  APFloat Two(APFloat::Float8E4M3FN(), "2.0");
  EXPECT_EQ(2.0F, Two.convertToFloat());

  APFloat PosLargest = APFloat::getLargest(APFloat::Float8E4M3FN(), false);
  EXPECT_EQ(448., PosLargest.convertToFloat());
  APFloat NegLargest = APFloat::getLargest(APFloat::Float8E4M3FN(), true);
  EXPECT_EQ(-448, NegLargest.convertToFloat());
  APFloat PosSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E4M3FN(), false);
  EXPECT_EQ(0x1.p-6, PosSmallest.convertToFloat());
  APFloat NegSmallest =
      APFloat::getSmallestNormalized(APFloat::Float8E4M3FN(), true);
  EXPECT_EQ(-0x1.p-6, NegSmallest.convertToFloat());

  APFloat SmallestDenorm = APFloat::getSmallest(APFloat::Float8E4M3FN(), false);
  EXPECT_TRUE(SmallestDenorm.isDenormal());
  EXPECT_EQ(0x1.p-9, SmallestDenorm.convertToFloat());

  APFloat QNaN = APFloat::getQNaN(APFloat::Float8E4M3FN());
  EXPECT_TRUE(std::isnan(QNaN.convertToFloat()));
}

TEST(APFloatTest, FloatTF32ToFloat) {
  APFloat PosZero = APFloat::getZero(APFloat::FloatTF32());
  APFloat PosZeroToFloat(PosZero.convertToFloat());
  EXPECT_TRUE(PosZeroToFloat.isPosZero());
  APFloat NegZero = APFloat::getZero(APFloat::FloatTF32(), true);
  APFloat NegZeroToFloat(NegZero.convertToFloat());
  EXPECT_TRUE(NegZeroToFloat.isNegZero());

  APFloat One(APFloat::FloatTF32(), "1.0");
  EXPECT_EQ(1.0F, One.convertToFloat());
  APFloat Two(APFloat::FloatTF32(), "2.0");
  EXPECT_EQ(2.0F, Two.convertToFloat());

  APFloat PosLargest = APFloat::getLargest(APFloat::FloatTF32(), false);
  EXPECT_EQ(3.40116213421e+38F, PosLargest.convertToFloat());

  APFloat NegLargest = APFloat::getLargest(APFloat::FloatTF32(), true);
  EXPECT_EQ(-3.40116213421e+38F, NegLargest.convertToFloat());

  APFloat PosSmallest =
      APFloat::getSmallestNormalized(APFloat::FloatTF32(), false);
  EXPECT_EQ(/*0x1.p-126*/ 1.1754943508222875e-38F,
            PosSmallest.convertToFloat());
  APFloat NegSmallest =
      APFloat::getSmallestNormalized(APFloat::FloatTF32(), true);
  EXPECT_EQ(/*-0x1.p-126*/ -1.1754943508222875e-38F,
            NegSmallest.convertToFloat());

  APFloat SmallestDenorm = APFloat::getSmallest(APFloat::FloatTF32(), false);
  EXPECT_TRUE(SmallestDenorm.isDenormal());
  EXPECT_EQ(0x0.004p-126, SmallestDenorm.convertToFloat());

  APFloat QNaN = APFloat::getQNaN(APFloat::FloatTF32());
  EXPECT_TRUE(std::isnan(QNaN.convertToFloat()));
}

} // namespace
