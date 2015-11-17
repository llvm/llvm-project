//===- llvm/unittest/ADT/APFloat.cpp - APFloat unit tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <cmath>
#include <ostream>
#include <string>

using namespace llvm;

static double convertToDoubleFromString(const char *Str) {
  llvm::APFloat F(0.0);
  F.convertFromString(Str, llvm::APFloat::rmNearestTiesToEven);
  return F.convertToDouble();
}

static std::string convertToString(double d, unsigned Prec, unsigned Pad) {
  llvm::SmallVector<char, 100> Buffer;
  llvm::APFloat F(d);
  F.toString(Buffer, Prec, Pad);
  return std::string(Buffer.data(), Buffer.size());
}

namespace {

TEST(APFloatTest, isSignaling) {
  // We test qNaN, -qNaN, +sNaN, -sNaN with and without payloads. *NOTE* The
  // positive/negative distinction is included only since the getQNaN/getSNaN
  // API provides the option.
  APInt payload = APInt::getOneBitSet(4, 2);
  EXPECT_FALSE(APFloat::getQNaN(APFloat::IEEEsingle, false).isSignaling());
  EXPECT_FALSE(APFloat::getQNaN(APFloat::IEEEsingle, true).isSignaling());
  EXPECT_FALSE(APFloat::getQNaN(APFloat::IEEEsingle, false, &payload).isSignaling());
  EXPECT_FALSE(APFloat::getQNaN(APFloat::IEEEsingle, true, &payload).isSignaling());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, false).isSignaling());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, true).isSignaling());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, false, &payload).isSignaling());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, true, &payload).isSignaling());
}

TEST(APFloatTest, next) {

  APFloat test(APFloat::IEEEquad, APFloat::uninitialized);
  APFloat expected(APFloat::IEEEquad, APFloat::uninitialized);

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
  test = APFloat::getInf(APFloat::IEEEquad, false);
  expected = APFloat::getInf(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+inf) = -nextUp(-inf) = -(-getLargest()) = getLargest()
  test = APFloat::getInf(APFloat::IEEEquad, false);
  expected = APFloat::getLargest(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-inf) = -getLargest()
  test = APFloat::getInf(APFloat::IEEEquad, true);
  expected = APFloat::getLargest(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-inf) = -nextUp(+inf) = -(+inf) = -inf.
  test = APFloat::getInf(APFloat::IEEEquad, true);
  expected = APFloat::getInf(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity() && test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(getLargest()) = +inf
  test = APFloat::getLargest(APFloat::IEEEquad, false);
  expected = APFloat::getInf(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity() && !test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(getLargest()) = -nextUp(-getLargest())
  //                        = -(-getLargest() + inc)
  //                        = getLargest() - inc.
  test = APFloat::getLargest(APFloat::IEEEquad, false);
  expected = APFloat(APFloat::IEEEquad,
                     "0x1.fffffffffffffffffffffffffffep+16383");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isInfinity() && !test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-getLargest()) = -getLargest() + inc.
  test = APFloat::getLargest(APFloat::IEEEquad, true);
  expected = APFloat(APFloat::IEEEquad,
                     "-0x1.fffffffffffffffffffffffffffep+16383");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-getLargest()) = -nextUp(getLargest()) = -(inf) = -inf.
  test = APFloat::getLargest(APFloat::IEEEquad, true);
  expected = APFloat::getInf(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity() && test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(getSmallest()) = getSmallest() + inc.
  test = APFloat(APFloat::IEEEquad, "0x0.0000000000000000000000000001p-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "0x0.0000000000000000000000000002p-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(getSmallest()) = -nextUp(-getSmallest()) = -(-0) = +0.
  test = APFloat(APFloat::IEEEquad, "0x0.0000000000000000000000000001p-16382");
  expected = APFloat::getZero(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isZero() && !test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-getSmallest()) = -0.
  test = APFloat(APFloat::IEEEquad, "-0x0.0000000000000000000000000001p-16382");
  expected = APFloat::getZero(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isZero() && test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-getSmallest()) = -nextUp(getSmallest()) = -getSmallest() - inc.
  test = APFloat(APFloat::IEEEquad, "-0x0.0000000000000000000000000001p-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x0.0000000000000000000000000002p-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(qNaN) = qNaN
  test = APFloat::getQNaN(APFloat::IEEEquad, false);
  expected = APFloat::getQNaN(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(qNaN) = qNaN
  test = APFloat::getQNaN(APFloat::IEEEquad, false);
  expected = APFloat::getQNaN(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(sNaN) = qNaN
  test = APFloat::getSNaN(APFloat::IEEEquad, false);
  expected = APFloat::getQNaN(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opInvalidOp);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(sNaN) = qNaN
  test = APFloat::getSNaN(APFloat::IEEEquad, false);
  expected = APFloat::getQNaN(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(true), APFloat::opInvalidOp);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+0) = +getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad, false);
  expected = APFloat::getSmallest(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+0) = -nextUp(-0) = -getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad, false);
  expected = APFloat::getSmallest(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-0) = +getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad, true);
  expected = APFloat::getSmallest(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-0) = -nextUp(0) = -getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad, true);
  expected = APFloat::getSmallest(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 2. Binade Boundary Tests.

  // 2a. Test denormal <-> normal binade boundaries.
  //     * nextUp(+Largest Denormal) -> +Smallest Normal.
  //     * nextDown(-Largest Denormal) -> -Smallest Normal.
  //     * nextUp(-Smallest Normal) -> -Largest Denormal.
  //     * nextDown(+Smallest Normal) -> +Largest Denormal.

  // nextUp(+Largest Denormal) -> +Smallest Normal.
  test = APFloat(APFloat::IEEEquad, "0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "0x1.0000000000000000000000000000p-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Largest Denormal) -> -Smallest Normal.
  test = APFloat(APFloat::IEEEquad,
                 "-0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x1.0000000000000000000000000000p-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-Smallest Normal) -> -LargestDenormal.
  test = APFloat(APFloat::IEEEquad,
                 "-0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x0.ffffffffffffffffffffffffffffp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Smallest Normal) -> +Largest Denormal.
  test = APFloat(APFloat::IEEEquad,
                 "+0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad,
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
  test = APFloat(APFloat::IEEEquad, "-0x1p+1");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x1.ffffffffffffffffffffffffffffp+0");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
  test = APFloat(APFloat::IEEEquad, "0x1p+1");
  expected = APFloat(APFloat::IEEEquad, "0x1.ffffffffffffffffffffffffffffp+0");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+Normal Binade Boundary - 1) -> +Normal Binade Boundary.
  test = APFloat(APFloat::IEEEquad, "0x1.ffffffffffffffffffffffffffffp+0");
  expected = APFloat(APFloat::IEEEquad, "0x1p+1");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Normal Binade Boundary + 1) -> -Normal Binade Boundary.
  test = APFloat(APFloat::IEEEquad, "-0x1.ffffffffffffffffffffffffffffp+0");
  expected = APFloat(APFloat::IEEEquad, "-0x1p+1");
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
  test = APFloat(APFloat::IEEEquad, "-0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x0.fffffffffffffffffffffffffffep-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Largest Denormal) -> +Largest Denormal - inc.
  test = APFloat(APFloat::IEEEquad, "0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "0x0.fffffffffffffffffffffffffffep-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+Smallest Normal) -> +Smallest Normal + inc.
  test = APFloat(APFloat::IEEEquad, "0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "0x1.0000000000000000000000000001p-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Smallest Normal) -> -Smallest Normal - inc.
  test = APFloat(APFloat::IEEEquad, "-0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad,
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
  test = APFloat(APFloat::IEEEquad, "-0x1p-16381");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x1.ffffffffffffffffffffffffffffp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-0x1.ffffffffffffffffffffffffffffp-16382) ->
  //         -0x1p-16381
  test = APFloat(APFloat::IEEEquad, "-0x1.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad, "-0x1p-16381");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(0x1.ffffffffffffffffffffffffffffp-16382) -> 0x1p-16381
  test = APFloat(APFloat::IEEEquad, "0x1.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad, "0x1p-16381");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(0x1p-16381) -> 0x1.ffffffffffffffffffffffffffffp-16382
  test = APFloat(APFloat::IEEEquad, "0x1p-16381");
  expected = APFloat(APFloat::IEEEquad,
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
  test = APFloat(APFloat::IEEEquad,
                 "0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad,
                 "0x0.ffffffffffffffffffffffff000dp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Denormal) -> +Denormal.
  test = APFloat(APFloat::IEEEquad,
                 "0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad,
                 "0x0.ffffffffffffffffffffffff000bp-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-Denormal) -> -Denormal.
  test = APFloat(APFloat::IEEEquad,
                 "-0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad,
                 "-0x0.ffffffffffffffffffffffff000bp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Denormal) -> -Denormal
  test = APFloat(APFloat::IEEEquad,
                 "-0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad,
                 "-0x0.ffffffffffffffffffffffff000dp-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+Normal) -> +Normal.
  test = APFloat(APFloat::IEEEquad,
                 "0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad,
                 "0x1.ffffffffffffffffffffffff000dp-16000");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Normal) -> +Normal.
  test = APFloat(APFloat::IEEEquad,
                 "0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad,
                 "0x1.ffffffffffffffffffffffff000bp-16000");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-Normal) -> -Normal.
  test = APFloat(APFloat::IEEEquad,
                 "-0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad,
                 "-0x1.ffffffffffffffffffffffff000bp-16000");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Normal) -> -Normal.
  test = APFloat(APFloat::IEEEquad,
                 "-0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad,
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
    APFloat f1(APFloat::IEEEdouble,  "-0x1p-1074");
    APFloat f2(APFloat::IEEEdouble, "+0x1p-1074");
    APFloat f3(0.0);
    f1.fusedMultiplyAdd(f2, f3, APFloat::rmNearestTiesToEven);
    EXPECT_TRUE(f1.isNegative() && f1.isZero());
  }

  // Test x87 extended precision case from http://llvm.org/PR20728.
  {
    APFloat M1(APFloat::x87DoubleExtended, 1.0);
    APFloat M2(APFloat::x87DoubleExtended, 1.0);
    APFloat A(APFloat::x87DoubleExtended, 3.0);

    bool losesInfo = false;
    M1.fusedMultiplyAdd(M1, A, APFloat::rmNearestTiesToEven);
    M1.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven, &losesInfo);
    EXPECT_FALSE(losesInfo);
    EXPECT_EQ(4.0f, M1.convertToFloat());
  }
}

TEST(APFloatTest, MinNum) {
  APFloat f1(1.0);
  APFloat f2(2.0);
  APFloat nan = APFloat::getNaN(APFloat::IEEEdouble);

  EXPECT_EQ(1.0, minnum(f1, f2).convertToDouble());
  EXPECT_EQ(1.0, minnum(f2, f1).convertToDouble());
  EXPECT_EQ(1.0, minnum(f1, nan).convertToDouble());
  EXPECT_EQ(1.0, minnum(nan, f1).convertToDouble());
}

TEST(APFloatTest, MaxNum) {
  APFloat f1(1.0);
  APFloat f2(2.0);
  APFloat nan = APFloat::getNaN(APFloat::IEEEdouble);

  EXPECT_EQ(2.0, maxnum(f1, f2).convertToDouble());
  EXPECT_EQ(2.0, maxnum(f2, f1).convertToDouble());
  EXPECT_EQ(1.0, maxnum(f1, nan).convertToDouble());
  EXPECT_EQ(1.0, minnum(nan, f1).convertToDouble());
}

TEST(APFloatTest, Denormal) {
  APFloat::roundingMode rdmd = APFloat::rmNearestTiesToEven;

  // Test single precision
  {
    const char *MinNormalStr = "1.17549435082228750797e-38";
    EXPECT_FALSE(APFloat(APFloat::IEEEsingle, MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::IEEEsingle, 0.0).isDenormal());

    APFloat Val2(APFloat::IEEEsingle, 2.0e0);
    APFloat T(APFloat::IEEEsingle, MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
  }

  // Test double precision
  {
    const char *MinNormalStr = "2.22507385850720138309e-308";
    EXPECT_FALSE(APFloat(APFloat::IEEEdouble, MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::IEEEdouble, 0.0).isDenormal());

    APFloat Val2(APFloat::IEEEdouble, 2.0e0);
    APFloat T(APFloat::IEEEdouble, MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
  }

  // Test Intel double-ext
  {
    const char *MinNormalStr = "3.36210314311209350626e-4932";
    EXPECT_FALSE(APFloat(APFloat::x87DoubleExtended, MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::x87DoubleExtended, 0.0).isDenormal());

    APFloat Val2(APFloat::x87DoubleExtended, 2.0e0);
    APFloat T(APFloat::x87DoubleExtended, MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
  }

  // Test quadruple precision
  {
    const char *MinNormalStr = "3.36210314311209350626267781732175260e-4932";
    EXPECT_FALSE(APFloat(APFloat::IEEEquad, MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::IEEEquad, 0.0).isDenormal());

    APFloat Val2(APFloat::IEEEquad, 2.0e0);
    APFloat T(APFloat::IEEEquad, MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
  }
}

TEST(APFloatTest, Zero) {
  EXPECT_EQ(0.0f,  APFloat(0.0f).convertToFloat());
  EXPECT_EQ(-0.0f, APFloat(-0.0f).convertToFloat());
  EXPECT_TRUE(APFloat(-0.0f).isNegative());

  EXPECT_EQ(0.0,  APFloat(0.0).convertToDouble());
  EXPECT_EQ(-0.0, APFloat(-0.0).convertToDouble());
  EXPECT_TRUE(APFloat(-0.0).isNegative());
}

TEST(APFloatTest, DecimalStringsWithoutNullTerminators) {
  // Make sure that we can parse strings without null terminators.
  // rdar://14323230.
  APFloat Val(APFloat::IEEEdouble);
  Val.convertFromString(StringRef("0.00", 3),
                        llvm::APFloat::rmNearestTiesToEven);
  EXPECT_EQ(Val.convertToDouble(), 0.0);
  Val.convertFromString(StringRef("0.01", 3),
                        llvm::APFloat::rmNearestTiesToEven);
  EXPECT_EQ(Val.convertToDouble(), 0.0);
  Val.convertFromString(StringRef("0.09", 3),
                        llvm::APFloat::rmNearestTiesToEven);
  EXPECT_EQ(Val.convertToDouble(), 0.0);
  Val.convertFromString(StringRef("0.095", 4),
                        llvm::APFloat::rmNearestTiesToEven);
  EXPECT_EQ(Val.convertToDouble(), 0.09);
  Val.convertFromString(StringRef("0.00e+3", 7),
                        llvm::APFloat::rmNearestTiesToEven);
  EXPECT_EQ(Val.convertToDouble(), 0.00);
  Val.convertFromString(StringRef("0e+3", 4),
                        llvm::APFloat::rmNearestTiesToEven);
  EXPECT_EQ(Val.convertToDouble(), 0.00);

}

TEST(APFloatTest, fromZeroDecimalString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  ".0").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+.0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-.0").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.0").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.0").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "00000.").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+00000.").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-00000.").convertToDouble());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, ".00000").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+.00000").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-.00000").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0000.00000").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0000.00000").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0000.00000").convertToDouble());
}

TEST(APFloatTest, fromZeroDecimalSingleExponentString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,   "0e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble,  "+0e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble,  "-0e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0e-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,   "0.e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble,  "+0.e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble,  "-0.e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.e-1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,   ".0e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble,  "+.0e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble,  "-.0e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  ".0e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+.0e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-.0e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  ".0e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+.0e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-.0e-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,   "0.0e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble,  "+0.0e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble,  "-0.0e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.0e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.0e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.0e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.0e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.0e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.0e-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "000.0000e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+000.0000e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-000.0000e+1").convertToDouble());
}

TEST(APFloatTest, fromZeroDecimalLargeExponentString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0e1234").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0e1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0e1234").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0e+1234").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0e+1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0e+1234").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0e-1234").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0e-1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0e-1234").convertToDouble());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "000.0000e1234").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "000.0000e-1234").convertToDouble());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, StringRef("0e1234\02", 6)).convertToDouble());
}

TEST(APFloatTest, fromZeroHexadecimalString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x.0p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x.0p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x.0p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x.0p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x.0p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x.0p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x.0p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x.0p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x.0p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.0p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.0p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.0p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.0p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.0p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.0p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.0p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.0p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.0p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x00000.p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x0000.00000p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x.00000p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x0.p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x0p1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x00000.p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x0000.00000p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x.00000p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x0.p1234").convertToDouble());
}

TEST(APFloatTest, fromDecimalString) {
  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble, "1").convertToDouble());
  EXPECT_EQ(2.0,      APFloat(APFloat::IEEEdouble, "2.").convertToDouble());
  EXPECT_EQ(0.5,      APFloat(APFloat::IEEEdouble, ".5").convertToDouble());
  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble, "1.0").convertToDouble());
  EXPECT_EQ(-2.0,     APFloat(APFloat::IEEEdouble, "-2").convertToDouble());
  EXPECT_EQ(-4.0,     APFloat(APFloat::IEEEdouble, "-4.").convertToDouble());
  EXPECT_EQ(-0.5,     APFloat(APFloat::IEEEdouble, "-.5").convertToDouble());
  EXPECT_EQ(-1.5,     APFloat(APFloat::IEEEdouble, "-1.5").convertToDouble());
  EXPECT_EQ(1.25e12,  APFloat(APFloat::IEEEdouble, "1.25e12").convertToDouble());
  EXPECT_EQ(1.25e+12, APFloat(APFloat::IEEEdouble, "1.25e+12").convertToDouble());
  EXPECT_EQ(1.25e-12, APFloat(APFloat::IEEEdouble, "1.25e-12").convertToDouble());
  EXPECT_EQ(1024.0,   APFloat(APFloat::IEEEdouble, "1024.").convertToDouble());
  EXPECT_EQ(1024.05,  APFloat(APFloat::IEEEdouble, "1024.05000").convertToDouble());
  EXPECT_EQ(0.05,     APFloat(APFloat::IEEEdouble, ".05000").convertToDouble());
  EXPECT_EQ(2.0,      APFloat(APFloat::IEEEdouble, "2.").convertToDouble());
  EXPECT_EQ(2.0e2,    APFloat(APFloat::IEEEdouble, "2.e2").convertToDouble());
  EXPECT_EQ(2.0e+2,   APFloat(APFloat::IEEEdouble, "2.e+2").convertToDouble());
  EXPECT_EQ(2.0e-2,   APFloat(APFloat::IEEEdouble, "2.e-2").convertToDouble());
  EXPECT_EQ(2.05e2,    APFloat(APFloat::IEEEdouble, "002.05000e2").convertToDouble());
  EXPECT_EQ(2.05e+2,   APFloat(APFloat::IEEEdouble, "002.05000e+2").convertToDouble());
  EXPECT_EQ(2.05e-2,   APFloat(APFloat::IEEEdouble, "002.05000e-2").convertToDouble());
  EXPECT_EQ(2.05e12,   APFloat(APFloat::IEEEdouble, "002.05000e12").convertToDouble());
  EXPECT_EQ(2.05e+12,  APFloat(APFloat::IEEEdouble, "002.05000e+12").convertToDouble());
  EXPECT_EQ(2.05e-12,  APFloat(APFloat::IEEEdouble, "002.05000e-12").convertToDouble());

  // These are "carefully selected" to overflow the fast log-base
  // calculations in APFloat.cpp
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble, "99e99999").isInfinity());
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble, "-99e99999").isInfinity());
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble, "1e-99999").isPosZero());
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble, "-1e-99999").isNegZero());

  EXPECT_EQ(2.71828, convertToDoubleFromString("2.71828"));
}

TEST(APFloatTest, fromHexadecimalString) {
  EXPECT_EQ( 1.0, APFloat(APFloat::IEEEdouble,  "0x1p0").convertToDouble());
  EXPECT_EQ(+1.0, APFloat(APFloat::IEEEdouble, "+0x1p0").convertToDouble());
  EXPECT_EQ(-1.0, APFloat(APFloat::IEEEdouble, "-0x1p0").convertToDouble());

  EXPECT_EQ( 1.0, APFloat(APFloat::IEEEdouble,  "0x1p+0").convertToDouble());
  EXPECT_EQ(+1.0, APFloat(APFloat::IEEEdouble, "+0x1p+0").convertToDouble());
  EXPECT_EQ(-1.0, APFloat(APFloat::IEEEdouble, "-0x1p+0").convertToDouble());

  EXPECT_EQ( 1.0, APFloat(APFloat::IEEEdouble,  "0x1p-0").convertToDouble());
  EXPECT_EQ(+1.0, APFloat(APFloat::IEEEdouble, "+0x1p-0").convertToDouble());
  EXPECT_EQ(-1.0, APFloat(APFloat::IEEEdouble, "-0x1p-0").convertToDouble());


  EXPECT_EQ( 2.0, APFloat(APFloat::IEEEdouble,  "0x1p1").convertToDouble());
  EXPECT_EQ(+2.0, APFloat(APFloat::IEEEdouble, "+0x1p1").convertToDouble());
  EXPECT_EQ(-2.0, APFloat(APFloat::IEEEdouble, "-0x1p1").convertToDouble());

  EXPECT_EQ( 2.0, APFloat(APFloat::IEEEdouble,  "0x1p+1").convertToDouble());
  EXPECT_EQ(+2.0, APFloat(APFloat::IEEEdouble, "+0x1p+1").convertToDouble());
  EXPECT_EQ(-2.0, APFloat(APFloat::IEEEdouble, "-0x1p+1").convertToDouble());

  EXPECT_EQ( 0.5, APFloat(APFloat::IEEEdouble,  "0x1p-1").convertToDouble());
  EXPECT_EQ(+0.5, APFloat(APFloat::IEEEdouble, "+0x1p-1").convertToDouble());
  EXPECT_EQ(-0.5, APFloat(APFloat::IEEEdouble, "-0x1p-1").convertToDouble());


  EXPECT_EQ( 3.0, APFloat(APFloat::IEEEdouble,  "0x1.8p1").convertToDouble());
  EXPECT_EQ(+3.0, APFloat(APFloat::IEEEdouble, "+0x1.8p1").convertToDouble());
  EXPECT_EQ(-3.0, APFloat(APFloat::IEEEdouble, "-0x1.8p1").convertToDouble());

  EXPECT_EQ( 3.0, APFloat(APFloat::IEEEdouble,  "0x1.8p+1").convertToDouble());
  EXPECT_EQ(+3.0, APFloat(APFloat::IEEEdouble, "+0x1.8p+1").convertToDouble());
  EXPECT_EQ(-3.0, APFloat(APFloat::IEEEdouble, "-0x1.8p+1").convertToDouble());

  EXPECT_EQ( 0.75, APFloat(APFloat::IEEEdouble,  "0x1.8p-1").convertToDouble());
  EXPECT_EQ(+0.75, APFloat(APFloat::IEEEdouble, "+0x1.8p-1").convertToDouble());
  EXPECT_EQ(-0.75, APFloat(APFloat::IEEEdouble, "-0x1.8p-1").convertToDouble());


  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble,  "0x1000.000p1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble, "+0x1000.000p1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble, "-0x1000.000p1").convertToDouble());

  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble,  "0x1000.000p+1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble, "+0x1000.000p+1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble, "-0x1000.000p+1").convertToDouble());

  EXPECT_EQ( 2048.0, APFloat(APFloat::IEEEdouble,  "0x1000.000p-1").convertToDouble());
  EXPECT_EQ(+2048.0, APFloat(APFloat::IEEEdouble, "+0x1000.000p-1").convertToDouble());
  EXPECT_EQ(-2048.0, APFloat(APFloat::IEEEdouble, "-0x1000.000p-1").convertToDouble());


  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble,  "0x1000p1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble, "+0x1000p1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble, "-0x1000p1").convertToDouble());

  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble,  "0x1000p+1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble, "+0x1000p+1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble, "-0x1000p+1").convertToDouble());

  EXPECT_EQ( 2048.0, APFloat(APFloat::IEEEdouble,  "0x1000p-1").convertToDouble());
  EXPECT_EQ(+2048.0, APFloat(APFloat::IEEEdouble, "+0x1000p-1").convertToDouble());
  EXPECT_EQ(-2048.0, APFloat(APFloat::IEEEdouble, "-0x1000p-1").convertToDouble());


  EXPECT_EQ( 16384.0, APFloat(APFloat::IEEEdouble,  "0x10p10").convertToDouble());
  EXPECT_EQ(+16384.0, APFloat(APFloat::IEEEdouble, "+0x10p10").convertToDouble());
  EXPECT_EQ(-16384.0, APFloat(APFloat::IEEEdouble, "-0x10p10").convertToDouble());

  EXPECT_EQ( 16384.0, APFloat(APFloat::IEEEdouble,  "0x10p+10").convertToDouble());
  EXPECT_EQ(+16384.0, APFloat(APFloat::IEEEdouble, "+0x10p+10").convertToDouble());
  EXPECT_EQ(-16384.0, APFloat(APFloat::IEEEdouble, "-0x10p+10").convertToDouble());

  EXPECT_EQ( 0.015625, APFloat(APFloat::IEEEdouble,  "0x10p-10").convertToDouble());
  EXPECT_EQ(+0.015625, APFloat(APFloat::IEEEdouble, "+0x10p-10").convertToDouble());
  EXPECT_EQ(-0.015625, APFloat(APFloat::IEEEdouble, "-0x10p-10").convertToDouble());

  EXPECT_EQ(1.0625, APFloat(APFloat::IEEEdouble, "0x1.1p0").convertToDouble());
  EXPECT_EQ(1.0, APFloat(APFloat::IEEEdouble, "0x1p0").convertToDouble());

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
}

TEST(APFloatTest, toInteger) {
  bool isExact = false;
  APSInt result(5, /*isUnsigned=*/true);

  EXPECT_EQ(APFloat::opOK,
            APFloat(APFloat::IEEEdouble, "10")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_TRUE(isExact);
  EXPECT_EQ(APSInt(APInt(5, 10), true), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble, "-10")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMinValue(5, true), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble, "32")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMaxValue(5, true), result);

  EXPECT_EQ(APFloat::opInexact,
            APFloat(APFloat::IEEEdouble, "7.9")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt(APInt(5, 7), true), result);

  result.setIsUnsigned(false);
  EXPECT_EQ(APFloat::opOK,
            APFloat(APFloat::IEEEdouble, "-10")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_TRUE(isExact);
  EXPECT_EQ(APSInt(APInt(5, -10, true), false), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble, "-17")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMinValue(5, false), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble, "16")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMaxValue(5, false), result);
}

static APInt nanbits(const fltSemantics &Sem,
                     bool SNaN, bool Negative, uint64_t fill) {
  APInt apfill(64, fill);
  if (SNaN)
    return APFloat::getSNaN(Sem, Negative, &apfill).bitcastToAPInt();
  else
    return APFloat::getQNaN(Sem, Negative, &apfill).bitcastToAPInt();
}

TEST(APFloatTest, makeNaN) {
  ASSERT_EQ(0x7fc00000, nanbits(APFloat::IEEEsingle, false, false, 0));
  ASSERT_EQ(0xffc00000, nanbits(APFloat::IEEEsingle, false, true, 0));
  ASSERT_EQ(0x7fc0ae72, nanbits(APFloat::IEEEsingle, false, false, 0xae72));
  ASSERT_EQ(0x7fffae72, nanbits(APFloat::IEEEsingle, false, false, 0xffffae72));
  ASSERT_EQ(0x7fa00000, nanbits(APFloat::IEEEsingle, true, false, 0));
  ASSERT_EQ(0xffa00000, nanbits(APFloat::IEEEsingle, true, true, 0));
  ASSERT_EQ(0x7f80ae72, nanbits(APFloat::IEEEsingle, true, false, 0xae72));
  ASSERT_EQ(0x7fbfae72, nanbits(APFloat::IEEEsingle, true, false, 0xffffae72));

  ASSERT_EQ(0x7ff8000000000000ULL, nanbits(APFloat::IEEEdouble, false, false, 0));
  ASSERT_EQ(0xfff8000000000000ULL, nanbits(APFloat::IEEEdouble, false, true, 0));
  ASSERT_EQ(0x7ff800000000ae72ULL, nanbits(APFloat::IEEEdouble, false, false, 0xae72));
  ASSERT_EQ(0x7fffffffffffae72ULL, nanbits(APFloat::IEEEdouble, false, false, 0xffffffffffffae72ULL));
  ASSERT_EQ(0x7ff4000000000000ULL, nanbits(APFloat::IEEEdouble, true, false, 0));
  ASSERT_EQ(0xfff4000000000000ULL, nanbits(APFloat::IEEEdouble, true, true, 0));
  ASSERT_EQ(0x7ff000000000ae72ULL, nanbits(APFloat::IEEEdouble, true, false, 0xae72));
  ASSERT_EQ(0x7ff7ffffffffae72ULL, nanbits(APFloat::IEEEdouble, true, false, 0xffffffffffffae72ULL));
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(APFloatTest, SemanticsDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEsingle, 0.0f).convertToDouble(), "Float semantics are not IEEEdouble");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, 0.0 ).convertToFloat(),  "Float semantics are not IEEEsingle");
}

TEST(APFloatTest, StringDecimalDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  ""), "Invalid string length");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+"), "String has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-"), "String has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("\0", 1)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1\0", 2)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1\02", 3)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1\02e1", 5)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1e\0", 3)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1e1\0", 4)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1e1\02", 5)), "Invalid character in exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1.0f"), "Invalid character in significand");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, ".."), "String contains multiple dots");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "..0"), "String contains multiple dots");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1.0.0"), "String contains multiple dots");
}

TEST(APFloatTest, StringDecimalSignificandDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "."), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+."), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-."), "Significand has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "e"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+e"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-e"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "e1"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+e1"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-e1"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  ".e1"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+.e1"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-.e1"), "Significand has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  ".e"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+.e"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-.e"), "Significand has no digits");
}

TEST(APFloatTest, StringDecimalExponentDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,   "1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "+1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "-1e"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,   "1.e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "+1.e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "-1.e"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,   ".1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "+.1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "-.1e"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,   "1.1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "+1.1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "-1.1e"), "Exponent has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1e+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1e-"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  ".1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, ".1e+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, ".1e-"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "1.0e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1.0e+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1.0e-"), "Exponent has no digits");
}

TEST(APFloatTest, StringHexadecimalDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x"), "Invalid string");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x"), "Invalid string");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x"), "Invalid string");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x0"), "Hex strings require an exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x0."), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x0."), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x0."), "Hex strings require an exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.0"), "Hex strings require an exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x0.0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x0.0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x0.0"), "Hex strings require an exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x\0", 3)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1\0", 4)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1\02", 5)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1\02p1", 7)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1p\0", 5)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1p1\0", 6)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1p1\02", 7)), "Invalid character in exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x1p0f"), "Invalid character in exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x..p1"), "String contains multiple dots");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x..0p1"), "String contains multiple dots");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x1.0.0p1"), "String contains multiple dots");
}

TEST(APFloatTest, StringHexadecimalSignificandDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x."), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x."), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x."), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0xp"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0xp"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0xp"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0xp+"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0xp+"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0xp+"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0xp-"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0xp-"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0xp-"), "Significand has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.p"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.p"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.p"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.p+"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.p+"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.p+"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.p-"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.p-"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.p-"), "Significand has no digits");
}

TEST(APFloatTest, StringHexadecimalExponentDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1p"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1p+"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1p-"), "Exponent has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.p"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.p+"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.p-"), "Exponent has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.1p"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.1p+"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.1p-"), "Exponent has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.1p"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.1p+"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.1p-"), "Exponent has no digits");
}
#endif
#endif

TEST(APFloatTest, exactInverse) {
  APFloat inv(0.0f);

  // Trivial operation.
  EXPECT_TRUE(APFloat(2.0).getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(0.5)));
  EXPECT_TRUE(APFloat(2.0f).getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(0.5f)));
  EXPECT_TRUE(APFloat(APFloat::IEEEquad, "2.0").getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(APFloat::IEEEquad, "0.5")));
  EXPECT_TRUE(APFloat(APFloat::PPCDoubleDouble, "2.0").getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(APFloat::PPCDoubleDouble, "0.5")));
  EXPECT_TRUE(APFloat(APFloat::x87DoubleExtended, "2.0").getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(APFloat::x87DoubleExtended, "0.5")));

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
  APFloat T(-0.5), S(3.14), R(APFloat::getLargest(APFloat::IEEEdouble)), P(0.0);

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

  P = APFloat::getZero(APFloat::IEEEdouble);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(0.0, P.convertToDouble());
  P = APFloat::getZero(APFloat::IEEEdouble, true);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(-0.0, P.convertToDouble());
  P = APFloat::getNaN(APFloat::IEEEdouble);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(std::isnan(P.convertToDouble()));
  P = APFloat::getInf(APFloat::IEEEdouble);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(std::isinf(P.convertToDouble()) && P.convertToDouble() > 0.0);
  P = APFloat::getInf(APFloat::IEEEdouble, true);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(std::isinf(P.convertToDouble()) && P.convertToDouble() < 0.0);
}
  
TEST(APFloatTest, isInteger) {
  APFloat T(-0.0);
  EXPECT_TRUE(T.isInteger());
  T = APFloat(3.14159);
  EXPECT_FALSE(T.isInteger());
  T = APFloat::getNaN(APFloat::IEEEdouble);
  EXPECT_FALSE(T.isInteger());
  T = APFloat::getInf(APFloat::IEEEdouble);
  EXPECT_FALSE(T.isInteger());
  T = APFloat::getInf(APFloat::IEEEdouble, true);
  EXPECT_FALSE(T.isInteger());
  T = APFloat::getLargest(APFloat::IEEEdouble);
  EXPECT_TRUE(T.isInteger());
}

TEST(APFloatTest, getLargest) {
  EXPECT_EQ(3.402823466e+38f, APFloat::getLargest(APFloat::IEEEsingle).convertToFloat());
  EXPECT_EQ(1.7976931348623158e+308, APFloat::getLargest(APFloat::IEEEdouble).convertToDouble());
}

TEST(APFloatTest, getSmallest) {
  APFloat test = APFloat::getSmallest(APFloat::IEEEsingle, false);
  APFloat expected = APFloat(APFloat::IEEEsingle, "0x0.000002p-126");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::IEEEsingle, true);
  expected = APFloat(APFloat::IEEEsingle, "-0x0.000002p-126");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::IEEEquad, false);
  expected = APFloat(APFloat::IEEEquad, "0x0.0000000000000000000000000001p-16382");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::IEEEquad, true);
  expected = APFloat(APFloat::IEEEquad, "-0x0.0000000000000000000000000001p-16382");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
}

TEST(APFloatTest, getSmallestNormalized) {
  APFloat test = APFloat::getSmallestNormalized(APFloat::IEEEsingle, false);
  APFloat expected = APFloat(APFloat::IEEEsingle, "0x1p-126");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallestNormalized(APFloat::IEEEsingle, true);
  expected = APFloat(APFloat::IEEEsingle, "-0x1p-126");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallestNormalized(APFloat::IEEEquad, false);
  expected = APFloat(APFloat::IEEEquad, "0x1p-16382");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallestNormalized(APFloat::IEEEquad, true);
  expected = APFloat(APFloat::IEEEquad, "-0x1p-16382");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
}

TEST(APFloatTest, getZero) {
  struct {
    const fltSemantics *semantics;
    const bool sign;
    const unsigned long long bitPattern[2];
    const unsigned bitPatternLength;
  } const GetZeroTest[] = {
    { &APFloat::IEEEhalf, false, {0, 0}, 1},
    { &APFloat::IEEEhalf, true, {0x8000ULL, 0}, 1},
    { &APFloat::IEEEsingle, false, {0, 0}, 1},
    { &APFloat::IEEEsingle, true, {0x80000000ULL, 0}, 1},
    { &APFloat::IEEEdouble, false, {0, 0}, 1},
    { &APFloat::IEEEdouble, true, {0x8000000000000000ULL, 0}, 1},
    { &APFloat::IEEEquad, false, {0, 0}, 2},
    { &APFloat::IEEEquad, true, {0, 0x8000000000000000ULL}, 2},
    { &APFloat::PPCDoubleDouble, false, {0, 0}, 2},
    { &APFloat::PPCDoubleDouble, true, {0x8000000000000000ULL, 0}, 2},
    { &APFloat::x87DoubleExtended, false, {0, 0}, 2},
    { &APFloat::x87DoubleExtended, true, {0, 0x8000ULL}, 2},
  };
  const unsigned NumGetZeroTests = 12;
  for (unsigned i = 0; i < NumGetZeroTests; ++i) {
    APFloat test = APFloat::getZero(*GetZeroTest[i].semantics,
				    GetZeroTest[i].sign);
    const char *pattern = GetZeroTest[i].sign? "-0x0p+0" : "0x0p+0";
    APFloat expected = APFloat(*GetZeroTest[i].semantics,
			       pattern);
    EXPECT_TRUE(test.isZero());
    EXPECT_TRUE(GetZeroTest[i].sign? test.isNegative() : !test.isNegative());
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
}

TEST(APFloatTest, convert) {
  bool losesInfo;
  APFloat test(APFloat::IEEEdouble, "1.0");
  test.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0f, test.convertToFloat());
  EXPECT_FALSE(losesInfo);

  test = APFloat(APFloat::x87DoubleExtended, "0x1p-53");
  test.add(APFloat(APFloat::x87DoubleExtended, "1.0"), APFloat::rmNearestTiesToEven);
  test.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0, test.convertToDouble());
  EXPECT_TRUE(losesInfo);

  test = APFloat(APFloat::IEEEquad, "0x1p-53");
  test.add(APFloat(APFloat::IEEEquad, "1.0"), APFloat::rmNearestTiesToEven);
  test.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0, test.convertToDouble());
  EXPECT_TRUE(losesInfo);

  test = APFloat(APFloat::x87DoubleExtended, "0xf.fffffffp+28");
  test.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(4294967295.0, test.convertToDouble());
  EXPECT_FALSE(losesInfo);

  test = APFloat::getSNaN(APFloat::IEEEsingle);
  APFloat X87SNaN = APFloat::getSNaN(APFloat::x87DoubleExtended);
  test.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven,
               &losesInfo);
  EXPECT_TRUE(test.bitwiseIsEqual(X87SNaN));
  EXPECT_FALSE(losesInfo);

  test = APFloat::getQNaN(APFloat::IEEEsingle);
  APFloat X87QNaN = APFloat::getQNaN(APFloat::x87DoubleExtended);
  test.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven,
               &losesInfo);
  EXPECT_TRUE(test.bitwiseIsEqual(X87QNaN));
  EXPECT_FALSE(losesInfo);

  test = APFloat::getSNaN(APFloat::x87DoubleExtended);
  test.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven,
               &losesInfo);
  EXPECT_TRUE(test.bitwiseIsEqual(X87SNaN));
  EXPECT_FALSE(losesInfo);

  test = APFloat::getQNaN(APFloat::x87DoubleExtended);
  test.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven,
               &losesInfo);
  EXPECT_TRUE(test.bitwiseIsEqual(X87QNaN));
  EXPECT_FALSE(losesInfo);
}

TEST(APFloatTest, PPCDoubleDouble) {
  APFloat test(APFloat::PPCDoubleDouble, "1.0");
  EXPECT_EQ(0x3ff0000000000000ull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x0000000000000000ull, test.bitcastToAPInt().getRawData()[1]);

  test.divide(APFloat(APFloat::PPCDoubleDouble, "3.0"), APFloat::rmNearestTiesToEven);
  EXPECT_EQ(0x3fd5555555555555ull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x3c75555555555556ull, test.bitcastToAPInt().getRawData()[1]);

  // LDBL_MAX
  test = APFloat(APFloat::PPCDoubleDouble, "1.79769313486231580793728971405301e+308");
  EXPECT_EQ(0x7fefffffffffffffull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x7c8ffffffffffffeull, test.bitcastToAPInt().getRawData()[1]);

  // LDBL_MIN
  test = APFloat(APFloat::PPCDoubleDouble, "2.00416836000897277799610805135016e-292");
  EXPECT_EQ(0x0360000000000000ull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x0000000000000000ull, test.bitcastToAPInt().getRawData()[1]);

  test = APFloat(APFloat::PPCDoubleDouble, "1.0");
  test.add(APFloat(APFloat::PPCDoubleDouble, "0x1p-105"), APFloat::rmNearestTiesToEven);
  EXPECT_EQ(0x3ff0000000000000ull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x3960000000000000ull, test.bitcastToAPInt().getRawData()[1]);

  test = APFloat(APFloat::PPCDoubleDouble, "1.0");
  test.add(APFloat(APFloat::PPCDoubleDouble, "0x1p-106"), APFloat::rmNearestTiesToEven);
  EXPECT_EQ(0x3ff0000000000000ull, test.bitcastToAPInt().getRawData()[0]);
#if 0 // XFAIL
  // This is what we would expect with a true double-double implementation
  EXPECT_EQ(0x3950000000000000ull, test.bitcastToAPInt().getRawData()[1]);
#else
  // This is what we get with our 106-bit mantissa approximation
  EXPECT_EQ(0x0000000000000000ull, test.bitcastToAPInt().getRawData()[1]);
#endif
}

TEST(APFloatTest, isNegative) {
  APFloat t(APFloat::IEEEsingle, "0x1p+0");
  EXPECT_FALSE(t.isNegative());
  t = APFloat(APFloat::IEEEsingle, "-0x1p+0");
  EXPECT_TRUE(t.isNegative());

  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, false).isNegative());
  EXPECT_TRUE(APFloat::getInf(APFloat::IEEEsingle, true).isNegative());

  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, false).isNegative());
  EXPECT_TRUE(APFloat::getZero(APFloat::IEEEsingle, true).isNegative());

  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, false).isNegative());
  EXPECT_TRUE(APFloat::getNaN(APFloat::IEEEsingle, true).isNegative());

  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, false).isNegative());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, true).isNegative());
}

TEST(APFloatTest, isNormal) {
  APFloat t(APFloat::IEEEsingle, "0x1p+0");
  EXPECT_TRUE(t.isNormal());

  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, false).isNormal());
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, false).isNormal());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, false).isNormal());
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, false).isNormal());
  EXPECT_FALSE(APFloat(APFloat::IEEEsingle, "0x1p-149").isNormal());
}

TEST(APFloatTest, isFinite) {
  APFloat t(APFloat::IEEEsingle, "0x1p+0");
  EXPECT_TRUE(t.isFinite());
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, false).isFinite());
  EXPECT_TRUE(APFloat::getZero(APFloat::IEEEsingle, false).isFinite());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, false).isFinite());
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, false).isFinite());
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle, "0x1p-149").isFinite());
}

TEST(APFloatTest, isInfinity) {
  APFloat t(APFloat::IEEEsingle, "0x1p+0");
  EXPECT_FALSE(t.isInfinity());
  EXPECT_TRUE(APFloat::getInf(APFloat::IEEEsingle, false).isInfinity());
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, false).isInfinity());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, false).isInfinity());
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, false).isInfinity());
  EXPECT_FALSE(APFloat(APFloat::IEEEsingle, "0x1p-149").isInfinity());
}

TEST(APFloatTest, isNaN) {
  APFloat t(APFloat::IEEEsingle, "0x1p+0");
  EXPECT_FALSE(t.isNaN());
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, false).isNaN());
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, false).isNaN());
  EXPECT_TRUE(APFloat::getNaN(APFloat::IEEEsingle, false).isNaN());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, false).isNaN());
  EXPECT_FALSE(APFloat(APFloat::IEEEsingle, "0x1p-149").isNaN());
}

TEST(APFloatTest, isFiniteNonZero) {
  // Test positive/negative normal value.
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle, "0x1p+0").isFiniteNonZero());
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle, "-0x1p+0").isFiniteNonZero());

  // Test positive/negative denormal value.
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle, "0x1p-149").isFiniteNonZero());
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle, "-0x1p-149").isFiniteNonZero());

  // Test +/- Infinity.
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, false).isFiniteNonZero());
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, true).isFiniteNonZero());

  // Test +/- Zero.
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, false).isFiniteNonZero());
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, true).isFiniteNonZero());

  // Test +/- qNaN. +/- dont mean anything with qNaN but paranoia can't hurt in
  // this instance.
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, false).isFiniteNonZero());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, true).isFiniteNonZero());

  // Test +/- sNaN. +/- dont mean anything with sNaN but paranoia can't hurt in
  // this instance.
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, false).isFiniteNonZero());
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, true).isFiniteNonZero());
}

TEST(APFloatTest, add) {
  // Test Special Cases against each other and normal values.

  // TODOS/NOTES:
  // 1. Since we perform only default exception handling all operations with
  // signaling NaNs should have a result that is a quiet NaN. Currently they
  // return sNaN.

  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle, false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle, true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle, false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle, true);
  APFloat QNaN = APFloat::getNaN(APFloat::IEEEsingle, false);
  APFloat SNaN = APFloat::getSNaN(APFloat::IEEEsingle, false);
  APFloat PNormalValue = APFloat(APFloat::IEEEsingle, "0x1p+0");
  APFloat MNormalValue = APFloat(APFloat::IEEEsingle, "-0x1p+0");
  APFloat PLargestValue = APFloat::getLargest(APFloat::IEEEsingle, false);
  APFloat MLargestValue = APFloat::getLargest(APFloat::IEEEsingle, true);
  APFloat PSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle, false);
  APFloat MSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle, true);
  APFloat PSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle, false);
  APFloat MSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle, true);

  const int OverflowStatus = APFloat::opOverflow | APFloat::opInexact;

  const unsigned NumTests = 169;
  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
  } SpecialCaseTests[NumTests] = {
    { PInf, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PZero, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MZero, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
#if 0
    // See Note 1.
    { PInf, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
    // See Note 1.
    { MInf, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
    // See Note 1.
    { PZero, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
    // See Note 1.
    { MZero, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
    // See Note 1.
    { QNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { QNaN, PNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
#if 0
    // See Note 1.
    { SNaN, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, QNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { PNormalValue, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, PZero, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MZero, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
#if 0
    // See Note 1.
    { PNormalValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
    // See Note 1.
    { MNormalValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
    // See Note 1.
    { PLargestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
    // See Note 1.
    { MLargestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
    // See Note 1.
    { PSmallestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
    // See Note 1.
    { MSmallestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { PSmallestNormalized, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
    // See Note 1.
    { MSmallestNormalized, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { MSmallestNormalized, PNormalValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, MNormalValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, PLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, MLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestValue, "-0x1.fffffcp-127", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MSmallestValue, "-0x1.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { MSmallestNormalized, MSmallestNormalized, "-0x1p-125", APFloat::opOK, APFloat::fcNormal }
  };

  for (size_t i = 0; i < NumTests; ++i) {
    APFloat x(SpecialCaseTests[i].x);
    APFloat y(SpecialCaseTests[i].y);
    APFloat::opStatus status = x.add(y, APFloat::rmNearestTiesToEven);

    APFloat result(APFloat::IEEEsingle, SpecialCaseTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_TRUE((int)status == SpecialCaseTests[i].status);
    EXPECT_TRUE((int)x.getCategory() == SpecialCaseTests[i].category);
  }
}

TEST(APFloatTest, subtract) {
  // Test Special Cases against each other and normal values.

  // TODOS/NOTES:
  // 1. Since we perform only default exception handling all operations with
  // signaling NaNs should have a result that is a quiet NaN. Currently they
  // return sNaN.

  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle, false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle, true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle, false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle, true);
  APFloat QNaN = APFloat::getNaN(APFloat::IEEEsingle, false);
  APFloat SNaN = APFloat::getSNaN(APFloat::IEEEsingle, false);
  APFloat PNormalValue = APFloat(APFloat::IEEEsingle, "0x1p+0");
  APFloat MNormalValue = APFloat(APFloat::IEEEsingle, "-0x1p+0");
  APFloat PLargestValue = APFloat::getLargest(APFloat::IEEEsingle, false);
  APFloat MLargestValue = APFloat::getLargest(APFloat::IEEEsingle, true);
  APFloat PSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle, false);
  APFloat MSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle, true);
  APFloat PSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle, false);
  APFloat MSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle, true);

  const int OverflowStatus = APFloat::opOverflow | APFloat::opInexact;

  const unsigned NumTests = 169;
  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
  } SpecialCaseTests[NumTests] = {
    { PInf, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PZero, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MZero, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { PInf, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
    { MInf, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { MInf, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
    { PZero, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { PZero, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
    { MZero, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { MZero, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { QNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { QNaN, PNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { SNaN, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, QNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { PNormalValue, PInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, MInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, PZero, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, MZero, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { PNormalValue, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { PNormalValue, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
    { MNormalValue, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { MNormalValue, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
    { PLargestValue, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { PLargestValue, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
    { MLargestValue, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { MLargestValue, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
    { PSmallestValue, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { PSmallestValue, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
    { MSmallestValue, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { MSmallestValue, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
    { PSmallestNormalized, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { PSmallestNormalized, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
    { MSmallestNormalized, QNaN, "-nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { MSmallestNormalized, SNaN, "-nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { MSmallestNormalized, PNormalValue, "-0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, MNormalValue, "0x1p+0", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, PLargestValue, "-0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, MLargestValue, "0x1.fffffep+127", APFloat::opInexact, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestValue, "-0x1.000002p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MSmallestValue, "-0x1.fffffcp-127", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestNormalized, "-0x1p-125", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MSmallestNormalized, "0x0p+0", APFloat::opOK, APFloat::fcZero }
  };

  for (size_t i = 0; i < NumTests; ++i) {
    APFloat x(SpecialCaseTests[i].x);
    APFloat y(SpecialCaseTests[i].y);
    APFloat::opStatus status = x.subtract(y, APFloat::rmNearestTiesToEven);

    APFloat result(APFloat::IEEEsingle, SpecialCaseTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_TRUE((int)status == SpecialCaseTests[i].status);
    EXPECT_TRUE((int)x.getCategory() == SpecialCaseTests[i].category);
  }
}

TEST(APFloatTest, multiply) {
  // Test Special Cases against each other and normal values.

  // TODOS/NOTES:
  // 1. Since we perform only default exception handling all operations with
  // signaling NaNs should have a result that is a quiet NaN. Currently they
  // return sNaN.

  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle, false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle, true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle, false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle, true);
  APFloat QNaN = APFloat::getNaN(APFloat::IEEEsingle, false);
  APFloat SNaN = APFloat::getSNaN(APFloat::IEEEsingle, false);
  APFloat PNormalValue = APFloat(APFloat::IEEEsingle, "0x1p+0");
  APFloat MNormalValue = APFloat(APFloat::IEEEsingle, "-0x1p+0");
  APFloat PLargestValue = APFloat::getLargest(APFloat::IEEEsingle, false);
  APFloat MLargestValue = APFloat::getLargest(APFloat::IEEEsingle, true);
  APFloat PSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle, false);
  APFloat MSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle, true);
  APFloat PSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle, false);
  APFloat MSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle, true);

  const int OverflowStatus = APFloat::opOverflow | APFloat::opInexact;
  const int UnderflowStatus = APFloat::opUnderflow | APFloat::opInexact;

  const unsigned NumTests = 169;
  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
  } SpecialCaseTests[NumTests] = {
    { PInf, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { PInf, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MInf, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { PZero, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MZero, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { QNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { QNaN, PNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { SNaN, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, QNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { PNormalValue, PInf, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, MInf, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PNormalValue, PZero, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, MZero, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { PNormalValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MNormalValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { PLargestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MLargestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { PSmallestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MSmallestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { PSmallestNormalized, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MSmallestNormalized, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { MSmallestNormalized, PNormalValue, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MNormalValue, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PLargestValue, "-0x1.fffffep+1", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MLargestValue, "0x1.fffffep+1", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestValue, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestNormalized, MSmallestValue, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestNormalized, PSmallestNormalized, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestNormalized, MSmallestNormalized, "0x0p+0", UnderflowStatus, APFloat::fcZero }
  };

  for (size_t i = 0; i < NumTests; ++i) {
    APFloat x(SpecialCaseTests[i].x);
    APFloat y(SpecialCaseTests[i].y);
    APFloat::opStatus status = x.multiply(y, APFloat::rmNearestTiesToEven);

    APFloat result(APFloat::IEEEsingle, SpecialCaseTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_TRUE((int)status == SpecialCaseTests[i].status);
    EXPECT_TRUE((int)x.getCategory() == SpecialCaseTests[i].category);
  }
}

TEST(APFloatTest, divide) {
  // Test Special Cases against each other and normal values.

  // TODOS/NOTES:
  // 1. Since we perform only default exception handling all operations with
  // signaling NaNs should have a result that is a quiet NaN. Currently they
  // return sNaN.

  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle, false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle, true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle, false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle, true);
  APFloat QNaN = APFloat::getNaN(APFloat::IEEEsingle, false);
  APFloat SNaN = APFloat::getSNaN(APFloat::IEEEsingle, false);
  APFloat PNormalValue = APFloat(APFloat::IEEEsingle, "0x1p+0");
  APFloat MNormalValue = APFloat(APFloat::IEEEsingle, "-0x1p+0");
  APFloat PLargestValue = APFloat::getLargest(APFloat::IEEEsingle, false);
  APFloat MLargestValue = APFloat::getLargest(APFloat::IEEEsingle, true);
  APFloat PSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle, false);
  APFloat MSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle, true);
  APFloat PSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle, false);
  APFloat MSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle, true);

  const int OverflowStatus = APFloat::opOverflow | APFloat::opInexact;
  const int UnderflowStatus = APFloat::opUnderflow | APFloat::opInexact;

  const unsigned NumTests = 169;
  struct {
    APFloat x;
    APFloat y;
    const char *result;
    int status;
    int category;
  } SpecialCaseTests[NumTests] = {
    { PInf, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { PInf, PZero, "inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, MZero, "-inf", APFloat::opOK, APFloat::fcInfinity },
    { PInf, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { PInf, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MInf, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { PZero, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MZero, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { QNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { QNaN, PNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MNormalValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MLargestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestValue, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, PSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
    { QNaN, MSmallestNormalized, "nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { SNaN, PInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MInf, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MZero, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, QNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MNormalValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MLargestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestValue, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, PSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
    { SNaN, MSmallestNormalized, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { PNormalValue, PInf, "0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, MInf, "-0x0p+0", APFloat::opOK, APFloat::fcZero },
    { PNormalValue, PZero, "inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { PNormalValue, MZero, "-inf", APFloat::opDivByZero, APFloat::fcInfinity },
    { PNormalValue, QNaN, "nan", APFloat::opOK, APFloat::fcNaN },
#if 0
// See Note 1.
    { PNormalValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MNormalValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { PLargestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MLargestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { PSmallestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MSmallestValue, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { PSmallestNormalized, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
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
#if 0
// See Note 1.
    { MSmallestNormalized, SNaN, "nan", APFloat::opInvalidOp, APFloat::fcNaN },
#endif
    { MSmallestNormalized, PNormalValue, "-0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MNormalValue, "0x1p-126", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PLargestValue, "-0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestNormalized, MLargestValue, "0x0p+0", UnderflowStatus, APFloat::fcZero },
    { MSmallestNormalized, PSmallestValue, "-0x1p+23", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MSmallestValue, "0x1p+23", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, PSmallestNormalized, "-0x1p+0", APFloat::opOK, APFloat::fcNormal },
    { MSmallestNormalized, MSmallestNormalized, "0x1p+0", APFloat::opOK, APFloat::fcNormal },
  };

  for (size_t i = 0; i < NumTests; ++i) {
    APFloat x(SpecialCaseTests[i].x);
    APFloat y(SpecialCaseTests[i].y);
    APFloat::opStatus status = x.divide(y, APFloat::rmNearestTiesToEven);

    APFloat result(APFloat::IEEEsingle, SpecialCaseTests[i].result);

    EXPECT_TRUE(result.bitwiseIsEqual(x));
    EXPECT_TRUE((int)status == SpecialCaseTests[i].status);
    EXPECT_TRUE((int)x.getCategory() == SpecialCaseTests[i].category);
  }
}

TEST(APFloatTest, operatorOverloads) {
  // This is mostly testing that these operator overloads compile.
  APFloat One = APFloat(APFloat::IEEEsingle, "0x1p+0");
  APFloat Two = APFloat(APFloat::IEEEsingle, "0x2p+0");
  EXPECT_TRUE(Two.bitwiseIsEqual(One + One));
  EXPECT_TRUE(One.bitwiseIsEqual(Two - One));
  EXPECT_TRUE(Two.bitwiseIsEqual(One * Two));
  EXPECT_TRUE(One.bitwiseIsEqual(Two / Two));
}

TEST(APFloatTest, abs) {
  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle, false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle, true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle, false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle, true);
  APFloat PQNaN = APFloat::getNaN(APFloat::IEEEsingle, false);
  APFloat MQNaN = APFloat::getNaN(APFloat::IEEEsingle, true);
  APFloat PSNaN = APFloat::getSNaN(APFloat::IEEEsingle, false);
  APFloat MSNaN = APFloat::getSNaN(APFloat::IEEEsingle, true);
  APFloat PNormalValue = APFloat(APFloat::IEEEsingle, "0x1p+0");
  APFloat MNormalValue = APFloat(APFloat::IEEEsingle, "-0x1p+0");
  APFloat PLargestValue = APFloat::getLargest(APFloat::IEEEsingle, false);
  APFloat MLargestValue = APFloat::getLargest(APFloat::IEEEsingle, true);
  APFloat PSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle, false);
  APFloat MSmallestValue = APFloat::getSmallest(APFloat::IEEEsingle, true);
  APFloat PSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle, false);
  APFloat MSmallestNormalized =
    APFloat::getSmallestNormalized(APFloat::IEEEsingle, true);

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

TEST(APFloatTest, ilogb) {
  EXPECT_EQ(0, ilogb(APFloat(APFloat::IEEEsingle, "0x1p+0")));
  EXPECT_EQ(0, ilogb(APFloat(APFloat::IEEEsingle, "-0x1p+0")));
  EXPECT_EQ(42, ilogb(APFloat(APFloat::IEEEsingle, "0x1p+42")));
  EXPECT_EQ(-42, ilogb(APFloat(APFloat::IEEEsingle, "0x1p-42")));

  EXPECT_EQ(APFloat::IEK_Inf,
            ilogb(APFloat::getInf(APFloat::IEEEsingle, false)));
  EXPECT_EQ(APFloat::IEK_Inf,
            ilogb(APFloat::getInf(APFloat::IEEEsingle, true)));
  EXPECT_EQ(APFloat::IEK_Zero,
            ilogb(APFloat::getZero(APFloat::IEEEsingle, false)));
  EXPECT_EQ(APFloat::IEK_Zero,
            ilogb(APFloat::getZero(APFloat::IEEEsingle, true)));
  EXPECT_EQ(APFloat::IEK_NaN,
            ilogb(APFloat::getNaN(APFloat::IEEEsingle, false)));
  EXPECT_EQ(APFloat::IEK_NaN,
            ilogb(APFloat::getSNaN(APFloat::IEEEsingle, false)));

  EXPECT_EQ(127, ilogb(APFloat::getLargest(APFloat::IEEEsingle, false)));
  EXPECT_EQ(127, ilogb(APFloat::getLargest(APFloat::IEEEsingle, true)));
  EXPECT_EQ(-126, ilogb(APFloat::getSmallest(APFloat::IEEEsingle, false)));
  EXPECT_EQ(-126, ilogb(APFloat::getSmallest(APFloat::IEEEsingle, true)));
  EXPECT_EQ(-126,
            ilogb(APFloat::getSmallestNormalized(APFloat::IEEEsingle, false)));
  EXPECT_EQ(-126,
            ilogb(APFloat::getSmallestNormalized(APFloat::IEEEsingle, true)));
}

TEST(APFloatTest, scalbn) {
  EXPECT_TRUE(
      APFloat(APFloat::IEEEsingle, "0x1p+0")
          .bitwiseIsEqual(scalbn(APFloat(APFloat::IEEEsingle, "0x1p+0"), 0)));
  EXPECT_TRUE(
      APFloat(APFloat::IEEEsingle, "0x1p+42")
          .bitwiseIsEqual(scalbn(APFloat(APFloat::IEEEsingle, "0x1p+0"), 42)));
  EXPECT_TRUE(
      APFloat(APFloat::IEEEsingle, "0x1p-42")
          .bitwiseIsEqual(scalbn(APFloat(APFloat::IEEEsingle, "0x1p+0"), -42)));

  APFloat PInf = APFloat::getInf(APFloat::IEEEsingle, false);
  APFloat MInf = APFloat::getInf(APFloat::IEEEsingle, true);
  APFloat PZero = APFloat::getZero(APFloat::IEEEsingle, false);
  APFloat MZero = APFloat::getZero(APFloat::IEEEsingle, true);
  APFloat QPNaN = APFloat::getNaN(APFloat::IEEEsingle, false);
  APFloat QMNaN = APFloat::getNaN(APFloat::IEEEsingle, true);
  APFloat SNaN = APFloat::getSNaN(APFloat::IEEEsingle, false);

  EXPECT_TRUE(PInf.bitwiseIsEqual(scalbn(PInf, 0)));
  EXPECT_TRUE(MInf.bitwiseIsEqual(scalbn(MInf, 0)));
  EXPECT_TRUE(PZero.bitwiseIsEqual(scalbn(PZero, 0)));
  EXPECT_TRUE(MZero.bitwiseIsEqual(scalbn(MZero, 0)));
  EXPECT_TRUE(QPNaN.bitwiseIsEqual(scalbn(QPNaN, 0)));
  EXPECT_TRUE(QMNaN.bitwiseIsEqual(scalbn(QMNaN, 0)));
  EXPECT_TRUE(SNaN.bitwiseIsEqual(scalbn(SNaN, 0)));

  EXPECT_TRUE(
      PInf.bitwiseIsEqual(scalbn(APFloat(APFloat::IEEEsingle, "0x1p+0"), 128)));
  EXPECT_TRUE(MInf.bitwiseIsEqual(
      scalbn(APFloat(APFloat::IEEEsingle, "-0x1p+0"), 128)));
  EXPECT_TRUE(
      PInf.bitwiseIsEqual(scalbn(APFloat(APFloat::IEEEsingle, "0x1p+127"), 1)));
  EXPECT_TRUE(PZero.bitwiseIsEqual(
      scalbn(APFloat(APFloat::IEEEsingle, "0x1p+0"), -127)));
  EXPECT_TRUE(MZero.bitwiseIsEqual(
      scalbn(APFloat(APFloat::IEEEsingle, "-0x1p+0"), -127)));
  EXPECT_TRUE(PZero.bitwiseIsEqual(
      scalbn(APFloat(APFloat::IEEEsingle, "0x1p-126"), -1)));
  EXPECT_TRUE(PZero.bitwiseIsEqual(
      scalbn(APFloat(APFloat::IEEEsingle, "0x1p-126"), -1)));
}
}
