//===- llvm/unittest/ADT/APFloat.cpp - APFloat unit tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <ostream>
#include <string>
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

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

TEST(APFloatTest, Zero) {
  EXPECT_EQ(0.0f,  APFloat(APFloat::IEEEsingle,  0.0f).convertToFloat());
  EXPECT_EQ(-0.0f, APFloat(APFloat::IEEEsingle, -0.0f).convertToFloat());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble,  0.0).convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, -0.0).convertToDouble());
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

  EXPECT_EQ(2.71828, convertToDoubleFromString("2.71828"));
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
  ASSERT_EQ("0.7853981633974483", convertToString(0.78539816339744830961, 0, 3));
  ASSERT_EQ("4.940656458412465E-324", convertToString(4.9406564584124654e-324, 0, 3));
  ASSERT_EQ("873.1834", convertToString(873.1834, 0, 1));
  ASSERT_EQ("8.731834E+2", convertToString(873.1834, 0, 0));
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

}
