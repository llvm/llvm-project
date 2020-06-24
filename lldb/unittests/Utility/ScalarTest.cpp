//===-- ScalarTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/Testing/Support/Error.h"

using namespace lldb_private;
using llvm::APInt;
using llvm::Failed;
using llvm::Succeeded;

template <typename T>
bool checkInequality(T c1, T c2) {
  return (Scalar(c1) != Scalar(c2));
}

template <typename T>
bool checkEquality(T c1, T c2) {
  return (Scalar(c1) == Scalar(c2));
}

TEST(ScalarTest, Equality) {
  ASSERT_TRUE(checkInequality<int>(23, 24));
  ASSERT_TRUE(checkEquality<int>(96, 96));
  ASSERT_TRUE(checkInequality<float>(4.0f, 4.5f));
  ASSERT_TRUE(checkEquality<float>(4.0f, 4.0f));

  auto apint1 = APInt(64, 234);
  auto apint2 = APInt(64, 246);
  ASSERT_TRUE(checkInequality<APInt>(apint1, apint2));
  ASSERT_TRUE(checkEquality<APInt>(apint1, apint1));

  Scalar void1;
  Scalar void2;
  float f1 = 2.0;
  ASSERT_TRUE(void1 == void2);
  ASSERT_FALSE(void1 == Scalar(f1));
}

TEST(ScalarTest, Comparison) {
  auto s1 = Scalar(23);
  auto s2 = Scalar(46);
  ASSERT_TRUE(s1 < s2);
  ASSERT_TRUE(s1 <= s2);
  ASSERT_TRUE(s2 > s1);
  ASSERT_TRUE(s2 >= s1);
}

TEST(ScalarTest, ComparisonFloat) {
  auto s1 = Scalar(23.0f);
  auto s2 = Scalar(46.0f);
  ASSERT_TRUE(s1 < s2);
  ASSERT_TRUE(s1 <= s2);
  ASSERT_TRUE(s2 > s1);
  ASSERT_TRUE(s2 >= s1);
}

TEST(ScalarTest, RightShiftOperator) {
  int a = 0x00001000;
  int b = 0xFFFFFFFF;
  int c = 4;
  Scalar a_scalar(a);
  Scalar b_scalar(b);
  Scalar c_scalar(c);
  ASSERT_EQ(a >> c, a_scalar >> c_scalar);
  ASSERT_EQ(b >> c, b_scalar >> c_scalar);
}

TEST(ScalarTest, GetBytes) {
  int a = 0x01020304;
  long long b = 0x0102030405060708LL;
  float c = 1234567.89e32f;
  double d = 1234567.89e42;
  char e[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  char f[32] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  Scalar a_scalar(a);
  Scalar b_scalar(b);
  Scalar c_scalar(c);
  Scalar d_scalar(d);
  Scalar e_scalar;
  Scalar f_scalar;
  DataExtractor e_data(e, sizeof(e), endian::InlHostByteOrder(),
                       sizeof(void *));
  Status e_error =
      e_scalar.SetValueFromData(e_data, lldb::eEncodingUint, sizeof(e));
  DataExtractor f_data(f, sizeof(f), endian::InlHostByteOrder(),
                       sizeof(void *));
  Status f_error =
      f_scalar.SetValueFromData(f_data, lldb::eEncodingUint, sizeof(f));
  ASSERT_EQ(0, memcmp(&a, a_scalar.GetBytes(), sizeof(a)));
  ASSERT_EQ(0, memcmp(&b, b_scalar.GetBytes(), sizeof(b)));
  ASSERT_EQ(0, memcmp(&c, c_scalar.GetBytes(), sizeof(c)));
  ASSERT_EQ(0, memcmp(&d, d_scalar.GetBytes(), sizeof(d)));
  ASSERT_EQ(0, e_error.Fail());
  ASSERT_EQ(0, memcmp(e, e_scalar.GetBytes(), sizeof(e)));
  ASSERT_EQ(0, f_error.Fail());
  ASSERT_EQ(0, memcmp(f, f_scalar.GetBytes(), sizeof(f)));
}

TEST(ScalarTest, CastOperations) {
  long long a = 0xf1f2f3f4f5f6f7f8LL;
  Scalar a_scalar(a);
  ASSERT_EQ((signed char)a, a_scalar.SChar());
  ASSERT_EQ((unsigned char)a, a_scalar.UChar());
  ASSERT_EQ((signed short)a, a_scalar.SShort());
  ASSERT_EQ((unsigned short)a, a_scalar.UShort());
  ASSERT_EQ((signed int)a, a_scalar.SInt());
  ASSERT_EQ((unsigned int)a, a_scalar.UInt());
  ASSERT_EQ((signed long)a, a_scalar.SLong());
  ASSERT_EQ((unsigned long)a, a_scalar.ULong());
  ASSERT_EQ((signed long long)a, a_scalar.SLongLong());
  ASSERT_EQ((unsigned long long)a, a_scalar.ULongLong());

  int a2 = 23;
  Scalar a2_scalar(a2);
  ASSERT_EQ((float)a2, a2_scalar.Float());
  ASSERT_EQ((double)a2, a2_scalar.Double());
  ASSERT_EQ((long double)a2, a2_scalar.LongDouble());
}

TEST(ScalarTest, ExtractBitfield) {
  uint32_t len = sizeof(long long) * 8;

  long long a1 = 0xf1f2f3f4f5f6f7f8LL;
  long long b1 = 0xff1f2f3f4f5f6f7fLL;
  Scalar s_scalar(a1);
  ASSERT_TRUE(s_scalar.ExtractBitfield(0, 0));
  ASSERT_EQ(0, memcmp(&a1, s_scalar.GetBytes(), sizeof(a1)));
  ASSERT_TRUE(s_scalar.ExtractBitfield(len, 0));
  ASSERT_EQ(0, memcmp(&a1, s_scalar.GetBytes(), sizeof(a1)));
  ASSERT_TRUE(s_scalar.ExtractBitfield(len - 4, 4));
  ASSERT_EQ(0, memcmp(&b1, s_scalar.GetBytes(), sizeof(b1)));

  unsigned long long a2 = 0xf1f2f3f4f5f6f7f8ULL;
  unsigned long long b2 = 0x0f1f2f3f4f5f6f7fULL;
  Scalar u_scalar(a2);
  ASSERT_TRUE(u_scalar.ExtractBitfield(0, 0));
  ASSERT_EQ(0, memcmp(&a2, u_scalar.GetBytes(), sizeof(a2)));
  ASSERT_TRUE(u_scalar.ExtractBitfield(len, 0));
  ASSERT_EQ(0, memcmp(&a2, u_scalar.GetBytes(), sizeof(a2)));
  ASSERT_TRUE(u_scalar.ExtractBitfield(len - 4, 4));
  ASSERT_EQ(0, memcmp(&b2, u_scalar.GetBytes(), sizeof(b2)));
}

template <typename T> static std::string ScalarGetValue(T value) {
  StreamString stream;
  Scalar(value).GetValue(&stream, false);
  return std::string(stream.GetString());
}

TEST(ScalarTest, GetValue) {
  EXPECT_EQ("12345", ScalarGetValue<signed short>(12345));
  EXPECT_EQ("-12345", ScalarGetValue<signed short>(-12345));
  EXPECT_EQ("12345", ScalarGetValue<unsigned short>(12345));
  EXPECT_EQ(std::to_string(std::numeric_limits<unsigned short>::max()),
            ScalarGetValue(std::numeric_limits<unsigned short>::max()));

  EXPECT_EQ("12345", ScalarGetValue<signed int>(12345));
  EXPECT_EQ("-12345", ScalarGetValue<signed int>(-12345));
  EXPECT_EQ("12345", ScalarGetValue<unsigned int>(12345));
  EXPECT_EQ(std::to_string(std::numeric_limits<unsigned int>::max()),
            ScalarGetValue(std::numeric_limits<unsigned int>::max()));

  EXPECT_EQ("12345678", ScalarGetValue<signed long>(12345678L));
  EXPECT_EQ("-12345678", ScalarGetValue<signed long>(-12345678L));
  EXPECT_EQ("12345678", ScalarGetValue<unsigned long>(12345678UL));
  EXPECT_EQ(std::to_string(std::numeric_limits<unsigned long>::max()),
            ScalarGetValue(std::numeric_limits<unsigned long>::max()));

  EXPECT_EQ("1234567890123", ScalarGetValue<signed long long>(1234567890123LL));
  EXPECT_EQ("-1234567890123",
            ScalarGetValue<signed long long>(-1234567890123LL));
  EXPECT_EQ("1234567890123",
            ScalarGetValue<unsigned long long>(1234567890123ULL));
  EXPECT_EQ(std::to_string(std::numeric_limits<unsigned long long>::max()),
            ScalarGetValue(std::numeric_limits<unsigned long long>::max()));
}

TEST(ScalarTest, LongLongAssigmentOperator) {
  Scalar ull;
  ull = std::numeric_limits<unsigned long long>::max();
  EXPECT_EQ(std::numeric_limits<unsigned long long>::max(), ull.ULongLong());

  Scalar sll;
  sll = std::numeric_limits<signed long long>::max();
  EXPECT_EQ(std::numeric_limits<signed long long>::max(), sll.SLongLong());
}

TEST(ScalarTest, Division) {
  Scalar lhs(5.0);
  Scalar rhs(2.0);
  Scalar r = lhs / rhs;
  EXPECT_TRUE(r.IsValid());
  EXPECT_EQ(r, Scalar(2.5));
}

TEST(ScalarTest, Promotion) {
  static Scalar::Type int_types[] = {
      Scalar::e_sint,    Scalar::e_uint,      Scalar::e_slong,
      Scalar::e_ulong,   Scalar::e_slonglong, Scalar::e_ulonglong,
      Scalar::e_sint128, Scalar::e_uint128,   Scalar::e_sint256,
      Scalar::e_uint256,
      Scalar::e_void // sentinel
  };

  static Scalar::Type float_types[] = {
      Scalar::e_float, Scalar::e_double, Scalar::e_long_double,
      Scalar::e_void // sentinel
  };

  for (int i = 0; int_types[i] != Scalar::e_void; ++i) {
    for (int j = 0; float_types[j] != Scalar::e_void; ++j) {
      Scalar lhs(2);
      EXPECT_TRUE(lhs.Promote(int_types[i])) << "int promotion #" << i;
      Scalar rhs(0.5f);
      EXPECT_TRUE(rhs.Promote(float_types[j])) << "float promotion #" << j;
      Scalar x(2.5f);
      EXPECT_TRUE(x.Promote(float_types[j]));
      EXPECT_EQ(lhs + rhs, x);
    }
  }

  for (int i = 0; float_types[i] != Scalar::e_void; ++i) {
    for (int j = 0; float_types[j] != Scalar::e_void; ++j) {
      Scalar lhs(2);
      EXPECT_TRUE(lhs.Promote(float_types[i])) << "float promotion #" << i;
      Scalar rhs(0.5f);
      EXPECT_TRUE(rhs.Promote(float_types[j])) << "float promotion #" << j;
      Scalar x(2.5f);
      EXPECT_TRUE(x.Promote(float_types[j]));
      EXPECT_EQ(lhs + rhs, x);
    }
  }
}

TEST(ScalarTest, SetValueFromCString) {
  Scalar a;

  EXPECT_THAT_ERROR(
      a.SetValueFromCString("1234567890123", lldb::eEncodingUint, 8).ToError(),
      Succeeded());
  EXPECT_EQ(1234567890123ull, a);

  EXPECT_THAT_ERROR(
      a.SetValueFromCString("-1234567890123", lldb::eEncodingSint, 8).ToError(),
      Succeeded());
  EXPECT_EQ(-1234567890123ll, a);

  EXPECT_THAT_ERROR(
      a.SetValueFromCString("asdf", lldb::eEncodingSint, 8).ToError(),
      Failed());
  EXPECT_THAT_ERROR(
      a.SetValueFromCString("asdf", lldb::eEncodingUint, 8).ToError(),
      Failed());
  EXPECT_THAT_ERROR(
      a.SetValueFromCString("1234567890123", lldb::eEncodingUint, 4).ToError(),
      Failed());
  EXPECT_THAT_ERROR(a.SetValueFromCString("123456789012345678901234567890",
                                          lldb::eEncodingUint, 8)
                        .ToError(),
                    Failed());
  EXPECT_THAT_ERROR(
      a.SetValueFromCString("-123", lldb::eEncodingUint, 8).ToError(),
      Failed());
}

TEST(ScalarTest, APIntConstructor) {
  auto width_array = {8, 16, 32};
  for (auto &w : width_array) {
    Scalar A(APInt(w, 24));
    EXPECT_EQ(A.GetType(), Scalar::e_sint);
  }

  Scalar B(APInt(64, 42));
  EXPECT_EQ(B.GetType(), Scalar::GetBestTypeForBitSize(64, true));
  Scalar C(APInt(128, 96));
  EXPECT_EQ(C.GetType(), Scalar::e_sint128);
  Scalar D(APInt(256, 156));
  EXPECT_EQ(D.GetType(), Scalar::e_sint256);
  Scalar E(APInt(512, 456));
  EXPECT_EQ(E.GetType(), Scalar::e_sint512);
}

TEST(ScalarTest, Scalar_512) {
  Scalar Z(APInt(512, 0));
  ASSERT_TRUE(Z.IsZero());
  Z.MakeUnsigned();
  ASSERT_TRUE(Z.IsZero());

  Scalar S(APInt(512, 2000));
  ASSERT_STREQ(S.GetTypeAsCString(), "int512_t");
  ASSERT_STREQ(S.GetValueTypeAsCString(Scalar::e_sint512), "int512_t");

  ASSERT_TRUE(S.MakeUnsigned());
  EXPECT_EQ(S.GetType(), Scalar::e_uint512);
  ASSERT_STREQ(S.GetTypeAsCString(), "unsigned int512_t");
  ASSERT_STREQ(S.GetValueTypeAsCString(Scalar::e_uint512), "uint512_t");
  EXPECT_EQ(S.GetByteSize(), 64U);

  ASSERT_TRUE(S.MakeSigned());
  EXPECT_EQ(S.GetType(), Scalar::e_sint512);
  EXPECT_EQ(S.GetByteSize(), 64U);
}

TEST(ScalarTest, TruncOrExtendTo) {
  Scalar S(0xffff);
  S.TruncOrExtendTo(12, true);
  EXPECT_EQ(S.ULong(), 0xfffu);
  S.TruncOrExtendTo(20, true);
  EXPECT_EQ(S.ULong(), 0xfffffu);
  S.TruncOrExtendTo(24, false);
  EXPECT_EQ(S.ULong(), 0x0fffffu);
  S.TruncOrExtendTo(16, false);
  EXPECT_EQ(S.ULong(), 0xffffu);
}
