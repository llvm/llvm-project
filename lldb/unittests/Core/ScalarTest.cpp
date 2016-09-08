//===-- ScalarTest.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER) && (_HAS_EXCEPTIONS == 0)
// Workaround for MSVC standard library bug, which fails to include <thread>
// when
// exceptions are disabled.
#include <eh.h>
#endif

#include "gtest/gtest.h"

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Host/Endian.h"

using namespace lldb_private;

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
  float c = 1234567.89e42;
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
  Error e_error =
      e_scalar.SetValueFromData(e_data, lldb::eEncodingUint, sizeof(e));
  DataExtractor f_data(f, sizeof(f), endian::InlHostByteOrder(),
                       sizeof(void *));
  Error f_error =
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
