//===- llvm/unittest/Support/MD5Test.cpp - MD5 tests ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the MD5 functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MD5.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
/// \brief Tests an arbitrary set of bytes passed as \p Input.
void TestMD5Sum(ArrayRef<uint8_t> Input, StringRef Final) {
  MD5 Hash;
  Hash.update(Input);
  MD5::MD5Result MD5Res;
  Hash.final(MD5Res);
  SmallString<32> Res;
  MD5::stringifyResult(MD5Res, Res);
  EXPECT_EQ(Res, Final);
}

void TestMD5Sum(StringRef Input, StringRef Final) {
  MD5 Hash;
  Hash.update(Input);
  MD5::MD5Result MD5Res;
  Hash.final(MD5Res);
  SmallString<32> Res;
  MD5::stringifyResult(MD5Res, Res);
  EXPECT_EQ(Res, Final);
}

TEST(MD5Test, MD5) {
  TestMD5Sum(makeArrayRef((const uint8_t *)"", (size_t) 0),
             "d41d8cd98f00b204e9800998ecf8427e");
  TestMD5Sum(makeArrayRef((const uint8_t *)"a", (size_t) 1),
             "0cc175b9c0f1b6a831c399e269772661");
  TestMD5Sum(makeArrayRef((const uint8_t *)"abcdefghijklmnopqrstuvwxyz",
                          (size_t) 26),
             "c3fcd3d76192e4007dfb496cca67e13b");
  TestMD5Sum(makeArrayRef((const uint8_t *)"\0", (size_t) 1),
             "93b885adfe0da089cdf634904fd59f71");
  TestMD5Sum(makeArrayRef((const uint8_t *)"a\0", (size_t) 2),
             "4144e195f46de78a3623da7364d04f11");
  TestMD5Sum(makeArrayRef((const uint8_t *)"abcdefghijklmnopqrstuvwxyz\0",
                          (size_t) 27),
             "81948d1f1554f58cd1a56ebb01f808cb");
  TestMD5Sum("abcdefghijklmnopqrstuvwxyz", "c3fcd3d76192e4007dfb496cca67e13b");
}

TEST(MD5HashTest, MD5) {
  ArrayRef<uint8_t> Input((const uint8_t *)"abcdefghijklmnopqrstuvwxyz", 26);
  std::array<uint8_t, 16> Vec = MD5::hash(Input);
  MD5::MD5Result MD5Res;
  SmallString<32> Res;
  memcpy(MD5Res, Vec.data(), Vec.size());
  MD5::stringifyResult(MD5Res, Res);
  EXPECT_EQ(Res, "c3fcd3d76192e4007dfb496cca67e13b");
}
}
