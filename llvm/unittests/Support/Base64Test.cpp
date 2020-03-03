//===- llvm/unittest/Support/Base64Test.cpp - Base64 tests
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the Base64 functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Base64.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
/// Tests an arbitrary set of bytes passed as \p Input.
void TestBase64(StringRef Input, StringRef Final) {
  auto Res = encodeBase64(Input);
  EXPECT_EQ(Res, Final);
}

} // namespace

TEST(Base64Test, Base64) {
  // from: https://tools.ietf.org/html/rfc4648#section-10
  TestBase64("", "");
  TestBase64("f", "Zg==");
  TestBase64("fo", "Zm8=");
  TestBase64("foo", "Zm9v");
  TestBase64("foob", "Zm9vYg==");
  TestBase64("fooba", "Zm9vYmE=");
  TestBase64("foobar", "Zm9vYmFy");

  // With non-printable values.
  char NonPrintableVector[] = {0x00, 0x00, 0x00,       0x46,
                               0x00, 0x08, (char)0xff, (char)0xee};
  TestBase64({NonPrintableVector, sizeof(NonPrintableVector)}, "AAAARgAI/+4=");

  // Large test case
  char LargeVector[] = {0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b,
                        0x20, 0x62, 0x72, 0x6f, 0x77, 0x6e, 0x20, 0x66, 0x6f,
                        0x78, 0x20, 0x6a, 0x75, 0x6d, 0x70, 0x73, 0x20, 0x6f,
                        0x76, 0x65, 0x72, 0x20, 0x31, 0x33, 0x20, 0x6c, 0x61,
                        0x7a, 0x79, 0x20, 0x64, 0x6f, 0x67, 0x73, 0x2e};
  TestBase64({LargeVector, sizeof(LargeVector)},
             "VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIDEzIGxhenkgZG9ncy4=");
}
