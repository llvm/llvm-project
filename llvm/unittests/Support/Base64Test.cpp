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
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
/// Tests an arbitrary set of bytes passed as \p Input.
void TestBase64(StringRef Input, StringRef Final) {
  auto Res = encodeBase64(Input);
  EXPECT_EQ(Res, Final);
}

void TestBase64Decode(StringRef Input, StringRef Expected,
                      StringRef ExpectedErrorMessage = {}) {
  std::vector<char> DecodedBytes;
  if (ExpectedErrorMessage.empty()) {
    ASSERT_THAT_ERROR(decodeBase64(Input, DecodedBytes), Succeeded());
    EXPECT_EQ(llvm::ArrayRef<char>(DecodedBytes),
              llvm::ArrayRef<char>(Expected.data(), Expected.size()));
  } else {
    ASSERT_THAT_ERROR(decodeBase64(Input, DecodedBytes),
                      FailedWithMessage(ExpectedErrorMessage));
  }
}

char NonPrintableVector[] = {0x00, 0x00, 0x00,       0x46,
                             0x00, 0x08, (char)0xff, (char)0xee};

char LargeVector[] = {0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b,
                      0x20, 0x62, 0x72, 0x6f, 0x77, 0x6e, 0x20, 0x66, 0x6f,
                      0x78, 0x20, 0x6a, 0x75, 0x6d, 0x70, 0x73, 0x20, 0x6f,
                      0x76, 0x65, 0x72, 0x20, 0x31, 0x33, 0x20, 0x6c, 0x61,
                      0x7a, 0x79, 0x20, 0x64, 0x6f, 0x67, 0x73, 0x2e};

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
  TestBase64({NonPrintableVector, sizeof(NonPrintableVector)}, "AAAARgAI/+4=");

  // Large test case
  TestBase64({LargeVector, sizeof(LargeVector)},
             "VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIDEzIGxhenkgZG9ncy4=");
}

TEST(Base64Test, DecodeBase64) {
  std::vector<llvm::StringRef> Outputs = {"",     "f",     "fo",    "foo",
                                          "foob", "fooba", "foobar"};
  Outputs.push_back(
      llvm::StringRef(NonPrintableVector, sizeof(NonPrintableVector)));

  Outputs.push_back(llvm::StringRef(LargeVector, sizeof(LargeVector)));
  // Make sure we can encode and decode any byte.
  std::vector<char> AllChars;
  for (int Ch = INT8_MIN; Ch <= INT8_MAX; ++Ch)
    AllChars.push_back(Ch);
  Outputs.push_back(llvm::StringRef(AllChars.data(), AllChars.size()));

  for (const auto &Output : Outputs) {
    // We trust that encoding is working after running the Base64Test::Base64()
    // test function above, so we can use it to encode the string and verify we
    // can decode it correctly.
    auto Input = encodeBase64(Output);
    TestBase64Decode(Input, Output);
  }
  struct ErrorInfo {
    llvm::StringRef Input;
    llvm::StringRef ErrorMessage;
  };
  std::vector<ErrorInfo> ErrorInfos = {
      {"f", "Base64 encoded strings must be a multiple of 4 bytes in length"},
      {"=abc", "Invalid Base64 character 0x3d at index 0"},
      {"a=bc", "Invalid Base64 character 0x3d at index 1"},
      {"ab=c", "Invalid Base64 character 0x3d at index 2"},
      {"fun!", "Invalid Base64 character 0x21 at index 3"},
  };

  for (const auto &EI : ErrorInfos)
    TestBase64Decode(EI.Input, "", EI.ErrorMessage);
}
