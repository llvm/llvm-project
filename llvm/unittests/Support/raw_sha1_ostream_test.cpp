//===- llvm/unittest/Support/raw_ostream_test.cpp - raw_ostream tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_sha1_ostream.h"
#include "gtest/gtest.h"

#include <string>

using namespace llvm;

static std::string toHex(ArrayRef<uint8_t> Input) {
  static const char *const LUT = "0123456789ABCDEF";
  size_t Length = Input.size();

  std::string Output;
  Output.reserve(2 * Length);
  for (size_t i = 0; i < Length; ++i) {
    const unsigned char c = Input[i];
    Output.push_back(LUT[c >> 4]);
    Output.push_back(LUT[c & 15]);
  }
  return Output;
}

TEST(raw_sha1_ostreamTest, Basic) {
  llvm::raw_sha1_ostream Sha1Stream;
  Sha1Stream << "Hello World!";
  auto Hash = toHex(Sha1Stream.sha1());

  ASSERT_EQ("2EF7BDE608CE5404E97D5F042F95F89F1C232871", Hash);
}

TEST(sha1_hash_test, Basic) {
  ArrayRef<uint8_t> Input((const uint8_t *)"Hello World!", 12);
  std::array<uint8_t, 20> Vec = SHA1::hash(Input);
  std::string Hash = toHex(Vec);
  ASSERT_EQ("2EF7BDE608CE5404E97D5F042F95F89F1C232871", Hash);
}

TEST(sha1_hash_test, Update) {
  SHA1 sha1;
  std::string Input = "123456789012345678901234567890";
  ASSERT_EQ(Input.size(), 30UL);
  // 3 short updates.
  sha1.update(Input);
  sha1.update(Input);
  sha1.update(Input);
  // Long update that gets into the optimized loop with prefix/suffix.
  sha1.update(Input + Input + Input + Input);
  // 18 bytes buffered now.

  std::string Hash = toHex(sha1.final());
  ASSERT_EQ("3E4A614101AD84985AB0FE54DC12A6D71551E5AE", Hash);
}

// Check that getting the intermediate hash in the middle of the stream does
// not invalidate the final result.
TEST(raw_sha1_ostreamTest, Intermediate) {
  llvm::raw_sha1_ostream Sha1Stream;
  Sha1Stream << "Hello";
  auto Hash = toHex(Sha1Stream.sha1());

  ASSERT_EQ("F7FF9E8B7BB2E09B70935A5D785E0CC5D9D0ABF0", Hash);
  Sha1Stream << " World!";
  Hash = toHex(Sha1Stream.sha1());

  // Compute the non-split hash separately as a reference.
  llvm::raw_sha1_ostream NonSplitSha1Stream;
  NonSplitSha1Stream << "Hello World!";
  auto NonSplitHash = toHex(NonSplitSha1Stream.sha1());

  ASSERT_EQ(NonSplitHash, Hash);
}

TEST(raw_sha1_ostreamTest, Reset) {
  llvm::raw_sha1_ostream Sha1Stream;
  Sha1Stream << "Hello";
  auto Hash = toHex(Sha1Stream.sha1());

  ASSERT_EQ("F7FF9E8B7BB2E09B70935A5D785E0CC5D9D0ABF0", Hash);

  Sha1Stream.resetHash();
  Sha1Stream << " World!";
  Hash = toHex(Sha1Stream.sha1());

  ASSERT_EQ("7447F2A5A42185C8CF91E632789C431830B59067", Hash);
}
