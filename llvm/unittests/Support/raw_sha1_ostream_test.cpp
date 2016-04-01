//===- llvm/unittest/Support/raw_ostream_test.cpp - raw_ostream tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_sha1_ostream.h"

#include <string>

using namespace llvm;

static std::string toHex(StringRef Input) {
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
