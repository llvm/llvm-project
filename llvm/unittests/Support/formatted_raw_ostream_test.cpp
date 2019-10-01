//===- llvm/unittest/Support/formatted_raw_ostream_test.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(formatted_raw_ostreamTest, Test_Tell) {
  // Check offset when underlying stream has buffer contents.
  SmallString<128> A;
  raw_svector_ostream B(A);
  formatted_raw_ostream C(B);
  char tmp[100] = "";

  for (unsigned i = 0; i != 3; ++i) {
    C.write(tmp, 100);

    EXPECT_EQ(100*(i+1), (unsigned) C.tell());
  }
}

}
