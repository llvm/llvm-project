//===- llvm/unittests/IR/MemoryModelRelaxationAnnotationsTest.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(MMRATest, Equality_Ordering) {
  MMRAMetadata A, B;

  std::array<MMRAMetadata::TagT, 5> Tags{{{"opencl-fence-mem", "local"},
                                          {"opencl-fence-mem", "global"},
                                          {"foo", "0"},
                                          {"foo", "2"},
                                          {"foo", "4"}}};

  // Test that ordering does not matter.
  for (unsigned K : {0, 2, 3, 1, 4})
    A.addTag(Tags[K]);
  for (unsigned K : {2, 3, 0, 4, 1})
    B.addTag(Tags[K]);

  EXPECT_EQ(A, B);
}

} // namespace
