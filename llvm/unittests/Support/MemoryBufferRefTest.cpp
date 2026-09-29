//===- MemoryBufferRefTest.cpp - MemoryBufferRef tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the MemoryBufferRef support class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(MemoryBufferRefTest, construct) {
  std::unique_ptr<MemoryBuffer> MB(MemoryBuffer::getMemBuffer("data", "id"));
  MemoryBufferRef MBR(*MB);

  EXPECT_EQ(MB->getBufferStart(), MBR.getBufferStart());
  EXPECT_EQ(MB->getBufferIdentifier(), MBR.getBufferIdentifier());
}

TEST(MemoryBufferRefTest, compareEquals) {
  std::string Data = "data";
  std::unique_ptr<MemoryBuffer> MB(MemoryBuffer::getMemBuffer(Data, "id"));
  MemoryBufferRef Ref(*MB);
  MemoryBufferRef Empty;
  MemoryBufferRef NoIdentifier(MB->getBuffer(), "");
  MemoryBufferRef NoData("", MB->getBufferIdentifier());
  MemoryBufferRef Same(MB->getBuffer(), MB->getBufferIdentifier());

  EXPECT_NE(Empty, Ref);
  EXPECT_NE(NoIdentifier, Ref);
  EXPECT_NE(NoData, Ref);
  EXPECT_EQ(Same, Ref);

  // Confirm NE when content matches but pointer identity does not.
  std::unique_ptr<MemoryBuffer> Copy(
      MemoryBuffer::getMemBufferCopy(Data, "id"));
  MemoryBufferRef CopyRef(*Copy);
  EXPECT_EQ(Ref.getBuffer(), CopyRef.getBuffer());
  EXPECT_NE(Ref, CopyRef);
}

} // end namespace
