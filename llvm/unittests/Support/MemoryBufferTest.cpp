//===- llvm/unittest/Support/MemoryBufferTest.cpp - MemoryBuffer tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the MemoryBuffer support class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/OwningPtr.h"

#include "gtest/gtest.h"

using namespace llvm;

class MemoryBufferTest : public testing::Test {
protected:
  MemoryBufferTest()
  : data("this is some data")
  { }

  virtual void SetUp() { }

  typedef OwningPtr<MemoryBuffer> OwningBuffer;

  std::string data;
};

namespace {

TEST_F(MemoryBufferTest, get) {
  // Default name and null-terminator flag
  OwningBuffer MB1(MemoryBuffer::getMemBuffer(data));
  EXPECT_TRUE(0 != MB1.get());

  // RequiresNullTerminator = false
  OwningBuffer MB2(MemoryBuffer::getMemBuffer(data, "one", false));
  EXPECT_TRUE(0 != MB2.get());

  // RequiresNullTerminator = true
  OwningBuffer MB3(MemoryBuffer::getMemBuffer(data, "two", true));
  EXPECT_TRUE(0 != MB3.get());

  // verify all 3 buffers point to the same address
  EXPECT_EQ(MB1->getBufferStart(), MB2->getBufferStart());
  EXPECT_EQ(MB2->getBufferStart(), MB3->getBufferStart());

  // verify the original data is unmodified after deleting the buffers
  MB1.reset();
  MB2.reset();
  MB3.reset();
  EXPECT_EQ("this is some data", data);
}

TEST_F(MemoryBufferTest, copy) {
  // copy with no name
  OwningBuffer MBC1(MemoryBuffer::getMemBufferCopy(data));
  EXPECT_TRUE(0 != MBC1.get());

  // copy with a name
  OwningBuffer MBC2(MemoryBuffer::getMemBufferCopy(data, "copy"));
  EXPECT_TRUE(0 != MBC2.get());

  // verify the two copies do not point to the same place
  EXPECT_NE(MBC1->getBufferStart(), MBC2->getBufferStart());
}

TEST_F(MemoryBufferTest, make_new) {
  // 0-sized buffer
  OwningBuffer Zero(MemoryBuffer::getNewUninitMemBuffer(0));
  EXPECT_TRUE(0 != Zero.get());

  // uninitialized buffer with no name
  OwningBuffer One(MemoryBuffer::getNewUninitMemBuffer(321));
  EXPECT_TRUE(0 != One.get());

  // uninitialized buffer with name
  OwningBuffer Two(MemoryBuffer::getNewUninitMemBuffer(123, "bla"));
  EXPECT_TRUE(0 != Two.get());

  // 0-initialized buffer with no name
  OwningBuffer Three(MemoryBuffer::getNewMemBuffer(321, data));
  EXPECT_TRUE(0 != Three.get());
  for (size_t i = 0; i < 321; ++i)
    EXPECT_EQ(0, Three->getBufferStart()[0]);

  // 0-initialized buffer with name
  OwningBuffer Four(MemoryBuffer::getNewMemBuffer(123, "zeros"));
  EXPECT_TRUE(0 != Four.get());
  for (size_t i = 0; i < 123; ++i)
    EXPECT_EQ(0, Four->getBufferStart()[0]);
}

}
