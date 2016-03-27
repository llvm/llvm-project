//===- unittests/Support/StreamingMemoryObjectTest.cpp --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/StreamingMemoryObject.h"
#include "gtest/gtest.h"
#include <string.h>

using namespace llvm;

namespace {

class NullDataStreamer : public DataStreamer {
  size_t GetBytes(unsigned char *Buffer, size_t Length) override {
    memset(Buffer, 0, Length);
    return Length;
  }
};

TEST(StreamingMemoryObjectTest, isValidAddress) {
  auto DS = make_unique<NullDataStreamer>();
  StreamingMemoryObject O(std::move(DS));
  EXPECT_TRUE(O.isValidAddress(32 * 1024));
}

TEST(StreamingMemoryObjectTest, setKnownObjectSize) {
  auto DS = make_unique<NullDataStreamer>();
  StreamingMemoryObject O(std::move(DS));
  uint8_t Buf[32];
  EXPECT_EQ(16u, O.readBytes(Buf, 16, 0));
  O.setKnownObjectSize(24);
  EXPECT_EQ(8u, O.readBytes(Buf, 16, 16));
}

} // end namespace
