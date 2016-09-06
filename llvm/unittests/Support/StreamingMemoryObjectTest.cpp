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

class BufferStreamer : public DataStreamer {
  StringRef Buffer;

public:
  BufferStreamer(StringRef Buffer) : Buffer(Buffer) {}
  size_t GetBytes(unsigned char *OutBuffer, size_t Length) override {
    if (Length >= Buffer.size())
      Length = Buffer.size();

    std::copy(Buffer.begin(), Buffer.begin() + Length, OutBuffer);
    Buffer = Buffer.drop_front(Length);
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

TEST(StreamingMemoryObjectTest, getPointer) {
  uint8_t InputBuffer[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
  StreamingMemoryObject O(make_unique<BufferStreamer>(StringRef(
      reinterpret_cast<const char *>(InputBuffer), sizeof(InputBuffer))));

  EXPECT_TRUE(std::equal(InputBuffer + 1, InputBuffer + 2, O.getPointer(1, 2)));
  EXPECT_TRUE(std::equal(InputBuffer + 3, InputBuffer + 7, O.getPointer(3, 4)));
  EXPECT_TRUE(std::equal(InputBuffer + 4, InputBuffer + 8, O.getPointer(4, 5)));
  EXPECT_TRUE(std::equal(InputBuffer, InputBuffer + 8, O.getPointer(0, 20)));
}

} // end namespace
