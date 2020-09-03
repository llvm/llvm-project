//===-- StreamTeeTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StreamTee.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(StreamTeeTest, DefaultConstructor) {
  // Test the default constructor.
  StreamTee tee;
  ASSERT_EQ(0U, tee.GetNumStreams());
}

TEST(StreamTeeTest, Constructor1Stream) {
  // Test the constructor for a single stream.
  lldb::StreamSP s1(std::make_shared<StreamString>());
  StreamTee tee(s1);

  ASSERT_EQ(1U, tee.GetNumStreams());
  EXPECT_EQ(s1, tee.GetStreamAtIndex(0U));
}

TEST(StreamTeeTest, Constructor2Streams) {
  // Test the constructor for two streams.
  lldb::StreamSP s1(std::make_shared<StreamString>());
  lldb::StreamSP s2(std::make_shared<StreamString>());
  StreamTee tee(s1, s2);

  ASSERT_EQ(2U, tee.GetNumStreams());
  EXPECT_EQ(s1, tee.GetStreamAtIndex(0U));
  EXPECT_EQ(s2, tee.GetStreamAtIndex(1U));
}

TEST(StreamTeeTest, CopyConstructor) {
  // Test the copy constructor.
  lldb::StreamSP s1(std::make_shared<StreamString>());
  lldb::StreamSP s2(std::make_shared<StreamString>());
  StreamTee tee1(s1, s2);
  StreamTee tee2(tee1);

  ASSERT_EQ(2U, tee2.GetNumStreams());
  EXPECT_EQ(s1, tee2.GetStreamAtIndex(0U));
  EXPECT_EQ(s2, tee2.GetStreamAtIndex(1U));
}

TEST(StreamTeeTest, Assignment) {
  // Test the assignment of StreamTee.
  lldb::StreamSP s1(std::make_shared<StreamString>());
  lldb::StreamSP s2(std::make_shared<StreamString>());
  StreamTee tee1(s1, s2);
  StreamTee tee2 = tee1;

  ASSERT_EQ(2U, tee2.GetNumStreams());
  EXPECT_EQ(s1, tee2.GetStreamAtIndex(0U));
  EXPECT_EQ(s2, tee2.GetStreamAtIndex(1U));
}

TEST(StreamTeeTest, Write) {
  // Test that write is sent out to all children.
  auto ss1 = new StreamString();
  auto ss2 = new StreamString();
  lldb::StreamSP s1(ss1);
  lldb::StreamSP s2(ss2);
  StreamTee tee(s1, s2);

  tee << "foo";
  tee.Flush();

  ASSERT_EQ(2U, tee.GetNumStreams());
  EXPECT_EQ("foo", ss1->GetString().str());
  EXPECT_EQ("foo", ss2->GetString().str());

  tee << "bar";
  tee.Flush();
  EXPECT_EQ("foobar", ss1->GetString().str());
  EXPECT_EQ("foobar", ss2->GetString().str());
}

namespace {
  struct FlushTestStream : public Stream {
    unsigned m_flush_count = false;
    void Flush() override {
      ++m_flush_count;
    }
    size_t WriteImpl(const void *src, size_t src_len) override {
      return src_len;
    }
  };
}

TEST(StreamTeeTest, Flush) {
  // Check that Flush is distributed to all streams.
  auto fs1 = new FlushTestStream();
  auto fs2 = new FlushTestStream();
  lldb::StreamSP s1(fs1);
  lldb::StreamSP s2(fs2);
  StreamTee tee(s1, s2);

  tee << "foo";
  tee.Flush();

  ASSERT_EQ(2U, tee.GetNumStreams());
  EXPECT_EQ(1U, fs1->m_flush_count);
  EXPECT_EQ(1U, fs2->m_flush_count);

  tee << "bar";
  tee.Flush();
  EXPECT_EQ(2U, fs1->m_flush_count);
  EXPECT_EQ(2U, fs2->m_flush_count);
}

TEST(StreamTeeTest, AppendStream) {
  // Append new streams to our StreamTee.
  auto ss1 = new StreamString();
  auto ss2 = new StreamString();
  lldb::StreamSP s1(ss1);
  lldb::StreamSP s2(ss2);

  StreamTee tee;

  ASSERT_EQ(0U, tee.GetNumStreams());

  tee.AppendStream(s1);
  ASSERT_EQ(1U, tee.GetNumStreams());
  EXPECT_EQ(s1, tee.GetStreamAtIndex(0U));

  tee.AppendStream(s2);
  ASSERT_EQ(2U, tee.GetNumStreams());
  EXPECT_EQ(s1, tee.GetStreamAtIndex(0U));
  EXPECT_EQ(s2, tee.GetStreamAtIndex(1U));
}

TEST(StreamTeeTest, GetStreamAtIndexOutOfBounds) {
  // The index we check for is not in the bounds of the StreamTee.
  lldb::StreamSP s1(std::make_shared<StreamString>());
  StreamTee tee(s1);

  ASSERT_EQ(1U, tee.GetNumStreams());
  EXPECT_EQ(lldb::StreamSP(), tee.GetStreamAtIndex(1));
}

TEST(StreamTeeTest, GetStreamAtIndexOutOfBoundsEmpty) {
  // Same as above, but with an empty StreamTee.
  StreamTee tee;
  ASSERT_EQ(0U, tee.GetNumStreams());
  EXPECT_EQ(lldb::StreamSP(), tee.GetStreamAtIndex(0U));
  EXPECT_EQ(lldb::StreamSP(), tee.GetStreamAtIndex(1U));
}

TEST(StreamTeeTest, SetStreamAtIndexOverwrite) {
  // We overwrite an existing stream at a given index.
  lldb::StreamSP s1(std::make_shared<StreamString>());
  StreamTee tee(s1);

  ASSERT_EQ(1U, tee.GetNumStreams());
  EXPECT_EQ(s1, tee.GetStreamAtIndex(0U));
  EXPECT_EQ(lldb::StreamSP(), tee.GetStreamAtIndex(1U));

  lldb::StreamSP s2(std::make_shared<StreamString>());
  tee.SetStreamAtIndex(0U, s2);
  EXPECT_EQ(1U, tee.GetNumStreams());
  EXPECT_EQ(s2, tee.GetStreamAtIndex(0U));
  EXPECT_EQ(lldb::StreamSP(), tee.GetStreamAtIndex(1));
}

TEST(StreamTeeTest, SetStreamAtIndexOutOfBounds) {
  // We place a new stream out of the bounds of the current StreamTee.
  lldb::StreamSP s1(std::make_shared<StreamString>());
  StreamTee tee(s1);

  ASSERT_EQ(1U, tee.GetNumStreams());
  EXPECT_EQ(s1, tee.GetStreamAtIndex(0U));
  EXPECT_EQ(lldb::StreamSP(), tee.GetStreamAtIndex(1U));

  // Place a new stream out of bounds of the current array. The StreamTee should
  // resize itself until it can contain this index.
  lldb::StreamSP s2(std::make_shared<StreamString>());
  tee.SetStreamAtIndex(4U, s2);
  // Check that the vector has been resized.
  EXPECT_EQ(5U, tee.GetNumStreams());
  // Is our stream at the right place?
  EXPECT_EQ(s2, tee.GetStreamAtIndex(4U));

  // Existing stream should still be there.
  EXPECT_EQ(s1, tee.GetStreamAtIndex(0U));
  // Other elements are all invalid StreamSPs.
  EXPECT_EQ(lldb::StreamSP(), tee.GetStreamAtIndex(1U));
  EXPECT_EQ(lldb::StreamSP(), tee.GetStreamAtIndex(2U));
  EXPECT_EQ(lldb::StreamSP(), tee.GetStreamAtIndex(3U));
  EXPECT_EQ(lldb::StreamSP(), tee.GetStreamAtIndex(5U));
}
