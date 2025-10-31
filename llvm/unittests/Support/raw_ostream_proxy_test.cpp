//===- raw_ostream_proxy_test.cpp - Tests for raw ostream proxies ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_ostream_proxy.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

/// Naive version of raw_svector_ostream that is buffered (by default) and
/// doesn't support pwrite.
class BufferedNoPwriteSmallVectorStream : public raw_ostream {
public:
  // Choose a strange buffer size to ensure it doesn't collide with the default
  // on \a raw_ostream.
  static constexpr size_t PreferredBufferSize = 63;

  size_t preferred_buffer_size() const override { return PreferredBufferSize; }
  uint64_t current_pos() const override { return Vector.size(); }
  void write_impl(const char *Ptr, size_t Size) override {
    Vector.append(Ptr, Ptr + Size);
  }

  bool is_displayed() const override { return IsDisplayed; }

  explicit BufferedNoPwriteSmallVectorStream(SmallVectorImpl<char> &Vector)
      : Vector(Vector) {}
  ~BufferedNoPwriteSmallVectorStream() override { flush(); }

  SmallVectorImpl<char> &Vector;
  bool IsDisplayed = false;
};

constexpr size_t BufferedNoPwriteSmallVectorStream::PreferredBufferSize;

TEST(raw_ostream_proxyTest, write) {
  // Besides confirming that "write" works, this test confirms that the proxy
  // takes on the buffer from the stream it's proxying, such that writes to the
  // proxy are flushed to the underlying stream as if no proxy were present.
  SmallString<128> Dest;
  {
    // Confirm that BufferedNoPwriteSmallVectorStream is buffered by default,
    // and that setting up a proxy effectively transfers a buffer of the same
    // size to the proxy.
    BufferedNoPwriteSmallVectorStream DestOS(Dest);
    EXPECT_EQ(BufferedNoPwriteSmallVectorStream::PreferredBufferSize,
              DestOS.GetBufferSize());
    raw_ostream_proxy ProxyOS(DestOS);
    EXPECT_EQ(0u, DestOS.GetBufferSize());
    EXPECT_EQ(BufferedNoPwriteSmallVectorStream::PreferredBufferSize,
              ProxyOS.GetBufferSize());

    // Flushing should send through to Dest.
    ProxyOS << "abcd";
    EXPECT_EQ("", Dest);
    ProxyOS.flush();
    EXPECT_EQ("abcd", Dest);

    // Buffer should still work.
    ProxyOS << "e";
    EXPECT_EQ("abcd", Dest);
  }

  // Destructing ProxyOS should flush (and not crash).
  EXPECT_EQ("abcde", Dest);

  {
    // Set up another stream, this time unbuffered.
    BufferedNoPwriteSmallVectorStream DestOS(Dest);
    DestOS.SetUnbuffered();
    EXPECT_EQ(0u, DestOS.GetBufferSize());
    raw_ostream_proxy ProxyOS(DestOS);
    EXPECT_EQ(0u, DestOS.GetBufferSize());
    EXPECT_EQ(0u, ProxyOS.GetBufferSize());

    // Flushing should not be required.
    ProxyOS << "f";
    EXPECT_EQ("abcdef", Dest);
  }
  EXPECT_EQ("abcdef", Dest);
}

TEST(raw_ostream_proxyTest, pwrite) {
  // This test confirms that the proxy takes on the buffer from the stream it's
  // proxying, such that writes to the proxy are flushed to the underlying
  // stream as if no proxy were present.
  SmallString<128> Dest;
  raw_svector_ostream DestOS(Dest);
  raw_pwrite_stream_proxy ProxyOS(DestOS);
  EXPECT_EQ(0u, ProxyOS.GetBufferSize());

  // Get some initial data.
  ProxyOS << "abcd";
  EXPECT_EQ("abcd", Dest);

  // Confirm that pwrite works.
  ProxyOS.pwrite("BC", 2, 1);
  EXPECT_EQ("aBCd", Dest);
}

TEST(raw_ostream_proxyTest, pwriteWithBuffer) {
  // This test confirms that when a buffer is configured, pwrite still works.
  SmallString<128> Dest;
  raw_svector_ostream DestOS(Dest);
  DestOS.SetBufferSize(256);
  EXPECT_EQ(256u, DestOS.GetBufferSize());

  // Confirm that the proxy steals the buffer.
  raw_pwrite_stream_proxy ProxyOS(DestOS);
  EXPECT_EQ(0u, DestOS.GetBufferSize());
  EXPECT_EQ(256u, ProxyOS.GetBufferSize());

  // Check that the buffer is working.
  ProxyOS << "abcd";
  EXPECT_EQ("", Dest);

  // Confirm that pwrite flushes.
  ProxyOS.pwrite("BC", 2, 1);
  EXPECT_EQ("aBCd", Dest);
}

class ProxyWithReset : public raw_ostream_proxy_adaptor<> {
public:
  ProxyWithReset(raw_ostream &OS) : raw_ostream_proxy_adaptor<>(OS) {}

  // Allow this to be called outside the class.
  using raw_ostream_proxy_adaptor<>::hasProxiedOS;
  using raw_ostream_proxy_adaptor<>::getProxiedOS;
  using raw_ostream_proxy_adaptor<>::resetProxiedOS;
};

TEST(raw_ostream_proxyTest, resetProxiedOS) {
  // Confirm that base classes can drop the proxied OS before destruction and
  // get consistent crashes.
  SmallString<128> Dest;
  BufferedNoPwriteSmallVectorStream DestOS(Dest);
  ProxyWithReset ProxyOS(DestOS);
  EXPECT_TRUE(ProxyOS.hasProxiedOS());
  EXPECT_EQ(&DestOS, &ProxyOS.getProxiedOS());

  // Write some data.
  ProxyOS << "abcd";
  EXPECT_EQ("", Dest);

  // Reset the underlying stream.
  ProxyOS.resetProxiedOS();
  EXPECT_EQ("abcd", Dest);
  EXPECT_EQ(0u, ProxyOS.GetBufferSize());
  EXPECT_FALSE(ProxyOS.hasProxiedOS());

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
  EXPECT_DEATH(ProxyOS << "e", "use after reset");
  EXPECT_DEATH(ProxyOS.getProxiedOS(), "use after reset");
#endif
}

TEST(raw_ostream_proxyTest, ColorMode) {
  {
    SmallString<128> Dest;
    BufferedNoPwriteSmallVectorStream DestOS(Dest);
    DestOS.IsDisplayed = true;
    raw_ostream_proxy ProxyOS(DestOS);
    ProxyOS.enable_colors(true);

    WithColor(ProxyOS, raw_ostream::Colors::RED, /*Bold=*/true, /*BG=*/false,
              ColorMode::Disable)
        << "test";
    EXPECT_EQ("", Dest);
    ProxyOS.flush();
    EXPECT_EQ("test", Dest);
  }

  {
    SmallString<128> Dest;
    BufferedNoPwriteSmallVectorStream DestOS(Dest);
    raw_ostream_proxy ProxyOS(DestOS);

    WithColor(ProxyOS, raw_ostream::Colors::RED, /*Bold=*/true, /*BG=*/false,
              ColorMode::Auto)
        << "test";
    EXPECT_EQ("", Dest);
    ProxyOS.flush();
    EXPECT_EQ("test", Dest);
  }

#ifdef LLVM_ON_UNIX
  {
    SmallString<128> Dest;
    BufferedNoPwriteSmallVectorStream DestOS(Dest);
    raw_ostream_proxy ProxyOS(DestOS);
    ProxyOS.enable_colors(true);

    WithColor(ProxyOS, raw_ostream::Colors::RED, /*Bold=*/true, /*BG=*/false,
              ColorMode::Enable)
        << "test";
    EXPECT_EQ("", Dest);
    ProxyOS.flush();
    EXPECT_EQ("\x1B[0;1;31mtest\x1B[0m", Dest);
  }

  {
    SmallString<128> Dest;
    BufferedNoPwriteSmallVectorStream DestOS(Dest);
    DestOS.IsDisplayed = true;
    raw_ostream_proxy ProxyOS(DestOS);
    ProxyOS.enable_colors(true);

    WithColor(ProxyOS, HighlightColor::Error, ColorMode::Auto) << "test";
    EXPECT_EQ("", Dest);
    ProxyOS.flush();
    EXPECT_EQ("\x1B[0;1;31mtest\x1B[0m", Dest);
  }
#endif
}

} // end namespace
