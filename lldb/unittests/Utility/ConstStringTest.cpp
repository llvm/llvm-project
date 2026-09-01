//===-- ConstStringTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/ConstString.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

#include <atomic>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace lldb_private;

TEST(ConstStringTest, format_provider) {
  EXPECT_EQ("foo", llvm::formatv("{0}", ConstString("foo")).str());
}

TEST(ConstStringTest, MangledCounterpart) {
  ConstString uvw("uvw");
  ConstString counterpart;
  EXPECT_FALSE(uvw.GetMangledCounterpart(counterpart));
  EXPECT_EQ("", counterpart.GetStringRef());

  ConstString xyz;
  xyz.SetStringWithMangledCounterpart("xyz", uvw);
  EXPECT_EQ("xyz", xyz.GetStringRef());

  EXPECT_TRUE(xyz.GetMangledCounterpart(counterpart));
  EXPECT_EQ("uvw", counterpart.GetStringRef());

  EXPECT_TRUE(uvw.GetMangledCounterpart(counterpart));
  EXPECT_EQ("xyz", counterpart.GetStringRef());
}

TEST(ConstStringTest, UpdateMangledCounterpart) {
  { // Add counterpart
    ConstString some1;
    some1.SetStringWithMangledCounterpart("some", ConstString(""));
  }
  { // Overwrite empty string
    ConstString some2;
    some2.SetStringWithMangledCounterpart("some", ConstString("one"));
  }
  { // Overwrite with identical value
    ConstString some2;
    some2.SetStringWithMangledCounterpart("some", ConstString("one"));
  }
  { // Check counterpart is set
    ConstString counterpart;
    EXPECT_TRUE(ConstString("some").GetMangledCounterpart(counterpart));
    EXPECT_EQ("one", counterpart.GetStringRef());
  }
}

TEST(ConstStringTest, FromMidOfBufferStringRef) {
  // StringRef's into bigger buffer: no null termination
  const char *buffer = "abcdefghi";
  llvm::StringRef foo_ref(buffer, 3);
  llvm::StringRef bar_ref(buffer + 3, 3);

  ConstString foo(foo_ref);

  ConstString bar;
  bar.SetStringWithMangledCounterpart(bar_ref, foo);
  EXPECT_EQ("def", bar.GetStringRef());

  ConstString counterpart;
  EXPECT_TRUE(bar.GetMangledCounterpart(counterpart));
  EXPECT_EQ("abc", counterpart.GetStringRef());

  EXPECT_TRUE(foo.GetMangledCounterpart(counterpart));
  EXPECT_EQ("def", counterpart.GetStringRef());
}

TEST(ConstStringTest, NullAndEmptyStates) {
  ConstString foo("foo");
  EXPECT_FALSE(!foo);
  EXPECT_FALSE(foo.IsEmpty());
  EXPECT_FALSE(foo.IsNull());

  ConstString empty("");
  EXPECT_TRUE(!empty);
  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_FALSE(empty.IsNull());

  ConstString null;
  EXPECT_TRUE(!null);
  EXPECT_TRUE(null.IsEmpty());
  EXPECT_TRUE(null.IsNull());
  EXPECT_TRUE(null.GetString().empty());
}

TEST(ConstStringTest, CompareConstString) {
  ConstString foo("foo");
  ConstString foo2("foo");
  ConstString bar("bar");

  EXPECT_TRUE(foo == foo2);
  EXPECT_TRUE(foo2 == foo);
  EXPECT_TRUE(foo == ConstString("foo"));

  EXPECT_FALSE(foo == bar);
  EXPECT_FALSE(foo2 == bar);
  EXPECT_FALSE(foo == ConstString("bar"));
  EXPECT_FALSE(foo == ConstString("different"));
  EXPECT_FALSE(foo == ConstString(""));
  EXPECT_FALSE(foo == ConstString());

  ConstString empty("");
  EXPECT_FALSE(empty == ConstString("bar"));
  EXPECT_FALSE(empty == ConstString());
  EXPECT_TRUE(empty == ConstString(""));

  ConstString null;
  EXPECT_FALSE(null == ConstString("bar"));
  EXPECT_TRUE(null == ConstString());
  EXPECT_FALSE(null == ConstString(""));
}

TEST(ConstStringTest, CompareStringRef) {
  ConstString foo("foo");

  EXPECT_TRUE(foo == "foo");
  EXPECT_TRUE(foo != "");
  EXPECT_FALSE(foo == static_cast<const char *>(nullptr));
  EXPECT_TRUE(foo != "bar");

  ConstString empty("");
  EXPECT_FALSE(empty == "foo");
  EXPECT_FALSE(empty != "");
  EXPECT_FALSE(empty == static_cast<const char *>(nullptr));
  EXPECT_TRUE(empty != "bar");

  ConstString null;
  EXPECT_FALSE(null == "foo");
  EXPECT_TRUE(null != "");
  EXPECT_TRUE(null == static_cast<const char *>(nullptr));
  EXPECT_TRUE(null != "bar");
}

TEST(ConstStringTest, StringConversions) {
  ConstString foo("foo");

  // Member functions.
  EXPECT_EQ(llvm::StringRef("foo"), foo.GetStringRef());
  EXPECT_EQ(std::string("foo"), foo.GetString());
  EXPECT_STREQ("foo", foo.AsCString(nullptr));

  // Conversion operators.
  EXPECT_EQ(llvm::StringRef("foo"), llvm::StringRef(foo));
  EXPECT_EQ(std::string("foo"), std::string_view(foo));
  EXPECT_EQ(std::string("foo"), std::string(foo));
}

// Stress-tests the ConstString pool from many threads at once with a mix of
// reads (lookups of already-interned strings) and writes (interning new
// strings). Intended both as a thread-safety regression check and as a
// hand-runnable benchmark for changes to the pool's locking strategy. Run
// timing is left to an external tool (e.g. hyperfine).
TEST(ConstStringTest, ConcurrentReadsAndWritesBenchmark) {
  // Pre-intern a set of "hot" strings that read operations will look up.
  // Sized to spread across the pool's 256 shards so contention is realistic.
  constexpr size_t kHotStringCount = 4096;
  std::vector<std::string> hot_strings;
  hot_strings.reserve(kHotStringCount);
  for (size_t i = 0; i < kHotStringCount; ++i) {
    hot_strings.push_back("ConstStringBench_hot_" + std::to_string(i));
    ConstString cs(hot_strings.back());
    (void)cs;
  }

  const unsigned num_threads =
      std::max(2u, std::thread::hardware_concurrency());
  constexpr size_t kIterationsPerThread = 200000;
  // 80% reads / 20% writes, which is probably generous as most lock contentions
  // is expected for writes while parsing the symtab in parallel.
  constexpr int kReadPercent = 80;

  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (unsigned t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      while (!start.load(std::memory_order_acquire))
        std::this_thread::yield();

      std::mt19937 rng(static_cast<uint32_t>(t) * 1337u + 17u);
      std::uniform_int_distribution<int> ratio_dist(0, 99);
      std::uniform_int_distribution<size_t> hot_dist(0, kHotStringCount - 1);
      uint64_t write_counter = 0;

      for (size_t i = 0; i < kIterationsPerThread; ++i) {
        if (ratio_dist(rng) < kReadPercent) {
          ConstString cs(hot_strings[hot_dist(rng)]);
          (void)cs;
        } else {
          std::string s = "ConstStringBench_tw_" + std::to_string(t) + "_" +
                          std::to_string(write_counter++);
          ConstString cs(s);
          (void)cs;
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto &th : threads)
    th.join();
}
