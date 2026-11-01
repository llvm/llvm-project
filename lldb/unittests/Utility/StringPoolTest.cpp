//===-- StringPoolTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StringPool.h"
#include "gtest/gtest.h"

#include <string>
#include <thread>
#include <vector>

using namespace lldb_private;

TEST(StringPoolTest, InternDeduplicates) {
  StringPool pool;
  std::string a = "foo";
  const char *p1 = pool.Intern(a);
  // The pool owns a copy, so changing the source does not affect it.
  a[0] = 'x';
  const char *p2 = pool.Intern("foo");
  EXPECT_EQ(p1, p2);
  EXPECT_STREQ("foo", p1);
  EXPECT_NE(p1, pool.Intern("bar"));
}

TEST(StringPoolTest, InternNullAndEmpty) {
  StringPool pool;
  EXPECT_EQ(nullptr, pool.Intern(llvm::StringRef()));
  const char *empty = pool.Intern("");
  ASSERT_NE(nullptr, empty);
  EXPECT_STREQ("", empty);
}

TEST(StringPoolTest, InternNonEmpty) {
  StringPool pool;
  StringPoolRef ref(pool);
  EXPECT_EQ(nullptr, ref.InternNonEmpty(""));
  EXPECT_EQ(nullptr, ref.InternNonEmpty(llvm::StringRef()));
  EXPECT_EQ(pool.Intern("foo"), ref.InternNonEmpty("foo"));
}

TEST(StringPoolTest, GlobalPoolBacksConstString) {
  ConstString cs("global_pool_string");
  EXPECT_EQ(cs.GetCString(),
            StringPool::GetGlobal().Intern("global_pool_string"));
  EXPECT_EQ(cs.GetCString(), StringPoolRef().Intern("global_pool_string"));
}

TEST(StringPoolTest, SystemPool) {
  StringPool::Initialize();
  EXPECT_EQ(StringPool::GetGlobal().Intern("system"),
            StringPool::GetSystemPool().Intern("system"));
  StringPool::Terminate();
}

TEST(StringPoolTest, PoolsAreIndependent) {
  StringPool a, b;
  EXPECT_NE(a.Intern("shared"), b.Intern("shared"));
  EXPECT_NE(a.Intern("shared"), StringPool::GetGlobal().Intern("shared"));
  EXPECT_EQ(5u, StringPool::GetConstCStringLength(a.Intern("12345")));
}

TEST(StringPoolTest, ConcurrentIntern) {
  StringPool pool;
  constexpr int num_threads = 8;
  constexpr int num_strings = 1000;
  std::vector<std::vector<const char *>> results(num_threads);
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t)
    threads.emplace_back([&, t] {
      for (int i = 0; i < num_strings; ++i)
        results[t].push_back(pool.Intern(std::to_string(i)));
    });
  for (std::thread &t : threads)
    t.join();
  for (int t = 1; t < num_threads; ++t)
    EXPECT_EQ(results[0], results[t]);
}
