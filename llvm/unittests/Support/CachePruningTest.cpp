//===- CachePruningTest.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CachePruning.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(CachePruningPolicyParser, Empty) {
  auto P = parseCachePruningPolicy("");
  ASSERT_TRUE(bool(P));
  EXPECT_EQ(std::chrono::seconds(1200), P->Interval);
  EXPECT_EQ(std::chrono::hours(7 * 24), P->Expiration);
  EXPECT_EQ(75u, P->PercentageOfAvailableSpace);
}

TEST(CachePruningPolicyParser, Interval) {
  auto P = parseCachePruningPolicy("prune_interval=1s");
  ASSERT_TRUE(bool(P));
  EXPECT_EQ(std::chrono::seconds(1), P->Interval);
  P = parseCachePruningPolicy("prune_interval=2m");
  ASSERT_TRUE(bool(P));
  EXPECT_EQ(std::chrono::minutes(2), P->Interval);
  P = parseCachePruningPolicy("prune_interval=3h");
  ASSERT_TRUE(bool(P));
  EXPECT_EQ(std::chrono::hours(3), P->Interval);
}

TEST(CachePruningPolicyParser, Expiration) {
  auto P = parseCachePruningPolicy("prune_after=1s");
  ASSERT_TRUE(bool(P));
  EXPECT_EQ(std::chrono::seconds(1), P->Expiration);
}

TEST(CachePruningPolicyParser, PercentageOfAvailableSpace) {
  auto P = parseCachePruningPolicy("cache_size=100%");
  ASSERT_TRUE(bool(P));
  EXPECT_EQ(100u, P->PercentageOfAvailableSpace);
}

TEST(CachePruningPolicyParser, Multiple) {
  auto P = parseCachePruningPolicy("prune_after=1s:cache_size=50%");
  ASSERT_TRUE(bool(P));
  EXPECT_EQ(std::chrono::seconds(1200), P->Interval);
  EXPECT_EQ(std::chrono::seconds(1), P->Expiration);
  EXPECT_EQ(50u, P->PercentageOfAvailableSpace);
}

TEST(CachePruningPolicyParser, Errors) {
  EXPECT_EQ("Duration must not be empty",
            toString(parseCachePruningPolicy("prune_interval=").takeError()));
  EXPECT_EQ("'foo' not an integer",
            toString(parseCachePruningPolicy("prune_interval=foos").takeError()));
  EXPECT_EQ("'24x' must end with one of 's', 'm' or 'h'",
            toString(parseCachePruningPolicy("prune_interval=24x").takeError()));
  EXPECT_EQ("'foo' must be a percentage",
            toString(parseCachePruningPolicy("cache_size=foo").takeError()));
  EXPECT_EQ("'foo' not an integer",
            toString(parseCachePruningPolicy("cache_size=foo%").takeError()));
  EXPECT_EQ("'101' must be between 0 and 100",
            toString(parseCachePruningPolicy("cache_size=101%").takeError()));
  EXPECT_EQ("Unknown key: 'foo'",
            toString(parseCachePruningPolicy("foo=bar").takeError()));
}
