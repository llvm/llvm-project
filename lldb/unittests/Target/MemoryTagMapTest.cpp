//===-- MemoryTagMapTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/MemoryTagMap.h"
#include "Plugins/Process/Utility/MemoryTagManagerAArch64MTE.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

using namespace lldb_private;
using namespace lldb;

// In these tests we use the AArch64 MTE tag manager because it is the only
// implementation of a memory tag manager. MemoryTagMap itself is generic.

TEST(MemoryTagMapTest, EmptyTagMap) {
  MemoryTagManagerAArch64MTE manager;
  MemoryTagMap tag_map(&manager);

  tag_map.InsertTags(0, {});
  ASSERT_TRUE(tag_map.Empty());
  tag_map.InsertTags(0, {0});
  ASSERT_FALSE(tag_map.Empty());
}

TEST(MemoryTagMapTest, GetTags) {
  using TagsVec = std::vector<std::optional<lldb::addr_t>>;

  MemoryTagManagerAArch64MTE manager;
  MemoryTagMap tag_map(&manager);

  // No tags for an address not in the map
  ASSERT_TRUE(tag_map.GetTags(0, 16).empty());

  tag_map.InsertTags(0, {0, 1});

  // No tags if you read zero length
  ASSERT_TRUE(tag_map.GetTags(0, 0).empty());

  EXPECT_THAT(tag_map.GetTags(0, 16), ::testing::ContainerEq(TagsVec{0}));

  EXPECT_THAT(tag_map.GetTags(0, 32), ::testing::ContainerEq(TagsVec{0, 1}));

  // Last granule of the range is not tagged
  EXPECT_THAT(tag_map.GetTags(0, 48),
              ::testing::ContainerEq(TagsVec{0, 1, std::nullopt}));

  EXPECT_THAT(tag_map.GetTags(16, 32),
              ::testing::ContainerEq(TagsVec{1, std::nullopt}));

  // Reading beyond that address gives you no tags at all
  EXPECT_THAT(tag_map.GetTags(32, 16), ::testing::ContainerEq(TagsVec{}));

  // Address is granule aligned for you
  // The length here is set such that alignment doesn't produce a 2 granule
  // range.
  EXPECT_THAT(tag_map.GetTags(8, 8), ::testing::ContainerEq(TagsVec{0}));

  EXPECT_THAT(tag_map.GetTags(30, 2), ::testing::ContainerEq(TagsVec{1}));

  // Here the length pushes the range into the next granule. When aligned
  // this produces 2 granules.
  EXPECT_THAT(tag_map.GetTags(30, 4),
              ::testing::ContainerEq(TagsVec{1, std::nullopt}));

  // A range can also have gaps at the beginning or in the middle.
  // Add more tags, 1 granule away from the first range.
  tag_map.InsertTags(48, {3, 4});

  // Untagged first granule
  EXPECT_THAT(tag_map.GetTags(32, 32),
              ::testing::ContainerEq(TagsVec{std::nullopt, 3}));

  // Untagged middle granule
  EXPECT_THAT(tag_map.GetTags(16, 48),
              ::testing::ContainerEq(TagsVec{1, std::nullopt, 3}));
}
