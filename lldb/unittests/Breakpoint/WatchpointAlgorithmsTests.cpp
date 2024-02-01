//===-- WatchpointAlgorithmsTests.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Breakpoint/WatchpointAlgorithms.h"

#include <utility>
#include <vector>

using namespace lldb;
using namespace lldb_private;

struct testcase {
  WatchpointAlgorithms::Region user; // What the user requested
  std::vector<WatchpointAlgorithms::Region>
      hw; // The hardware watchpoints we'll use
};

class WatchpointAlgorithmsTest : public WatchpointAlgorithms {
public:
  using WatchpointAlgorithms::PowerOf2Watchpoints;
};

void check_testcase(testcase test,
                    std::vector<WatchpointAlgorithms::Region> result,
                    size_t min_byte_size, size_t max_byte_size,
                    uint32_t address_byte_size) {

  EXPECT_EQ(result.size(), test.hw.size());
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(result[i].addr, test.hw[i].addr);
    EXPECT_EQ(result[i].size, test.hw[i].size);
  }
}

TEST(WatchpointAlgorithmsTests, PowerOf2Watchpoints) {

#if 0
  // clang-format off
  std::vector<testcase> doubleword_max = {
    {
      {0x7fffffffe83b, 1},
      {{0x7fffffffe83b, 1}}
    },
    {
      {0x7fffffffe838, 2},
      {{0x7fffffffe838, 2}}
    },
    {
      {0x1012, 8},
      {{0x1010, 8}, {0x1018, 8}}
    },
    {
      {0x1002, 4},
      {{0x1000, 8}}
    },
    {
      {0x1006, 4},
      {{0x1004, 4}, {0x1008, 4}}
    },
    {
      {0x1006, 8},
      {{0x1000, 8}, {0x1008, 8}}
    },
    {
      {0x1000, 24},
      {{0x1000, 8}, {0x1008, 8}, {0x1010, 8}}
    },
    {
      {0x1014, 26},
      {{0x1010, 8}, {0x1018, 8}, {0x1020, 8}, {0x1028, 8}}
    },
  };
  // clang-format on
  printf("Running doubleword max test cases\n");
  for (testcase test : doubleword_max) {
    addr_t user_addr = test.user.addr;
    size_t user_size = test.user.size;
    size_t min_byte_size = 1;
    size_t max_byte_size = 8;
    size_t address_byte_size = 8;
    printf("Calling with addr 0x%llx size %zu\n", user_addr, user_size);
    auto result = WatchpointAlgorithmsTest::PowerOf2Watchpoints(
        user_addr, user_size, min_byte_size, max_byte_size, address_byte_size);

    printf("Checking returned values\n");
    check_testcase(test, result, min_byte_size, max_byte_size,
                   address_byte_size);
  }
#endif

  // clang-format off
  std::vector<testcase> word_max = {
    {
      {0x00411050, 4},
      {{0x00411050, 4}}
    },
    {
      {0x1002, 4},
      {{0x1000, 4}, {0x1004, 4}}
    },
  };
  // clang-format on
  printf("Running word max test cases\n");
  for (testcase test : word_max) {
    addr_t user_addr = test.user.addr;
    size_t user_size = test.user.size;
    size_t min_byte_size = 1;
    size_t max_byte_size = 4;
    size_t address_byte_size = 4;
    printf("Calling with addr 0x%llx size %zu\n", user_addr, user_size);
    auto result = WatchpointAlgorithmsTest::PowerOf2Watchpoints(
        user_addr, user_size, min_byte_size, max_byte_size, address_byte_size);

    printf("Checking returned values\n");
    check_testcase(test, result, min_byte_size, max_byte_size,
                   address_byte_size);
  }

#if 0
  // clang-format off
  std::vector<testcase> twogig_max = {
    {
      {0x1010, 16},
      {{0x1010, 16}}
    },
    {
      {0x1010, 24},
      {{0x1000, 64}}
    },

    // We increase 36 to the aligned 64 byte size, but
    // 0x1000-0x1040 doesn't cover the requested region.  Then
    // we expand to 128 bytes starting at 0x1000 that does
    // cover it.  Is this a good tradeoff for a 36 byte region?
    {
      {0x1024, 36},
      {{0x1000, 128}}
    },
    {
      {0x1000, 192},
      {{0x1000, 256}}
    },
    {
      {0x1080, 192},
      {{0x1000, 512}}
    },

    // In this case, our aligned size is 128, and increasing it to 256
    // still can't watch the requested region.  The algorithm
    // falls back to using two 128 byte watchpoints.  
    // The alternative would be to use a 1024B watchpoint 
    // starting at 0x1000, to watch this 120 byte user request.
    //
    // This still isn't ideal.  The user is asking to watch 0x12e0-1358
    // and could be optimally handled by a 
    // 16-byte watchpoint at 0x12e0 and a 128-byte watchpoint at 0x1300
    {
      {0x12e0, 120},
      {{0x1280, 128}, {0x1300, 128}}
    },
  };
  // clang-format on
  printf("Running word twogig test cases\n");
  for (testcase test : twogig_max) {
    addr_t user_addr = test.user.addr;
    size_t user_size = test.user.size;
    size_t min_byte_size = 1;
    size_t max_byte_size = INT32_MAX;
    size_t address_byte_size = 8;
    printf("Calling with addr 0x%llx size %zu\n", user_addr, user_size);
    auto result = WatchpointAlgorithmsTest::PowerOf2Watchpoints(
        user_addr, user_size, min_byte_size, max_byte_size, address_byte_size);

    printf("Checking returned values\n");
    check_testcase(test, result, min_byte_size, max_byte_size,
                   address_byte_size);
  }
#endif

}
