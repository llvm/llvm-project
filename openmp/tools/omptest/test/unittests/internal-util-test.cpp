#include "InternalEvent.h"
#include <omp-tools.h>

#include "gtest/gtest.h"

using namespace omptest;
using namespace util;

TEST(InternalUtility, ExpectedDefault_Integer) {
  // int: -2147483648 (decimal) = 0x80000000 (hexadecimal)
  EXPECT_EQ(expectedDefault(int), 0x80000000);
  EXPECT_EQ(expectedDefault(int), (0x1 << 31));
  // int64_t: -9223372036854775808 (decimal) = 0x8000000000000000 (hexadecimal)
  EXPECT_EQ(expectedDefault(int64_t), 0x8000000000000000);
  EXPECT_EQ(expectedDefault(int64_t), (0x1L << 63));
}

TEST(InternalUtility, ExpectedDefault_Zero) {
  // Expectedly zero
  EXPECT_EQ(expectedDefault(size_t), 0);
  EXPECT_EQ(expectedDefault(unsigned int), 0);
  EXPECT_EQ(expectedDefault(ompt_id_t), 0);
  EXPECT_EQ(expectedDefault(ompt_dispatch_t), 0);
  EXPECT_EQ(expectedDefault(ompt_device_time_t), 0);
}

TEST(InternalUtility, ExpectedDefault_Nullpointer) {
  // Expectedly nullptr
  EXPECT_EQ(expectedDefault(const char *), nullptr);
  EXPECT_EQ(expectedDefault(const void *), nullptr);
  EXPECT_EQ(expectedDefault(int *), nullptr);
  EXPECT_EQ(expectedDefault(void *), nullptr);
  EXPECT_EQ(expectedDefault(ompt_data_t *), nullptr);
  EXPECT_EQ(expectedDefault(ompt_device_t *), nullptr);
  EXPECT_EQ(expectedDefault(ompt_frame_t *), nullptr);
  EXPECT_EQ(expectedDefault(ompt_function_lookup_t), nullptr);
  EXPECT_EQ(expectedDefault(ompt_id_t *), nullptr);
}

TEST(InternalUtility, MakeHexString_PointerValues) {
  // IsPointer should only affect zero value
  EXPECT_EQ(makeHexString(0, /*IsPointer=*/true), "(nil)");
  EXPECT_EQ(makeHexString(0, /*IsPointer=*/false), "0x0");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true), "0xff");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/false), "0xff");
}

TEST(InternalUtility, MakeHexString_MinimumBytes) {
  // Return a minimum length, based on the (minimum) requested bytes
  EXPECT_EQ(makeHexString(15, /*IsPointer=*/true, /*MinBytes=*/0), "0xf");
  EXPECT_EQ(makeHexString(15, /*IsPointer=*/true, /*MinBytes=*/1), "0x0f");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true, /*MinBytes=*/0), "0xff");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true, /*MinBytes=*/1), "0xff");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true, /*MinBytes=*/2), "0x00ff");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true, /*MinBytes=*/3), "0x0000ff");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true, /*MinBytes=*/4),
            "0x000000ff");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true, /*MinBytes=*/5),
            "0x00000000ff");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true, /*MinBytes=*/6),
            "0x0000000000ff");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true, /*MinBytes=*/7),
            "0x000000000000ff");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true, /*MinBytes=*/8),
            "0x00000000000000ff");

  // Default to four bytes, if request exceeds eight byte range
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true, /*MinBytes=*/9),
            "0x000000ff");

  // Disregard requested minimum byte width, if actual value exceeds it
  EXPECT_EQ(makeHexString(1024, /*IsPointer=*/true, /*MinBytes=*/1), "0x400");
}

TEST(InternalUtility, MakeHexString_HexBase) {
  // Cut off "0x" when requested
  EXPECT_EQ(makeHexString(0, /*IsPointer=*/true, /*MinBytes=*/0,
                          /*ShowHexBase=*/false),
            "(nil)");
  EXPECT_EQ(makeHexString(0, /*IsPointer=*/false, /*MinBytes=*/0,
                          /*ShowHexBase=*/false),
            "0");
  EXPECT_EQ(makeHexString(0, /*IsPointer=*/false, /*MinBytes=*/1,
                          /*ShowHexBase=*/false),
            "00");
  EXPECT_EQ(makeHexString(255, /*IsPointer=*/true,
                          /*MinBytes=*/2,
                          /*ShowHexBase=*/false),
            "00ff");
}
