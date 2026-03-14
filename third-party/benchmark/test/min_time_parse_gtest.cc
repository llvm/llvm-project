#include "../src/benchmark_runner.h"
#include "gtest/gtest.h"

namespace {

TEST(ParseMinTimeTest, InvalidInput) {
#if GTEST_HAS_DEATH_TEST
  // Tests only runnable in debug mode (when BM_CHECK is enabled).
#ifndef NDEBUG
#ifndef TEST_BENCHMARK_LIBRARY_HAS_NO_ASSERTIONS
  ASSERT_DEATH_IF_SUPPORTED(
      { benchmark::internal::ParseBenchMinTime("abc"); },
      "Malformed seconds value passed to --benchmark_min_time: `abc`");

  ASSERT_DEATH_IF_SUPPORTED(
      { benchmark::internal::ParseBenchMinTime("123ms"); },
      "Malformed seconds value passed to --benchmark_min_time: `123ms`");

  ASSERT_DEATH_IF_SUPPORTED(
      { benchmark::internal::ParseBenchMinTime("1z"); },
      "Malformed seconds value passed to --benchmark_min_time: `1z`");

  ASSERT_DEATH_IF_SUPPORTED(
      { benchmark::internal::ParseBenchMinTime("1hs"); },
      "Malformed seconds value passed to --benchmark_min_time: `1hs`");
#endif
#endif
#endif
}
}  // namespace
