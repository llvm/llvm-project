//===---------------------------------------------------------------------===//
// string_util_test - Unit tests for src/string_util.cc
//===---------------------------------------------------------------------===//

#include <tuple>

#include "../src/internal_macros.h"
#include "../src/string_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {
TEST(StringUtilTest, stoul) {
  {
    size_t pos = 0;
    EXPECT_EQ(0ul, benchmark::stoul("0", &pos));
    EXPECT_EQ(1ul, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(7ul, benchmark::stoul("7", &pos));
    EXPECT_EQ(1ul, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(135ul, benchmark::stoul("135", &pos));
    EXPECT_EQ(3ul, pos);
  }
#if ULONG_MAX == 0xFFFFFFFFul
  {
    size_t pos = 0;
    EXPECT_EQ(0xFFFFFFFFul, benchmark::stoul("4294967295", &pos));
    EXPECT_EQ(10ul, pos);
  }
#elif ULONG_MAX == 0xFFFFFFFFFFFFFFFFul
  {
    size_t pos = 0;
    EXPECT_EQ(0xFFFFFFFFFFFFFFFFul,
              benchmark::stoul("18446744073709551615", &pos));
    EXPECT_EQ(20ul, pos);
  }
#endif
  {
    size_t pos = 0;
    EXPECT_EQ(10ul, benchmark::stoul("1010", &pos, 2));
    EXPECT_EQ(4ul, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(520ul, benchmark::stoul("1010", &pos, 8));
    EXPECT_EQ(4ul, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(1010ul, benchmark::stoul("1010", &pos, 10));
    EXPECT_EQ(4ul, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(4112ul, benchmark::stoul("1010", &pos, 16));
    EXPECT_EQ(4ul, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(0xBEEFul, benchmark::stoul("BEEF", &pos, 16));
    EXPECT_EQ(4ul, pos);
  }
#ifndef BENCHMARK_HAS_NO_EXCEPTIONS
  {
    ASSERT_THROW(std::ignore = benchmark::stoul("this is a test"),
                 std::invalid_argument);
  }
#endif
}

TEST(StringUtilTest, stoi){{size_t pos = 0;
EXPECT_EQ(0, benchmark::stoi("0", &pos));
EXPECT_EQ(1ul, pos);
}  // namespace
{
  size_t pos = 0;
  EXPECT_EQ(-17, benchmark::stoi("-17", &pos));
  EXPECT_EQ(3ul, pos);
}
{
  size_t pos = 0;
  EXPECT_EQ(1357, benchmark::stoi("1357", &pos));
  EXPECT_EQ(4ul, pos);
}
{
  size_t pos = 0;
  EXPECT_EQ(10, benchmark::stoi("1010", &pos, 2));
  EXPECT_EQ(4ul, pos);
}
{
  size_t pos = 0;
  EXPECT_EQ(520, benchmark::stoi("1010", &pos, 8));
  EXPECT_EQ(4ul, pos);
}
{
  size_t pos = 0;
  EXPECT_EQ(1010, benchmark::stoi("1010", &pos, 10));
  EXPECT_EQ(4ul, pos);
}
{
  size_t pos = 0;
  EXPECT_EQ(4112, benchmark::stoi("1010", &pos, 16));
  EXPECT_EQ(4ul, pos);
}
{
  size_t pos = 0;
  EXPECT_EQ(0xBEEF, benchmark::stoi("BEEF", &pos, 16));
  EXPECT_EQ(4ul, pos);
}
#ifndef BENCHMARK_HAS_NO_EXCEPTIONS
{
  ASSERT_THROW(std::ignore = benchmark::stoi("this is a test"),
               std::invalid_argument);
}
#endif
}

TEST(StringUtilTest, stod){{size_t pos = 0;
EXPECT_EQ(0.0, benchmark::stod("0", &pos));
EXPECT_EQ(1ul, pos);
}
{
  size_t pos = 0;
  EXPECT_EQ(-84.0, benchmark::stod("-84", &pos));
  EXPECT_EQ(3ul, pos);
}
{
  size_t pos = 0;
  EXPECT_EQ(1234.0, benchmark::stod("1234", &pos));
  EXPECT_EQ(4ul, pos);
}
{
  size_t pos = 0;
  EXPECT_EQ(1.5, benchmark::stod("1.5", &pos));
  EXPECT_EQ(3ul, pos);
}
{
  size_t pos = 0;
  /* Note: exactly representable as double */
  EXPECT_EQ(-1.25e+9, benchmark::stod("-1.25e+9", &pos));
  EXPECT_EQ(8ul, pos);
}
#ifndef BENCHMARK_HAS_NO_EXCEPTIONS
{
  ASSERT_THROW(std::ignore = benchmark::stod("this is a test"),
               std::invalid_argument);
}
#endif
}

TEST(StringUtilTest, StrSplit) {
  EXPECT_EQ(benchmark::StrSplit("", ','), std::vector<std::string>{});
  EXPECT_EQ(benchmark::StrSplit("hello", ','),
            std::vector<std::string>({"hello"}));
  EXPECT_EQ(benchmark::StrSplit("hello,there,is,more", ','),
            std::vector<std::string>({"hello", "there", "is", "more"}));
}

using HumanReadableFixture = ::testing::TestWithParam<
    std::tuple<double, benchmark::Counter::OneK, std::string>>;

INSTANTIATE_TEST_SUITE_P(
    HumanReadableTests, HumanReadableFixture,
    ::testing::Values(
        std::make_tuple(0.0, benchmark::Counter::kIs1024, "0"),
        std::make_tuple(999.0, benchmark::Counter::kIs1024, "999"),
        std::make_tuple(1000.0, benchmark::Counter::kIs1024, "1000"),
        std::make_tuple(1024.0, benchmark::Counter::kIs1024, "1Ki"),
        std::make_tuple(1000 * 1000.0, benchmark::Counter::kIs1024,
                        "976\\.56.Ki"),
        std::make_tuple(1024 * 1024.0, benchmark::Counter::kIs1024, "1Mi"),
        std::make_tuple(1000 * 1000 * 1000.0, benchmark::Counter::kIs1024,
                        "953\\.674Mi"),
        std::make_tuple(1024 * 1024 * 1024.0, benchmark::Counter::kIs1024,
                        "1Gi"),
        std::make_tuple(0.0, benchmark::Counter::kIs1000, "0"),
        std::make_tuple(999.0, benchmark::Counter::kIs1000, "999"),
        std::make_tuple(1000.0, benchmark::Counter::kIs1000, "1k"),
        std::make_tuple(1024.0, benchmark::Counter::kIs1000, "1.024k"),
        std::make_tuple(1000 * 1000.0, benchmark::Counter::kIs1000, "1M"),
        std::make_tuple(1024 * 1024.0, benchmark::Counter::kIs1000,
                        "1\\.04858M"),
        std::make_tuple(1000 * 1000 * 1000.0, benchmark::Counter::kIs1000,
                        "1G"),
        std::make_tuple(1024 * 1024 * 1024.0, benchmark::Counter::kIs1000,
                        "1\\.07374G")));

TEST_P(HumanReadableFixture, HumanReadableNumber) {
  std::string str = benchmark::HumanReadableNumber(std::get<0>(GetParam()),
                                                   std::get<1>(GetParam()));
  ASSERT_THAT(str, ::testing::MatchesRegex(std::get<2>(GetParam())));
}

}  // end namespace
