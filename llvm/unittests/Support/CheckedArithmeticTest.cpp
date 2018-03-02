#include "llvm/Support/CheckedArithmetic.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(CheckedArithmetic, CheckedAdd) {
  int64_t Out;
  const int64_t Max = std::numeric_limits<int64_t>::max();
  const int64_t Min = std::numeric_limits<int64_t>::min();
  EXPECT_EQ(checkedAdd<int64_t>(Max, Max, &Out), true);
  EXPECT_EQ(checkedAdd<int64_t>(Min, -1, &Out), true);
  EXPECT_EQ(checkedAdd<int64_t>(Max, 1, &Out), true);
  EXPECT_EQ(checkedAdd<int64_t>(10, 1, &Out), false);
  EXPECT_EQ(Out, 11);
}

TEST(CheckedArithmetic, CheckedAddSmall) {
  int16_t Out;
  const int16_t Max = std::numeric_limits<int16_t>::max();
  const int16_t Min = std::numeric_limits<int16_t>::min();
  EXPECT_EQ(checkedAdd<int16_t>(Max, Max, &Out), true);
  EXPECT_EQ(checkedAdd<int16_t>(Min, -1, &Out), true);
  EXPECT_EQ(checkedAdd<int16_t>(Max, 1, &Out), true);
  EXPECT_EQ(checkedAdd<int16_t>(10, 1, &Out), false);
  EXPECT_EQ(Out, 11);
}

TEST(CheckedArithmetic, CheckedMul) {
  int64_t Out;
  const int64_t Max = std::numeric_limits<int64_t>::max();
  const int64_t Min = std::numeric_limits<int64_t>::min();
  EXPECT_EQ(checkedMul<int64_t>(Max, 2, &Out), true);
  EXPECT_EQ(checkedMul<int64_t>(Max, Max, &Out), true);
  EXPECT_EQ(checkedMul<int64_t>(Min, 2, &Out), true);
  EXPECT_EQ(checkedMul<int64_t>(10, 2, &Out), false);
  EXPECT_EQ(Out, 20);
}

TEST(CheckedArithmetic, CheckedMulSmall) {
  int16_t Out;
  const int16_t Max = std::numeric_limits<int16_t>::max();
  const int16_t Min = std::numeric_limits<int16_t>::min();
  EXPECT_EQ(checkedMul<int16_t>(Max, 2, &Out), true);
  EXPECT_EQ(checkedMul<int16_t>(Max, Max, &Out), true);
  EXPECT_EQ(checkedMul<int16_t>(Min, 2, &Out), true);
  EXPECT_EQ(checkedMul<int16_t>(10, 2, &Out), false);
  EXPECT_EQ(Out, 20);
}

TEST(CheckedArithmetic, CheckedAddUnsigned) {
  uint64_t Out;
  const uint64_t Max = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(checkedAddUnsigned<uint64_t>(Max, Max, &Out), true);
  EXPECT_EQ(checkedAddUnsigned<uint64_t>(Max, 1, &Out), true);
  EXPECT_EQ(checkedAddUnsigned<uint64_t>(10, 1, &Out), false);
  EXPECT_EQ(Out, uint64_t(11));
}

TEST(CheckedArithmetic, CheckedMulUnsigned) {
  uint64_t Out;
  const uint64_t Max = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(checkedMulUnsigned<uint64_t>(Max, 2, &Out), true);
  EXPECT_EQ(checkedMulUnsigned<uint64_t>(Max, Max, &Out), true);
  EXPECT_EQ(checkedMulUnsigned<uint64_t>(10, 2, &Out), false);
  EXPECT_EQ(Out, uint64_t(20));
}


} // namespace
