#include "llvm/Support/CheckedArithmetic.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(CheckedArithmetic, CheckedAdd) {
  const int64_t Max = std::numeric_limits<int64_t>::max();
  const int64_t Min = std::numeric_limits<int64_t>::min();
  EXPECT_EQ(checkedAdd<int64_t>(Max, Max), std::nullopt);
  EXPECT_EQ(checkedAdd<int64_t>(Min, -1), std::nullopt);
  EXPECT_EQ(checkedAdd<int64_t>(Max, 1), std::nullopt);
  EXPECT_EQ(checkedAdd<int64_t>(10, 1), Optional<int64_t>(11));
}

TEST(CheckedArithmetic, CheckedAddSmall) {
  const int16_t Max = std::numeric_limits<int16_t>::max();
  const int16_t Min = std::numeric_limits<int16_t>::min();
  EXPECT_EQ(checkedAdd<int16_t>(Max, Max), std::nullopt);
  EXPECT_EQ(checkedAdd<int16_t>(Min, -1), std::nullopt);
  EXPECT_EQ(checkedAdd<int16_t>(Max, 1), std::nullopt);
  EXPECT_EQ(checkedAdd<int16_t>(10, 1), Optional<int64_t>(11));
}

TEST(CheckedArithmetic, CheckedMul) {
  const int64_t Max = std::numeric_limits<int64_t>::max();
  const int64_t Min = std::numeric_limits<int64_t>::min();
  EXPECT_EQ(checkedMul<int64_t>(Max, 2), std::nullopt);
  EXPECT_EQ(checkedMul<int64_t>(Max, Max), std::nullopt);
  EXPECT_EQ(checkedMul<int64_t>(Min, 2), std::nullopt);
  EXPECT_EQ(checkedMul<int64_t>(10, 2), Optional<int64_t>(20));
}

TEST(CheckedArithmetic, CheckedMulAdd) {
  const int64_t Max = std::numeric_limits<int64_t>::max();
  const int64_t Min = std::numeric_limits<int64_t>::min();
  EXPECT_EQ(checkedMulAdd<int64_t>(Max, 1, 2), std::nullopt);
  EXPECT_EQ(checkedMulAdd<int64_t>(1, 1, Max), std::nullopt);
  EXPECT_EQ(checkedMulAdd<int64_t>(1, -1, Min), std::nullopt);
  EXPECT_EQ(checkedMulAdd<int64_t>(10, 2, 3), Optional<int64_t>(23));
}

TEST(CheckedArithmetic, CheckedMulSmall) {
  const int16_t Max = std::numeric_limits<int16_t>::max();
  const int16_t Min = std::numeric_limits<int16_t>::min();
  EXPECT_EQ(checkedMul<int16_t>(Max, 2), std::nullopt);
  EXPECT_EQ(checkedMul<int16_t>(Max, Max), std::nullopt);
  EXPECT_EQ(checkedMul<int16_t>(Min, 2), std::nullopt);
  EXPECT_EQ(checkedMul<int16_t>(10, 2), Optional<int16_t>(20));
}

TEST(CheckedArithmetic, CheckedMulAddSmall) {
  const int16_t Max = std::numeric_limits<int16_t>::max();
  const int16_t Min = std::numeric_limits<int16_t>::min();
  EXPECT_EQ(checkedMulAdd<int16_t>(Max, 1, 2), std::nullopt);
  EXPECT_EQ(checkedMulAdd<int16_t>(1, 1, Max), std::nullopt);
  EXPECT_EQ(checkedMulAdd<int16_t>(1, -1, Min), std::nullopt);
  EXPECT_EQ(checkedMulAdd<int16_t>(10, 2, 3), Optional<int16_t>(23));
}

TEST(CheckedArithmetic, CheckedAddUnsigned) {
  const uint64_t Max = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(checkedAddUnsigned<uint64_t>(Max, Max), std::nullopt);
  EXPECT_EQ(checkedAddUnsigned<uint64_t>(Max, 1), std::nullopt);
  EXPECT_EQ(checkedAddUnsigned<uint64_t>(10, 1), Optional<uint64_t>(11));
}

TEST(CheckedArithmetic, CheckedMulUnsigned) {
  const uint64_t Max = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(checkedMulUnsigned<uint64_t>(Max, 2), std::nullopt);
  EXPECT_EQ(checkedMulUnsigned<uint64_t>(Max, Max), std::nullopt);
  EXPECT_EQ(checkedMulUnsigned<uint64_t>(10, 2), Optional<uint64_t>(20));
}

TEST(CheckedArithmetic, CheckedMulAddUnsigned) {
  const uint64_t Max = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(checkedMulAddUnsigned<uint64_t>(Max, 1, 2), std::nullopt);
  EXPECT_EQ(checkedMulAddUnsigned<uint64_t>(1, 1, Max), std::nullopt);
  EXPECT_EQ(checkedMulAddUnsigned<uint64_t>(10, 2, 3), Optional<uint64_t>(23));
}


} // namespace
