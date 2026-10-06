#include "llvm/Support/CHERICapabilityFormat.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(CHERICapabilityFormat, RV32Y) {
  using RV32Y = RV32YCapabilityFormat;

  EXPECT_EQ(RV32Y::AddressMask, 0xFFFFFFFF);

  // Lengths up to 255 are byte-aligned.
  for (uint64_t Len = 1; Len <= 255; ++Len) {
    EXPECT_EQ(RV32Y::getRepresentableLength(Len), Len);
    EXPECT_EQ(RV32Y::getRequiredAlignment(Len), 1);
    EXPECT_EQ(RV32Y::getAlignmentMask(Len), 0xFFFFFFFF);
  }

  // Lengths up to 511 are 8-byte-aligned.
  for (uint64_t Len = 256; Len <= 511; ++Len) {
    EXPECT_EQ(RV32Y::getRepresentableLength(Len), (Len + 7) & 0xFFFFFFF8);
    EXPECT_EQ(RV32Y::getRequiredAlignment(Len), 8);
    EXPECT_EQ(RV32Y::getAlignmentMask(Len), 0xFFFFFFF8);
  }

  // Lengths up to 1023 are 16-byte-aligned.
  for (uint64_t Len = 512; Len <= 1023; ++Len) {
    EXPECT_EQ(RV32Y::getRepresentableLength(Len), (Len + 15) & 0xFFFFFFF0);
    EXPECT_EQ(RV32Y::getRequiredAlignment(Len), 16);
    EXPECT_EQ(RV32Y::getAlignmentMask(Len), 0xFFFFFFF0);
  }

  EXPECT_EQ(RV32Y::getRepresentableLength(0xFFFFFFF0), 0ULL);
  EXPECT_EQ(RV32Y::getRequiredAlignment(0xFFFFFFF0), 67108864);
  EXPECT_EQ(RV32Y::getAlignmentMask(0xFFFFFFF0), 0xFC000000);
}

TEST(CHERICapabilityFormat, RV64Y) {
  using RV64Y = RV64YCapabilityFormat;

  EXPECT_EQ(RV64Y::AddressMask, 0xFFFFFFFFFFFFFFFF);

  // Lengths up to 4095 are byte-aligned.
  for (uint64_t Len = 1; Len <= 4095; ++Len) {
    EXPECT_EQ(RV64Y::getRepresentableLength(Len), Len);
    EXPECT_EQ(RV64Y::getRequiredAlignment(Len), 1);
    EXPECT_EQ(RV64Y::getAlignmentMask(Len), 0xFFFFFFFFFFFFFFFF);
  }

  // Lengths up to 8191 are 8-byte-aligned.
  for (uint64_t Len = 4096; Len <= 8191; ++Len) {
    assert(RV64Y::getRepresentableLength(Len) ==
           ((Len + 7) & 0xFFFFFFFFFFFFFFF8));
    EXPECT_EQ(RV64Y::getRequiredAlignment(Len), 8);
    EXPECT_EQ(RV64Y::getAlignmentMask(Len), 0xFFFFFFFFFFFFFFF8);
  }

  // Lengths up to 16383 are 16-byte-aligned.
  for (uint64_t Len = 8192; Len <= 16383; ++Len) {
    EXPECT_EQ(RV64Y::getRepresentableLength(Len),
              (Len + 15) & 0xFFFFFFFFFFFFFFF0);
    EXPECT_EQ(RV64Y::getRequiredAlignment(Len), 16);
    EXPECT_EQ(RV64Y::getAlignmentMask(Len), 0xFFFFFFFFFFFFFFF0);
  }

  EXPECT_EQ(RV64Y::getRepresentableLength(0xFFFFFFFFFFFFFFF0), 0ULL);
  EXPECT_EQ(RV64Y::getRequiredAlignment(0xFFFFFFFFFFFFFFF0),
            18014398509481984ULL);
  EXPECT_EQ(RV64Y::getAlignmentMask(0xFFFFFFFFFFFFFFF0), 0xFFC0000000000000);
}

TEST(CHERICapabilityFormat, CHERIoT) {
  using CHERIoT = CHERIoTCapabilityFormat;

  EXPECT_EQ(CHERIoT::AddressMask, 0xFFFFFFFF);

  // Lengths up to 511 are byte-aligned.
  for (uint64_t Len = 1; Len <= 511; ++Len) {
    EXPECT_EQ(CHERIoT::getRepresentableLength(Len), Len);
    assert(CHERIoT::getRequiredAlignment(Len) == 1);
    EXPECT_EQ(CHERIoT::getAlignmentMask(Len), 0xFFFFFFFF);
  }

  // Lengths up to 1022 are 2-byte-aligned.
  for (uint64_t Len = 512; Len <= 1022; ++Len) {
    EXPECT_EQ(CHERIoT::getRepresentableLength(Len), (Len + 1) & 0xFFFFFFFE);
    EXPECT_EQ(CHERIoT::getRequiredAlignment(Len), 2);
    EXPECT_EQ(CHERIoT::getAlignmentMask(Len), 0xFFFFFFFE);
  }

  // Lengths up to 2044 are 4-byte-aligned.
  for (uint64_t Len = 1023; Len <= 2044; ++Len) {
    EXPECT_EQ(CHERIoT::getRepresentableLength(Len), (Len + 3) & 0xFFFFFFFC);
    assert(CHERIoT::getRequiredAlignment(Len) == 4);
    EXPECT_EQ(CHERIoT::getAlignmentMask(Len), 0xFFFFFFFC);
  }

  EXPECT_EQ(CHERIoT::getRepresentableLength(0xFFFFFFF0), 0ULL);
  EXPECT_EQ(CHERIoT::getRequiredAlignment(0xFFFFFFF0), 16777216ULL);
  EXPECT_EQ(CHERIoT::getAlignmentMask(0xFFFFFFF0), 0xFF000000);
}

} // namespace
