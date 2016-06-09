//===----------- TargetParser.cpp - Target Parser -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Support/TargetParser.h"

using namespace llvm;

namespace {
static const unsigned kAArch64ArchKinds[] = {
#define AARCH64_ARCH(NAME, ID, CPU_ATTR, SUB_ARCH, ARCH_ATTR, ARCH_FPU,        \
                     ARCH_BASE_EXT)                                            \
  llvm::ARM::ID,
#include "llvm/Support/AArch64TargetParser.def"
#undef AARCH64_ARCH
};

template <typename T, size_t N>
bool contains(const T (&array)[N], const T element) {
  return std::find(std::begin(array), std::end(array), element) !=
         std::end(array);
}

TEST(TargetParserTest, ARMArchName) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK <= ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_TRUE(AK == ARM::AK_LAST ? ARM::getArchName(AK).empty()
                                   : !ARM::getArchName(AK).empty());
}

TEST(TargetParserTest, ARMCPUAttr) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK <= ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_TRUE((AK == ARM::AK_INVALID || AK == ARM::AK_LAST)
                    ? ARM::getCPUAttr(AK).empty()
                    : !ARM::getCPUAttr(AK).empty());
}

TEST(TargetParserTest, ARMSubArch) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK <= ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_TRUE((AK == ARM::AK_INVALID || AK == ARM::AK_IWMMXT ||
                 AK == ARM::AK_IWMMXT2 || AK == ARM::AK_LAST)
                    ? ARM::getSubArch(AK).empty()
                    : !ARM::getSubArch(AK).empty());
}

TEST(TargetParserTest, ARMFPUName) {
  for (ARM::FPUKind FK = static_cast<ARM::FPUKind>(0);
       FK <= ARM::FPUKind::FK_LAST;
       FK = static_cast<ARM::FPUKind>(static_cast<unsigned>(FK) + 1))
    EXPECT_TRUE(FK == ARM::FK_LAST ? ARM::getFPUName(FK).empty()
                                   : !ARM::getFPUName(FK).empty());
}

TEST(TargetParserTest, AArch64ArchName) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK <= ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_TRUE(contains(kAArch64ArchKinds, static_cast<unsigned>(AK))
                    ? !AArch64::getArchName(AK).empty()
                    : AArch64::getArchName(AK).empty());
}

TEST(TargetParserTest, AArch64CPUAttr) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK <= ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_TRUE(contains(kAArch64ArchKinds, static_cast<unsigned>(AK))
                    ? !AArch64::getCPUAttr(AK).empty()
                    : AArch64::getCPUAttr(AK).empty());
}

TEST(TargetParserTest, AArch64SubArch) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK <= ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_TRUE(contains(kAArch64ArchKinds, static_cast<unsigned>(AK))
                    ? !AArch64::getSubArch(AK).empty()
                    : AArch64::getSubArch(AK).empty());
}
}

