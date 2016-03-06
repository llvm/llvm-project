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
}

