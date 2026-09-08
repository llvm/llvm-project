//===------- IntelGPUTargetParserTest.cpp - Intel GPU Target Parser -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/IntelGPUTargetParser.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Build a GMDID the way the Level Zero driver reports it.
constexpr uint32_t gmdid(uint32_t Architecture, uint32_t Release,
                         uint32_t Revision) {
  return (Architecture << 22) | (Release << 14) | Revision;
}

TEST(IntelGPUTargetParserTest, DecodeGMDID) {
  IntelGPU::GMDID ID = IntelGPU::decodeGMDID(gmdid(35, 11, 7));
  EXPECT_EQ(ID.Architecture, 35u);
  EXPECT_EQ(ID.Release, 11u);
  EXPECT_EQ(ID.Revision, 7u);

  // The revision occupies the low 6 bits and the release the 8 above it, so
  // neither can bleed into the architecture.
  ID = IntelGPU::decodeGMDID(gmdid(12, 0xff, 0x3f));
  EXPECT_EQ(ID.Architecture, 12u);
  EXPECT_EQ(ID.Release, 0xffu);
  EXPECT_EQ(ID.Revision, 0x3fu);
}

TEST(IntelGPUTargetParserTest, KindForGMDID) {
  EXPECT_EQ(IntelGPU::getKindForGMDID({12, 60, 7}), IntelGPU::GK_XE_PVC);
  EXPECT_EQ(IntelGPU::getKindForGMDID({35, 11, 0}), IntelGPU::GK_XE_CRI);
  // The revision is not part of the key: every stepping of a release is the
  // same device.
  EXPECT_EQ(IntelGPU::getKindForGMDID({12, 60, 0}), IntelGPU::GK_XE_PVC);
  EXPECT_EQ(IntelGPU::getKindForGMDID({12, 60, 63}), IntelGPU::GK_XE_PVC);
  // When several devices share an architecture and a release, the first row of
  // the group wins.
  EXPECT_EQ(IntelGPU::getKindForGMDID({30, 5, 0}), IntelGPU::GK_XE_NVL_U);
  EXPECT_EQ(IntelGPU::getKindForGMDID({12, 55, 0}), IntelGPU::GK_XE_ACM_G10);
  // A device that is not in the table has no kind at all.
  EXPECT_EQ(IntelGPU::getKindForGMDID({40, 11, 0}), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::getKindForGMDID({9, 0, 9}), IntelGPU::GK_NONE);
}

TEST(IntelGPUTargetParserTest, ArchNames) {
  EXPECT_EQ(IntelGPU::getArchName(IntelGPU::GK_XE_PVC), "xe-pvc");
  EXPECT_EQ(IntelGPU::getArchName(IntelGPU::GK_XE_ATS_M150), "xe-ats-m150");
  EXPECT_EQ(IntelGPU::getArchName(IntelGPU::GK_XE_MTL), "xe-mtl");
  EXPECT_EQ(IntelGPU::getArchName(IntelGPU::GK_NONE), "");
}

TEST(IntelGPUTargetParserTest, EveryKindIsNamed) {
  // A row with no name would make the offload-arch utility print an empty
  // architecture, so every kind the table declares must have a spelling.
#define INTEL_GPU(NAME, KIND, ARCHITECTURE, RELEASE, IGCA_LEVEL, IGCA_SUFFIX)  \
  EXPECT_FALSE(IntelGPU::getArchName(IntelGPU::GK_##KIND).empty()) << #KIND;
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_LEVEL, IGCA_SUFFIX)                  \
  EXPECT_FALSE(IntelGPU::getArchName(IntelGPU::GK_##KIND).empty()) << #KIND;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
}

TEST(IntelGPUTargetParserTest, NumericArchName) {
  EXPECT_EQ(IntelGPU::getNumericArchName({35, 11, 0}), "xe_35.11.0");
  EXPECT_EQ(IntelGPU::getNumericArchName({12, 99, 3}), "xe_12.99.3");
  // Pre-Xe devices report a GMDID too, and none of them are in the table.
  EXPECT_EQ(IntelGPU::getNumericArchName({9, 0, 9}), "xe_9.0.9");
}

} // namespace
