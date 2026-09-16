//===-- IntelGPUTargetParserTest.cpp - Intel GPU Target Parser Test -------===//
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

// Build a GPU IP version the way the Level Zero driver reports it.
constexpr uint32_t gmdid(uint32_t Architecture, uint32_t Release,
                         uint32_t Revision) {
  return (Architecture << 22) | (Release << 14) | Revision;
}

// The bits between the release and the revision, which no component uses.
constexpr uint32_t GMDIDReservedBits = 0x3fc0;

TEST(IntelGPUTargetParserTest, ArchNames) {
  EXPECT_EQ(IntelGPU::getArchName(gmdid(12, 60, 7)), "xe-pvc");
  EXPECT_EQ(IntelGPU::getArchName(gmdid(35, 11, 0)), "xe-cri");
  EXPECT_EQ(IntelGPU::getArchName(gmdid(12, 55, 3)), "xe-acm-g10");
  // The revision is not part of the key: every stepping of a release is the
  // same device.
  EXPECT_EQ(IntelGPU::getArchName(gmdid(12, 60, 0)), "xe-pvc");
  EXPECT_EQ(IntelGPU::getArchName(gmdid(12, 60, 63)), "xe-pvc");
  // Neither are the reserved bits, whatever a driver reports in them.
  EXPECT_EQ(IntelGPU::getArchName(gmdid(12, 60, 7) | GMDIDReservedBits),
            "xe-pvc");
  // When several devices share an architecture and a release, the first row of
  // the group names the whole group.
  EXPECT_EQ(IntelGPU::getArchName(gmdid(30, 5, 0)), "xe-nvl-u");
  // A device that is not in the table has no name at all.
  EXPECT_EQ(IntelGPU::getArchName(gmdid(40, 11, 0)), "");
  EXPECT_EQ(IntelGPU::getArchName(gmdid(9, 0, 9)), "");
}

TEST(IntelGPUTargetParserTest, EveryDeviceIsNamed) {
  // A row that no GMDID can reach would make the offload-arch utility print an
  // empty architecture for a device the table does list, so every physical
  // device must resolve to some name. It need not be the name of the row
  // itself: a row that shares its architecture and release with an earlier one
  // is named after that earlier row. Compatibility names have no GMDID and so
  // cannot be looked up at all, which is why INTEL_GPU_COMPAT is left alone.
#define INTEL_GPU(NAME, KIND, ARCHITECTURE, RELEASE, IGCA_TARGET, IGCA_SUFFIX) \
  EXPECT_FALSE(IntelGPU::getArchName(gmdid(ARCHITECTURE, RELEASE, 0)).empty()) \
      << NAME;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
}

TEST(IntelGPUTargetParserTest, NumericArchName) {
  EXPECT_EQ(IntelGPU::getNumericArchName(gmdid(35, 11, 0)), "xe_35.11.0");
  EXPECT_EQ(IntelGPU::getNumericArchName(gmdid(12, 99, 3)), "xe_12.99.3");
  // Pre-Xe devices report a GMDID too, and none of them are in the table.
  EXPECT_EQ(IntelGPU::getNumericArchName(gmdid(9, 0, 9)), "xe_9.0.9");
  // The revision occupies the low 6 bits and the release the 8 above it, so
  // neither can bleed into the architecture.
  EXPECT_EQ(IntelGPU::getNumericArchName(gmdid(12, 0xff, 0x3f)),
            "xe_12.255.63");
  // The reserved bits belong to no component, so they are not spelled out.
  EXPECT_EQ(IntelGPU::getNumericArchName(gmdid(12, 60, 7) | GMDIDReservedBits),
            "xe_12.60.7");
}

} // namespace
