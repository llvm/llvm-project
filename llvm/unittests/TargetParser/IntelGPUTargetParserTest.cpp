//===-- IntelGPUTargetParserTest.cpp - Intel GPU Target Parser Test -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/IntelGPUTargetParser.h"
#include "gtest/gtest.h"
#include <cassert>

using namespace llvm;

namespace {

// Build a GPU IP version the way the Level Zero driver reports it. A component
// too wide for its field would corrupt the fields above it and quietly test
// something other than what it spells out.
constexpr uint32_t gpuIPVersion(uint32_t Major, uint32_t Minor,
                                uint32_t Revision) {
  assert((Major & ~0x3ffu) == 0 && "major version too wide");
  assert((Minor & ~0xffu) == 0 && "minor version too wide");
  assert((Revision & ~0x3fu) == 0 && "revision too wide");
  return (Major << 22) | (Minor << 14) | Revision;
}

// The bits between the minor version and the revision, which no component uses.
constexpr uint32_t GPUIPReservedBits = 0x3fc0;

TEST(IntelGPUTargetParserTest, ArchNames) {
  EXPECT_EQ(IntelGPU::getArchName(gpuIPVersion(12, 60, 7)), "xe-pvc");
  EXPECT_EQ(IntelGPU::getArchName(gpuIPVersion(35, 11, 0)), "xe-cri");
  EXPECT_EQ(IntelGPU::getArchName(gpuIPVersion(12, 55, 3)), "xe-acm-g10");
  // The revision is not part of the key: every revision of a device is that
  // same device.
  EXPECT_EQ(IntelGPU::getArchName(gpuIPVersion(12, 60, 0)), "xe-pvc");
  EXPECT_EQ(IntelGPU::getArchName(gpuIPVersion(12, 60, 63)), "xe-pvc");
  // Neither are the reserved bits, whatever a driver reports in them.
  EXPECT_EQ(IntelGPU::getArchName(gpuIPVersion(12, 60, 7) | GPUIPReservedBits),
            "xe-pvc");
  // When several devices share a major and a minor version, the first row of
  // the group names the whole group.
  EXPECT_EQ(IntelGPU::getArchName(gpuIPVersion(30, 5, 0)), "xe-nvl-u");
  // A device that is not in the table has no name at all.
  EXPECT_EQ(IntelGPU::getArchName(gpuIPVersion(40, 11, 0)), "");
  EXPECT_EQ(IntelGPU::getArchName(gpuIPVersion(9, 0, 9)), "");
}

TEST(IntelGPUTargetParserTest, EveryDeviceIsNamed) {
  // A row that no GPU IP version can reach would make the offload-arch utility
  // print an empty architecture for a device the table does list, so every
  // physical device must resolve to some name. It need not be the name of the
  // row itself: a row that shares its major and minor version with an earlier
  // one is named after that earlier row. Compatibility names have no version of
  // their own and so cannot be looked up, which is why INTEL_GPU_COMPAT is left
  // alone.
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  EXPECT_FALSE(IntelGPU::getArchName(gpuIPVersion(MAJOR, MINOR, 0)).empty())   \
      << NAME;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
}

TEST(IntelGPUTargetParserTest, NumericArchName) {
  EXPECT_EQ(IntelGPU::getNumericArchName(gpuIPVersion(35, 11, 0)),
            "xe_35.11.0");
  EXPECT_EQ(IntelGPU::getNumericArchName(gpuIPVersion(12, 99, 3)),
            "xe_12.99.3");
  // Pre-Xe devices report a version too, and none of them are in the table.
  EXPECT_EQ(IntelGPU::getNumericArchName(gpuIPVersion(9, 0, 9)), "xe_9.0.9");
  // The revision occupies the low 6 bits and the minor version the 8 above it,
  // so neither can bleed into the major version.
  EXPECT_EQ(IntelGPU::getNumericArchName(gpuIPVersion(12, 0xff, 0x3f)),
            "xe_12.255.63");
  // The reserved bits belong to no component, so they are not spelled out.
  EXPECT_EQ(
      IntelGPU::getNumericArchName(gpuIPVersion(12, 60, 7) | GPUIPReservedBits),
      "xe_12.60.7");
}

TEST(IntelGPUTargetParserTest, IGCATargetBehavior) {
  IntelGPU::IGCATarget T(60, IntelGPU::IGCA_COMPUTE, true);
  EXPECT_TRUE(T.isValid());
  EXPECT_EQ(T.getTarget(), 60);
  EXPECT_EQ(T.getFeatureSet(), IntelGPU::IGCA_COMPUTE);
  EXPECT_TRUE(T.isExact());
  EXPECT_TRUE(T.isComputeExact());
  EXPECT_FALSE(T.isRender());
  EXPECT_EQ(IntelGPU::IGCATarget::unpack(T.pack()), T);
  EXPECT_NE(T, IntelGPU::IGCATarget(60, IntelGPU::IGCA_COMPUTE, false));
  // Invalid behavior:
  EXPECT_FALSE(IntelGPU::IGCATarget::invalid());
  EXPECT_EQ(IntelGPU::IGCATarget::invalid().pack(), 0u);
}

TEST(IntelGPUTargetParserTest, ParseIGCATarget) {
  using IntelGPU::IGCATarget;
  EXPECT_EQ(IntelGPU::parseIGCATarget("igca_10"),
            IGCATarget(10, IntelGPU::IGCA_CORE, false));
  EXPECT_EQ(IntelGPU::parseIGCATarget("igca_20c"),
            IGCATarget(20, IntelGPU::IGCA_COMPUTE, false));
  EXPECT_EQ(IntelGPU::parseIGCATarget("igca_20ca"),
            IGCATarget(20, IntelGPU::IGCA_COMPUTE, true));
  EXPECT_EQ(IntelGPU::parseIGCATarget("igca_15r"),
            IGCATarget(15, IntelGPU::IGCA_RENDER, false));
  EXPECT_EQ(IntelGPU::parseIGCATarget("igca_15ra"),
            IGCATarget(15, IntelGPU::IGCA_RENDER, true));
  // Only levels in the table are valid targets.
  EXPECT_FALSE(IntelGPU::parseIGCATarget("igca_42"));
  EXPECT_FALSE(IntelGPU::parseIGCATarget("igca_0"));
  // Cannot have exact without naming feature set:
  EXPECT_FALSE(IntelGPU::parseIGCATarget("igca_10a"));
  // Malformed spellings:
  EXPECT_FALSE(IntelGPU::parseIGCATarget("igca_"));
  EXPECT_FALSE(IntelGPU::parseIGCATarget("igca_10x"));
  EXPECT_FALSE(IntelGPU::parseIGCATarget("igca_10cr"));
  EXPECT_FALSE(IntelGPU::parseIGCATarget("igca_10caa"));
  EXPECT_FALSE(IntelGPU::parseIGCATarget("igca10"));
  EXPECT_FALSE(IntelGPU::parseIGCATarget("IGCA_10"));
  EXPECT_FALSE(IntelGPU::parseIGCATarget(""));
}

TEST(IntelGPUTargetParserTest, IGCATargetName) {
  using IntelGPU::IGCATarget;
  EXPECT_EQ(
      IntelGPU::getIGCATargetName(IGCATarget(60, IntelGPU::IGCA_CORE, false)),
      "igca_60");
  EXPECT_EQ(
      IntelGPU::getIGCATargetName(IGCATarget(60, IntelGPU::IGCA_COMPUTE, true)),
      "igca_60ca");
  EXPECT_EQ(
      IntelGPU::getIGCATargetName(IGCATarget(15, IntelGPU::IGCA_RENDER, false)),
      "igca_15r");
  EXPECT_EQ(IntelGPU::getIGCATargetName(IGCATarget::invalid()), "");
  // A level that is not in the table has no spelling, and neither does an
  // exact core target.
  EXPECT_EQ(
      IntelGPU::getIGCATargetName(IGCATarget(11, IntelGPU::IGCA_CORE, false)),
      "");
  EXPECT_EQ(
      IntelGPU::getIGCATargetName(IGCATarget(10, IntelGPU::IGCA_CORE, true)),
      "");
}

TEST(IntelGPUTargetParserTest, EveryIGCASpellingRoundTrips) {
  SmallVector<StringRef> Names;
  IntelGPU::fillValidIGCATargetList(Names);
  EXPECT_FALSE(Names.empty());
  for (StringRef Name : Names) {
    IntelGPU::IGCATarget T = IntelGPU::parseIGCATarget(Name);
    EXPECT_TRUE(T.isValid()) << Name;
    EXPECT_EQ(IntelGPU::getIGCATargetName(T), Name);
  }
}

TEST(IntelGPUTargetParserTest, EveryDeviceHasValidIGCATargetLevel) {
  // Every device's defined IGCA target should have a corresponding entry in
  // INTEL_IGCA_TARGET:
#define INTEL_IGCA_TARGET_CHECK(NAME, IGCA_TARGET)                             \
  EXPECT_TRUE(IntelGPU::parseIGCATarget("igca_" #IGCA_TARGET).isValid())       \
      << NAME;
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  INTEL_IGCA_TARGET_CHECK(NAME, IGCA_TARGET)
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_TARGET, IGCA_FEATURE_SETS)           \
  INTEL_IGCA_TARGET_CHECK(NAME, IGCA_TARGET)
#include "llvm/TargetParser/IntelGPUTargetParser.def"
#undef INTEL_IGCA_TARGET_CHECK
}

} // namespace
