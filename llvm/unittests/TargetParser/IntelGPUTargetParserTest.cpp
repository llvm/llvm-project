//===-- IntelGPUTargetParserTest.cpp - Intel GPU Target Parser Test -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/IntelGPUTargetParser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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

TEST(IntelGPUTargetParserTest, KindArchNames) {
  EXPECT_EQ(IntelGPU::getArchName(IntelGPU::GK_XE_PVC), "xe-pvc");
  EXPECT_EQ(IntelGPU::getArchName(IntelGPU::GK_XE_DG2), "xe-dg2");
  EXPECT_EQ(IntelGPU::getArchName(IntelGPU::GK_NONE), "");
}

TEST(IntelGPUTargetParserTest, ParseArch) {
  // A human-friendly name, a compatibility name, and an alias.
  EXPECT_EQ(IntelGPU::parseArch("xe-pvc"), IntelGPU::GK_XE_PVC);
  EXPECT_EQ(IntelGPU::parseArch("xe-dg2"), IntelGPU::GK_XE_DG2);
  EXPECT_EQ(IntelGPU::parseArch("bmg_g21"), IntelGPU::GK_XE_BMG_G21);
  // Products built on one GPU IP version are one device, named after the one
  // the offload-arch utility prints.
  EXPECT_EQ(IntelGPU::parseArch("xe-arl-u"), IntelGPU::GK_XE_MTL_U);
  EXPECT_EQ(IntelGPU::parseArch("xe-ats-m150"), IntelGPU::GK_XE_ACM_G10);
  EXPECT_EQ(IntelGPU::parseArch("xe_12.70"), IntelGPU::GK_XE_MTL_U);
  // Except PVC-SDV, which the AOT compiler builds for its own stepping.
  EXPECT_EQ(IntelGPU::parseArch("xe-pvc-sdv"), IntelGPU::GK_XE_PVC_SDV);

  // A numeric name names the same device the driver would have reported.
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60.0"), IntelGPU::GK_XE_PVC);
  // The revision takes no part in the lookup, and may be left out.
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60.7"), IntelGPU::GK_XE_PVC);
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60"), IntelGPU::GK_XE_PVC);
  // A numeric name for a device this build does not know names no device.
  EXPECT_EQ(IntelGPU::parseArch("xe_40.11.0"), IntelGPU::GK_NONE);

  // Neither an empty name nor a malformed one names a device.
  EXPECT_EQ(IntelGPU::parseArch(""), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("pvc"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_12"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_12.pvc"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60.7.1"), IntelGPU::GK_NONE);
  // A trailing separator is not an omitted revision.
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60."), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_12."), IntelGPU::GK_NONE);

  // A field wider than the GPU IP version field it spells describes no GPU
  // that could ever report it, so it names no device however well the rest of
  // the name matches a row. The major version holds 10 bits, the minor 8, the
  // revision 6.
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60.63"), IntelGPU::GK_XE_PVC);
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60.64"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_12.256"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_1024.60"), IntelGPU::GK_NONE);
  // The widest values that fit name no row either.
  EXPECT_EQ(IntelGPU::parseArch("xe_12.255"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_1023.60"), IntelGPU::GK_NONE);
  // Truncated to its field, each of these would name xe-pvc.
  EXPECT_EQ(IntelGPU::parseArch("xe_12.316"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_1036.60"), IntelGPU::GK_NONE);
  // Wider than the whole version, so the integer itself does not fit.
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60.99999999999"), IntelGPU::GK_NONE);
}

TEST(IntelGPUTargetParserTest, IsNumericArchName) {
  EXPECT_TRUE(IntelGPU::isNumericArchName("xe_12.60.7"));
  EXPECT_TRUE(IntelGPU::isNumericArchName("xe_12.60"));
  // A device the table does not know yet is still well-formed.
  EXPECT_TRUE(IntelGPU::isNumericArchName("xe_40.11.0"));
  EXPECT_FALSE(IntelGPU::isNumericArchName("xe-pvc"));
  EXPECT_FALSE(IntelGPU::isNumericArchName("xe_12.60."));
  EXPECT_FALSE(IntelGPU::isNumericArchName("xe_12.60.64"));
  EXPECT_FALSE(IntelGPU::isNumericArchName(""));
  EXPECT_FALSE(IntelGPU::isNumericArchName("xe_40"));
  EXPECT_FALSE(IntelGPU::isNumericArchName("XE_12.60"));
}

TEST(IntelGPUTargetParserTest, EveryNameParses) {
  // Every name the table declares has to be an --offload-arch value, and has to
  // name the row it came from.
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  EXPECT_EQ(IntelGPU::parseArch(NAME), IntelGPU::GK_##KIND) << NAME;
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_TARGET, IGCA_FEATURE_SETS)           \
  EXPECT_EQ(IntelGPU::parseArch(NAME), IntelGPU::GK_##KIND) << NAME;
#define INTEL_GPU_ALIAS(NAME, KIND)                                            \
  EXPECT_EQ(IntelGPU::parseArch(NAME), IntelGPU::GK_##KIND) << NAME;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
}

TEST(IntelGPUTargetParserTest, AliasesAreNeverReported) {
  // An alias declares no kind of its own, so the name reported for a device is
  // the row's own name. A device with two spellings would otherwise depend on
  // which one the table happened to reach first.
#define INTEL_GPU_ALIAS(NAME, KIND)                                            \
  EXPECT_NE(IntelGPU::getArchName(IntelGPU::GK_##KIND), NAME) << NAME;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
}

TEST(IntelGPUTargetParserTest, IGCANames) {
  // One of each suffix, since the target and the feature sets are pasted
  // together.
  EXPECT_EQ(IntelGPU::getIGCAName(IntelGPU::GK_XE_CRI), "igca_60c");
  EXPECT_EQ(IntelGPU::getIGCAName(IntelGPU::GK_XE_NVL_P), "igca_60r");
  EXPECT_EQ(IntelGPU::getIGCAName(IntelGPU::GK_XE_PVC), "igca_20ca");
  EXPECT_EQ(IntelGPU::getIGCAName(IntelGPU::GK_XE_DG2), "igca_15ra");
  EXPECT_EQ(IntelGPU::getIGCAName(IntelGPU::GK_NONE), "");
}

TEST(IntelGPUTargetParserTest, ValidArchList) {
  SmallVector<StringRef> Values;
  IntelGPU::fillValidArchList(Values);

  EXPECT_FALSE(Values.empty());
  EXPECT_NE(llvm::find(Values, "xe-pvc"), Values.end());
  // A compatibility name and an alias are as valid as any other name.
  EXPECT_NE(llvm::find(Values, "xe-dg2"), Values.end());
  EXPECT_NE(llvm::find(Values, "bmg_g21"), Values.end());
  // The list is offered to a user whose name did not parse, so every entry has
  // to be a name that would have.
  for (StringRef Value : Values)
    EXPECT_NE(IntelGPU::parseArch(Value), IntelGPU::GK_NONE) << Value;
}

} // namespace
