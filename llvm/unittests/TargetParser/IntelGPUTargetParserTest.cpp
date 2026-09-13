//===------- IntelGPUTargetParserTest.cpp - Intel GPU Target Parser -------===//
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

TEST(IntelGPUTargetParserTest, CompatibilityNamesHaveNoGMDID) {
  // A compatibility name stands for a group of releases and no device reports
  // it, so no GMDID may ever resolve to one. A row that grew GMDID columns
  // would silently start being printed by the offload-arch utility.
  for (unsigned Architecture = 0; Architecture != 64; ++Architecture)
    for (unsigned Release = 0; Release != 256; ++Release) {
      IntelGPU::GPUKind Kind =
          IntelGPU::getKindForGMDID({Architecture, Release, 0});
      switch (Kind) {
      default:
        break;
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_LEVEL, IGCA_SUFFIX)                  \
  case IntelGPU::GK_##KIND:                                                    \
    FAIL() << NAME << " matched a GMDID";
#include "llvm/TargetParser/IntelGPUTargetParser.def"
      }
    }
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

TEST(IntelGPUTargetParserTest, ParseArch) {
  // A human-friendly name, a compatibility name, and an alias.
  EXPECT_EQ(IntelGPU::parseArch("xe-pvc"), IntelGPU::GK_XE_PVC);
  EXPECT_EQ(IntelGPU::parseArch("xe-ats-m150"), IntelGPU::GK_XE_ATS_M150);
  EXPECT_EQ(IntelGPU::parseArch("xe-dg2"), IntelGPU::GK_XE_DG2);
  EXPECT_EQ(IntelGPU::parseArch("bmg_g21"), IntelGPU::GK_XE_BMG_G21);

  // A numeric name names the same device the driver would have reported.
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60.0"), IntelGPU::GK_XE_PVC);
  // The revision takes no part in the lookup, and may be left out.
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60.7"), IntelGPU::GK_XE_PVC);
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60"), IntelGPU::GK_XE_PVC);
  // A numeric name for a device this build does not know names no device.
  EXPECT_EQ(IntelGPU::parseArch("xe_35.32.0"), IntelGPU::GK_NONE);

  // Neither an empty name nor a malformed one names a device.
  EXPECT_EQ(IntelGPU::parseArch(""), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("pvc"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_12"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_12.pvc"), IntelGPU::GK_NONE);
  EXPECT_EQ(IntelGPU::parseArch("xe_12.60.7.1"), IntelGPU::GK_NONE);
}

TEST(IntelGPUTargetParserTest, EveryNameParses) {
  // Every name the table declares has to be an --offload-arch value, and has to
  // name the row it came from.
#define INTEL_GPU(NAME, KIND, ARCHITECTURE, RELEASE, IGCA_LEVEL, IGCA_SUFFIX)  \
  EXPECT_EQ(IntelGPU::parseArch(NAME), IntelGPU::GK_##KIND) << NAME;
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_LEVEL, IGCA_SUFFIX)                  \
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
  // One of each suffix, since the level and the suffix are pasted together.
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
