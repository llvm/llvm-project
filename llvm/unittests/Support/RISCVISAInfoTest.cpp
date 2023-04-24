//===-- unittests/RISCVISAInfoTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/RISCVISAInfo.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using ::testing::ElementsAre;

using namespace llvm;

bool operator==(const llvm::RISCVExtensionInfo &A,
                const llvm::RISCVExtensionInfo &B) {
  return A.ExtName == B.ExtName && A.MajorVersion == B.MajorVersion &&
         A.MinorVersion == B.MinorVersion;
}

TEST(ParseNormalizedArchString, RejectsUpperCase) {
  for (StringRef Input : {"RV32", "rV64", "rv32i2P0", "rv64i2p0_A2p0"}) {
    EXPECT_EQ(
        toString(RISCVISAInfo::parseNormalizedArchString(Input).takeError()),
        "string must be lowercase");
  }
}

TEST(ParseNormalizedArchString, RejectsInvalidBaseISA) {
  for (StringRef Input : {"rv32", "rv64", "rv32j", "rv65i"}) {
    EXPECT_EQ(
        toString(RISCVISAInfo::parseNormalizedArchString(Input).takeError()),
        "arch string must begin with valid base ISA");
  }
}

TEST(ParseNormalizedArchString, RejectsMalformedInputs) {
  for (StringRef Input : {"rv64i2p0_", "rv32i2p0__a2p0", "rv32e2.0", "rv64e2p",
                          "rv32i", "rv64ip1"}) {
    EXPECT_EQ(
        toString(RISCVISAInfo::parseNormalizedArchString(Input).takeError()),
        "extension lacks version in expected format");
  }
}

TEST(ParseNormalizedArchString, AcceptsValidBaseISAsAndSetsXLen) {
  auto MaybeRV32I = RISCVISAInfo::parseNormalizedArchString("rv32i2p0");
  ASSERT_THAT_EXPECTED(MaybeRV32I, Succeeded());
  RISCVISAInfo &InfoRV32I = **MaybeRV32I;
  EXPECT_EQ(InfoRV32I.getExtensions().size(), 1UL);
  EXPECT_TRUE(InfoRV32I.getExtensions().at("i") ==
              (RISCVExtensionInfo{"i", 2, 0}));
  EXPECT_EQ(InfoRV32I.getXLen(), 32U);

  auto MaybeRV32E = RISCVISAInfo::parseNormalizedArchString("rv32e2p0");
  ASSERT_THAT_EXPECTED(MaybeRV32E, Succeeded());
  RISCVISAInfo &InfoRV32E = **MaybeRV32E;
  EXPECT_EQ(InfoRV32E.getExtensions().size(), 1UL);
  EXPECT_TRUE(InfoRV32E.getExtensions().at("e") ==
              (RISCVExtensionInfo{"e", 2, 0}));
  EXPECT_EQ(InfoRV32I.getXLen(), 32U);

  auto MaybeRV64I = RISCVISAInfo::parseNormalizedArchString("rv64i2p0");
  ASSERT_THAT_EXPECTED(MaybeRV64I, Succeeded());
  RISCVISAInfo &InfoRV64I = **MaybeRV64I;
  EXPECT_EQ(InfoRV64I.getExtensions().size(), 1UL);
  EXPECT_TRUE(InfoRV64I.getExtensions().at("i") ==
              (RISCVExtensionInfo{"i", 2, 0}));
  EXPECT_EQ(InfoRV64I.getXLen(), 64U);

  auto MaybeRV64E = RISCVISAInfo::parseNormalizedArchString("rv64e2p0");
  ASSERT_THAT_EXPECTED(MaybeRV64E, Succeeded());
  RISCVISAInfo &InfoRV64E = **MaybeRV64E;
  EXPECT_EQ(InfoRV64E.getExtensions().size(), 1UL);
  EXPECT_TRUE(InfoRV64E.getExtensions().at("e") ==
              (RISCVExtensionInfo{"e", 2, 0}));
  EXPECT_EQ(InfoRV64I.getXLen(), 64U);
}

TEST(ParseNormalizedArchString, AcceptsArbitraryExtensionsAndVersions) {
  auto MaybeISAInfo = RISCVISAInfo::parseNormalizedArchString(
      "rv64i5p1_m3p2_zmadeup11p12_sfoo2p0_xbar3p0");
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  RISCVISAInfo &Info = **MaybeISAInfo;
  EXPECT_EQ(Info.getExtensions().size(), 5UL);
  EXPECT_TRUE(Info.getExtensions().at("i") == (RISCVExtensionInfo{"i", 5, 1}));
  EXPECT_TRUE(Info.getExtensions().at("m") == (RISCVExtensionInfo{"m", 3, 2}));
  EXPECT_TRUE(Info.getExtensions().at("zmadeup") ==
              (RISCVExtensionInfo{"zmadeup", 11, 12}));
  EXPECT_TRUE(Info.getExtensions().at("sfoo") ==
              (RISCVExtensionInfo{"sfoo", 2, 0}));
  EXPECT_TRUE(Info.getExtensions().at("xbar") ==
              (RISCVExtensionInfo{"xbar", 3, 0}));
}

TEST(ParseNormalizedArchString, UpdatesFLenMinVLenMaxELen) {
  auto MaybeISAInfo = RISCVISAInfo::parseNormalizedArchString(
      "rv64i2p0_d2p0_zvl64b1p0_zve64d1p0");
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  RISCVISAInfo &Info = **MaybeISAInfo;
  EXPECT_EQ(Info.getXLen(), 64U);
  EXPECT_EQ(Info.getFLen(), 64U);
  EXPECT_EQ(Info.getMinVLen(), 64U);
  EXPECT_EQ(Info.getMaxELen(), 64U);
}

TEST(ToFeatureVector, IIsDroppedAndExperimentalExtensionsArePrefixed) {
  auto MaybeISAInfo1 =
      RISCVISAInfo::parseArchString("rv64im_zihintntl", true, false);
  ASSERT_THAT_EXPECTED(MaybeISAInfo1, Succeeded());
  EXPECT_THAT((*MaybeISAInfo1)->toFeatureVector(),
              ElementsAre("+m", "+experimental-zihintntl"));

  auto MaybeISAInfo2 = RISCVISAInfo::parseArchString(
      "rv32e_zihintntl_xventanacondops", true, false);
  ASSERT_THAT_EXPECTED(MaybeISAInfo2, Succeeded());
  EXPECT_THAT((*MaybeISAInfo2)->toFeatureVector(),
              ElementsAre("+e", "+experimental-zihintntl", "+xventanacondops"));
}

TEST(ToFeatureVector, UnsupportedExtensionsAreDropped) {
  auto MaybeISAInfo =
      RISCVISAInfo::parseNormalizedArchString("rv64i2p0_m2p0_xmadeup1p0");
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  EXPECT_THAT((*MaybeISAInfo)->toFeatureVector(), ElementsAre("+m"));
}
