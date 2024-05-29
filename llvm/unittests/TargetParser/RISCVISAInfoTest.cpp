//===-- unittests/RISCVISAInfoTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using ::testing::ElementsAre;

using namespace llvm;

bool operator==(const RISCVISAUtils::ExtensionVersion &A,
                const RISCVISAUtils::ExtensionVersion &B) {
  return A.Major == B.Major && A.Minor == B.Minor;
}

TEST(ParseNormalizedArchString, RejectsInvalidChars) {
  for (StringRef Input : {"RV32", "rV64", "rv32i2P0", "rv64i2p0_A2p0",
                          "rv32e2.0", "rva20u64+zbc"}) {
    EXPECT_EQ(
        toString(RISCVISAInfo::parseNormalizedArchString(Input).takeError()),
        "string may only contain [a-z0-9_]");
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
  for (StringRef Input : {"rv64e2p", "rv32i", "rv64ip1"}) {
    EXPECT_EQ(
        toString(RISCVISAInfo::parseNormalizedArchString(Input).takeError()),
        "extension lacks version in expected format");
  }

  for (StringRef Input : {"rv64i2p0_", "rv32i2p0__a2p0"}) {
    EXPECT_EQ(
        toString(RISCVISAInfo::parseNormalizedArchString(Input).takeError()),
        "extension name missing after separator '_'");
  }
}

TEST(ParseNormalizedArchString, RejectsOnlyVersion) {
  for (StringRef Input : {"rv64i2p0_1p0", "rv32i2p0_1p0"}) {
    EXPECT_EQ(
        toString(RISCVISAInfo::parseNormalizedArchString(Input).takeError()),
        "missing extension name");
  }
}

TEST(ParseNormalizedArchString, RejectsBadZ) {
  for (StringRef Input : {"rv64i2p0_z1p0", "rv32i2p0_z2a1p0"}) {
    EXPECT_EQ(
        toString(RISCVISAInfo::parseNormalizedArchString(Input).takeError()),
        "'z' must be followed by a letter");
  }
}

TEST(ParseNormalizedArchString, RejectsBadS) {
  for (StringRef Input : {"rv64i2p0_s1p0", "rv32i2p0_s2a1p0"}) {
    EXPECT_EQ(
        toString(RISCVISAInfo::parseNormalizedArchString(Input).takeError()),
        "'s' must be followed by a letter");
  }
}

TEST(ParseNormalizedArchString, RejectsBadX) {
  for (StringRef Input : {"rv64i2p0_x1p0", "rv32i2p0_x2a1p0"}) {
    EXPECT_EQ(
        toString(RISCVISAInfo::parseNormalizedArchString(Input).takeError()),
        "'x' must be followed by a letter");
  }
}

TEST(ParseNormalizedArchString, DuplicateExtension) {
  for (StringRef Input : {"rv64i2p0_a2p0_a1p0"}) {
    EXPECT_EQ(
        toString(RISCVISAInfo::parseNormalizedArchString(Input).takeError()),
        "duplicate extension 'a'");
  }
}

TEST(ParseNormalizedArchString, AcceptsValidBaseISAsAndSetsXLen) {
  auto MaybeRV32I = RISCVISAInfo::parseNormalizedArchString("rv32i2p0");
  ASSERT_THAT_EXPECTED(MaybeRV32I, Succeeded());
  RISCVISAInfo &InfoRV32I = **MaybeRV32I;
  EXPECT_EQ(InfoRV32I.getExtensions().size(), 1UL);
  EXPECT_TRUE(InfoRV32I.getExtensions().at("i") ==
              (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_EQ(InfoRV32I.getXLen(), 32U);

  auto MaybeRV32E = RISCVISAInfo::parseNormalizedArchString("rv32e2p0");
  ASSERT_THAT_EXPECTED(MaybeRV32E, Succeeded());
  RISCVISAInfo &InfoRV32E = **MaybeRV32E;
  EXPECT_EQ(InfoRV32E.getExtensions().size(), 1UL);
  EXPECT_TRUE(InfoRV32E.getExtensions().at("e") ==
              (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_EQ(InfoRV32E.getXLen(), 32U);

  auto MaybeRV64I = RISCVISAInfo::parseNormalizedArchString("rv64i2p0");
  ASSERT_THAT_EXPECTED(MaybeRV64I, Succeeded());
  RISCVISAInfo &InfoRV64I = **MaybeRV64I;
  EXPECT_EQ(InfoRV64I.getExtensions().size(), 1UL);
  EXPECT_TRUE(InfoRV64I.getExtensions().at("i") ==
              (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_EQ(InfoRV64I.getXLen(), 64U);

  auto MaybeRV64E = RISCVISAInfo::parseNormalizedArchString("rv64e2p0");
  ASSERT_THAT_EXPECTED(MaybeRV64E, Succeeded());
  RISCVISAInfo &InfoRV64E = **MaybeRV64E;
  EXPECT_EQ(InfoRV64E.getExtensions().size(), 1UL);
  EXPECT_TRUE(InfoRV64E.getExtensions().at("e") ==
              (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_EQ(InfoRV64E.getXLen(), 64U);
}

TEST(ParseNormalizedArchString, AcceptsArbitraryExtensionsAndVersions) {
  auto MaybeISAInfo = RISCVISAInfo::parseNormalizedArchString(
      "rv64i5p1_m3p2_zmadeup11p12_sfoo2p0_xbar3p0");
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  RISCVISAInfo &Info = **MaybeISAInfo;
  EXPECT_EQ(Info.getExtensions().size(), 5UL);
  EXPECT_TRUE(Info.getExtensions().at("i") ==
              (RISCVISAUtils::ExtensionVersion{5, 1}));
  EXPECT_TRUE(Info.getExtensions().at("m") ==
              (RISCVISAUtils::ExtensionVersion{3, 2}));
  EXPECT_TRUE(Info.getExtensions().at("zmadeup") ==
              (RISCVISAUtils::ExtensionVersion{11, 12}));
  EXPECT_TRUE(Info.getExtensions().at("sfoo") ==
              (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_TRUE(Info.getExtensions().at("xbar") ==
              (RISCVISAUtils::ExtensionVersion{3, 0}));
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
  EXPECT_EQ(Info.getMaxELenFp(), 64U);
}

TEST(ParseNormalizedArchString, AcceptsUnknownMultiletter) {
  auto MaybeISAInfo = RISCVISAInfo::parseNormalizedArchString(
      "rv64i2p0_f2p0_d2p0_zicsr2p0_ykk1p0");
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  RISCVISAInfo &Info = **MaybeISAInfo;
  EXPECT_EQ(Info.toString(), "rv64i2p0_f2p0_d2p0_zicsr2p0_ykk1p0");
}

TEST(ParseArchString, RejectsInvalidChars) {
  for (StringRef Input : {"RV32", "rV64", "rv32i2P0", "rv64i2p0_A2p0"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "string may only contain [a-z0-9_]");
  }
}

TEST(ParseArchString, RejectsInvalidBaseISA) {
  for (StringRef Input : {"rv32", "rv64", "rv65i"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "string must begin with rv32{i,e,g}, rv64{i,e,g}, or a supported "
              "profile name");
  }

  for (StringRef Input : {"rv32j", "rv32_i"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "first letter after 'rv32' should be 'e', 'i' or 'g'");
  }

  EXPECT_EQ(toString(RISCVISAInfo::parseArchString("rv64k", true).takeError()),
            "first letter after 'rv64' should be 'e', 'i' or 'g'");
}

TEST(ParseArchString, RejectsUnsupportedBaseISA) {
  for (StringRef Input : {"rv128i", "rv128g"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "string must begin with rv32{i,e,g}, rv64{i,e,g}, or a supported "
              "profile name");
  }
}

TEST(ParseArchString, AcceptsSupportedBaseISAsAndSetsXLenAndFLen) {
  auto MaybeRV32I = RISCVISAInfo::parseArchString("rv32i", true);
  ASSERT_THAT_EXPECTED(MaybeRV32I, Succeeded());
  RISCVISAInfo &InfoRV32I = **MaybeRV32I;
  const auto &ExtsRV32I = InfoRV32I.getExtensions();
  EXPECT_EQ(ExtsRV32I.size(), 1UL);
  EXPECT_TRUE(ExtsRV32I.at("i") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  EXPECT_EQ(InfoRV32I.getXLen(), 32U);
  EXPECT_EQ(InfoRV32I.getFLen(), 0U);
  EXPECT_EQ(InfoRV32I.getMinVLen(), 0U);
  EXPECT_EQ(InfoRV32I.getMaxELen(), 0U);
  EXPECT_EQ(InfoRV32I.getMaxELenFp(), 0U);

  auto MaybeRV32E = RISCVISAInfo::parseArchString("rv32e", true);
  ASSERT_THAT_EXPECTED(MaybeRV32E, Succeeded());
  RISCVISAInfo &InfoRV32E = **MaybeRV32E;
  const auto &ExtsRV32E = InfoRV32E.getExtensions();
  EXPECT_EQ(ExtsRV32E.size(), 1UL);
  EXPECT_TRUE(ExtsRV32E.at("e") == (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_EQ(InfoRV32E.getXLen(), 32U);
  EXPECT_EQ(InfoRV32E.getFLen(), 0U);
  EXPECT_EQ(InfoRV32E.getMinVLen(), 0U);
  EXPECT_EQ(InfoRV32E.getMaxELen(), 0U);
  EXPECT_EQ(InfoRV32E.getMaxELenFp(), 0U);

  auto MaybeRV32G = RISCVISAInfo::parseArchString("rv32g", true);
  ASSERT_THAT_EXPECTED(MaybeRV32G, Succeeded());
  RISCVISAInfo &InfoRV32G = **MaybeRV32G;
  const auto &ExtsRV32G = InfoRV32G.getExtensions();
  EXPECT_EQ(ExtsRV32G.size(), 7UL);
  EXPECT_TRUE(ExtsRV32G.at("i") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  EXPECT_TRUE(ExtsRV32G.at("m") == (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_TRUE(ExtsRV32G.at("a") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  EXPECT_TRUE(ExtsRV32G.at("f") == (RISCVISAUtils::ExtensionVersion{2, 2}));
  EXPECT_TRUE(ExtsRV32G.at("d") == (RISCVISAUtils::ExtensionVersion{2, 2}));
  EXPECT_TRUE(ExtsRV32G.at("zicsr") == (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_TRUE(ExtsRV32G.at("zifencei") ==
              (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_EQ(InfoRV32G.getXLen(), 32U);
  EXPECT_EQ(InfoRV32G.getFLen(), 64U);
  EXPECT_EQ(InfoRV32G.getMinVLen(), 0U);
  EXPECT_EQ(InfoRV32G.getMaxELen(), 0U);
  EXPECT_EQ(InfoRV32G.getMaxELenFp(), 0U);

  auto MaybeRV64I = RISCVISAInfo::parseArchString("rv64i", true);
  ASSERT_THAT_EXPECTED(MaybeRV64I, Succeeded());
  RISCVISAInfo &InfoRV64I = **MaybeRV64I;
  const auto &ExtsRV64I = InfoRV64I.getExtensions();
  EXPECT_EQ(ExtsRV64I.size(), 1UL);
  EXPECT_TRUE(ExtsRV64I.at("i") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  EXPECT_EQ(InfoRV64I.getXLen(), 64U);
  EXPECT_EQ(InfoRV64I.getFLen(), 0U);
  EXPECT_EQ(InfoRV64I.getMinVLen(), 0U);
  EXPECT_EQ(InfoRV64I.getMaxELen(), 0U);
  EXPECT_EQ(InfoRV64I.getMaxELenFp(), 0U);

  auto MaybeRV64E = RISCVISAInfo::parseArchString("rv64e", true);
  ASSERT_THAT_EXPECTED(MaybeRV64E, Succeeded());
  RISCVISAInfo &InfoRV64E = **MaybeRV64E;
  const auto &ExtsRV64E = InfoRV64E.getExtensions();
  EXPECT_EQ(ExtsRV64E.size(), 1UL);
  EXPECT_TRUE(ExtsRV64E.at("e") == (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_EQ(InfoRV64E.getXLen(), 64U);
  EXPECT_EQ(InfoRV64E.getFLen(), 0U);
  EXPECT_EQ(InfoRV64E.getMinVLen(), 0U);
  EXPECT_EQ(InfoRV64E.getMaxELen(), 0U);
  EXPECT_EQ(InfoRV64E.getMaxELenFp(), 0U);

  auto MaybeRV64G = RISCVISAInfo::parseArchString("rv64g", true);
  ASSERT_THAT_EXPECTED(MaybeRV64G, Succeeded());
  RISCVISAInfo &InfoRV64G = **MaybeRV64G;
  const auto &ExtsRV64G = InfoRV64G.getExtensions();
  EXPECT_EQ(ExtsRV64G.size(), 7UL);
  EXPECT_TRUE(ExtsRV64G.at("i") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  EXPECT_TRUE(ExtsRV64G.at("m") == (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_TRUE(ExtsRV64G.at("a") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  EXPECT_TRUE(ExtsRV64G.at("f") == (RISCVISAUtils::ExtensionVersion{2, 2}));
  EXPECT_TRUE(ExtsRV64G.at("d") == (RISCVISAUtils::ExtensionVersion{2, 2}));
  EXPECT_TRUE(ExtsRV64G.at("zicsr") == (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_TRUE(ExtsRV64G.at("zifencei") ==
              (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_EQ(InfoRV64G.getXLen(), 64U);
  EXPECT_EQ(InfoRV64G.getFLen(), 64U);
  EXPECT_EQ(InfoRV64G.getMinVLen(), 0U);
  EXPECT_EQ(InfoRV64G.getMaxELen(), 0U);
  EXPECT_EQ(InfoRV64G.getMaxELenFp(), 0U);

  auto MaybeRV64GCV = RISCVISAInfo::parseArchString("rv64gcv", true);
  ASSERT_THAT_EXPECTED(MaybeRV64GCV, Succeeded());
  RISCVISAInfo &InfoRV64GCV = **MaybeRV64GCV;
  const auto &ExtsRV64GCV = InfoRV64GCV.getExtensions();
  EXPECT_EQ(ExtsRV64GCV.size(), 17UL);
  EXPECT_TRUE(ExtsRV64GCV.at("i") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  EXPECT_TRUE(ExtsRV64GCV.at("m") == (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("a") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  EXPECT_TRUE(ExtsRV64GCV.at("f") == (RISCVISAUtils::ExtensionVersion{2, 2}));
  EXPECT_TRUE(ExtsRV64GCV.at("d") == (RISCVISAUtils::ExtensionVersion{2, 2}));
  EXPECT_TRUE(ExtsRV64GCV.at("c") == (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("zicsr") == (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("zifencei") ==
              (RISCVISAUtils::ExtensionVersion{2, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("v") == (RISCVISAUtils::ExtensionVersion{1, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("zve32x") == (RISCVISAUtils::ExtensionVersion{1, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("zve32f") == (RISCVISAUtils::ExtensionVersion{1, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("zve64x") == (RISCVISAUtils::ExtensionVersion{1, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("zve64f") == (RISCVISAUtils::ExtensionVersion{1, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("zve64d") == (RISCVISAUtils::ExtensionVersion{1, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("zvl32b") == (RISCVISAUtils::ExtensionVersion{1, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("zvl64b") == (RISCVISAUtils::ExtensionVersion{1, 0}));
  EXPECT_TRUE(ExtsRV64GCV.at("zvl128b") == (RISCVISAUtils::ExtensionVersion{1, 0}));
  EXPECT_EQ(InfoRV64GCV.getXLen(), 64U);
  EXPECT_EQ(InfoRV64GCV.getFLen(), 64U);
  EXPECT_EQ(InfoRV64GCV.getMinVLen(), 128U);
  EXPECT_EQ(InfoRV64GCV.getMaxELen(), 64U);
  EXPECT_EQ(InfoRV64GCV.getMaxELenFp(), 64U);
}

TEST(ParseArchString, RejectsUnrecognizedExtensionNamesByDefault) {
  EXPECT_EQ(toString(RISCVISAInfo::parseArchString("rv64ib", true).takeError()),
            "unsupported standard user-level extension 'b'");
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv32i_zmadeup", true).takeError()),
      "unsupported standard user-level extension 'zmadeup'");
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv64g_smadeup", true).takeError()),
      "unsupported standard supervisor-level extension 'smadeup'");
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv64g_xmadeup", true).takeError()),
      "unsupported non-standard user-level extension 'xmadeup'");
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv64ib1p0", true).takeError()),
      "unsupported standard user-level extension 'b'");
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv32i_zmadeup1p0", true).takeError()),
      "unsupported standard user-level extension 'zmadeup'");
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv64g_smadeup1p0", true).takeError()),
      "unsupported standard supervisor-level extension 'smadeup'");
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv64g_xmadeup1p0", true).takeError()),
      "unsupported non-standard user-level extension 'xmadeup'");
}

TEST(ParseArchString, IgnoresUnrecognizedExtensionNamesWithIgnoreUnknown) {
  for (StringRef Input : {"rv32ib", "rv32i_zmadeup",
                          "rv64i_smadeup", "rv64i_xmadeup"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true, false, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    RISCVISAInfo &Info = **MaybeISAInfo;
    const auto &Exts = Info.getExtensions();
    EXPECT_EQ(Exts.size(), 1UL);
    EXPECT_TRUE(Exts.at("i") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  }

  // Checks that supported extensions aren't incorrectly ignored when a
  // version is present (an early version of the patch had this mistake).
  auto MaybeISAInfo =
      RISCVISAInfo::parseArchString("rv32i_zbc1p0_xmadeup", true, false, true);
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  const auto &Exts = (*MaybeISAInfo)->getExtensions();
  EXPECT_TRUE(Exts.at("zbc") == (RISCVISAUtils::ExtensionVersion{1, 0}));
}

TEST(ParseArchString, AcceptsVersionInLongOrShortForm) {
  for (StringRef Input : {"rv64i2p1"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    const auto &Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_TRUE(Exts.at("i") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  }
  for (StringRef Input : {"rv32i_zfinx1", "rv32i_zfinx1p0"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    const auto &Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_TRUE(Exts.at("zfinx") == (RISCVISAUtils::ExtensionVersion{1, 0}));
  }
}

TEST(ParseArchString, RejectsUnrecognizedExtensionVersionsByDefault) {
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv64i2p", true).takeError()),
      "minor version number missing after 'p' for extension 'i'");
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv64i1p0", true).takeError()),
      "unsupported version number 1.0 for extension 'i'");
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv64i9p9", true).takeError()),
      "unsupported version number 9.9 for extension 'i'");
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv32im0p1", true).takeError()),
      "unsupported version number 0.1 for extension 'm'");
  EXPECT_EQ(toString(RISCVISAInfo::parseArchString("rv32izifencei10p10", true)
                         .takeError()),
            "unsupported version number 10.10 for extension 'zifencei'");
}

TEST(ParseArchString,
     UsesDefaultVersionForUnrecognisedBaseISAVersionWithIgnoreUnknown) {
  for (StringRef Input : {"rv32i0p1", "rv32i99p99", "rv64i0p1", "rv64i99p99"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true, false, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    const auto &Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_EQ(Exts.size(), 1UL);
    EXPECT_TRUE(Exts.at("i") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  }
  for (StringRef Input : {"rv32e0p1", "rv32e99p99", "rv64e0p1", "rv64e99p99"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true, false, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    const auto &Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_EQ(Exts.size(), 1UL);
    EXPECT_TRUE(Exts.at("e") == (RISCVISAUtils::ExtensionVersion{2, 0}));
  }
}

TEST(ParseArchString,
     IgnoresExtensionsWithUnrecognizedVersionsWithIgnoreUnknown) {
  for (StringRef Input : {"rv32im1p1", "rv64i_svnapot10p9", "rv32i_zicsr0p5"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true, false, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    const auto &Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_EQ(Exts.size(), 1UL);
    EXPECT_TRUE(Exts.at("i") == (RISCVISAUtils::ExtensionVersion{2, 1}));
  }
}

TEST(ParseArchString, AcceptsUnderscoreSplittingExtensions) {
  for (StringRef Input : {"rv32imafdczifencei", "rv32i_m_a_f_d_c_zifencei"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    const auto &Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_EQ(Exts.size(), 8UL);
    EXPECT_EQ(Exts.count("i"), 1U);
    EXPECT_EQ(Exts.count("m"), 1U);
    EXPECT_EQ(Exts.count("a"), 1U);
    EXPECT_EQ(Exts.count("f"), 1U);
    EXPECT_EQ(Exts.count("d"), 1U);
    EXPECT_EQ(Exts.count("c"), 1U);
    EXPECT_EQ(Exts.count("zicsr"), 1U);
    EXPECT_EQ(Exts.count("zifencei"), 1U);
  }
}

TEST(ParseArchString, AcceptsRelaxSingleLetterExtensions) {
  for (StringRef Input :
       {"rv32imfad", "rv32im_fa_d", "rv32im2p0fad", "rv32i2p1m2p0fad"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    const auto &Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_EQ(Exts.size(), 6UL);
    EXPECT_EQ(Exts.count("i"), 1U);
    EXPECT_EQ(Exts.count("m"), 1U);
    EXPECT_EQ(Exts.count("f"), 1U);
    EXPECT_EQ(Exts.count("a"), 1U);
    EXPECT_EQ(Exts.count("d"), 1U);
    EXPECT_EQ(Exts.count("zicsr"), 1U);
  }
}

TEST(ParseArchString, AcceptsRelaxMixedLetterExtensions) {
  for (StringRef Input :
       {"rv32i_zihintntl_m_a_f_d_svinval", "rv32izihintntl_mafdsvinval",
        "rv32i_zihintntl_mafd_svinval"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    const auto &Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_EQ(Exts.size(), 8UL);
    EXPECT_EQ(Exts.count("i"), 1U);
    EXPECT_EQ(Exts.count("m"), 1U);
    EXPECT_EQ(Exts.count("a"), 1U);
    EXPECT_EQ(Exts.count("f"), 1U);
    EXPECT_EQ(Exts.count("d"), 1U);
    EXPECT_EQ(Exts.count("zihintntl"), 1U);
    EXPECT_EQ(Exts.count("svinval"), 1U);
    EXPECT_EQ(Exts.count("zicsr"), 1U);
  }
}

TEST(ParseArchString, AcceptsAmbiguousFromRelaxExtensions) {
  for (StringRef Input : {"rv32i_zba_m", "rv32izba_m", "rv32izba1p0_m2p0"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    const auto &Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_EQ(Exts.size(), 3UL);
    EXPECT_EQ(Exts.count("i"), 1U);
    EXPECT_EQ(Exts.count("zba"), 1U);
    EXPECT_EQ(Exts.count("m"), 1U);
  }
  for (StringRef Input :
       {"rv32ia_zba_m", "rv32iazba_m", "rv32ia2p1zba1p0_m2p0"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    const auto &Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_EQ(Exts.size(), 4UL);
    EXPECT_EQ(Exts.count("i"), 1U);
    EXPECT_EQ(Exts.count("zba"), 1U);
    EXPECT_EQ(Exts.count("m"), 1U);
    EXPECT_EQ(Exts.count("a"), 1U);
  }
}

TEST(ParseArchString, RejectsRelaxExtensionsNotStartWithEorIorG) {
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv32zba_im", true).takeError()),
      "first letter after 'rv32' should be 'e', 'i' or 'g'");
}

TEST(ParseArchString,
     RejectsMultiLetterExtensionFollowBySingleLetterExtensions) {
  for (StringRef Input : {"rv32izbam", "rv32i_zbam"})
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "unsupported standard user-level extension 'zbam'");
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv32izbai_m", true).takeError()),
      "unsupported standard user-level extension 'zbai'");
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv32izbaim", true).takeError()),
      "unsupported standard user-level extension 'zbaim'");
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv32i_zba1p0m", true).takeError()),
      "unsupported standard user-level extension 'zba1p0m'");
}

TEST(ParseArchString, RejectsDoubleOrTrailingUnderscore) {
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv64i__m", true).takeError()),
      "extension name missing after separator '_'");

  for (StringRef Input :
       {"rv32ezicsr__zifencei", "rv32i_", "rv32izicsr_", "rv64im_"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "extension name missing after separator '_'");
  }
}

TEST(ParseArchString, RejectsDuplicateExtensionNames) {
  EXPECT_EQ(toString(RISCVISAInfo::parseArchString("rv64ii", true).takeError()),
            "invalid standard user-level extension 'i'");
  EXPECT_EQ(toString(RISCVISAInfo::parseArchString("rv32ee", true).takeError()),
            "invalid standard user-level extension 'e'");
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv64imm", true).takeError()),
      "duplicated standard user-level extension 'm'");
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv32i_zicsr_zicsr", true).takeError()),
      "duplicated standard user-level extension 'zicsr'");
}

TEST(ParseArchString,
     RejectsExperimentalExtensionsIfNotEnableExperimentalExtension) {
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv64iztso", false).takeError()),
      "requires '-menable-experimental-extensions' for experimental extension "
      "'ztso'");
}

TEST(ParseArchString,
     AcceptsExperimentalExtensionsIfEnableExperimentalExtension) {
  // Note: If ztso becomes none-experimental, this test will need
  // updating (and unfortunately, it will still pass). The failure of
  // RejectsExperimentalExtensionsIfNotEnableExperimentalExtension will
  // hopefully serve as a reminder to update.
  auto MaybeISAInfo = RISCVISAInfo::parseArchString("rv64iztso", true, false);
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  const auto &Exts = (*MaybeISAInfo)->getExtensions();
  EXPECT_EQ(Exts.size(), 2UL);
  EXPECT_EQ(Exts.count("ztso"), 1U);
  auto MaybeISAInfo2 = RISCVISAInfo::parseArchString("rv64iztso0p1", true);
  ASSERT_THAT_EXPECTED(MaybeISAInfo2, Succeeded());
  const auto &Exts2 = (*MaybeISAInfo2)->getExtensions();
  EXPECT_EQ(Exts2.size(), 2UL);
  EXPECT_EQ(Exts2.count("ztso"), 1U);
}

TEST(ParseArchString,
     RequiresExplicitVersionNumberForExperimentalExtensionByDefault) {
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv64iztso", true).takeError()),
      "experimental extension requires explicit version number `ztso`");
}

TEST(ParseArchString,
     AcceptsUnrecognizedVersionIfNotExperimentalExtensionVersionCheck) {
  auto MaybeISAInfo =
      RISCVISAInfo::parseArchString("rv64iztso9p9", true, false);
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  const auto &Exts = (*MaybeISAInfo)->getExtensions();
  EXPECT_EQ(Exts.size(), 2UL);
  EXPECT_TRUE(Exts.at("ztso") == (RISCVISAUtils::ExtensionVersion{9, 9}));
}

TEST(ParseArchString, RejectsUnrecognizedVersionForExperimentalExtension) {
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv64iztso9p9", true).takeError()),
      "unsupported version number 9.9 for experimental extension 'ztso' "
      "(this compiler supports 0.1)");
}

TEST(ParseArchString, RejectsExtensionVersionForG) {
  for (StringRef Input : {"rv32g1c", "rv64g2p0"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "version not supported for 'g'");
  }
}

TEST(ParseArchString, AddsImpliedExtensions) {
  // Does not attempt to exhaustively test all implications.
  auto MaybeRV64ID = RISCVISAInfo::parseArchString("rv64id", true);
  ASSERT_THAT_EXPECTED(MaybeRV64ID, Succeeded());
  const auto &ExtsRV64ID = (*MaybeRV64ID)->getExtensions();
  EXPECT_EQ(ExtsRV64ID.size(), 4UL);
  EXPECT_EQ(ExtsRV64ID.count("i"), 1U);
  EXPECT_EQ(ExtsRV64ID.count("f"), 1U);
  EXPECT_EQ(ExtsRV64ID.count("d"), 1U);
  EXPECT_EQ(ExtsRV64ID.count("zicsr"), 1U);

  auto MaybeRV32IZKN = RISCVISAInfo::parseArchString("rv64izkn", true);
  ASSERT_THAT_EXPECTED(MaybeRV32IZKN, Succeeded());
  const auto &ExtsRV32IZKN = (*MaybeRV32IZKN)->getExtensions();
  EXPECT_EQ(ExtsRV32IZKN.size(), 8UL);
  EXPECT_EQ(ExtsRV32IZKN.count("i"), 1U);
  EXPECT_EQ(ExtsRV32IZKN.count("zbkb"), 1U);
  EXPECT_EQ(ExtsRV32IZKN.count("zbkc"), 1U);
  EXPECT_EQ(ExtsRV32IZKN.count("zbkx"), 1U);
  EXPECT_EQ(ExtsRV32IZKN.count("zkne"), 1U);
  EXPECT_EQ(ExtsRV32IZKN.count("zknd"), 1U);
  EXPECT_EQ(ExtsRV32IZKN.count("zknh"), 1U);
  EXPECT_EQ(ExtsRV32IZKN.count("zkn"), 1U);
}

TEST(ParseArchString, RejectsConflictingExtensions) {
  for (StringRef Input : {"rv32ifzfinx", "rv64gzdinx"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "'f' and 'zfinx' extensions are incompatible");
  }

  for (StringRef Input : {"rv32idc_zcmp1p0", "rv64idc_zcmp1p0"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "'zcmp' extension is incompatible with 'c' extension when 'd' "
              "extension is enabled");
  }

  for (StringRef Input : {"rv32id_zcd1p0_zcmp1p0", "rv64id_zcd1p0_zcmp1p0"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "'zcmp' extension is incompatible with 'zcd' extension when 'd' "
              "extension is enabled");
  }

  for (StringRef Input : {"rv32idc_zcmt1p0", "rv64idc_zcmt1p0"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "'zcmt' extension is incompatible with 'c' extension when 'd' "
              "extension is enabled");
  }

  for (StringRef Input : {"rv32id_zcd1p0_zcmt1p0", "rv64id_zcd1p0_zcmt1p0"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "'zcmt' extension is incompatible with 'zcd' extension when 'd' "
              "extension is enabled");
  }

  for (StringRef Input : {"rv64if_zcf"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "'zcf' is only supported for 'rv32'");
  }
}

TEST(ParseArchString, RejectsUnrecognizedProfileNames) {
  for (StringRef Input : {"rvi23u99", "rvz23u64", "rva99u32"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "string must begin with rv32{i,e,g}, rv64{i,e,g}, or a supported "
              "profile name");
  }
}

TEST(ParseArchString, RejectsProfilesWithUnseparatedExtraExtensions) {
  for (StringRef Input : {"rvi20u32m", "rvi20u64c"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "additional extensions must be after separator '_'");
  }
}

TEST(ParseArchString, AcceptsBareProfileNames) {
  auto MaybeRVA20U64 = RISCVISAInfo::parseArchString("rva20u64", true);
  ASSERT_THAT_EXPECTED(MaybeRVA20U64, Succeeded());
  const auto &Exts = (*MaybeRVA20U64)->getExtensions();
  EXPECT_EQ(Exts.size(), 13UL);
  EXPECT_EQ(Exts.count("i"), 1U);
  EXPECT_EQ(Exts.count("m"), 1U);
  EXPECT_EQ(Exts.count("f"), 1U);
  EXPECT_EQ(Exts.count("a"), 1U);
  EXPECT_EQ(Exts.count("d"), 1U);
  EXPECT_EQ(Exts.count("c"), 1U);
  EXPECT_EQ(Exts.count("za128rs"), 1U);
  EXPECT_EQ(Exts.count("zicntr"), 1U);
  EXPECT_EQ(Exts.count("ziccif"), 1U);
  EXPECT_EQ(Exts.count("zicsr"), 1U);
  EXPECT_EQ(Exts.count("ziccrse"), 1U);
  EXPECT_EQ(Exts.count("ziccamoa"), 1U);
  EXPECT_EQ(Exts.count("zicclsm"), 1U);

  auto MaybeRVA23U64 = RISCVISAInfo::parseArchString("rva23u64", true);
  ASSERT_THAT_EXPECTED(MaybeRVA23U64, Succeeded());
  EXPECT_GT((*MaybeRVA23U64)->getExtensions().size(), 13UL);
}

TEST(ParseArchSTring, AcceptsProfileNamesWithSeparatedAdditionalExtensions) {
  auto MaybeRVI20U64 = RISCVISAInfo::parseArchString("rvi20u64_m_zba", true);
  ASSERT_THAT_EXPECTED(MaybeRVI20U64, Succeeded());
  const auto &Exts = (*MaybeRVI20U64)->getExtensions();
  EXPECT_EQ(Exts.size(), 3UL);
  EXPECT_EQ(Exts.count("i"), 1U);
  EXPECT_EQ(Exts.count("m"), 1U);
  EXPECT_EQ(Exts.count("zba"), 1U);
}

TEST(ParseArchString,
     RejectsProfilesWithAdditionalExtensionsGivenAlreadyInProfile) {
  // This test was added to document the current behaviour. Discussion isn't
  // believed to have taken place about if this is desirable or not.
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rva20u64_zicntr", true).takeError()),
      "duplicated standard user-level extension 'zicntr'");
}

TEST(ParseArchString,
     RejectsExperimentalProfilesIfEnableExperimentalExtensionsNotSet) {
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rva23u64", false).takeError()),
      "requires '-menable-experimental-extensions' for profile 'rva23u64'");
}

TEST(ToFeatures, IIsDroppedAndExperimentalExtensionsArePrefixed) {
  auto MaybeISAInfo1 =
      RISCVISAInfo::parseArchString("rv64im_ztso", true, false);
  ASSERT_THAT_EXPECTED(MaybeISAInfo1, Succeeded());
  EXPECT_THAT((*MaybeISAInfo1)->toFeatures(),
              ElementsAre("+m", "+experimental-ztso"));

  auto MaybeISAInfo2 =
      RISCVISAInfo::parseArchString("rv32e_ztso_xventanacondops", true, false);
  ASSERT_THAT_EXPECTED(MaybeISAInfo2, Succeeded());
  EXPECT_THAT((*MaybeISAInfo2)->toFeatures(),
              ElementsAre("+e", "+experimental-ztso", "+xventanacondops"));
}

TEST(ToFeatures, UnsupportedExtensionsAreDropped) {
  auto MaybeISAInfo =
      RISCVISAInfo::parseNormalizedArchString("rv64i2p0_m2p0_xmadeup1p0");
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  EXPECT_THAT((*MaybeISAInfo)->toFeatures(), ElementsAre("+m"));
}

TEST(ToFeatures, UnsupportedExtensionsAreKeptIfIgnoreUnknownIsFalse) {
  auto MaybeISAInfo =
      RISCVISAInfo::parseNormalizedArchString("rv64i2p0_m2p0_xmadeup1p0");
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  EXPECT_THAT((*MaybeISAInfo)->toFeatures(false, false),
              ElementsAre("+m", "+xmadeup"));
}

TEST(ToFeatures, AddAllExtensionsAddsNegativeExtensions) {
  auto MaybeISAInfo = RISCVISAInfo::parseNormalizedArchString("rv64i2p0_m2p0");
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());

  auto Features = (*MaybeISAInfo)->toFeatures(true);
  EXPECT_GT(Features.size(), 1UL);
  EXPECT_EQ(Features.front(), "+m");
  // Every feature after should be a negative feature
  for (auto &NegativeExt : llvm::drop_begin(Features))
    EXPECT_TRUE(NegativeExt.substr(0, 1) == "-");
}

TEST(OrderedExtensionMap, ExtensionsAreCorrectlyOrdered) {
  RISCVISAUtils::OrderedExtensionMap Exts;
  for (auto ExtName : {"y", "l", "m", "c", "i", "xfoo", "xbar", "sfoo", "sbar",
                       "zmfoo", "zzfoo", "zfinx", "zicsr"})
    Exts[ExtName] = {1, 0};

  std::vector<std::string> ExtNames;
  for (const auto &Ext : Exts)
    ExtNames.push_back(Ext.first);

  // FIXME: 'l' and 'y' should be ordered after 'i', 'm', 'c'.
  EXPECT_THAT(ExtNames,
              ElementsAre("i", "m", "l", "c", "y", "zicsr", "zmfoo", "zfinx",
                           "zzfoo", "sbar", "sfoo", "xbar", "xfoo"));
}

TEST(ParseArchString, ZceImplication) {
  auto MaybeRV32IZce = RISCVISAInfo::parseArchString("rv32izce", true);
  ASSERT_THAT_EXPECTED(MaybeRV32IZce, Succeeded());
  const auto &ExtsRV32IZce = (*MaybeRV32IZce)->getExtensions();
  EXPECT_EQ(ExtsRV32IZce.size(), 7UL);
  EXPECT_EQ(ExtsRV32IZce.count("i"), 1U);
  EXPECT_EQ(ExtsRV32IZce.count("zicsr"), 1U);
  EXPECT_EQ(ExtsRV32IZce.count("zca"), 1U);
  EXPECT_EQ(ExtsRV32IZce.count("zcb"), 1U);
  EXPECT_EQ(ExtsRV32IZce.count("zce"), 1U);
  EXPECT_EQ(ExtsRV32IZce.count("zcmp"), 1U);
  EXPECT_EQ(ExtsRV32IZce.count("zcmt"), 1U);

  auto MaybeRV32IFZce = RISCVISAInfo::parseArchString("rv32ifzce", true);
  ASSERT_THAT_EXPECTED(MaybeRV32IFZce, Succeeded());
  const auto &ExtsRV32IFZce = (*MaybeRV32IFZce)->getExtensions();
  EXPECT_EQ(ExtsRV32IFZce.size(), 9UL);
  EXPECT_EQ(ExtsRV32IFZce.count("i"), 1U);
  EXPECT_EQ(ExtsRV32IFZce.count("zicsr"), 1U);
  EXPECT_EQ(ExtsRV32IFZce.count("f"), 1U);
  EXPECT_EQ(ExtsRV32IFZce.count("zca"), 1U);
  EXPECT_EQ(ExtsRV32IFZce.count("zcb"), 1U);
  EXPECT_EQ(ExtsRV32IFZce.count("zce"), 1U);
  EXPECT_EQ(ExtsRV32IFZce.count("zcf"), 1U);
  EXPECT_EQ(ExtsRV32IFZce.count("zcmp"), 1U);
  EXPECT_EQ(ExtsRV32IFZce.count("zcmt"), 1U);

  auto MaybeRV32IDZce = RISCVISAInfo::parseArchString("rv32idzce", true);
  ASSERT_THAT_EXPECTED(MaybeRV32IDZce, Succeeded());
  const auto &ExtsRV32IDZce = (*MaybeRV32IDZce)->getExtensions();
  EXPECT_EQ(ExtsRV32IDZce.size(), 10UL);
  EXPECT_EQ(ExtsRV32IDZce.count("i"), 1U);
  EXPECT_EQ(ExtsRV32IDZce.count("zicsr"), 1U);
  EXPECT_EQ(ExtsRV32IDZce.count("f"), 1U);
  EXPECT_EQ(ExtsRV32IDZce.count("d"), 1U);
  EXPECT_EQ(ExtsRV32IDZce.count("zca"), 1U);
  EXPECT_EQ(ExtsRV32IDZce.count("zcb"), 1U);
  EXPECT_EQ(ExtsRV32IDZce.count("zce"), 1U);
  EXPECT_EQ(ExtsRV32IDZce.count("zcf"), 1U);
  EXPECT_EQ(ExtsRV32IDZce.count("zcmp"), 1U);
  EXPECT_EQ(ExtsRV32IDZce.count("zcmt"), 1U);

  auto MaybeRV64IZce = RISCVISAInfo::parseArchString("rv64izce", true);
  ASSERT_THAT_EXPECTED(MaybeRV64IZce, Succeeded());
  const auto &ExtsRV64IZce = (*MaybeRV64IZce)->getExtensions();
  EXPECT_EQ(ExtsRV64IZce.size(), 7UL);
  EXPECT_EQ(ExtsRV64IZce.count("i"), 1U);
  EXPECT_EQ(ExtsRV64IZce.count("zicsr"), 1U);
  EXPECT_EQ(ExtsRV64IZce.count("zca"), 1U);
  EXPECT_EQ(ExtsRV64IZce.count("zcb"), 1U);
  EXPECT_EQ(ExtsRV64IZce.count("zce"), 1U);
  EXPECT_EQ(ExtsRV64IZce.count("zcmp"), 1U);
  EXPECT_EQ(ExtsRV64IZce.count("zcmt"), 1U);

  auto MaybeRV64IFZce = RISCVISAInfo::parseArchString("rv64ifzce", true);
  ASSERT_THAT_EXPECTED(MaybeRV64IFZce, Succeeded());
  const auto &ExtsRV64IFZce = (*MaybeRV64IFZce)->getExtensions();
  EXPECT_EQ(ExtsRV64IFZce.size(), 8UL);
  EXPECT_EQ(ExtsRV64IFZce.count("i"), 1U);
  EXPECT_EQ(ExtsRV64IFZce.count("zicsr"), 1U);
  EXPECT_EQ(ExtsRV64IFZce.count("f"), 1U);
  EXPECT_EQ(ExtsRV64IFZce.count("zca"), 1U);
  EXPECT_EQ(ExtsRV64IFZce.count("zcb"), 1U);
  EXPECT_EQ(ExtsRV64IFZce.count("zce"), 1U);
  EXPECT_EQ(ExtsRV64IFZce.count("zcmp"), 1U);
  EXPECT_EQ(ExtsRV64IFZce.count("zcmt"), 1U);

  EXPECT_EQ(ExtsRV64IFZce.count("zca"), 1U);
  EXPECT_EQ(ExtsRV64IFZce.count("zcb"), 1U);
  EXPECT_EQ(ExtsRV64IFZce.count("zce"), 1U);
  EXPECT_EQ(ExtsRV64IFZce.count("zcmp"), 1U);
  EXPECT_EQ(ExtsRV64IFZce.count("zcmt"), 1U);

  auto MaybeRV64IDZce = RISCVISAInfo::parseArchString("rv64idzce", true);
  ASSERT_THAT_EXPECTED(MaybeRV64IDZce, Succeeded());
  const auto &ExtsRV64IDZce = (*MaybeRV64IDZce)->getExtensions();
  EXPECT_EQ(ExtsRV64IDZce.size(), 9UL);
  EXPECT_EQ(ExtsRV64IDZce.count("i"), 1U);
  EXPECT_EQ(ExtsRV64IDZce.count("zicsr"), 1U);
  EXPECT_EQ(ExtsRV64IDZce.count("f"), 1U);
  EXPECT_EQ(ExtsRV64IDZce.count("d"), 1U);
  EXPECT_EQ(ExtsRV64IDZce.count("zca"), 1U);
  EXPECT_EQ(ExtsRV64IDZce.count("zcb"), 1U);
  EXPECT_EQ(ExtsRV64IDZce.count("zce"), 1U);
  EXPECT_EQ(ExtsRV64IDZce.count("zcmp"), 1U);
  EXPECT_EQ(ExtsRV64IDZce.count("zcmt"), 1U);
}

TEST(isSupportedExtensionWithVersion, AcceptsSingleExtensionWithVersion) {
  EXPECT_TRUE(RISCVISAInfo::isSupportedExtensionWithVersion("zbb1p0"));
  EXPECT_FALSE(RISCVISAInfo::isSupportedExtensionWithVersion("zbb"));
  EXPECT_FALSE(RISCVISAInfo::isSupportedExtensionWithVersion("zfoo1p0"));
  EXPECT_FALSE(RISCVISAInfo::isSupportedExtensionWithVersion("zfoo"));
  EXPECT_FALSE(RISCVISAInfo::isSupportedExtensionWithVersion(""));
  EXPECT_FALSE(RISCVISAInfo::isSupportedExtensionWithVersion("c2p0zbb1p0"));
}

TEST(getTargetFeatureForExtension, RetrieveTargetFeatureFromOneExt) {
  EXPECT_EQ(RISCVISAInfo::getTargetFeatureForExtension("zbb"), "zbb");
  EXPECT_EQ(RISCVISAInfo::getTargetFeatureForExtension("ztso0p1"),
            "experimental-ztso");
  EXPECT_EQ(RISCVISAInfo::getTargetFeatureForExtension("ztso"),
            "experimental-ztso");
  EXPECT_EQ(RISCVISAInfo::getTargetFeatureForExtension("zihintntl1234p4321"),
            "");
  EXPECT_EQ(RISCVISAInfo::getTargetFeatureForExtension("zfoo"), "");
  EXPECT_EQ(RISCVISAInfo::getTargetFeatureForExtension(""), "");
  EXPECT_EQ(RISCVISAInfo::getTargetFeatureForExtension("zbbzihintntl"), "");
}

TEST(RiscvExtensionsHelp, CheckExtensions) {
  // clang-format off
  std::string ExpectedOutput =
R"(All available -march extensions for RISC-V

    Name                 Version   Description
    i                    2.1       This is a long dummy description
    e                    2.0
    m                    2.0
    a                    2.1
    f                    2.2
    d                    2.2
    c                    2.0
    v                    1.0
    h                    1.0
    zic64b               1.0
    zicbom               1.0
    zicbop               1.0
    zicboz               1.0
    ziccamoa             1.0
    ziccif               1.0
    zicclsm              1.0
    ziccrse              1.0
    zicntr               2.0
    zicond               1.0
    zicsr                2.0
    zifencei             2.0
    zihintntl            1.0
    zihintpause          2.0
    zihpm                2.0
    zimop                1.0
    zmmul                1.0
    za128rs              1.0
    za64rs               1.0
    zaamo                1.0
    zacas                1.0
    zalrsc               1.0
    zama16b              1.0
    zawrs                1.0
    zfa                  1.0
    zfh                  1.0
    zfhmin               1.0
    zfinx                1.0
    zdinx                1.0
    zca                  1.0
    zcb                  1.0
    zcd                  1.0
    zce                  1.0
    zcf                  1.0
    zcmop                1.0
    zcmp                 1.0
    zcmt                 1.0
    zba                  1.0
    zbb                  1.0
    zbc                  1.0
    zbkb                 1.0
    zbkc                 1.0
    zbkx                 1.0
    zbs                  1.0
    zk                   1.0
    zkn                  1.0
    zknd                 1.0
    zkne                 1.0
    zknh                 1.0
    zkr                  1.0
    zks                  1.0
    zksed                1.0
    zksh                 1.0
    zkt                  1.0
    zvbb                 1.0
    zvbc                 1.0
    zve32f               1.0
    zve32x               1.0
    zve64d               1.0
    zve64f               1.0
    zve64x               1.0
    zvfh                 1.0
    zvfhmin              1.0
    zvkb                 1.0
    zvkg                 1.0
    zvkn                 1.0
    zvknc                1.0
    zvkned               1.0
    zvkng                1.0
    zvknha               1.0
    zvknhb               1.0
    zvks                 1.0
    zvksc                1.0
    zvksed               1.0
    zvksg                1.0
    zvksh                1.0
    zvkt                 1.0
    zvl1024b             1.0
    zvl128b              1.0
    zvl16384b            1.0
    zvl2048b             1.0
    zvl256b              1.0
    zvl32768b            1.0
    zvl32b               1.0
    zvl4096b             1.0
    zvl512b              1.0
    zvl64b               1.0
    zvl65536b            1.0
    zvl8192b             1.0
    zhinx                1.0
    zhinxmin             1.0
    shcounterenw         1.0
    shgatpa              1.0
    shtvala              1.0
    shvsatpa             1.0
    shvstvala            1.0
    shvstvecd            1.0
    smaia                1.0
    smepmp               1.0
    smstateen            1.0
    ssaia                1.0
    ssccptr              1.0
    sscofpmf             1.0
    sscounterenw         1.0
    ssstateen            1.0
    ssstrict             1.0
    sstc                 1.0
    sstvala              1.0
    sstvecd              1.0
    ssu64xl              1.0
    svade                1.0
    svadu                1.0
    svbare               1.0
    svinval              1.0
    svnapot              1.0
    svpbmt               1.0
    xcvalu               1.0
    xcvbi                1.0
    xcvbitmanip          1.0
    xcvelw               1.0
    xcvmac               1.0
    xcvmem               1.0
    xcvsimd              1.0
    xsfcease             1.0
    xsfvcp               1.0
    xsfvfnrclipxfqf      1.0
    xsfvfwmaccqqq        1.0
    xsfvqmaccdod         1.0
    xsfvqmaccqoq         1.0
    xsifivecdiscarddlone 1.0
    xsifivecflushdlone   1.0
    xtheadba             1.0
    xtheadbb             1.0
    xtheadbs             1.0
    xtheadcmo            1.0
    xtheadcondmov        1.0
    xtheadfmemidx        1.0
    xtheadmac            1.0
    xtheadmemidx         1.0
    xtheadmempair        1.0
    xtheadsync           1.0
    xtheadvdot           1.0
    xventanacondops      1.0

Experimental extensions
    zicfilp              0.4       This is a long dummy description
    zicfiss              0.4
    zabha                1.0
    zalasr               0.1
    zfbfmin              1.0
    ztso                 0.1
    zvfbfmin             1.0
    zvfbfwma             1.0
    smmpm                0.8
    smnpm                0.8
    ssnpm                0.8
    sspm                 0.8
    ssqosid              1.0
    supm                 0.8

Supported Profiles
    rva20s64
    rva20u64
    rva22s64
    rva22u64
    rvi20u32
    rvi20u64

Experimental Profiles
    rva23s64
    rva23u64
    rvb23s64
    rvb23u64
    rvm23u32

Use -march to specify the target's extension.
For example, clang -march=rv32i_v1p0)";
  // clang-format on

  StringMap<StringRef> DummyMap;
  DummyMap["i"] = "This is a long dummy description";
  DummyMap["experimental-zicfilp"] = "This is a long dummy description";

  outs().flush();
  testing::internal::CaptureStdout();
  riscvExtensionsHelp(DummyMap);
  outs().flush();

  std::string CapturedOutput = testing::internal::GetCapturedStdout();
  EXPECT_TRUE([](std::string &Captured, std::string &Expected) {
                return Captured.find(Expected) != std::string::npos;
              }(CapturedOutput, ExpectedOutput));
}
