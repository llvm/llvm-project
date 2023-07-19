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
  return A.MajorVersion == B.MajorVersion && A.MinorVersion == B.MinorVersion;
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
  EXPECT_TRUE(InfoRV32I.getExtensions().at("i") == (RISCVExtensionInfo{2, 0}));
  EXPECT_EQ(InfoRV32I.getXLen(), 32U);

  auto MaybeRV32E = RISCVISAInfo::parseNormalizedArchString("rv32e2p0");
  ASSERT_THAT_EXPECTED(MaybeRV32E, Succeeded());
  RISCVISAInfo &InfoRV32E = **MaybeRV32E;
  EXPECT_EQ(InfoRV32E.getExtensions().size(), 1UL);
  EXPECT_TRUE(InfoRV32E.getExtensions().at("e") == (RISCVExtensionInfo{2, 0}));
  EXPECT_EQ(InfoRV32E.getXLen(), 32U);

  auto MaybeRV64I = RISCVISAInfo::parseNormalizedArchString("rv64i2p0");
  ASSERT_THAT_EXPECTED(MaybeRV64I, Succeeded());
  RISCVISAInfo &InfoRV64I = **MaybeRV64I;
  EXPECT_EQ(InfoRV64I.getExtensions().size(), 1UL);
  EXPECT_TRUE(InfoRV64I.getExtensions().at("i") == (RISCVExtensionInfo{2, 0}));
  EXPECT_EQ(InfoRV64I.getXLen(), 64U);

  auto MaybeRV64E = RISCVISAInfo::parseNormalizedArchString("rv64e2p0");
  ASSERT_THAT_EXPECTED(MaybeRV64E, Succeeded());
  RISCVISAInfo &InfoRV64E = **MaybeRV64E;
  EXPECT_EQ(InfoRV64E.getExtensions().size(), 1UL);
  EXPECT_TRUE(InfoRV64E.getExtensions().at("e") == (RISCVExtensionInfo{2, 0}));
  EXPECT_EQ(InfoRV64E.getXLen(), 64U);
}

TEST(ParseNormalizedArchString, AcceptsArbitraryExtensionsAndVersions) {
  auto MaybeISAInfo = RISCVISAInfo::parseNormalizedArchString(
      "rv64i5p1_m3p2_zmadeup11p12_sfoo2p0_xbar3p0");
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  RISCVISAInfo &Info = **MaybeISAInfo;
  EXPECT_EQ(Info.getExtensions().size(), 5UL);
  EXPECT_TRUE(Info.getExtensions().at("i") == (RISCVExtensionInfo{5, 1}));
  EXPECT_TRUE(Info.getExtensions().at("m") == (RISCVExtensionInfo{3, 2}));
  EXPECT_TRUE(Info.getExtensions().at("zmadeup") ==
              (RISCVExtensionInfo{11, 12}));
  EXPECT_TRUE(Info.getExtensions().at("sfoo") == (RISCVExtensionInfo{2, 0}));
  EXPECT_TRUE(Info.getExtensions().at("xbar") == (RISCVExtensionInfo{3, 0}));
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

TEST(ParseArchString, RejectsUpperCase) {
  for (StringRef Input : {"RV32", "rV64", "rv32i2P0", "rv64i2p0_A2p0"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "string must be lowercase");
  }
}

TEST(ParseArchString, RejectsInvalidBaseISA) {
  for (StringRef Input : {"rv32", "rv64", "rv65i"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "string must begin with rv32{i,e,g} or rv64{i,e,g}");
  }
  for (StringRef Input : {"rv32j", "rv64k", "rv32_i"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "first letter should be 'e', 'i' or 'g'");
  }
}

TEST(ParseArchString, RejectsUnsupportedBaseISA) {
  for (StringRef Input : {"rv128i", "rv128g"}) {
    EXPECT_EQ(toString(RISCVISAInfo::parseArchString(Input, true).takeError()),
              "string must begin with rv32{i,e,g} or rv64{i,e,g}");
  }
}

TEST(ParseArchString, AcceptsSupportedBaseISAsAndSetsXLenAndFLen) {
  auto MaybeRV32I = RISCVISAInfo::parseArchString("rv32i", true);
  ASSERT_THAT_EXPECTED(MaybeRV32I, Succeeded());
  RISCVISAInfo &InfoRV32I = **MaybeRV32I;
  RISCVISAInfo::OrderedExtensionMap ExtsRV32I = InfoRV32I.getExtensions();
  EXPECT_EQ(ExtsRV32I.size(), 1UL);
  EXPECT_TRUE(ExtsRV32I.at("i") == (RISCVExtensionInfo{2, 1}));
  EXPECT_EQ(InfoRV32I.getXLen(), 32U);
  EXPECT_EQ(InfoRV32I.getFLen(), 0U);

  auto MaybeRV32E = RISCVISAInfo::parseArchString("rv32e", true);
  ASSERT_THAT_EXPECTED(MaybeRV32E, Succeeded());
  RISCVISAInfo &InfoRV32E = **MaybeRV32E;
  RISCVISAInfo::OrderedExtensionMap ExtsRV32E = InfoRV32E.getExtensions();
  EXPECT_EQ(ExtsRV32E.size(), 1UL);
  EXPECT_TRUE(ExtsRV32E.at("e") == (RISCVExtensionInfo{2, 0}));
  EXPECT_EQ(InfoRV32E.getXLen(), 32U);
  EXPECT_EQ(InfoRV32E.getFLen(), 0U);

  auto MaybeRV32G = RISCVISAInfo::parseArchString("rv32g", true);
  ASSERT_THAT_EXPECTED(MaybeRV32G, Succeeded());
  RISCVISAInfo &InfoRV32G = **MaybeRV32G;
  RISCVISAInfo::OrderedExtensionMap ExtsRV32G = InfoRV32G.getExtensions();
  EXPECT_EQ(ExtsRV32G.size(), 7UL);
  EXPECT_TRUE(ExtsRV32G.at("i") == (RISCVExtensionInfo{2, 1}));
  EXPECT_TRUE(ExtsRV32G.at("m") == (RISCVExtensionInfo{2, 0}));
  EXPECT_TRUE(ExtsRV32G.at("a") == (RISCVExtensionInfo{2, 1}));
  EXPECT_TRUE(ExtsRV32G.at("f") == (RISCVExtensionInfo{2, 2}));
  EXPECT_TRUE(ExtsRV32G.at("d") == (RISCVExtensionInfo{2, 2}));
  EXPECT_TRUE(ExtsRV32G.at("zicsr") == (RISCVExtensionInfo{2, 0}));
  EXPECT_TRUE(ExtsRV32G.at("zifencei") == (RISCVExtensionInfo{2, 0}));
  EXPECT_EQ(InfoRV32G.getXLen(), 32U);
  EXPECT_EQ(InfoRV32G.getFLen(), 64U);

  auto MaybeRV64I = RISCVISAInfo::parseArchString("rv64i", true);
  ASSERT_THAT_EXPECTED(MaybeRV64I, Succeeded());
  RISCVISAInfo &InfoRV64I = **MaybeRV64I;
  RISCVISAInfo::OrderedExtensionMap ExtsRV64I = InfoRV64I.getExtensions();
  EXPECT_EQ(ExtsRV64I.size(), 1UL);
  EXPECT_TRUE(ExtsRV64I.at("i") == (RISCVExtensionInfo{2, 1}));
  EXPECT_EQ(InfoRV64I.getXLen(), 64U);
  EXPECT_EQ(InfoRV64I.getFLen(), 0U);

  auto MaybeRV64E = RISCVISAInfo::parseArchString("rv64e", true);
  ASSERT_THAT_EXPECTED(MaybeRV64E, Succeeded());
  RISCVISAInfo &InfoRV64E = **MaybeRV64E;
  RISCVISAInfo::OrderedExtensionMap ExtsRV64E = InfoRV64E.getExtensions();
  EXPECT_EQ(ExtsRV64E.size(), 1UL);
  EXPECT_TRUE(ExtsRV64E.at("e") == (RISCVExtensionInfo{2, 0}));
  EXPECT_EQ(InfoRV64E.getXLen(), 64U);
  EXPECT_EQ(InfoRV64E.getFLen(), 0U);

  auto MaybeRV64G = RISCVISAInfo::parseArchString("rv64g", true);
  ASSERT_THAT_EXPECTED(MaybeRV64G, Succeeded());
  RISCVISAInfo &InfoRV64G = **MaybeRV64G;
  RISCVISAInfo::OrderedExtensionMap ExtsRV64G = InfoRV64G.getExtensions();
  EXPECT_EQ(ExtsRV64G.size(), 7UL);
  EXPECT_TRUE(ExtsRV64G.at("i") == (RISCVExtensionInfo{2, 1}));
  EXPECT_TRUE(ExtsRV64G.at("m") == (RISCVExtensionInfo{2, 0}));
  EXPECT_TRUE(ExtsRV64G.at("a") == (RISCVExtensionInfo{2, 1}));
  EXPECT_TRUE(ExtsRV64G.at("f") == (RISCVExtensionInfo{2, 2}));
  EXPECT_TRUE(ExtsRV64G.at("d") == (RISCVExtensionInfo{2, 2}));
  EXPECT_TRUE(ExtsRV64G.at("zicsr") == (RISCVExtensionInfo{2, 0}));
  EXPECT_TRUE(ExtsRV64G.at("zifencei") == (RISCVExtensionInfo{2, 0}));
  EXPECT_EQ(InfoRV64G.getXLen(), 64U);
  EXPECT_EQ(InfoRV64G.getFLen(), 64U);
}

TEST(ParseArchString, RequiresCanonicalOrderForSingleLetterExtensions) {
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv64idf", true).takeError()),
      "standard user-level extension not given in canonical order 'f'");
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv32iam", true).takeError()),
      "standard user-level extension not given in canonical order 'm'");
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv32i_zfinx_a", true).takeError()),
      "invalid extension prefix 'a'");
  // Canonical ordering not required for z*, s*, and x* extensions.
  EXPECT_THAT_EXPECTED(
      RISCVISAInfo::parseArchString(
          "rv64imafdc_xsfvcp_zicsr_xtheadba_svnapot_zawrs", true),
      Succeeded());
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
}

TEST(ParseArchString, IgnoresUnrecognizedExtensionNamesWithIgnoreUnknown) {
  for (StringRef Input : {"rv32ib", "rv32i_zmadeup",
                          "rv64i_smadeup", "rv64i_xmadeup"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true, false, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    RISCVISAInfo &Info = **MaybeISAInfo;
    RISCVISAInfo::OrderedExtensionMap Exts = Info.getExtensions();
    EXPECT_EQ(Exts.size(), 1UL);
    EXPECT_TRUE(Exts.at("i") == (RISCVExtensionInfo{2, 1}));
  }

  // Checks that supported extensions aren't incorrectly ignored when a
  // version is present (an early version of the patch had this mistake).
  auto MaybeISAInfo =
      RISCVISAInfo::parseArchString("rv32i_zbc1p0_xmadeup", true, false, true);
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  RISCVISAInfo::OrderedExtensionMap Exts = (*MaybeISAInfo)->getExtensions();
  EXPECT_TRUE(Exts.at("zbc") == (RISCVExtensionInfo{1, 0}));
}

TEST(ParseArchString, AcceptsVersionInLongOrShortForm) {
  for (StringRef Input : {"rv64i2p1"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    RISCVISAInfo::OrderedExtensionMap Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_TRUE(Exts.at("i") == (RISCVExtensionInfo{2, 1}));
  }
  for (StringRef Input : {"rv32i_zfinx1", "rv32i_zfinx1p0"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    RISCVISAInfo::OrderedExtensionMap Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_TRUE(Exts.at("zfinx") == (RISCVExtensionInfo{1, 0}));
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
    RISCVISAInfo::OrderedExtensionMap Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_EQ(Exts.size(), 1UL);
    EXPECT_TRUE(Exts.at("i") == (RISCVExtensionInfo{2, 1}));
  }
  for (StringRef Input : {"rv32e0p1", "rv32e99p99", "rv64e0p1", "rv64e99p99"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true, false, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    RISCVISAInfo::OrderedExtensionMap Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_EQ(Exts.size(), 1UL);
    EXPECT_TRUE(Exts.at("e") == (RISCVExtensionInfo{2, 0}));
  }
}

TEST(ParseArchString,
     IgnoresExtensionsWithUnrecognizedVersionsWithIgnoreUnknown) {
  for (StringRef Input : {"rv32im1p1", "rv64i_svnapot10p9", "rv32i_zicsr0p5"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true, false, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    RISCVISAInfo::OrderedExtensionMap Exts = (*MaybeISAInfo)->getExtensions();
    EXPECT_EQ(Exts.size(), 1UL);
    EXPECT_TRUE(Exts.at("i") == (RISCVExtensionInfo{2, 1}));
  }
}

TEST(ParseArchString, AcceptsUnderscoreSplittingExtensions) {
  for (StringRef Input : {"rv32imafdczifencei", "rv32i_m_a_f_d_c_zifencei"}) {
    auto MaybeISAInfo = RISCVISAInfo::parseArchString(Input, true);
    ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
    RISCVISAInfo::OrderedExtensionMap Exts = (*MaybeISAInfo)->getExtensions();
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

TEST(ParseArchString, RejectsDoubleOrTrailingUnderscore) {
  EXPECT_EQ(
      toString(RISCVISAInfo::parseArchString("rv64i__m", true).takeError()),
      "invalid standard user-level extension '_'");

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
      "standard user-level extension not given in canonical order 'm'");
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv32i_zicsr_zicsr", true).takeError()),
      "duplicated standard user-level extension 'zicsr'");
}

TEST(ParseArchString,
     RejectsExperimentalExtensionsIfNotEnableExperimentalExtension) {
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv64izicond", false).takeError()),
      "requires '-menable-experimental-extensions' for experimental extension "
      "'zicond'");
}

TEST(ParseArchString,
     AcceptsExperimentalExtensionsIfEnableExperimentalExtension) {
  // Note: If zicond becomes none-experimental, this test will need
  // updating (and unfortunately, it will still pass). The failure of
  // RejectsExperimentalExtensionsIfNotEnableExperimentalExtension will
  // hopefully serve as a reminder to update.
  auto MaybeISAInfo =
      RISCVISAInfo::parseArchString("rv64izicond", true, false);
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  RISCVISAInfo::OrderedExtensionMap Exts = (*MaybeISAInfo)->getExtensions();
  EXPECT_EQ(Exts.size(), 2UL);
  EXPECT_EQ(Exts.count("zicond"), 1U);
  auto MaybeISAInfo2 = RISCVISAInfo::parseArchString("rv64izicond1p0", true);
  ASSERT_THAT_EXPECTED(MaybeISAInfo2, Succeeded());
  RISCVISAInfo::OrderedExtensionMap Exts2 = (*MaybeISAInfo2)->getExtensions();
  EXPECT_EQ(Exts2.size(), 2UL);
  EXPECT_EQ(Exts2.count("zicond"), 1U);
}

TEST(ParseArchString,
     RequiresExplicitVersionNumberForExperimentalExtensionByDefault) {
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv64izicond", true).takeError()),
      "experimental extension requires explicit version number `zicond`");
}

TEST(ParseArchString,
     AcceptsUnrecognizedVersionIfNotExperimentalExtensionVersionCheck) {
  auto MaybeISAInfo =
      RISCVISAInfo::parseArchString("rv64izicond9p9", true, false);
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  RISCVISAInfo::OrderedExtensionMap Exts = (*MaybeISAInfo)->getExtensions();
  EXPECT_EQ(Exts.size(), 2UL);
  EXPECT_TRUE(Exts.at("zicond") == (RISCVExtensionInfo{9, 9}));
}

TEST(ParseArchString, RejectsUnrecognizedVersionForExperimentalExtension) {
  EXPECT_EQ(
      toString(
          RISCVISAInfo::parseArchString("rv64izicond9p9", true).takeError()),
      "unsupported version number 9.9 for experimental extension 'zicond' "
      "(this compiler supports 1.0)");
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
  RISCVISAInfo::OrderedExtensionMap ExtsRV64ID =
      (*MaybeRV64ID)->getExtensions();
  EXPECT_EQ(ExtsRV64ID.size(), 4UL);
  EXPECT_EQ(ExtsRV64ID.count("i"), 1U);
  EXPECT_EQ(ExtsRV64ID.count("f"), 1U);
  EXPECT_EQ(ExtsRV64ID.count("d"), 1U);
  EXPECT_EQ(ExtsRV64ID.count("zicsr"), 1U);

  auto MaybeRV32IZKN = RISCVISAInfo::parseArchString("rv64izkn", true);
  ASSERT_THAT_EXPECTED(MaybeRV32IZKN, Succeeded());
  RISCVISAInfo::OrderedExtensionMap ExtsRV32IZKN =
      (*MaybeRV32IZKN)->getExtensions();
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

TEST(ToFeatureVector, IIsDroppedAndExperimentalExtensionsArePrefixed) {
  auto MaybeISAInfo1 =
      RISCVISAInfo::parseArchString("rv64im_zicond", true, false);
  ASSERT_THAT_EXPECTED(MaybeISAInfo1, Succeeded());
  EXPECT_THAT((*MaybeISAInfo1)->toFeatureVector(),
              ElementsAre("+m", "+experimental-zicond"));

  auto MaybeISAInfo2 = RISCVISAInfo::parseArchString(
      "rv32e_zicond_xventanacondops", true, false);
  ASSERT_THAT_EXPECTED(MaybeISAInfo2, Succeeded());
  EXPECT_THAT((*MaybeISAInfo2)->toFeatureVector(),
              ElementsAre("+e", "+experimental-zicond", "+xventanacondops"));
}

TEST(ToFeatureVector, UnsupportedExtensionsAreDropped) {
  auto MaybeISAInfo =
      RISCVISAInfo::parseNormalizedArchString("rv64i2p0_m2p0_xmadeup1p0");
  ASSERT_THAT_EXPECTED(MaybeISAInfo, Succeeded());
  EXPECT_THAT((*MaybeISAInfo)->toFeatureVector(), ElementsAre("+m"));
}

TEST(OrderedExtensionMap, ExtensionsAreCorrectlyOrdered) {
  RISCVISAInfo::OrderedExtensionMap Exts;
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
  RISCVISAInfo::OrderedExtensionMap ExtsRV32IZce =
      (*MaybeRV32IZce)->getExtensions();
  EXPECT_EQ(ExtsRV32IZce.size(), 6UL);
  EXPECT_EQ(ExtsRV32IZce.count("i"), 1U);
  EXPECT_EQ(ExtsRV32IZce.count("zca"), 1U);
  EXPECT_EQ(ExtsRV32IZce.count("zcb"), 1U);
  EXPECT_EQ(ExtsRV32IZce.count("zce"), 1U);
  EXPECT_EQ(ExtsRV32IZce.count("zcmp"), 1U);
  EXPECT_EQ(ExtsRV32IZce.count("zcmt"), 1U);

  auto MaybeRV32IFZce = RISCVISAInfo::parseArchString("rv32ifzce", true);
  ASSERT_THAT_EXPECTED(MaybeRV32IFZce, Succeeded());
  RISCVISAInfo::OrderedExtensionMap ExtsRV32IFZce =
      (*MaybeRV32IFZce)->getExtensions();
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
  RISCVISAInfo::OrderedExtensionMap ExtsRV32IDZce =
      (*MaybeRV32IDZce)->getExtensions();
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
  RISCVISAInfo::OrderedExtensionMap ExtsRV64IZce =
      (*MaybeRV64IZce)->getExtensions();
  EXPECT_EQ(ExtsRV64IZce.size(), 6UL);
  EXPECT_EQ(ExtsRV64IZce.count("i"), 1U);
  EXPECT_EQ(ExtsRV64IZce.count("zca"), 1U);
  EXPECT_EQ(ExtsRV64IZce.count("zcb"), 1U);
  EXPECT_EQ(ExtsRV64IZce.count("zce"), 1U);
  EXPECT_EQ(ExtsRV64IZce.count("zcmp"), 1U);
  EXPECT_EQ(ExtsRV64IZce.count("zcmt"), 1U);

  auto MaybeRV64IFZce = RISCVISAInfo::parseArchString("rv64ifzce", true);
  ASSERT_THAT_EXPECTED(MaybeRV64IFZce, Succeeded());
  RISCVISAInfo::OrderedExtensionMap ExtsRV64IFZce =
      (*MaybeRV64IFZce)->getExtensions();
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
  RISCVISAInfo::OrderedExtensionMap ExtsRV64IDZce =
      (*MaybeRV64IDZce)->getExtensions();
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
