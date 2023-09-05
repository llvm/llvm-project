//===----------- TargetParser.cpp - Target Parser -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/TargetParser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/ARMTargetParserCommon.h"
#include "llvm/TargetParser/Triple.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>
#include <string>

using namespace llvm;

namespace {
const char *AArch64Arch[] = {
    "armv4",        "armv4t",      "armv5",          "armv5t",      "armv5e",
    "armv5te",      "armv5tej",    "armv6",          "armv6j",      "armv6k",
    "armv6hl",      "armv6t2",     "armv6kz",        "armv6z",      "armv6zk",
    "armv6-m",      "armv6m",      "armv6sm",        "armv6s-m",    "armv7-a",
    "armv7",        "armv7a",      "armv7ve",        "armv7hl",     "armv7l",
    "armv7-r",      "armv7r",      "armv7-m",        "armv7m",      "armv7k",
    "armv7s",       "armv7e-m",    "armv7em",        "armv8-a",     "armv8",
    "armv8a",       "armv8l",      "armv8.1-a",      "armv8.1a",    "armv8.2-a",
    "armv8.2a",     "armv8.3-a",   "armv8.3a",       "armv8.4-a",   "armv8.4a",
    "armv8.5-a",    "armv8.5a",    "armv8.6-a",      "armv8.6a",    "armv8.7-a",
    "armv8.7a",     "armv8.8-a",   "armv8.8a",       "armv8.9-a",   "armv8.9a",
    "armv8-r",      "armv8r",      "armv8-m.base",   "armv8m.base", "armv8-m.main",
    "armv8m.main",  "iwmmxt",      "iwmmxt2",        "xscale",      "armv8.1-m.main",
    "armv9-a",      "armv9",       "armv9a",         "armv9.1-a",   "armv9.1a",
    "armv9.2-a",    "armv9.2a",    "armv9.3-a",      "armv9.3a",    "armv9.4-a",
    "armv9.4a",
};

std::string FormatExtensionFlags(BitVector Flags) {
  std::vector<StringRef> Features;

  // AEK_NONE is not meant to be shown to the user so the target parser
  // does not recognise it. It is relevant here though.
  if (Flags.test(AArch64::AEK_NONE))
    Features.push_back("none");
  AArch64::getExtensionFeatures(Flags, Features);

  // The target parser also includes every extension you don't have.
  // E.g. if AEK_CRC is not set then it adds "-crc". Not useful here.
  Features.erase(std::remove_if(Features.begin(), Features.end(),
                                [](StringRef extension) {
                                  return extension.startswith("-");
                                }),
                 Features.end());

  return llvm::join(Features, ", ");
}

std::string SerializeExtensionFlags(BitVector Flags) {
  std::string SerializedFlags;

  for(unsigned int i = 0; i < Flags.size(); i ++)
    SerializedFlags += (int)Flags[i];
  return SerializedFlags;
}

struct AssertSameExtensionFlags {
  AssertSameExtensionFlags(StringRef CPUName) : CPUName(CPUName) {}

  testing::AssertionResult operator()(const char *m_expr, const char *n_expr,
                                      BitVector ExpectedFlags,
                                      BitVector GotFlags) {
    if (ExpectedFlags == GotFlags)
      return testing::AssertionSuccess();

    return testing::AssertionFailure() << llvm::formatv(
               "CPU: {4}\n"
               "Expected extension flags: {0} ({1:x})\n"
               "     Got extension flags: {2} ({3:x})\n",
               FormatExtensionFlags(ExpectedFlags), SerializeExtensionFlags(ExpectedFlags),
               FormatExtensionFlags(GotFlags), SerializeExtensionFlags(GotFlags), CPUName);
  }

private:
  StringRef CPUName;
};

struct AArch64CPUTestParams {
  AArch64CPUTestParams(StringRef CPUName, StringRef ExpectedArch,
                   StringRef ExpectedFPU, BitVector ExpectedFlags,
                   StringRef CPUAttr)
      : CPUName(CPUName), ExpectedArch(ExpectedArch), ExpectedFPU(ExpectedFPU),
        ExpectedFlags(ExpectedFlags), CPUAttr(CPUAttr) {}

  friend std::ostream &operator<<(std::ostream &os,
                                  const AArch64CPUTestParams &params) {
    return os << "\"" << params.CPUName.str() << "\", \""
              << params.ExpectedArch.str() << "\", \""
              << params.ExpectedFPU.str() << "\", 0x" << std::hex
              << SerializeExtensionFlags(params.ExpectedFlags) << ", \"" << params.CPUAttr.str() << "\"";
  }

  StringRef CPUName;
  StringRef ExpectedArch;
  StringRef ExpectedFPU;
  BitVector ExpectedFlags;
  StringRef CPUAttr;
};

class CPUTestFixture
    : public ::testing::TestWithParam<AArch64CPUTestParams> {};

TEST_P(CPUTestFixture, testAArch64CPU) {
  AArch64CPUTestParams params = GetParam();

  const std::optional<AArch64::CpuInfo> Cpu = AArch64::parseCpu(params.CPUName);
  EXPECT_TRUE(Cpu);
  EXPECT_EQ(params.ExpectedArch, Cpu->Arch.Name);

  EXPECT_PRED_FORMAT2(
      AssertSameExtensionFlags(params.CPUName),
      params.ExpectedFlags, Cpu->getImpliedExtensions());
}

INSTANTIATE_TEST_SUITE_P(
    AArch64CPUTests, CPUTestFixture,
    ::testing::Values(
        AArch64CPUTestParams("cortex-a34", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("cortex-a35", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("cortex-a53", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("cortex-a55", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_FP16).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_RCPC)),
                         "8.2-A"),
        AArch64CPUTestParams(
            "cortex-a510", "armv9-a", "neon-fp-armv8",
            (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_FP).set(AArch64::AEK_SIMD).
                set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).set(AArch64::AEK_RDM).
                set(AArch64::AEK_RCPC).set(AArch64::AEK_DOTPROD).set(AArch64::AEK_BF16).
                set(AArch64::AEK_I8MM).set(AArch64::AEK_SVE).set(AArch64::AEK_SVE2).
                set(AArch64::AEK_SVE2BITPERM).set(AArch64::AEK_PAUTH).
                set(AArch64::AEK_MTE).set(AArch64::AEK_SSBS).set(AArch64::AEK_FP16).
                set(AArch64::AEK_FP16FML).set(AArch64::AEK_SB)),
            "9-A"),
        AArch64CPUTestParams("cortex-a57", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("cortex-a65", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_FP).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_SSBS)),
                         "8.2-A"),
        AArch64CPUTestParams("cortex-a65ae", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_FP).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_SSBS)),
                         "8.2-A"),
        AArch64CPUTestParams("cortex-a72", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("cortex-a73", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("cortex-a75", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_FP16).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_RCPC)),
                         "8.2-A"),
        AArch64CPUTestParams("cortex-a76", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_FP16).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_SSBS)),
                         "8.2-A"),
        AArch64CPUTestParams("cortex-a76ae", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_FP16).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_SSBS)),
                         "8.2-A"),
        AArch64CPUTestParams("cortex-a77", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_FP16).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_SSBS)),
                         "8.2-A"),
        AArch64CPUTestParams("cortex-a78", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_FP16).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_SSBS).
                             set(AArch64::AEK_PROFILE)),
                         "8.2-A"),
        AArch64CPUTestParams(
            "cortex-a78c", "armv8.2-a", "crypto-neon-fp-armv8",
            (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_RAS).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).set(AArch64::AEK_SIMD).
                set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).set(AArch64::AEK_RDM).
                set(AArch64::AEK_FP16).set(AArch64::AEK_DOTPROD).set(AArch64::AEK_RCPC).
                set(AArch64::AEK_SSBS).set(AArch64::AEK_PROFILE).set(AArch64::AEK_FLAGM).
                set(AArch64::AEK_PAUTH).set(AArch64::AEK_FP16FML)),
            "8.2-A"),
        AArch64CPUTestParams(
            "cortex-a710", "armv9-a", "neon-fp-armv8",
            (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_FP).set(AArch64::AEK_SIMD).
                set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).set(AArch64::AEK_RDM).
                set(AArch64::AEK_RCPC).set(AArch64::AEK_DOTPROD).set(AArch64::AEK_MTE).
                set(AArch64::AEK_FP16).set(AArch64::AEK_FP16FML).set(AArch64::AEK_SVE).
                set(AArch64::AEK_SVE2).set(AArch64::AEK_SVE2BITPERM).
                set(AArch64::AEK_PAUTH).set(AArch64::AEK_FLAGM).set(AArch64::AEK_SB).
                set(AArch64::AEK_I8MM).set(AArch64::AEK_BF16)),
            "9-A"),
        AArch64CPUTestParams(
            "cortex-a715", "armv9-a", "neon-fp-armv8",
            (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_FP).set(AArch64::AEK_BF16).
                set(AArch64::AEK_SIMD).set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).set(AArch64::AEK_DOTPROD).
                set(AArch64::AEK_MTE).set(AArch64::AEK_PAUTH).set(AArch64::AEK_SVE).
                set(AArch64::AEK_SVE2).set(AArch64::AEK_SVE2BITPERM).
                set(AArch64::AEK_SSBS).set(AArch64::AEK_SB).set(AArch64::AEK_I8MM).
                set(AArch64::AEK_PERFMON).set(AArch64::AEK_PREDRES).
                set(AArch64::AEK_PROFILE).set(AArch64::AEK_FP16FML).
                set(AArch64::AEK_FP16).set(AArch64::AEK_FLAGM)),
            "9-A"),
        AArch64CPUTestParams(
            "neoverse-v1", "armv8.4-a", "crypto-neon-fp-armv8",
            (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_RAS).set(AArch64::AEK_SVE).set(AArch64::AEK_SSBS).
                set(AArch64::AEK_RCPC).set(AArch64::AEK_CRC).set(AArch64::AEK_FP).
                set(AArch64::AEK_SIMD).set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).set(AArch64::AEK_DOTPROD).
                set(AArch64::AEK_AES).set(AArch64::AEK_SHA2).set(AArch64::AEK_SHA3).
                set(AArch64::AEK_SM4).set(AArch64::AEK_FP16).set(AArch64::AEK_BF16).
                set(AArch64::AEK_PROFILE).set(AArch64::AEK_RAND).
                set(AArch64::AEK_FP16FML).set(AArch64::AEK_I8MM)),
            "8.4-A"),
        AArch64CPUTestParams("neoverse-v2", "armv9-a", "neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_RAS).set(AArch64::AEK_SVE).
                             set(AArch64::AEK_SSBS).set(AArch64::AEK_RCPC).
                             set(AArch64::AEK_CRC).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_MTE).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_FP16).set(AArch64::AEK_BF16).
                             set(AArch64::AEK_SVE2).set(AArch64::AEK_PROFILE).
                             set(AArch64::AEK_FP16FML).set(AArch64::AEK_I8MM).
                             set(AArch64::AEK_SVE2BITPERM).set(AArch64::AEK_RAND)),
                         "9-A"),
        AArch64CPUTestParams("cortex-r82", "armv8-r", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_SSBS).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_FP).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_FP16).set(AArch64::AEK_FP16FML).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_RCPC).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_SB)),
                         "8-R"),
        AArch64CPUTestParams("cortex-x1", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_FP16).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_SSBS).
                             set(AArch64::AEK_PROFILE)),
                         "8.2-A"),
        AArch64CPUTestParams("cortex-x1c", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_FP16).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_SSBS).
                             set(AArch64::AEK_PAUTH).set(AArch64::AEK_PROFILE)),
                         "8.2-A"),
        AArch64CPUTestParams("cortex-x2", "armv9-a", "neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_MTE).set(AArch64::AEK_PAUTH).
                             set(AArch64::AEK_I8MM).set(AArch64::AEK_BF16).
                             set(AArch64::AEK_SVE).set(AArch64::AEK_SVE2).
                             set(AArch64::AEK_SVE2BITPERM).set(AArch64::AEK_SSBS).
                             set(AArch64::AEK_SB).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_FP16FML)),
                         "9-A"),
        AArch64CPUTestParams(
            "cortex-x3",
            "armv9-a", "neon-fp-armv8",
            (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_FP).set(AArch64::AEK_BF16).
                set(AArch64::AEK_SIMD).set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).set(AArch64::AEK_DOTPROD).
                set(AArch64::AEK_MTE).set(AArch64::AEK_PAUTH).set(AArch64::AEK_SVE).
                set(AArch64::AEK_SVE2).set(AArch64::AEK_SVE2BITPERM).set(AArch64::AEK_SB).
                set(AArch64::AEK_PROFILE).set(AArch64::AEK_PERFMON).
                set(AArch64::AEK_I8MM).set(AArch64::AEK_FP16).set(AArch64::AEK_FP16FML).
                set(AArch64::AEK_PREDRES).set(AArch64::AEK_FLAGM).set(AArch64::AEK_SSBS)),
            "9-A"),
        AArch64CPUTestParams("cyclone", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_NONE).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("apple-a7", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_NONE).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("apple-a8", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_NONE).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("apple-a9", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_NONE).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("apple-a10", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("apple-a11", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_FP16)),
                         "8.2-A"),
        AArch64CPUTestParams("apple-a12", "armv8.3-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_FP16)),
                         "8.3-A"),
        AArch64CPUTestParams("apple-a13", "armv8.4-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_SHA3).
                             set(AArch64::AEK_FP).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).
                             set(AArch64::AEK_DOTPROD).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_FP16FML).set(AArch64::AEK_SHA3)),
                         "8.4-A"),
        AArch64CPUTestParams("apple-a14", "armv8.5-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_SHA3).
                             set(AArch64::AEK_FP).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).
                             set(AArch64::AEK_DOTPROD).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_FP16FML).set(AArch64::AEK_SHA3)),
                         "8.5-A"),
        AArch64CPUTestParams("apple-a15", "armv8.5-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_SHA3).
                             set(AArch64::AEK_FP).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).
                             set(AArch64::AEK_DOTPROD).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_FP16FML).set(AArch64::AEK_SHA3).
                             set(AArch64::AEK_BF16).set(AArch64::AEK_I8MM)),
                         "8.5-A"),
        AArch64CPUTestParams("apple-a16", "armv8.5-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_SHA3).
                             set(AArch64::AEK_FP).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).
                             set(AArch64::AEK_DOTPROD).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_FP16FML).set(AArch64::AEK_SHA3).
                             set(AArch64::AEK_BF16).set(AArch64::AEK_I8MM)),
                         "8.5-A"),
        AArch64CPUTestParams("apple-m1", "armv8.5-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_SHA3).
                             set(AArch64::AEK_FP).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).
                             set(AArch64::AEK_DOTPROD).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_FP16FML).set(AArch64::AEK_SHA3)),
                         "8.5-A"),
        AArch64CPUTestParams("apple-m2", "armv8.5-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_SHA3).
                             set(AArch64::AEK_FP).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).
                             set(AArch64::AEK_DOTPROD).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_FP16FML).set(AArch64::AEK_SHA3).
                             set(AArch64::AEK_BF16).set(AArch64::AEK_I8MM)),
                         "8.5-A"),
        AArch64CPUTestParams("apple-s4", "armv8.3-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_FP16)),
                         "8.3-A"),
        AArch64CPUTestParams("apple-s5", "armv8.3-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_FP16)),
                         "8.3-A"),
        AArch64CPUTestParams("exynos-m3", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("exynos-m4", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_FP).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_SIMD)),
                         "8.2-A"),
        AArch64CPUTestParams("exynos-m5", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_FP).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_SIMD)),
                         "8.2-A"),
        AArch64CPUTestParams("falkor", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_RDM)),
                         "8-A"),
        AArch64CPUTestParams("kryo", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8-A"),
        AArch64CPUTestParams("neoverse-e1", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_FP).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RCPC).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_SSBS)),
                         "8.2-A"),
        AArch64CPUTestParams("neoverse-n1", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_DOTPROD).
                             set(AArch64::AEK_FP).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_PROFILE).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_RCPC).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_SSBS)),
                         "8.2-A"),
        AArch64CPUTestParams(
            "neoverse-n2", "armv8.5-a", "crypto-neon-fp-armv8",
            (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).set(AArch64::AEK_SHA2).
                set(AArch64::AEK_SHA3).set(AArch64::AEK_SM4).set(AArch64::AEK_FP).
                set(AArch64::AEK_SIMD).set(AArch64::AEK_FP16).set(AArch64::AEK_RAS).
                set(AArch64::AEK_LSE).set(AArch64::AEK_SVE).set(AArch64::AEK_DOTPROD).
                set(AArch64::AEK_RCPC).set(AArch64::AEK_RDM).set(AArch64::AEK_MTE).
                set(AArch64::AEK_SSBS).set(AArch64::AEK_SB).set(AArch64::AEK_SVE2).
                set(AArch64::AEK_SVE2BITPERM).set(AArch64::AEK_BF16).
                set(AArch64::AEK_I8MM)),
            "8.5-A"),
        AArch64CPUTestParams(
            "ampere1", "armv8.6-a", "crypto-neon-fp-armv8",
            (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_FP).set(AArch64::AEK_FP16).
                set(AArch64::AEK_SIMD).set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).set(AArch64::AEK_DOTPROD).
                set(AArch64::AEK_SHA3).set(AArch64::AEK_BF16).set(AArch64::AEK_SHA2).
                set(AArch64::AEK_AES).set(AArch64::AEK_I8MM).set(AArch64::AEK_SSBS).
                set(AArch64::AEK_SB).set(AArch64::AEK_RAND)),
            "8.6-A"),
        AArch64CPUTestParams(
            "ampere1a", "armv8.6-a", "crypto-neon-fp-armv8",
            (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_FP).set(AArch64::AEK_FP16).
                set(AArch64::AEK_SIMD).set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).set(AArch64::AEK_DOTPROD).
                set(AArch64::AEK_SM4).set(AArch64::AEK_SHA3).set(AArch64::AEK_BF16).
                set(AArch64::AEK_SHA2).set(AArch64::AEK_AES).set(AArch64::AEK_I8MM).
                set(AArch64::AEK_SSBS).set(AArch64::AEK_SB).set(AArch64::AEK_RAND).
                set(AArch64::AEK_MTE)),
            "8.6-A"),
        AArch64CPUTestParams(
            "neoverse-512tvb", "armv8.4-a", "crypto-neon-fp-armv8",
            (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_RAS).set(AArch64::AEK_SVE).set(AArch64::AEK_SSBS).
                set(AArch64::AEK_RCPC).set(AArch64::AEK_CRC).set(AArch64::AEK_FP).
                set(AArch64::AEK_SIMD).set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                set(AArch64::AEK_RDM).set(AArch64::AEK_RCPC).set(AArch64::AEK_DOTPROD).
                set(AArch64::AEK_AES).set(AArch64::AEK_SHA2).set(AArch64::AEK_SHA3).
                set(AArch64::AEK_SM4).set(AArch64::AEK_FP16).set(AArch64::AEK_BF16).
                set(AArch64::AEK_PROFILE).set(AArch64::AEK_RAND).
                set(AArch64::AEK_FP16FML).set(AArch64::AEK_I8MM)),
            "8.4-A"),
        AArch64CPUTestParams("thunderx2t99", "armv8.1-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD)),
                         "8.1-A"),
        AArch64CPUTestParams("thunderx3t110", "armv8.3-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_RDM).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_RCPC)),
                         "8.3-A"),
        AArch64CPUTestParams("thunderx", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_FP)),
                         "8-A"),
        AArch64CPUTestParams("thunderxt81", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_FP)),
                         "8-A"),
        AArch64CPUTestParams("thunderxt83", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_FP)),
                         "8-A"),
        AArch64CPUTestParams("thunderxt88", "armv8-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_SIMD).
                             set(AArch64::AEK_FP)),
                         "8-A"),
        AArch64CPUTestParams("tsv110", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_RAS).
                             set(AArch64::AEK_LSE).set(AArch64::AEK_RDM).
                             set(AArch64::AEK_PROFILE).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_FP16FML).set(AArch64::AEK_DOTPROD)),
                         "8.2-A"),
        AArch64CPUTestParams("a64fx", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_SVE).set(AArch64::AEK_RDM)),
                         "8.2-A"),
        AArch64CPUTestParams("carmel", "armv8.2-a", "crypto-neon-fp-armv8",
                         (BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_CRC).set(AArch64::AEK_AES).
                             set(AArch64::AEK_SHA2).set(AArch64::AEK_FP).
                             set(AArch64::AEK_SIMD).set(AArch64::AEK_FP16).
                             set(AArch64::AEK_RAS).set(AArch64::AEK_LSE).
                             set(AArch64::AEK_RDM)),
                         "8.2-A")));

// Note: number of CPUs includes aliases.
static constexpr unsigned NumAArch64CPUArchs = 62;

TEST(TargetParserTest, testAArch64CPUArchList) {
  SmallVector<StringRef, NumAArch64CPUArchs> List;
  AArch64::fillValidCPUArchList(List);

  // No list exists for these in this test suite, so ensure all are
  // valid, and match the expected 'magic' count.
  EXPECT_EQ(List.size(), NumAArch64CPUArchs);
  for(StringRef CPU : List) {
    EXPECT_TRUE(AArch64::parseCpu(CPU));
  }
}

bool testAArch64Arch(StringRef Arch, StringRef DefaultCPU, StringRef SubArch,
                     unsigned ArchAttr) {
  const std::optional<AArch64::ArchInfo> AI = AArch64::parseArch(Arch);
  return AI.has_value();
}

TEST(TargetParserTest, testAArch64Arch) {
  EXPECT_TRUE(testAArch64Arch("armv8-a", "cortex-a53", "v8a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv8.1-a", "generic", "v8.1a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv8.2-a", "generic", "v8.2a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv8.3-a", "generic", "v8.3a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv8.4-a", "generic", "v8.4a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv8.5-a", "generic", "v8.5a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv8.6-a", "generic", "v8.6a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv8.7-a", "generic", "v8.7a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv8.8-a", "generic", "v8.8a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv8.9-a", "generic", "v8.9a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv9-a", "generic", "v9a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv9.1-a", "generic", "v9.1a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv9.2-a", "generic", "v9.2a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv9.3-a", "generic", "v9.3a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv9.4-a", "generic", "v9.4a",
                              ARMBuildAttrs::CPUArch::v8_A));
}

bool testAArch64Extension(StringRef CPUName, StringRef ArchExt) {
  std::optional<AArch64::ExtensionInfo> Extension =
      AArch64::parseArchExtension(ArchExt);
  if (!Extension)
    return false;
  std::optional<AArch64::CpuInfo> CpuInfo = AArch64::parseCpu(CPUName);
  return CpuInfo->getImpliedExtensions().test(Extension->ID);
}

bool testAArch64Extension(const AArch64::ArchInfo &AI, StringRef ArchExt) {
  std::optional<AArch64::ExtensionInfo> Extension =
      AArch64::parseArchExtension(ArchExt);
  if (!Extension)
    return false;
  return AI.DefaultExts.test(Extension->ID);
}

TEST(TargetParserTest, testAArch64Extension) {
  EXPECT_FALSE(testAArch64Extension("cortex-a34", "ras"));
  EXPECT_FALSE(testAArch64Extension("cortex-a35", "ras"));
  EXPECT_FALSE(testAArch64Extension("cortex-a53", "ras"));
  EXPECT_TRUE(testAArch64Extension("cortex-a55", "ras"));
  EXPECT_TRUE(testAArch64Extension("cortex-a55", "fp16"));
  EXPECT_FALSE(testAArch64Extension("cortex-a55", "fp16fml"));
  EXPECT_TRUE(testAArch64Extension("cortex-a55", "dotprod"));
  EXPECT_FALSE(testAArch64Extension("cortex-a57", "ras"));
  EXPECT_FALSE(testAArch64Extension("cortex-a72", "ras"));
  EXPECT_FALSE(testAArch64Extension("cortex-a73", "ras"));
  EXPECT_TRUE(testAArch64Extension("cortex-a75", "ras"));
  EXPECT_TRUE(testAArch64Extension("cortex-a75", "fp16"));
  EXPECT_FALSE(testAArch64Extension("cortex-a75", "fp16fml"));
  EXPECT_TRUE(testAArch64Extension("cortex-a75", "dotprod"));
  EXPECT_TRUE(testAArch64Extension("cortex-r82", "ras"));
  EXPECT_TRUE(testAArch64Extension("cortex-r82", "fp16"));
  EXPECT_TRUE(testAArch64Extension("cortex-r82", "fp16fml"));
  EXPECT_TRUE(testAArch64Extension("cortex-r82", "dotprod"));
  EXPECT_TRUE(testAArch64Extension("cortex-r82", "lse"));
  EXPECT_FALSE(testAArch64Extension("cyclone", "ras"));
  EXPECT_FALSE(testAArch64Extension("exynos-m3", "ras"));
  EXPECT_TRUE(testAArch64Extension("exynos-m4", "dotprod"));
  EXPECT_TRUE(testAArch64Extension("exynos-m4", "fp16"));
  EXPECT_TRUE(testAArch64Extension("exynos-m4", "lse"));
  EXPECT_TRUE(testAArch64Extension("exynos-m4", "ras"));
  EXPECT_TRUE(testAArch64Extension("exynos-m4", "rdm"));
  EXPECT_TRUE(testAArch64Extension("exynos-m5", "dotprod"));
  EXPECT_TRUE(testAArch64Extension("exynos-m5", "fp16"));
  EXPECT_TRUE(testAArch64Extension("exynos-m5", "lse"));
  EXPECT_TRUE(testAArch64Extension("exynos-m5", "ras"));
  EXPECT_TRUE(testAArch64Extension("exynos-m5", "rdm"));
  EXPECT_TRUE(testAArch64Extension("falkor", "rdm"));
  EXPECT_FALSE(testAArch64Extension("kryo", "ras"));
  EXPECT_TRUE(testAArch64Extension("saphira", "crc"));
  EXPECT_TRUE(testAArch64Extension("saphira", "lse"));
  EXPECT_TRUE(testAArch64Extension("saphira", "rdm"));
  EXPECT_TRUE(testAArch64Extension("saphira", "ras"));
  EXPECT_TRUE(testAArch64Extension("saphira", "rcpc"));
  EXPECT_TRUE(testAArch64Extension("saphira", "profile"));
  EXPECT_FALSE(testAArch64Extension("saphira", "fp16"));
  EXPECT_FALSE(testAArch64Extension("thunderx2t99", "ras"));
  EXPECT_FALSE(testAArch64Extension("thunderx", "lse"));
  EXPECT_FALSE(testAArch64Extension("thunderxt81", "lse"));
  EXPECT_FALSE(testAArch64Extension("thunderxt83", "lse"));
  EXPECT_FALSE(testAArch64Extension("thunderxt88", "lse"));
  EXPECT_TRUE(testAArch64Extension("tsv110", "aes"));
  EXPECT_TRUE(testAArch64Extension("tsv110", "sha2"));
  EXPECT_FALSE(testAArch64Extension("tsv110", "sha3"));
  EXPECT_FALSE(testAArch64Extension("tsv110", "sm4"));
  EXPECT_TRUE(testAArch64Extension("tsv110", "ras"));
  EXPECT_TRUE(testAArch64Extension("tsv110", "profile"));
  EXPECT_TRUE(testAArch64Extension("tsv110", "fp16"));
  EXPECT_TRUE(testAArch64Extension("tsv110", "fp16fml"));
  EXPECT_TRUE(testAArch64Extension("tsv110", "dotprod"));
  EXPECT_TRUE(testAArch64Extension("a64fx", "fp16"));
  EXPECT_TRUE(testAArch64Extension("a64fx", "sve"));
  EXPECT_FALSE(testAArch64Extension("a64fx", "sve2"));
  EXPECT_TRUE(testAArch64Extension("carmel", "aes"));
  EXPECT_TRUE(testAArch64Extension("carmel", "sha2"));
  EXPECT_TRUE(testAArch64Extension("carmel", "fp16"));

  EXPECT_FALSE(testAArch64Extension(AArch64::ARMV8A, "ras"));
  EXPECT_FALSE(testAArch64Extension(AArch64::ARMV8_1A, "ras"));
  EXPECT_FALSE(testAArch64Extension(AArch64::ARMV8_2A, "profile"));
  EXPECT_FALSE(testAArch64Extension(AArch64::ARMV8_2A, "fp16"));
  EXPECT_FALSE(testAArch64Extension(AArch64::ARMV8_2A, "fp16fml"));
  EXPECT_FALSE(testAArch64Extension(AArch64::ARMV8_3A, "fp16"));
  EXPECT_FALSE(testAArch64Extension(AArch64::ARMV8_3A, "fp16fml"));
  EXPECT_FALSE(testAArch64Extension(AArch64::ARMV8_4A, "fp16"));
  EXPECT_FALSE(testAArch64Extension(AArch64::ARMV8_4A, "fp16fml"));
}

TEST(TargetParserTest, AArch64ExtensionFeatures) {
  std::vector<uint64_t> Extensions = {
      AArch64::AEK_CRC,           AArch64::AEK_LSE,           AArch64::AEK_RDM,
      AArch64::AEK_CRYPTO,        AArch64::AEK_SM4,           AArch64::AEK_SHA3,
      AArch64::AEK_SHA2,          AArch64::AEK_AES,           AArch64::AEK_DOTPROD,
      AArch64::AEK_FP,            AArch64::AEK_SIMD,          AArch64::AEK_FP16,
      AArch64::AEK_FP16FML,       AArch64::AEK_PROFILE,       AArch64::AEK_RAS,
      AArch64::AEK_SVE,           AArch64::AEK_SVE2,          AArch64::AEK_SVE2AES,
      AArch64::AEK_SVE2SM4,       AArch64::AEK_SVE2SHA3,      AArch64::AEK_SVE2BITPERM,
      AArch64::AEK_RCPC,          AArch64::AEK_RAND,          AArch64::AEK_MTE,
      AArch64::AEK_SSBS,          AArch64::AEK_SB,            AArch64::AEK_PREDRES,
      AArch64::AEK_BF16,          AArch64::AEK_I8MM,          AArch64::AEK_F32MM,
      AArch64::AEK_F64MM,         AArch64::AEK_TME,           AArch64::AEK_LS64,
      AArch64::AEK_BRBE,          AArch64::AEK_PAUTH,         AArch64::AEK_FLAGM,
      AArch64::AEK_SME,           AArch64::AEK_SMEF64F64,     AArch64::AEK_SMEI16I64,
      AArch64::AEK_SME2,          AArch64::AEK_HBC,           AArch64::AEK_MOPS,
      AArch64::AEK_PERFMON,       AArch64::AEK_SVE2p1,        AArch64::AEK_SME2p1,
      AArch64::AEK_B16B16,        AArch64::AEK_SMEF16F16,     AArch64::AEK_CSSC,
      AArch64::AEK_RCPC3,         AArch64::AEK_THE,           AArch64::AEK_D128,
      AArch64::AEK_LSE128,        AArch64::AEK_SPECRES2,      AArch64::AEK_RASv2,
      AArch64::AEK_ITE,           AArch64::AEK_GCS,
  };

  std::vector<StringRef> Features;

  BitVector ExtVal(AArch64::AEK_EXTENTIONS_NUM);
  for (auto Ext : Extensions)
    ExtVal.set(Ext);

  // NONE has no feature names.
  // We return True here because NONE is a valid choice.
  EXPECT_TRUE(AArch64::getExtensionFeatures(BitVector(AArch64::AEK_EXTENTIONS_NUM).set(AArch64::AEK_NONE), Features));
  EXPECT_TRUE(!Features.size());

  AArch64::getExtensionFeatures(ExtVal, Features);
  EXPECT_EQ(Extensions.size(), Features.size());

  EXPECT_TRUE(llvm::is_contained(Features, "+crc"));
  EXPECT_TRUE(llvm::is_contained(Features, "+lse"));
  EXPECT_TRUE(llvm::is_contained(Features, "+rdm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+crypto"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sm4"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sha3"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sha2"));
  EXPECT_TRUE(llvm::is_contained(Features, "+aes"));
  EXPECT_TRUE(llvm::is_contained(Features, "+dotprod"));
  EXPECT_TRUE(llvm::is_contained(Features, "+fp-armv8"));
  EXPECT_TRUE(llvm::is_contained(Features, "+neon"));
  EXPECT_TRUE(llvm::is_contained(Features, "+fullfp16"));
  EXPECT_TRUE(llvm::is_contained(Features, "+fp16fml"));
  EXPECT_TRUE(llvm::is_contained(Features, "+spe"));
  EXPECT_TRUE(llvm::is_contained(Features, "+ras"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2-aes"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2-sm4"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2-sha3"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2-bitperm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2p1"));
  EXPECT_TRUE(llvm::is_contained(Features, "+b16b16"));
  EXPECT_TRUE(llvm::is_contained(Features, "+rcpc"));
  EXPECT_TRUE(llvm::is_contained(Features, "+rand"));
  EXPECT_TRUE(llvm::is_contained(Features, "+mte"));
  EXPECT_TRUE(llvm::is_contained(Features, "+ssbs"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sb"));
  EXPECT_TRUE(llvm::is_contained(Features, "+predres"));
  EXPECT_TRUE(llvm::is_contained(Features, "+bf16"));
  EXPECT_TRUE(llvm::is_contained(Features, "+i8mm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+f32mm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+f64mm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+tme"));
  EXPECT_TRUE(llvm::is_contained(Features, "+ls64"));
  EXPECT_TRUE(llvm::is_contained(Features, "+brbe"));
  EXPECT_TRUE(llvm::is_contained(Features, "+pauth"));
  EXPECT_TRUE(llvm::is_contained(Features, "+flagm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-f64f64"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-i16i64"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-f16f16"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme2"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme2p1"));
  EXPECT_TRUE(llvm::is_contained(Features, "+hbc"));
  EXPECT_TRUE(llvm::is_contained(Features, "+mops"));
  EXPECT_TRUE(llvm::is_contained(Features, "+perfmon"));
  EXPECT_TRUE(llvm::is_contained(Features, "+cssc"));
  EXPECT_TRUE(llvm::is_contained(Features, "+rcpc3"));
  EXPECT_TRUE(llvm::is_contained(Features, "+the"));
  EXPECT_TRUE(llvm::is_contained(Features, "+d128"));
  EXPECT_TRUE(llvm::is_contained(Features, "+lse128"));
  EXPECT_TRUE(llvm::is_contained(Features, "+specres2"));
  EXPECT_TRUE(llvm::is_contained(Features, "+ite"));
  EXPECT_TRUE(llvm::is_contained(Features, "+gcs"));

  // Assuming we listed every extension above, this should produce the same
  // result. (note that AEK_NONE doesn't have a name so it won't be in the
  // result despite its bit being set)
  std::vector<StringRef> AllFeatures;
  EXPECT_TRUE(AArch64::getExtensionFeatures(BitVector(AArch64::AEK_EXTENTIONS_NUM, true), AllFeatures));
  EXPECT_THAT(Features, ::testing::ContainerEq(AllFeatures));
}

TEST(TargetParserTest, AArch64ArchFeatures) {
  EXPECT_EQ(AArch64::ARMV8A.ArchFeature, "+v8a");
  EXPECT_EQ(AArch64::ARMV8_1A.ArchFeature, "+v8.1a");
  EXPECT_EQ(AArch64::ARMV8_2A.ArchFeature, "+v8.2a");
  EXPECT_EQ(AArch64::ARMV8_3A.ArchFeature, "+v8.3a");
  EXPECT_EQ(AArch64::ARMV8_4A.ArchFeature, "+v8.4a");
  EXPECT_EQ(AArch64::ARMV8_5A.ArchFeature, "+v8.5a");
  EXPECT_EQ(AArch64::ARMV8_6A.ArchFeature, "+v8.6a");
  EXPECT_EQ(AArch64::ARMV8_7A.ArchFeature, "+v8.7a");
  EXPECT_EQ(AArch64::ARMV8_8A.ArchFeature, "+v8.8a");
  EXPECT_EQ(AArch64::ARMV8_9A.ArchFeature, "+v8.9a");
  EXPECT_EQ(AArch64::ARMV9A.ArchFeature, "+v9a");
  EXPECT_EQ(AArch64::ARMV9_1A.ArchFeature, "+v9.1a");
  EXPECT_EQ(AArch64::ARMV9_2A.ArchFeature, "+v9.2a");
  EXPECT_EQ(AArch64::ARMV9_3A.ArchFeature, "+v9.3a");
  EXPECT_EQ(AArch64::ARMV9_4A.ArchFeature, "+v9.4a");
  EXPECT_EQ(AArch64::ARMV8R.ArchFeature, "+v8r");
}

TEST(TargetParserTest, AArch64ArchPartialOrder) {
  for (const auto *A : AArch64::ArchInfos) {
    EXPECT_EQ(*A, *A);

    // v8r has no relation to other valid architectures
    if (*A != AArch64::ARMV8R) {
      EXPECT_FALSE(A->implies(AArch64::ARMV8R));
      EXPECT_FALSE(AArch64::ARMV8R.implies(*A));
    }
  }

  for (const auto *A : {
           &AArch64::ARMV8_1A,
           &AArch64::ARMV8_2A,
           &AArch64::ARMV8_3A,
           &AArch64::ARMV8_4A,
           &AArch64::ARMV8_5A,
           &AArch64::ARMV8_6A,
           &AArch64::ARMV8_7A,
           &AArch64::ARMV8_8A,
           &AArch64::ARMV8_9A,
       })
    EXPECT_TRUE(A->implies(AArch64::ARMV8A));

  for (const auto *A : {&AArch64::ARMV9_1A, &AArch64::ARMV9_2A,
                        &AArch64::ARMV9_3A, &AArch64::ARMV9_4A})
    EXPECT_TRUE(A->implies(AArch64::ARMV9A));

  EXPECT_TRUE(AArch64::ARMV8_1A.implies(AArch64::ARMV8A));
  EXPECT_TRUE(AArch64::ARMV8_2A.implies(AArch64::ARMV8_1A));
  EXPECT_TRUE(AArch64::ARMV8_3A.implies(AArch64::ARMV8_2A));
  EXPECT_TRUE(AArch64::ARMV8_4A.implies(AArch64::ARMV8_3A));
  EXPECT_TRUE(AArch64::ARMV8_5A.implies(AArch64::ARMV8_4A));
  EXPECT_TRUE(AArch64::ARMV8_6A.implies(AArch64::ARMV8_5A));
  EXPECT_TRUE(AArch64::ARMV8_7A.implies(AArch64::ARMV8_6A));
  EXPECT_TRUE(AArch64::ARMV8_8A.implies(AArch64::ARMV8_7A));
  EXPECT_TRUE(AArch64::ARMV8_9A.implies(AArch64::ARMV8_8A));

  EXPECT_TRUE(AArch64::ARMV9_1A.implies(AArch64::ARMV9A));
  EXPECT_TRUE(AArch64::ARMV9_2A.implies(AArch64::ARMV9_1A));
  EXPECT_TRUE(AArch64::ARMV9_3A.implies(AArch64::ARMV9_2A));
  EXPECT_TRUE(AArch64::ARMV9_4A.implies(AArch64::ARMV9_3A));

  EXPECT_TRUE(AArch64::ARMV9A.implies(AArch64::ARMV8_5A));
  EXPECT_TRUE(AArch64::ARMV9_1A.implies(AArch64::ARMV8_6A));
  EXPECT_TRUE(AArch64::ARMV9_2A.implies(AArch64::ARMV8_7A));
  EXPECT_TRUE(AArch64::ARMV9_3A.implies(AArch64::ARMV8_8A));
  EXPECT_TRUE(AArch64::ARMV9_4A.implies(AArch64::ARMV8_9A));
}

TEST(TargetParserTest, AArch64ArchExtFeature) {
  const char *ArchExt[][4] = {
      {"crc", "nocrc", "+crc", "-crc"},
      {"crypto", "nocrypto", "+crypto", "-crypto"},
      {"flagm", "noflagm", "+flagm", "-flagm"},
      {"fp", "nofp", "+fp-armv8", "-fp-armv8"},
      {"simd", "nosimd", "+neon", "-neon"},
      {"fp16", "nofp16", "+fullfp16", "-fullfp16"},
      {"fp16fml", "nofp16fml", "+fp16fml", "-fp16fml"},
      {"profile", "noprofile", "+spe", "-spe"},
      {"ras", "noras", "+ras", "-ras"},
      {"lse", "nolse", "+lse", "-lse"},
      {"rdm", "nordm", "+rdm", "-rdm"},
      {"sve", "nosve", "+sve", "-sve"},
      {"sve2", "nosve2", "+sve2", "-sve2"},
      {"sve2-aes", "nosve2-aes", "+sve2-aes", "-sve2-aes"},
      {"sve2-sm4", "nosve2-sm4", "+sve2-sm4", "-sve2-sm4"},
      {"sve2-sha3", "nosve2-sha3", "+sve2-sha3", "-sve2-sha3"},
      {"sve2p1", "nosve2p1", "+sve2p1", "-sve2p1"},
      {"b16b16", "nob16b16", "+b16b16", "-b16b16"},
      {"sve2-bitperm", "nosve2-bitperm", "+sve2-bitperm", "-sve2-bitperm"},
      {"dotprod", "nodotprod", "+dotprod", "-dotprod"},
      {"rcpc", "norcpc", "+rcpc", "-rcpc"},
      {"rng", "norng", "+rand", "-rand"},
      {"memtag", "nomemtag", "+mte", "-mte"},
      {"tme", "notme", "+tme", "-tme"},
      {"pauth", "nopauth", "+pauth", "-pauth"},
      {"ssbs", "nossbs", "+ssbs", "-ssbs"},
      {"sb", "nosb", "+sb", "-sb"},
      {"predres", "nopredres", "+predres", "-predres"},
      {"i8mm", "noi8mm", "+i8mm", "-i8mm"},
      {"f32mm", "nof32mm", "+f32mm", "-f32mm"},
      {"f64mm", "nof64mm", "+f64mm", "-f64mm"},
      {"sme", "nosme", "+sme", "-sme"},
      {"sme-f64f64", "nosme-f64f64", "+sme-f64f64", "-sme-f64f64"},
      {"sme-i16i64", "nosme-i16i64", "+sme-i16i64", "-sme-i16i64"},
      {"sme-f16f16", "nosme-f16f16", "+sme-f16f16", "-sme-f16f16"},
      {"sme2", "nosme2", "+sme2", "-sme2"},
      {"sme2p1", "nosme2p1", "+sme2p1", "-sme2p1"},
      {"hbc", "nohbc", "+hbc", "-hbc"},
      {"mops", "nomops", "+mops", "-mops"},
      {"pmuv3", "nopmuv3", "+perfmon", "-perfmon"},
      {"predres2", "nopredres2", "+specres2", "-specres2"},
      {"rasv2", "norasv2", "+rasv2", "-rasv2"},
      {"gcs", "nogcs", "+gcs", "-gcs"},
  };

  for (unsigned i = 0; i < std::size(ArchExt); i++) {
    EXPECT_EQ(StringRef(ArchExt[i][2]),
              AArch64::getArchExtFeature(ArchExt[i][0]));
    EXPECT_EQ(StringRef(ArchExt[i][3]),
              AArch64::getArchExtFeature(ArchExt[i][1]));
  }
}

} // namespace
