//===----------- TargetParser.cpp - Target Parser -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/TargetParser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/ARMTargetParserCommon.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>
#include <sstream>
#include <string>

using namespace llvm;

using ::testing::Contains;
using ::testing::StrEq;

namespace {
const char *ARMArch[] = {
    "armv4",       "armv4t",    "armv5",        "armv5t",      "armv5e",
    "armv5te",     "armv5tej",  "armv6",        "armv6j",      "armv6k",
    "armv6hl",     "armv6t2",   "armv6kz",      "armv6z",      "armv6zk",
    "armv6-m",     "armv6m",    "armv6sm",      "armv6s-m",    "armv7-a",
    "armv7",       "armv7a",    "armv7ve",      "armv7hl",     "armv7l",
    "armv7-r",     "armv7r",    "armv7-m",      "armv7m",      "armv7k",
    "armv7s",      "armv7e-m",  "armv7em",      "armv8-a",     "armv8",
    "armv8a",      "armv8l",    "armv8.1-a",    "armv8.1a",    "armv8.2-a",
    "armv8.2a",    "armv8.3-a", "armv8.3a",     "armv8.4-a",   "armv8.4a",
    "armv8.5-a",   "armv8.5a",  "armv8.6-a",    "armv8.6a",    "armv8.7-a",
    "armv8.7a",    "armv8.8-a", "armv8.8a",     "armv8.9-a",   "armv8.9a",
    "armv8-r",     "armv8r",    "armv8-m.base", "armv8m.base", "armv8-m.main",
    "armv8m.main", "iwmmxt",    "iwmmxt2",      "xscale",      "armv8.1-m.main",
    "armv9-a",     "armv9",     "armv9a",       "armv9.1-a",   "armv9.1a",
    "armv9.2-a",   "armv9.2a",  "armv9.3-a",    "armv9.3a",    "armv9.4-a",
    "armv9.4a",    "armv9.5-a", "armv9.5a",     "armv9.6a",    "armv9.6-a",
    "armv9.7a",    "armv9.7-a",
};

std::string FormatExtensionFlags(int64_t Flags) {
  std::vector<StringRef> Features;

  if (Flags & ARM::AEK_NONE)
    Features.push_back("none");
  ARM::getExtensionFeatures(Flags, Features);

  // The target parser also includes every extension you don't have.
  // E.g. if AEK_CRC is not set then it adds "-crc". Not useful here.
  llvm::erase_if(
      Features, [](StringRef extension) { return extension.starts_with("-"); });

  return llvm::join(Features, ", ");
}

std::string SerializeExtensionFlags(AArch64::ExtensionBitset Flags) {
  std::string SerializedFlags;
  std::ostringstream ss;
  int HexValue = 0;
  for (unsigned int i = 0; i < AArch64::AEK_NUM_EXTENSIONS; i++) {
    HexValue <<= 1;
    HexValue |= (int)Flags[i];
    if ((i + 1) % 4 == 0) {
      ss << std::hex << HexValue;
      HexValue = 0;
    }
  }
  // check if there are remainig unhandled bits
  if ((AArch64::AEK_NUM_EXTENSIONS % 4) != 0)
    ss << std::hex << HexValue;

  SerializedFlags = ss.str();
  return SerializedFlags;
}

template <ARM::ISAKind ISAKind> struct AssertSameExtensionFlags {
  AssertSameExtensionFlags(StringRef CPUName) : CPUName(CPUName) {}

  testing::AssertionResult operator()(const char *m_expr, const char *n_expr,
                                      uint64_t ExpectedFlags,
                                      uint64_t GotFlags) {
    if (ExpectedFlags == GotFlags)
      return testing::AssertionSuccess();

    return testing::AssertionFailure()
           << llvm::formatv("CPU: {4}\n"
                            "Expected extension flags: {0} ({1:x})\n"
                            "     Got extension flags: {2} ({3:x})\n"
                            "                    Diff: {5} ({6:x})\n",
                            FormatExtensionFlags(ExpectedFlags), ExpectedFlags,
                            FormatExtensionFlags(GotFlags), GotFlags, CPUName,
                            FormatExtensionFlags(ExpectedFlags ^ GotFlags));
  }

private:
  StringRef CPUName;
};

template <typename T> struct ARMCPUTestParams {
  ARMCPUTestParams(StringRef CPUName, StringRef ExpectedArch,
                   StringRef ExpectedFPU, T ExpectedFlags, StringRef CPUAttr)
      : CPUName(CPUName), ExpectedArch(ExpectedArch), ExpectedFPU(ExpectedFPU),
        ExpectedFlags(ExpectedFlags), CPUAttr(CPUAttr) {}

  friend std::ostream &operator<<(std::ostream &os,
                                  const ARMCPUTestParams<T> &params) {
    os << "\"" << params.CPUName.str() << "\", \"" << params.ExpectedArch.str()
       << "\", \"" << params.ExpectedFPU.str() << "\", 0x";
    if constexpr (std::is_same<T, uint64_t>::value)
      os << std::hex << params.ExpectedFlags;
    else
      os << SerializeExtensionFlags(params.ExpectedFlags);
    os << ", \"" << params.CPUAttr.str() << "\"";
    return os;
  }

  /// Print a gtest-compatible facsimile of the CPUName, to make the test's name
  /// human-readable.
  ///
  /// https://github.com/google/googletest/blob/main/docs/advanced.md#specifying-names-for-value-parameterized-test-parameters
  static std::string PrintToStringParamName(
      const testing::TestParamInfo<ARMCPUTestParams<T>> &Info) {
    std::string Name = Info.param.CPUName.str();
    for (char &C : Name)
      if (!std::isalnum(C))
        C = '_';
    return Name;
  }

  StringRef CPUName;
  StringRef ExpectedArch;
  StringRef ExpectedFPU;
  T ExpectedFlags;
  StringRef CPUAttr;
};

class ARMCPUTestFixture
    : public ::testing::TestWithParam<ARMCPUTestParams<uint64_t>> {};

TEST_P(ARMCPUTestFixture, ARMCPUTests) {
  auto params = GetParam();

  ARM::ArchKind AK = ARM::parseCPUArch(params.CPUName);
  EXPECT_EQ(params.ExpectedArch, ARM::getArchName(AK));

  ARM::FPUKind FPUKind = ARM::getDefaultFPU(params.CPUName, AK);
  EXPECT_EQ(params.ExpectedFPU, ARM::getFPUName(FPUKind));

  uint64_t default_extensions = ARM::getDefaultExtensions(params.CPUName, AK);
  EXPECT_PRED_FORMAT2(
      AssertSameExtensionFlags<ARM::ISAKind::ARM>(params.CPUName),
      params.ExpectedFlags, default_extensions);

  EXPECT_EQ(params.CPUAttr, ARM::getCPUAttr(AK));
}

// Note that we include ARM::AEK_NONE even when there are other extensions
// we expect. This is because the default extensions for a CPU are the sum
// of the default extensions for its architecture and for the CPU.
// So if a CPU has no extra extensions, it adds AEK_NONE.
INSTANTIATE_TEST_SUITE_P(
    ARMCPUTestsPart1, ARMCPUTestFixture,
    ::testing::Values(
        ARMCPUTestParams<uint64_t>("invalid", "invalid", "invalid",
                                   ARM::AEK_NONE, ""),
        ARMCPUTestParams<uint64_t>("generic", "invalid", "none", ARM::AEK_NONE,
                                   ""),

        ARMCPUTestParams<uint64_t>("arm8", "armv4", "none", ARM::AEK_NONE, "4"),
        ARMCPUTestParams<uint64_t>("arm810", "armv4", "none", ARM::AEK_NONE,
                                   "4"),
        ARMCPUTestParams<uint64_t>("strongarm", "armv4", "none", ARM::AEK_NONE,
                                   "4"),
        ARMCPUTestParams<uint64_t>("strongarm110", "armv4", "none",
                                   ARM::AEK_NONE, "4"),
        ARMCPUTestParams<uint64_t>("strongarm1100", "armv4", "none",
                                   ARM::AEK_NONE, "4"),
        ARMCPUTestParams<uint64_t>("strongarm1110", "armv4", "none",
                                   ARM::AEK_NONE, "4"),
        ARMCPUTestParams<uint64_t>("arm7tdmi", "armv4t", "none", ARM::AEK_NONE,
                                   "4T"),
        ARMCPUTestParams<uint64_t>("arm7tdmi-s", "armv4t", "none",
                                   ARM::AEK_NONE, "4T"),
        ARMCPUTestParams<uint64_t>("arm710t", "armv4t", "none", ARM::AEK_NONE,
                                   "4T"),
        ARMCPUTestParams<uint64_t>("arm720t", "armv4t", "none", ARM::AEK_NONE,
                                   "4T"),
        ARMCPUTestParams<uint64_t>("arm9", "armv4t", "none", ARM::AEK_NONE,
                                   "4T"),
        ARMCPUTestParams<uint64_t>("arm9tdmi", "armv4t", "none", ARM::AEK_NONE,
                                   "4T"),
        ARMCPUTestParams<uint64_t>("arm920", "armv4t", "none", ARM::AEK_NONE,
                                   "4T"),
        ARMCPUTestParams<uint64_t>("arm920t", "armv4t", "none", ARM::AEK_NONE,
                                   "4T"),
        ARMCPUTestParams<uint64_t>("arm922t", "armv4t", "none", ARM::AEK_NONE,
                                   "4T"),
        ARMCPUTestParams<uint64_t>("arm940t", "armv4t", "none", ARM::AEK_NONE,
                                   "4T"),
        ARMCPUTestParams<uint64_t>("ep9312", "armv4t", "none", ARM::AEK_NONE,
                                   "4T"),
        ARMCPUTestParams<uint64_t>("arm10tdmi", "armv5t", "none", ARM::AEK_NONE,
                                   "5T"),
        ARMCPUTestParams<uint64_t>("arm1020t", "armv5t", "none", ARM::AEK_NONE,
                                   "5T"),
        ARMCPUTestParams<uint64_t>("arm9e", "armv5te", "none",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "5TE"),
        ARMCPUTestParams<uint64_t>("arm946e-s", "armv5te", "none",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "5TE"),
        ARMCPUTestParams<uint64_t>("arm966e-s", "armv5te", "none",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "5TE"),
        ARMCPUTestParams<uint64_t>("arm968e-s", "armv5te", "none",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "5TE"),
        ARMCPUTestParams<uint64_t>("arm10e", "armv5te", "none",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "5TE"),
        ARMCPUTestParams<uint64_t>("arm1020e", "armv5te", "none",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "5TE"),
        ARMCPUTestParams<uint64_t>("arm1022e", "armv5te", "none",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "5TE"),
        ARMCPUTestParams<uint64_t>("arm926ej-s", "armv5tej", "none",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "5TEJ"),
        ARMCPUTestParams<uint64_t>("arm1136j-s", "armv6", "none",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "6"),
        ARMCPUTestParams<uint64_t>("arm1136jf-s", "armv6", "vfpv2",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "6"),
        ARMCPUTestParams<uint64_t>("arm1176jz-s", "armv6kz", "none",
                                   ARM::AEK_NONE | ARM::AEK_SEC | ARM::AEK_DSP,
                                   "6KZ"),
        ARMCPUTestParams<uint64_t>("mpcore", "armv6k", "vfpv2",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "6K"),
        ARMCPUTestParams<uint64_t>("mpcorenovfp", "armv6k", "none",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "6K"),
        ARMCPUTestParams<uint64_t>("arm1176jzf-s", "armv6kz", "vfpv2",
                                   ARM::AEK_NONE | ARM::AEK_SEC | ARM::AEK_DSP,
                                   "6KZ"),
        ARMCPUTestParams<uint64_t>("arm1156t2-s", "armv6t2", "none",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "6T2"),
        ARMCPUTestParams<uint64_t>("arm1156t2f-s", "armv6t2", "vfpv2",
                                   ARM::AEK_NONE | ARM::AEK_DSP, "6T2"),
        ARMCPUTestParams<uint64_t>("cortex-m0", "armv6-m", "none",
                                   ARM::AEK_NONE, "6-M"),
        ARMCPUTestParams<uint64_t>("cortex-m0plus", "armv6-m", "none",
                                   ARM::AEK_NONE, "6-M"),
        ARMCPUTestParams<uint64_t>("cortex-m1", "armv6-m", "none",
                                   ARM::AEK_NONE, "6-M"),
        ARMCPUTestParams<uint64_t>("sc000", "armv6-m", "none", ARM::AEK_NONE,
                                   "6-M"),
        ARMCPUTestParams<uint64_t>("cortex-a5", "armv7-a", "neon-vfpv4",
                                   ARM::AEK_MP | ARM::AEK_SEC | ARM::AEK_DSP,
                                   "7-A"),
        ARMCPUTestParams<uint64_t>("cortex-a7", "armv7-a", "neon-vfpv4",
                                   ARM::AEK_HWDIVTHUMB | ARM::AEK_HWDIVARM |
                                       ARM::AEK_MP | ARM::AEK_SEC |
                                       ARM::AEK_VIRT | ARM::AEK_DSP,
                                   "7-A"),
        ARMCPUTestParams<uint64_t>("cortex-a8", "armv7-a", "neon",
                                   ARM::AEK_SEC | ARM::AEK_DSP, "7-A")),
    ARMCPUTestParams<uint64_t>::PrintToStringParamName);

// gtest in llvm has a limit of 50 test cases when using ::Values so we split
// them into 2 blocks
INSTANTIATE_TEST_SUITE_P(
    ARMCPUTestsPart2, ARMCPUTestFixture,
    ::testing::Values(
        ARMCPUTestParams<uint64_t>("cortex-a9", "armv7-a", "neon-fp16",
                                   ARM::AEK_MP | ARM::AEK_SEC | ARM::AEK_DSP,
                                   "7-A"),
        ARMCPUTestParams<uint64_t>("cortex-a12", "armv7-a", "neon-vfpv4",
                                   ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                                       ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                                       ARM::AEK_DSP,
                                   "7-A"),
        ARMCPUTestParams<uint64_t>("cortex-a15", "armv7-a", "neon-vfpv4",
                                   ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                                       ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                                       ARM::AEK_DSP,
                                   "7-A"),
        ARMCPUTestParams<uint64_t>("cortex-a17", "armv7-a", "neon-vfpv4",
                                   ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                                       ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                                       ARM::AEK_DSP,
                                   "7-A"),
        ARMCPUTestParams<uint64_t>("krait", "armv7-a", "neon-vfpv4",
                                   ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                                       ARM::AEK_DSP,
                                   "7-A"),
        ARMCPUTestParams<uint64_t>("cortex-r4", "armv7-r", "none",
                                   ARM::AEK_NONE | ARM::AEK_HWDIVTHUMB |
                                       ARM::AEK_DSP,
                                   "7-R"),
        ARMCPUTestParams<uint64_t>("cortex-r4f", "armv7-r", "vfpv3-d16",
                                   ARM::AEK_NONE | ARM::AEK_HWDIVTHUMB |
                                       ARM::AEK_DSP,
                                   "7-R"),
        ARMCPUTestParams<uint64_t>("cortex-r5", "armv7-r", "vfpv3-d16",
                                   ARM::AEK_MP | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "7-R"),
        ARMCPUTestParams<uint64_t>("cortex-r7", "armv7-r", "vfpv3-d16-fp16",
                                   ARM::AEK_MP | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "7-R"),
        ARMCPUTestParams<uint64_t>("cortex-r8", "armv7-r", "vfpv3-d16-fp16",
                                   ARM::AEK_MP | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "7-R"),
        ARMCPUTestParams<uint64_t>("cortex-r52", "armv8-r", "neon-fp-armv8",
                                   ARM::AEK_NONE | ARM::AEK_CRC | ARM::AEK_MP |
                                       ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-R"),
        ARMCPUTestParams<uint64_t>("cortex-r52plus", "armv8-r", "neon-fp-armv8",
                                   ARM::AEK_NONE | ARM::AEK_CRC | ARM::AEK_MP |
                                       ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-R"),
        ARMCPUTestParams<uint64_t>("sc300", "armv7-m", "none",
                                   ARM::AEK_NONE | ARM::AEK_HWDIVTHUMB, "7-M"),
        ARMCPUTestParams<uint64_t>("cortex-m3", "armv7-m", "none",
                                   ARM::AEK_NONE | ARM::AEK_HWDIVTHUMB, "7-M"),
        ARMCPUTestParams<uint64_t>("cortex-m4", "armv7e-m", "fpv4-sp-d16",
                                   ARM::AEK_NONE | ARM::AEK_HWDIVTHUMB |
                                       ARM::AEK_DSP,
                                   "7E-M"),
        ARMCPUTestParams<uint64_t>("cortex-m7", "armv7e-m", "fpv5-d16",
                                   ARM::AEK_NONE | ARM::AEK_HWDIVTHUMB |
                                       ARM::AEK_DSP,
                                   "7E-M"),
        ARMCPUTestParams<uint64_t>("cortex-a32", "armv8-a",
                                   "crypto-neon-fp-armv8",
                                   ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                                       ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-A"),
        ARMCPUTestParams<uint64_t>("cortex-a35", "armv8-a",
                                   "crypto-neon-fp-armv8",
                                   ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                                       ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-A"),
        ARMCPUTestParams<uint64_t>("cortex-a53", "armv8-a",
                                   "crypto-neon-fp-armv8",
                                   ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                                       ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-A"),
        ARMCPUTestParams<uint64_t>(
            "cortex-a55", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP |
                ARM::AEK_FP16 | ARM::AEK_RAS | ARM::AEK_DOTPROD,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>("cortex-a57", "armv8-a",
                                   "crypto-neon-fp-armv8",
                                   ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                                       ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-A"),
        ARMCPUTestParams<uint64_t>("cortex-a72", "armv8-a",
                                   "crypto-neon-fp-armv8",
                                   ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                                       ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-A"),
        ARMCPUTestParams<uint64_t>("cortex-a73", "armv8-a",
                                   "crypto-neon-fp-armv8",
                                   ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                                       ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-A"),
        ARMCPUTestParams<uint64_t>(
            "cortex-a75", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP |
                ARM::AEK_FP16 | ARM::AEK_RAS | ARM::AEK_DOTPROD,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>(
            "cortex-a76", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP |
                ARM::AEK_FP16 | ARM::AEK_RAS | ARM::AEK_DOTPROD,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>(
            "cortex-a76ae", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP |
                ARM::AEK_FP16 | ARM::AEK_RAS | ARM::AEK_DOTPROD,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>(
            "cortex-a78c", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP | ARM::AEK_CRC |
                ARM::AEK_RAS | ARM::AEK_FP16 | ARM::AEK_DOTPROD,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>("cortex-a510", "armv9-a", "neon-fp-armv8",
                                   ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                                       ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                                       ARM::AEK_DSP | ARM::AEK_CRC |
                                       ARM::AEK_RAS | ARM::AEK_DOTPROD |
                                       ARM::AEK_FP16FML | ARM::AEK_BF16 |
                                       ARM::AEK_I8MM | ARM::AEK_SB,
                                   "9-A"),
        ARMCPUTestParams<uint64_t>("cortex-a710", "armv9-a", "neon-fp-armv8",
                                   ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                                       ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                                       ARM::AEK_DSP | ARM::AEK_CRC |
                                       ARM::AEK_RAS | ARM::AEK_DOTPROD |
                                       ARM::AEK_FP16FML | ARM::AEK_BF16 |
                                       ARM::AEK_I8MM | ARM::AEK_SB,
                                   "9-A"),
        ARMCPUTestParams<uint64_t>(
            "cortex-a77", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP |
                ARM::AEK_FP16 | ARM::AEK_RAS | ARM::AEK_DOTPROD,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>(
            "cortex-a78", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_DOTPROD | ARM::AEK_FP16 | ARM::AEK_SEC | ARM::AEK_MP |
                ARM::AEK_VIRT | ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                ARM::AEK_DSP | ARM::AEK_CRC | ARM::AEK_RAS,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>(
            "cortex-a78ae", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_RAS | ARM::AEK_DOTPROD | ARM::AEK_SEC | ARM::AEK_MP |
                ARM::AEK_VIRT | ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                ARM::AEK_DSP | ARM::AEK_CRC | ARM::AEK_RAS,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>(
            "cortex-x1", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_RAS | ARM::AEK_FP16 | ARM::AEK_DOTPROD | ARM::AEK_SEC |
                ARM::AEK_MP | ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP | ARM::AEK_CRC,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>(
            "cortex-x1c", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_RAS | ARM::AEK_FP16 | ARM::AEK_DOTPROD | ARM::AEK_SEC |
                ARM::AEK_MP | ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP | ARM::AEK_CRC,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>(
            "neoverse-n1", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP |
                ARM::AEK_FP16 | ARM::AEK_RAS | ARM::AEK_DOTPROD,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>(
            "neoverse-n2", "armv9-a", "neon-fp-armv8",
            ARM::AEK_CRC | ARM::AEK_HWDIVTHUMB | ARM::AEK_HWDIVARM |
                ARM::AEK_MP | ARM::AEK_SEC | ARM::AEK_VIRT | ARM::AEK_DSP |
                ARM::AEK_BF16 | ARM::AEK_DOTPROD | ARM::AEK_RAS |
                ARM::AEK_I8MM | ARM::AEK_FP16FML | ARM::AEK_SB,
            "9-A"),
        ARMCPUTestParams<uint64_t>(
            "neoverse-v1", "armv8.4-a", "crypto-neon-fp-armv8",
            ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP | ARM::AEK_CRC |
                ARM::AEK_RAS | ARM::AEK_FP16 | ARM::AEK_BF16 | ARM::AEK_DOTPROD,
            "8.4-A"),
        ARMCPUTestParams<uint64_t>("cyclone", "armv8-a", "crypto-neon-fp-armv8",
                                   ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                                       ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-A"),
        ARMCPUTestParams<uint64_t>("exynos-m3", "armv8-a",
                                   "crypto-neon-fp-armv8",
                                   ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                                       ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                                       ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-A"),
        ARMCPUTestParams<uint64_t>(
            "exynos-m4", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP |
                ARM::AEK_DOTPROD | ARM::AEK_FP16 | ARM::AEK_RAS,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>(
            "exynos-m5", "armv8.2-a", "crypto-neon-fp-armv8",
            ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP |
                ARM::AEK_DOTPROD | ARM::AEK_FP16 | ARM::AEK_RAS,
            "8.2-A"),
        ARMCPUTestParams<uint64_t>("cortex-m23", "armv8-m.base", "none",
                                   ARM::AEK_NONE | ARM::AEK_HWDIVTHUMB,
                                   "8-M.Baseline"),
        ARMCPUTestParams<uint64_t>("cortex-m33", "armv8-m.main", "fpv5-sp-d16",
                                   ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-M.Mainline"),
        ARMCPUTestParams<uint64_t>("star-mc1", "armv8-m.main", "fpv5-sp-d16",
                                   ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-M.Mainline"),
        ARMCPUTestParams<uint64_t>("cortex-m35p", "armv8-m.main", "fpv5-sp-d16",
                                   ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                                   "8-M.Mainline"),
        ARMCPUTestParams<uint64_t>(
            "cortex-m55", "armv8.1-m.main", "fp-armv8-fullfp16-d16",
            ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP | ARM::AEK_MVE | ARM::AEK_FP |
                ARM::AEK_RAS | ARM::AEK_LOB | ARM::AEK_FP16,
            "8.1-M.Mainline"),
        ARMCPUTestParams<uint64_t>(
            "cortex-m85", "armv8.1-m.main", "fp-armv8-fullfp16-d16",
            ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP | ARM::AEK_MVE | ARM::AEK_FP |
                ARM::AEK_RAS | ARM::AEK_LOB | ARM::AEK_FP16 | ARM::AEK_PACBTI,
            "8.1-M.Mainline"),
        ARMCPUTestParams<uint64_t>(
            "cortex-m52", "armv8.1-m.main", "fp-armv8-fullfp16-d16",
            ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP | ARM::AEK_MVE | ARM::AEK_FP |
                ARM::AEK_RAS | ARM::AEK_LOB | ARM::AEK_FP16 | ARM::AEK_PACBTI,
            "8.1-M.Mainline"),
        ARMCPUTestParams<uint64_t>(
            "star-mc3", "armv8.1-m.main", "fp-armv8-fullfp16-d16",
            ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP | ARM::AEK_MVE | ARM::AEK_FP |
                ARM::AEK_RAS | ARM::AEK_LOB | ARM::AEK_FP16 | ARM::AEK_PACBTI,
            "8.1-M.Mainline"),
        ARMCPUTestParams<uint64_t>("iwmmxt", "iwmmxt", "none", ARM::AEK_NONE,
                                   "iwmmxt"),
        ARMCPUTestParams<uint64_t>("xscale", "xscale", "none", ARM::AEK_NONE,
                                   "xscale"),
        ARMCPUTestParams<uint64_t>("swift", "armv7s", "neon-vfpv4",
                                   ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                                       ARM::AEK_DSP,
                                   "7-S")),
    ARMCPUTestParams<uint64_t>::PrintToStringParamName);

static constexpr unsigned NumARMCPUArchs = 95;

TEST(FeatureBitsetTest, Iterator) {
  // Empty bitset yields nothing.
  FeatureBitset Empty;
  EXPECT_EQ(Empty.begin(), Empty.end());
  for (unsigned Index : Empty) {
    (void)Index;
    FAIL() << "empty bitset should yield no set bits";
  }

  // Yields the set indices in order, crossing the 63/64 word boundary.
  FeatureBitset Bits;
  Bits.set(0).set(5).set(63).set(64).set(200);
  std::vector<unsigned> SetIndices;
  for (unsigned Index : Bits)
    SetIndices.push_back(Index);
  EXPECT_EQ(SetIndices, (std::vector<unsigned>{0, 5, 63, 64, 200}));

  // Every yielded index is set, and the count matches.
  for (unsigned Index : Bits)
    EXPECT_TRUE(Bits[Index]);
  EXPECT_EQ(SetIndices.size(), Bits.count());

  // The final position is reached.
  FeatureBitset Last;
  unsigned LastIndex = Last.size() - 1;
  Last.set(LastIndex);
  SetIndices.assign(Last.begin(), Last.end());
  EXPECT_EQ(SetIndices, (std::vector<unsigned>{LastIndex}));
}

TEST(TargetParserTest, testARMCPUArchList) {
  SmallVector<StringRef, NumARMCPUArchs> List;
  ARM::fillValidCPUArchList(List);

  // No list exists for these in this test suite, so ensure all are
  // valid, and match the expected 'magic' count.
  EXPECT_EQ(List.size(), NumARMCPUArchs);
  for (StringRef CPU : List) {
    EXPECT_NE(ARM::parseCPUArch(CPU), ARM::ArchKind::INVALID);
  }
}

TEST(TargetParserTest, testInvalidARMArch) {
  auto InvalidArchStrings = {"armv", "armv99", "noarm"};
  for (const char *InvalidArch : InvalidArchStrings)
    EXPECT_EQ(ARM::parseArch(InvalidArch), ARM::ArchKind::INVALID);
}

bool testARMArch(StringRef Arch, StringRef DefaultCPU, StringRef SubArch,
                 unsigned ArchAttr) {
  ARM::ArchKind AK = ARM::parseArch(Arch);
  bool Result = (AK != ARM::ArchKind::INVALID);
  Result &= ARM::getDefaultCPU(Arch) == DefaultCPU;
  Result &= ARM::getSubArch(AK) == SubArch;
  Result &= (ARM::getArchAttr(AK) == ArchAttr);
  return Result;
}

TEST(TargetParserTest, testARMArch) {
  EXPECT_TRUE(
      testARMArch("armv4", "strongarm", "v4", ARMBuildAttrs::CPUArch::v4));
  EXPECT_TRUE(
      testARMArch("armv4t", "arm7tdmi", "v4t", ARMBuildAttrs::CPUArch::v4T));
  EXPECT_TRUE(
      testARMArch("armv5t", "arm10tdmi", "v5", ARMBuildAttrs::CPUArch::v5T));
  EXPECT_TRUE(
      testARMArch("armv5te", "arm1022e", "v5e", ARMBuildAttrs::CPUArch::v5TE));
  EXPECT_TRUE(testARMArch("armv5tej", "arm926ej-s", "v5e",
                          ARMBuildAttrs::CPUArch::v5TEJ));
  EXPECT_TRUE(
      testARMArch("armv6", "arm1136jf-s", "v6", ARMBuildAttrs::CPUArch::v6));
  EXPECT_TRUE(
      testARMArch("armv6k", "mpcore", "v6k", ARMBuildAttrs::CPUArch::v6K));
  EXPECT_TRUE(testARMArch("armv6t2", "arm1156t2-s", "v6t2",
                          ARMBuildAttrs::CPUArch::v6T2));
  EXPECT_TRUE(testARMArch("armv6kz", "arm1176jzf-s", "v6kz",
                          ARMBuildAttrs::CPUArch::v6KZ));
  EXPECT_TRUE(
      testARMArch("armv6-m", "cortex-m0", "v6m", ARMBuildAttrs::CPUArch::v6_M));
  EXPECT_TRUE(
      testARMArch("armv7-a", "generic", "v7", ARMBuildAttrs::CPUArch::v7));
  EXPECT_TRUE(
      testARMArch("armv7ve", "generic", "v7ve", ARMBuildAttrs::CPUArch::v7));
  EXPECT_TRUE(
      testARMArch("armv7-r", "cortex-r4", "v7r", ARMBuildAttrs::CPUArch::v7));
  EXPECT_TRUE(
      testARMArch("armv7-m", "cortex-m3", "v7m", ARMBuildAttrs::CPUArch::v7));
  EXPECT_TRUE(testARMArch("armv7e-m", "cortex-m4", "v7em",
                          ARMBuildAttrs::CPUArch::v7E_M));
  EXPECT_TRUE(
      testARMArch("armv8-a", "generic", "v8a", ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testARMArch("armv8.1-a", "generic", "v8.1a",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testARMArch("armv8.2-a", "generic", "v8.2a",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testARMArch("armv8.3-a", "generic", "v8.3a",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testARMArch("armv8.4-a", "generic", "v8.4a",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testARMArch("armv8.5-a", "generic", "v8.5a",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testARMArch("armv8.6-a", "generic", "v8.6a",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testARMArch("armv8.7-a", "generic", "v8.7a",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testARMArch("armv8.8-a", "generic", "v8.8a",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testARMArch("armv8.9-a", "generic", "v8.9a",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(
      testARMArch("armv9-a", "generic", "v9a", ARMBuildAttrs::CPUArch::v9_A));
  EXPECT_TRUE(testARMArch("armv9.1-a", "generic", "v9.1a",
                          ARMBuildAttrs::CPUArch::v9_A));
  EXPECT_TRUE(testARMArch("armv9.2-a", "generic", "v9.2a",
                          ARMBuildAttrs::CPUArch::v9_A));
  EXPECT_TRUE(testARMArch("armv9.3-a", "generic", "v9.3a",
                          ARMBuildAttrs::CPUArch::v9_A));
  EXPECT_TRUE(testARMArch("armv9.4-a", "generic", "v9.4a",
                          ARMBuildAttrs::CPUArch::v9_A));
  EXPECT_TRUE(testARMArch("armv9.5-a", "generic", "v9.5a",
                          ARMBuildAttrs::CPUArch::v9_A));
  EXPECT_TRUE(testARMArch("armv9.6-a", "generic", "v9.6a",
                          ARMBuildAttrs::CPUArch::v9_A));
  EXPECT_TRUE(testARMArch("armv9.7-a", "generic", "v9.7a",
                          ARMBuildAttrs::CPUArch::v9_A));
  EXPECT_TRUE(
      testARMArch("armv8-r", "generic", "v8r", ARMBuildAttrs::CPUArch::v8_R));
  EXPECT_TRUE(testARMArch("armv8-m.base", "generic", "v8m.base",
                          ARMBuildAttrs::CPUArch::v8_M_Base));
  EXPECT_TRUE(testARMArch("armv8-m.main", "generic", "v8m.main",
                          ARMBuildAttrs::CPUArch::v8_M_Main));
  EXPECT_TRUE(testARMArch("armv8.1-m.main", "generic", "v8.1m.main",
                          ARMBuildAttrs::CPUArch::v8_1_M_Main));
  EXPECT_TRUE(
      testARMArch("iwmmxt", "iwmmxt", "", ARMBuildAttrs::CPUArch::v5TE));
  EXPECT_TRUE(
      testARMArch("iwmmxt2", "generic", "", ARMBuildAttrs::CPUArch::v5TE));
  EXPECT_TRUE(
      testARMArch("xscale", "xscale", "v5e", ARMBuildAttrs::CPUArch::v5TE));
  EXPECT_TRUE(
      testARMArch("armv7s", "swift", "v7s", ARMBuildAttrs::CPUArch::v7));
  EXPECT_TRUE(
      testARMArch("armv7k", "generic", "v7k", ARMBuildAttrs::CPUArch::v7));
}

bool testARMExtension(StringRef CPUName, ARM::ArchKind ArchKind,
                      StringRef ArchExt) {
  return ARM::getDefaultExtensions(CPUName, ArchKind) &
         ARM::parseArchExt(ArchExt);
}

TEST(TargetParserTest, testARMExtension) {
  EXPECT_FALSE(testARMExtension("strongarm", ARM::ArchKind::INVALID, "dsp"));
  EXPECT_FALSE(testARMExtension("arm7tdmi", ARM::ArchKind::INVALID, "dsp"));
  EXPECT_FALSE(testARMExtension("arm10tdmi", ARM::ArchKind::INVALID, "simd"));
  EXPECT_FALSE(testARMExtension("arm1022e", ARM::ArchKind::INVALID, "simd"));
  EXPECT_FALSE(testARMExtension("arm926ej-s", ARM::ArchKind::INVALID, "simd"));
  EXPECT_FALSE(
      testARMExtension("arm1136jf-s", ARM::ArchKind::INVALID, "crypto"));
  EXPECT_FALSE(
      testARMExtension("arm1156t2-s", ARM::ArchKind::INVALID, "crypto"));
  EXPECT_FALSE(
      testARMExtension("arm1176jzf-s", ARM::ArchKind::INVALID, "crypto"));
  EXPECT_FALSE(testARMExtension("cortex-m0", ARM::ArchKind::INVALID, "crypto"));
  EXPECT_FALSE(testARMExtension("cortex-a8", ARM::ArchKind::INVALID, "crypto"));
  EXPECT_FALSE(testARMExtension("cortex-r4", ARM::ArchKind::INVALID, "crypto"));
  EXPECT_FALSE(testARMExtension("cortex-m3", ARM::ArchKind::INVALID, "crypto"));
  EXPECT_FALSE(testARMExtension("cortex-a53", ARM::ArchKind::INVALID, "ras"));
  EXPECT_FALSE(testARMExtension("cortex-a53", ARM::ArchKind::INVALID, "fp16"));
  EXPECT_TRUE(testARMExtension("cortex-a55", ARM::ArchKind::INVALID, "fp16"));
  EXPECT_FALSE(
      testARMExtension("cortex-a55", ARM::ArchKind::INVALID, "fp16fml"));
  EXPECT_TRUE(testARMExtension("cortex-a75", ARM::ArchKind::INVALID, "fp16"));
  EXPECT_FALSE(
      testARMExtension("cortex-a75", ARM::ArchKind::INVALID, "fp16fml"));
  EXPECT_FALSE(testARMExtension("cortex-r52", ARM::ArchKind::INVALID, "ras"));
  EXPECT_FALSE(
      testARMExtension("cortex-r52plus", ARM::ArchKind::INVALID, "ras"));
  EXPECT_FALSE(testARMExtension("iwmmxt", ARM::ArchKind::INVALID, "crc"));
  EXPECT_FALSE(testARMExtension("xscale", ARM::ArchKind::INVALID, "crc"));
  EXPECT_FALSE(testARMExtension("swift", ARM::ArchKind::INVALID, "crc"));

  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV4, "dsp"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV4T, "dsp"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV5T, "simd"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV5TE, "simd"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV5TEJ, "simd"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV6, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV6K, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV6T2, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV6KZ, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV6M, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV7A, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV7R, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV7M, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV7EM, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV8A, "ras"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV8_1A, "ras"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV8_2A, "profile"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV8_2A, "fp16"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV8_2A, "fp16fml"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV8_3A, "fp16"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV8_3A, "fp16fml"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV8_4A, "fp16"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV8_4A, "fp16fml"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV8R, "ras"));
  EXPECT_FALSE(
      testARMExtension("generic", ARM::ArchKind::ARMV8MBaseline, "crc"));
  EXPECT_FALSE(
      testARMExtension("generic", ARM::ArchKind::ARMV8MMainline, "crc"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::IWMMXT, "crc"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::IWMMXT2, "crc"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::XSCALE, "crc"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV7S, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::ArchKind::ARMV7K, "crypto"));
}

TEST(TargetParserTest, ARMFPUVersion) {
  for (ARM::FPUKind FK = static_cast<ARM::FPUKind>(0);
       FK <= ARM::FPUKind::FK_LAST;
       FK = static_cast<ARM::FPUKind>(static_cast<unsigned>(FK) + 1))
    if (FK == ARM::FK_LAST || ARM::getFPUName(FK) == "invalid" ||
        ARM::getFPUName(FK) == "none" || ARM::getFPUName(FK) == "softvfp")
      EXPECT_EQ(ARM::FPUVersion::NONE, ARM::getFPUVersion(FK));
    else
      EXPECT_NE(ARM::FPUVersion::NONE, ARM::getFPUVersion(FK));
}

TEST(TargetParserTest, ARMFPUNeonSupportLevel) {
  for (ARM::FPUKind FK = static_cast<ARM::FPUKind>(0);
       FK <= ARM::FPUKind::FK_LAST;
       FK = static_cast<ARM::FPUKind>(static_cast<unsigned>(FK) + 1))
    if (FK == ARM::FK_LAST || !ARM::getFPUName(FK).contains("neon"))
      EXPECT_EQ(ARM::NeonSupportLevel::None, ARM::getFPUNeonSupportLevel(FK));
    else
      EXPECT_NE(ARM::NeonSupportLevel::None, ARM::getFPUNeonSupportLevel(FK));
}

TEST(TargetParserTest, ARMFPURestriction) {
  for (ARM::FPUKind FK = static_cast<ARM::FPUKind>(0);
       FK <= ARM::FPUKind::FK_LAST;
       FK = static_cast<ARM::FPUKind>(static_cast<unsigned>(FK) + 1)) {
    if (FK == ARM::FK_LAST || (!ARM::getFPUName(FK).contains("d16") &&
                               !ARM::getFPUName(FK).contains("vfpv3xd")))
      EXPECT_EQ(ARM::FPURestriction::None, ARM::getFPURestriction(FK));
    else
      EXPECT_NE(ARM::FPURestriction::None, ARM::getFPURestriction(FK));
  }
}

TEST(TargetParserTest, ARMExtensionFeatures) {
  std::map<uint64_t, std::vector<StringRef>> Extensions;

  for (auto &Ext : ARM::ARCHExtNames) {
    if (!Ext.Feature.empty() && !Ext.NegFeature.empty())
      Extensions[Ext.ID] = {Ext.Feature, Ext.NegFeature};
  }

  Extensions[ARM::AEK_HWDIVARM] = {"+hwdiv-arm", "-hwdiv-arm"};
  Extensions[ARM::AEK_HWDIVTHUMB] = {"+hwdiv", "-hwdiv"};

  std::vector<StringRef> Features;

  EXPECT_FALSE(ARM::getExtensionFeatures(ARM::AEK_INVALID, Features));

  for (auto &E : Extensions) {
    // test +extension
    Features.clear();
    ARM::getExtensionFeatures(E.first, Features);
    EXPECT_TRUE(llvm::is_contained(Features, E.second.at(0)));
    EXPECT_EQ(Extensions.size(), Features.size());

    // test -extension
    Features.clear();
    ARM::getExtensionFeatures(~E.first, Features);
    EXPECT_TRUE(llvm::is_contained(Features, E.second.at(1)));
    EXPECT_EQ(Extensions.size(), Features.size());
  }
}

TEST(TargetParserTest, ARMFPUFeatures) {
  std::vector<StringRef> Features;
  for (ARM::FPUKind FK = static_cast<ARM::FPUKind>(0);
       FK <= ARM::FPUKind::FK_LAST;
       FK = static_cast<ARM::FPUKind>(static_cast<unsigned>(FK) + 1)) {
    if (FK == ARM::FK_INVALID || FK >= ARM::FK_LAST)
      EXPECT_FALSE(ARM::getFPUFeatures(FK, Features));
    else
      EXPECT_TRUE(ARM::getFPUFeatures(FK, Features));
  }
}

TEST(TargetParserTest, ARMArchExtFeature) {
  const char *ArchExt[][4] = {{"crc", "nocrc", "+crc", "-crc"},
                              {"crypto", "nocrypto", "+crypto", "-crypto"},
                              {"dsp", "nodsp", "+dsp", "-dsp"},
                              {"fp", "nofp", nullptr, nullptr},
                              {"idiv", "noidiv", nullptr, nullptr},
                              {"mp", "nomp", nullptr, nullptr},
                              {"simd", "nosimd", "+neon", "-neon"},
                              {"sec", "nosec", nullptr, nullptr},
                              {"virt", "novirt", nullptr, nullptr},
                              {"fp16", "nofp16", "+fullfp16", "-fullfp16"},
                              {"fp16fml", "nofp16fml", "+fp16fml", "-fp16fml"},
                              {"ras", "noras", "+ras", "-ras"},
                              {"dotprod", "nodotprod", "+dotprod", "-dotprod"},
                              {"os", "noos", nullptr, nullptr},
                              {"iwmmxt", "noiwmmxt", nullptr, nullptr},
                              {"iwmmxt2", "noiwmmxt2", nullptr, nullptr},
                              {"maverick", "maverick", nullptr, nullptr},
                              {"xscale", "noxscale", nullptr, nullptr},
                              {"sb", "nosb", "+sb", "-sb"},
                              {"i8mm", "noi8mm", "+i8mm", "-i8mm"},
                              {"mve", "nomve", "+mve", "-mve"},
                              {"mve.fp", "nomve.fp", "+mve.fp", "-mve.fp"}};

  for (unsigned i = 0; i < std::size(ArchExt); i++) {
    EXPECT_EQ(StringRef(ArchExt[i][2]), ARM::getArchExtFeature(ArchExt[i][0]));
    EXPECT_EQ(StringRef(ArchExt[i][3]), ARM::getArchExtFeature(ArchExt[i][1]));
  }
}

static bool
testArchExtDependency(const char *ArchExt,
                      const std::initializer_list<const char *> &Expected) {
  std::vector<StringRef> Features;
  ARM::FPUKind FPUKind;

  if (!ARM::appendArchExtFeatures("", ARM::ArchKind::ARMV8_1MMainline, ArchExt,
                                  Features, FPUKind))
    return false;

  return llvm::all_of(Expected, [&](StringRef Ext) {
    return llvm::is_contained(Features, Ext);
  });
}

TEST(TargetParserTest, ARMArchExtDependencies) {
  EXPECT_TRUE(testArchExtDependency("mve", {"+mve", "+dsp"}));
  EXPECT_TRUE(testArchExtDependency("mve.fp", {"+mve.fp", "+mve", "+dsp"}));
  EXPECT_TRUE(testArchExtDependency("nodsp", {"-dsp", "-mve", "-mve.fp"}));
  EXPECT_TRUE(testArchExtDependency("nomve", {"-mve", "-mve.fp"}));
}

TEST(TargetParserTest, ARMparseHWDiv) {
  const char *hwdiv[] = {"thumb", "arm", "arm,thumb", "thumb,arm"};

  for (unsigned i = 0; i < std::size(hwdiv); i++)
    EXPECT_NE(ARM::AEK_INVALID, ARM::parseHWDiv((StringRef)hwdiv[i]));
}

TEST(TargetParserTest, ARMparseArchEndianAndISA) {
  const char *Arch[] = {
      "v2",        "v2a",    "v3",    "v3m",    "v4",       "v4t",
      "v5",        "v5t",    "v5e",   "v5te",   "v5tej",    "v6",
      "v6j",       "v6k",    "v6hl",  "v6t2",   "v6kz",     "v6z",
      "v6zk",      "v6-m",   "v6m",   "v6sm",   "v6s-m",    "v7-a",
      "v7",        "v7a",    "v7ve",  "v7hl",   "v7l",      "v7-r",
      "v7r",       "v7-m",   "v7m",   "v7k",    "v7s",      "v7e-m",
      "v7em",      "v8-a",   "v8",    "v8a",    "v8l",      "v8.1-a",
      "v8.1a",     "v8.2-a", "v8.2a", "v8.3-a", "v8.3a",    "v8.4-a",
      "v8.4a",     "v8.5-a", "v8.5a", "v8.6-a", "v8.6a",    "v8.7-a",
      "v8.7a",     "v8.8-a", "v8.8a", "v8-r",   "v8m.base", "v8m.main",
      "v8.1m.main"};

  for (unsigned i = 0; i < std::size(Arch); i++) {
    std::string arm_1 = "armeb" + (std::string)(Arch[i]);
    std::string arm_2 = "arm" + (std::string)(Arch[i]) + "eb";
    std::string arm_3 = "arm" + (std::string)(Arch[i]);
    std::string thumb_1 = "thumbeb" + (std::string)(Arch[i]);
    std::string thumb_2 = "thumb" + (std::string)(Arch[i]) + "eb";
    std::string thumb_3 = "thumb" + (std::string)(Arch[i]);

    EXPECT_EQ(ARM::EndianKind::BIG, ARM::parseArchEndian(arm_1));
    EXPECT_EQ(ARM::EndianKind::BIG, ARM::parseArchEndian(arm_2));
    EXPECT_EQ(ARM::EndianKind::LITTLE, ARM::parseArchEndian(arm_3));

    EXPECT_EQ(ARM::ISAKind::ARM, ARM::parseArchISA(arm_1));
    EXPECT_EQ(ARM::ISAKind::ARM, ARM::parseArchISA(arm_2));
    EXPECT_EQ(ARM::ISAKind::ARM, ARM::parseArchISA(arm_3));
    if (i >= 4) {
      EXPECT_EQ(ARM::EndianKind::BIG, ARM::parseArchEndian(thumb_1));
      EXPECT_EQ(ARM::EndianKind::BIG, ARM::parseArchEndian(thumb_2));
      EXPECT_EQ(ARM::EndianKind::LITTLE, ARM::parseArchEndian(thumb_3));

      EXPECT_EQ(ARM::ISAKind::THUMB, ARM::parseArchISA(thumb_1));
      EXPECT_EQ(ARM::ISAKind::THUMB, ARM::parseArchISA(thumb_2));
      EXPECT_EQ(ARM::ISAKind::THUMB, ARM::parseArchISA(thumb_3));
    }
  }

  EXPECT_EQ(ARM::EndianKind::LITTLE, ARM::parseArchEndian("aarch64"));
  EXPECT_EQ(ARM::EndianKind::LITTLE, ARM::parseArchEndian("arm64_32"));
  EXPECT_EQ(ARM::EndianKind::BIG, ARM::parseArchEndian("aarch64_be"));

  EXPECT_EQ(ARM::ISAKind::AARCH64, ARM::parseArchISA("aarch64"));
  EXPECT_EQ(ARM::ISAKind::AARCH64, ARM::parseArchISA("aarch64_be"));
  EXPECT_EQ(ARM::ISAKind::AARCH64, ARM::parseArchISA("arm64"));
  EXPECT_EQ(ARM::ISAKind::AARCH64, ARM::parseArchISA("arm64_be"));
  EXPECT_EQ(ARM::ISAKind::AARCH64, ARM::parseArchISA("arm64_32"));
  EXPECT_EQ(ARM::ISAKind::AARCH64, ARM::parseArchISA("aarch64_32"));
}

TEST(TargetParserTest, ARMparseArchProfile) {
  for (unsigned i = 0; i < std::size(ARMArch); i++) {
    switch (ARM::parseArch(ARMArch[i])) {
    case ARM::ArchKind::ARMV6M:
    case ARM::ArchKind::ARMV7M:
    case ARM::ArchKind::ARMV7EM:
    case ARM::ArchKind::ARMV8MMainline:
    case ARM::ArchKind::ARMV8MBaseline:
    case ARM::ArchKind::ARMV8_1MMainline:
      EXPECT_EQ(ARM::ProfileKind::M, ARM::parseArchProfile(ARMArch[i]));
      break;
    case ARM::ArchKind::ARMV7R:
    case ARM::ArchKind::ARMV8R:
      EXPECT_EQ(ARM::ProfileKind::R, ARM::parseArchProfile(ARMArch[i]));
      break;
    case ARM::ArchKind::ARMV7A:
    case ARM::ArchKind::ARMV7VE:
    case ARM::ArchKind::ARMV7K:
    case ARM::ArchKind::ARMV8A:
    case ARM::ArchKind::ARMV8_1A:
    case ARM::ArchKind::ARMV8_2A:
    case ARM::ArchKind::ARMV8_3A:
    case ARM::ArchKind::ARMV8_4A:
    case ARM::ArchKind::ARMV8_5A:
    case ARM::ArchKind::ARMV8_6A:
    case ARM::ArchKind::ARMV8_7A:
    case ARM::ArchKind::ARMV8_8A:
    case ARM::ArchKind::ARMV8_9A:
    case ARM::ArchKind::ARMV9A:
    case ARM::ArchKind::ARMV9_1A:
    case ARM::ArchKind::ARMV9_2A:
    case ARM::ArchKind::ARMV9_3A:
    case ARM::ArchKind::ARMV9_4A:
    case ARM::ArchKind::ARMV9_5A:
    case ARM::ArchKind::ARMV9_6A:
    case ARM::ArchKind::ARMV9_7A:
      EXPECT_EQ(ARM::ProfileKind::A, ARM::parseArchProfile(ARMArch[i]));
      break;
    default:
      EXPECT_EQ(ARM::ProfileKind::INVALID, ARM::parseArchProfile(ARMArch[i]));
      break;
    }
  }
}

TEST(TargetParserTest, ARMparseArchVersion) {
  for (unsigned i = 0; i < std::size(ARMArch); i++)
    if (((std::string)ARMArch[i]).substr(0, 4) == "armv")
      EXPECT_EQ((ARMArch[i][4] - 48u), ARM::parseArchVersion(ARMArch[i]));
    else
      EXPECT_EQ(5u, ARM::parseArchVersion(ARMArch[i]));
}

TEST(TargetParserTest, getARMCPUForArch) {
  // Platform specific defaults.
  {
    llvm::Triple Triple("arm--openbsd");
    EXPECT_EQ("cortex-a8", ARM::getARMCPUForArch(Triple));
  }
  {
    llvm::Triple Triple("armv6-unknown-freebsd");
    EXPECT_EQ("arm1176jzf-s", ARM::getARMCPUForArch(Triple));
  }
  {
    llvm::Triple Triple("thumbv6-unknown-freebsd");
    EXPECT_EQ("arm1176jzf-s", ARM::getARMCPUForArch(Triple));
  }
  {
    llvm::Triple Triple("armebv6-unknown-freebsd");
    EXPECT_EQ("arm1176jzf-s", ARM::getARMCPUForArch(Triple));
  }
  {
    llvm::Triple Triple("arm--win32");
    EXPECT_EQ("cortex-a9", ARM::getARMCPUForArch(Triple));
    EXPECT_EQ("generic", ARM::getARMCPUForArch(Triple, "armv8-a"));
  }
  // Some alternative architectures
  {
    llvm::Triple Triple("armv7k-apple-ios9");
    EXPECT_EQ("cortex-a7", ARM::getARMCPUForArch(Triple));
  }
  {
    llvm::Triple Triple("armv7k-apple-watchos3");
    EXPECT_EQ("cortex-a7", ARM::getARMCPUForArch(Triple));
  }
  {
    llvm::Triple Triple("armv7k-apple-tvos9");
    EXPECT_EQ("cortex-a7", ARM::getARMCPUForArch(Triple));
  }
  // armeb is permitted, but armebeb is not
  {
    llvm::Triple Triple("armeb-none-eabi");
    EXPECT_EQ("arm7tdmi", ARM::getARMCPUForArch(Triple));
  }
  {
    llvm::Triple Triple("armebeb-none-eabi");
    EXPECT_EQ("", ARM::getARMCPUForArch(Triple));
  }
  {
    llvm::Triple Triple("armebv6eb-none-eabi");
    EXPECT_EQ("", ARM::getARMCPUForArch(Triple));
  }
  // xscaleeb is permitted, but armebxscale is not
  {
    llvm::Triple Triple("xscaleeb-none-eabi");
    EXPECT_EQ("xscale", ARM::getARMCPUForArch(Triple));
  }
  {
    llvm::Triple Triple("armebxscale-none-eabi");
    EXPECT_EQ("", ARM::getARMCPUForArch(Triple));
  }
}

TEST(TargetParserTest, ARMPrintSupportedExtensions) {
  std::string expected =
      "All available -march extensions for ARM\n\n"
      "    Name                Description\n"
      "    crc                 This is a long dummy description\n"
      "    crypto\n"
      "    sha2\n";

  StringMap<StringRef> DummyMap;
  DummyMap["crc"] = "This is a long dummy description";

  outs().flush();
  testing::internal::CaptureStdout();
  ARM::PrintSupportedExtensions(DummyMap);
  outs().flush();
  std::string captured = testing::internal::GetCapturedStdout();

  // Check that the start of the output is as expected.
  EXPECT_EQ(0ULL, captured.find(expected));

  // Should not include "none" or "invalid".
  EXPECT_EQ(std::string::npos, captured.find("none"));
  EXPECT_EQ(std::string::npos, captured.find("invalid"));
  // Should not include anything that lacks a feature name. Checking a few here
  // but not all as if one is hidden correctly the rest should be.
  EXPECT_EQ(std::string::npos, captured.find("maverick"));
  EXPECT_EQ(std::string::npos, captured.find("xscale"));
}

struct AArch64CPUTestParams
    : public ARMCPUTestParams<AArch64::ExtensionBitset> {
  AArch64CPUTestParams(StringRef CPUName, StringRef ExpectedArch)
      : ARMCPUTestParams<AArch64::ExtensionBitset>(CPUName, ExpectedArch,
                                                   /*ignored*/ "", {},
                                                   /*ignored*/ "") {}
  /// Print a gtest-compatible facsimile of the CPUName, to make the test's name
  /// human-readable.
  ///
  /// https://github.com/google/googletest/blob/main/docs/advanced.md#specifying-names-for-value-parameterized-test-parameters
  static std::string PrintToStringParamName(
      const testing::TestParamInfo<AArch64CPUTestParams> &Info) {
    std::string Name = Info.param.CPUName.str();
    for (char &C : Name)
      if (!std::isalnum(C))
        C = '_';
    return Name;
  }
};

class AArch64CPUTestFixture
    : public ::testing::TestWithParam<AArch64CPUTestParams> {};

TEST_P(AArch64CPUTestFixture, testAArch64CPU) {
  auto params = GetParam();

  const std::optional<AArch64::CpuInfo> Cpu = AArch64::parseCpu(params.CPUName);
  EXPECT_TRUE(Cpu);
  EXPECT_EQ(params.ExpectedArch,
            AArch64::StrTab[AArch64::ArchInfos[Cpu->ArchIdx].Name]);
}

INSTANTIATE_TEST_SUITE_P(
    AArch64CPUTests, AArch64CPUTestFixture,
    ::testing::Values(AArch64CPUTestParams("cortex-a34", "armv8-a"),
                      AArch64CPUTestParams("cortex-a35", "armv8-a"),
                      AArch64CPUTestParams("cortex-a320", "armv9.2-a"),
                      AArch64CPUTestParams("cortex-a53", "armv8-a"),
                      AArch64CPUTestParams("cortex-a55", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-a510", "armv9-a"),
                      AArch64CPUTestParams("cortex-a520", "armv9.2-a"),
                      AArch64CPUTestParams("cortex-a520ae", "armv9.2-a"),
                      AArch64CPUTestParams("cortex-a57", "armv8-a"),
                      AArch64CPUTestParams("cortex-a65", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-a65ae", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-a72", "armv8-a"),
                      AArch64CPUTestParams("cortex-a73", "armv8-a"),
                      AArch64CPUTestParams("cortex-a75", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-a76", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-a76ae", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-a77", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-a78", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-a78ae", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-a78c", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-a710", "armv9-a"),
                      AArch64CPUTestParams("cortex-a715", "armv9-a"),
                      AArch64CPUTestParams("cortex-a720", "armv9.2-a"),
                      AArch64CPUTestParams("cortex-a720ae", "armv9.2-a"),
                      AArch64CPUTestParams("cortex-a725", "armv9.2-a"),
                      AArch64CPUTestParams("neoverse-v1", "armv8.4-a"),
                      AArch64CPUTestParams("neoverse-v2", "armv9-a"),
                      AArch64CPUTestParams("neoverse-v3", "armv9.2-a"),
                      AArch64CPUTestParams("neoverse-v3ae", "armv9.2-a"),
                      AArch64CPUTestParams("armagicpu", "armv9.2-a"),
                      AArch64CPUTestParams("cortex-r82", "armv8-r"),
                      AArch64CPUTestParams("cortex-r82ae", "armv8-r"),
                      AArch64CPUTestParams("cortex-x1", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-x1c", "armv8.2-a"),
                      AArch64CPUTestParams("cortex-x2", "armv9-a"),
                      AArch64CPUTestParams("cortex-x3", "armv9-a"),
                      AArch64CPUTestParams("cortex-x4", "armv9.2-a"),
                      AArch64CPUTestParams("cortex-x925", "armv9.2-a"),
                      AArch64CPUTestParams("c1-nano", "armv9.3-a"),
                      AArch64CPUTestParams("c1-premium", "armv9.3-a"),
                      AArch64CPUTestParams("c1-pro", "armv9.3-a"),
                      AArch64CPUTestParams("c1-ultra", "armv9.3-a"),
                      AArch64CPUTestParams("cyclone", "armv8-a"),
                      AArch64CPUTestParams("apple-a7", "armv8-a"),
                      AArch64CPUTestParams("apple-a8", "armv8-a"),
                      AArch64CPUTestParams("apple-a9", "armv8-a"),
                      AArch64CPUTestParams("apple-a10", "armv8-a"),
                      AArch64CPUTestParams("apple-a11", "armv8.2-a"),
                      AArch64CPUTestParams("apple-a12", "armv8.3-a"),
                      AArch64CPUTestParams("apple-s4", "armv8.3-a"),
                      AArch64CPUTestParams("apple-s5", "armv8.3-a"),
                      AArch64CPUTestParams("apple-a13", "armv8.4-a"),
                      AArch64CPUTestParams("apple-s6", "armv8.4-a"),
                      AArch64CPUTestParams("apple-s7", "armv8.4-a"),
                      AArch64CPUTestParams("apple-s8", "armv8.4-a"),
                      AArch64CPUTestParams("apple-a14", "armv8.4-a"),
                      AArch64CPUTestParams("apple-m1", "armv8.4-a"),
                      AArch64CPUTestParams("apple-a15", "armv8.6-a"),
                      AArch64CPUTestParams("apple-m2", "armv8.6-a"),
                      AArch64CPUTestParams("apple-a16", "armv8.6-a"),
                      AArch64CPUTestParams("apple-m3", "armv8.6-a"),
                      AArch64CPUTestParams("apple-s9", "armv8.6-a"),
                      AArch64CPUTestParams("apple-s10", "armv8.6-a"),
                      AArch64CPUTestParams("apple-a17", "armv8.6-a"),
                      AArch64CPUTestParams("apple-m4", "armv8.7-a"),
                      AArch64CPUTestParams("apple-a18", "armv8.7-a"),
                      AArch64CPUTestParams("apple-m5", "armv8.7-a"),
                      AArch64CPUTestParams("exynos-m3", "armv8-a"),
                      AArch64CPUTestParams("exynos-m4", "armv8.2-a"),
                      AArch64CPUTestParams("exynos-m5", "armv8.2-a"),
                      AArch64CPUTestParams("falkor", "armv8-a"),
                      AArch64CPUTestParams("kryo", "armv8-a"),
                      AArch64CPUTestParams("neoverse-e1", "armv8.2-a"),
                      AArch64CPUTestParams("neoverse-n1", "armv8.2-a"),
                      AArch64CPUTestParams("neoverse-n2", "armv9-a"),
                      AArch64CPUTestParams("neoverse-n3", "armv9.2-a"),
                      AArch64CPUTestParams("ampere1", "armv8.6-a"),
                      AArch64CPUTestParams("ampere1a", "armv8.6-a"),
                      AArch64CPUTestParams("ampere1b", "armv8.7-a"),
                      AArch64CPUTestParams("ampere1c", "armv9.2-a"),
                      AArch64CPUTestParams("neoverse-512tvb", "armv8.4-a"),
                      AArch64CPUTestParams("thunderx2t99", "armv8.1-a"),
                      AArch64CPUTestParams("thunderx3t110", "armv8.3-a"),
                      AArch64CPUTestParams("thunderx", "armv8-a"),
                      AArch64CPUTestParams("thunderxt81", "armv8-a"),
                      AArch64CPUTestParams("thunderxt83", "armv8-a"),
                      AArch64CPUTestParams("thunderxt88", "armv8-a"),
                      AArch64CPUTestParams("tsv110", "armv8.2-a"),
                      AArch64CPUTestParams("hip12", "armv8.7-a"),
                      AArch64CPUTestParams("a64fx", "armv8.2-a"),
                      AArch64CPUTestParams("fujitsu-monaka", "armv9.3-a"),
                      AArch64CPUTestParams("carmel", "armv8.2-a"),
                      AArch64CPUTestParams("gb10", "armv9.2-a"),
                      AArch64CPUTestParams("grace", "armv9-a"),
                      AArch64CPUTestParams("olympus", "armv9.2-a"),
                      AArch64CPUTestParams("rigel", "armv9.2-a"),
                      AArch64CPUTestParams("saphira", "armv8.4-a"),
                      AArch64CPUTestParams("oryon-1", "armv8.6-a")),
    AArch64CPUTestParams::PrintToStringParamName);

struct AArch64CPUAliasTestParams {
  AArch64CPUAliasTestParams(std::vector<StringRef> Aliases)
      : Aliases(Aliases) {}

  friend std::ostream &operator<<(std::ostream &os,
                                  const AArch64CPUAliasTestParams &params) {
    raw_os_ostream oss(os);
    interleave(params.Aliases, oss, ", ");
    return os;
  }

  /// Print a gtest-compatible facsimile of the first cpu (the aliasee), to make
  /// the test's name human-readable.
  static std::string PrintToStringParamName(
      const testing::TestParamInfo<AArch64CPUAliasTestParams> &Info) {
    std::string Name = Info.param.Aliases.front().str();
    for (char &C : Name)
      if (!std::isalnum(C))
        C = '_';
    return Name;
  }

  std::vector<StringRef> Aliases;
};

class AArch64CPUAliasTestFixture
    : public ::testing::TestWithParam<AArch64CPUAliasTestParams> {};

static std::string aarch64FeaturesFromBits(AArch64::ExtensionBitset BitFlags) {
  std::vector<StringRef> Flags;
  bool OK = AArch64::getExtensionFeatures(BitFlags, Flags);
  assert(OK);
  (void)OK;
  std::string S;
  raw_string_ostream OS(S);
  interleave(Flags, OS, ", ");
  return OS.str();
}

TEST_P(AArch64CPUAliasTestFixture, testCPUAlias) {
  AArch64CPUAliasTestParams params = GetParam();

  StringRef MainName = params.Aliases[0];
  const std::optional<AArch64::CpuInfo> Cpu = AArch64::parseCpu(MainName);
  const AArch64::ArchInfo &MainAI = AArch64::ArchInfos[Cpu->ArchIdx];
  AArch64::ExtensionBitset MainFlags = Cpu->DefaultExtensions;

  for (size_t I = 1, E = params.Aliases.size(); I != E; ++I) {
    StringRef OtherName = params.Aliases[I];
    const std::optional<AArch64::CpuInfo> OtherCpu =
        AArch64::parseCpu(OtherName);
    const AArch64::ArchInfo &OtherAI = AArch64::ArchInfos[OtherCpu->ArchIdx];

    EXPECT_EQ(MainAI.Version, OtherAI.Version)
        << MainName << " vs " << OtherName;
    EXPECT_EQ(MainAI.Name, OtherAI.Name) << MainName << " vs " << OtherName;
    EXPECT_EQ(MainAI.Profile, OtherAI.Profile)
        << MainName << " vs " << OtherName;
    EXPECT_EQ(MainAI.DefaultExts, OtherAI.DefaultExts)
        << MainName << " vs " << OtherName;
    EXPECT_EQ(MainAI, OtherAI) << MainName << " vs " << OtherName;

    AArch64::ExtensionBitset OtherFlags = OtherCpu->DefaultExtensions;

    EXPECT_EQ(MainFlags, OtherFlags) << MainName << " vs " << OtherName;

    EXPECT_EQ(aarch64FeaturesFromBits(MainFlags),
              aarch64FeaturesFromBits(OtherFlags))
        << MainName << " vs " << OtherName << "\n        Diff: "
        << (aarch64FeaturesFromBits(MainFlags ^ OtherFlags));
  }
}

INSTANTIATE_TEST_SUITE_P(
    AArch64CPUAliasTests, AArch64CPUAliasTestFixture,
    ::testing::Values(AArch64CPUAliasTestParams({"neoverse-n2", "cobalt-100"}),
                      AArch64CPUAliasTestParams({"apple-a7", "cyclone",
                                                 "apple-a8", "apple-a9"}),
                      AArch64CPUAliasTestParams({"apple-a12", "apple-s4",
                                                 "apple-s5"}),
                      AArch64CPUAliasTestParams({"apple-a13", "apple-s6",
                                                 "apple-s7", "apple-s8"}),
                      AArch64CPUAliasTestParams({"apple-a14", "apple-m1"}),
                      AArch64CPUAliasTestParams({"apple-a15", "apple-m2"}),
                      AArch64CPUAliasTestParams({"apple-a16", "apple-m3",
                                                 "apple-s9", "apple-s10"}),
                      AArch64CPUAliasTestParams({"apple-m4", "apple-a18"}),
                      AArch64CPUAliasTestParams({"apple-m5", "apple-a19"})),
    AArch64CPUAliasTestParams::PrintToStringParamName);

// Note: number of CPUs includes aliases.
static constexpr unsigned NumAArch64CPUArchs = 101;

TEST(TargetParserTest, testAArch64CPUArchList) {
  SmallVector<StringRef, NumAArch64CPUArchs> List;
  AArch64::fillValidCPUArchList(List);

  // No list exists for these in this test suite, so ensure all are
  // valid, and match the expected 'magic' count.
  EXPECT_EQ(List.size(), NumAArch64CPUArchs);
  for (StringRef CPU : List) {
    EXPECT_TRUE(AArch64::parseCpu(CPU));
  }
}

bool testAArch64Arch(StringRef Arch) {
  const AArch64::ArchInfo *AI = AArch64::parseArch(Arch);
  return AI != nullptr;
}

TEST(TargetParserTest, testAArch64Arch) {
  EXPECT_TRUE(testAArch64Arch("armv8-a"));
  EXPECT_TRUE(testAArch64Arch("armv8.1-a"));
  EXPECT_TRUE(testAArch64Arch("armv8.2-a"));
  EXPECT_TRUE(testAArch64Arch("armv8.3-a"));
  EXPECT_TRUE(testAArch64Arch("armv8.4-a"));
  EXPECT_TRUE(testAArch64Arch("armv8.5-a"));
  EXPECT_TRUE(testAArch64Arch("armv8.6-a"));
  EXPECT_TRUE(testAArch64Arch("armv8.7-a"));
  EXPECT_TRUE(testAArch64Arch("armv8.8-a"));
  EXPECT_TRUE(testAArch64Arch("armv8.9-a"));
  EXPECT_TRUE(testAArch64Arch("armv9-a"));
  EXPECT_TRUE(testAArch64Arch("armv9.1-a"));
  EXPECT_TRUE(testAArch64Arch("armv9.2-a"));
  EXPECT_TRUE(testAArch64Arch("armv9.3-a"));
  EXPECT_TRUE(testAArch64Arch("armv9.4-a"));
  EXPECT_TRUE(testAArch64Arch("armv9.5-a"));
  EXPECT_TRUE(testAArch64Arch("armv9.6-a"));
  EXPECT_TRUE(testAArch64Arch("armv9.7-a"));
}

bool testAArch64Extension(StringRef CPUName, StringRef ArchExt) {
  std::optional<AArch64::ExtensionInfo> Extension =
      AArch64::parseArchExtension(ArchExt);
  if (!Extension)
    return false;
  std::optional<AArch64::CpuInfo> CpuInfo = AArch64::parseCpu(CPUName);
  return CpuInfo->DefaultExtensions.test(Extension->ID);
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
  EXPECT_TRUE(testAArch64Extension("tsv110", "jscvt"));
  EXPECT_TRUE(testAArch64Extension("tsv110", "fcma"));
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
  EXPECT_FALSE(testAArch64Extension(AArch64::ARMV9_5A, "hinte"));
  EXPECT_TRUE(testAArch64Extension(AArch64::ARMV9_6A, "hinte"));
  EXPECT_TRUE(testAArch64Extension(AArch64::ARMV9_7A, "hinte"));
}

TEST(TargetParserTest, AArch64ExtensionFeatures) {
  std::vector<uint64_t> Extensions = {
      AArch64::AEK_CRC,          AArch64::AEK_LSE,
      AArch64::AEK_RDM,          AArch64::AEK_CRYPTO,
      AArch64::AEK_SM4,          AArch64::AEK_SHA3,
      AArch64::AEK_SHA2,         AArch64::AEK_AES,
      AArch64::AEK_DOTPROD,      AArch64::AEK_FP,
      AArch64::AEK_SIMD,         AArch64::AEK_FP16,
      AArch64::AEK_FP16FML,      AArch64::AEK_PROFILE,
      AArch64::AEK_RAS,          AArch64::AEK_SVE,
      AArch64::AEK_SVE2,         AArch64::AEK_SVE2AES,
      AArch64::AEK_SVE2SM4,      AArch64::AEK_SVE2SHA3,
      AArch64::AEK_SVE2BITPERM,  AArch64::AEK_RCPC,
      AArch64::AEK_RAND,         AArch64::AEK_MTE,
      AArch64::AEK_SSBS,         AArch64::AEK_SB,
      AArch64::AEK_PREDRES,      AArch64::AEK_BF16,
      AArch64::AEK_I8MM,         AArch64::AEK_F32MM,
      AArch64::AEK_LS64,         AArch64::AEK_BRBE,
      AArch64::AEK_PAUTH,        AArch64::AEK_FLAGM,
      AArch64::AEK_SME,          AArch64::AEK_SMEF64F64,
      AArch64::AEK_SMEI16I64,    AArch64::AEK_SME2,
      AArch64::AEK_HBC,          AArch64::AEK_MOPS,
      AArch64::AEK_PERFMON,      AArch64::AEK_SVE2P1,
      AArch64::AEK_SME2P1,       AArch64::AEK_SMEB16B16,
      AArch64::AEK_SMEF16F16,    AArch64::AEK_CSSC,
      AArch64::AEK_RCPC3,        AArch64::AEK_THE,
      AArch64::AEK_D128,         AArch64::AEK_LSE128,
      AArch64::AEK_SPECRES2,     AArch64::AEK_RASV2,
      AArch64::AEK_ITE,          AArch64::AEK_GCS,
      AArch64::AEK_FAMINMAX,     AArch64::AEK_FP8FMA,
      AArch64::AEK_SSVE_FP8FMA,  AArch64::AEK_FP8DOT2,
      AArch64::AEK_SSVE_FP8DOT2, AArch64::AEK_FP8DOT4,
      AArch64::AEK_SSVE_FP8DOT4, AArch64::AEK_LUT,
      AArch64::AEK_SME_LUTV2,    AArch64::AEK_SMEF8F16,
      AArch64::AEK_SMEF8F32,     AArch64::AEK_SMEFA64,
      AArch64::AEK_CPA,          AArch64::AEK_PAUTHLR,
      AArch64::AEK_TLBIW,        AArch64::AEK_JSCVT,
      AArch64::AEK_FCMA,         AArch64::AEK_FP8,
      AArch64::AEK_SVEB16B16,    AArch64::AEK_SVE2P2,
      AArch64::AEK_SME2P2,       AArch64::AEK_SVE_BFSCALE,
      AArch64::AEK_SVE_F16F32MM, AArch64::AEK_SVE_AES2,
      AArch64::AEK_SSVE_AES,     AArch64::AEK_F8F32MM,
      AArch64::AEK_F8F16MM,      AArch64::AEK_LSFE,
      AArch64::AEK_FPRCVT,       AArch64::AEK_CMPBR,
      AArch64::AEK_LSUI,         AArch64::AEK_OCCMO,
      AArch64::AEK_SVEAES,       AArch64::AEK_SME_MOP4,
      AArch64::AEK_SME_TMOP,     AArch64::AEK_SVEBITPERM,
      AArch64::AEK_SSVE_BITPERM, AArch64::AEK_SVESHA3,
      AArch64::AEK_LSCP,         AArch64::AEK_TLBID,
      AArch64::AEK_GCIE,         AArch64::AEK_SME2P3,
      AArch64::AEK_SVE2P3,       AArch64::AEK_SVE_B16MM,
      AArch64::AEK_F16MM,        AArch64::AEK_F16F32DOT,
      AArch64::AEK_F16F32MM,     AArch64::AEK_MOPS_GO,
      AArch64::AEK_POE2,         AArch64::AEK_TEV,
      AArch64::AEK_BTIE,         AArch64::AEK_F64MM,
      AArch64::AEK_POPS,         AArch64::AEK_SVESM4,
      AArch64::AEK_MTETC,        AArch64::AEK_HINTE,
  };

  std::vector<StringRef> Features;

  AArch64::ExtensionBitset ExtVal;
  for (auto Ext : Extensions)
    ExtVal.set(Ext);

  // Test an empty set of features.
  EXPECT_TRUE(AArch64::getExtensionFeatures({}, Features));
  EXPECT_TRUE(Features.size() == 0);

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
  EXPECT_TRUE(llvm::is_contained(Features, "+sve-b16b16"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve-bfscale"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve-f16f32mm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve-aes"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2-aes"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve-sm4"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2-sm4"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve-sha3"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2-sha3"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve-bitperm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2-bitperm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+ssve-bitperm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve-aes2"));
  EXPECT_TRUE(llvm::is_contained(Features, "+ssve-aes"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2p1"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2p2"));
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
  EXPECT_TRUE(llvm::is_contained(Features, "+ls64"));
  EXPECT_TRUE(llvm::is_contained(Features, "+brbe"));
  EXPECT_TRUE(llvm::is_contained(Features, "+pauth"));
  EXPECT_TRUE(llvm::is_contained(Features, "+flagm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-f64f64"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-i16i64"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-f16f16"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme2"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-b16b16"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme2p1"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme2p2"));
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
  EXPECT_TRUE(llvm::is_contained(Features, "+fp8"));
  EXPECT_TRUE(llvm::is_contained(Features, "+faminmax"));
  EXPECT_TRUE(llvm::is_contained(Features, "+fp8fma"));
  EXPECT_TRUE(llvm::is_contained(Features, "+ssve-fp8fma"));
  EXPECT_TRUE(llvm::is_contained(Features, "+fp8dot2"));
  EXPECT_TRUE(llvm::is_contained(Features, "+ssve-fp8dot2"));
  EXPECT_TRUE(llvm::is_contained(Features, "+fp8dot4"));
  EXPECT_TRUE(llvm::is_contained(Features, "+ssve-fp8dot4"));
  EXPECT_TRUE(llvm::is_contained(Features, "+f8f32mm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+f8f16mm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+lut"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-lutv2"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-f8f16"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-f8f32"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-fa64"));
  EXPECT_TRUE(llvm::is_contained(Features, "+cpa"));
  EXPECT_TRUE(llvm::is_contained(Features, "+pauth-lr"));
  EXPECT_TRUE(llvm::is_contained(Features, "+tlbiw"));
  EXPECT_TRUE(llvm::is_contained(Features, "+jsconv"));
  EXPECT_TRUE(llvm::is_contained(Features, "+complxnum"));
  EXPECT_TRUE(llvm::is_contained(Features, "+lsfe"));
  EXPECT_TRUE(llvm::is_contained(Features, "+fprcvt"));
  EXPECT_TRUE(llvm::is_contained(Features, "+cmpbr"));
  EXPECT_TRUE(llvm::is_contained(Features, "+lsui"));
  EXPECT_TRUE(llvm::is_contained(Features, "+occmo"));
  EXPECT_TRUE(llvm::is_contained(Features, "+pops"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-mop4"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme-tmop"));
  EXPECT_TRUE(llvm::is_contained(Features, "+lscp"));
  EXPECT_TRUE(llvm::is_contained(Features, "+tlbid"));
  EXPECT_TRUE(llvm::is_contained(Features, "+mtetc"));
  EXPECT_TRUE(llvm::is_contained(Features, "+gcie"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sme2p3"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve2p3"));
  EXPECT_TRUE(llvm::is_contained(Features, "+sve-b16mm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+f16mm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+f16f32dot"));
  EXPECT_TRUE(llvm::is_contained(Features, "+f16f32mm"));
  EXPECT_TRUE(llvm::is_contained(Features, "+mops-go"));
  EXPECT_TRUE(llvm::is_contained(Features, "+poe2"));
  EXPECT_TRUE(llvm::is_contained(Features, "+tev"));
  EXPECT_TRUE(llvm::is_contained(Features, "+btie"));
  EXPECT_TRUE(llvm::is_contained(Features, "+hinte"));

  // Assuming we listed every extension above, this should produce the same
  // result.
  std::vector<StringRef> AllFeatures;
  EXPECT_TRUE(AArch64::getExtensionFeatures(ExtVal, AllFeatures));
  EXPECT_THAT(Features, ::testing::ContainerEq(AllFeatures));
}

TEST(TargetParserTest, AArch64ArchFeatures) {
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV8A.ArchFeature], "+v8a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV8_1A.ArchFeature], "+v8.1a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV8_2A.ArchFeature], "+v8.2a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV8_3A.ArchFeature], "+v8.3a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV8_4A.ArchFeature], "+v8.4a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV8_5A.ArchFeature], "+v8.5a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV8_6A.ArchFeature], "+v8.6a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV8_7A.ArchFeature], "+v8.7a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV8_8A.ArchFeature], "+v8.8a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV8_9A.ArchFeature], "+v8.9a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV9A.ArchFeature], "+v9a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV9_1A.ArchFeature], "+v9.1a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV9_2A.ArchFeature], "+v9.2a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV9_3A.ArchFeature], "+v9.3a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV9_4A.ArchFeature], "+v9.4a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV9_5A.ArchFeature], "+v9.5a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV9_6A.ArchFeature], "+v9.6a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV9_7A.ArchFeature], "+v9.7a");
  EXPECT_EQ(AArch64::StrTab[AArch64::ARMV8R.ArchFeature], "+v8r");
}

TEST(TargetParserTest, AArch64ArchPartialOrder) {
  for (const auto &A : AArch64::ArchInfos) {
    EXPECT_EQ(A, A);

    // v8r has no relation to other valid architectures
    if (A != AArch64::ARMV8R) {
      EXPECT_FALSE(A.implies(AArch64::ARMV8R));
      EXPECT_FALSE(AArch64::ARMV8R.implies(A));
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

  for (const auto *A :
       {&AArch64::ARMV9_1A, &AArch64::ARMV9_2A, &AArch64::ARMV9_3A,
        &AArch64::ARMV9_4A, &AArch64::ARMV9_5A, &AArch64::ARMV9_6A,
        &AArch64::ARMV9_7A})
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
  EXPECT_TRUE(AArch64::ARMV9_5A.implies(AArch64::ARMV9_4A));
  EXPECT_TRUE(AArch64::ARMV9_6A.implies(AArch64::ARMV9_5A));
  EXPECT_TRUE(AArch64::ARMV9_7A.implies(AArch64::ARMV9_6A));

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
      {"sve-b16b16", "nosve-b16b16", "+sve-b16b16", "-sve-b16b16"},
      {"sve-bfscale", "nosve-bfscale", "+sve-bfscale", "-sve-bfscale"},
      {"sve-f16f32mm", "nosve-f16f32mm", "+sve-f16f32mm", "-sve-f16f32mm"},
      {"sve2", "nosve2", "+sve2", "-sve2"},
      {"sve-aes", "nosve-aes", "+sve-aes", "-sve-aes"},
      {"sve-sm4", "nosve-sm4", "+sve-sm4", "-sve-sm4"},
      {"sve-sha3", "nosve-sha3", "+sve-sha3", "-sve-sha3"},
      {"sve2-aes", "nosve2-aes", "+sve2-aes", "-sve2-aes"},
      {"sve2-sm4", "nosve2-sm4", "+sve2-sm4", "-sve2-sm4"},
      {"sve2-sha3", "nosve2-sha3", "+sve2-sha3", "-sve2-sha3"},
      {"sve2p1", "nosve2p1", "+sve2p1", "-sve2p1"},
      {"sve2p2", "nosve2p2", "+sve2p2", "-sve2p2"},
      {"sve-bitperm", "nosve-bitperm", "+sve-bitperm", "-sve-bitperm"},
      {"ssve-bitperm", "nossve-bitperm", "+ssve-bitperm", "-ssve-bitperm"},
      {"sve2-bitperm", "nosve2-bitperm", "+sve2-bitperm", "-sve2-bitperm"},
      {"sve-aes2", "nosve-aes2", "+sve-aes2", "-sve-aes2"},
      {"ssve-aes", "nossve-aes", "+ssve-aes", "-ssve-aes"},
      {"dotprod", "nodotprod", "+dotprod", "-dotprod"},
      {"rcpc", "norcpc", "+rcpc", "-rcpc"},
      {"rng", "norng", "+rand", "-rand"},
      {"memtag", "nomemtag", "+mte", "-mte"},
      {"pauth", "nopauth", "+pauth", "-pauth"},
      {"ssbs", "nossbs", "+ssbs", "-ssbs"},
      {"sb", "nosb", "+sb", "-sb"},
      {"predres", "nopredres", "+predres", "-predres"},
      {"i8mm", "noi8mm", "+i8mm", "-i8mm"},
      {"f32mm", "nof32mm", "+f32mm", "-f32mm"},
      {"f64mm", "nof64mm", "+f64mm", "-f64mm"},
      {"f8f32mm", "nof8f32mm", "+f8f32mm", "-f8f32mm"},
      {"f8f16mm", "nof8f16mm", "+f8f16mm", "-f8f16mm"},
      {"sme", "nosme", "+sme", "-sme"},
      {"sme-fa64", "nosme-fa64", "+sme-fa64", "-sme-fa64"},
      {"sme-f64f64", "nosme-f64f64", "+sme-f64f64", "-sme-f64f64"},
      {"sme-i16i64", "nosme-i16i64", "+sme-i16i64", "-sme-i16i64"},
      {"sme-f16f16", "nosme-f16f16", "+sme-f16f16", "-sme-f16f16"},
      {"sme2", "nosme2", "+sme2", "-sme2"},
      {"sme-b16b16", "nosme-b16b16", "+sme-b16b16", "-sme-b16b16"},
      {"sme2p1", "nosme2p1", "+sme2p1", "-sme2p1"},
      {"sme2p2", "nosme2p2", "+sme2p2", "-sme2p2"},
      {"hbc", "nohbc", "+hbc", "-hbc"},
      {"mops", "nomops", "+mops", "-mops"},
      {"pmuv3", "nopmuv3", "+perfmon", "-perfmon"},
      {"predres2", "nopredres2", "+specres2", "-specres2"},
      {"rasv2", "norasv2", "+rasv2", "-rasv2"},
      {"gcs", "nogcs", "+gcs", "-gcs"},
      {"fp8", "nofp8", "+fp8", "-fp8"},
      {"faminmax", "nofaminmax", "+faminmax", "-faminmax"},
      {"fp8fma", "nofp8fma", "+fp8fma", "-fp8fma"},
      {"ssve-fp8fma", "nossve-fp8fma", "+ssve-fp8fma", "-ssve-fp8fma"},
      {"fp8dot2", "nofp8dot2", "+fp8dot2", "-fp8dot2"},
      {"ssve-fp8dot2", "nossve-fp8dot2", "+ssve-fp8dot2", "-ssve-fp8dot2"},
      {"fp8dot4", "nofp8dot4", "+fp8dot4", "-fp8dot4"},
      {"ssve-fp8dot4", "nossve-fp8dot4", "+ssve-fp8dot4", "-ssve-fp8dot4"},
      {"lut", "nolut", "+lut", "-lut"},
      {"sme-lutv2", "nosme-lutv2", "+sme-lutv2", "-sme-lutv2"},
      {"sme-f8f16", "nosme-f8f16", "+sme-f8f16", "-sme-f8f16"},
      {"sme-f8f32", "nosme-f8f32", "+sme-f8f32", "-sme-f8f32"},
      {"lsfe", "nolsfe", "+lsfe", "-lsfe"},
      {"fprcvt", "nofprcvt", "+fprcvt", "-fprcvt"},
      {"cmpbr", "nocmpbr", "+cmpbr", "-cmpbr"},
      {"lsui", "nolsui", "+lsui", "-lsui"},
      {"occmo", "nooccmo", "+occmo", "-occmo"},
      {"pops", "nopops", "+pops", "-pops"},
      {"sme-mop4", "nosme-mop4", "+sme-mop4", "-sme-mop4"},
      {"sme-tmop", "nosme-tmop", "+sme-tmop", "-sme-tmop"},
      {"lscp", "nolscp", "+lscp", "-lscp"},
      {"tlbid", "notlbid", "+tlbid", "-tlbid"},
      {"mtetc", "nomtetc", "+mtetc", "-mtetc"},
      {"gcie", "nogcie", "+gcie", "-gcie"},
      {"sme2p3", "nosme2p3", "+sme2p3", "-sme2p3"},
      {"sve2p3", "nosve2p3", "+sve2p3", "-sve2p3"},
      {"sve-b16mm", "nosve-b16mm", "+sve-b16mm", "-sve-b16mm"},
      {"f16mm", "nof16mm", "+f16mm", "-f16mm"},
      {"f16f32dot", "nof16f32dot", "+f16f32dot", "-f16f32dot"},
      {"f16f32mm", "nof16f32mm", "+f16f32mm", "-f16f32mm"},
      {"mops-go", "nomops-go", "+mops-go", "-mops-go"},
      {"poe2", "nopoe2", "+poe2", "-poe2"},
      {"tev", "notev", "+tev", "-tev"},
      {"btie", "nobtie", "+btie", "-btie"},
      {"hinte", "nohinte", "+hinte", "-hinte"},
  };

  for (unsigned i = 0; i < std::size(ArchExt); i++) {
    EXPECT_EQ(StringRef(ArchExt[i][2]),
              AArch64::getArchExtFeature(ArchExt[i][0]));
    EXPECT_EQ(StringRef(ArchExt[i][3]),
              AArch64::getArchExtFeature(ArchExt[i][1]));
  }
}

TEST(TargetParserTest, AArch64PrintSupportedExtensions) {
  std::string expected = "All available -march extensions for AArch64\n\n"
                         "    Name                Architecture Feature(s)      "
                         "                          Description\n";

  outs().flush();
  testing::internal::CaptureStdout();
  AArch64::PrintSupportedExtensions();
  outs().flush();
  std::string captured = testing::internal::GetCapturedStdout();

  // Check that the start of the output is as expected.
  EXPECT_EQ(0ULL, captured.find(expected));

  // Should not include "none".
  EXPECT_EQ(std::string::npos, captured.find("none"));
  // Should not include anything that lacks a feature name. Checking a few here
  // but not all as if one is hidden correctly the rest should be.
  EXPECT_EQ(std::string::npos, captured.find("memtag3"));
  EXPECT_EQ(std::string::npos, captured.find("sha1"));
  EXPECT_EQ(std::string::npos, captured.find("ssbs2"));
}

struct AArch64ExtensionDependenciesBaseArchTestParams {
  const llvm::AArch64::ArchInfo &Arch;
  std::vector<StringRef> Modifiers;
  std::vector<StringRef> ExpectedPos;
  std::vector<StringRef> ExpectedNeg;
};

class AArch64ExtensionDependenciesBaseArchTestFixture
    : public ::testing::TestWithParam<
          AArch64ExtensionDependenciesBaseArchTestParams> {};

struct AArch64ExtensionDependenciesBaseCPUTestParams {
  StringRef CPUName;
  std::vector<StringRef> Modifiers;
  std::vector<StringRef> ExpectedPos;
  std::vector<StringRef> ExpectedNeg;
};

class AArch64ExtensionDependenciesBaseCPUTestFixture
    : public ::testing::TestWithParam<
          AArch64ExtensionDependenciesBaseCPUTestParams> {};

TEST_P(AArch64ExtensionDependenciesBaseArchTestFixture,
       AArch64ExtensionDependenciesBaseArch) {
  auto Params = GetParam();

  llvm::AArch64::ExtensionSet Extensions;
  Extensions.addArchDefaults(Params.Arch);
  for (auto M : Params.Modifiers) {
    bool success = Extensions.parseModifier(M);
    EXPECT_TRUE(success);
  }
  std::vector<StringRef> Features;
  Extensions.toLLVMFeatureList(Features);

  for (auto E : Params.ExpectedPos) {
    std::string PosString = "+";
    PosString += E;
    std::string NegString = "-";
    NegString += E;
    ASSERT_THAT(Features, Contains(StrEq(PosString)));
    ASSERT_THAT(Features, Not(Contains(StrEq(NegString))));
  }

  for (auto E : Params.ExpectedNeg) {
    std::string PosString = "+";
    PosString += E;
    ASSERT_THAT(Features, Not(Contains(StrEq(PosString))));
    // Features default to off, so the negative string is not expected in many
    // cases.
  }
}

TEST_P(AArch64ExtensionDependenciesBaseCPUTestFixture,
       AArch64ExtensionDependenciesBaseCPU) {
  auto Params = GetParam();

  llvm::AArch64::ExtensionSet Extensions;
  const std::optional<llvm::AArch64::CpuInfo> CPU =
      llvm::AArch64::parseCpu(Params.CPUName);
  EXPECT_TRUE(CPU);
  Extensions.addCPUDefaults(*CPU);
  for (auto M : Params.Modifiers) {
    bool success = Extensions.parseModifier(M);
    EXPECT_TRUE(success);
  }
  std::vector<StringRef> Features;
  Extensions.toLLVMFeatureList(Features);

  for (auto E : Params.ExpectedPos) {
    std::string PosString = "+";
    PosString += E;
    std::string NegString = "-";
    NegString += E;
    ASSERT_THAT(Features, Contains(StrEq(PosString)));
    ASSERT_THAT(Features, Not(Contains(StrEq(NegString))));
  }

  for (auto E : Params.ExpectedNeg) {
    std::string PosString = "+";
    PosString += E;
    ASSERT_THAT(Features, Not(Contains(StrEq(PosString))));
    // Features default to off, so the negative string is not expected in many
    // cases.
  }
}

TEST(TargetParserTest, testAArch64ReconstructFromParsedFeatures) {
  AArch64::ExtensionSet Extensions;
  std::vector<std::string> FeatureOptions = {
      "-sve2", "-Baz", "+sve", "+FooBar", "+sve2", "+neon", "-sve",
  };
  std::vector<std::string> NonExtensions;
  Extensions.reconstructFromParsedFeatures(FeatureOptions, NonExtensions);

  std::vector<std::string> NonExtensionsExpected = {"-Baz", "+FooBar"};
  ASSERT_THAT(NonExtensions, testing::ContainerEq(NonExtensionsExpected));
  std::vector<StringRef> Features;
  Extensions.toLLVMFeatureList(Features);
  std::vector<StringRef> FeaturesExpected = {"+neon", "-sve", "+sve2"};
  ASSERT_THAT(Features, testing::ContainerEq(FeaturesExpected));
}

AArch64ExtensionDependenciesBaseArchTestParams
    AArch64ExtensionDependenciesArchData[] = {
        // Base architecture features
        {AArch64::ARMV8A, {}, {"v8a", "fp-armv8", "neon"}, {}},
        {AArch64::ARMV8_1A,
         {},
         {"v8.1a", "crc", "fp-armv8", "lse", "rdm", "neon"},
         {}},
        {AArch64::ARMV9_5A, {}, {"v9.5a", "mops", "cpa"}, {}},
        {AArch64::ARMV9_7A, {}, {"v9.7a", "f16f32dot"}, {}},

        // Positive modifiers
        {AArch64::ARMV8A, {"fp16"}, {"fullfp16"}, {}},
        {AArch64::ARMV8A, {"dotprod"}, {"dotprod"}, {}},

        // Negative modifiers
        {AArch64::ARMV8A, {"nofp"}, {"v8a"}, {"fp-armv8", "neon"}},

        // Mixed modifiers
        {AArch64::ARMV8A,
         {"fp16", "nofp16"},
         {"v8a", "fp-armv8", "neon"},
         {"fullfp16"}},
        {AArch64::ARMV8A,
         {"fp16", "nofp"},
         {"v8a"},
         {"fp-armv8", "neon", "fullfp16"}},

        // Long dependency chains: sve2-bitperm -> sve2 -> sve -> fp16 -> fp
        {AArch64::ARMV8A,
         {"nofp", "sve2", "sve-bitperm"},
         {"fp-armv8", "fullfp16", "sve", "sve2", "sve-bitperm"},
         {}},
        {AArch64::ARMV8A,
         {"sve2", "sve-bitperm", "nofp16"},
         {"fp-armv8"},
         {"full-fp16", "sve", "sve2", "sve-bitperm"}},

        // Meaning of +crypto varies with base architecture.
        {AArch64::ARMV8A, {"crypto"}, {"aes", "sha2"}, {}},
        {AArch64::ARMV8_4A, {"crypto"}, {"aes", "sha2", "sha3", "sm4"}, {}},
        {AArch64::ARMV9A, {"crypto"}, {"aes", "sha2", "sha3", "sm4"}, {}},

        // -crypto always disables all crypto features, even if it wouldn't
        // enable them.
        {AArch64::ARMV8A,
         {"aes", "sha2", "sha3", "sm4", "nocrypto"},
         {},
         {"aes", "sha2", "sha3", "sm4"}},
        {AArch64::ARMV8_4A,
         {"aes", "sha2", "sha3", "sm4", "nocrypto"},
         {},
         {"aes", "sha2", "sha3", "sm4"}},

        // +fp16 implies +fp16fml for v8.4A+, but not v9.0-A+
        {AArch64::ARMV8_3A, {"fp16"}, {"fullfp16"}, {"fp16fml"}},
        {AArch64::ARMV9A, {"fp16"}, {"fullfp16"}, {"fp16fml"}},
        {AArch64::ARMV8_4A, {"fp16"}, {"fullfp16", "fp16fml"}, {}},

        // fp -> fp16
        {AArch64::ARMV8A, {"nofp", "fp16"}, {"fp-armv8", "fullfp16"}, {}},
        {AArch64::ARMV8A, {"fp16", "nofp"}, {}, {"fp-armv8", "fullfp16"}},

        // fp -> simd
        {AArch64::ARMV8A, {"nofp", "simd"}, {"fp-armv8", "neon"}, {}},
        {AArch64::ARMV8A, {"simd", "nofp"}, {}, {"fp-armv8", "neon"}},

        // fp -> jscvt
        {AArch64::ARMV8A, {"nofp", "jscvt"}, {"fp-armv8", "jsconv"}, {}},
        {AArch64::ARMV8A, {"jscvt", "nofp"}, {}, {"fp-armv8", "jsconv"}},

        // fp -> lsfe
        {AArch64::ARMV9_6A, {"nofp", "lsfe"}, {"fp-armv8", "lsfe"}, {}},
        {AArch64::ARMV9_6A, {"lsfe", "nofp"}, {}, {"fp-armv8", "lsfe"}},

        // fp -> fprcvt
        {AArch64::ARMV9_6A, {"nofp", "fprcvt"}, {"fp-armv8", "fprcvt"}, {}},
        {AArch64::ARMV9_6A, {"fprcvt", "nofp"}, {}, {"fp-armv8", "fprcvt"}},

        // simd -> {aes, sha2, sha3, sm4, f8f16mm, f8f32mm, faminmax, lut, fp8,
        // f16f32dot, f16f32mm}
        {AArch64::ARMV8A, {"nosimd", "aes"}, {"neon", "aes"}, {}},
        {AArch64::ARMV8A, {"aes", "nosimd"}, {}, {"neon", "aes"}},
        {AArch64::ARMV8A, {"nosimd", "sha2"}, {"neon", "sha2"}, {}},
        {AArch64::ARMV8A, {"sha2", "nosimd"}, {}, {"neon", "sha2"}},
        {AArch64::ARMV8A, {"nosimd", "sha3"}, {"neon", "sha3"}, {}},
        {AArch64::ARMV8A, {"sha3", "nosimd"}, {}, {"neon", "sha3"}},
        {AArch64::ARMV8A, {"nosimd", "sm4"}, {"neon", "sm4"}, {}},
        {AArch64::ARMV8A, {"sm4", "nosimd"}, {}, {"neon", "sm4"}},
        {AArch64::ARMV9_6A, {"nosimd", "f8f16mm"}, {"neon", "f8f16mm"}, {}},
        {AArch64::ARMV9_6A, {"f8f16mm", "nosimd"}, {}, {"neon", "f8f16mm"}},
        {AArch64::ARMV9_6A, {"nosimd", "f8f32mm"}, {"neon", "f8f32mm"}, {}},
        {AArch64::ARMV9_6A, {"f8f32mm", "nosimd"}, {}, {"neon", "f8f32mm"}},
        {AArch64::ARMV9_6A, {"faminmax", "nosimd"}, {}, {"neon", "faminmax"}},
        {AArch64::ARMV9_6A, {"nosimd", "faminmax"}, {"neon", "faminmax"}, {}},
        {AArch64::ARMV9_6A, {"lut", "nosimd"}, {}, {"neon", "lut"}},
        {AArch64::ARMV9_6A, {"nosimd", "lut"}, {"neon", "lut"}, {}},
        {AArch64::ARMV9_6A, {"fp8", "nosimd"}, {}, {"neon", "fp8"}},
        {AArch64::ARMV9_6A, {"nosimd", "fp8"}, {"neon", "fp8"}, {}},
        {AArch64::ARMV9_7A, {"nosimd", "f16f32mm"}, {"neon", "f16f32mm"}, {}},
        {AArch64::ARMV9_7A, {"f16f32mm", "nosimd"}, {}, {"neon", "f16f32mm"}},
        {AArch64::ARMV9_7A, {"nosimd", "f16f32dot"}, {"neon", "f16f32dot"}, {}},
        {AArch64::ARMV9_7A, {"f16f32dot", "nosimd"}, {}, {"neon", "f16f32dot"}},

        // fp8 -> {fp8dot4, fp8dot2}
        {AArch64::ARMV9_6A, {"nofp8", "fp8dot4"}, {"fp8", "fp8dot4"}, {}},
        {AArch64::ARMV9_6A, {"fp8dot4", "nofp8"}, {}, {"fp8", "fp8dot4"}},
        {AArch64::ARMV9_6A, {"nofp8", "fp8dot2"}, {"fp8", "fp8dot2"}, {}},
        {AArch64::ARMV9_6A, {"fp8dot2", "nofp8"}, {}, {"fp8", "fp8dot2"}},

        // simd -> {rdm, dotprod, fcma}
        {AArch64::ARMV8A, {"nosimd", "rdm"}, {"neon", "rdm"}, {}},
        {AArch64::ARMV8A, {"rdm", "nosimd"}, {}, {"neon", "rdm"}},
        {AArch64::ARMV8A, {"nosimd", "dotprod"}, {"neon", "dotprod"}, {}},
        {AArch64::ARMV8A, {"dotprod", "nosimd"}, {}, {"neon", "dotprod"}},
        {AArch64::ARMV8A, {"nosimd", "fcma"}, {"neon", "complxnum"}, {}},
        {AArch64::ARMV8A, {"fcma", "nosimd"}, {}, {"neon", "complxnum"}},

        // fp16 -> {fp16fml, sve, f16f32dot, f16f32mm, f16mm}
        {AArch64::ARMV8A, {"nofp16", "fp16fml"}, {"fullfp16", "fp16fml"}, {}},
        {AArch64::ARMV8A, {"fp16fml", "nofp16"}, {}, {"fullfp16", "fp16fml"}},
        {AArch64::ARMV8A, {"nofp16", "sve"}, {"fullfp16", "sve"}, {}},
        {AArch64::ARMV8A, {"sve", "nofp16"}, {}, {"fullfp16", "sve"}},
        {AArch64::ARMV9_7A, {"nofp16", "f16mm"}, {"fullfp16", "f16mm"}, {}},
        {AArch64::ARMV9_7A, {"f16mm", "nofp16"}, {}, {"fullfp16", "f16mm"}},
        {AArch64::ARMV9_7A, {"nosimd", "f16mm"}, {"neon", "f16mm"}, {}},
        {AArch64::ARMV9_7A, {"f16mm", "nosimd"}, {}, {"neon", "f16mm"}},
        {AArch64::ARMV9_7A,
         {"nofp16", "f16f32mm"},
         {"fullfp16", "f16f32mm"},
         {}},
        {AArch64::ARMV9_7A,
         {"f16f32mm", "nofp16"},
         {},
         {"fullfp16", "f16f32mm"}},
        {AArch64::ARMV9_7A,
         {"nofp16", "f16f32dot"},
         {"fullfp16", "f16f32dot"},
         {}},
        {AArch64::ARMV9_7A,
         {"f16f32dot", "nofp16"},
         {},
         {"fullfp16", "f16f32dot"}},

        // bf16 -> {sme}
        {AArch64::ARMV8A, {"nobf16", "sme"}, {"bf16", "sme"}, {}},
        {AArch64::ARMV8A, {"sme", "nobf16"}, {}, {"bf16", "sme"}},

        // sve -> {sve2, f32mm, f64mm, sve-f16f32mm, sve-b16mm}
        {AArch64::ARMV8A, {"nosve", "sve2"}, {"sve", "sve2"}, {}},
        {AArch64::ARMV8A, {"sve2", "nosve"}, {}, {"sve", "sve2"}},
        {AArch64::ARMV8A, {"nosve", "f32mm"}, {"sve", "f32mm"}, {}},
        {AArch64::ARMV8A, {"f32mm", "nosve"}, {}, {"sve", "f32mm"}},
        {AArch64::ARMV8A, {"nosve", "f64mm"}, {"sve", "f64mm"}, {}},
        {AArch64::ARMV8A, {"f64mm", "nosve"}, {}, {"sve", "f64mm"}},
        {AArch64::ARMV9_6A,
         {"nosve", "sve-f16f32mm"},
         {"sve", "sve-f16f32mm"},
         {}},
        {AArch64::ARMV9_6A,
         {"sve-f16f32mm", "nosve"},
         {},
         {"sve", "sve-f16f32mm"}},
        {AArch64::ARMV9_7A, {"nosve", "sve-b16mm"}, {"sve", "sve-b16mm"}, {}},
        {AArch64::ARMV9_7A, {"sve-b16mm", "nosve"}, {}, {"sve", "sve-b16mm"}},

        // aes -> {sve-aes}
        {AArch64::ARMV8A, {"noaes", "sve-aes"}, {"aes", "sve-aes"}, {}},
        {AArch64::ARMV8A, {"sve-aes", "noaes"}, {}, {"aes", "sve-aes"}},

        // sve2 -> {sve2p1, sve2-bitperm, sve2-sha3, sve2-sm4, sve2-aes}
        {AArch64::ARMV8A, {"nosve2", "sve2p1"}, {"sve2", "sve2p1"}, {}},
        {AArch64::ARMV8A, {"sve2p1", "nosve2"}, {}, {"sve2", "sve2p1"}},
        {AArch64::ARMV8A,
         {"nosve2", "sve2-bitperm"},
         {"sve2", "sve-bitperm"},
         {}},
        {AArch64::ARMV8A,
         {"sve2-bitperm", "nosve2"},
         {"sve"},
         {"sve-bitperm", "sve2", "sve2-bitperm"}},
        {AArch64::ARMV8A,
         {"ssve-bitperm", "nosve-bitperm"},
         {"sme"},
         {"ssve-bitperm", "sve-bitperm"}},
        {AArch64::ARMV8A,
         {"nosve-bitperm", "ssve-bitperm"},
         {"sve-bitperm", "sve-bitperm"},
         {""}},
        {AArch64::ARMV8A, {"nosve2", "sve2-sha3"}, {"sve2", "sve2-sha3"}, {}},
        {AArch64::ARMV8A, {"sve2-sha3", "nosve2"}, {}, {"sve2", "sve2-sha3"}},
        {AArch64::ARMV8A, {"nosve2", "sve2-sm4"}, {"sve2", "sve2-sm4"}, {}},
        {AArch64::ARMV8A, {"sve2-sm4", "nosve2"}, {}, {"sve2", "sve2-sm4"}},
        {AArch64::ARMV8A, {"nosve2", "sve2-aes"}, {"sve2", "sve2-aes"}, {}},
        {AArch64::ARMV8A, {"sve2-aes", "nosve2"}, {}, {"sve2", "sve2-aes"}},

        // sve-b16b16 -> {sme-b16b16}
        {AArch64::ARMV9_4A,
         {"nosve-b16b16", "sme-b16b16"},
         {"sve-b16b16", "sme-b16b16"},
         {}},
        {AArch64::ARMV9_4A,
         {"sme-b16b16", "nosve-b16b16"},
         {},
         {"sve-b16b16", "sme-b16b16"}},

        // sve2p1 -> {sve2p2}
        {AArch64::ARMV9_6A, {"nosve2p1", "sve2p2"}, {"sve2p1", "sve2p2"}, {}},
        {AArch64::ARMV9_6A, {"sve2p2", "nosve2p1"}, {}, {"sve2p1", "sve2p2"}},

        // sve2p2 -> {sve2p3}
        {AArch64::ARMV9_7A, {"nosve2p2", "sve2p3"}, {"sve2p2", "sve2p3"}, {}},
        {AArch64::ARMV9_7A, {"sve2p3", "nosve2p2"}, {}, {"sve2p2", "sve2p3"}},

        // sme -> {sme2, sme-f16f16, sme-f64f64, sme-i16i64, sme-fa64}
        {AArch64::ARMV8A, {"nosme", "sme2"}, {"sme", "sme2"}, {}},
        {AArch64::ARMV8A, {"sme2", "nosme"}, {}, {"sme", "sme2"}},
        {AArch64::ARMV8A, {"nosme", "sme-f16f16"}, {"sme", "sme-f16f16"}, {}},
        {AArch64::ARMV8A, {"sme-f16f16", "nosme"}, {}, {"sme", "sme-f16f16"}},
        {AArch64::ARMV8A, {"nosme", "sme-f64f64"}, {"sme", "sme-f64f64"}, {}},
        {AArch64::ARMV8A, {"sme-f64f64", "nosme"}, {}, {"sme", "sme-f64f64"}},
        {AArch64::ARMV8A, {"nosme", "sme-i16i64"}, {"sme", "sme-i16i64"}, {}},
        {AArch64::ARMV8A, {"sme-i16i64", "nosme"}, {}, {"sme", "sme-i16i64"}},
        {AArch64::ARMV8A, {"nosme", "sme-fa64"}, {"sme", "sme-fa64"}, {}},
        {AArch64::ARMV8A, {"sme-fa64", "nosme"}, {}, {"sme", "sme-fa64"}},

        // sme2 -> {sme2p1, ssve-fp8fma, ssve-fp8dot2, ssve-fp8dot4, sme-f8f16,
        // sme-f8f32, sme-b16b16, ssve-aes}
        {AArch64::ARMV8A, {"nosme2", "sme2p1"}, {"sme2", "sme2p1"}, {}},
        {AArch64::ARMV8A, {"sme2p1", "nosme2"}, {}, {"sme2", "sme2p1"}},
        {AArch64::ARMV8A,
         {"nosme2", "ssve-fp8fma"},
         {"sme2", "ssve-fp8fma"},
         {}},
        {AArch64::ARMV8A,
         {"ssve-fp8fma", "nosme2"},
         {},
         {"sme2", "ssve-fp8fma"}},
        {AArch64::ARMV8A,
         {"nosme2", "ssve-fp8dot2"},
         {"sme2", "ssve-fp8dot2"},
         {}},
        {AArch64::ARMV8A,
         {"ssve-fp8dot2", "nosme2"},
         {},
         {"sme2", "ssve-fp8dot2"}},
        {AArch64::ARMV8A,
         {"nosme2", "ssve-fp8dot4"},
         {"sme2", "ssve-fp8dot4"},
         {}},
        {AArch64::ARMV8A,
         {"ssve-fp8dot4", "nosme2"},
         {},
         {"sme2", "ssve-fp8dot4"}},
        {AArch64::ARMV8A, {"nosme2", "sme-f8f16"}, {"sme2", "sme-f8f16"}, {}},
        {AArch64::ARMV8A, {"sme-f8f16", "nosme2"}, {}, {"sme2", "sme-f8f16"}},
        {AArch64::ARMV8A, {"nosme2", "sme-f8f32"}, {"sme2", "sme-f8f32"}, {}},
        {AArch64::ARMV8A, {"sme-f8f32", "nosme2"}, {}, {"sme2", "sme-f8f32"}},
        {AArch64::ARMV8A, {"nosme2", "sme-b16b16"}, {"sme2", "sme-b16b16"}, {}},
        {AArch64::ARMV8A, {"sme-b16b16", "nosme2"}, {}, {"sme2", "sme-b16b16"}},
        {AArch64::ARMV9_6A, {"nosme2", "ssve-aes"}, {"sme2", "ssve-aes"}, {}},
        {AArch64::ARMV9_6A, {"ssve-aes", "nosme2"}, {}, {"ssve-aes", "sme2"}},

        // sme2p1 -> {sme2p2}
        {AArch64::ARMV9_6A, {"nosme2p1", "sme2p2"}, {"sme2p2", "sme2p1"}, {}},
        {AArch64::ARMV9_6A, {"sme2p2", "nosme2p1"}, {}, {"sme2p1", "sme2p2"}},

        // sme2p2 -> {sme2p3}
        {AArch64::ARMV9_7A, {"nosme2p2", "sme2p3"}, {"sme2p3", "sme2p2"}, {}},
        {AArch64::ARMV9_7A, {"sme2p3", "nosme2p2"}, {}, {"sme2p2", "sme2p3"}},

        // fp8 -> {sme-f8f16, sme-f8f32, f8f16mm, f8f32mm, fp8dot4, fp8dot2,
        // ssve-fp8dot4, ssve-fp8dot2}
        {AArch64::ARMV8A, {"nofp8", "sme-f8f16"}, {"fp8", "sme-f8f16"}, {}},
        {AArch64::ARMV8A, {"sme-f8f16", "nofp8"}, {}, {"fp8", "sme-f8f16"}},
        {AArch64::ARMV8A, {"nofp8", "sme-f8f32"}, {"fp8", "sme-f8f32"}, {}},
        {AArch64::ARMV8A, {"sme-f8f32", "nofp8"}, {}, {"fp8", "sme-f8f32"}},
        {AArch64::ARMV9_6A, {"nofp8", "f8f16mm"}, {"fp8", "f8f16mm"}, {}},
        {AArch64::ARMV9_6A, {"f8f16mm", "nofp8"}, {}, {"fp8", "f8f16mm"}},
        {AArch64::ARMV9_6A, {"nofp8", "f8f32mm"}, {"fp8", "f8f32mm"}, {}},
        {AArch64::ARMV9_6A, {"f8f32mm", "nofp8"}, {}, {"fp8", "f8f32mm"}},
        {AArch64::ARMV9_6A, {"nofp8", "fp8dot4"}, {"fp8", "fp8dot4"}, {}},
        {AArch64::ARMV9_6A, {"fp8dot4", "nofp8"}, {}, {"fp8", "fp8dot4"}},
        {AArch64::ARMV9_6A, {"nofp8", "fp8dot2"}, {"fp8", "fp8dot2"}, {}},
        {AArch64::ARMV9_6A, {"fp8dot2", "nofp8"}, {}, {"fp8", "fp8dot2"}},
        {AArch64::ARMV9_6A,
         {"nofp8", "ssve-fp8dot4"},
         {"fp8", "ssve-fp8dot4"},
         {}},
        {AArch64::ARMV9_6A,
         {"ssve-fp8dot4", "nofp8"},
         {},
         {"fp8", "ssve-fp8dot4"}},
        {AArch64::ARMV9_6A,
         {"nofp8", "ssve-fp8dot2"},
         {"fp8", "ssve-fp8dot2"},
         {}},
        {AArch64::ARMV9_6A,
         {"ssve-fp8dot2", "nofp8"},
         {},
         {"fp8", "ssve-fp8dot2"}},

        // lse -> lse128
        {AArch64::ARMV8A, {"nolse", "lse128"}, {"lse", "lse128"}, {}},
        {AArch64::ARMV8A, {"lse128", "nolse"}, {}, {"lse", "lse128"}},

        // mtetc -> mte
        {AArch64::ARMV9_7A, {"nomemtag", "mtetc"}, {"mte"}, {}},

        // mops-go -> mops, mte
        {AArch64::ARMV9_7A,
         {"nomops", "nomemtag", "mops-go"},
         {"mops", "mte"},
         {}},

        // predres -> predres2
        {AArch64::ARMV8A,
         {"nopredres", "predres2"},
         {"predres", "specres2"},
         {}},
        {AArch64::ARMV8A,
         {"predres2", "nopredres"},
         {},
         {"predres", "specres2"}},

        // ras -> ras2
        {AArch64::ARMV8A, {"noras", "rasv2"}, {"ras", "rasv2"}, {}},
        {AArch64::ARMV8A, {"rasv2", "noras"}, {}, {"ras", "rasv2"}},

        // rcpc -> rcpc3
        {AArch64::ARMV8A, {"norcpc", "rcpc3"}, {"rcpc", "rcpc3"}, {}},
        {AArch64::ARMV8A, {"rcpc3", "norcpc"}, {}, {"rcpc", "rcpc3"}},

        // sve-aes -> {ssve-aes, sve2-aes}
        {AArch64::ARMV9_6A,
         {"nosve-aes", "ssve-aes"},
         {"sve-aes", "ssve-aes"},
         {}},
        {AArch64::ARMV9_6A,
         {"ssve-aes", "nosve-aes"},
         {},
         {"ssve-aes", "sve-aes"}},
        {AArch64::ARMV9_6A,
         {"nosve-aes", "sve2-aes"},
         {"sve2-aes", "sve-aes"},
         {}},
        {AArch64::ARMV9_6A,
         {"sve2-aes", "nosve-aes"},
         {},
         {"sve2-aes", "sve-aes"}},

        // -sve2-aes should disable sve-aes (only)
        {AArch64::ARMV9_6A,
         {"sve2", "sve-aes", "nosve2-aes"},
         {"sve2"},
         {"sve2-aes", "sve-aes"}},

        // -sve2-sm4 should disable sve-sm4 (only)
        {AArch64::ARMV9_6A,
         {"sve2", "sve-sm4", "nosve2-sm4"},
         {"sve2"},
         {"sve2-sm4", "sve-sm4"}},

        // -sve2-sha3 should disable sve-sha3 (only)
        {AArch64::ARMV9_6A,
         {"sve2", "sve-sha3", "nosve2-sha3"},
         {"sve2"},
         {"sve2-sha3", "sve-sha3"}},

        // sme-tmop -> sme
        {AArch64::ARMV8A, {"nosme2", "sme-tmop"}, {"sme2", "sme-tmop"}, {}},
        {AArch64::ARMV8A, {"sme-tmop", "nosme2"}, {}, {"sme2", "sme-tmop"}},

        // sme-mop4 -> sme
        {AArch64::ARMV8A, {"nosme", "sme-mop4"}, {"sme", "sme-mop4"}, {}},
        {AArch64::ARMV8A, {"sme-mop4", "nosme"}, {}, {"sme", "sme-mop4"}}};

INSTANTIATE_TEST_SUITE_P(
    AArch64ExtensionDependenciesBaseArch,
    AArch64ExtensionDependenciesBaseArchTestFixture,
    ::testing::ValuesIn(AArch64ExtensionDependenciesArchData));

AArch64ExtensionDependenciesBaseCPUTestParams
    AArch64ExtensionDependenciesCPUData[] = {
        // Base CPU features
        {"cortex-a57",
         {},
         {"v8a", "aes", "crc", "fp-armv8", "sha2", "neon"},
         {}},
        {"cortex-r82",
         {},
         {"v8r", "crc", "dotprod", "fp-armv8", "fullfp16", "fp16fml", "lse",
          "ras", "rcpc", "rdm", "sb", "neon", "ssbs"},
         {}},
        {"cortex-a520",
         {},
         {"v9.2a",    "bf16",    "crc",  "dotprod",     "flagm", "fp-armv8",
          "fullfp16", "fp16fml", "i8mm", "lse",         "mte",   "pauth",
          "perfmon",  "predres", "ras",  "rcpc",        "rdm",   "sb",
          "neon",     "ssbs",    "sve",  "sve-bitperm", "sve2"},
         {}},

        // Negative modifiers
        {"cortex-r82",
         {"nofp"},
         {"v8r", "crc", "lse", "ras", "rcpc", "sb", "ssbs"},
         {"fp-armv8", "neon", "fullfp16", "fp16fml", "dotprod", "rdm"}},
};

INSTANTIATE_TEST_SUITE_P(
    AArch64ExtensionDependenciesBaseCPU,
    AArch64ExtensionDependenciesBaseCPUTestFixture,
    ::testing::ValuesIn(AArch64ExtensionDependenciesCPUData));

struct CheckFindSinglePrecisionFpuTest {
  StringRef Cpu;
  ARM::ArchKind Arch;
  StringRef Archext;
  std::vector<StringRef> Features;
  ARM::FPUKind Fpu;
  ARM::FPUKind Output;
};

TEST(TargetParserTest, checkFindSinglePrecisionFPU) {
  CheckFindSinglePrecisionFpuTest tests[] = {
      {"cortex-r4f",
       ARM::ArchKind::ARMV7R,
       "nofp.dp",
       {},
       ARM::FK_INVALID,
       ARM::FK_VFPV3XD},
      {"cortex-r7",
       ARM::ArchKind::ARMV7R,
       "nofp.dp",
       {},
       ARM::FK_INVALID,
       ARM::FK_VFPV3XD_FP16},
      {"cortex-a7",
       ARM::ArchKind::ARMV7A,
       "nofp.dp",
       {},
       ARM::FK_INVALID,
       ARM::FK_FPV4_SP_D16},
      {"cortex-r52",
       ARM::ArchKind::ARMV8R,
       "nofp.dp",
       {},
       ARM::FK_INVALID,
       ARM::FK_FPV5_SP_D16},
      {"cortex-m55",
       ARM::ArchKind::ARMV8_1MMainline,
       "nofp.dp",
       {},
       ARM::FK_INVALID,
       ARM::FK_FP_ARMV8_FULLFP16_SP_D16}};

  for (auto X : tests) {
    ARM::FPUKind FPU = X.Fpu;
    EXPECT_TRUE(
        ARM::appendArchExtFeatures(X.Cpu, X.Arch, X.Archext, X.Features, FPU));
    EXPECT_EQ(FPU, X.Output);
  }
}

TEST(TargetParserTest, testAMDGPUArch) {
  EXPECT_EQ(Triple("amdgpu--").getArch(), Triple::amdgpu);
  EXPECT_EQ(Triple("amdgpu--").getSubArch(), Triple::NoSubArch);
  EXPECT_EQ(Triple("amdgpu0--").getSubArch(), Triple::NoSubArch);
  EXPECT_EQ(Triple("amdgpufoo--").getSubArch(), Triple::NoSubArch);

  {
    EXPECT_EQ(Triple("amdgpu6--").getSubArch(), Triple::AMDGPUSubArch6);
    EXPECT_EQ(Triple("amdgpu6.0").getSubArch(), Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu6.00").getSubArch(), Triple::AMDGPUSubArch600);
    EXPECT_EQ(Triple("amdgpu6.01").getSubArch(), Triple::AMDGPUSubArch601);
    EXPECT_EQ(Triple("amdgpu6.02").getSubArch(), Triple::AMDGPUSubArch602);
    EXPECT_EQ(Triple("amdgpu6.03").getSubArch(), Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu60").getSubArch(), Triple::NoSubArch);   // Unknown
  }

  {
    EXPECT_EQ(Triple("amdgpu7--").getSubArch(), Triple::AMDGPUSubArch7);
    EXPECT_EQ(Triple("amdgpu7.0--").getSubArch(), Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu7.00").getSubArch(), Triple::AMDGPUSubArch700);
    EXPECT_EQ(Triple("amdgpu7.01").getSubArch(), Triple::AMDGPUSubArch701);
    EXPECT_EQ(Triple("amdgpu7.02").getSubArch(), Triple::AMDGPUSubArch702);
    EXPECT_EQ(Triple("amdgpu7.03").getSubArch(), Triple::AMDGPUSubArch703);
    EXPECT_EQ(Triple("amdgpu7.04").getSubArch(), Triple::AMDGPUSubArch704);
    EXPECT_EQ(Triple("amdgpu7.05").getSubArch(), Triple::AMDGPUSubArch705);
    EXPECT_EQ(Triple("amdgpu7.06").getSubArch(), Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu71").getSubArch(), Triple::NoSubArch);   // Unknown
  }

  {
    EXPECT_EQ(Triple("amdgpu8--").getSubArch(), Triple::AMDGPUSubArch8);
    EXPECT_EQ(Triple("amdgpu8.0--").getSubArch(), Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu8.01--").getSubArch(), Triple::AMDGPUSubArch801);
    EXPECT_EQ(Triple("amdgpu8.02--").getSubArch(), Triple::AMDGPUSubArch802);
    EXPECT_EQ(Triple("amdgpu8.03--").getSubArch(), Triple::AMDGPUSubArch803);
    EXPECT_EQ(Triple("amdgpu8.05--").getSubArch(), Triple::AMDGPUSubArch805);
    EXPECT_EQ(Triple("amdgpu8.1--").getSubArch(), Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu8.10--").getSubArch(), Triple::AMDGPUSubArch810);
    EXPECT_EQ(Triple("amdgpu81--").getSubArch(), Triple::NoSubArch); // Unknown
  }

  {
    EXPECT_EQ(Triple("amdgpu9--").getArch(), Triple::amdgpu);
    EXPECT_EQ(Triple("amdgpu9--").getSubArch(), Triple::AMDGPUSubArch9);
    EXPECT_EQ(Triple("amdgpu9.0--").getSubArch(), Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu9.00--").getSubArch(), Triple::AMDGPUSubArch900);
    EXPECT_EQ(Triple("amdgpu9.02--").getSubArch(), Triple::AMDGPUSubArch902);
    EXPECT_EQ(Triple("amdgpu9.04--").getSubArch(), Triple::AMDGPUSubArch904);
    EXPECT_EQ(Triple("amdgpu9.06--").getSubArch(), Triple::AMDGPUSubArch906);
    EXPECT_EQ(Triple("amdgpu9.08--").getSubArch(), Triple::AMDGPUSubArch908);
    EXPECT_EQ(Triple("amdgpu9.0a--").getSubArch(), Triple::AMDGPUSubArch90A);
    EXPECT_EQ(Triple("amdgpu9.0c--").getSubArch(), Triple::AMDGPUSubArch90C);

    EXPECT_EQ(Triple("amdgpu9.4--").getSubArch(), Triple::AMDGPUSubArch9_4);
    EXPECT_EQ(Triple("amdgpu9.42--").getSubArch(), Triple::AMDGPUSubArch942);
    EXPECT_EQ(Triple("amdgpu9.50--").getSubArch(), Triple::AMDGPUSubArch950);
    EXPECT_EQ(Triple("amdgpu99").getSubArch(), Triple::NoSubArch); // Unknown
  }

  {
    EXPECT_EQ(Triple("amdgpu10--").getArch(), Triple::amdgpu);
    EXPECT_EQ(Triple("amdgpu10--").getSubArch(), Triple::AMDGPUSubArch10_1);
    EXPECT_EQ(Triple("amdgpu10.1--").getSubArch(), Triple::AMDGPUSubArch10_1);
    EXPECT_EQ(Triple("amdgpu10.10--").getSubArch(), Triple::AMDGPUSubArch1010);
    EXPECT_EQ(Triple("amdgpu10.11--").getSubArch(), Triple::AMDGPUSubArch1011);
    EXPECT_EQ(Triple("amdgpu10.12--").getSubArch(), Triple::AMDGPUSubArch1012);
    EXPECT_EQ(Triple("amdgpu10.13--").getSubArch(), Triple::AMDGPUSubArch1013);
    EXPECT_EQ(Triple("amdgpu101").getSubArch(), Triple::NoSubArch); // Unknown
  }

  {
    EXPECT_EQ(Triple("amdgpu10.3--").getSubArch(), Triple::AMDGPUSubArch10_3);
    EXPECT_EQ(Triple("amdgpu10.30--").getSubArch(), Triple::AMDGPUSubArch1030);
    EXPECT_EQ(Triple("amdgpu10.31--").getSubArch(), Triple::AMDGPUSubArch1031);
    EXPECT_EQ(Triple("amdgpu10.32--").getSubArch(), Triple::AMDGPUSubArch1032);
    EXPECT_EQ(Triple("amdgpu10.33--").getSubArch(), Triple::AMDGPUSubArch1033);
    EXPECT_EQ(Triple("amdgpu10.34--").getSubArch(), Triple::AMDGPUSubArch1034);
    EXPECT_EQ(Triple("amdgpu10.35--").getSubArch(), Triple::AMDGPUSubArch1035);
    EXPECT_EQ(Triple("amdgpu10.36--").getSubArch(), Triple::AMDGPUSubArch1036);
    EXPECT_EQ(Triple("amdgpu10.39").getSubArch(), Triple::NoSubArch); // Unknown
  }

  {
    EXPECT_EQ(Triple("amdgpu11--").getSubArch(), Triple::AMDGPUSubArch11);
    EXPECT_EQ(Triple("amdgpu11.0--").getSubArch(),
              Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu11.00--").getSubArch(), Triple::AMDGPUSubArch1100);
    EXPECT_EQ(Triple("amdgpu11.01--").getSubArch(), Triple::AMDGPUSubArch1101);
    EXPECT_EQ(Triple("amdgpu11.02--").getSubArch(), Triple::AMDGPUSubArch1102);
    EXPECT_EQ(Triple("amdgpu11.03--").getSubArch(), Triple::AMDGPUSubArch1103);
    EXPECT_EQ(Triple("amdgpu11.5--").getSubArch(),
              Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu11.50--").getSubArch(), Triple::AMDGPUSubArch1150);
    EXPECT_EQ(Triple("amdgpu11.51--").getSubArch(), Triple::AMDGPUSubArch1151);
    EXPECT_EQ(Triple("amdgpu11.52--").getSubArch(), Triple::AMDGPUSubArch1152);
    EXPECT_EQ(Triple("amdgpu11.53--").getSubArch(), Triple::AMDGPUSubArch1153);
    EXPECT_EQ(Triple("amdgpu11.54--").getSubArch(), Triple::AMDGPUSubArch1154);
    EXPECT_EQ(Triple("amdgpu11.7--").getSubArch(), Triple::AMDGPUSubArch11_7);
    EXPECT_EQ(Triple("amdgpu11.70--").getSubArch(), Triple::AMDGPUSubArch1170);
    EXPECT_EQ(Triple("amdgpu11.71--").getSubArch(), Triple::AMDGPUSubArch1171);
    EXPECT_EQ(Triple("amdgpu11.72--").getSubArch(), Triple::AMDGPUSubArch1172);
    EXPECT_EQ(Triple("amdgpu111--").getSubArch(), Triple::NoSubArch); // Unknown
  }

  {
    EXPECT_EQ(Triple("amdgpu12--").getSubArch(), Triple::AMDGPUSubArch12);
    EXPECT_EQ(Triple("amdgpu12.0--").getSubArch(),
              Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu12.00--").getSubArch(), Triple::AMDGPUSubArch1200);
    EXPECT_EQ(Triple("amdgpu12.01--").getSubArch(), Triple::AMDGPUSubArch1201);
    EXPECT_EQ(Triple("amdgpu12.5--").getSubArch(), Triple::AMDGPUSubArch12_5);
    EXPECT_EQ(Triple("amdgpu12.50--").getSubArch(), Triple::AMDGPUSubArch1250);
    EXPECT_EQ(Triple("amdgpu12.51--").getSubArch(), Triple::AMDGPUSubArch1251);
    EXPECT_EQ(Triple("amdgpu122--").getSubArch(), Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu12.59--").getSubArch(),
              Triple::NoSubArch); // Unknown
  }

  {
    EXPECT_EQ(Triple("amdgpu13--").getSubArch(), Triple::AMDGPUSubArch13);
    EXPECT_EQ(Triple("amdgpu13.10--").getSubArch(), Triple::AMDGPUSubArch1310);
    EXPECT_EQ(Triple("amdgpu13.1").getSubArch(), Triple::NoSubArch); // Unknown
    EXPECT_EQ(Triple("amdgpu131").getSubArch(), Triple::NoSubArch);  // Unknown
  }
}

TEST(TargetParserTest, testAMDGPUgetMajorSubArch) {
  // Make sure every subarch is handled by the function
  for (unsigned SubArch = Triple::FirstAMDGPUSubArch;
       SubArch <= Triple::LastAMDGPUSubArch; ++SubArch) {

    EXPECT_NE(
        AMDGPU::getMajorSubArch(static_cast<Triple::SubArchType>(SubArch)),
        Triple::NoSubArch);
  }
}

TEST(TargetParserTest, testAMDGPUisSubArchCompatible) {
  // A major-family subarch is compatible with itself and with any specific
  // subarch in its family.
  EXPECT_TRUE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch9,
                                          Triple::AMDGPUSubArch9));
  EXPECT_TRUE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch9,
                                          Triple::AMDGPUSubArch900));
  EXPECT_TRUE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch900,
                                          Triple::AMDGPUSubArch9));
  EXPECT_TRUE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch9,
                                          Triple::AMDGPUSubArch90C));

  // NoSubArch is compatible with anything.
  EXPECT_TRUE(
      AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch9, Triple::NoSubArch));
  EXPECT_TRUE(
      AMDGPU::isSubArchCompatible(Triple::NoSubArch, Triple::AMDGPUSubArch9));
  EXPECT_TRUE(
      AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch900, Triple::NoSubArch));
  EXPECT_TRUE(
      AMDGPU::isSubArchCompatible(Triple::NoSubArch, Triple::AMDGPUSubArch900));

  // gfx9 and gfx9.4 are distinct major families.
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch9,
                                           Triple::AMDGPUSubArch9_4));
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch9,
                                           Triple::AMDGPUSubArch942));

  // gfx908 and gfx90a are their own major families, not part of gfx9-generic,
  // because they have ABI-incompatible feature differences (e.g. gfx90a
  // requires aligned VGPRs).
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch9,
                                           Triple::AMDGPUSubArch908));
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch9,
                                           Triple::AMDGPUSubArch90A));
  EXPECT_TRUE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch90A,
                                          Triple::AMDGPUSubArch90A));

  // gfx11 and gfx11.7 are distinct major families.
  EXPECT_TRUE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch11_7,
                                          Triple::AMDGPUSubArch11_7));
  EXPECT_TRUE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch11_7,
                                          Triple::AMDGPUSubArch1170));
  EXPECT_TRUE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch1172,
                                          Triple::AMDGPUSubArch11_7));
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch11,
                                           Triple::AMDGPUSubArch11_7));
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch11,
                                           Triple::AMDGPUSubArch1170));

  // Two specific subarches in the same family are not compatible with each
  // other; only the major-family subarch is.
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch900,
                                           Triple::AMDGPUSubArch90A));

  // Different major families are incompatible.
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch9,
                                           Triple::AMDGPUSubArch10_3));
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch10_3,
                                           Triple::AMDGPUSubArch9));
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch900,
                                           Triple::AMDGPUSubArch1030));
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch1030,
                                           Triple::AMDGPUSubArch900));

  // NoSubArch is compatible with anything.
  EXPECT_TRUE(
      AMDGPU::isSubArchCompatible(Triple::NoSubArch, Triple::NoSubArch));
  EXPECT_TRUE(
      AMDGPU::isSubArchCompatible(Triple::NoSubArch, Triple::AMDGPUSubArch9));
  EXPECT_TRUE(
      AMDGPU::isSubArchCompatible(Triple::AMDGPUSubArch9, Triple::NoSubArch));
  EXPECT_TRUE(
      AMDGPU::isSubArchCompatible(Triple::NoSubArch, Triple::AMDGPUSubArch900));

  // The Triple-based overload checks the same compatibility.
  EXPECT_TRUE(AMDGPU::isSubArchCompatible(Triple("amdgpu9-amd-amdhsa"),
                                          Triple("amdgpu9.00-amd-amdhsa")));
  EXPECT_TRUE(AMDGPU::isSubArchCompatible(Triple("amdgpu-amd-amdhsa"),
                                          Triple("amdgpu9.00-amd-amdhsa")));

  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple("amdgpu9.00-amd-amdhsa"),
                                           Triple("amdgpu9.0a-amd-amdhsa")));

  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple("amdgpu9.0a-amd-amdhsa"),
                                           Triple("amdgpu9.00-amd-amdhsa")));

  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple("amdgpu9-amd-amdhsa"),
                                           Triple("amdgpu10.3-amd-amdhsa")));
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple("amdgpu10.3-amd-amdhsa"),
                                           Triple("amdgpu9-amd-amdhsa")));

  // An unrecognized subarch is incompatible with any recognized subarch.
  EXPECT_FALSE(AMDGPU::isSubArchCompatible(Triple("amdgpu9.99-amd-amdhsa"),
                                           Triple("amdgpu9.00-amd-amdhsa")));
}

TEST(TargetParserTest, testAMDGPUisCPUValidForSubArch) {
  EXPECT_TRUE(
      AMDGPU::isCPUValidForSubArch(Triple::AMDGPUSubArch9, AMDGPU::GK_GFX900));
  EXPECT_TRUE(AMDGPU::isCPUValidForSubArch(Triple::AMDGPUSubArch9, "gfx900"));

  EXPECT_TRUE(AMDGPU::isCPUValidForSubArch(Triple::AMDGPUSubArch900,
                                           AMDGPU::GK_GFX900));
  EXPECT_TRUE(AMDGPU::isCPUValidForSubArch(Triple::AMDGPUSubArch900, "gfx900"));

  EXPECT_FALSE(AMDGPU::isCPUValidForSubArch(Triple::AMDGPUSubArch900,
                                            AMDGPU::GK_GFX902));
  EXPECT_FALSE(
      AMDGPU::isCPUValidForSubArch(Triple::AMDGPUSubArch900, "gfx902"));

  EXPECT_TRUE(
      AMDGPU::isCPUValidForSubArch(Triple::NoSubArch, AMDGPU::GK_GFX900));
  EXPECT_TRUE(AMDGPU::isCPUValidForSubArch(Triple::NoSubArch, "gfx900"));

  EXPECT_FALSE(
      AMDGPU::isCPUValidForSubArch(Triple::NoSubArch, AMDGPU::GK_NONE));
  EXPECT_FALSE(AMDGPU::isCPUValidForSubArch(Triple::NoSubArch, ""));
}

TEST(TargetParserTest, testAMDGPUfillValidArchListAMDGCN) {
  SmallVector<StringRef, 0> All;
  AMDGPU::fillValidArchListAMDGCN(All, Triple::NoSubArch);
  EXPECT_FALSE(All.empty());

  SmallVector<StringRef, 16> Filtered;

  // For every subarch, the filtered list must agree exactly with
  // isCPUValidForSubArch over the full set of names.
  for (unsigned SA = Triple::FirstAMDGPUSubArch;
       SA <= Triple::LastAMDGPUSubArch; ++SA) {
    auto SubArch = static_cast<Triple::SubArchType>(SA);
    Triple::SubArchType Major = AMDGPU::getMajorSubArch(SubArch);

    AMDGPU::fillValidArchListAMDGCN(Filtered, SubArch);

    for (StringRef Name : All) {
      AMDGPU::GPUKind Kind = AMDGPU::parseArchAMDGCN(Name);
      bool Valid = AMDGPU::isCPUValidForSubArch(SubArch, Kind);
      EXPECT_EQ(Valid, llvm::is_contained(Filtered, Name))
          << "subarch " << Triple::getArchName(Triple::amdgpu, SubArch)
          << " name " << Name;
    }

    for (StringRef Name : Filtered)
      EXPECT_TRUE(llvm::is_contained(All, Name)) << Name;

    // A specific subarch matches its own concrete GPU, plus the family-generic
    // GPU. Names may repeat through aliases, so compare distinct GPUKinds.
    if (Major != SubArch) {
      SmallSet<AMDGPU::GPUKind, 2> Kinds;
      for (StringRef Name : Filtered)
        Kinds.insert(AMDGPU::parseArchAMDGCN(Name));

      bool FamilyHasGeneric =
          AMDGPU::getGPUKindFromSubArch(Major) != AMDGPU::GK_NONE;
      EXPECT_EQ(Kinds.size(), FamilyHasGeneric ? 2u : 1u)
          << "subarch " << Triple::getArchName(Triple::amdgpu, SubArch);
    }

    Filtered.clear();
  }
}

TEST(TargetParserTest, testAMDGPUgetGPUKindFromSubArch) {
  EXPECT_EQ(AMDGPU::getGPUKindFromSubArch(Triple::NoSubArch), AMDGPU::GK_NONE);

  struct SubArchMapping {
    Triple::SubArchType SubArch;
    AMDGPU::GPUKind Kind;
  };

  static const SubArchMapping Mappings[] = {
      // GFX6 family
      {Triple::AMDGPUSubArch6, AMDGPU::GK_NONE},
      {Triple::AMDGPUSubArch600, AMDGPU::GK_GFX600},
      {Triple::AMDGPUSubArch601, AMDGPU::GK_GFX601},
      {Triple::AMDGPUSubArch602, AMDGPU::GK_GFX602},

      // GFX7 family
      {Triple::AMDGPUSubArch7, AMDGPU::GK_NONE},
      {Triple::AMDGPUSubArch700, AMDGPU::GK_GFX700},
      {Triple::AMDGPUSubArch701, AMDGPU::GK_GFX701},
      {Triple::AMDGPUSubArch702, AMDGPU::GK_GFX702},
      {Triple::AMDGPUSubArch703, AMDGPU::GK_GFX703},
      {Triple::AMDGPUSubArch704, AMDGPU::GK_GFX704},
      {Triple::AMDGPUSubArch705, AMDGPU::GK_GFX705},

      // GFX8 family
      {Triple::AMDGPUSubArch8, AMDGPU::GK_NONE},
      {Triple::AMDGPUSubArch801, AMDGPU::GK_GFX801},
      {Triple::AMDGPUSubArch802, AMDGPU::GK_GFX802},
      {Triple::AMDGPUSubArch803, AMDGPU::GK_GFX803},
      {Triple::AMDGPUSubArch805, AMDGPU::GK_GFX805},
      {Triple::AMDGPUSubArch810, AMDGPU::GK_GFX810},

      // GFX9 family
      {Triple::AMDGPUSubArch9, AMDGPU::GK_GFX9_GENERIC},
      {Triple::AMDGPUSubArch900, AMDGPU::GK_GFX900},
      {Triple::AMDGPUSubArch902, AMDGPU::GK_GFX902},
      {Triple::AMDGPUSubArch904, AMDGPU::GK_GFX904},
      {Triple::AMDGPUSubArch906, AMDGPU::GK_GFX906},
      {Triple::AMDGPUSubArch908, AMDGPU::GK_GFX908},
      {Triple::AMDGPUSubArch909, AMDGPU::GK_GFX909},
      {Triple::AMDGPUSubArch90A, AMDGPU::GK_GFX90A},
      {Triple::AMDGPUSubArch90C, AMDGPU::GK_GFX90C},
      {Triple::AMDGPUSubArch9_4, AMDGPU::GK_GFX9_4_GENERIC},
      {Triple::AMDGPUSubArch942, AMDGPU::GK_GFX942},
      {Triple::AMDGPUSubArch950, AMDGPU::GK_GFX950},

      // GFX10 family
      {Triple::AMDGPUSubArch10_1, AMDGPU::GK_GFX10_1_GENERIC},
      {Triple::AMDGPUSubArch1010, AMDGPU::GK_GFX1010},
      {Triple::AMDGPUSubArch1011, AMDGPU::GK_GFX1011},
      {Triple::AMDGPUSubArch1012, AMDGPU::GK_GFX1012},
      {Triple::AMDGPUSubArch1013, AMDGPU::GK_GFX1013},
      {Triple::AMDGPUSubArch10_3, AMDGPU::GK_GFX10_3_GENERIC},
      {Triple::AMDGPUSubArch1030, AMDGPU::GK_GFX1030},
      {Triple::AMDGPUSubArch1031, AMDGPU::GK_GFX1031},
      {Triple::AMDGPUSubArch1032, AMDGPU::GK_GFX1032},
      {Triple::AMDGPUSubArch1033, AMDGPU::GK_GFX1033},
      {Triple::AMDGPUSubArch1034, AMDGPU::GK_GFX1034},
      {Triple::AMDGPUSubArch1035, AMDGPU::GK_GFX1035},
      {Triple::AMDGPUSubArch1036, AMDGPU::GK_GFX1036},

      // GFX11 family
      {Triple::AMDGPUSubArch11, AMDGPU::GK_GFX11_GENERIC},
      {Triple::AMDGPUSubArch1100, AMDGPU::GK_GFX1100},
      {Triple::AMDGPUSubArch1101, AMDGPU::GK_GFX1101},
      {Triple::AMDGPUSubArch1102, AMDGPU::GK_GFX1102},
      {Triple::AMDGPUSubArch1103, AMDGPU::GK_GFX1103},
      {Triple::AMDGPUSubArch1150, AMDGPU::GK_GFX1150},
      {Triple::AMDGPUSubArch1151, AMDGPU::GK_GFX1151},
      {Triple::AMDGPUSubArch1152, AMDGPU::GK_GFX1152},
      {Triple::AMDGPUSubArch1153, AMDGPU::GK_GFX1153},
      {Triple::AMDGPUSubArch1154, AMDGPU::GK_GFX1154},
      {Triple::AMDGPUSubArch11_7, AMDGPU::GK_GFX11_7_GENERIC},
      {Triple::AMDGPUSubArch1170, AMDGPU::GK_GFX1170},
      {Triple::AMDGPUSubArch1171, AMDGPU::GK_GFX1171},
      {Triple::AMDGPUSubArch1172, AMDGPU::GK_GFX1172},

      // GFX12 family
      {Triple::AMDGPUSubArch12, AMDGPU::GK_GFX12_GENERIC},
      {Triple::AMDGPUSubArch1200, AMDGPU::GK_GFX1200},
      {Triple::AMDGPUSubArch1201, AMDGPU::GK_GFX1201},
      {Triple::AMDGPUSubArch12_5, AMDGPU::GK_GFX12_5_GENERIC},
      {Triple::AMDGPUSubArch1250, AMDGPU::GK_GFX1250},
      {Triple::AMDGPUSubArch1251, AMDGPU::GK_GFX1251},

      // GFX13 family
      {Triple::AMDGPUSubArch13, AMDGPU::GK_GFX13_GENERIC},
      {Triple::AMDGPUSubArch1310, AMDGPU::GK_GFX1310},
  };

  for (const auto &Mapping : Mappings)
    EXPECT_EQ(AMDGPU::getGPUKindFromSubArch(Mapping.SubArch), Mapping.Kind);
}

TEST(TargetParserTest, testAMDGPUgetIsaVersionFromSubArch) {
  // Concrete subarches map to their exact ISA version.
  EXPECT_EQ(AMDGPU::getIsaVersion(Triple::AMDGPUSubArch600),
            (AMDGPU::IsaVersion{6, 0, 0}));
  EXPECT_EQ(AMDGPU::getIsaVersion(Triple::AMDGPUSubArch900),
            (AMDGPU::IsaVersion{9, 0, 0}));
  EXPECT_EQ(AMDGPU::getIsaVersion(Triple::AMDGPUSubArch1250),
            (AMDGPU::IsaVersion{12, 5, 0}));
  EXPECT_EQ(AMDGPU::getIsaVersion(Triple::AMDGPUSubArch1251),
            (AMDGPU::IsaVersion{12, 5, 1}));
  EXPECT_EQ(AMDGPU::getIsaVersion(Triple::AMDGPUSubArch1310),
            (AMDGPU::IsaVersion{13, 1, 0}));

  // Generic subarches map to the generic target's version.
  EXPECT_EQ(AMDGPU::getIsaVersion(Triple::AMDGPUSubArch9),
            (AMDGPU::IsaVersion{9, 0, 0}));
  EXPECT_EQ(AMDGPU::getIsaVersion(Triple::AMDGPUSubArch12),
            (AMDGPU::IsaVersion{12, 0, 0}));
  EXPECT_EQ(AMDGPU::getIsaVersion(Triple::AMDGPUSubArch12_5),
            (AMDGPU::IsaVersion{12, 5, 0}));

  // Major-family subarches with no associated GPU have no version.
  EXPECT_EQ(AMDGPU::getIsaVersion(Triple::AMDGPUSubArch6),
            (AMDGPU::IsaVersion{0, 0, 0}));
  EXPECT_EQ(AMDGPU::getIsaVersion(Triple::AMDGPUSubArch13),
            (AMDGPU::IsaVersion{13, 1, 0}));
  EXPECT_EQ(AMDGPU::getIsaVersion(Triple::NoSubArch),
            (AMDGPU::IsaVersion{0, 0, 0}));
}

TEST(TargetParserTest, testAMDGPUgetNumSGPRs) {
  // getTotalNumSGPRs: 800 for GFX8+, 512 otherwise.
  EXPECT_EQ(AMDGPU::getTotalNumSGPRs(Triple::AMDGPUSubArch600), 512u);
  EXPECT_EQ(AMDGPU::getTotalNumSGPRs(Triple::AMDGPUSubArch700), 512u);
  EXPECT_EQ(AMDGPU::getTotalNumSGPRs(Triple::AMDGPUSubArch900), 800u);
  EXPECT_EQ(AMDGPU::getTotalNumSGPRs(Triple::AMDGPUSubArch1030), 800u);

  // getAddressableNumSGPRs resolves the SGPR init bug from the device kind:
  // 106 for GFX10+, 102 for GFX8/GFX9, 104 otherwise; gfx802/gfx805 have the
  // init bug and return the fixed size.
  EXPECT_EQ(AMDGPU::getAddressableNumSGPRs(Triple::AMDGPUSubArch600), 104u);
  EXPECT_EQ(AMDGPU::getAddressableNumSGPRs(Triple::AMDGPUSubArch803), 102u);
  EXPECT_EQ(AMDGPU::getAddressableNumSGPRs(Triple::AMDGPUSubArch900), 102u);
  EXPECT_EQ(AMDGPU::getAddressableNumSGPRs(Triple::AMDGPUSubArch1030), 106u);
  EXPECT_EQ(AMDGPU::getAddressableNumSGPRs(Triple::AMDGPUSubArch802),
            AMDGPU::FIXED_NUM_SGPRS_FOR_INIT_BUG);
  EXPECT_EQ(AMDGPU::getAddressableNumSGPRs(Triple::AMDGPUSubArch805),
            AMDGPU::FIXED_NUM_SGPRS_FOR_INIT_BUG);

  // The GPUKind overloads resolve to the same values.
  EXPECT_EQ(AMDGPU::getTotalNumSGPRs(AMDGPU::GK_GFX600), 512u);
  EXPECT_EQ(AMDGPU::getTotalNumSGPRs(AMDGPU::GK_GFX900), 800u);
  EXPECT_EQ(AMDGPU::getAddressableNumSGPRs(AMDGPU::GK_GFX900), 102u);
  EXPECT_EQ(AMDGPU::getAddressableNumSGPRs(AMDGPU::GK_GFX1030), 106u);
  EXPECT_EQ(AMDGPU::getAddressableNumSGPRs(AMDGPU::GK_GFX802),
            AMDGPU::FIXED_NUM_SGPRS_FOR_INIT_BUG);
  EXPECT_EQ(AMDGPU::getAddressableNumSGPRs(AMDGPU::GK_GFX805),
            AMDGPU::FIXED_NUM_SGPRS_FOR_INIT_BUG);
}

TEST(TargetParserTest, testAMDGPUgetSGPRAllocGranule) {
  // 8 for pre-GFX8, 16 for GFX8/GFX9, addressable count for GFX10+.
  EXPECT_EQ(AMDGPU::getSGPRAllocGranule(Triple::AMDGPUSubArch600), 8u);
  EXPECT_EQ(AMDGPU::getSGPRAllocGranule(Triple::AMDGPUSubArch802), 16u);
  EXPECT_EQ(AMDGPU::getSGPRAllocGranule(Triple::AMDGPUSubArch900), 16u);
  EXPECT_EQ(AMDGPU::getSGPRAllocGranule(Triple::AMDGPUSubArch1030), 106u);

  // The GPUKind overload resolves to the same values.
  EXPECT_EQ(AMDGPU::getSGPRAllocGranule(AMDGPU::GK_GFX600), 8u);
  EXPECT_EQ(AMDGPU::getSGPRAllocGranule(AMDGPU::GK_GFX802), 16u);
  EXPECT_EQ(AMDGPU::getSGPRAllocGranule(AMDGPU::GK_GFX1030), 106u);
}

TEST(TargetParserTest, testAMDGPUParseTargetIDString) {
  using AMDGPU::TargetID;
  using AMDGPU::TargetIDSetting;

  // A well-formed target id parses, canonicalizing the processor and
  // features.
  {
    std::optional<TargetID> TID =
        TargetID::parseTargetIDString("amdgcn-amd-amdhsa-unknown-gfx90a");
    ASSERT_TRUE(TID.has_value());
    EXPECT_EQ(TID->getGPUKind(), AMDGPU::GK_GFX90A);
    EXPECT_EQ(TID->getXnackSetting(), TargetIDSetting::Any);
    EXPECT_EQ(TID->getSramEccSetting(), TargetIDSetting::Any);
  }

  // Explicit feature modifiers are applied.
  {
    std::optional<TargetID> TID = TargetID::parseTargetIDString(
        "amdgcn-amd-amdhsa-unknown-gfx90a:xnack+:sramecc-");
    ASSERT_TRUE(TID.has_value());
    EXPECT_EQ(TID->getXnackSetting(), TargetIDSetting::On);
    EXPECT_EQ(TID->getSramEccSetting(), TargetIDSetting::Off);
  }

  // The processor+features field may be empty; the ISA is taken from the
  // triple subarch.
  EXPECT_TRUE(TargetID::parseTargetIDString("amdgpu9.0a-amd-amdhsa-unknown-")
                  .has_value());

  // Structurally malformed strings (missing the processor+features field or a
  // non-AMDGCN triple) are rejected.
  EXPECT_FALSE(TargetID::parseTargetIDString("not-a-valid-target-id"));
  EXPECT_FALSE(
      TargetID::parseTargetIDString("x86_64-unknown-linux-gnu-gfx90a"));

  // An unrecognized processor is rejected.
  EXPECT_FALSE(
      TargetID::parseTargetIDString("amdgcn-amd-amdhsa-unknown-gfxbogus"));

  // A feature the processor does not support is rejected: gfx600 has neither
  // xnack nor sramecc.
  EXPECT_FALSE(
      TargetID::parseTargetIDString("amdgcn-amd-amdhsa-unknown-gfx600:xnack+"));
  EXPECT_FALSE(TargetID::parseTargetIDString(
      "amdgcn-amd-amdhsa-unknown-gfx900:sramecc+"));

  // xnack is only a valid modifier when the processor supports on/off modes.
  // gfx1250 has xnack permanently enabled (FEATURE_XNACK without
  // FEATURE_XNACK_ON_OFF_MODES), so an xnack modifier is rejected.
  EXPECT_FALSE(TargetID::parseTargetIDString(
      "amdgcn-amd-amdhsa-unknown-gfx1250:xnack+"));
  EXPECT_FALSE(TargetID::parseTargetIDString(
      "amdgcn-amd-amdhsa-unknown-gfx1250:xnack-"));

  // A feature modifier with no "+"/"-" sign is rejected.
  EXPECT_FALSE(
      TargetID::parseTargetIDString("amdgcn-amd-amdhsa-unknown-gfx908:xnack"));

  // An unknown feature name is rejected, even alongside a valid one.
  EXPECT_FALSE(TargetID::parseTargetIDString(
      "amdgcn-amd-amdhsa-unknown-gfx908:unknown+"));
  EXPECT_FALSE(TargetID::parseTargetIDString(
      "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:unknown+"));

  // A repeated feature is rejected.
  EXPECT_FALSE(TargetID::parseTargetIDString(
      "amdgcn-amd-amdhsa-unknown-gfx908:xnack+:xnack+"));

  // Empty processor and/or feature components must be handled without
  // crashing.
  EXPECT_FALSE(TargetID::parseTargetIDString("amdgcn-amd-amdhsa-unknown-:"));
  EXPECT_FALSE(TargetID::parseTargetIDString("amdgcn-amd-amdhsa-unknown-::"));
  EXPECT_FALSE(
      TargetID::parseTargetIDString("amdgcn-amd-amdhsa-unknown-gfx900:"));
  EXPECT_FALSE(
      TargetID::parseTargetIDString("amdgcn-amd-amdhsa-unknown-gfx900::"));
  EXPECT_FALSE(TargetID::parseTargetIDString(
      "amdgcn-amd-amdhsa-unknown-gfx900:xnack+:"));

  // Constructing directly from a triple and processor+features string must
  // also be crash-safe on empty components.
  Triple AMDHSA("amdgcn-amd-amdhsa");
  (void)TargetID(AMDHSA, ":");
  (void)TargetID(AMDHSA, "::");
  (void)TargetID(AMDHSA, "gfx900:");

  // TargetID::parse validates a separate triple + processor/features string.
  {
    std::optional<TargetID> TID = TargetID::parse(AMDHSA, "gfx90a:xnack+");
    ASSERT_TRUE(TID.has_value());
    EXPECT_EQ(TID->getGPUKind(), AMDGPU::GK_GFX90A);
    EXPECT_EQ(TID->getXnackSetting(), AMDGPU::TargetIDSetting::On);
  }

  EXPECT_EQ(TargetID::parse(AMDHSA, "gfx908:xnack+:sramecc-")
                ->getCanonicalFeatureString(),
            "gfx908:sramecc-:xnack+");
  EXPECT_EQ(TargetID::parse(AMDHSA, "gfx908")->getCanonicalFeatureString(),
            "gfx908");
  EXPECT_EQ(TargetID::parse(Triple("amdgcn-amd-amdpal"), "gfx908:xnack-")
                ->getCanonicalFeatureString(),
            "gfx908:xnack-");
  EXPECT_TRUE(TargetID::parse(AMDHSA, "").has_value());
  EXPECT_FALSE(TargetID::parse(AMDHSA, "gfxbogus").has_value());
  EXPECT_FALSE(TargetID::parse(AMDHSA, "gfx600:xnack+").has_value());
  EXPECT_FALSE(TargetID::parse(AMDHSA, "gfx900:").has_value());
  // A non-AMDGCN triple has no target-id features.
  EXPECT_FALSE(
      TargetID::parse(Triple("r600-unknown-unknown"), "cypress").has_value());
}

TEST(TargetParserTest, testAMDGPUTargetIDProvidesFor) {
  using AMDGPU::TargetID;
  Triple AMDHSA("amdgcn-amd-amdhsa");
  Triple AMDHSACanon("amdgpu-amd-amdhsa");

  // TargetID::providesFor is directional: it asks whether an image built
  // for *this target can provide the device code for a request for the other
  // target. An unspecified feature acts as a wildcard that can provide for a
  // specific request, but not the other way around.

  // Exact match provides for itself.
  EXPECT_TRUE(
      TargetID(AMDHSA, "gfx90a").providesFor(TargetID(AMDHSA, "gfx90a")));

  // Different processors are never compatible.
  EXPECT_FALSE(
      TargetID(AMDHSA, "gfx90a").providesFor(TargetID(AMDHSA, "gfx908")));

  // The pure "generic"/empty wildcard processor provides for any specific
  // processor, in both directions (unknown arch acts as a wildcard).
  EXPECT_TRUE(
      TargetID(AMDHSA, "generic").providesFor(TargetID(AMDHSA, "gfx90a")));
  EXPECT_TRUE(
      TargetID(AMDHSA, "gfx90a").providesFor(TargetID(AMDHSA, "generic")));
  EXPECT_TRUE(TargetID(AMDHSA, "").providesFor(TargetID(AMDHSA, "gfx908")));

  // A major-family/generic processor provides for a specific member of its
  // family (gfx900), but a specific processor does not provide for the
  // generic family.
  Triple AMDHSAMajor9("amdgpu9-amd-amdhsa");
  EXPECT_TRUE(
      TargetID(AMDHSAMajor9, "").providesFor(TargetID(AMDHSA, "gfx900")));
  EXPECT_TRUE(
      TargetID(AMDHSAMajor9, "").providesFor(TargetID(AMDHSA, "gfx902")));

  EXPECT_TRUE(TargetID(AMDHSAMajor9, "").providesFor(TargetID(AMDHSA, "")));
  EXPECT_TRUE(
      TargetID(AMDHSAMajor9, "").providesFor(TargetID(AMDHSA, "gfx9-generic")));
  EXPECT_TRUE(
      TargetID(AMDHSAMajor9, "").providesFor(TargetID(AMDHSA, "generic")));
  EXPECT_TRUE(TargetID(AMDHSAMajor9, "gfx9-generic")
                  .providesFor(TargetID(AMDHSA, "gfx9-generic")));

  EXPECT_FALSE(
      TargetID(AMDHSA, "gfx900").providesFor(TargetID(AMDHSAMajor9, "")));
  // A specific processor and a member of a different major family do not
  // provide for each other.
  EXPECT_FALSE(
      TargetID(AMDHSAMajor9, "").providesFor(TargetID(AMDHSA, "gfx1030")));

  // xnack: an unspecified (wildcard) image provides for an explicit request,
  // but an explicit image does not provide for a request that lacks it, and
  // explicit On vs Off never match.
  EXPECT_TRUE(TargetID(AMDHSA, "gfx90a")
                  .providesFor(TargetID(AMDHSA, "gfx90a:xnack+")));
  EXPECT_TRUE(TargetID(AMDHSA, "gfx90a")
                  .providesFor(TargetID(AMDHSA, "gfx90a:xnack-")));
  EXPECT_FALSE(TargetID(AMDHSA, "gfx90a:xnack+")
                   .providesFor(TargetID(AMDHSA, "gfx90a")));
  EXPECT_FALSE(TargetID(AMDHSA, "gfx90a:xnack-")
                   .providesFor(TargetID(AMDHSA, "gfx90a")));
  EXPECT_TRUE(TargetID(AMDHSA, "gfx90a:xnack+")
                  .providesFor(TargetID(AMDHSA, "gfx90a:xnack+")));
  EXPECT_FALSE(TargetID(AMDHSA, "gfx90a:xnack+")
                   .providesFor(TargetID(AMDHSA, "gfx90a:xnack-")));
  EXPECT_FALSE(TargetID(AMDHSA, "gfx90a:xnack-")
                   .providesFor(TargetID(AMDHSA, "gfx90a:xnack+")));

  // sramecc behaves the same way as xnack.
  EXPECT_TRUE(TargetID(AMDHSA, "gfx90a")
                  .providesFor(TargetID(AMDHSA, "gfx90a:sramecc+")));
  EXPECT_TRUE(TargetID(AMDHSA, "gfx90a")
                  .providesFor(TargetID(AMDHSA, "gfx90a:sramecc-")));
  EXPECT_FALSE(TargetID(AMDHSA, "gfx90a:sramecc+")
                   .providesFor(TargetID(AMDHSA, "gfx90a")));
  EXPECT_FALSE(TargetID(AMDHSA, "gfx90a:sramecc-")
                   .providesFor(TargetID(AMDHSA, "gfx90a")));
  EXPECT_FALSE(TargetID(AMDHSA, "gfx90a:sramecc+")
                   .providesFor(TargetID(AMDHSA, "gfx90a:sramecc-")));

  // Both feature modifiers together: a wildcard provides for a
  // fully-specified request, but a partially-specified image only provides
  // when its explicit features match.
  EXPECT_TRUE(TargetID(AMDHSA, "gfx90a")
                  .providesFor(TargetID(AMDHSA, "gfx90a:sramecc+:xnack+")));
  EXPECT_TRUE(TargetID(AMDHSA, "gfx90a:sramecc+")
                  .providesFor(TargetID(AMDHSA, "gfx90a:sramecc+:xnack+")));
  EXPECT_FALSE(TargetID(AMDHSA, "gfx90a:sramecc-")
                   .providesFor(TargetID(AMDHSA, "gfx90a:sramecc+:xnack+")));

  // The legacy "amdgcn" and canonical "amdgpu" triple spellings are unified.
  EXPECT_TRUE(
      TargetID(AMDHSA, "gfx90a").providesFor(TargetID(AMDHSACanon, "gfx90a")));
  EXPECT_TRUE(
      TargetID(AMDHSACanon, "gfx90a").providesFor(TargetID(AMDHSA, "gfx90a")));
}

TEST(TargetParserTest, testAMDGPUTargetIDEquivalent) {
  using AMDGPU::TargetID;
  Triple AMDHSA("amdgcn-amd-amdhsa");
  Triple AMDHSACanon("amdgpu-amd-amdhsa");

  // TargetID::isEquivalent is a symmetric semantic equality: same processor
  // and features, looking through triple spelling differences.
  auto Equivalent = [](const TargetID &A, const TargetID &B) {
    EXPECT_EQ(A.isEquivalent(B), B.isEquivalent(A));
    return A.isEquivalent(B);
  };

  // Same target is equivalent to itself.
  EXPECT_TRUE(
      Equivalent(TargetID(AMDHSA, "gfx90a"), TargetID(AMDHSA, "gfx90a")));

  // Different processors are not equivalent.
  EXPECT_FALSE(
      Equivalent(TargetID(AMDHSA, "gfx90a"), TargetID(AMDHSA, "gfx908")));

  // A specific processor and a member of its generic family are distinct and
  // not equivalent (this is the over-merge that produced duplicate symbols).
  EXPECT_FALSE(
      Equivalent(TargetID(AMDHSA, "gfx900"), TargetID(AMDHSA, "gfx9-generic")));

  // Legacy "amdgcn" and canonical "amdgpu" spellings are equivalent.
  EXPECT_TRUE(
      Equivalent(TargetID(AMDHSA, "gfx90a"), TargetID(AMDHSACanon, "gfx90a")));

  // The same processor spelled via the triple subarch (with no processor
  // name) and via the processor name (with a major-family subarch triple) are
  // equivalent, e.g. amdgpu9-amd-amdhsa + "gfx900" and amdgpu9.00-amd-amdhsa.
  Triple AMDHSAMajor9("amdgpu9-amd-amdhsa");
  Triple AMDHSASub900("amdgpu9.00-amd-amdhsa");
  EXPECT_TRUE(
      Equivalent(TargetID(AMDHSAMajor9, "gfx900"), TargetID(AMDHSASub900, "")));
  EXPECT_TRUE(
      Equivalent(TargetID(AMDHSASub900, ""), TargetID(AMDHSA, "gfx900")));
  EXPECT_TRUE(Equivalent(TargetID(AMDHSAMajor9, "gfx9-generic"),
                         TargetID(AMDHSAMajor9, "")));

  // The same processor with different feature settings is not equivalent.
  EXPECT_FALSE(Equivalent(TargetID(AMDHSA, "gfx90a"),
                          TargetID(AMDHSA, "gfx90a:xnack+")));
  EXPECT_FALSE(Equivalent(TargetID(AMDHSA, "gfx90a:xnack+"),
                          TargetID(AMDHSA, "gfx90a:xnack-")));
  EXPECT_FALSE(Equivalent(TargetID(AMDHSA, "gfx90a:sramecc+"),
                          TargetID(AMDHSA, "gfx90a:sramecc-")));
  EXPECT_TRUE(Equivalent(TargetID(AMDHSA, "gfx90a:xnack+"),
                         TargetID(AMDHSA, "gfx90a:xnack+")));

  // Target IDs that set both xnack and sramecc are equivalent only when both
  // settings match; a difference in either feature makes them distinct.
  EXPECT_TRUE(Equivalent(TargetID(AMDHSA, "gfx90a:sramecc+:xnack+"),
                         TargetID(AMDHSA, "gfx90a:sramecc+:xnack+")));
  EXPECT_FALSE(Equivalent(TargetID(AMDHSA, "gfx90a:sramecc+:xnack+"),
                          TargetID(AMDHSA, "gfx90a:sramecc+:xnack-")));
  EXPECT_FALSE(Equivalent(TargetID(AMDHSA, "gfx90a:sramecc+:xnack+"),
                          TargetID(AMDHSA, "gfx90a:sramecc-:xnack+")));
  EXPECT_FALSE(Equivalent(TargetID(AMDHSA, "gfx90a:sramecc+:xnack+"),
                          TargetID(AMDHSA, "gfx90a:sramecc-:xnack-")));
  // A fully-specified target ID is not equivalent to one that leaves a
  // feature unspecified, even if the specified features agree.
  EXPECT_FALSE(Equivalent(TargetID(AMDHSA, "gfx90a:sramecc+:xnack+"),
                          TargetID(AMDHSA, "gfx90a:sramecc+")));
  EXPECT_FALSE(Equivalent(TargetID(AMDHSA, "gfx90a:sramecc+:xnack+"),
                          TargetID(AMDHSA, "gfx90a")));
}

} // namespace
