//===-------- CSKYTargetParser.cpp - CSKY Target Parser -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CSKYTargetParser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CSKYAttributes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetParser.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

namespace {

std::string FormatExtensionFlags(uint64_t Flags) {
  std::vector<StringRef> Features;

  if (Flags & CSKY::AEK_NONE)
    Features.push_back("none");
  CSKY::getExtensionFeatures(Flags, Features);

  Features.erase(std::remove_if(Features.begin(), Features.end(),
                                [](StringRef extension) {
                                  return extension.startswith("-");
                                }),
                 Features.end());

  return llvm::join(Features, ", ");
}

testing::AssertionResult AssertSameExtensionFlags(const char *m_expr,
                                                  const char *n_expr,
                                                  uint64_t ExpectedFlags,
                                                  uint64_t GotFlags) {
  if (ExpectedFlags == GotFlags)
    return testing::AssertionSuccess();

  return testing::AssertionFailure()
         << llvm::formatv("Expected extension flags: {0} ({1:x})\n"
                          "     Got extension flags: {2} ({3:x})\n",
                          FormatExtensionFlags(ExpectedFlags), ExpectedFlags,
                          FormatExtensionFlags(GotFlags), GotFlags);
}

struct CSKYCPUTestParams {
  CSKYCPUTestParams(StringRef CPUName, StringRef ExpectedArch,
                    uint64_t ExpectedFlags)
      : CPUName(CPUName), ExpectedArch(ExpectedArch),
        ExpectedFlags(ExpectedFlags) {}

  friend std::ostream &operator<<(std::ostream &os,
                                  const CSKYCPUTestParams &params) {
    return os << "\"" << params.CPUName.str() << "\", \""
              << params.ExpectedArch.str() << "\", \"" << params.ExpectedFlags
              << "\"";
  }

  StringRef CPUName;
  StringRef ExpectedArch;
  uint64_t ExpectedFlags;
};

class CSKYCPUTestFixture : public ::testing::TestWithParam<CSKYCPUTestParams> {
};

TEST_P(CSKYCPUTestFixture, CSKYCPUTests) {
  auto params = GetParam();

  CSKY::ArchKind AK = CSKY::parseCPUArch(params.CPUName);
  EXPECT_EQ(params.ExpectedArch, CSKY::getArchName(AK));

  uint64_t default_extensions = CSKY::getDefaultExtensions(params.CPUName);
  EXPECT_PRED_FORMAT2(AssertSameExtensionFlags, params.ExpectedFlags,
                      default_extensions);
}

// Note that we include CSKY::AEK_NONE even when there are other extensions
// we expect. This is because the default extensions for a CPU are the sum
// of the default extensions for its architecture and for the CPU.
// So if a CPU has no extra extensions, it adds AEK_NONE.
INSTANTIATE_TEST_SUITE_P(
    CSKYCPUTests, CSKYCPUTestFixture,
    ::testing::Values(

        CSKYCPUTestParams("ck801", "ck801",
                          CSKY::AEK_NONE | CSKY::MAEK_E1 | CSKY::AEK_TRUST),
        CSKYCPUTestParams("ck801t", "ck801",
                          CSKY::AEK_NONE | CSKY::MAEK_E1 | CSKY::AEK_TRUST),
        CSKYCPUTestParams("e801", "ck801",
                          CSKY::AEK_NONE | CSKY::MAEK_E1 | CSKY::AEK_TRUST),

        CSKYCPUTestParams("ck802", "ck802",
                          CSKY::AEK_NONE | CSKY::MAEK_E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC),
        CSKYCPUTestParams("ck802t", "ck802",
                          CSKY::AEK_NONE | CSKY::MAEK_E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC),
        CSKYCPUTestParams("ck802j", "ck802",
                          CSKY::AEK_JAVA | CSKY::MAEK_E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC),
        CSKYCPUTestParams("e802", "ck802",
                          CSKY::AEK_NONE | CSKY::MAEK_E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC),
        CSKYCPUTestParams("e802t", "ck802",
                          CSKY::AEK_NONE | CSKY::MAEK_E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC),
        CSKYCPUTestParams("s802", "ck802",
                          CSKY::AEK_NONE | CSKY::MAEK_E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC),
        CSKYCPUTestParams("s802t", "ck802",
                          CSKY::AEK_NONE | CSKY::MAEK_E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC),

        CSKYCPUTestParams("ck803", "ck803",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803h", "ck803",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803t", "ck803",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ht", "ck803",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803f", "ck803",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803fh", "ck803",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803e", "ck803",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803eh", "ck803",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803et", "ck803",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803eht", "ck803",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ef", "ck803",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803efh", "ck803",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ft", "ck803",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803eft", "ck803",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803efht", "ck803",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803r1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803r2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803r3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803hr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803hr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803hr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803tr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803tr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803tr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803htr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803htr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803htr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803fr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803fr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803fr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803fhr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803fhr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803fhr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803er1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803er2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803er3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ehr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ehr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ehr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803etr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803etr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803etr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ehtr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ehtr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ehtr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                              CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803efr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803efr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803efr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803efhr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803efhr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803efhr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ftr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ftr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803ftr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803eftr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803eftr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803eftr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803efhtr1", "ck803",
                          CSKY::MAEK_3E3R1 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803efhtr2", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803efhtr3", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("s803", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("s803t", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("e803", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("e803t", "ck803",
                          CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),

        CSKYCPUTestParams("ck803s", "ck803s",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803st", "ck803s",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803se", "ck803s",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803sf", "ck803s",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803sef", "ck803s",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),
        CSKYCPUTestParams("ck803seft", "ck803s",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV),

        CSKYCPUTestParams("ck804", "ck804",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804h", "ck804",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804t", "ck804",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804ht", "ck804",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804f", "ck804",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804fh", "ck804",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804e", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804eh", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804et", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804eht", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804ef", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804efh", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804ft", "ck804",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804eft", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3),
        CSKYCPUTestParams("ck804efht", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3),
        CSKYCPUTestParams("e804d", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("e804dt", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("e804f", "ck804",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("e804ft", "ck804",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3),
        CSKYCPUTestParams("e804df", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3),
        CSKYCPUTestParams("e804dft", "ck804",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3),

        CSKYCPUTestParams("ck805", "ck805",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_VDSPV2 | CSKY::AEK_VDSP2E3),
        CSKYCPUTestParams("ck805e", "ck805",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3 | CSKY::AEK_VDSPV2 |
                              CSKY::AEK_VDSP2E3),
        CSKYCPUTestParams("ck805f", "ck805",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_VDSPV2 | CSKY::AEK_VDSP2E3),
        CSKYCPUTestParams("ck805t", "ck805",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_VDSPV2 | CSKY::AEK_VDSP2E3),
        CSKYCPUTestParams("ck805ef", "ck805",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_VDSPV2 | CSKY::AEK_VDSP2E3),
        CSKYCPUTestParams("ck805et", "ck805",
                          CSKY::AEK_DSPV2 | CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3 | CSKY::AEK_VDSPV2 |
                              CSKY::AEK_VDSP2E3),
        CSKYCPUTestParams("ck805ft", "ck805",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_VDSPV2 | CSKY::AEK_VDSP2E3),
        CSKYCPUTestParams("ck805eft", "ck805",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::AEK_DSPV2 |
                              CSKY::AEK_3E3R1 | CSKY::AEK_3E3R3 |
                              CSKY::MAEK_2E3 | CSKY::AEK_MP | CSKY::AEK_TRUST |
                              CSKY::AEK_NVIC | CSKY::AEK_HWDIV |
                              CSKY::AEK_HIGHREG | CSKY::MAEK_3E3R2 |
                              CSKY::AEK_3E3R3 | CSKY::AEK_VDSPV2 |
                              CSKY::AEK_VDSP2E3),
        CSKYCPUTestParams("i805", "ck805",
                          CSKY::AEK_NONE | CSKY::MAEK_2E3 | CSKY::AEK_MP |
                              CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_VDSPV2 | CSKY::AEK_VDSP2E3),
        CSKYCPUTestParams("i805f", "ck805",
                          CSKY::AEK_FPUV2SF | CSKY::AEK_FLOATE1 |
                              CSKY::AEK_FLOAT1E3 | CSKY::MAEK_2E3 |
                              CSKY::AEK_MP | CSKY::AEK_TRUST | CSKY::AEK_NVIC |
                              CSKY::AEK_HWDIV | CSKY::AEK_HIGHREG |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_VDSPV2 | CSKY::AEK_VDSP2E3),

        CSKYCPUTestParams("ck807", "ck807",
                          CSKY::AEK_NONE | CSKY::MAEK_3E7 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams("ck807e", "ck807",
                          CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::MAEK_3E7 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams(
            "ck807f", "ck807",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::AEK_FLOAT1E3 |
                CSKY::AEK_FLOAT3E4 | CSKY::MAEK_3E7 | CSKY::MAEK_MP |
                CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST | CSKY::AEK_HWDIV |
                CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP | CSKY::AEK_NVIC |
                CSKY::AEK_CACHE),
        CSKYCPUTestParams(
            "ck807ef", "ck807",
            CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::AEK_FLOAT1E3 |
                CSKY::AEK_FLOAT3E4 | CSKY::MAEK_3E7 | CSKY::MAEK_MP |
                CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST | CSKY::AEK_HWDIV |
                CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP | CSKY::AEK_NVIC |
                CSKY::AEK_CACHE),
        CSKYCPUTestParams("c807", "ck807",
                          CSKY::AEK_NONE | CSKY::MAEK_3E7 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams(
            "c807f", "ck807",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::AEK_FLOAT1E3 |
                CSKY::AEK_FLOAT3E4 | CSKY::MAEK_3E7 | CSKY::MAEK_MP |
                CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST | CSKY::AEK_HWDIV |
                CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP | CSKY::AEK_NVIC |
                CSKY::AEK_CACHE),

        CSKYCPUTestParams("r807", "ck807",
                          CSKY::AEK_NONE | CSKY::MAEK_3E7 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams(
            "r807f", "ck807",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::AEK_FLOAT1E3 |
                CSKY::AEK_FLOAT3E4 | CSKY::MAEK_3E7 | CSKY::MAEK_MP |
                CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST | CSKY::AEK_HWDIV |
                CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP | CSKY::AEK_NVIC |
                CSKY::AEK_CACHE),

        CSKYCPUTestParams("ck810e", "ck810",
                          CSKY::AEK_NONE | CSKY::MAEK_7E10 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE),

        CSKYCPUTestParams("ck810et", "ck810",
                          CSKY::AEK_NONE | CSKY::MAEK_7E10 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams(
            "ck810ef", "ck810",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams(
            "ck810eft", "ck810",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams("ck810", "ck810",
                          CSKY::AEK_NONE | CSKY::MAEK_7E10 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams(
            "ck810f", "ck810",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams("ck810t", "ck810",
                          CSKY::AEK_NONE | CSKY::MAEK_7E10 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams(
            "ck810ft", "ck810",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams(
            "c810", "ck810",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE),
        CSKYCPUTestParams(
            "c810t", "ck810",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE),

        CSKYCPUTestParams("ck810v", "ck810v",
                          CSKY::AEK_NONE | CSKY::MAEK_7E10 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE |
                              CSKY::AEK_VDSPV1),

        CSKYCPUTestParams("ck810ev", "ck810v",
                          CSKY::AEK_NONE | CSKY::MAEK_7E10 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE |
                              CSKY::AEK_VDSPV1),

        CSKYCPUTestParams("ck810tv", "ck810v",
                          CSKY::AEK_NONE | CSKY::MAEK_7E10 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE |
                              CSKY::AEK_VDSPV1),

        CSKYCPUTestParams("ck810etv", "ck810v",
                          CSKY::AEK_NONE | CSKY::MAEK_7E10 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_EDSP |
                              CSKY::AEK_DSP1E2 | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE |
                              CSKY::AEK_VDSPV1),

        CSKYCPUTestParams(
            "c810v", "ck810v",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE | CSKY::AEK_VDSPV1),

        CSKYCPUTestParams(
            "ck810fv", "ck810v",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE | CSKY::AEK_VDSPV1),

        CSKYCPUTestParams(
            "ck810efv", "ck810v",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE | CSKY::AEK_VDSPV1),

        CSKYCPUTestParams(
            "ck810ftv", "ck810v",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE | CSKY::AEK_VDSPV1),

        CSKYCPUTestParams(
            "c810tv", "ck810v",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE | CSKY::AEK_VDSPV1),

        CSKYCPUTestParams(
            "c810eftv", "ck810v",
            CSKY::AEK_FPUV2SF | CSKY::AEK_FPUV2DF | CSKY::AEK_FDIVDU |
                CSKY::AEK_FLOATE1 | CSKY::AEK_FLOAT1E2 | CSKY::MAEK_7E10 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_EDSP | CSKY::AEK_DSP1E2 |
                CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                CSKY::AEK_NVIC | CSKY::AEK_CACHE | CSKY::AEK_VDSPV1),

        CSKYCPUTestParams("ck860", "ck860",
                          CSKY::AEK_NONE | CSKY::MAEK_10E60 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3),

        CSKYCPUTestParams(
            "ck860f", "ck860",
            CSKY::AEK_FPUV3HI | CSKY::AEK_FPUV3HF | CSKY::AEK_FPUV3SF |
                CSKY::AEK_FPUV3DF | CSKY::AEK_FLOAT7E60 | CSKY::MAEK_10E60 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                CSKY::AEK_HARDTP | CSKY::AEK_NVIC | CSKY::AEK_CACHE |
                CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3),

        CSKYCPUTestParams(
            "c860", "ck860",
            CSKY::AEK_FPUV3HI | CSKY::AEK_FPUV3HF | CSKY::AEK_FPUV3SF |
                CSKY::AEK_FPUV3DF | CSKY::AEK_FLOAT7E60 | CSKY::MAEK_10E60 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                CSKY::AEK_HARDTP | CSKY::AEK_NVIC | CSKY::AEK_CACHE |
                CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3),

        CSKYCPUTestParams("ck860v", "ck860v",
                          CSKY::AEK_NONE | CSKY::MAEK_10E60 | CSKY::MAEK_MP |
                              CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                              CSKY::AEK_HWDIV | CSKY::AEK_DSPE60 |
                              CSKY::AEK_HIGHREG | CSKY::AEK_HARDTP |
                              CSKY::AEK_NVIC | CSKY::AEK_CACHE |
                              CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 |
                              CSKY::AEK_VDSPV2 | CSKY::AEK_VDSP2E60F),

        CSKYCPUTestParams(
            "ck860fv", "ck860v",
            CSKY::AEK_FPUV3HI | CSKY::AEK_FPUV3HF | CSKY::AEK_FPUV3SF |
                CSKY::AEK_FPUV3DF | CSKY::AEK_FLOAT7E60 | CSKY::MAEK_10E60 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                CSKY::AEK_HARDTP | CSKY::AEK_NVIC | CSKY::AEK_CACHE |
                CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_VDSPV2 |
                CSKY::AEK_VDSP2E60F),

        CSKYCPUTestParams(
            "c860v", "ck860v",
            CSKY::AEK_FPUV3HI | CSKY::AEK_FPUV3HF | CSKY::AEK_FPUV3SF |
                CSKY::AEK_FPUV3DF | CSKY::AEK_FLOAT7E60 | CSKY::MAEK_10E60 |
                CSKY::MAEK_MP | CSKY::MAEK_MP1E2 | CSKY::AEK_TRUST |
                CSKY::AEK_HWDIV | CSKY::AEK_DSPE60 | CSKY::AEK_HIGHREG |
                CSKY::AEK_HARDTP | CSKY::AEK_NVIC | CSKY::AEK_CACHE |
                CSKY::MAEK_3E3R2 | CSKY::AEK_3E3R3 | CSKY::AEK_VDSPV2 |
                CSKY::AEK_VDSP2E60F)));

static constexpr unsigned NumCSKYCPUArchs = 145;

TEST(TargetParserTest, testCSKYCPUArchList) {
  SmallVector<StringRef, NumCSKYCPUArchs> List;
  CSKY::fillValidCPUArchList(List);

  // No list exists for these in this test suite, so ensure all are
  // valid, and match the expected 'magic' count.
  EXPECT_EQ(List.size(), NumCSKYCPUArchs);
  for (StringRef CPU : List) {
    EXPECT_NE(CSKY::parseCPUArch(CPU), CSKY::ArchKind::INVALID);
  }
}

TEST(TargetParserTest, testInvalidCSKYArch) {
  auto InvalidArchStrings = {"ckv", "ckv99", "nock"};
  for (const char *InvalidArch : InvalidArchStrings)
    EXPECT_EQ(CSKY::parseArch(InvalidArch), CSKY::ArchKind::INVALID);
}

bool testCSKYArch(StringRef Arch, StringRef DefaultCPU) {
  CSKY::ArchKind AK = CSKY::parseArch(Arch);
  bool Result = (AK != CSKY::ArchKind::INVALID);
  Result &= CSKY::getDefaultCPU(Arch).equals(DefaultCPU);
  return Result;
}

TEST(TargetParserTest, testCSKYArch) {
  EXPECT_TRUE(testCSKYArch("ck801", "ck801"));
  EXPECT_TRUE(testCSKYArch("ck802", "ck802"));
  EXPECT_TRUE(testCSKYArch("ck803", "ck803"));
  EXPECT_TRUE(testCSKYArch("ck803s", "ck803s"));
  EXPECT_TRUE(testCSKYArch("ck804", "ck804"));
  EXPECT_TRUE(testCSKYArch("ck805", "ck805"));
  EXPECT_TRUE(testCSKYArch("ck807", "ck807"));
  EXPECT_TRUE(testCSKYArch("ck810", "ck810"));
  EXPECT_TRUE(testCSKYArch("ck810v", "ck810v"));
  EXPECT_TRUE(testCSKYArch("ck860", "ck860"));
  EXPECT_TRUE(testCSKYArch("ck860v", "ck860v"));
}

TEST(TargetParserTest, CSKYFPUVersion) {
  for (CSKY::CSKYFPUKind FK = static_cast<CSKY::CSKYFPUKind>(0);
       FK <= CSKY::CSKYFPUKind::FK_LAST;
       FK = static_cast<CSKY::CSKYFPUKind>(static_cast<unsigned>(FK) + 1))
    if (FK == CSKY::FK_LAST || CSKY::getFPUName(FK) == "invalid")
      EXPECT_EQ(CSKY::FPUVersion::NONE, CSKY::getFPUVersion(FK));
    else
      EXPECT_NE(CSKY::FPUVersion::NONE, CSKY::getFPUVersion(FK));
}

TEST(TargetParserTest, CSKYExtensionFeatures) {
  std::map<uint64_t, std::vector<StringRef>> Extensions;

  for (auto &Ext : CSKY::CSKYARCHExtNames) {
    if (Ext.Feature && Ext.NegFeature)
      Extensions[Ext.ID] = {StringRef(Ext.Feature), StringRef(Ext.NegFeature)};
  }

  std::vector<StringRef> Features;

  EXPECT_FALSE(CSKY::getExtensionFeatures(CSKY::AEK_INVALID, Features));

  for (auto &E : Extensions) {
    CSKY::getExtensionFeatures(E.first, Features);
    EXPECT_TRUE(llvm::is_contained(Features, E.second.at(0)));
  }

  EXPECT_EQ(Extensions.size(), Features.size());
}

TEST(TargetParserTest, CSKYFPUFeatures) {
  std::vector<StringRef> Features;
  for (CSKY::CSKYFPUKind FK = static_cast<CSKY::CSKYFPUKind>(0);
       FK <= CSKY::CSKYFPUKind::FK_LAST;
       FK = static_cast<CSKY::CSKYFPUKind>(static_cast<unsigned>(FK) + 1)) {
    if (FK == CSKY::FK_INVALID || FK >= CSKY::FK_LAST)
      EXPECT_FALSE(CSKY::getFPUFeatures(FK, Features));
    else
      EXPECT_TRUE(CSKY::getFPUFeatures(FK, Features));
  }
}

TEST(TargetParserTest, CSKYArchExtFeature) {
  const char *ArchExt[][4] = {
      {"fpuv2_sf", "nofpuv2_sf", "+fpuv2_sf", "-fpuv2_sf"},
      {"fpuv2_df", "nofpuv2_df", "+fpuv2_df", "-fpuv2_df"},
      {"fdivdu", "nofdivdu", "+fdivdu", "-fdivdu"},
      {"fpuv3_hi", "nofpuv3_hi", "+fpuv3_hi", "-fpuv3_hi"},
      {"fpuv3_hf", "nofpuv3_hf", "+fpuv3_hf", "-fpuv3_hf"},
      {"fpuv2_df", "nofpuv2_df", "+fpuv2_df", "-fpuv2_df"},
      {"fpuv3_sf", "nofpuv3_sf", "+fpuv3_sf", "-fpuv3_sf"},
      {"fpuv3_df", "nofpuv3_df", "+fpuv3_df", "-fpuv3_df"},
      {"floate1", "nofloate1", "+floate1", "-floate1"},
      {"float1e2", "nofloat1e2", "+float1e2", "-float1e2"},
      {"float1e3", "nofloat1e3", "+float1e3", "-float1e3"},
      {"float3e4", "nofloat3e4", "+float3e4", "-float3e4"},
      {"float7e60", "nofloat7e60", "+float7e60", "-float7e60"},
      {"hwdiv", "nohwdiv", "+hwdiv", "-hwdiv"},
      {"multiple_stld", "nomultiple_stld", "+multiple_stld", "-multiple_stld"},
      {"pushpop", "nopushpop", "+pushpop", "-pushpop"},
      {"edsp", "noedsp", "+edsp", "-edsp"},
      {"dsp1e2", "nodsp1e2", "+dsp1e2", "-dsp1e2"},
      {"dspe60", "nodspe60", "+dspe60", "-dspe60"},
      {"dspv2", "nodspv2", "+dspv2", "-dspv2"},
      {"dsp_silan", "nodsp_silan", "+dsp_silan", "-dsp_silan"},
      {"elrw", "noelrw", "+elrw", "-elrw"},
      {"trust", "notrust", "+trust", "-trust"},
      {"java", "nojava", "+java", "-java"},
      {"cache", "nocache", "+cache", "-cache"},
      {"nvic", "nonvic", "+nvic", "-nvic"},
      {"doloop", "nodoloop", "+doloop", "-doloop"},
      {"high-registers", "nohigh-registers", "+high-registers",
       "-high-registers"},
      {"smart", "nosmart", "+smart", "-smart"},
      {"vdsp2e3", "novdsp2e3", "+vdsp2e3", "-vdsp2e3"},
      {"vdsp2e60f", "novdsp2e60f", "+vdsp2e60f", "-vdsp2e60f"},
      {"vdspv2", "novdspv2", "+vdspv2", "-vdspv2"},
      {"hard-tp", "nohard-tp", "+hard-tp", "-hard-tp"},
      {"soft-tp", "nosoft-tp", "+soft-tp", "-soft-tp"},
      {"istack", "noistack", "+istack", "-istack"},
      {"constpool", "noconstpool", "+constpool", "-constpool"},
      {"stack-size", "nostack-size", "+stack-size", "-stack-size"},
      {"ccrt", "noccrt", "+ccrt", "-ccrt"},
      {"vdspv1", "novdspv1", "+vdspv1", "-vdspv1"},
      {"e1", "noe1", "+e1", "-e1"},
      {"e2", "noe2", "+e2", "-e2"},
      {"2e3", "no2e3", "+2e3", "-2e3"},
      {"mp", "nomp", "+mp", "-mp"},
      {"3e3r1", "no3e3r1", "+3e3r1", "-3e3r1"},
      {"3e3r2", "no3e3r2", "+3e3r2", "-3e3r2"},
      {"3e3r3", "no3e3r3", "+3e3r3", "-3e3r3"},
      {"3e7", "no3e7", "+3e7", "-3e7"},
      {"mp1e2", "nomp1e2", "+mp1e2", "-mp1e2"},
      {"7e10", "no7e10", "+7e10", "-7e10"},
      {"10e60", "no10e60", "+10e60", "-10e60"},
  };

  for (unsigned i = 0; i < array_lengthof(ArchExt); i++) {
    EXPECT_EQ(StringRef(ArchExt[i][2]), CSKY::getArchExtFeature(ArchExt[i][0]));
    EXPECT_EQ(StringRef(ArchExt[i][3]), CSKY::getArchExtFeature(ArchExt[i][1]));
  }
}

} // namespace
