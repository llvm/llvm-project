//===---- RISCVTargetParserTest.cpp - RISCVTargetParser unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
TEST(RISCVVType, CheckSameRatioLMUL) {
  // Smaller LMUL.
  EXPECT_EQ(RISCVVType::LMUL_1,
            RISCVVType::getSameRatioLMUL(16, RISCVVType::LMUL_2, 8));
  EXPECT_EQ(RISCVVType::LMUL_F2,
            RISCVVType::getSameRatioLMUL(16, RISCVVType::LMUL_1, 8));
  // Smaller fractional LMUL.
  EXPECT_EQ(RISCVVType::LMUL_F8,
            RISCVVType::getSameRatioLMUL(16, RISCVVType::LMUL_F4, 8));
  // Bigger LMUL.
  EXPECT_EQ(RISCVVType::LMUL_2,
            RISCVVType::getSameRatioLMUL(8, RISCVVType::LMUL_1, 16));
  EXPECT_EQ(RISCVVType::LMUL_1,
            RISCVVType::getSameRatioLMUL(8, RISCVVType::LMUL_F2, 16));
  // Bigger fractional LMUL.
  EXPECT_EQ(RISCVVType::LMUL_F2,
            RISCVVType::getSameRatioLMUL(8, RISCVVType::LMUL_F4, 16));
}

TEST(RISCVTuneFeature, AllTuneFeatures) {
  SmallVector<StringRef> AllTuneFeatures;
  RISCV::getAllTuneFeatures(AllTuneFeatures);
  // Only allowed subtarget features that are explicitly marked by
  // special TableGen class.
  EXPECT_EQ(AllTuneFeatures.size(), 28U);
  for (auto F : {"conditional-cmv-fusion",
                 "dlen-factor-2",
                 "disable-latency-sched-heuristic",
                 "disable-misched-load-clustering",
                 "disable-misched-store-clustering",
                 "disable-postmisched-load-clustering",
                 "disable-postmisched-store-clustering",
                 "single-element-vec-fp64",
                 "log-vrgather",
                 "no-default-unroll",
                 "no-sink-splat-operands",
                 "optimized-nf2-segment-load-store",
                 "optimized-nf3-segment-load-store",
                 "optimized-nf4-segment-load-store",
                 "optimized-nf5-segment-load-store",
                 "optimized-nf6-segment-load-store",
                 "optimized-nf7-segment-load-store",
                 "optimized-nf8-segment-load-store",
                 "optimized-zero-stride-load",
                 "use-postra-scheduler",
                 "predictable-select-expensive",
                 "prefer-vsetvli-over-read-vlenb",
                 "prefer-w-inst",
                 "short-forward-branch-i-minmax",
                 "short-forward-branch-i-mul",
                 "short-forward-branch-opt",
                 "vl-dependent-latency",
                 "vxrm-pipeline-flush"})
    EXPECT_TRUE(is_contained(AllTuneFeatures, F));
}

TEST(RISCVTuneFeature, LegalTuneFeatureStrings) {
  SmallVector<std::string> Result;
  EXPECT_FALSE(errorToBool(RISCV::parseTuneFeatureString(
      "log-vrgather,no-short-forward-branch-opt,vl-dependent-latency",
      Result)));
  EXPECT_TRUE(is_contained(Result, "+log-vrgather"));
  EXPECT_TRUE(is_contained(Result, "+vl-dependent-latency"));
  EXPECT_TRUE(is_contained(Result, "-short-forward-branch-opt"));
  EXPECT_TRUE(is_contained(Result, "-short-forward-branch-i-minmax"));
  EXPECT_TRUE(is_contained(Result, "-short-forward-branch-i-mul"));

  Result.clear();
  // Test inverse implied features.
  EXPECT_FALSE(errorToBool(RISCV::parseTuneFeatureString(
      "no-short-forward-branch-i-mul,short-forward-branch-i-minmax", Result)));
  EXPECT_TRUE(is_contained(Result, "+short-forward-branch-i-minmax"));
  EXPECT_TRUE(is_contained(Result, "+short-forward-branch-opt"));
  EXPECT_TRUE(is_contained(Result, "-short-forward-branch-i-mul"));

  Result.clear();
  // Test custom directive names.
  EXPECT_FALSE(errorToBool(
      RISCV::parseTuneFeatureString("enable-default-unroll,no-sink-splat-"
                                    "operands,enable-latency-sched-heuristic",
                                    Result)));
  EXPECT_TRUE(is_contained(Result, "+no-sink-splat-operands"));
  EXPECT_TRUE(is_contained(Result, "-no-default-unroll"));
  EXPECT_TRUE(is_contained(Result, "-disable-latency-sched-heuristic"));
}

TEST(RISCVTuneFeature, IgnoreUnrecognizedTuneFeature) {
  SmallVector<std::string> Result;
  auto Err = RISCV::parseTuneFeatureString("32bit,log-vrgather", Result);
  // This should be an warning.
  EXPECT_TRUE(Err.isA<RISCV::ParserWarning>());
  EXPECT_EQ(toString(std::move(Err)),
            "unrecognized tune feature directive '32bit'");
  EXPECT_TRUE(is_contained(Result, "+log-vrgather"));
}

TEST(RISCVTuneFeature, DuplicatedFeatures) {
  SmallVector<std::string> Result;
  EXPECT_EQ(toString(RISCV::parseTuneFeatureString("log-vrgather,log-vrgather",
                                                   Result)),
            "cannot specify more than one instance of 'log-vrgather'");

  EXPECT_EQ(toString(RISCV::parseTuneFeatureString(
                "log-vrgather,no-log-vrgather,short-forward-branch-i-mul,no-"
                "short-forward-branch-i-mul",
                Result)),
            "Feature(s) 'log-vrgather', 'short-forward-branch-i-mul' cannot "
            "appear in both positive and negative directives");

  // The error message should show the feature name for those using custom
  // directives.
  EXPECT_EQ(
      toString(RISCV::parseTuneFeatureString(
          "disable-latency-sched-heuristic,enable-latency-sched-heuristic",
          Result)),
      "Feature(s) 'disable-latency-sched-heuristic' cannot appear in both "
      "positive and negative directives");

  EXPECT_EQ(
      toString(RISCV::parseTuneFeatureString(
          "short-forward-branch-i-mul,no-short-forward-branch-opt", Result)),
      "Feature(s) 'short-forward-branch-i-mul', 'short-forward-branch-opt' "
      "were implied by both positive and negative directives");
}
} // namespace
