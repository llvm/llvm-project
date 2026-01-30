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
            RISCVVType::getSameRatioLMUL(
                RISCVVType::getSEWLMULRatio(16, RISCVVType::LMUL_2), 8));
  EXPECT_EQ(RISCVVType::LMUL_F2,
            RISCVVType::getSameRatioLMUL(
                RISCVVType::getSEWLMULRatio(16, RISCVVType::LMUL_1), 8));
  // Smaller fractional LMUL.
  EXPECT_EQ(RISCVVType::LMUL_F8,
            RISCVVType::getSameRatioLMUL(
                RISCVVType::getSEWLMULRatio(16, RISCVVType::LMUL_F4), 8));
  // Bigger LMUL.
  EXPECT_EQ(RISCVVType::LMUL_2,
            RISCVVType::getSameRatioLMUL(
                RISCVVType::getSEWLMULRatio(8, RISCVVType::LMUL_1), 16));
  EXPECT_EQ(RISCVVType::LMUL_1,
            RISCVVType::getSameRatioLMUL(
                RISCVVType::getSEWLMULRatio(8, RISCVVType::LMUL_F2), 16));
  // Bigger fractional LMUL.
  EXPECT_EQ(RISCVVType::LMUL_F2,
            RISCVVType::getSameRatioLMUL(
                RISCVVType::getSEWLMULRatio(8, RISCVVType::LMUL_F4), 16));
}

TEST(RISCVTuneFeature, AllTuneFeatures) {
  SmallVector<StringRef> AllTuneFeatures;
  RISCV::getAllTuneFeatures(AllTuneFeatures);
  // Only allowed subtarget features that are explicitly marked by
  // special TableGen class.
  EXPECT_EQ(AllTuneFeatures.size(), 19U);
  for (auto F :
       {"conditional-cmv-fusion", "disable-latency-sched-heuristic",
        "disable-misched-load-clustering", "disable-misched-store-clustering",
        "disable-postmisched-load-clustering",
        "disable-postmisched-store-clustering", "single-element-vec-fp64",
        "no-default-unroll", "no-sink-splat-operands", "use-postra-scheduler",
        "predictable-select-expensive", "prefer-vsetvli-over-read-vlenb",
        "prefer-w-inst", "short-forward-branch-ialu",
        "short-forward-branch-iminmax", "short-forward-branch-imul",
        "short-forward-branch-iload", "vl-dependent-latency",
        "vxrm-pipeline-flush"})
    EXPECT_TRUE(is_contained(AllTuneFeatures, F));
}

TEST(RISCVTuneFeature, LegalTuneFeatureStrings) {
  SmallVector<std::string> Result;
  EXPECT_FALSE(errorToBool(RISCV::parseTuneFeatureString(
      /*ProcName=*/"",
      "prefer-w-inst,no-short-forward-branch-ialu,vl-dependent-latency",
      Result)));
  EXPECT_TRUE(is_contained(Result, "+prefer-w-inst"));
  EXPECT_TRUE(is_contained(Result, "+vl-dependent-latency"));
  EXPECT_TRUE(is_contained(Result, "-short-forward-branch-ialu"));
  EXPECT_TRUE(is_contained(Result, "-short-forward-branch-iminmax"));
  EXPECT_TRUE(is_contained(Result, "-short-forward-branch-imul"));
  EXPECT_TRUE(is_contained(Result, "-short-forward-branch-iload"));

  Result.clear();
  // Test inverse implied features.
  EXPECT_FALSE(errorToBool(RISCV::parseTuneFeatureString(
      /*ProcName=*/"",
      "no-short-forward-branch-imul,short-forward-branch-iminmax", Result)));
  EXPECT_TRUE(is_contained(Result, "+short-forward-branch-iminmax"));
  EXPECT_TRUE(is_contained(Result, "+short-forward-branch-ialu"));
  EXPECT_TRUE(is_contained(Result, "-short-forward-branch-imul"));

  Result.clear();
  // Test custom directive names.
  EXPECT_FALSE(errorToBool(
      RISCV::parseTuneFeatureString(/*ProcName=*/"",
                                    "enable-default-unroll,no-sink-splat-"
                                    "operands,enable-latency-sched-heuristic",
                                    Result)));
  EXPECT_TRUE(is_contained(Result, "+no-sink-splat-operands"));
  EXPECT_TRUE(is_contained(Result, "-no-default-unroll"));
  EXPECT_TRUE(is_contained(Result, "-disable-latency-sched-heuristic"));
}

TEST(RISCVTuneFeature, IgnoreUnrecognizedTuneFeature) {
  SmallVector<std::string> Result;
  auto Err = RISCV::parseTuneFeatureString(/*ProcName=*/"",
                                           "32bit,prefer-w-inst", Result);
  // This should be an warning.
  EXPECT_TRUE(Err.isA<RISCV::ParserWarning>());
  EXPECT_EQ(toString(std::move(Err)),
            "unrecognized tune feature directive '32bit'");
  EXPECT_TRUE(is_contained(Result, "+prefer-w-inst"));
}

TEST(RISCVTuneFeature, DuplicatedFeatures) {
  SmallVector<std::string> Result;
  EXPECT_EQ(toString(RISCV::parseTuneFeatureString(
                /*ProcName=*/"", "prefer-w-inst,prefer-w-inst", Result)),
            "cannot specify more than one instance of 'prefer-w-inst'");

  EXPECT_EQ(toString(RISCV::parseTuneFeatureString(
                /*ProcName=*/"",
                "prefer-w-inst,no-prefer-w-inst,short-forward-branch-imul,no-"
                "short-forward-branch-imul",
                Result)),
            "Feature(s) 'prefer-w-inst', 'short-forward-branch-imul' cannot "
            "appear in both positive and negative directives");

  // The error message should show the feature name for those using custom
  // directives.
  EXPECT_EQ(
      toString(RISCV::parseTuneFeatureString(
          /*ProcName=*/"",
          "disable-latency-sched-heuristic,enable-latency-sched-heuristic",
          Result)),
      "Feature(s) 'disable-latency-sched-heuristic' cannot appear in both "
      "positive and negative directives");

  EXPECT_EQ(
      toString(RISCV::parseTuneFeatureString(
          /*ProcName=*/"",
          "short-forward-branch-imul,no-short-forward-branch-ialu", Result)),
      "Feature(s) 'short-forward-branch-imul', 'short-forward-branch-ialu' "
      "were implied by both positive and negative directives");
}

TEST(RISCVTuneFeature, ProcConfigurableFeatures) {
  SmallVector<std::string> Result;
  EXPECT_FALSE(errorToBool(RISCV::parseTuneFeatureString(
      "sifive-x280", "single-element-vec-fp64", Result)));
  EXPECT_TRUE(is_contained(Result, "+single-element-vec-fp64"));

  Result.clear();
  EXPECT_EQ(
      toString(RISCV::parseTuneFeatureString(
          "sifive-x280", "single-element-vec-fp64,prefer-w-inst", Result)),
      "Directive 'prefer-w-inst' is not allowed to be used with processor "
      "'sifive-x280'");
}

TEST(RISCVTuneFeature, AllProcConfigurableFeatures) {
  SmallVector<StringRef> Result;
  RISCV::getCPUConfigurableTuneFeatures("sifive-x280", Result);
  EXPECT_TRUE(is_contained(Result, "single-element-vec-fp64"));
  EXPECT_TRUE(is_contained(Result, "full-vec-fp64"));
  EXPECT_EQ(Result.size(), 2U);

  Result.clear();
  RISCV::getCPUConfigurableTuneFeatures("sifive-x390", Result);
  EXPECT_TRUE(is_contained(Result, "single-element-vec-fp64"));
  EXPECT_TRUE(is_contained(Result, "full-vec-fp64"));
  EXPECT_EQ(Result.size(), 2U);

  Result.clear();
  RISCV::getCPUConfigurableTuneFeatures("rocket", Result);
  EXPECT_TRUE(Result.empty());
}
} // namespace
