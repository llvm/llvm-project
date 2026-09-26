//=- AArch64Processors.td - Describe AArch64 Processors ------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AArch64 Processor subtarget features.
//===----------------------------------------------------------------------===//


// Processor families only select tuning, so like the tuning features they
// imply they do not prevent inlining.
class ProcFamily<string n, string v, string d,
                 list<SubtargetFeature> i = []>
    : SubtargetFeature<n, "ARMProcFamily", v, d, i, InlineIgnore>;

def TuneA35     : ProcFamily<"a35", "CortexA35",
                                "Cortex-A35 ARM processors">;

def TuneA320 : ProcFamily<"a320", "CortexA320",
                                   "Cortex-A320 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeaturePostRAScheduler,
                                   FeatureUseWzrToVecMove,
                                   FeatureUseFixedOverScalableIfEqualCost,
                                   FeatureAggressiveInterleaving]>;

def TuneA53     : ProcFamily<"a53", "CortexA53",
                                   "Cortex-A53 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureBalanceFPOps,
                                   FeaturePostRAScheduler,
                                   FeatureUseWzrToVecMove]>;

def TuneA55     : ProcFamily<"a55", "CortexA55",
                                   "Cortex-A55 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeaturePostRAScheduler,
                                   FeatureFuseAddress,
                                   FeatureUseWzrToVecMove]>;

def TuneA510    : ProcFamily<"a510", "CortexA510",
                                   "Cortex-A510 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeaturePostRAScheduler,
                                   FeatureUseWzrToVecMove,
                                   FeatureUseFixedOverScalableIfEqualCost
                                   ]>;

def TuneA520    : ProcFamily<"a520", "CortexA520",
                                   "Cortex-A520 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeaturePostRAScheduler,
                                   FeatureUseWzrToVecMove,
                                   FeatureUseFixedOverScalableIfEqualCost]>;

def TuneA520AE  : ProcFamily<"a520ae", "CortexA520",
                                   "Cortex-A520AE ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeaturePostRAScheduler,
                                   FeatureUseWzrToVecMove]>;

def TuneC1Nano : ProcFamily<"c1-nano", "C1Nano", "C1-Nano ARM Processors",[
                                  FeatureFuseAES,
                                  FeatureFuseAdrpAdd,
                                  FeaturePostRAScheduler,
                                  FeatureUseWzrToVecMove,
                                  FeatureUseFixedOverScalableIfEqualCost]>;

def TuneA57     : ProcFamily<"a57", "CortexA57",
                                   "Cortex-A57 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureBalanceFPOps,
                                   FeatureFuseAdrpAdd,
                                   FeatureFuseLiterals,
                                   FeatureAddrLSLSlow14,
                                   FeaturePostRAScheduler,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureMaxInterleaveFactor4]>;

def TuneA65     : ProcFamily<"a65", "CortexA65",
                                   "Cortex-A65 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAddress,
                                   FeatureFuseAdrpAdd,
                                   FeatureFuseLiterals,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA72     : ProcFamily<"a72", "CortexA72",
                                   "Cortex-A72 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureFuseLiterals,
                                   FeatureAddrLSLSlow14,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA73     : ProcFamily<"a73", "CortexA73",
                                   "Cortex-A73 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureAddrLSLSlow14,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA75     : ProcFamily<"a75", "CortexA75",
                                   "Cortex-A75 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureAddrLSLSlow14,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA76     : ProcFamily<"a76", "CortexA76",
                                   "Cortex-A76 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureAddrLSLSlow14,
                                   FeatureALULSLFast,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA77     : ProcFamily<"a77", "CortexA77",
                                   "Cortex-A77 ARM processors", [
                                   FeatureCmpBccFusion,
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureAddrLSLSlow14,
                                   FeatureALULSLFast,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA78 : ProcFamily<"a78", "CortexA78",
                               "Cortex-A78 ARM processors", [
                               FeatureCmpBccFusion,
                               FeatureFuseAES,
                               FeatureFuseAdrpAdd,
                               FeatureFuseCmpCSel,
                               FeatureFuseCmpCSet,
                               FeatureAddrLSLSlow14,
                               FeatureALULSLFast,
                               FeaturePostRAScheduler,
                               FeatureEnableSelectOptimize,
                               FeaturePredictableSelectIsExpensive]>;

def TuneA78AE : ProcFamily<"a78ae", "CortexA78AE",
                                 "Cortex-A78AE ARM processors", [
                                 FeatureCmpBccFusion,
                                 FeatureFuseAES,
                                 FeatureFuseAdrpAdd,
                                 FeatureFuseCmpCSel,
                                 FeatureFuseCmpCSet,
                                 FeatureAddrLSLSlow14,
                                 FeatureALULSLFast,
                                 FeaturePostRAScheduler,
                                 FeatureEnableSelectOptimize,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneA78C : ProcFamily<"a78c", "CortexA78C",
                                "Cortex-A78C ARM processors", [
                                FeatureCmpBccFusion,
                                FeatureFuseAES,
                                FeatureFuseAdrpAdd,
                                FeatureFuseCmpCSel,
                                FeatureFuseCmpCSet,
                                FeatureAddrLSLSlow14,
                                FeatureALULSLFast,
                                FeaturePostRAScheduler,
                                FeatureEnableSelectOptimize,
                                FeaturePredictableSelectIsExpensive]>;

def TuneA710    : ProcFamily<"a710", "CortexA710",
                                   "Cortex-A710 ARM processors", [
                                   FeatureCmpBccFusion,
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureFuseCmpCSel,
                                   FeatureFuseCmpCSet,
                                   FeatureALULSLFast,
                                   FeaturePostRAScheduler,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA715 : ProcFamily<"a715", "CortexA715",
                                 "Cortex-A715 ARM processors", [
                                 FeatureFuseAES,
                                 FeaturePostRAScheduler,
                                 FeatureCmpBccFusion,
                                 FeatureALULSLFast,
                                 FeatureFuseAdrpAdd,
                                 FeatureFuseCmpCSel,
                                 FeatureFuseCmpCSet,
                                 FeatureEnableSelectOptimize,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneA720 : ProcFamily<"a720", "CortexA720",
                                "Cortex-A720 ARM processors", [
                                 FeatureFuseAES,
                                 FeaturePostRAScheduler,
                                 FeatureCmpBccFusion,
                                 FeatureALULSLFast,
                                 FeatureFuseAdrpAdd,
                                 FeatureFuseCmpCSel,
                                 FeatureFuseCmpCSet,
                                 FeatureEnableSelectOptimize,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneA720AE : ProcFamily<"a720ae", "CortexA720",
                                "Cortex-A720AE ARM processors", [
                                 FeatureFuseAES,
                                 FeaturePostRAScheduler,
                                 FeatureCmpBccFusion,
                                 FeatureALULSLFast,
                                 FeatureFuseAdrpAdd,
                                 FeatureFuseCmpCSel,
                                 FeatureFuseCmpCSet,
                                 FeatureEnableSelectOptimize,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneA725 : ProcFamily<"cortex-a725", "CortexA725",
                                "Cortex-A725 ARM processors", [
                                FeatureFuseAES,
                                FeaturePostRAScheduler,
                                FeatureCmpBccFusion,
                                FeatureALULSLFast,
                                FeatureFuseAdrpAdd,
                                FeatureFuseCmpCSel,
                                FeatureFuseCmpCSet,
                                FeatureEnableSelectOptimize,
                                FeaturePredictableSelectIsExpensive]>;

def TuneC1Pro : ProcFamily<"c1-pro", "C1Pro", "C1-Pro ARM Processors",[
                                  FeatureFuseAES,
                                  FeaturePostRAScheduler,
                                  FeatureCmpBccFusion,
                                  FeatureALULSLFast,
                                  FeatureFuseAdrpAdd,
                                  FeatureFuseCmpCSel,
                                  FeatureFuseCmpCSet,
                                  FeatureEnableSelectOptimize,
                                  FeaturePredictableSelectIsExpensive]>;

def TuneR82 : ProcFamily<"cortex-r82", "CortexR82",
                               "Cortex-R82 ARM processors", [
                               FeaturePostRAScheduler]>;

def TuneR82AE : ProcFamily<"cortex-r82ae", "CortexR82AE",
                                 "Cortex-R82-AE ARM processors",
                                 [FeaturePostRAScheduler]>;

def TuneX1 : ProcFamily<"cortex-x1", "CortexX1",
                                  "Cortex-X1 ARM processors", [
                                  FeatureCmpBccFusion,
                                  FeatureFuseAES,
                                  FeatureFuseAdrpAdd,
                                  FeatureAddrLSLSlow14,
                                  FeatureALULSLFast,
                                  FeaturePostRAScheduler,
                                  FeatureEnableSelectOptimize,
                                  FeatureMaxInterleaveFactor4,
                                  FeaturePredictableSelectIsExpensive]>;

def TuneX2 : ProcFamily<"cortex-x2", "CortexX2",
                                  "Cortex-X2 ARM processors", [
                                  FeatureCmpBccFusion,
                                  FeatureFuseAES,
                                  FeatureFuseAdrpAdd,
                                  FeatureALULSLFast,
                                  FeaturePostRAScheduler,
                                  FeatureEnableSelectOptimize,
                                  FeatureUseFixedOverScalableIfEqualCost,
                                  FeatureMaxInterleaveFactor4,
                                  FeaturePredictableSelectIsExpensive]>;

def TuneX3 : ProcFamily<"cortex-x3", "CortexX3",
                              "Cortex-X3 ARM processors", [
                               FeatureALULSLFast,
                               FeatureFuseAdrpAdd,
                               FeatureFuseAES,
                               FeaturePostRAScheduler,
                               FeatureEnableSelectOptimize,
                               FeatureUseFixedOverScalableIfEqualCost,
                               FeatureMaxInterleaveFactor4,
                               FeatureAvoidLDAPUR,
                               FeaturePredictableSelectIsExpensive]>;

def TuneX4 : ProcFamily<"cortex-x4", "CortexX4",
                              "Cortex-X4 ARM processors", [
                               FeatureALULSLFast,
                               FeatureFuseAdrpAdd,
                               FeatureFuseCmpCSel,
                               FeatureFuseCmpCSet,
                               FeatureFuseAES,
                               FeaturePostRAScheduler,
                               FeatureEnableSelectOptimize,
                               FeatureUseFixedOverScalableIfEqualCost,
                               FeatureMaxInterleaveFactor4,
                               FeatureAvoidLDAPUR,
                               FeaturePredictableSelectIsExpensive]>;

def TuneX925 : ProcFamily<"cortex-x925", "CortexX925", "Cortex-X925 ARM processors",[
                                FeatureALULSLFast,
                                FeatureFuseAdrpAdd,
                                FeatureFuseCmpCSel,
                                FeatureFuseCmpCSet,
                                FeatureFuseAES,
                                FeaturePostRAScheduler,
                                FeatureEnableSelectOptimize,
                                FeatureUseFixedOverScalableIfEqualCost,
                                FeatureMaxInterleaveFactor4,
                                FeatureAvoidLDAPUR,
                                FeaturePredictableSelectIsExpensive]>;

def TuneC1Premium : ProcFamily<"c1-premium", "C1Premium", "C1-Premium ARM Processors",[
                                  FeatureALULSLFast,
                                  FeatureFuseAdrpAdd,
                                  FeatureFuseCmpCSel,
                                  FeatureFuseCmpCSet,
                                  FeatureFuseAES,
                                  FeaturePostRAScheduler,
                                  FeatureEnableSelectOptimize,
                                  FeatureUseFixedOverScalableIfEqualCost,
                                  FeatureAvoidLDAPUR,
                                  FeaturePredictableSelectIsExpensive]>;

def TuneC1Ultra : ProcFamily<"c1-ultra", "C1Ultra", "C1-Ultra ARM Processors",[
                                  FeatureALULSLFast,
                                  FeatureFuseAdrpAdd,
                                  FeatureFuseCmpCSel,
                                  FeatureFuseCmpCSet,
                                  FeatureFuseAES,
                                  FeaturePostRAScheduler,
                                  FeatureEnableSelectOptimize,
                                  FeatureUseFixedOverScalableIfEqualCost,
                                  FeatureMaxInterleaveFactor4,
                                  FeatureAvoidLDAPUR,
                                  FeaturePredictableSelectIsExpensive]>;

def TuneA64FX : ProcFamily<"a64fx", "A64FX",
                                 "Fujitsu A64FX processors", [
                                 FeaturePostRAScheduler,
                                 FeatureAggressiveFMA,
                                 FeatureArithmeticBccFusion,
                                 FeatureStorePairSuppress,
                                 FeaturePredictableSelectIsExpensive,
                                 FeatureDisableUnpredicatedLdStLower,
                                 FeatureMaxInterleaveFactor4,
                                 FeatureUseReciprocalFDivCombineThreshold2]>;

def TuneMONAKA : ProcFamily<"fujitsu-monaka", "MONAKA",
                                 "Fujitsu FUJITSU-MONAKA processors", [
                                 FeaturePredictableSelectIsExpensive,
                                 FeatureEnableSelectOptimize,
                                 FeaturePostRAScheduler,
                                 FeatureArithmeticBccFusion,
                                 ]>;

def TuneCarmel : ProcFamily<"carmel", "Carmel",
                                  "Nvidia Carmel processors">;

def TuneOlympus : ProcFamily<"olympus", "Olympus",
                                   "NVIDIA Olympus processors", [
                                   FeatureALULSLFast,
                                   FeatureCmpBccFusion,
                                   FeatureEnableSelectOptimize,
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureUseFixedOverScalableIfEqualCost,
                                   FeatureMaxInterleaveFactor4]>;

def TuneAppleA7  : ProcFamily<"apple-a7", "AppleA7",
                                    "Apple A7 (the CPU formerly known as Cyclone)",
                                    [FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFuseAES,
                                    FeatureFuseCryptoEOR,
                                    FeatureStorePairSuppress,
                                    FeatureZCRegMoveGPR64,
                                    FeatureZCRegMoveFPR128,
                                    FeatureZCZeroingFPWorkaround,
                                    FeatureZCZeroingGPR32,
                                    FeatureZCZeroingGPR64,
                                    FeatureNoZCZeroingFPR64,
                                    FeatureZCZeroingFPR128]>;

def TuneAppleA10 : ProcFamily<"apple-a10", "AppleA10",
                                    "Apple A10",
                                    !listremove(TuneAppleA7.Implies,
                                    [FeatureZCZeroingFPWorkaround])>;

def TuneAppleA11 : ProcFamily<"apple-a11", "AppleA11",
                                    "Apple A11", TuneAppleA10.Implies>;

def TuneAppleA12 : ProcFamily<"apple-a12", "AppleA12",
                                    "Apple A12", TuneAppleA11.Implies>;

def TuneAppleA13 : ProcFamily<"apple-a13", "AppleA13",
                                    "Apple A13", TuneAppleA12.Implies>;

def TuneAppleA14 : ProcFamily<"apple-a14", "AppleA14",
                                    "Apple A14",
                                    !listconcat(TuneAppleA13.Implies, [
                                    FeatureAggressiveFMA,
                                    FeatureAlignCmpCSelPairs,
                                    FeatureFastLD1Single,
                                    FeatureFuseArithmeticLogic,
                                    FeatureFuseCmpCSel,
                                    FeatureFuseLiterals,
                                    FeatureMaxInterleaveFactor4])>;

def TuneAppleA15 : ProcFamily<"apple-a15", "AppleA15",
                                    "Apple A15",
                                    !listconcat(
                                    !listremove(TuneAppleA14.Implies,
                                    [FeatureAggressiveFMA]),
                                    [FeatureFuseAdrpAdd])>;

def TuneAppleA16 : ProcFamily<"apple-a16", "AppleA16",
                                    "Apple A16",
                                    !listconcat(TuneAppleA15.Implies,
                                    [FeatureFuseFCmpFCSel])>;

def TuneAppleA17 : ProcFamily<"apple-a17", "AppleA17",
                                    "Apple A17", TuneAppleA16.Implies>;

def TuneAppleM4 : ProcFamily<"apple-m4", "AppleM4",
                                    "Apple M4",
                                    !listconcat(
                                    !listremove(TuneAppleA17.Implies,
                                    [FeatureStorePairSuppress]),
                                    [FeatureFuseAppleSMECompute])>;

def TuneAppleM5 : ProcFamily<"apple-m5", "AppleM5",
                                    "Apple M5",
                                    !listconcat(TuneAppleM4.Implies,
                                    [FeatureFuseFMinFMax])>;

def TuneExynosM3 : ProcFamily<"exynosm3", "ExynosM3",
                                    "Samsung Exynos-M3 processors",
                                    [FeatureExynosCheapAsMoveHandling,
                                     FeatureForce32BitJumpTables,
                                     FeatureFuseAddress,
                                     FeatureFuseAES,
                                     FeatureFuseCmpCSel,
                                     FeatureFuseAdrpAdd,
                                     FeatureFuseLiterals,
                                     FeatureStorePairSuppress,
                                     FeatureALULSLFast,
                                     FeaturePostRAScheduler,
                                     FeaturePredictableSelectIsExpensive,
                                     FeatureMaxInterleaveFactor4]>;

// Re-uses some scheduling and tunings from the ExynosM3 proc family.
def TuneExynosM4 : ProcFamily<"exynosm4", "ExynosM3",
                                    "Samsung Exynos-M4 processors",
                                    [FeatureArithmeticBccFusion,
                                     FeatureArithmeticCbzFusion,
                                     FeatureExynosCheapAsMoveHandling,
                                     FeatureForce32BitJumpTables,
                                     FeatureFuseAddress,
                                     FeatureFuseAES,
                                     FeatureFuseArithmeticLogic,
                                     FeatureFuseCmpCSel,
                                     FeatureFuseAdrpAdd,
                                     FeatureFuseLiterals,
                                     FeatureStorePairSuppress,
                                     FeatureALULSLFast,
                                     FeaturePostRAScheduler,
                                     FeatureZCZeroingGPR32,
                                     FeatureZCZeroingGPR64,
                                     FeatureMaxInterleaveFactor4]>;

def TuneKryo    : ProcFamily<"kryo", "Kryo",
                                   "Qualcomm Kryo processors", [
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroingGPR32,
                                   FeatureZCZeroingGPR64,
                                   FeatureALULSLFast,
                                   FeatureStorePairSuppress,
                                   FeatureMaxInterleaveFactor4]>;

def TuneFalkor  : ProcFamily<"falkor", "Falkor",
                                   "Qualcomm Falkor processors", [
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroingGPR32,
                                   FeatureZCZeroingGPR64,
                                   FeatureStorePairSuppress,
                                   FeatureALULSLFast,
                                   FeatureSlowSTRQro,
                                   FeatureMaxInterleaveFactor4]>;

def TuneNeoverseE1 : ProcFamily<"neoversee1", "NeoverseE1",
                                      "Neoverse E1 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeaturePostRAScheduler]>;

def TuneNeoverseN1 : ProcFamily<"neoversen1", "NeoverseN1",
                                      "Neoverse N1 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureAddrLSLSlow14,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneNeoverseN2 : ProcFamily<"neoversen2", "NeoverseN2",
                                      "Neoverse N2 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureFuseCmpCSel,
                                      FeatureFuseCmpCSet,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeatureUseFixedOverScalableIfEqualCost,
                                      FeaturePredictableSelectIsExpensive,
                                      FeatureLimited64bitVectorMulBandwidth]>;

def TuneNeoverseN3 : ProcFamily<"neoversen3", "NeoverseN3",
                                      "Neoverse N3 ARM processors", [
                                      FeatureFuseAES,
                                      FeaturePostRAScheduler,
                                      FeatureALULSLFast,
                                      FeatureFuseAdrpAdd,
                                      FeatureFuseCmpCSel,
                                      FeatureFuseCmpCSet,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneNeoverse512TVB : ProcFamily<"neoverse512tvb", "Neoverse512TVB",
                                      "Neoverse 512-TVB ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive,
                                      FeatureMaxInterleaveFactor4]>;

def TuneNeoverseV1 : ProcFamily<"neoversev1", "NeoverseV1",
                                      "Neoverse V1 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureFuseCmpCSel,
                                      FeatureFuseCmpCSet,
                                      FeatureAddrLSLSlow14,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive,
                                      FeatureDisableMaximizeScalableBandwidth,
                                      FeatureNoSVEFPLD1R,
                                      FeatureDontPreferPredicatedLD1R]>;

def TuneNeoverseV2 : ProcFamily<"neoversev2", "NeoverseV2",
                                      "Neoverse V2 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureCmpBccFusion,
                                      FeatureFuseAdrpAdd,
                                      FeatureFuseCmpCSel,
                                      FeatureFuseCmpCSet,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeatureUseFixedOverScalableIfEqualCost,
                                      FeatureAvoidLDAPUR,
                                      FeaturePredictableSelectIsExpensive,
                                      FeatureDisableLatencySchedHeuristic,
                                      FeatureMaxInterleaveFactor4,
                                      FeatureDontPreferPredicatedLD1R]>;

def TuneNeoverseV3 : ProcFamily<"neoversev3", "NeoverseV3",
                                      "Neoverse V3 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureALULSLFast,
                                      FeatureFuseAdrpAdd,
                                      FeatureFuseCmpCSel,
                                      FeatureFuseCmpCSet,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeatureUseFixedOverScalableIfEqualCost,
                                      FeatureAvoidLDAPUR,
                                      FeaturePredictableSelectIsExpensive,
                                      FeatureMaxInterleaveFactor4]>;

def TuneNeoverseV3AE : ProcFamily<"neoversev3AE", "NeoverseV3AE",
                                      "Neoverse V3AE ARM processors", [
                                      FeatureFuseAES,
                                      FeatureALULSLFast,
                                      FeatureFuseAdrpAdd,
                                      FeatureFuseCmpCSel,
                                      FeatureFuseCmpCSet,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeatureUseFixedOverScalableIfEqualCost,
                                      FeatureAvoidLDAPUR,
                                      FeaturePredictableSelectIsExpensive,
                                      FeatureLimited64bitVectorMulBandwidth]>;

def TuneSaphira  : ProcFamily<"saphira", "Saphira",
                                   "Qualcomm Saphira processors", [
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroingGPR32,
                                   FeatureZCZeroingGPR64,
                                   FeatureStorePairSuppress,
                                   FeatureALULSLFast,
                                   FeatureMaxInterleaveFactor4]>;

def TuneThunderX2T99  : ProcFamily<"thunderx2t99", "ThunderX2T99",
                                         "Cavium ThunderX2 processors", [
                                          FeatureAggressiveFMA,
                                          FeatureArithmeticBccFusion,
                                          FeaturePostRAScheduler,
                                          FeatureStorePairSuppress,
                                          FeaturePredictableSelectIsExpensive,
                                          FeatureMaxInterleaveFactor4]>;

def TuneThunderX3T110  : ProcFamily<"thunderx3t110", "ThunderX3T110",
                                          "Marvell ThunderX3 processors", [
                                           FeatureAggressiveFMA,
                                           FeatureArithmeticBccFusion,
                                           FeaturePostRAScheduler,
                                           FeaturePredictableSelectIsExpensive,
                                           FeatureBalanceFPOps,
                                           FeatureStorePairSuppress,
                                           FeatureStrictAlign,
                                           FeatureMaxInterleaveFactor4]>;

def TuneThunderX : ProcFamily<"thunderx", "ThunderX",
                                    "Cavium ThunderX processors", [
                                    FeaturePostRAScheduler,
                                    FeatureStorePairSuppress,
                                    FeaturePredictableSelectIsExpensive]>;

def TuneThunderXT88 : ProcFamily<"thunderxt88", "ThunderXT88",
                                       "Cavium ThunderX processors", [
                                       FeaturePostRAScheduler,
                                       FeatureStorePairSuppress,
                                       FeaturePredictableSelectIsExpensive]>;

def TuneThunderXT81 : ProcFamily<"thunderxt81", "ThunderXT81",
                                       "Cavium ThunderX processors", [
                                       FeaturePostRAScheduler,
                                       FeatureStorePairSuppress,
                                       FeaturePredictableSelectIsExpensive]>;

def TuneThunderXT83 : ProcFamily<"thunderxt83", "ThunderXT83",
                                       "Cavium ThunderX processors", [
                                       FeaturePostRAScheduler,
                                       FeatureStorePairSuppress,
                                       FeaturePredictableSelectIsExpensive]>;

def TuneTSV110 : ProcFamily<"tsv110", "TSV110",
                                  "HiSilicon TS-V110 processors", [
                                  FeatureFuseAES,
                                  FeatureStorePairSuppress,
                                  FeaturePostRAScheduler]>;

def TuneHIP12 : ProcFamily<"hip12", "HIP12",
                                 "HiSilicon HIP12 processors", [
                                 FeatureCmpBccFusion,
                                 FeatureFuseAES,
                                 FeatureAddrLSLSlow14,
                                 FeatureALULSLFast,
                                 FeatureStorePairSuppress,
                                 FeaturePostRAScheduler]>;

def TuneAmpere1 : ProcFamily<"ampere1", "Ampere1",
                                   "Ampere Computing Ampere-1 processors", [
                                   FeaturePostRAScheduler,
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureALULSLFast,
                                   FeatureAggressiveFMA,
                                   FeatureArithmeticBccFusion,
                                   FeatureCmpBccFusion,
                                   FeatureFuseAddress,
                                   FeatureFuseLiterals,
                                   FeatureStorePairSuppress,
                                   FeatureLdpAlignedOnly,
                                   FeatureStpAlignedOnly,
                                   FeatureMaxInterleaveFactor4]>;

def TuneAmpere1A : ProcFamily<"ampere1a", "Ampere1A",
                                    "Ampere Computing Ampere-1A processors", [
                                    FeaturePostRAScheduler,
                                    FeatureFuseAES,
                                    FeatureFuseAdrpAdd,
                                    FeatureALULSLFast,
                                    FeatureAggressiveFMA,
                                    FeatureArithmeticBccFusion,
                                    FeatureCmpBccFusion,
                                    FeatureFuseAddress,
                                    FeatureFuseLiterals,
                                    FeatureFuseAddSub2RegAndConstOne,
                                    FeatureStorePairSuppress,
                                    FeatureLdpAlignedOnly,
                                    FeatureStpAlignedOnly,
                                    FeatureMaxInterleaveFactor4]>;

def TuneAmpere1B : ProcFamily<"ampere1b", "Ampere1B",
                                    "Ampere Computing Ampere-1B processors", [
                                    FeaturePostRAScheduler,
                                    FeatureFuseAES,
                                    FeatureFuseAdrpAdd,
                                    FeatureALULSLFast,
                                    FeatureAggressiveFMA,
                                    FeatureArithmeticBccFusion,
                                    FeatureCmpBccFusion,
                                    FeatureFuseAddress,
                                    FeatureFuseLiterals,
                                    FeatureStorePairSuppress,
                                    FeatureEnableSelectOptimize,
                                    FeaturePredictableSelectIsExpensive,
                                    FeatureLdpAlignedOnly,
                                    FeatureStpAlignedOnly,
                                    FeatureMaxInterleaveFactor4]>;

def TuneAmpere1C : ProcFamily<"ampere1c", "Ampere1C",
                                    "Ampere Computing Ampere-1C processors", [
                                    FeaturePostRAScheduler,
                                    FeatureFuseAES,
                                    FeatureFuseAdrpAdd,
                                    FeatureALULSLFast,
                                    FeatureAggressiveFMA,
                                    FeatureArithmeticBccFusion,
                                    FeatureCmpBccFusion,
                                    FeatureFuseAddress,
                                    FeatureFuseLiterals,
                                    FeatureStorePairSuppress,
                                    FeatureEnableSelectOptimize,
                                    FeaturePredictableSelectIsExpensive,
                                    FeatureMaxInterleaveFactor4]>;

def TuneOryon  : ProcFamily<"oryon-1", "Oryon",
                                    "Nuvia Inc Oryon processors", [
                                    FeatureSHA2,
                                    FeatureAES,
                                    FeatureFPARMv8,
                                    FeatureNEON,
                                    FeatureFuseAES,
                                    FeatureFuseAdrpAdd,
                                    FeatureEnableSelectOptimize,
                                    FeatureFuseCryptoEOR,
                                    FeatureFuseAddress,
                                    FeatureSM4,
                                    FeatureSHA2,
                                    FeatureSHA3,
                                    FeatureAES,
                                    FeatureFullFP16,
                                    FeatureFP16FML,
                                    FeaturePerfMon,
                                    FeatureSPE,
                                    FeaturePostRAScheduler,
                                    HasV8_6aOps,
                                    FeatureMaxInterleaveFactor4]>;

def ProcessorFeatures {
  list<SubtargetFeature> A320 = [HasV9_2aOps, FeatureNEON, FeatureMTE,
                                 FeatureSVEBitPerm, FeatureFP16FML, FeatureFullFP16,
                                 FeatureETE, FeaturePerfMon, FeatureCCIDX,
                                 FeatureCCPP, FeatureComplxNum, FeatureCRC,
                                 FeatureDotProd, FeatureJS, FeatureLOR,
                                 FeatureLSE, FeaturePAN, FeaturePAN_RWV,
                                 FeaturePAuth, FeaturePsUAO, FeatureRAS,
                                 FeatureRDM, FeatureTRBE, FeatureVH,
                                 FeatureFlagM, FeaturePredRes, FeatureSB, FeatureSSBS,
                                 FeatureSVE, FeatureSVE2];
  list<SubtargetFeature> A53  = [HasV8_0aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                 FeatureFPARMv8, FeatureNEON, FeaturePerfMon];
  list<SubtargetFeature> A55  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeaturePerfMon, FeatureCRC,
                                 FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> A510 = [HasV9_0aOps, FeatureNEON, FeaturePerfMon,
                                 FeatureMatMulInt8, FeatureBF16, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVEBitPerm,
                                 FeatureFP16FML,
                                 FeatureCCIDX,
                                 FeatureSB, FeaturePAuth, FeatureSSBS, FeatureSVE, FeatureSVE2,
                                 FeatureComplxNum, FeatureCRC, FeatureDotProd,
                                 FeatureFPARMv8,FeatureFullFP16, FeatureJS, FeatureLSE,
                                 FeatureRAS, FeatureRCPC, FeatureRDM, FeatureFPAC];
  list<SubtargetFeature> A520 = [HasV9_2aOps, FeaturePerfMon, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVEBitPerm,
                                 FeatureFP16FML,
                                 FeatureCCIDX,
                                 FeatureSB, FeatureSSBS, FeaturePAuth, FeatureFlagM, FeaturePredRes,
                                 FeatureSVE, FeatureSVE2, FeatureBF16, FeatureComplxNum, FeatureCRC,
                                 FeatureFPARMv8, FeatureFullFP16, FeatureMatMulInt8, FeatureJS,
                                 FeatureNEON, FeatureLSE, FeatureRAS, FeatureRCPC, FeatureRDM,
                                 FeatureDotProd, FeatureFPAC];
  list<SubtargetFeature> A520AE = [HasV9_2aOps, FeaturePerfMon, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVEBitPerm,
                                 FeatureFP16FML,
                                 FeatureCCIDX,
                                 FeatureSB, FeatureSSBS, FeaturePAuth, FeatureFlagM, FeaturePredRes,
                                 FeatureSVE, FeatureSVE2, FeatureBF16, FeatureComplxNum, FeatureCRC,
                                 FeatureFPARMv8, FeatureFullFP16, FeatureMatMulInt8, FeatureJS,
                                 FeatureNEON, FeatureLSE, FeatureRAS, FeatureRCPC, FeatureRDM,
                                 FeatureDotProd, FeatureFPAC];
  list<SubtargetFeature> C1Nano = [HasV9_3aOps, FeatureNEON, FeatureCLRBHB,
                                   FeatureCHK, FeatureFPARMv8, FeaturePerfMon,
                                   FeatureSPECRES2, FeatureSSBS, FeatureRDM,
                                   FeatureVH, FeatureBF16, FeatureDotProd,
                                   FeatureFP16FML, FeatureFullFP16, FeatureMPAM,
                                   FeatureSVE, FeatureCCIDX, FeatureComplxNum,
                                   FeatureFPAC, FeatureJS, FeatureAM,
                                   FeatureRAS, FeatureSEL2, FeatureTRACEV8_4,
                                   FeatureAltFPCmp, FeatureFRInt3264,
                                   FeatureMTE, FeatureFineGrainedTraps,
                                   FeatureRCPC3, FeatureETE,
                                   FeatureSVEBitPerm, FeatureSVE2, FeatureTRBE,
                                   FeatureSME, FeatureSME2];
  list<SubtargetFeature> A65  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeatureSSBS, FeatureRAS,
                                 FeaturePerfMon, FeatureCRC, FeatureLSE, FeatureRDM];
  list<SubtargetFeature> A76  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeatureSSBS, FeaturePerfMon,
                                 FeatureCRC, FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> A77  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeaturePerfMon, FeatureSSBS,
                                 FeatureCRC, FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> A78  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeaturePerfMon, FeatureSPE,
                                 FeatureSSBS, FeatureCRC, FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> A78AE = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                  FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                  FeatureRCPC, FeaturePerfMon, FeatureSPE,
                                  FeatureSSBS, FeatureCRC, FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> A78C = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureFlagM, FeaturePAuth,
                                 FeaturePerfMon, FeatureRCPC, FeatureSPE,
                                 FeatureSSBS, FeatureCRC, FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> A710 = [HasV9_0aOps, FeatureNEON, FeaturePerfMon,
                                 FeatureCCIDX, FeatureSSBS,
                                 FeatureETE, FeatureMTE, FeatureFP16FML,
                                 FeatureSVEBitPerm, FeatureBF16, FeatureMatMulInt8,
                                 FeaturePAuth, FeatureFlagM, FeatureSB, FeatureSVE, FeatureSVE2, FeatureFPAC,
                                 FeatureComplxNum, FeatureCRC, FeatureDotProd, FeatureFPARMv8,
                                 FeatureFullFP16, FeatureJS, FeatureLSE, FeatureRAS, FeatureRCPC, FeatureRDM];
  list<SubtargetFeature> A715 = [HasV9_0aOps, FeatureNEON, FeatureMTE,
                                 FeatureCCIDX,
                                 FeatureFP16FML, FeatureSVE, FeatureTRBE,
                                 FeatureSVEBitPerm, FeatureBF16, FeatureETE,
                                 FeaturePerfMon, FeatureMatMulInt8, FeatureSPE,
                                 FeatureSB, FeatureSSBS, FeatureFullFP16, FeaturePAuth, FeaturePredRes, FeatureFlagM,
                                 FeatureSVE2, FeatureComplxNum, FeatureCRC,
                                 FeatureDotProd, FeatureFPARMv8,
                                 FeatureJS, FeatureLSE, FeatureRAS,
                                 FeatureRCPC, FeatureRDM, FeatureFPAC];
  list<SubtargetFeature> A720 = [HasV9_2aOps, FeatureMTE, FeatureFP16FML,
                                 FeatureCCIDX,
                                 FeatureTRBE, FeatureSVEBitPerm, FeatureETE,
                                 FeaturePerfMon, FeatureSPE, FeatureSPE_EEF,
                                 FeatureSB, FeatureSSBS, FeaturePAuth, FeatureFlagM, FeaturePredRes,
                                 FeatureSVE, FeatureSVE2, FeatureBF16, FeatureComplxNum, FeatureCRC,
                                 FeatureDotProd, FeatureFPARMv8, FeatureFullFP16, FeatureMatMulInt8,
                                 FeatureJS, FeatureLSE, FeatureNEON, FeatureRAS,
                                 FeatureRCPC, FeatureRDM, FeatureFPAC];
  list<SubtargetFeature> A720AE = [HasV9_2aOps, FeatureMTE, FeatureFP16FML,
                                 FeatureCCIDX,
                                 FeatureTRBE, FeatureSVEBitPerm, FeatureETE,
                                 FeaturePerfMon, FeatureSPE, FeatureSPE_EEF,
                                 FeatureSB, FeatureSSBS, FeaturePAuth, FeatureFlagM, FeaturePredRes,
                                 FeatureSVE, FeatureSVE2, FeatureBF16, FeatureComplxNum, FeatureCRC,
                                 FeatureDotProd, FeatureFPARMv8, FeatureFullFP16, FeatureMatMulInt8,
                                 FeatureJS, FeatureLSE, FeatureNEON, FeatureRAS,
                                 FeatureRCPC, FeatureRDM, FeatureFPAC];
  list<SubtargetFeature> A725 = [HasV9_2aOps, FeatureMTE, FeatureFP16FML,
                                 FeatureCCIDX,
                                 FeatureETE, FeaturePerfMon, FeatureSPE,
                                 FeatureSVEBitPerm, FeatureSPE_EEF, FeatureTRBE,
                                 FeatureFlagM, FeaturePredRes, FeatureSB, FeatureSSBS,
                                 FeatureSVE, FeatureSVE2, FeatureBF16, FeatureComplxNum, FeatureCRC,
                                 FeatureDotProd, FeatureFPARMv8, FeatureFullFP16, FeatureMatMulInt8,
                                 FeatureJS, FeatureLSE, FeatureNEON, FeaturePAuth, FeatureRAS,
                                 FeatureRCPC, FeatureRDM, FeatureFPAC];
  list<SubtargetFeature> C1Pro = [HasV9_3aOps, FeatureNEON, FeatureCLRBHB,
                                  FeatureCHK, FeatureFPARMv8, FeaturePerfMon,
                                  FeatureSPECRES2, FeatureSSBS, FeatureRDM,
                                  FeatureVH, FeatureBF16, FeatureDotProd,
                                  FeatureFP16FML, FeatureFullFP16, FeatureMPAM,
                                  FeatureSPE, FeatureSVE, FeatureCCIDX,
                                  FeatureComplxNum, FeatureFPAC, FeatureJS,
                                  FeatureAM, FeatureRAS, FeatureSEL2,
                                  FeatureTRACEV8_4, FeatureAltFPCmp,
                                  FeatureFRInt3264, FeatureMTE,
                                  FeatureFineGrainedTraps,
                                  FeatureSPE_EEF, FeatureRCPC3, FeatureETE,
                                  FeatureSVEBitPerm, FeatureSVE2, FeatureTRBE,
                                  FeatureSME, FeatureSME2];
  list<SubtargetFeature> R82  = [HasV8_0rOps, FeaturePerfMon, FeatureFullFP16,
                                 FeatureFP16FML, FeatureSSBS, FeaturePredRes,
                                 FeatureSB, FeatureRDM, FeatureDotProd,
                                 FeatureComplxNum, FeatureJS,
                                 FeatureCacheDeepPersist,
                                 FeatureFlagM, FeatureCRC, FeatureLSE, FeatureRAS, FeatureFPARMv8,
                                 FeatureNEON, FeaturePAuth, FeatureRCPC, FeatureFPAC];
  list<SubtargetFeature> R82AE = [HasV8_0rOps, FeaturePerfMon, FeatureFullFP16,
                                  FeatureFP16FML, FeatureSSBS, FeaturePredRes,
                                  FeatureSB, FeatureRDM, FeatureDotProd,
                                  FeatureComplxNum, FeatureJS,
                                  FeatureCacheDeepPersist,
                                  FeatureLSE, FeatureFlagM, FeatureCRC, FeatureFPARMv8, FeatureNEON,
                                  FeaturePAuth, FeatureRAS, FeatureRCPC, FeatureFPAC];
  list<SubtargetFeature> X1   = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureRCPC, FeaturePerfMon,
                                 FeatureSPE, FeatureFullFP16, FeatureDotProd,
                                 FeatureSSBS, FeatureCRC, FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> X1C  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureRCPC_IMMO, FeaturePerfMon,
                                 FeatureSPE, FeatureFullFP16, FeatureDotProd,
                                 FeaturePAuth, FeatureSSBS, FeatureFlagM,
                                 FeatureLSE2,
                                 FeatureRCPC, FeatureCRC, FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> X2   = [HasV9_0aOps, FeatureNEON, FeaturePerfMon,
                                 FeatureMatMulInt8, FeatureBF16, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVEBitPerm,
                                 FeatureFP16FML,
                                 FeatureCCIDX,
                                 FeaturePAuth, FeatureSSBS, FeatureSB, FeatureSVE, FeatureSVE2, FeatureFlagM,
                                 FeatureComplxNum, FeatureCRC, FeatureDotProd, FeatureFPARMv8, FeatureFullFP16,
                                 FeatureJS, FeatureLSE, FeatureRAS, FeatureRCPC, FeatureRDM, FeatureFPAC];
  list<SubtargetFeature> X3 =   [HasV9_0aOps, FeatureSVE, FeatureNEON,
                                 FeaturePerfMon, FeatureETE, FeatureTRBE,
                                 FeatureSPE, FeatureBF16, FeatureMatMulInt8,
                                 FeatureMTE, FeatureSVEBitPerm, FeatureFullFP16,
                                 FeatureFP16FML,
                                 FeatureCCIDX,
                                 FeatureSB, FeaturePAuth, FeaturePredRes, FeatureFlagM, FeatureSSBS,
                                 FeatureSVE2, FeatureComplxNum, FeatureCRC, FeatureFPARMv8, FeatureJS,
                                 FeatureLSE, FeatureRAS, FeatureRCPC, FeatureRDM, FeatureDotProd, FeatureFPAC];
  list<SubtargetFeature> X4 =   [HasV9_2aOps,
                                 FeaturePerfMon, FeatureETE, FeatureTRBE,
                                 FeatureSPE, FeatureMTE, FeatureSVEBitPerm,
                                 FeatureFP16FML, FeatureSPE_EEF,
                                 FeatureCCIDX,
                                 FeatureSB, FeatureSSBS, FeaturePAuth, FeatureFlagM, FeaturePredRes,
                                 FeatureSVE, FeatureSVE2, FeatureComplxNum, FeatureCRC, FeatureDotProd,
                                 FeatureFPARMv8, FeatureFullFP16, FeatureMatMulInt8, FeatureJS, FeatureLSE,
                                 FeatureNEON, FeatureRAS, FeatureRCPC, FeatureRDM, FeatureBF16, FeatureFPAC];
  list<SubtargetFeature> X925 = [HasV9_2aOps, FeatureMTE, FeatureFP16FML,
                                 FeatureCCIDX,
                                 FeatureETE, FeaturePerfMon, FeatureSPE,
                                 FeatureSVEBitPerm, FeatureSPE_EEF, FeatureTRBE,
                                 FeatureFlagM, FeaturePredRes, FeatureSB, FeatureSSBS,
                                 FeatureSVE, FeatureSVE2, FeatureBF16, FeatureComplxNum, FeatureCRC,
                                 FeatureDotProd, FeatureFPARMv8, FeatureFullFP16, FeatureMatMulInt8,
                                 FeatureJS, FeatureLSE, FeatureNEON, FeaturePAuth, FeatureRAS,
                                 FeatureRCPC, FeatureRDM, FeatureFPAC];
  list<SubtargetFeature> C1Premium = [HasV9_3aOps, FeatureNEON, FeatureCLRBHB,
                                      FeatureCHK, FeatureFPARMv8,
                                      FeaturePerfMon, FeatureSPECRES2,
                                      FeatureSSBS, FeatureRDM, FeatureVH,
                                      FeatureBF16, FeatureDotProd,
                                      FeatureFP16FML, FeatureFullFP16,
                                      FeatureMPAM, FeatureSPE, FeatureSVE,
                                      FeatureCCIDX, FeatureComplxNum,
                                      FeatureFPAC, FeatureJS, FeatureAM,
                                      FeatureRAS, FeatureSEL2, FeatureTRACEV8_4,
                                      FeatureAltFPCmp, FeatureFRInt3264,
                                      FeatureMTE, FeatureFineGrainedTraps,
                                      FeatureSPE_EEF, FeatureRCPC3,
                                      FeatureETE, FeatureSVEBitPerm,
                                      FeatureSVE2, FeatureTRBE, FeatureSME,
                                      FeatureSME2];
  list<SubtargetFeature> C1Ultra = [HasV9_3aOps, FeatureNEON, FeatureCLRBHB,
                                    FeatureCHK, FeatureFPARMv8, FeaturePerfMon,
                                    FeatureSPECRES2, FeatureSSBS, FeatureRDM,
                                    FeatureVH, FeatureBF16, FeatureDotProd,
                                    FeatureFP16FML, FeatureFullFP16,
                                    FeatureMPAM, FeatureSPE, FeatureSVE,
                                    FeatureCCIDX, FeatureComplxNum, FeatureFPAC,
                                    FeatureJS, FeatureAM, FeatureRAS,
                                    FeatureSEL2, FeatureTRACEV8_4,
                                    FeatureAltFPCmp, FeatureFRInt3264,
                                    FeatureMTE, FeatureFineGrainedTraps,
                                    FeatureSPE_EEF, FeatureRCPC3,
                                    FeatureETE, FeatureSVEBitPerm, FeatureSVE2,
                                    FeatureTRBE, FeatureSME, FeatureSME2];
  list<SubtargetFeature> A64FX    = [HasV8_2aOps, FeatureFPARMv8, FeatureNEON,
                                     FeatureSHA2, FeaturePerfMon, FeatureFullFP16,
                                     FeatureSVE, FeatureComplxNum,
                                     FeatureAES, FeatureCRC, FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> MONAKA = [HasV9_3aOps, FeaturePerfMon, FeatureCCIDX,
                                   FeatureFPAC, FeatureFP16FML, FeatureRandGen,
                                   FeatureSSBS, FeatureLS64, FeatureCLRBHB,
                                   FeatureSPECRES2, FeatureSVEAES, FeatureSVESM4,
                                   FeatureSVESHA3, FeatureSVE2, FeatureSVEBitPerm, FeatureETE,
                                   FeatureMEC, FeatureFAMINMAX, FeatureFP8DOT2, FeatureFP8DOT4,
                                   FeatureFP8FMA, FeatureLUT];
  list<SubtargetFeature> Carmel   = [HasV8_2aOps, FeatureNEON, FeatureSHA2, FeatureAES,
                                     FeatureFullFP16, FeatureCRC, FeatureLSE, FeatureRAS, FeatureRDM,
                                     FeatureFPARMv8];
  list<SubtargetFeature> Olympus = [HasV9_2aOps, FeatureBRBE, FeatureCCIDX,
                                    FeatureCHK, FeatureETE, FeatureFAMINMAX,
                                    FeatureFP16FML, FeatureFP8DOT2,
                                    FeatureFP8DOT4, FeatureFP8FMA, FeatureFPAC,
                                    FeatureLS64, FeatureLUT, FeatureMEC,
                                    FeatureMTE, FeaturePerfMon, FeatureRandGen,
                                    FeatureSPE, FeatureSPE_EEF, FeatureSSBS,
                                    FeatureSVEBitPerm, FeatureSVESHA3,
                                    FeatureSVESM4, FeatureSVEAES, FeatureSVE2];
  list<SubtargetFeature> AppleA7  = [HasV8_0aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON,FeaturePerfMon];
  list<SubtargetFeature> AppleA10 = [HasV8_0aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureCRC,
                                     FeatureRDM, FeaturePAN, FeatureLOR, FeatureVH];
  list<SubtargetFeature> AppleA11 = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureFullFP16, FeatureCRC,
                                     FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> AppleA12 = [HasV8_3aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureFullFP16,
                                     FeatureComplxNum, FeatureCRC, FeatureJS, FeatureLSE,
                                     FeaturePAuth, FeatureRAS, FeatureRCPC, FeatureRDM];
  list<SubtargetFeature> AppleA13 = [HasV8_4aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureFullFP16,
                                     FeatureFP16FML, FeatureSHA3, FeatureComplxNum, FeatureCRC, FeatureJS, FeatureLSE,
                                     FeaturePAuth, FeatureRAS, FeatureRCPC, FeatureRDM, FeatureDotProd];
  list<SubtargetFeature> AppleA14 = [HasV8_4aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon,
                                     FeatureFullFP16, FeatureFP16FML, FeatureSHA3,
                                     // ArmV8.5-a extensions, excluding BTI:
                                     FeatureAltFPCmp, FeatureFRInt3264,
                                     FeatureSpecRestrict, FeatureSSBS, FeatureSB,
                                     FeaturePredRes, FeatureCacheDeepPersist,
                                     FeatureComplxNum, FeatureCRC, FeatureJS, FeatureLSE,
                                     FeaturePAuth, FeatureRAS, FeatureRCPC, FeatureRDM,
                                     FeatureDotProd];
  list<SubtargetFeature> AppleA15 = [HasV8_6aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureSHA3,
                                     FeatureFullFP16, FeatureFP16FML,
                                     FeatureComplxNum, FeatureCRC, FeatureJS,
                                     FeatureLSE, FeaturePAuth, FeatureFPAC,
                                     FeatureRAS, FeatureRCPC, FeatureRDM,
                                     FeatureBF16, FeatureDotProd, FeatureMatMulInt8, FeatureSSBS];
  list<SubtargetFeature> AppleA16 = [HasV8_6aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureSHA3,
                                     FeatureFullFP16, FeatureFP16FML,
                                     FeatureComplxNum, FeatureCRC, FeatureJS,
                                     FeatureLSE, FeaturePAuth, FeatureFPAC,
                                     FeatureRAS, FeatureRCPC, FeatureRDM,
                                     FeatureBF16, FeatureDotProd, FeatureMatMulInt8, FeatureSSBS];
  list<SubtargetFeature> AppleA17 = [HasV8_6aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureSHA3,
                                     FeatureFullFP16, FeatureFP16FML,
                                     FeatureComplxNum, FeatureCRC, FeatureJS,
                                     FeatureLSE, FeaturePAuth, FeatureFPAC,
                                     FeatureRAS, FeatureRCPC, FeatureRDM,
                                     FeatureBF16, FeatureDotProd, FeatureMatMulInt8, FeatureSSBS];
  // Technically apple-m4 is v9.2a, but we can't use that here.
  // Historically, llvm defined v9.0a as requiring SVE, but it's optional
  // according to the Arm ARM, and not supported by the core.  We decoupled the
  // two in the clang driver and in the backend subtarget features, but it's
  // still an issue in the clang frontend.  v8.7a is the next closest choice.
  list<SubtargetFeature> AppleM4 = [HasV8_7aOps, FeatureSHA2, FeatureFPARMv8,
                                    FeatureNEON, FeaturePerfMon, FeatureSHA3,
                                    FeatureFullFP16, FeatureFP16FML,
                                    FeatureAES, FeatureBF16,
                                    FeatureSME, FeatureSME2,
                                    FeatureSMEF64F64, FeatureSMEI16I64,
                                    FeatureComplxNum, FeatureCRC, FeatureJS,
                                    FeatureLSE, FeaturePAuth, FeatureFPAC,
                                    FeatureRAS, FeatureRCPC, FeatureRDM,
                                    FeatureDotProd, FeatureMatMulInt8];

  list<SubtargetFeature> AppleM5 = [HasV8_7aOps, FeatureSHA2, FeatureFPARMv8,
                                    FeatureNEON, FeaturePerfMon, FeatureSHA3,
                                    FeatureFullFP16, FeatureFP16FML,
                                    FeatureAES, FeatureBF16,
                                    FeatureWFxT,
                                    FeatureSME, FeatureSME2,
                                    FeatureSMEF64F64, FeatureSMEI16I64,
                                    FeatureComplxNum, FeatureCRC, FeatureJS,
                                    FeatureLSE, FeaturePAuth, FeatureFPAC,
                                    FeatureRAS, FeatureRCPC, FeatureRDM,
                                    FeatureDotProd, FeatureMatMulInt8,
                                    FeatureMTE, FeatureCSSC, FeatureHBC,
                                    FeatureSME2p1, FeatureSMEB16B16, FeatureSMEF16F16,
                                    FeatureSPECRES2];

  list<SubtargetFeature> ExynosM3 = [HasV8_0aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                     FeaturePerfMon, FeatureNEON, FeatureFPARMv8];
  list<SubtargetFeature> ExynosM4 = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureDotProd,
                                     FeatureFullFP16, FeaturePerfMon, FeatureCRC, FeatureFPARMv8,
                                     FeatureLSE, FeatureNEON, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> Falkor   = [HasV8_0aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                     FeatureFPARMv8, FeatureNEON, FeaturePerfMon,
                                     FeatureRDM];
  list<SubtargetFeature> NeoverseE1 = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureDotProd,
                                       FeatureFPARMv8, FeatureFullFP16, FeatureNEON,
                                       FeatureRCPC, FeatureSSBS, FeaturePerfMon, FeatureCRC,
                                       FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> NeoverseN1 = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureDotProd,
                                       FeatureFPARMv8, FeatureFullFP16, FeatureNEON,
                                       FeatureRCPC, FeatureSPE, FeatureSSBS,
                                       FeaturePerfMon, FeatureCRC, FeatureLSE, FeatureRAS, FeatureRDM];
  list<SubtargetFeature> NeoverseN2 = [HasV9_0aOps, FeatureBF16, FeatureETE, FeatureFP16FML,
                                       FeatureMatMulInt8, FeatureMTE, FeatureSVE2,
                                       FeatureSVEBitPerm, FeatureTRBE,
                                       FeaturePerfMon,
                                       FeatureCCIDX,
                                       FeatureDotProd, FeatureFullFP16, FeatureSB, FeatureSSBS, FeatureSVE,
                                       FeatureComplxNum, FeatureCRC, FeatureFPARMv8, FeatureJS, FeatureLSE,
                                       FeatureNEON, FeaturePAuth, FeatureRAS, FeatureRCPC, FeatureRDM, FeatureFPAC];
  list<SubtargetFeature> NeoverseN3 = [HasV9_2aOps, FeatureETE, FeatureFP16FML,
                                      FeatureFullFP16, FeatureMTE, FeaturePerfMon,
                                      FeatureRandGen, FeatureSPE, FeatureSPE_EEF,
                                      FeatureSVEBitPerm,
                                      FeatureCCIDX,
                                      FeatureSSBS, FeatureSB, FeaturePredRes, FeaturePAuth, FeatureFlagM,
                                      FeatureSVE, FeatureSVE2, FeatureBF16, FeatureComplxNum,
                                      FeatureCRC, FeatureDotProd, FeatureFPARMv8, FeatureMatMulInt8,
                                      FeatureJS, FeatureLSE, FeatureRAS, FeatureRCPC, FeatureRDM,
                                      FeatureNEON, FeatureFPAC];
  list<SubtargetFeature> Neoverse512TVB = [HasV8_4aOps, FeatureBF16, FeatureCacheDeepPersist,
                                           FeatureSHA2, FeatureAES, FeatureFPARMv8, FeatureFP16FML,
                                           FeatureFullFP16, FeatureMatMulInt8, FeatureNEON,
                                           FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                           FeatureSSBS, FeatureSVE,
                                           FeatureCCIDX,
                                           FeatureSHA3, FeatureSM4, FeatureDotProd, FeatureComplxNum,
                                           FeatureCRC, FeatureJS, FeatureLSE, FeaturePAuth, FeatureRAS,
                                           FeatureRCPC, FeatureRDM, FeatureFPAC];
  list<SubtargetFeature> NeoverseV1 = [HasV8_4aOps, FeatureBF16, FeatureCacheDeepPersist,
                                       FeatureSHA2, FeatureAES, FeatureFPARMv8, FeatureFP16FML,
                                       FeatureFullFP16, FeatureMatMulInt8, FeatureNEON,
                                       FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                       FeatureSSBS, FeatureSVE,
                                       FeatureCCIDX,
                                       FeatureSHA3, FeatureSM4, FeatureDotProd, FeatureComplxNum,
                                       FeatureCRC, FeatureJS, FeatureLSE, FeaturePAuth, FeatureRAS,
                                       FeatureRCPC, FeatureRDM];
  list<SubtargetFeature> NeoverseV2 = [HasV9_0aOps, FeatureBF16, FeatureSPE,
                                       FeaturePerfMon, FeatureETE, FeatureMatMulInt8,
                                       FeatureNEON, FeatureSVEBitPerm, FeatureFP16FML,
                                       FeatureMTE, FeatureRandGen,
                                       FeatureCCIDX,
                                       FeatureSVE, FeatureSVE2, FeatureSSBS, FeatureFullFP16, FeatureDotProd,
                                       FeatureComplxNum, FeatureCRC, FeatureFPARMv8, FeatureJS, FeatureLSE,
                                       FeaturePAuth, FeatureRAS, FeatureRCPC, FeatureRDM, FeatureFPAC];
  list<SubtargetFeature> NeoverseV3 = [HasV9_2aOps, FeatureETE, FeatureFP16FML,
                                      FeatureFullFP16, FeatureLS64, FeatureMTE,
                                      FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                      FeatureCCIDX,
                                      FeatureSPE_EEF, FeatureSVEBitPerm, FeatureBRBE,
                                      FeatureSSBS, FeatureSB, FeaturePredRes, FeaturePAuth, FeatureFlagM,
                                      FeatureSVE, FeatureSVE2, FeatureBF16, FeatureComplxNum, FeatureCRC,
                                      FeatureDotProd, FeatureFPARMv8, FeatureMatMulInt8, FeatureJS, FeatureLSE,
                                      FeatureNEON, FeatureRAS, FeatureRCPC, FeatureRDM, FeatureRME, FeatureFPAC];
  list<SubtargetFeature> NeoverseV3AE = [HasV9_2aOps, FeatureETE, FeatureFP16FML,
                                      FeatureFullFP16, FeatureLS64, FeatureMTE,
                                      FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                      FeatureSPE_EEF, FeatureSVEBitPerm, FeatureBRBE,
                                      FeatureSSBS, FeatureSB, FeaturePredRes, FeaturePAuth, FeatureFlagM,
                                      FeatureCCIDX,
                                      FeatureSVE, FeatureSVE2, FeatureBF16, FeatureComplxNum, FeatureCRC,
                                      FeatureDotProd, FeatureFPARMv8, FeatureMatMulInt8, FeatureJS,
                                      FeatureLSE, FeatureNEON, FeatureRAS, FeatureRCPC, FeatureRDM,
                                      FeatureRME, FeatureFPAC];
  list<SubtargetFeature> Saphira    = [HasV8_4aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                       FeatureNEON, FeatureSPE, FeaturePerfMon, FeatureCRC,
                                       FeatureCCIDX,
                                       FeatureLSE, FeatureRDM, FeatureRAS, FeatureRCPC];
  list<SubtargetFeature> ThunderX   = [HasV8_0aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                       FeatureFPARMv8, FeaturePerfMon, FeatureNEON];
  list<SubtargetFeature> ThunderX2T99  = [HasV8_1aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                          FeatureFPARMv8, FeatureNEON, FeatureLSE,
                                          FeatureRDM];
  list<SubtargetFeature> ThunderX3T110 = [HasV8_3aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                          FeatureFPARMv8, FeatureNEON, FeatureLSE,
                                          FeatureCCIDX,
                                          FeaturePAuth, FeaturePerfMon, FeatureComplxNum,
                                          FeatureJS, FeatureRAS, FeatureRCPC, FeatureRDM];
  list<SubtargetFeature> TSV110 = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                   FeatureNEON, FeaturePerfMon, FeatureSPE,
                                   FeatureFullFP16, FeatureFP16FML, FeatureDotProd,
                                   FeatureJS, FeatureComplxNum, FeatureCRC, FeatureLSE,
                                   FeatureRAS, FeatureRDM];
  list<SubtargetFeature> HIP12 = [HasV8_7aOps, FeatureSVE, FeatureSVE2,
                                  FeatureSVEBitPerm, FeatureSVEAES,
                                  FeatureSVESM4, FeatureSVESHA3,
                                  FeatureFullFP16, FeatureFP16FML, FeaturePerfMon,
                                  FeatureFlagM, FeaturePredRes, FeatureSB, FeatureSSBS,
                                  FeatureCCIDX, FeatureRandGen,
                                  FeatureETE, FeatureTRBE, FeatureSPE,
                                  FeatureSPE_EEF, FeatureNMI,
                                  FeatureHBC, FeatureRCPC3, FeatureBF16,
                                  FeatureComplxNum, FeatureCRC,
                                  FeatureDotProd, FeatureFPARMv8,
                                  FeatureMatMulInt8, FeatureJS,
                                  FeatureLSE, FeatureNEON, FeaturePAuth,
                                  FeatureRAS, FeatureRCPC, FeatureRDM,
                                  FeatureFPAC, FeatureLS64, FeatureRME,
                                  FeatureBRBE];
  list<SubtargetFeature> Ampere1 = [HasV8_6aOps, FeatureNEON, FeaturePerfMon,
                                    FeatureSSBS, FeatureRandGen, FeatureSB,
                                    FeatureSHA2, FeatureSHA3, FeatureAES,
                                    FeatureFullFP16, FeatureBF16, FeatureComplxNum, FeatureCRC,
                                    FeatureDotProd, FeatureFPARMv8, FeatureMatMulInt8, FeatureJS,
                                    FeatureCCIDX,
                                    FeatureLSE, FeaturePAuth, FeatureRAS, FeatureRCPC, FeatureRDM];
  list<SubtargetFeature> Ampere1A = [HasV8_6aOps, FeatureNEON, FeaturePerfMon,
                                     FeatureMTE, FeatureSSBS, FeatureRandGen,
                                     FeatureSB, FeatureSM4, FeatureSHA2,
                                     FeatureSHA3, FeatureAES,
                                     FeatureFullFP16, FeatureBF16, FeatureComplxNum,
                                     FeatureCRC, FeatureDotProd, FeatureFPARMv8, FeatureMatMulInt8,
                                     FeatureJS, FeatureLSE, FeaturePAuth, FeatureRAS, FeatureRCPC,
                                     FeatureCCIDX,
                                     FeatureRDM];
  list<SubtargetFeature> Ampere1B = [HasV8_7aOps, FeatureNEON, FeaturePerfMon,
                                     FeatureMTE, FeatureSSBS, FeatureRandGen,
                                     FeatureSB, FeatureSM4, FeatureSHA2,
                                     FeatureSHA3, FeatureAES, FeatureCSSC,
                                     FeatureWFxT, FeatureFullFP16, FeatureBF16, FeatureComplxNum,
                                     FeatureCRC, FeatureDotProd, FeatureFPARMv8, FeatureMatMulInt8,
                                     FeatureJS, FeatureLSE, FeaturePAuth, FeatureRAS, FeatureRCPC,
                                     FeatureCCIDX,
                                     FeatureRDM];
  list<SubtargetFeature> Ampere1C = [HasV9_2aOps, FeatureNEON, FeatureCSSC, FeatureCCIDX,
                                     FeatureFP8FMA, FeatureFAMINMAX, FeatureLUT,
                                     FeatureSVEAES, FeatureSVESM4, FeatureSVESHA3,
                                     FeatureSVE2, FeatureSVEB16B16, FeatureMTE,
                                     FeatureFP16FML, FeatureSSBS, FeatureRandGen,
                                     FeaturePerfMon];
  list<SubtargetFeature> Oryon = [HasV8_6aOps, FeatureNEON, FeaturePerfMon,
                                     FeatureRandGen,
                                     FeaturePAuth, FeatureSM4, FeatureSHA2,
                                     FeatureSHA3, FeatureAES,
                                     FeatureSPE, FeatureBF16, FeatureComplxNum, FeatureCRC,
                                     FeatureDotProd, FeatureFPARMv8, FeatureMatMulInt8,
                                     FeatureSSBS, FeatureCCIDX,
                                     FeatureJS, FeatureLSE, FeatureRAS, FeatureRCPC, FeatureRDM];
  list<SubtargetFeature> GB10 = !listconcat(X925, [FeatureSVEAES, FeatureSVESHA3, FeatureSVESM4]);
  list<SubtargetFeature> Grace = !listconcat(!listremove(NeoverseV2, [FeatureRandGen]), [FeatureSVESM4, FeatureSVEAES, FeatureSVESHA3]);

  // ETE and TRBE are future architecture extensions. We temporarily enable them
  // by default for users targeting generic AArch64. The extensions do not
  // affect code generated by the compiler and can be used only by explicitly
  // mentioning the new system register names in assembly.
  list<SubtargetFeature> Generic = [FeatureFPARMv8, FeatureNEON, FeatureETE];
}

// A note on scheduling models - we do not have exact models for each core and
// attempt to use the "best equivalent" that we have for those that are
// missing. -mcpu=generic uses a cortex-a510 model to get good scheduling
// performance for in-order cores where it is most beneficial. Out-of-order
// cores are expected to be less sensitive and scheduling from an in-order
// model will be sufficient.

// FeatureFuseAdrpAdd is enabled under Generic to allow linker merging
// optimizations. We use a fixed load latency of 4 as the a510 scheduling model
// has a latency of 2 for most loads which is disproportionately low for many
// cpus.
def : ProcessorModel<"generic", CortexA510Model, ProcessorFeatures.Generic,
                     [FeatureFuseAES, FeatureFuseAdrpAdd, FeaturePostRAScheduler,
                      FeatureEnableSelectOptimize, FeatureFixedLoadLatency4]>;
def : ProcessorModel<"cortex-a34", CortexA53Model, ProcessorFeatures.A53,
                     [TuneA35]>;
def : ProcessorModel<"cortex-a35", CortexA53Model, ProcessorFeatures.A53,
                     [TuneA35]>;
def : ProcessorModel<"cortex-a320", CortexA320Model, ProcessorFeatures.A320,
                     [TuneA320]>;
def : ProcessorModel<"cortex-a53", CortexA53Model, ProcessorFeatures.A53,
                     [TuneA53]>;
def : ProcessorModel<"cortex-a55", CortexA55Model, ProcessorFeatures.A55,
                     [TuneA55]>;
def : ProcessorModel<"cortex-a510", CortexA510Model, ProcessorFeatures.A510,
                     [TuneA510]>;
def : ProcessorModel<"cortex-a520", CortexA510Model, ProcessorFeatures.A520,
                     [TuneA520]>;
def : ProcessorModel<"cortex-a520ae", CortexA510Model, ProcessorFeatures.A520AE,
                     [TuneA520AE]>;
def : ProcessorModel<"c1-nano", C1NanoModel,
                     ProcessorFeatures.C1Nano, [TuneC1Nano]>;
def : ProcessorModel<"cortex-a57", CortexA57Model, ProcessorFeatures.A53,
                     [TuneA57]>;
def : ProcessorModel<"cortex-a65", CortexA53Model, ProcessorFeatures.A65,
                     [TuneA65]>;
def : ProcessorModel<"cortex-a65ae", CortexA53Model, ProcessorFeatures.A65,
                     [TuneA65]>;
def : ProcessorModel<"cortex-a72", CortexA57Model, ProcessorFeatures.A53,
                     [TuneA72]>;
def : ProcessorModel<"cortex-a73", CortexA57Model, ProcessorFeatures.A53,
                     [TuneA73]>;
def : ProcessorModel<"cortex-a75", CortexA57Model, ProcessorFeatures.A55,
                     [TuneA75]>;
def : ProcessorModel<"cortex-a76", NeoverseN1Model, ProcessorFeatures.A76,
                     [TuneA76]>;
def : ProcessorModel<"cortex-a76ae", NeoverseN1Model, ProcessorFeatures.A76,
                     [TuneA76]>;
def : ProcessorModel<"cortex-a77", NeoverseN2Model, ProcessorFeatures.A77,
                     [TuneA77]>;
def : ProcessorModel<"cortex-a78", NeoverseN2Model, ProcessorFeatures.A78,
                     [TuneA78]>;
def : ProcessorModel<"cortex-a78ae", NeoverseN2Model, ProcessorFeatures.A78AE,
                     [TuneA78AE]>;
def : ProcessorModel<"cortex-a78c", NeoverseN2Model, ProcessorFeatures.A78C,
                     [TuneA78C]>;
def : ProcessorModel<"cortex-a710", NeoverseN2Model, ProcessorFeatures.A710,
                     [TuneA710]>;
def : ProcessorModel<"cortex-a715", NeoverseN2Model, ProcessorFeatures.A715,
                     [TuneA715]>;
def : ProcessorModel<"cortex-a720", NeoverseN2Model, ProcessorFeatures.A720,
                     [TuneA720]>;
def : ProcessorModel<"cortex-a720ae", NeoverseN2Model, ProcessorFeatures.A720AE,
                     [TuneA720AE]>;
def : ProcessorModel<"cortex-a725", NeoverseN3Model, ProcessorFeatures.A725,
                     [TuneA725]>;
def : ProcessorModel<"c1-pro", NeoverseN3Model,
                     ProcessorFeatures.C1Pro, [TuneC1Pro]>;
def : ProcessorModel<"cortex-r82", CortexA55Model, ProcessorFeatures.R82,
                     [TuneR82]>;
def : ProcessorModel<"cortex-r82ae", CortexA55Model, ProcessorFeatures.R82AE,
                     [TuneR82AE]>;
def : ProcessorModel<"cortex-x1", NeoverseV1Model, ProcessorFeatures.X1,
                     [TuneX1]>;
def : ProcessorModel<"cortex-x1c", NeoverseV1Model, ProcessorFeatures.X1C,
                     [TuneX1]>;
def : ProcessorModel<"cortex-x2", NeoverseV2Model, ProcessorFeatures.X2,
                     [TuneX2]>;
def : ProcessorModel<"cortex-x3", NeoverseV2Model, ProcessorFeatures.X3,
                     [TuneX3]>;
def : ProcessorModel<"cortex-x4", NeoverseV3Model, ProcessorFeatures.X4,
                     [TuneX4]>;
def : ProcessorModel<"cortex-x925", NeoverseV3Model, ProcessorFeatures.X925,
                     [TuneX925]>;
def : ProcessorModel<"c1-premium", C1PremiumModel,
                     ProcessorFeatures.C1Premium, [TuneC1Premium]>;
def : ProcessorModel<"c1-ultra", C1UltraModel,
                     ProcessorFeatures.C1Ultra, [TuneC1Ultra]>;
def : ProcessorModel<"gb10", NeoverseV3Model, ProcessorFeatures.GB10,
                     [TuneX925]>;
def : ProcessorModel<"grace", NeoverseV2Model, ProcessorFeatures.Grace,
                     [TuneNeoverseV2]>;
def : ProcessorModel<"neoverse-e1", CortexA53Model,
                     ProcessorFeatures.NeoverseE1, [TuneNeoverseE1]>;
def : ProcessorModel<"neoverse-n1", NeoverseN1Model,
                     ProcessorFeatures.NeoverseN1, [TuneNeoverseN1]>;
def : ProcessorModel<"neoverse-n2", NeoverseN2Model,
                     ProcessorFeatures.NeoverseN2, [TuneNeoverseN2]>;
def : ProcessorAlias<"cobalt-100", "neoverse-n2">;
def : ProcessorModel<"neoverse-n3", NeoverseN3Model,
                     ProcessorFeatures.NeoverseN3, [TuneNeoverseN3]>;
def : ProcessorModel<"neoverse-512tvb", NeoverseV1Model,
                     ProcessorFeatures.Neoverse512TVB, [TuneNeoverse512TVB]>;
def : ProcessorModel<"neoverse-v1", NeoverseV1Model,
                     ProcessorFeatures.NeoverseV1, [TuneNeoverseV1]>;
def : ProcessorModel<"neoverse-v2", NeoverseV2Model,
                     ProcessorFeatures.NeoverseV2, [TuneNeoverseV2]>;
def : ProcessorModel<"neoverse-v3", NeoverseV3Model,
                     ProcessorFeatures.NeoverseV3, [TuneNeoverseV3]>;
def : ProcessorModel<"neoverse-v3ae", NeoverseV3AEModel,
                     ProcessorFeatures.NeoverseV3AE, [TuneNeoverseV3AE]>;
def : ProcessorAlias<"armagicpu", "neoverse-v3ae">;
def : ProcessorModel<"exynos-m3", ExynosM3Model, ProcessorFeatures.ExynosM3,
                     [TuneExynosM3]>;
def : ProcessorModel<"exynos-m4", ExynosM4Model, ProcessorFeatures.ExynosM4,
                     [TuneExynosM4]>;
def : ProcessorModel<"exynos-m5", ExynosM5Model, ProcessorFeatures.ExynosM4,
                     [TuneExynosM4]>;
def : ProcessorModel<"falkor", FalkorModel, ProcessorFeatures.Falkor,
                     [TuneFalkor]>;
def : ProcessorModel<"saphira", FalkorModel, ProcessorFeatures.Saphira,
                     [TuneSaphira]>;
def : ProcessorModel<"kryo", KryoModel, ProcessorFeatures.A53, [TuneKryo]>;

// Cavium ThunderX/ThunderX T8X  Processors
def : ProcessorModel<"thunderx", ThunderXT8XModel,  ProcessorFeatures.ThunderX,
                     [TuneThunderX]>;
def : ProcessorModel<"thunderxt88", ThunderXT8XModel,
                     ProcessorFeatures.ThunderX, [TuneThunderXT88]>;
def : ProcessorModel<"thunderxt81", ThunderXT8XModel,
                     ProcessorFeatures.ThunderX, [TuneThunderXT81]>;
def : ProcessorModel<"thunderxt83", ThunderXT8XModel,
                     ProcessorFeatures.ThunderX, [TuneThunderXT83]>;
// Cavium ThunderX2T9X  Processors. Formerly Broadcom Vulcan.
def : ProcessorModel<"thunderx2t99", ThunderX2T99Model,
                     ProcessorFeatures.ThunderX2T99, [TuneThunderX2T99]>;
// Marvell ThunderX3T110 Processors.
def : ProcessorModel<"thunderx3t110", ThunderX3T110Model,
                     ProcessorFeatures.ThunderX3T110, [TuneThunderX3T110]>;

// HiSilicon Processors.
def : ProcessorModel<"tsv110", TSV110Model, ProcessorFeatures.TSV110,
                     [TuneTSV110]>;
def : ProcessorModel<"hip12", HIP12Model, ProcessorFeatures.HIP12,
                     [TuneHIP12]>;

// Apple CPUs

def : ProcessorModel<"apple-a7", CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7]>;
// Support cyclone as an alias for apple-a7 so we can still LTO old bitcode.
def : ProcessorAlias<"cyclone", "apple-a7">;
def : ProcessorAlias<"apple-a8", "apple-a7">;
def : ProcessorAlias<"apple-a9", "apple-a7">;

def : ProcessorModel<"apple-a10", CycloneModel, ProcessorFeatures.AppleA10,
                     [TuneAppleA10]>;

def : ProcessorModel<"apple-a11", CycloneModel, ProcessorFeatures.AppleA11,
                     [TuneAppleA11]>;

def : ProcessorModel<"apple-a12", CycloneModel, ProcessorFeatures.AppleA12,
                     [TuneAppleA12]>;
def : ProcessorAlias<"apple-s4", "apple-a12">;
def : ProcessorAlias<"apple-s5", "apple-a12">;

def : ProcessorModel<"apple-a13", CycloneModel, ProcessorFeatures.AppleA13,
                     [TuneAppleA13]>;
def : ProcessorAlias<"apple-s6", "apple-a13">;
def : ProcessorAlias<"apple-s7", "apple-a13">;
def : ProcessorAlias<"apple-s8", "apple-a13">;

def : ProcessorModel<"apple-a14", CycloneModel, ProcessorFeatures.AppleA14,
                     [TuneAppleA14]>;
def : ProcessorAlias<"apple-m1", "apple-a14">;

def : ProcessorModel<"apple-a15", CycloneModel, ProcessorFeatures.AppleA15,
                     [TuneAppleA15]>;
def : ProcessorAlias<"apple-m2", "apple-a15">;

def : ProcessorModel<"apple-a16", CycloneModel, ProcessorFeatures.AppleA16,
                     [TuneAppleA16]>;
def : ProcessorAlias<"apple-m3", "apple-a16">;
def : ProcessorAlias<"apple-s9", "apple-a16">;
def : ProcessorAlias<"apple-s10", "apple-a16">;

def : ProcessorModel<"apple-a17", CycloneModel, ProcessorFeatures.AppleA17,
                     [TuneAppleA17]>;

def : ProcessorModel<"apple-m4", CycloneModel, ProcessorFeatures.AppleM4,
                     [TuneAppleM4]>;
def : ProcessorAlias<"apple-a18", "apple-m4">;

def : ProcessorModel<"apple-m5", CycloneModel, ProcessorFeatures.AppleM5,
                     [TuneAppleM5]>;
def : ProcessorAlias<"apple-a19", "apple-m5">;

// Alias for the latest Apple processor model supported by LLVM.
def : ProcessorAlias<"apple-latest", "apple-m5">;


// Fujitsu A64FX
def : ProcessorModel<"a64fx", A64FXModel, ProcessorFeatures.A64FX,
                     [TuneA64FX]>;

// Fujitsu FUJITSU-MONAKA
def : ProcessorModel<"fujitsu-monaka", A64FXModel, ProcessorFeatures.MONAKA,
                     [TuneMONAKA]>;

// Nvidia Carmel
def : ProcessorModel<"carmel", NoSchedModel, ProcessorFeatures.Carmel,
                     [TuneCarmel]>;

// NVIDIA Olympus
def : ProcessorModel<"olympus", OlympusModel, ProcessorFeatures.Olympus,
                     [TuneOlympus]>;

// NVIDIA Rigel
def : ProcessorModel<"rigel", OlympusModel, ProcessorFeatures.Olympus,
                     [TuneOlympus]>;

// Ampere Computing
def : ProcessorModel<"ampere1", Ampere1Model, ProcessorFeatures.Ampere1,
                     [TuneAmpere1]>;

def : ProcessorModel<"ampere1a", Ampere1Model, ProcessorFeatures.Ampere1A,
                     [TuneAmpere1A]>;

def : ProcessorModel<"ampere1b", Ampere1BModel, ProcessorFeatures.Ampere1B,
                     [TuneAmpere1B]>;

def : ProcessorModel<"ampere1c", NeoverseN3Model, ProcessorFeatures.Ampere1C,
                     [TuneAmpere1C]>;

// Qualcomm Oryon
def : ProcessorModel<"oryon-1", OryonModel, ProcessorFeatures.Oryon,
                       [TuneOryon]>;
