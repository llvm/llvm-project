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


def TuneA35     : SubtargetFeature<"a35", "ARMProcFamily", "CortexA35",
                                "Cortex-A35 ARM processors">;

def TuneA53     : SubtargetFeature<"a53", "ARMProcFamily", "CortexA53",
                                   "Cortex-A53 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureBalanceFPOps,
                                   FeaturePostRAScheduler]>;

def TuneA55     : SubtargetFeature<"a55", "ARMProcFamily", "CortexA55",
                                   "Cortex-A55 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeaturePostRAScheduler,
                                   FeatureFuseAddress]>;

def TuneA510    : SubtargetFeature<"a510", "ARMProcFamily", "CortexA510",
                                   "Cortex-A510 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeaturePostRAScheduler
                                   ]>;

def TuneA520    : SubtargetFeature<"a520", "ARMProcFamily", "CortexA520",
                                   "Cortex-A520 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeaturePostRAScheduler]>;

def TuneA520AE  : SubtargetFeature<"a520ae", "ARMProcFamily", "CortexA520",
                                   "Cortex-A520AE ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeaturePostRAScheduler]>;

def TuneA57     : SubtargetFeature<"a57", "ARMProcFamily", "CortexA57",
                                   "Cortex-A57 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureBalanceFPOps,
                                   FeatureFuseAdrpAdd,
                                   FeatureFuseLiterals,
                                   FeatureAddrLSLSlow14,
                                   FeaturePostRAScheduler,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA65     : SubtargetFeature<"a65", "ARMProcFamily", "CortexA65",
                                   "Cortex-A65 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAddress,
                                   FeatureFuseAdrpAdd,
                                   FeatureFuseLiterals,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA72     : SubtargetFeature<"a72", "ARMProcFamily", "CortexA72",
                                   "Cortex-A72 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureFuseLiterals,
                                   FeatureAddrLSLSlow14,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA73     : SubtargetFeature<"a73", "ARMProcFamily", "CortexA73",
                                   "Cortex-A73 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureAddrLSLSlow14,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA75     : SubtargetFeature<"a75", "ARMProcFamily", "CortexA75",
                                   "Cortex-A75 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureAddrLSLSlow14,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA76     : SubtargetFeature<"a76", "ARMProcFamily", "CortexA76",
                                   "Cortex-A76 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureAddrLSLSlow14,
                                   FeatureALULSLFast,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA77     : SubtargetFeature<"a77", "ARMProcFamily", "CortexA77",
                                   "Cortex-A77 ARM processors", [
                                   FeatureCmpBccFusion,
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureAddrLSLSlow14,
                                   FeatureALULSLFast,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA78 : SubtargetFeature<"a78", "ARMProcFamily", "CortexA78",
                               "Cortex-A78 ARM processors", [
                               FeatureCmpBccFusion,
                               FeatureFuseAES,
                               FeatureFuseAdrpAdd,
                               FeatureAddrLSLSlow14,
                               FeatureALULSLFast,
                               FeaturePostRAScheduler,
                               FeatureEnableSelectOptimize,
                               FeaturePredictableSelectIsExpensive]>;

def TuneA78AE : SubtargetFeature<"a78ae", "ARMProcFamily",
                                 "CortexA78AE",
                                 "Cortex-A78AE ARM processors", [
                                 FeatureCmpBccFusion,
                                 FeatureFuseAES,
                                 FeatureFuseAdrpAdd,
                                 FeatureAddrLSLSlow14,
                                 FeatureALULSLFast,
                                 FeaturePostRAScheduler,
                                 FeatureEnableSelectOptimize,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneA78C : SubtargetFeature<"a78c", "ARMProcFamily",
                                "CortexA78C",
                                "Cortex-A78C ARM processors", [
                                FeatureCmpBccFusion,
                                FeatureFuseAES,
                                FeatureFuseAdrpAdd,
                                FeatureAddrLSLSlow14,
                                FeatureALULSLFast,
                                FeaturePostRAScheduler,
                                FeatureEnableSelectOptimize,
                                FeaturePredictableSelectIsExpensive]>;

def TuneA710    : SubtargetFeature<"a710", "ARMProcFamily", "CortexA710",
                                   "Cortex-A710 ARM processors", [
                                   FeatureCmpBccFusion,
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureALULSLFast,
                                   FeaturePostRAScheduler,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA715 : SubtargetFeature<"a715", "ARMProcFamily", "CortexA715",
                                 "Cortex-A715 ARM processors", [
                                 FeatureFuseAES,
                                 FeaturePostRAScheduler,
                                 FeatureCmpBccFusion,
                                 FeatureALULSLFast,
                                 FeatureFuseAdrpAdd,
                                 FeatureEnableSelectOptimize,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneA720 : SubtargetFeature<"a720", "ARMProcFamily", "CortexA720",
                                "Cortex-A720 ARM processors", [
                                 FeatureFuseAES,
                                 FeaturePostRAScheduler,
                                 FeatureCmpBccFusion,
                                 FeatureALULSLFast,
                                 FeatureFuseAdrpAdd,
                                 FeatureEnableSelectOptimize,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneA720AE : SubtargetFeature<"a720ae", "ARMProcFamily", "CortexA720",
                                "Cortex-A720AE ARM processors", [
                                 FeatureFuseAES,
                                 FeaturePostRAScheduler,
                                 FeatureCmpBccFusion,
                                 FeatureALULSLFast,
                                 FeatureFuseAdrpAdd,
                                 FeatureEnableSelectOptimize,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneR82 : SubtargetFeature<"cortex-r82", "ARMProcFamily",
                               "CortexR82",
                               "Cortex-R82 ARM processors", [
                               FeaturePostRAScheduler]>;

def TuneR82AE : SubtargetFeature<"cortex-r82ae", "ARMProcFamily",
                                 "CortexR82AE",
                                 "Cortex-R82-AE ARM processors",
                                 [FeaturePostRAScheduler]>;

def TuneX1 : SubtargetFeature<"cortex-x1", "ARMProcFamily", "CortexX1",
                                  "Cortex-X1 ARM processors", [
                                  FeatureCmpBccFusion,
                                  FeatureFuseAES,
                                  FeatureFuseAdrpAdd,
                                  FeatureAddrLSLSlow14,
                                  FeatureALULSLFast,
                                  FeaturePostRAScheduler,
                                  FeatureEnableSelectOptimize,
                                  FeaturePredictableSelectIsExpensive]>;

def TuneX2 : SubtargetFeature<"cortex-x2", "ARMProcFamily", "CortexX2",
                                  "Cortex-X2 ARM processors", [
                                  FeatureCmpBccFusion,
                                  FeatureFuseAES,
                                  FeatureFuseAdrpAdd,
                                  FeatureALULSLFast,
                                  FeaturePostRAScheduler,
                                  FeatureEnableSelectOptimize,
                                  FeaturePredictableSelectIsExpensive]>;

def TuneX3 : SubtargetFeature<"cortex-x3", "ARMProcFamily", "CortexX3",
                              "Cortex-X3 ARM processors", [
                               FeatureALULSLFast,
                               FeatureFuseAdrpAdd,
                               FeatureFuseAES,
                               FeaturePostRAScheduler,
                               FeatureEnableSelectOptimize,
                               FeaturePredictableSelectIsExpensive]>;

def TuneX4 : SubtargetFeature<"cortex-x4", "ARMProcFamily", "CortexX4",
                              "Cortex-X4 ARM processors", [
                               FeatureALULSLFast,
                               FeatureFuseAdrpAdd,
                               FeatureFuseAES,
                               FeaturePostRAScheduler,
                               FeatureEnableSelectOptimize,
                               FeaturePredictableSelectIsExpensive]>;

def TuneA64FX : SubtargetFeature<"a64fx", "ARMProcFamily", "A64FX",
                                 "Fujitsu A64FX processors", [
                                 FeaturePostRAScheduler,
                                 FeatureAggressiveFMA,
                                 FeatureArithmeticBccFusion,
                                 FeatureStorePairSuppress,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneCarmel : SubtargetFeature<"carmel", "ARMProcFamily", "Carmel",
                                  "Nvidia Carmel processors">;

// Note that cyclone does not fuse AES instructions, but newer apple chips do
// perform the fusion and cyclone is used by default when targetting apple OSes.
def TuneAppleA7  : SubtargetFeature<"apple-a7", "ARMProcFamily", "AppleA7",
                                    "Apple A7 (the CPU formerly known as Cyclone)", [
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFuseAES, FeatureFuseCryptoEOR,
                                    FeatureStorePairSuppress,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing,
                                    FeatureZCZeroingFPWorkaround]>;

def TuneAppleA10 : SubtargetFeature<"apple-a10", "ARMProcFamily", "AppleA10",
                                    "Apple A10", [
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFuseAES,
                                    FeatureFuseCryptoEOR,
                                    FeatureStorePairSuppress,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing]>;

def TuneAppleA11 : SubtargetFeature<"apple-a11", "ARMProcFamily", "AppleA11",
                                    "Apple A11", [
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFuseAES,
                                    FeatureFuseCryptoEOR,
                                    FeatureStorePairSuppress,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing]>;

def TuneAppleA12 : SubtargetFeature<"apple-a12", "ARMProcFamily", "AppleA12",
                                    "Apple A12", [
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFuseAES,
                                    FeatureFuseCryptoEOR,
                                    FeatureStorePairSuppress,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing]>;

def TuneAppleA13 : SubtargetFeature<"apple-a13", "ARMProcFamily", "AppleA13",
                                    "Apple A13", [
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFuseAES,
                                    FeatureFuseCryptoEOR,
                                    FeatureStorePairSuppress,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing]>;

def TuneAppleA14 : SubtargetFeature<"apple-a14", "ARMProcFamily", "AppleA14",
                                    "Apple A14", [
                                    FeatureAggressiveFMA,
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFuseAddress,
                                    FeatureFuseAES,
                                    FeatureFuseArithmeticLogic,
                                    FeatureFuseCCSelect,
                                    FeatureFuseCryptoEOR,
                                    FeatureFuseLiterals,
                                    FeatureStorePairSuppress,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing]>;

def TuneAppleA15 : SubtargetFeature<"apple-a15", "ARMProcFamily", "AppleA15",
                                    "Apple A15", [
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFuseAddress,
                                    FeatureFuseAES,
                                    FeatureFuseArithmeticLogic,
                                    FeatureFuseCCSelect,
                                    FeatureFuseCryptoEOR,
                                    FeatureFuseLiterals,
                                    FeatureStorePairSuppress,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing]>;

def TuneAppleA16 : SubtargetFeature<"apple-a16", "ARMProcFamily", "AppleA16",
                                    "Apple A16", [
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFuseAddress,
                                    FeatureFuseAdrpAdd,
                                    FeatureFuseAES,
                                    FeatureFuseArithmeticLogic,
                                    FeatureFuseCCSelect,
                                    FeatureFuseCryptoEOR,
                                    FeatureFuseLiterals,
                                    FeatureStorePairSuppress,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing]>;

def TuneAppleA17 : SubtargetFeature<"apple-a17", "ARMProcFamily", "AppleA17",
                                    "Apple A17", [
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFuseAddress,
                                    FeatureFuseAdrpAdd,
                                    FeatureFuseAES,
                                    FeatureFuseArithmeticLogic,
                                    FeatureFuseCCSelect,
                                    FeatureFuseCryptoEOR,
                                    FeatureFuseLiterals,
                                    FeatureStorePairSuppress,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing]>;

def TuneExynosM3 : SubtargetFeature<"exynosm3", "ARMProcFamily", "ExynosM3",
                                    "Samsung Exynos-M3 processors",
                                    [FeatureExynosCheapAsMoveHandling,
                                     FeatureForce32BitJumpTables,
                                     FeatureFuseAddress,
                                     FeatureFuseAES,
                                     FeatureFuseCCSelect,
                                     FeatureFuseAdrpAdd,
                                     FeatureFuseLiterals,
                                     FeatureStorePairSuppress,
                                     FeatureALULSLFast,
                                     FeaturePostRAScheduler,
                                     FeaturePredictableSelectIsExpensive]>;

// Re-uses some scheduling and tunings from the ExynosM3 proc family.
def TuneExynosM4 : SubtargetFeature<"exynosm4", "ARMProcFamily", "ExynosM3",
                                    "Samsung Exynos-M4 processors",
                                    [FeatureArithmeticBccFusion,
                                     FeatureArithmeticCbzFusion,
                                     FeatureExynosCheapAsMoveHandling,
                                     FeatureForce32BitJumpTables,
                                     FeatureFuseAddress,
                                     FeatureFuseAES,
                                     FeatureFuseArithmeticLogic,
                                     FeatureFuseCCSelect,
                                     FeatureFuseAdrpAdd,
                                     FeatureFuseLiterals,
                                     FeatureStorePairSuppress,
                                     FeatureALULSLFast,
                                     FeaturePostRAScheduler,
                                     FeatureZCZeroing]>;

def TuneKryo    : SubtargetFeature<"kryo", "ARMProcFamily", "Kryo",
                                   "Qualcomm Kryo processors", [
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroing,
                                   FeatureALULSLFast,
                                   FeatureStorePairSuppress]>;

def TuneFalkor  : SubtargetFeature<"falkor", "ARMProcFamily", "Falkor",
                                   "Qualcomm Falkor processors", [
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroing,
                                   FeatureStorePairSuppress,
                                   FeatureALULSLFast,
                                   FeatureSlowSTRQro]>;

def TuneNeoverseE1 : SubtargetFeature<"neoversee1", "ARMProcFamily", "NeoverseE1",
                                      "Neoverse E1 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeaturePostRAScheduler]>;

def TuneNeoverseN1 : SubtargetFeature<"neoversen1", "ARMProcFamily", "NeoverseN1",
                                      "Neoverse N1 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureAddrLSLSlow14,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneNeoverseN2 : SubtargetFeature<"neoversen2", "ARMProcFamily", "NeoverseN2",
                                      "Neoverse N2 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneNeoverseN3 : SubtargetFeature<"neoversen3", "ARMProcFamily", "NeoverseN3",
                                      "Neoverse N3 ARM processors", [
                                      FeatureFuseAES,
                                      FeaturePostRAScheduler,
                                      FeatureALULSLFast,
                                      FeatureFuseAdrpAdd,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneNeoverse512TVB : SubtargetFeature<"neoverse512tvb", "ARMProcFamily", "Neoverse512TVB",
                                      "Neoverse 512-TVB ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneNeoverseV1 : SubtargetFeature<"neoversev1", "ARMProcFamily", "NeoverseV1",
                                      "Neoverse V1 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureAddrLSLSlow14,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive,
                                      FeatureNoSVEFPLD1R]>;

def TuneNeoverseV2 : SubtargetFeature<"neoversev2", "ARMProcFamily", "NeoverseV2",
                                      "Neoverse V2 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneNeoverseV3 : SubtargetFeature<"neoversev3", "ARMProcFamily", "NeoverseV3",
                                      "Neoverse V3 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureALULSLFast,
                                      FeatureFuseAdrpAdd,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneNeoverseV3AE : SubtargetFeature<"neoversev3AE", "ARMProcFamily", "NeoverseV3",
                                      "Neoverse V3AE ARM processors", [
                                      FeatureFuseAES,
                                      FeatureALULSLFast,
                                      FeatureFuseAdrpAdd,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneSaphira  : SubtargetFeature<"saphira", "ARMProcFamily", "Saphira",
                                   "Qualcomm Saphira processors", [
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroing,
                                   FeatureStorePairSuppress,
                                   FeatureALULSLFast]>;

def TuneThunderX2T99  : SubtargetFeature<"thunderx2t99", "ARMProcFamily", "ThunderX2T99",
                                         "Cavium ThunderX2 processors", [
                                          FeatureAggressiveFMA,
                                          FeatureArithmeticBccFusion,
                                          FeaturePostRAScheduler,
                                          FeatureStorePairSuppress,
                                          FeaturePredictableSelectIsExpensive]>;

def TuneThunderX3T110  : SubtargetFeature<"thunderx3t110", "ARMProcFamily",
                                          "ThunderX3T110",
                                          "Marvell ThunderX3 processors", [
                                           FeatureAggressiveFMA,
                                           FeatureArithmeticBccFusion,
                                           FeaturePostRAScheduler,
                                           FeaturePredictableSelectIsExpensive,
                                           FeatureBalanceFPOps,
                                           FeatureStorePairSuppress,
                                           FeatureStrictAlign]>;

def TuneThunderX : SubtargetFeature<"thunderx", "ARMProcFamily", "ThunderX",
                                    "Cavium ThunderX processors", [
                                    FeaturePostRAScheduler,
                                    FeatureStorePairSuppress,
                                    FeaturePredictableSelectIsExpensive]>;

def TuneThunderXT88 : SubtargetFeature<"thunderxt88", "ARMProcFamily",
                                       "ThunderXT88",
                                       "Cavium ThunderX processors", [
                                       FeaturePostRAScheduler,
                                       FeatureStorePairSuppress,
                                       FeaturePredictableSelectIsExpensive]>;

def TuneThunderXT81 : SubtargetFeature<"thunderxt81", "ARMProcFamily",
                                       "ThunderXT81",
                                       "Cavium ThunderX processors", [
                                       FeaturePostRAScheduler,
                                       FeatureStorePairSuppress,
                                       FeaturePredictableSelectIsExpensive]>;

def TuneThunderXT83 : SubtargetFeature<"thunderxt83", "ARMProcFamily",
                                       "ThunderXT83",
                                       "Cavium ThunderX processors", [
                                       FeaturePostRAScheduler,
                                       FeatureStorePairSuppress,
                                       FeaturePredictableSelectIsExpensive]>;

def TuneTSV110 : SubtargetFeature<"tsv110", "ARMProcFamily", "TSV110",
                                  "HiSilicon TS-V110 processors", [
                                  FeatureFuseAES,
                                  FeatureStorePairSuppress,
                                  FeaturePostRAScheduler]>;

def TuneAmpere1 : SubtargetFeature<"ampere1", "ARMProcFamily", "Ampere1",
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
                                   FeatureStpAlignedOnly]>;

def TuneAmpere1A : SubtargetFeature<"ampere1a", "ARMProcFamily", "Ampere1A",
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
                                    FeatureStpAlignedOnly]>;

def TuneAmpere1B : SubtargetFeature<"ampere1b", "ARMProcFamily", "Ampere1B",
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
                                    FeatureStpAlignedOnly]>;


def ProcessorFeatures {
  list<SubtargetFeature> A53  = [HasV8_0aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                 FeatureFPARMv8, FeatureNEON, FeaturePerfMon];
  list<SubtargetFeature> A55  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeaturePerfMon];
  list<SubtargetFeature> A510 = [HasV9_0aOps, FeatureNEON, FeaturePerfMon,
                                 FeatureMatMulInt8, FeatureBF16, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVE2BitPerm,
                                 FeatureFP16FML,
                                 FeatureSB, FeaturePAuth, FeatureSSBS, FeatureSVE, FeatureSVE2];
  list<SubtargetFeature> A520 = [HasV9_2aOps, FeaturePerfMon, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVE2BitPerm,
                                 FeatureFP16FML,
                                 FeatureSB, FeatureSSBS, FeaturePAuth, FeatureFlagM, FeaturePredRes];
  list<SubtargetFeature> A520AE = [HasV9_2aOps, FeaturePerfMon, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVE2BitPerm,
                                 FeatureFP16FML,
                                 FeatureSB, FeatureSSBS, FeaturePAuth, FeatureFlagM, FeaturePredRes];
  list<SubtargetFeature> A65  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeatureSSBS, FeatureRAS,
                                 FeaturePerfMon];
  list<SubtargetFeature> A76  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeatureSSBS, FeaturePerfMon];
  list<SubtargetFeature> A77  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeaturePerfMon, FeatureSSBS];
  list<SubtargetFeature> A78  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeaturePerfMon, FeatureSPE,
                                 FeatureSSBS];
  list<SubtargetFeature> A78AE = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                  FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                  FeatureRCPC, FeaturePerfMon, FeatureSPE,
                                  FeatureSSBS];
  list<SubtargetFeature> A78C = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureFlagM, FeaturePAuth,
                                 FeaturePerfMon, FeatureRCPC, FeatureSPE,
                                 FeatureSSBS];
  list<SubtargetFeature> A710 = [HasV9_0aOps, FeatureNEON, FeaturePerfMon,
                                 FeatureETE, FeatureMTE, FeatureFP16FML,
                                 FeatureSVE2BitPerm, FeatureBF16, FeatureMatMulInt8,
                                 FeaturePAuth, FeatureFlagM, FeatureSB, FeatureSVE, FeatureSVE2];
  list<SubtargetFeature> A715 = [HasV9_0aOps, FeatureNEON, FeatureMTE,
                                 FeatureFP16FML, FeatureSVE, FeatureTRBE,
                                 FeatureSVE2BitPerm, FeatureBF16, FeatureETE,
                                 FeaturePerfMon, FeatureMatMulInt8, FeatureSPE,
                                 FeatureSB, FeatureSSBS, FeatureFullFP16, FeaturePAuth, FeaturePredRes, FeatureFlagM];
  list<SubtargetFeature> A720 = [HasV9_2aOps, FeatureMTE, FeatureFP16FML,
                                 FeatureTRBE, FeatureSVE2BitPerm, FeatureETE,
                                 FeaturePerfMon, FeatureSPE, FeatureSPE_EEF,
                                 FeatureSB, FeatureSSBS, FeaturePAuth, FeatureFlagM, FeaturePredRes];
  list<SubtargetFeature> A720AE = [HasV9_2aOps, FeatureMTE, FeatureFP16FML,
                                 FeatureTRBE, FeatureSVE2BitPerm, FeatureETE,
                                 FeaturePerfMon, FeatureSPE, FeatureSPE_EEF,
                                 FeatureSB, FeatureSSBS, FeaturePAuth, FeatureFlagM, FeaturePredRes];
  list<SubtargetFeature> R82  = [HasV8_0rOps, FeaturePerfMon, FeatureFullFP16,
                                 FeatureFP16FML, FeatureSSBS, FeaturePredRes,
                                 FeatureSB, FeatureRDM, FeatureDotProd,
                                 FeatureComplxNum, FeatureJS,
                                 FeatureCacheDeepPersist,
                                 FeatureLSE, FeatureFlagM];
  list<SubtargetFeature> R82AE = [HasV8_0rOps, FeaturePerfMon, FeatureFullFP16,
                                  FeatureFP16FML, FeatureSSBS, FeaturePredRes,
                                  FeatureSB, FeatureRDM, FeatureDotProd,
                                  FeatureComplxNum, FeatureJS,
                                  FeatureCacheDeepPersist,
                                  FeatureLSE, FeatureFlagM];
  list<SubtargetFeature> X1   = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureRCPC, FeaturePerfMon,
                                 FeatureSPE, FeatureFullFP16, FeatureDotProd,
                                 FeatureSSBS];
  list<SubtargetFeature> X1C  = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                 FeatureNEON, FeatureRCPC_IMMO, FeaturePerfMon,
                                 FeatureSPE, FeatureFullFP16, FeatureDotProd,
                                 FeaturePAuth, FeatureSSBS, FeatureFlagM,
                                 FeatureLSE2,
                                 FeatureRCPC];
  list<SubtargetFeature> X2   = [HasV9_0aOps, FeatureNEON, FeaturePerfMon,
                                 FeatureMatMulInt8, FeatureBF16, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVE2BitPerm,
                                 FeatureFP16FML,
                                 FeaturePAuth, FeatureSSBS, FeatureSB, FeatureSVE, FeatureSVE2, FeatureFlagM];
  list<SubtargetFeature> X3 =   [HasV9_0aOps, FeatureSVE, FeatureNEON,
                                 FeaturePerfMon, FeatureETE, FeatureTRBE,
                                 FeatureSPE, FeatureBF16, FeatureMatMulInt8,
                                 FeatureMTE, FeatureSVE2BitPerm, FeatureFullFP16,
                                 FeatureFP16FML,
                                 FeatureSB, FeaturePAuth, FeaturePredRes, FeatureFlagM, FeatureSSBS];
  list<SubtargetFeature> X4 =   [HasV9_2aOps,
                                 FeaturePerfMon, FeatureETE, FeatureTRBE,
                                 FeatureSPE, FeatureMTE, FeatureSVE2BitPerm,
                                 FeatureFP16FML, FeatureSPE_EEF,
                                 FeatureSB, FeatureSSBS, FeaturePAuth, FeatureFlagM, FeaturePredRes];
  list<SubtargetFeature> A64FX    = [HasV8_2aOps, FeatureFPARMv8, FeatureNEON,
                                     FeatureSHA2, FeaturePerfMon, FeatureFullFP16,
                                     FeatureSVE, FeatureComplxNum,
                                     FeatureAES];
  list<SubtargetFeature> Carmel   = [HasV8_2aOps, FeatureNEON, FeatureSHA2, FeatureAES,
                                     FeatureFullFP16];
  list<SubtargetFeature> AppleA7  = [HasV8_0aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON,FeaturePerfMon, FeatureAppleA7SysReg];
  list<SubtargetFeature> AppleA10 = [HasV8_0aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureCRC,
                                     FeatureRDM, FeaturePAN, FeatureLOR, FeatureVH];
  list<SubtargetFeature> AppleA11 = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureFullFP16];
  list<SubtargetFeature> AppleA12 = [HasV8_3aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureFullFP16];
  list<SubtargetFeature> AppleA13 = [HasV8_4aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureFullFP16,
                                     FeatureFP16FML, FeatureSHA3];
  list<SubtargetFeature> AppleA14 = [HasV8_5aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureFRInt3264,
                                     FeatureSpecRestrict, FeatureSSBS, FeatureSB,
                                     FeaturePredRes, FeatureCacheDeepPersist,
                                     FeatureFullFP16, FeatureFP16FML, FeatureSHA3,
                                     FeatureAltFPCmp];
  list<SubtargetFeature> AppleA15 = [HasV8_6aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureSHA3,
                                     FeatureFullFP16, FeatureFP16FML];
  list<SubtargetFeature> AppleA16 = [HasV8_6aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureSHA3,
                                     FeatureFullFP16, FeatureFP16FML,
                                     FeatureHCX];
  list<SubtargetFeature> AppleA17 = [HasV8_6aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureSHA3,
                                     FeatureFullFP16, FeatureFP16FML,
                                     FeatureHCX];
  list<SubtargetFeature> ExynosM3 = [HasV8_0aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                     FeaturePerfMon];
  list<SubtargetFeature> ExynosM4 = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureDotProd,
                                     FeatureFullFP16, FeaturePerfMon];
  list<SubtargetFeature> Falkor   = [HasV8_0aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                     FeatureFPARMv8, FeatureNEON, FeaturePerfMon,
                                     FeatureRDM];
  list<SubtargetFeature> NeoverseE1 = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureDotProd,
                                       FeatureFPARMv8, FeatureFullFP16, FeatureNEON,
                                       FeatureRCPC, FeatureSSBS, FeaturePerfMon];
  list<SubtargetFeature> NeoverseN1 = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureDotProd,
                                       FeatureFPARMv8, FeatureFullFP16, FeatureNEON,
                                       FeatureRCPC, FeatureSPE, FeatureSSBS,
                                       FeaturePerfMon];
  list<SubtargetFeature> NeoverseN2 = [HasV9_0aOps, FeatureBF16, FeatureETE, FeatureFP16FML,
                                       FeatureMatMulInt8, FeatureMTE, FeatureSVE2,
                                       FeatureSVE2BitPerm, FeatureTRBE,
                                       FeaturePerfMon,
                                       FeatureDotProd, FeatureFullFP16, FeatureSB, FeatureSSBS, FeatureSVE];
  list<SubtargetFeature> NeoverseN3 = [HasV9_2aOps, FeatureETE, FeatureFP16FML,
                                      FeatureFullFP16, FeatureMTE, FeaturePerfMon,
                                      FeatureRandGen, FeatureSPE, FeatureSPE_EEF,
                                      FeatureSVE2BitPerm,
                                      FeatureSSBS, FeatureSB, FeaturePredRes, FeaturePAuth, FeatureFlagM];
  list<SubtargetFeature> Neoverse512TVB = [HasV8_4aOps, FeatureBF16, FeatureCacheDeepPersist,
                                           FeatureSHA2, FeatureAES, FeatureFPARMv8, FeatureFP16FML,
                                           FeatureFullFP16, FeatureMatMulInt8, FeatureNEON,
                                           FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                           FeatureSSBS, FeatureSVE,
                                           FeatureSHA3, FeatureSM4, FeatureDotProd];
  list<SubtargetFeature> NeoverseV1 = [HasV8_4aOps, FeatureBF16, FeatureCacheDeepPersist,
                                       FeatureSHA2, FeatureAES, FeatureFPARMv8, FeatureFP16FML,
                                       FeatureFullFP16, FeatureMatMulInt8, FeatureNEON,
                                       FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                       FeatureSSBS, FeatureSVE,
                                       FeatureSHA3, FeatureSM4, FeatureDotProd];
  list<SubtargetFeature> NeoverseV2 = [HasV9_0aOps, FeatureBF16, FeatureSPE,
                                       FeaturePerfMon, FeatureETE, FeatureMatMulInt8,
                                       FeatureNEON, FeatureSVE2BitPerm, FeatureFP16FML,
                                       FeatureMTE, FeatureRandGen,
                                       FeatureSVE, FeatureSVE2, FeatureSSBS, FeatureFullFP16, FeatureDotProd];
  list<SubtargetFeature> NeoverseV3 = [HasV9_2aOps, FeatureETE, FeatureFP16FML,
                                      FeatureFullFP16, FeatureLS64, FeatureMTE,
                                      FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                      FeatureSPE_EEF, FeatureSVE2BitPerm, FeatureBRBE,
                                      FeatureSSBS, FeatureSB, FeaturePredRes, FeaturePAuth, FeatureFlagM];
  list<SubtargetFeature> NeoverseV3AE = [HasV9_2aOps, FeatureETE, FeatureFP16FML,
                                      FeatureFullFP16, FeatureLS64, FeatureMTE,
                                      FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                      FeatureSPE_EEF, FeatureSVE2BitPerm, FeatureBRBE,
                                      FeatureSSBS, FeatureSB, FeaturePredRes, FeaturePAuth, FeatureFlagM];
  list<SubtargetFeature> Saphira    = [HasV8_4aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                       FeatureNEON, FeatureSPE, FeaturePerfMon];
  list<SubtargetFeature> ThunderX   = [HasV8_0aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                       FeatureFPARMv8, FeaturePerfMon, FeatureNEON];
  list<SubtargetFeature> ThunderX2T99  = [HasV8_1aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                          FeatureFPARMv8, FeatureNEON, FeatureLSE];
  list<SubtargetFeature> ThunderX3T110 = [HasV8_3aOps, FeatureCRC, FeatureSHA2, FeatureAES,
                                          FeatureFPARMv8, FeatureNEON, FeatureLSE,
                                          FeaturePAuth, FeaturePerfMon];
  list<SubtargetFeature> TSV110 = [HasV8_2aOps, FeatureSHA2, FeatureAES, FeatureFPARMv8,
                                   FeatureNEON, FeaturePerfMon, FeatureSPE,
                                   FeatureFullFP16, FeatureFP16FML, FeatureDotProd,
                                   FeatureJS, FeatureComplxNum];
  list<SubtargetFeature> Ampere1 = [HasV8_6aOps, FeatureNEON, FeaturePerfMon,
                                    FeatureSSBS, FeatureRandGen, FeatureSB,
                                    FeatureSHA2, FeatureSHA3, FeatureAES,
                                    FeatureFullFP16];
  list<SubtargetFeature> Ampere1A = [HasV8_6aOps, FeatureNEON, FeaturePerfMon,
                                     FeatureMTE, FeatureSSBS, FeatureRandGen,
                                     FeatureSB, FeatureSM4, FeatureSHA2,
                                     FeatureSHA3, FeatureAES,
                                     FeatureFullFP16];
  list<SubtargetFeature> Ampere1B = [HasV8_7aOps, FeatureNEON, FeaturePerfMon,
                                     FeatureMTE, FeatureSSBS, FeatureRandGen,
                                     FeatureSB, FeatureSM4, FeatureSHA2,
                                     FeatureSHA3, FeatureAES, FeatureCSSC,
                                     FeatureWFxT, FeatureFullFP16];

  // ETE and TRBE are future architecture extensions. We temporarily enable them
  // by default for users targeting generic AArch64. The extensions do not
  // affect code generated by the compiler and can be used only by explicitly
  // mentioning the new system register names in assembly.
  list<SubtargetFeature> Generic = [FeatureFPARMv8, FeatureNEON, FeatureETE];
}

class AArch64Processor<
  string n,
  Architecture64 arch,
  SchedMachineModel m,
  list<SubtargetFeature> f,
  list<SubtargetFeature> tunef
> : ProcessorModel<n, m, f, tunef> {
  // The base architecture for this processor.
  Architecture64 Arch = arch;
}

// FeatureFuseAdrpAdd is enabled under Generic to allow linker merging
// optimizations.
def : AArch64Processor<"generic", HasV8_0aOps, CortexA510Model, ProcessorFeatures.Generic,
                     [FeatureFuseAES, FeatureFuseAdrpAdd, FeaturePostRAScheduler,
                      FeatureEnableSelectOptimize]>;

// Alias for the latest Apple processor model supported by LLVM.
def : ProcessorModel<"apple-latest", CycloneModel, ProcessorFeatures.AppleA16,
                     [TuneAppleA16]>;

def : AArch64Processor<"cortex-a34", HasV8_0aOps, CortexA53Model, ProcessorFeatures.A53,
                     [TuneA35]>;
def : AArch64Processor<"cortex-a35", HasV8_0aOps, CortexA53Model, ProcessorFeatures.A53,
                     [TuneA35]>;
def : AArch64Processor<"cortex-a53", HasV8_0aOps, CortexA53Model, ProcessorFeatures.A53,
                     [TuneA53]>;
def : AArch64Processor<"cortex-a55", HasV8_2aOps, CortexA55Model, ProcessorFeatures.A55,
                     [TuneA55]>;
def : AArch64Processor<"cortex-a510", HasV9_0aOps, CortexA510Model, ProcessorFeatures.A510,
                     [TuneA510]>;
def : AArch64Processor<"cortex-a520", HasV9_2aOps, CortexA510Model, ProcessorFeatures.A520,
                     [TuneA520]>;
def : AArch64Processor<"cortex-a520ae", HasV9_2aOps, CortexA510Model, ProcessorFeatures.A520AE,
                     [TuneA520AE]>;
def : AArch64Processor<"cortex-a57", HasV8_0aOps, CortexA57Model, ProcessorFeatures.A53,
                     [TuneA57]>;
def : AArch64Processor<"cortex-a65", HasV8_2aOps, CortexA53Model, ProcessorFeatures.A65,
                     [TuneA65]>;
def : AArch64Processor<"cortex-a65ae", HasV8_2aOps, CortexA53Model, ProcessorFeatures.A65,
                     [TuneA65]>;
def : AArch64Processor<"cortex-a72", HasV8_0aOps, CortexA57Model, ProcessorFeatures.A53,
                     [TuneA72]>;
def : AArch64Processor<"cortex-a73", HasV8_0aOps, CortexA57Model, ProcessorFeatures.A53,
                     [TuneA73]>;
def : AArch64Processor<"cortex-a75", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A55,
                     [TuneA75]>;
def : AArch64Processor<"cortex-a76", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A76,
                     [TuneA76]>;
def : AArch64Processor<"cortex-a76ae", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A76,
                     [TuneA76]>;
def : AArch64Processor<"cortex-a77", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A77,
                     [TuneA77]>;
def : AArch64Processor<"cortex-a78", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A78,
                     [TuneA78]>;
def : AArch64Processor<"cortex-a78ae", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A78AE,
                     [TuneA78AE]>;
def : AArch64Processor<"cortex-a78c", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A78C,
                     [TuneA78C]>;
def : AArch64Processor<"cortex-a710", HasV9_0aOps, NeoverseN2Model, ProcessorFeatures.A710,
                     [TuneA710]>;
def : AArch64Processor<"cortex-a715", HasV9_0aOps, NeoverseN2Model, ProcessorFeatures.A715,
                     [TuneA715]>;
def : AArch64Processor<"cortex-a720", HasV9_2aOps, NeoverseN2Model, ProcessorFeatures.A720,
                     [TuneA720]>;
def : AArch64Processor<"cortex-a720ae", HasV9_2aOps, NeoverseN2Model, ProcessorFeatures.A720AE,
                     [TuneA720AE]>;
def : AArch64Processor<"cortex-r82", HasV8_0rOps, CortexA55Model, ProcessorFeatures.R82,
                     [TuneR82]>;
def : AArch64Processor<"cortex-r82ae", HasV8_0rOps, CortexA55Model, ProcessorFeatures.R82AE,
                     [TuneR82AE]>;
def : AArch64Processor<"cortex-x1", HasV8_2aOps, CortexA57Model, ProcessorFeatures.X1,
                     [TuneX1]>;
def : AArch64Processor<"cortex-x1c", HasV8_2aOps, CortexA57Model, ProcessorFeatures.X1C,
                     [TuneX1]>;
def : AArch64Processor<"cortex-x2", HasV9_0aOps, NeoverseN2Model, ProcessorFeatures.X2,
                     [TuneX2]>;
def : AArch64Processor<"cortex-x3", HasV9_0aOps, NeoverseN2Model, ProcessorFeatures.X3,
                     [TuneX3]>;
def : AArch64Processor<"cortex-x4", HasV9_2aOps, NeoverseN2Model, ProcessorFeatures.X4,
                     [TuneX4]>;
def : AArch64Processor<"neoverse-e1", HasV8_2aOps, CortexA53Model, ProcessorFeatures.NeoverseE1,
                     [TuneNeoverseE1]>;
def : AArch64Processor<"neoverse-n1", HasV8_2aOps, NeoverseN1Model, ProcessorFeatures.NeoverseN1,
                     [TuneNeoverseN1]>;
def : AArch64Processor<"neoverse-n2", HasV9_0aOps, NeoverseN2Model, ProcessorFeatures.NeoverseN2,
                     [TuneNeoverseN2]>;
def : AArch64Processor<"neoverse-n3", HasV9_2aOps, NeoverseN2Model, ProcessorFeatures.NeoverseN3,
                     [TuneNeoverseN3]>;
def : AArch64Processor<"neoverse-512tvb", HasV8_4aOps, NeoverseV1Model, ProcessorFeatures.Neoverse512TVB,
                     [TuneNeoverse512TVB]>;
def : AArch64Processor<"neoverse-v1", HasV8_4aOps, NeoverseV1Model, ProcessorFeatures.NeoverseV1,
                     [TuneNeoverseV1]>;
def : AArch64Processor<"neoverse-v2", HasV9_0aOps, NeoverseV2Model, ProcessorFeatures.NeoverseV2,
                     [TuneNeoverseV2]>;
def : AArch64Processor<"neoverse-v3", HasV9_2aOps, NeoverseV2Model, ProcessorFeatures.NeoverseV3,
                     [TuneNeoverseV3]>;
def : AArch64Processor<"neoverse-v3ae", HasV9_2aOps, NeoverseV2Model, ProcessorFeatures.NeoverseV3AE,
                     [TuneNeoverseV3AE]>;
def : AArch64Processor<"exynos-m3", HasV8_0aOps, ExynosM3Model, ProcessorFeatures.ExynosM3,
                     [TuneExynosM3]>;
def : AArch64Processor<"exynos-m4", HasV8_2aOps, ExynosM4Model, ProcessorFeatures.ExynosM4,
                     [TuneExynosM4]>;
def : AArch64Processor<"exynos-m5", HasV8_2aOps, ExynosM5Model, ProcessorFeatures.ExynosM4,
                     [TuneExynosM4]>;
def : AArch64Processor<"falkor", HasV8_0aOps, FalkorModel, ProcessorFeatures.Falkor,
                     [TuneFalkor]>;
def : AArch64Processor<"saphira", HasV8_4aOps, FalkorModel, ProcessorFeatures.Saphira,
                     [TuneSaphira]>;
def : AArch64Processor<"kryo", HasV8_0aOps, KryoModel, ProcessorFeatures.A53,
                     [TuneKryo]>;

// Cavium ThunderX/ThunderX T8X  Processors
def : AArch64Processor<"thunderx", HasV8_0aOps, ThunderXT8XModel,  ProcessorFeatures.ThunderX,
                     [TuneThunderX]>;
def : AArch64Processor<"thunderxt88", HasV8_0aOps, ThunderXT8XModel, ProcessorFeatures.ThunderX,
                     [TuneThunderXT88]>;
def : AArch64Processor<"thunderxt81", HasV8_0aOps, ThunderXT8XModel, ProcessorFeatures.ThunderX,
                     [TuneThunderXT81]>;
def : AArch64Processor<"thunderxt83", HasV8_0aOps, ThunderXT8XModel, ProcessorFeatures.ThunderX,
                     [TuneThunderXT83]>;
// Cavium ThunderX2T9X  Processors. Formerly Broadcom Vulcan.
def : AArch64Processor<"thunderx2t99", HasV8_1aOps, ThunderX2T99Model, ProcessorFeatures.ThunderX2T99,
                     [TuneThunderX2T99]>;
// Marvell ThunderX3T110 Processors.
def : AArch64Processor<"thunderx3t110", HasV8_3aOps, ThunderX3T110Model, ProcessorFeatures.ThunderX3T110,
                     [TuneThunderX3T110]>;
def : AArch64Processor<"tsv110", HasV8_2aOps, TSV110Model, ProcessorFeatures.TSV110,
                     [TuneTSV110]>;

// Support cyclone as an alias for apple-a7 so we can still LTO old bitcode.
def : AArch64Processor<"cyclone", HasV8_0aOps, CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7]>;

// iPhone and iPad CPUs
def : AArch64Processor<"apple-a7", HasV8_0aOps, CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7]>;
def : AArch64Processor<"apple-a8", HasV8_0aOps, CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7]>;
def : AArch64Processor<"apple-a9", HasV8_0aOps, CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7]>;
def : AArch64Processor<"apple-a10", HasV8_0aOps, CycloneModel, ProcessorFeatures.AppleA10,
                     [TuneAppleA10]>;
def : AArch64Processor<"apple-a11", HasV8_2aOps, CycloneModel, ProcessorFeatures.AppleA11,
                     [TuneAppleA11]>;
def : AArch64Processor<"apple-a12", HasV8_3aOps, CycloneModel, ProcessorFeatures.AppleA12,
                     [TuneAppleA12]>;
def : AArch64Processor<"apple-a13", HasV8_4aOps, CycloneModel, ProcessorFeatures.AppleA13,
                     [TuneAppleA13]>;
def : AArch64Processor<"apple-a14", HasV8_5aOps, CycloneModel, ProcessorFeatures.AppleA14,
                     [TuneAppleA14]>;
def : AArch64Processor<"apple-a15", HasV8_6aOps, CycloneModel, ProcessorFeatures.AppleA15,
                     [TuneAppleA15]>;
// Alias for the latest Apple processor model supported by LLVM.
def : AArch64Processor<"apple-a16", HasV8_6aOps, CycloneModel, ProcessorFeatures.AppleA16,
                     [TuneAppleA16]>;
def : AArch64Processor<"apple-a17", HasV8_6aOps, CycloneModel, ProcessorFeatures.AppleA17,
                     [TuneAppleA17]>;
// Mac CPUs
def : AArch64Processor<"apple-m1", HasV8_5aOps, CycloneModel, ProcessorFeatures.AppleA14,
                     [TuneAppleA14]>;
def : AArch64Processor<"apple-m2", HasV8_6aOps, CycloneModel, ProcessorFeatures.AppleA15,
                     [TuneAppleA15]>;
def : AArch64Processor<"apple-m3", HasV8_6aOps, CycloneModel, ProcessorFeatures.AppleA16,
                     [TuneAppleA16]>;

// watch CPUs.
def : AArch64Processor<"apple-s4", HasV8_3aOps, CycloneModel, ProcessorFeatures.AppleA12,
                     [TuneAppleA12]>;
def : AArch64Processor<"apple-s5", HasV8_3aOps, CycloneModel, ProcessorFeatures.AppleA12,
                     [TuneAppleA12]>;

// Fujitsu A64FX
def : AArch64Processor<"a64fx", HasV8_2aOps, A64FXModel, ProcessorFeatures.A64FX,
                     [TuneA64FX]>;

// Nvidia Carmel
def : AArch64Processor<"carmel", HasV8_2aOps, NoSchedModel, ProcessorFeatures.Carmel,
                     [TuneCarmel]>;

// Ampere Computing
def : AArch64Processor<"ampere1", HasV8_6aOps, Ampere1Model, ProcessorFeatures.Ampere1,
                     [TuneAmpere1]>;

def : AArch64Processor<"ampere1a", HasV8_6aOps, Ampere1Model, ProcessorFeatures.Ampere1A,
                     [TuneAmpere1A]>;

def : AArch64Processor<"ampere1b", HasV8_7aOps, Ampere1BModel, ProcessorFeatures.Ampere1B,
                     [TuneAmpere1B]>;
