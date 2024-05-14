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
  list<SubtargetFeature> A53  = [HasV8_0aOps, FeatureCRC, FeatureCrypto,
                                 FeatureFPARMv8, FeatureNEON, FeaturePerfMon];
  list<SubtargetFeature> A55  = [HasV8_2aOps, FeatureCrypto, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeaturePerfMon];
  list<SubtargetFeature> A510 = [HasV9_0aOps, FeatureNEON, FeaturePerfMon,
                                 FeatureMatMulInt8, FeatureBF16, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVE2BitPerm,
                                 FeatureFP16FML];
  list<SubtargetFeature> A520 = [HasV9_2aOps, FeaturePerfMon, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVE2BitPerm,
                                 FeatureFP16FML];
  list<SubtargetFeature> A520AE = [HasV9_2aOps, FeaturePerfMon, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVE2BitPerm,
                                 FeatureFP16FML];
  list<SubtargetFeature> A65  = [HasV8_2aOps, FeatureCrypto, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeatureSSBS, FeatureRAS,
                                 FeaturePerfMon];
  list<SubtargetFeature> A76  = [HasV8_2aOps, FeatureCrypto, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeatureSSBS, FeaturePerfMon];
  list<SubtargetFeature> A77  = [HasV8_2aOps, FeatureCrypto, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeaturePerfMon, FeatureSSBS];
  list<SubtargetFeature> A78  = [HasV8_2aOps, FeatureCrypto, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureRCPC, FeaturePerfMon, FeatureSPE,
                                 FeatureSSBS];
  list<SubtargetFeature> A78AE = [HasV8_2aOps, FeatureCrypto, FeatureFPARMv8,
                                  FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                  FeatureRCPC, FeaturePerfMon, FeatureSPE,
                                  FeatureSSBS];
  list<SubtargetFeature> A78C = [HasV8_2aOps, FeatureCrypto, FeatureFPARMv8,
                                 FeatureNEON, FeatureFullFP16, FeatureDotProd,
                                 FeatureFlagM, FeaturePAuth,
                                 FeaturePerfMon, FeatureRCPC, FeatureSPE,
                                 FeatureSSBS];
  list<SubtargetFeature> A710 = [HasV9_0aOps, FeatureNEON, FeaturePerfMon,
                                 FeatureETE, FeatureMTE, FeatureFP16FML,
                                 FeatureSVE2BitPerm, FeatureBF16, FeatureMatMulInt8];
  list<SubtargetFeature> A715 = [HasV9_0aOps, FeatureNEON, FeatureMTE,
                                 FeatureFP16FML, FeatureSVE, FeatureTRBE,
                                 FeatureSVE2BitPerm, FeatureBF16, FeatureETE,
                                 FeaturePerfMon, FeatureMatMulInt8, FeatureSPE];
  list<SubtargetFeature> A720 = [HasV9_2aOps, FeatureMTE, FeatureFP16FML,
                                 FeatureTRBE, FeatureSVE2BitPerm, FeatureETE,
                                 FeaturePerfMon, FeatureSPE, FeatureSPE_EEF];
  list<SubtargetFeature> A720AE = [HasV9_2aOps, FeatureMTE, FeatureFP16FML,
                                 FeatureTRBE, FeatureSVE2BitPerm, FeatureETE,
                                 FeaturePerfMon, FeatureSPE, FeatureSPE_EEF];
  list<SubtargetFeature> R82  = [HasV8_0rOps, FeaturePerfMon, FeatureFullFP16,
                                 FeatureFP16FML, FeatureSSBS, FeaturePredRes,
                                 FeatureSB, FeatureRDM, FeatureDotProd,
                                 FeatureComplxNum, FeatureJS,
                                 FeatureCacheDeepPersist];
  list<SubtargetFeature> R82AE = [HasV8_0rOps, FeaturePerfMon, FeatureFullFP16,
                                  FeatureFP16FML, FeatureSSBS, FeaturePredRes,
                                  FeatureSB, FeatureRDM, FeatureDotProd,
                                  FeatureComplxNum, FeatureJS,
                                  FeatureCacheDeepPersist];
  list<SubtargetFeature> X1   = [HasV8_2aOps, FeatureCrypto, FeatureFPARMv8,
                                 FeatureNEON, FeatureRCPC, FeaturePerfMon,
                                 FeatureSPE, FeatureFullFP16, FeatureDotProd,
                                 FeatureSSBS];
  list<SubtargetFeature> X1C  = [HasV8_2aOps, FeatureCrypto, FeatureFPARMv8,
                                 FeatureNEON, FeatureRCPC_IMMO, FeaturePerfMon,
                                 FeatureSPE, FeatureFullFP16, FeatureDotProd,
                                 FeaturePAuth, FeatureSSBS, FeatureFlagM,
                                 FeatureLSE2];
  list<SubtargetFeature> X2   = [HasV9_0aOps, FeatureNEON, FeaturePerfMon,
                                 FeatureMatMulInt8, FeatureBF16, FeatureAM,
                                 FeatureMTE, FeatureETE, FeatureSVE2BitPerm,
                                 FeatureFP16FML];
  list<SubtargetFeature> X3 =   [HasV9_0aOps, FeatureSVE, FeatureNEON,
                                 FeaturePerfMon, FeatureETE, FeatureTRBE,
                                 FeatureSPE, FeatureBF16, FeatureMatMulInt8,
                                 FeatureMTE, FeatureSVE2BitPerm, FeatureFullFP16,
                                 FeatureFP16FML];
  list<SubtargetFeature> X4 =   [HasV9_2aOps,
                                 FeaturePerfMon, FeatureETE, FeatureTRBE,
                                 FeatureSPE, FeatureMTE, FeatureSVE2BitPerm,
                                 FeatureFP16FML, FeatureSPE_EEF];
  list<SubtargetFeature> A64FX    = [HasV8_2aOps, FeatureFPARMv8, FeatureNEON,
                                     FeatureSHA2, FeaturePerfMon, FeatureFullFP16,
                                     FeatureSVE, FeatureComplxNum];
  list<SubtargetFeature> Carmel   = [HasV8_2aOps, FeatureNEON, FeatureCrypto,
                                     FeatureFullFP16];
  list<SubtargetFeature> AppleA7  = [HasV8_0aOps, FeatureCrypto, FeatureFPARMv8,
                                     FeatureNEON,FeaturePerfMon, FeatureAppleA7SysReg];
  list<SubtargetFeature> AppleA10 = [HasV8_0aOps, FeatureCrypto, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureCRC,
                                     FeatureRDM, FeaturePAN, FeatureLOR, FeatureVH];
  list<SubtargetFeature> AppleA11 = [HasV8_2aOps, FeatureCrypto, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureFullFP16];
  list<SubtargetFeature> AppleA12 = [HasV8_3aOps, FeatureCrypto, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureFullFP16];
  list<SubtargetFeature> AppleA13 = [HasV8_4aOps, FeatureCrypto, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureFullFP16,
                                     FeatureFP16FML, FeatureSHA3];
  list<SubtargetFeature> AppleA14 = [HasV8_4aOps, FeatureCrypto, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureFRInt3264,
                                     FeatureSpecRestrict, FeatureSSBS, FeatureSB,
                                     FeaturePredRes, FeatureCacheDeepPersist,
                                     FeatureFullFP16, FeatureFP16FML, FeatureSHA3,
                                     FeatureAltFPCmp];
  list<SubtargetFeature> AppleA15 = [HasV8_6aOps, FeatureCrypto, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureSHA3,
                                     FeatureFullFP16, FeatureFP16FML];
  list<SubtargetFeature> AppleA16 = [HasV8_6aOps, FeatureCrypto, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureSHA3,
                                     FeatureFullFP16, FeatureFP16FML,
                                     FeatureHCX];
  list<SubtargetFeature> AppleA17 = [HasV8_6aOps, FeatureCrypto, FeatureFPARMv8,
                                     FeatureNEON, FeaturePerfMon, FeatureSHA3,
                                     FeatureFullFP16, FeatureFP16FML,
                                     FeatureHCX];
  list<SubtargetFeature> ExynosM3 = [HasV8_0aOps, FeatureCRC, FeatureCrypto,
                                     FeaturePerfMon];
  list<SubtargetFeature> ExynosM4 = [HasV8_2aOps, FeatureCrypto, FeatureDotProd,
                                     FeatureFullFP16, FeaturePerfMon];
  list<SubtargetFeature> Falkor   = [HasV8_0aOps, FeatureCRC, FeatureCrypto,
                                     FeatureFPARMv8, FeatureNEON, FeaturePerfMon,
                                     FeatureRDM];
  list<SubtargetFeature> NeoverseE1 = [HasV8_2aOps, FeatureCrypto, FeatureDotProd,
                                       FeatureFPARMv8, FeatureFullFP16, FeatureNEON,
                                       FeatureRCPC, FeatureSSBS, FeaturePerfMon];
  list<SubtargetFeature> NeoverseN1 = [HasV8_2aOps, FeatureCrypto, FeatureDotProd,
                                       FeatureFPARMv8, FeatureFullFP16, FeatureNEON,
                                       FeatureRCPC, FeatureSPE, FeatureSSBS,
                                       FeaturePerfMon];
  list<SubtargetFeature> NeoverseN2 = [HasV9_0aOps, FeatureBF16, FeatureETE, FeatureFP16FML,
                                       FeatureMatMulInt8, FeatureMTE, FeatureSVE2,
                                       FeatureSVE2BitPerm, FeatureTRBE,
                                       FeaturePerfMon];
  list<SubtargetFeature> NeoverseN3 = [HasV9_2aOps, FeatureETE, FeatureFP16FML,
                                      FeatureFullFP16, FeatureMTE, FeaturePerfMon,
                                      FeatureRandGen, FeatureSPE, FeatureSPE_EEF,
                                      FeatureSVE2BitPerm];
  list<SubtargetFeature> Neoverse512TVB = [HasV8_4aOps, FeatureBF16, FeatureCacheDeepPersist,
                                           FeatureCrypto, FeatureFPARMv8, FeatureFP16FML,
                                           FeatureFullFP16, FeatureMatMulInt8, FeatureNEON,
                                           FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                           FeatureSSBS, FeatureSVE];
  list<SubtargetFeature> NeoverseV1 = [HasV8_4aOps, FeatureBF16, FeatureCacheDeepPersist,
                                       FeatureCrypto, FeatureFPARMv8, FeatureFP16FML,
                                       FeatureFullFP16, FeatureMatMulInt8, FeatureNEON,
                                       FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                       FeatureSSBS, FeatureSVE];
  list<SubtargetFeature> NeoverseV2 = [HasV9_0aOps, FeatureBF16, FeatureSPE,
                                       FeaturePerfMon, FeatureETE, FeatureMatMulInt8,
                                       FeatureNEON, FeatureSVE2BitPerm, FeatureFP16FML,
                                       FeatureMTE, FeatureRandGen];
  list<SubtargetFeature> NeoverseV3 = [HasV9_2aOps, FeatureETE, FeatureFP16FML,
                                      FeatureFullFP16, FeatureLS64, FeatureMTE,
                                      FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                      FeatureSPE_EEF, FeatureSVE2BitPerm, FeatureBRBE];
  list<SubtargetFeature> NeoverseV3AE = [HasV9_2aOps, FeatureETE, FeatureFP16FML,
                                      FeatureFullFP16, FeatureLS64, FeatureMTE,
                                      FeaturePerfMon, FeatureRandGen, FeatureSPE,
                                      FeatureSPE_EEF, FeatureSVE2BitPerm, FeatureBRBE];
  list<SubtargetFeature> Saphira    = [HasV8_4aOps, FeatureCrypto, FeatureFPARMv8,
                                       FeatureNEON, FeatureSPE, FeaturePerfMon];
  list<SubtargetFeature> ThunderX   = [HasV8_0aOps, FeatureCRC, FeatureCrypto,
                                       FeatureFPARMv8, FeaturePerfMon, FeatureNEON];
  list<SubtargetFeature> ThunderX2T99  = [HasV8_1aOps, FeatureCRC, FeatureCrypto,
                                          FeatureFPARMv8, FeatureNEON, FeatureLSE];
  list<SubtargetFeature> ThunderX3T110 = [HasV8_3aOps, FeatureCRC, FeatureCrypto,
                                          FeatureFPARMv8, FeatureNEON, FeatureLSE,
                                          FeaturePAuth, FeaturePerfMon];
  list<SubtargetFeature> TSV110 = [HasV8_2aOps, FeatureCrypto, FeatureFPARMv8,
                                   FeatureNEON, FeaturePerfMon, FeatureSPE,
                                   FeatureFullFP16, FeatureFP16FML, FeatureDotProd,
                                   FeatureJS, FeatureComplxNum];
  list<SubtargetFeature> Ampere1 = [HasV8_6aOps, FeatureNEON, FeaturePerfMon,
                                    FeatureSSBS, FeatureRandGen, FeatureSB,
                                    FeatureSHA2, FeatureSHA3, FeatureAES];
  list<SubtargetFeature> Ampere1A = [HasV8_6aOps, FeatureNEON, FeaturePerfMon,
                                     FeatureMTE, FeatureSSBS, FeatureRandGen,
                                     FeatureSB, FeatureSM4, FeatureSHA2,
                                     FeatureSHA3, FeatureAES];
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
  list<SubtargetFeature> tunef,
  list<Extension> default_extensions
> : ProcessorModel<n, m, f, tunef> {
  // The base architecture for this processor.
  Architecture64 Arch = arch;

  // The set of extensions enabled by default for this processor.
  list<Extension> DefaultExts = default_extensions;
}

// FeatureFuseAdrpAdd is enabled under Generic to allow linker merging
// optimizations.
def : AArch64Processor<"generic", HasV9_0aOps, CortexA510Model, ProcessorFeatures.Generic,
                     [FeatureFuseAES, FeatureFuseAdrpAdd, FeaturePostRAScheduler,
                      FeatureEnableSelectOptimize],
                     [/*FIXME what are the default features for generic?*/]>;

// Alias for the latest Apple processor model supported by LLVM.
def : ProcessorModel<"apple-latest", CycloneModel, ProcessorFeatures.AppleA16,
                     [TuneAppleA16]>;

def : AArch64Processor<"cortex-a34", HasV8_0aOps, CortexA53Model, ProcessorFeatures.A53,
                     [TuneA35],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;
def : AArch64Processor<"cortex-a35", HasV8_0aOps, CortexA53Model, ProcessorFeatures.A53,
                     [TuneA35],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;
def : AArch64Processor<"cortex-a53", HasV8_0aOps, CortexA53Model, ProcessorFeatures.A53,
                     [TuneA53],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;
def : AArch64Processor<"cortex-a55", HasV8_2aOps, CortexA55Model, ProcessorFeatures.A55,
                     [TuneA55],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16, FeatureDotProd, FeatureRCPC]>;
def : AArch64Processor<"cortex-a510", HasV9_0aOps, CortexA510Model, ProcessorFeatures.A510,
                     [TuneA510],
                     [FeatureBF16, FeatureMatMulInt8, FeatureSB, FeaturePAuth, FeatureMTE, FeatureSSBS, FeatureSVE, FeatureSVE2, FeatureSVE2BitPerm, FeatureFP16FML]>;
def : AArch64Processor<"cortex-a520", HasV9_2aOps, CortexA510Model, ProcessorFeatures.A520,
                     [TuneA520],
                     [FeatureSB, FeatureSSBS, FeatureMTE, FeatureFP16FML, FeaturePAuth, FeatureSVE2BitPerm, FeatureFlagM, FeaturePerfMon, FeaturePredRes]>;
def : AArch64Processor<"cortex-a520ae", HasV9_2aOps, CortexA510Model, ProcessorFeatures.A520AE,
                     [TuneA520AE],
                     [FeatureSB, FeatureSSBS, FeatureMTE, FeatureFP16FML, FeaturePAuth, FeatureSVE2BitPerm, FeatureFlagM, FeaturePerfMon, FeaturePredRes]>;
def : AArch64Processor<"cortex-a57", HasV8_0aOps, CortexA57Model, ProcessorFeatures.A53,
                     [TuneA57],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;
def : AArch64Processor<"cortex-a65", HasV8_2aOps, CortexA53Model, ProcessorFeatures.A65,
                     [TuneA65],
                     [FeatureAES, FeatureSHA2, FeatureDotProd, FeatureFullFP16, FeatureRCPC, FeatureSSBS]>;
def : AArch64Processor<"cortex-a65ae", HasV8_2aOps, CortexA53Model, ProcessorFeatures.A65,
                     [TuneA65],
                     [FeatureAES, FeatureSHA2, FeatureDotProd, FeatureFullFP16, FeatureRCPC, FeatureSSBS]>;
def : AArch64Processor<"cortex-a72", HasV8_0aOps, CortexA57Model, ProcessorFeatures.A53,
                     [TuneA72],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;
def : AArch64Processor<"cortex-a73", HasV8_0aOps, CortexA57Model, ProcessorFeatures.A53,
                     [TuneA73],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;
def : AArch64Processor<"cortex-a75", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A55,
                     [TuneA75],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16, FeatureDotProd, FeatureRCPC]>;
def : AArch64Processor<"cortex-a76", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A76,
                     [TuneA76],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16, FeatureDotProd, FeatureRCPC, FeatureSSBS]>;
def : AArch64Processor<"cortex-a76ae", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A76,
                     [TuneA76],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16, FeatureDotProd, FeatureRCPC, FeatureSSBS]>;
def : AArch64Processor<"cortex-a77", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A77,
                     [TuneA77],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16, FeatureRCPC, FeatureDotProd, FeatureSSBS]>;
def : AArch64Processor<"cortex-a78", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A78,
                     [TuneA78],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16, FeatureDotProd, FeatureRCPC, FeatureSSBS, FeatureSPE]>;
def : AArch64Processor<"cortex-a78ae", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A78AE,
                     [TuneA78AE],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16, FeatureDotProd, FeatureRCPC, FeatureSSBS, FeatureSPE]>;
def : AArch64Processor<"cortex-a78c", HasV8_2aOps, CortexA57Model, ProcessorFeatures.A78C,
                     [TuneA78C],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16, FeatureDotProd, FeatureRCPC, FeatureSSBS, FeatureSPE, FeatureFlagM, FeaturePAuth]>;
def : AArch64Processor<"cortex-a710", HasV9_0aOps, NeoverseN2Model, ProcessorFeatures.A710,
                     [TuneA710],
                     [FeatureMTE, FeaturePAuth, FeatureFlagM, FeatureSB, FeatureMatMulInt8, FeatureFP16FML, FeatureSVE, FeatureSVE2, FeatureSVE2BitPerm, FeatureBF16]>;
def : AArch64Processor<"cortex-a715", HasV9_0aOps, NeoverseN2Model, ProcessorFeatures.A715,
                     [TuneA715],
                     [FeatureSB, FeatureSSBS, FeatureMTE, FeatureFullFP16, FeatureFP16FML, FeaturePAuth, FeatureMatMulInt8, FeaturePredRes, FeaturePerfMon, FeatureSPE, FeatureSVE, FeatureSVE2BitPerm, FeatureBF16, FeatureFlagM]>;
def : AArch64Processor<"cortex-a720", HasV9_2aOps, NeoverseN2Model, ProcessorFeatures.A720,
                     [TuneA720],
                     [FeatureSB, FeatureSSBS, FeatureMTE, FeatureFP16FML, FeaturePAuth, FeatureSVE2BitPerm, FeatureFlagM, FeaturePerfMon, FeaturePredRes, FeatureSPE]>;
def : AArch64Processor<"cortex-a720ae", HasV9_2aOps, NeoverseN2Model, ProcessorFeatures.A720AE,
                     [TuneA720AE],
                     [FeatureSB, FeatureSSBS, FeatureMTE, FeatureFP16FML, FeaturePAuth, FeatureSVE2BitPerm, FeatureFlagM, FeaturePerfMon, FeaturePredRes, FeatureSPE]>;
def : AArch64Processor<"cortex-r82", HasV8_0rOps, CortexA55Model, ProcessorFeatures.R82,
                     [TuneR82],
                     [FeatureLSE, FeatureFlagM, FeaturePerfMon, FeaturePredRes]>;
def : AArch64Processor<"cortex-r82ae", HasV8_0rOps, CortexA55Model, ProcessorFeatures.R82AE,
                     [TuneR82AE],
                     [FeatureLSE, FeatureFlagM, FeaturePerfMon, FeaturePredRes]>;
def : AArch64Processor<"cortex-x1", HasV8_2aOps, CortexA57Model, ProcessorFeatures.X1,
                     [TuneX1],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16, FeatureDotProd, FeatureRCPC, FeatureSSBS, FeatureSPE]>;
def : AArch64Processor<"cortex-x1c", HasV8_2aOps, CortexA57Model, ProcessorFeatures.X1C,
                     [TuneX1],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16, FeatureDotProd, FeatureRCPC, FeatureSSBS, FeaturePAuth, FeatureSPE, FeatureFlagM]>;
def : AArch64Processor<"cortex-x2", HasV9_0aOps, NeoverseN2Model, ProcessorFeatures.X2,
                     [TuneX2],
                     [FeatureMTE, FeatureBF16, FeatureMatMulInt8, FeaturePAuth, FeatureSSBS, FeatureSB, FeatureSVE, FeatureSVE2, FeatureSVE2BitPerm, FeatureFP16FML, FeatureFlagM]>;
def : AArch64Processor<"cortex-x3", HasV9_0aOps, NeoverseN2Model, ProcessorFeatures.X3,
                     [TuneX3],
                     [FeatureSVE, FeaturePerfMon, FeatureSPE, FeatureBF16, FeatureMatMulInt8, FeatureMTE, FeatureSVE2BitPerm, FeatureSB, FeaturePAuth, FeatureFullFP16, FeatureFP16FML, FeaturePredRes, FeatureFlagM, FeatureSSBS]>;
def : AArch64Processor<"cortex-x4", HasV9_2aOps, NeoverseN2Model, ProcessorFeatures.X4,
                     [TuneX4],
                     [FeatureSB, FeatureSSBS, FeatureMTE, FeatureFP16FML, FeaturePAuth, FeatureSVE2BitPerm, FeatureFlagM, FeaturePerfMon, FeaturePredRes, FeatureSPE]>;
def : AArch64Processor<"neoverse-e1", HasV8_2aOps, CortexA53Model, ProcessorFeatures.NeoverseE1,
                     [TuneNeoverseE1],
                     [FeatureAES, FeatureSHA2, FeatureDotProd, FeatureFullFP16, FeatureRCPC, FeatureSSBS]>;
def : AArch64Processor<"neoverse-n1", HasV8_2aOps, NeoverseN1Model, ProcessorFeatures.NeoverseN1,
                     [TuneNeoverseN1],
                     [FeatureAES, FeatureSHA2, FeatureDotProd, FeatureFullFP16, FeatureSPE, FeatureRCPC, FeatureSSBS]>;
def : AArch64Processor<"neoverse-n2", HasV9_0aOps, NeoverseN2Model, ProcessorFeatures.NeoverseN2,
                     [TuneNeoverseN2],
                     [FeatureBF16, FeatureDotProd, FeatureFullFP16, FeatureFP16FML, FeatureMatMulInt8, FeatureMTE, FeatureSB, FeatureSSBS, FeatureSVE, FeatureSVE2, FeatureSVE2BitPerm]>;
def : AArch64Processor<"neoverse-n3", HasV9_2aOps, NeoverseN2Model, ProcessorFeatures.NeoverseN3,
                     [TuneNeoverseN3],
                     [FeatureMTE, FeatureSSBS, FeatureSB, FeaturePredRes, FeatureFP16FML, FeaturePAuth, FeatureFlagM, FeaturePerfMon, FeatureRandGen, FeatureSVE2BitPerm, FeatureSPE, FeaturePerfMon]>;
def : AArch64Processor<"neoverse-512tvb", HasV8_4aOps, NeoverseV1Model, ProcessorFeatures.Neoverse512TVB,
                     [TuneNeoverse512TVB],
                     [FeatureAES, FeatureSHA2, FeatureSHA3, FeatureSM4, FeatureSVE, FeatureSSBS, FeatureFullFP16, FeatureBF16, FeatureDotProd, FeatureSPE, FeatureRandGen, FeatureFP16FML, FeatureMatMulInt8]>;
def : AArch64Processor<"neoverse-v1", HasV8_4aOps, NeoverseV1Model, ProcessorFeatures.NeoverseV1,
                     [TuneNeoverseV1],
                     [FeatureAES, FeatureSHA2, FeatureSHA3, FeatureSM4, FeatureSVE, FeatureSSBS, FeatureFullFP16, FeatureBF16, FeatureDotProd, FeatureSPE, FeatureRandGen, FeatureFP16FML, FeatureMatMulInt8]>;
def : AArch64Processor<"neoverse-v2", HasV9_0aOps, NeoverseV2Model, ProcessorFeatures.NeoverseV2,
                     [TuneNeoverseV2],
                     [FeatureSVE, FeatureSVE2, FeatureSSBS, FeatureFullFP16, FeatureBF16, FeatureRandGen, FeatureDotProd, FeatureSPE, FeatureSVE2BitPerm, FeatureFP16FML, FeatureMatMulInt8, FeatureMTE]>;
def : AArch64Processor<"neoverse-v3", HasV9_2aOps, NeoverseV2Model, ProcessorFeatures.NeoverseV3,
                     [TuneNeoverseV3],
                     [FeatureSPE, FeatureMTE, FeatureSSBS, FeatureSB, FeaturePredRes, FeatureLS64, FeatureBRBE, FeaturePAuth, FeatureFlagM, FeaturePerfMon, FeatureRandGen, FeatureSVE2BitPerm, FeatureFP16FML]>;
def : AArch64Processor<"neoverse-v3ae", HasV9_2aOps, NeoverseV2Model, ProcessorFeatures.NeoverseV3AE,
                     [TuneNeoverseV3AE],
                     [FeatureSPE, FeatureMTE, FeatureSSBS, FeatureSB, FeaturePredRes, FeatureLS64, FeatureBRBE, FeaturePAuth, FeatureFlagM, FeaturePerfMon, FeatureRandGen, FeatureSVE2BitPerm, FeatureFP16FML]>;
def : AArch64Processor<"exynos-m3", HasV8_0aOps, ExynosM3Model, ProcessorFeatures.ExynosM3,
                     [TuneExynosM3],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;
def : AArch64Processor<"exynos-m4", HasV8_2aOps, ExynosM4Model, ProcessorFeatures.ExynosM4,
                     [TuneExynosM4],
                     [FeatureAES, FeatureSHA2, FeatureDotProd, FeatureFullFP16]>;
def : AArch64Processor<"exynos-m5", HasV8_2aOps, ExynosM5Model, ProcessorFeatures.ExynosM4,
                     [TuneExynosM4],
                     [FeatureAES, FeatureSHA2, FeatureDotProd, FeatureFullFP16]>;
def : AArch64Processor<"falkor", HasV8_0aOps, FalkorModel, ProcessorFeatures.Falkor,
                     [TuneFalkor],
                     [FeatureAES, FeatureSHA2, FeatureCRC, FeatureRDM]>;
def : AArch64Processor<"saphira", HasV8_3aOps, FalkorModel, ProcessorFeatures.Saphira,
                     [TuneSaphira],
                     [FeatureAES, FeatureSHA2, FeatureSPE]>;
def : AArch64Processor<"kryo", HasV8_0aOps, KryoModel, ProcessorFeatures.A53,
                     [TuneKryo],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;

// Cavium ThunderX/ThunderX T8X  Processors
def : AArch64Processor<"thunderx", HasV8_0aOps, ThunderXT8XModel,  ProcessorFeatures.ThunderX,
                     [TuneThunderX],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;
def : AArch64Processor<"thunderxt88", HasV8_0aOps, ThunderXT8XModel, ProcessorFeatures.ThunderX,
                     [TuneThunderXT88],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;
def : AArch64Processor<"thunderxt81", HasV8_0aOps, ThunderXT8XModel, ProcessorFeatures.ThunderX,
                     [TuneThunderXT81],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;
def : AArch64Processor<"thunderxt83", HasV8_0aOps, ThunderXT8XModel, ProcessorFeatures.ThunderX,
                     [TuneThunderXT83],
                     [FeatureAES, FeatureSHA2, FeatureCRC]>;
// Cavium ThunderX2T9X  Processors. Formerly Broadcom Vulcan.
def : AArch64Processor<"thunderx2t99", HasV8_1aOps, ThunderX2T99Model, ProcessorFeatures.ThunderX2T99,
                     [TuneThunderX2T99],
                     [FeatureAES, FeatureSHA2]>;
// Marvell ThunderX3T110 Processors.
def : AArch64Processor<"thunderx3t110", HasV8_3aOps, ThunderX3T110Model, ProcessorFeatures.ThunderX3T110,
                     [TuneThunderX3T110],
                     [FeatureAES, FeatureSHA2]>;
def : AArch64Processor<"tsv110", HasV8_2aOps, TSV110Model, ProcessorFeatures.TSV110,
                     [TuneTSV110],
                     [FeatureAES, FeatureSHA2, FeatureDotProd, FeatureFullFP16, FeatureFP16FML, FeatureSPE, FeatureJS, FeatureComplxNum]>;

// Support cyclone as an alias for apple-a7 so we can still LTO old bitcode.
def : AArch64Processor<"cyclone", HasV8_0aOps, CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7],
                     [FeatureAES, FeatureSHA2]>;

// iPhone and iPad CPUs
def : AArch64Processor<"apple-a7", HasV8_0aOps, CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7],
                     [FeatureAES, FeatureSHA2]>;
def : AArch64Processor<"apple-a8", HasV8_0aOps, CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7],
                     [FeatureAES, FeatureSHA2]>;
def : AArch64Processor<"apple-a9", HasV8_0aOps, CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7],
                     [FeatureAES, FeatureSHA2]>;
def : AArch64Processor<"apple-a10", HasV8_0aOps, CycloneModel, ProcessorFeatures.AppleA10,
                     [TuneAppleA10],
                     [FeatureAES, FeatureSHA2, FeatureCRC, FeatureRDM]>;
def : AArch64Processor<"apple-a11", HasV8_2aOps, CycloneModel, ProcessorFeatures.AppleA11,
                     [TuneAppleA11],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16]>;
def : AArch64Processor<"apple-a12", HasV8_3aOps, CycloneModel, ProcessorFeatures.AppleA12,
                     [TuneAppleA12],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16]>;
def : AArch64Processor<"apple-a13", HasV8_4aOps, CycloneModel, ProcessorFeatures.AppleA13,
                     [TuneAppleA13],
                     [FeatureAES, FeatureSHA2, FeatureSHA3, FeatureFullFP16, FeatureFP16FML]>;
def : AArch64Processor<"apple-a14", HasV8_5aOps, CycloneModel, ProcessorFeatures.AppleA14,
                     [TuneAppleA14],
                     [FeatureAES, FeatureSHA2, FeatureSHA3, FeatureFullFP16, FeatureFP16FML]>;
def : AArch64Processor<"apple-a15", HasV8_6aOps, CycloneModel, ProcessorFeatures.AppleA15,
                     [TuneAppleA15],
                     [FeatureAES, FeatureSHA2, FeatureSHA3, FeatureFullFP16, FeatureFP16FML]>;
// Alias for the latest Apple processor model supported by LLVM.
def : AArch64Processor<"apple-a16", HasV8_6aOps, CycloneModel, ProcessorFeatures.AppleA16,
                     [TuneAppleA16],
                     [FeatureAES, FeatureSHA2, FeatureSHA3, FeatureFullFP16, FeatureFP16FML]>;
def : AArch64Processor<"apple-a17", HasV8_6aOps, CycloneModel, ProcessorFeatures.AppleA17,
                     [TuneAppleA17],
                     [FeatureAES, FeatureSHA2, FeatureSHA3, FeatureFullFP16, FeatureFP16FML]>;
// Mac CPUs
def : AArch64Processor<"apple-m1", HasV8_5aOps, CycloneModel, ProcessorFeatures.AppleA14,
                     [TuneAppleA14],
                     [FeatureAES, FeatureSHA2, FeatureSHA3, FeatureFullFP16, FeatureFP16FML]>;
def : AArch64Processor<"apple-m2", HasV8_6aOps, CycloneModel, ProcessorFeatures.AppleA15,
                     [TuneAppleA15],
                     [FeatureAES, FeatureSHA2, FeatureSHA3, FeatureFullFP16, FeatureFP16FML]>;
def : AArch64Processor<"apple-m3", HasV8_6aOps, CycloneModel, ProcessorFeatures.AppleA16,
                     [TuneAppleA16],
                     [FeatureAES, FeatureSHA2, FeatureSHA3, FeatureFullFP16, FeatureFP16FML]>;

// watch CPUs.
def : AArch64Processor<"apple-s4", HasV8_3aOps, CycloneModel, ProcessorFeatures.AppleA12,
                     [TuneAppleA12],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16]>;
def : AArch64Processor<"apple-s5", HasV8_3aOps, CycloneModel, ProcessorFeatures.AppleA12,
                     [TuneAppleA12],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16]>;

// Fujitsu A64FX
def : AArch64Processor<"a64fx", HasV8_2aOps, A64FXModel, ProcessorFeatures.A64FX,
                     [TuneA64FX],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16, FeatureSVE]>;

// Nvidia Carmel
def : AArch64Processor<"carmel", HasV8_2aOps, NoSchedModel, ProcessorFeatures.Carmel,
                     [TuneCarmel],
                     [FeatureAES, FeatureSHA2, FeatureFullFP16]>;

// Ampere Computing
def : AArch64Processor<"ampere1", HasV8_6aOps, Ampere1Model, ProcessorFeatures.Ampere1,
                     [TuneAmpere1],
                     [FeatureAES, FeatureSHA2, FeatureSHA3, FeatureFullFP16, FeatureSB, FeatureSSBS, FeatureRandGen]>;

def : AArch64Processor<"ampere1a", HasV8_6aOps, Ampere1Model, ProcessorFeatures.Ampere1A,
                     [TuneAmpere1A],
                     [FeatureFullFP16, FeatureRandGen, FeatureSM4, FeatureSHA3, FeatureSHA2, FeatureAES, FeatureMTE, FeatureSB, FeatureSSBS]>;

def : AArch64Processor<"ampere1b", HasV8_7aOps, Ampere1BModel, ProcessorFeatures.Ampere1B,
                     [TuneAmpere1B],
                     [FeatureFullFP16, FeatureRandGen, FeatureSM4, FeatureSHA3, FeatureSHA2, FeatureAES, FeatureMTE, FeatureSB, FeatureSSBS, FeatureCSSC]>;
