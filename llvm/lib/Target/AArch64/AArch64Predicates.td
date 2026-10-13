//=- AArch64Predicates.td - AArch64 Predicate Definitions ----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ARM Instruction Predicate Definitions.
//

class AssemblerPredicateWithAll<dag cond, string name="">
    : AssemblerPredicate<(any_of FeatureAll, cond), name>;

def HasV8_0a         : Predicate<"Subtarget->hasV8_0aOps()">,
                                 AssemblerPredicate<(all_of HasV8_0aOps), "armv8.0a">;
def HasV8_1a         : Predicate<"Subtarget->hasV8_1aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV8_1aOps), "armv8.1a">;
def HasV8_2a         : Predicate<"Subtarget->hasV8_2aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV8_2aOps), "armv8.2a">;
def HasV8_3a         : Predicate<"Subtarget->hasV8_3aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV8_3aOps), "armv8.3a">;
def HasV8_4a         : Predicate<"Subtarget->hasV8_4aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV8_4aOps), "armv8.4a">;
def HasV8_5a         : Predicate<"Subtarget->hasV8_5aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV8_5aOps), "armv8.5a">;
def HasV8_6a         : Predicate<"Subtarget->hasV8_6aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV8_6aOps), "armv8.6a">;
def HasV8_7a         : Predicate<"Subtarget->hasV8_7aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV8_7aOps), "armv8.7a">;
def HasV8_8a         : Predicate<"Subtarget->hasV8_8aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV8_8aOps), "armv8.8a">;
def HasV8_9a         : Predicate<"Subtarget->hasV8_9aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV8_9aOps), "armv8.9a">;
def HasV9_0a         : Predicate<"Subtarget->hasV9_0aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV9_0aOps), "armv9-a">;
def HasV9_1a         : Predicate<"Subtarget->hasV9_1aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV9_1aOps), "armv9.1a">;
def HasV9_2a         : Predicate<"Subtarget->hasV9_2aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV9_2aOps), "armv9.2a">;
def HasV9_3a         : Predicate<"Subtarget->hasV9_3aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV9_3aOps), "armv9.3a">;
def HasV9_4a         : Predicate<"Subtarget->hasV9_4aOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV9_4aOps), "armv9.4a">;
def HasV8_0r         : Predicate<"Subtarget->hasV8_0rOps()">,
                                 AssemblerPredicateWithAll<(all_of HasV8_0rOps), "armv8-r">;
def HasEL2VMSA       : Predicate<"Subtarget->hasEL2VMSA()">,
                                 AssemblerPredicateWithAll<(all_of FeatureEL2VMSA), "el2vmsa">;
def HasEL3           : Predicate<"Subtarget->hasEL3()">,
                                 AssemblerPredicateWithAll<(all_of FeatureEL3), "el3">;
def HasVH            : Predicate<"Subtarget->hasVH()">,
                                 AssemblerPredicateWithAll<(all_of FeatureVH), "vh">;
def HasLOR           : Predicate<"Subtarget->hasLOR()">,
                                 AssemblerPredicateWithAll<(all_of FeatureLOR), "lor">;
def HasPAuth         : Predicate<"Subtarget->hasPAuth()">,
                                 AssemblerPredicateWithAll<(all_of FeaturePAuth), "pauth">;
def HasPAuthLR       : Predicate<"Subtarget->hasPAuthLR()">,
                                 AssemblerPredicateWithAll<(all_of FeaturePAuthLR), "pauth-lr">;
def HasJS            : Predicate<"Subtarget->hasJS()">,
                                 AssemblerPredicateWithAll<(all_of FeatureJS), "jsconv">;
def HasCCIDX         : Predicate<"Subtarget->hasCCIDX()">,
                                 AssemblerPredicateWithAll<(all_of FeatureCCIDX), "ccidx">;
def HasComplxNum     : Predicate<"Subtarget->hasComplxNum()">,
                                 AssemblerPredicateWithAll<(all_of FeatureComplxNum), "fcma">;
def HasNV            : Predicate<"Subtarget->hasNV()">,
                                 AssemblerPredicateWithAll<(all_of FeatureNV), "nv">;
def HasMPAM          : Predicate<"Subtarget->hasMPAM()">,
                                 AssemblerPredicateWithAll<(all_of FeatureMPAM), "mpam">;
def HasDIT           : Predicate<"Subtarget->hasDIT()">,
                                 AssemblerPredicateWithAll<(all_of FeatureDIT), "dit">;
def HasTRACEV8_4     : Predicate<"Subtarget->hasTRACEV8_4()">,
                                 AssemblerPredicateWithAll<(all_of FeatureTRACEV8_4), "tracev8.4">;
def HasAM            : Predicate<"Subtarget->hasAM()">,
                                 AssemblerPredicateWithAll<(all_of FeatureAM), "am">;
def HasSEL2          : Predicate<"Subtarget->hasSEL2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSEL2), "sel2">;
def HasTLB_RMI       : Predicate<"Subtarget->hasTLB_RMI()">,
                                 AssemblerPredicateWithAll<(all_of FeatureTLB_RMI), "tlb-rmi">;
def HasFlagM         : Predicate<"Subtarget->hasFlagM()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFlagM), "flagm">;
def HasRCPC_IMMO     : Predicate<"Subtarget->hasRCPC_IMMO()">,
                                 AssemblerPredicateWithAll<(all_of FeatureRCPC_IMMO), "rcpc-immo">;
def HasFPARMv8       : Predicate<"Subtarget->hasFPARMv8()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFPARMv8), "fp-armv8">;
def HasNEON          : Predicate<"Subtarget->isNeonAvailable()">,
                                 AssemblerPredicateWithAll<(all_of FeatureNEON), "neon">;
def HasSM4           : Predicate<"Subtarget->hasSM4()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSM4), "sm4">;
def HasSHA3          : Predicate<"Subtarget->hasSHA3()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSHA3), "sha3">;
def HasSHA2          : Predicate<"Subtarget->hasSHA2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSHA2), "sha2">;
def HasAES           : Predicate<"Subtarget->hasAES()">,
                                 AssemblerPredicateWithAll<(all_of FeatureAES), "aes">;
def HasDotProd       : Predicate<"Subtarget->hasDotProd()">,
                                 AssemblerPredicateWithAll<(all_of FeatureDotProd), "dotprod">;
def HasCRC           : Predicate<"Subtarget->hasCRC()">,
                                 AssemblerPredicateWithAll<(all_of FeatureCRC), "crc">;
def HasCSSC          : Predicate<"Subtarget->hasCSSC()">,
                                 AssemblerPredicateWithAll<(all_of FeatureCSSC), "cssc">;
def HasNoCSSC        : Predicate<"!Subtarget->hasCSSC()">;
def HasLSE           : Predicate<"Subtarget->hasLSE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureLSE), "lse">;
def HasNoLSE         : Predicate<"!Subtarget->hasLSE()">;
def HasRAS           : Predicate<"Subtarget->hasRAS()">,
                                 AssemblerPredicateWithAll<(all_of FeatureRAS), "ras">;
def HasRDM           : Predicate<"Subtarget->hasRDM()">,
                                 AssemblerPredicateWithAll<(all_of FeatureRDM), "rdm">;
def HasFullFP16      : Predicate<"Subtarget->hasFullFP16()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFullFP16), "fullfp16">;
def HasNoFullFP16    : Predicate<"!Subtarget->hasFullFP16()">;
def HasFP16FML       : Predicate<"Subtarget->hasFP16FML()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFP16FML), "fp16fml">;
def HasSPE           : Predicate<"Subtarget->hasSPE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSPE), "spe">;
def HasFuseAES       : Predicate<"Subtarget->hasFuseAES()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFuseAES),
                                 "fuse-aes">;
def HasSVE           : Predicate<"Subtarget->isSVEAvailable()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVE), "sve">;
def HasSVEB16B16     : Predicate<"Subtarget->hasSVEB16B16()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVEB16B16), "sve-b16b16">;
def HasSVE2          : Predicate<"Subtarget->isSVEAvailable() && Subtarget->hasSVE2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVE2), "sve2">;
def HasSVE2p1        : Predicate<"Subtarget->isSVEAvailable() && Subtarget->hasSVE2p1()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVE2p1), "sve2p1">;
def HasSVEAES        : Predicate<"Subtarget->hasSVEAES()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVEAES), "sve-aes">;
def HasSVESM4        : Predicate<"Subtarget->isSVEAvailable() && Subtarget->hasSVESM4()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVESM4), "sve-sm4">;
def HasSVESHA3       : Predicate<"Subtarget->hasSVESHA3()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVESHA3), "sve-sha3">;
def HasSVEBitPerm    : Predicate<"Subtarget->hasSVEBitPerm()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVEBitPerm), "sve-bitperm">;
def HasSMEandIsNonStreamingSafe
                     : Predicate<"Subtarget->hasSME()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSME), "sme">;
def HasSME           : Predicate<"Subtarget->isStreaming() && Subtarget->hasSME()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSME), "sme">;
def HasSMEF64F64     : Predicate<"Subtarget->isStreaming() && Subtarget->hasSMEF64F64()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSMEF64F64), "sme-f64f64">;
def HasSMEF16F16     : Predicate<"Subtarget->isStreaming() && Subtarget->hasSMEF16F16()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSMEF16F16), "sme-f16f16">;
def HasSMEFA64       : Predicate<"Subtarget->isStreaming() && Subtarget->hasSMEFA64()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSMEFA64), "sme-fa64">;
def HasSMEI16I64     : Predicate<"Subtarget->isStreaming() && Subtarget->hasSMEI16I64()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSMEI16I64), "sme-i16i64">;
def HasSMEB16B16     : Predicate<"Subtarget->isStreaming() && Subtarget->hasSMEB16B16()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSMEB16B16), "sme-b16b16">;
def HasSME2andIsNonStreamingSafe
                     : Predicate<"Subtarget->hasSME2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSME2), "sme2">;
def HasSME2          : Predicate<"Subtarget->isStreaming() && Subtarget->hasSME2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSME2), "sme2">;
def HasSME2p1        : Predicate<"Subtarget->isStreaming() && Subtarget->hasSME2p1()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSME2p1), "sme2p1">;
def HasFP8           : Predicate<"Subtarget->hasFP8()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFP8), "fp8">;
def HasFAMINMAX      : Predicate<"Subtarget->hasFAMINMAX()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFAMINMAX), "faminmax">;
def HasFP8FMA        : Predicate<"Subtarget->hasFP8FMA()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFP8FMA), "fp8fma">;
def HasSSVE_FP8FMA   : Predicate<"Subtarget->hasSSVE_FP8FMA() || "
                                 "(Subtarget->hasSVE2() && Subtarget->hasFP8FMA())">,
                                 AssemblerPredicateWithAll<(any_of FeatureSSVE_FP8FMA,
                                                           (all_of FeatureSVE2, FeatureFP8FMA)),
                                                           "ssve-fp8fma or (sve2 and fp8fma)">;
def HasFP8DOT2       : Predicate<"Subtarget->hasFP8DOT2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFP8DOT2), "fp8dot2">;
def HasSSVE_FP8DOT2  : Predicate<"Subtarget->hasSSVE_FP8DOT2() || "
                                 "(Subtarget->hasSVE2() && Subtarget->hasFP8DOT2())">,
                                 AssemblerPredicateWithAll<(any_of FeatureSSVE_FP8DOT2,
                                                           (all_of FeatureSVE2, FeatureFP8DOT2)),
                                 "ssve-fp8dot2 or (sve2 and fp8dot2)">;
def HasFP8DOT4       : Predicate<"Subtarget->hasFP8DOT4()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFP8DOT4), "fp8dot4">;
def HasSSVE_FP8DOT4  : Predicate<"Subtarget->hasSSVE_FP8DOT4() || "
                                 "(Subtarget->hasSVE2() && Subtarget->hasFP8DOT4())">,
                                 AssemblerPredicateWithAll<(any_of FeatureSSVE_FP8DOT4,
                                                           (all_of FeatureSVE2, FeatureFP8DOT4)),
                                 "ssve-fp8dot4 or (sve2 and fp8dot4)">;
def HasLUT           : Predicate<"Subtarget->hasLUT()">,
                                 AssemblerPredicateWithAll<(all_of FeatureLUT), "lut">;
def HasSME_LUTv2     : Predicate<"Subtarget->isStreaming() && Subtarget->hasSME_LUTv2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSME_LUTv2), "sme-lutv2">;
def HasSMEF8F16      : Predicate<"Subtarget->isStreaming() && Subtarget->hasSMEF8F16()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSMEF8F16), "sme-f8f16">;
def HasSMEF8F32      : Predicate<"Subtarget->isStreaming() && Subtarget->hasSMEF8F32()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSMEF8F32), "sme-f8f32">;
def HasSME_MOP4      : Predicate<"(Subtarget->isStreaming() && Subtarget->hasSME_MOP4())">,
                                 AssemblerPredicateWithAll<(all_of FeatureSME_MOP4), "sme-mop4">;
def HasSME_TMOP      : Predicate<"(Subtarget->isStreaming() && Subtarget->hasSME_TMOP())">,
                                 AssemblerPredicateWithAll<(all_of FeatureSME_TMOP), "sme-tmop">;
def HasCMPBR         : Predicate<"Subtarget->hasCMPBR()">,
                                 AssemblerPredicateWithAll<(all_of FeatureCMPBR), "cmpbr">;
def HasF8F32MM       : Predicate<"Subtarget->hasF8F32MM()">,
                                 AssemblerPredicateWithAll<(all_of FeatureF8F32MM), "f8f32mm">;
def HasF8F16MM       : Predicate<"Subtarget->hasF8F16MM()">,
                                 AssemblerPredicateWithAll<(all_of FeatureF8F16MM), "f8f16mm">;
def HasFPRCVT        : Predicate<"Subtarget->hasFPRCVT()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFPRCVT), "fprcvt">;
def HasLSFE          : Predicate<"Subtarget->hasLSFE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureLSFE), "lsfe">;
def HasSME2p2        : Predicate<"Subtarget->isStreaming() && Subtarget->hasSME2p2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSME2p2), "sme2p2">;
def HasSVEAES2       : Predicate<"Subtarget->hasSVEAES2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVEAES2), "sve-aes2">;
def HasSVEBFSCALE    : Predicate<"Subtarget->isNonStreamingSVEorSME2Available() && Subtarget->hasSVE_BFSCALE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVEBFSCALE), "sve-bfscale">;
def HasSVE_F16F32MM  : Predicate<"Subtarget->isSVEAvailable() && Subtarget->hasSVE_F16F32MM()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVE_F16F32MM), "sve-f16f32mm">;
def HasLSUI          : Predicate<"Subtarget->hasLSUI()">,
                                 AssemblerPredicateWithAll<(all_of FeatureLSUI), "lsui">;
def HasOCCMO         : Predicate<"Subtarget->hasOCCMO()">,
                                 AssemblerPredicateWithAll<(all_of FeatureOCCMO), "occmo">;
def HasHINTE         : Predicate<"Subtarget->hasHINTE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureHINTE), "hinte">;
def HasLSCP          : Predicate<"Subtarget->hasLSCP()">,
                                 AssemblerPredicateWithAll<(all_of FeatureLSCP), "lscp">;
def HasSVE2p2        : Predicate<"Subtarget->isSVEAvailable() && Subtarget->hasSVE2p2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVE2p2), "sve2p2">;
def HasSVE_B16MM     : Predicate<"Subtarget->isSVEAvailable() && Subtarget->hasSVE_B16MM()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVE_B16MM), "sve-b16mm">;
def HasF16MM         : Predicate<"Subtarget->hasF16MM()">,
                                 AssemblerPredicateWithAll<(all_of FeatureF16MM), "f16mm">;
def HasSVE2p3        : Predicate<"Subtarget->isSVEAvailable() && Subtarget->hasSVE2p3()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSVE2p3), "sve2p3">;
def HasSME2p3        : Predicate<"Subtarget->isStreaming() && Subtarget->hasSME2p3()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSME2p3), "sme2p3">;
def HasF16F32DOT     : Predicate<"Subtarget->hasF16F32DOT()">,
                                 AssemblerPredicateWithAll<(all_of FeatureF16F32DOT), "f16f32dot">;
def HasF16F32MM      : Predicate<"Subtarget->hasF16F32MM()">,
                                 AssemblerPredicateWithAll<(all_of FeatureF16F32MM), "f16f32mm">;
def HasCFLT          : Predicate<"Subtarget->hasCFLT()">,
                                 AssemblerPredicateWithAll<(all_of FeatureCFLT), "cflt">;
def HasLSC64B        : Predicate<"Subtarget->hasLSC64B()">,
                                 AssemblerPredicateWithAll<(all_of FeatureLSC64B), "lsc64b">;

// A subset of SVE(2) instructions are legal in Streaming SVE execution mode,
// they should be enabled if either has been specified.
def HasSVE_or_SME
    : Predicate<"Subtarget->isSVEorStreamingSVEAvailable()">,
                AssemblerPredicateWithAll<(any_of FeatureSVE, FeatureSME),
                "sve or sme">;
def HasNonStreamingSVE_or_SME2
    : Predicate<"Subtarget->isNonStreamingSVEorSME2Available()">,
                AssemblerPredicateWithAll<(any_of FeatureSVE, FeatureSME2),
                "sve or sme2">;
def HasNonStreamingSVE_or_SME2p1
    : Predicate<"Subtarget->isSVEAvailable() ||"
                "(Subtarget->isSVEorStreamingSVEAvailable() && Subtarget->hasSME2p1())">,
                AssemblerPredicateWithAll<(any_of FeatureSVE, FeatureSME2p1),
                "sve or sme2p1">;
def HasNonStreamingSVE_or_SME2p2
    : Predicate<"Subtarget->isSVEAvailable() ||"
                "(Subtarget->isSVEorStreamingSVEAvailable() && Subtarget->hasSME2p2())">,
                AssemblerPredicateWithAll<(any_of FeatureSVE, FeatureSME2p2),
                "sve or sme2p2">;
def HasNonStreamingSVE_or_SSVE_AES
    : Predicate<"Subtarget->isSVEAvailable() ||"
                "(Subtarget->isSVEorStreamingSVEAvailable() && Subtarget->hasSSVE_AES())">,
                AssemblerPredicateWithAll<(any_of FeatureSVE, FeatureSSVE_AES),
                "sve or ssve-aes">;
def HasNonStreamingSVE_or_SSVE_BitPerm
    : Predicate<"Subtarget->isSVEAvailable() ||"
                "(Subtarget->isSVEorStreamingSVEAvailable() && Subtarget->hasSSVE_BitPerm())">,
                AssemblerPredicateWithAll<(any_of FeatureSVE, FeatureSSVE_BitPerm),
                "sve or ssve-bitperm">;
def HasNonStreamingSVE_or_SSVE_FEXPA
    : Predicate<"Subtarget->isSVEAvailable() ||"
                "(Subtarget->isSVEorStreamingSVEAvailable() && Subtarget->hasSSVE_FEXPA())">,
                AssemblerPredicateWithAll<(any_of FeatureSVE, FeatureSSVE_FEXPA),
                "sve or ssve-fexpa">;

def HasSVE2_or_SME
    : Predicate<"Subtarget->isSVEorStreamingSVEAvailable() && (Subtarget->hasSVE2() || Subtarget->hasSME())">,
                AssemblerPredicateWithAll<(any_of FeatureSVE2, FeatureSME),
                "sve2 or sme">;
def HasNonStreamingSVE2_or_SME2
    : Predicate<"(Subtarget->isSVEAvailable() && Subtarget->hasSVE2()) ||"
                "(Subtarget->isSVEorStreamingSVEAvailable() && Subtarget->hasSME2())">,
                AssemblerPredicateWithAll<(any_of FeatureSVE2, FeatureSME2),
                "sve2 or sme2">;

def HasSVE2p1_or_SME
    : Predicate<"Subtarget->isSVEorStreamingSVEAvailable() && (Subtarget->hasSVE2p1() || Subtarget->hasSME())">,
                AssemblerPredicateWithAll<(any_of FeatureSME, FeatureSVE2p1),
                "sme or sve2p1">;
def HasSVE2p1_or_SME2
    : Predicate<"Subtarget->isSVEorStreamingSVEAvailable() && (Subtarget->hasSVE2p1() || Subtarget->hasSME2())">,
                AssemblerPredicateWithAll<(any_of FeatureSME2, FeatureSVE2p1),
                "sme2 or sve2p1">;
def HasSVE2p1_or_SME2p1
    : Predicate<"Subtarget->isSVEorStreamingSVEAvailable() && (Subtarget->hasSVE2p1() || Subtarget->hasSME2p1())">,
                AssemblerPredicateWithAll<(any_of FeatureSME2p1, FeatureSVE2p1),
                "sme2p1 or sve2p1">;
def HasSVE2p1_or_StreamingSME2
    : Predicate<"(Subtarget->isSVEorStreamingSVEAvailable() && Subtarget->hasSVE2p1()) ||"
                "(Subtarget->isStreaming() && Subtarget->hasSME2())">,
                AssemblerPredicateWithAll<(any_of FeatureSME2, FeatureSVE2p1),
                "sme2 or sve2p1">;

def HasSVE2p2_or_SME2p2
    : Predicate<"Subtarget->isSVEorStreamingSVEAvailable() && (Subtarget->hasSVE2p2() || Subtarget->hasSME2p2())">,
                AssemblerPredicateWithAll<(any_of FeatureSME2p2, FeatureSVE2p2),
                "sme2p2 or sve2p2">;
def HasSVE2p3_or_SME2p3
    : Predicate<"Subtarget->isSVEorStreamingSVEAvailable() && (Subtarget->hasSVE2p3() || Subtarget->hasSME2p3())">,
                AssemblerPredicateWithAll<(any_of FeatureSME2p3, FeatureSVE2p3),
                "sme2p3 or sve2p3">;
def HasNonStreamingSVE2p2_or_SME2p2
    : Predicate<"(Subtarget->isSVEAvailable() && Subtarget->hasSVE2p2()) ||"
                "(Subtarget->isSVEorStreamingSVEAvailable() && Subtarget->hasSME2p2())">,
                AssemblerPredicateWithAll<(any_of FeatureSVE2p2, FeatureSME2p2),
                "sme2p2 or sve2p2">;

def HasSMEF16F16_or_SMEF8F16
    : Predicate<"Subtarget->isStreaming() && (Subtarget->hasSMEF16F16() || Subtarget->hasSMEF8F16())">,
                AssemblerPredicateWithAll<(any_of FeatureSMEF16F16, FeatureSMEF8F16),
                "sme-f16f16 or sme-f8f16">;

// A subset of NEON instructions are legal in Streaming SVE execution mode,
// so don't need the additional check for 'isNeonAvailable'.
def HasNEONandIsStreamingSafe
    : Predicate<"Subtarget->hasNEON()">,
      AssemblerPredicateWithAll<(any_of FeatureNEON), "neon">;
// A subset of NEON instructions are legal in Streaming SVE mode only with +fprcvt.
def HasNEONandIsFPRCVTStreamingSafe
                     : Predicate<"Subtarget->isNeonAvailable() || (Subtarget->hasNEON() && Subtarget->hasFPRCVT())">,
                                 AssemblerPredicateWithAll<(any_of FeatureNEON), "neon">;
def HasRCPC          : Predicate<"Subtarget->hasRCPC()">,
                                 AssemblerPredicateWithAll<(all_of FeatureRCPC), "rcpc">;
def HasAltNZCV       : Predicate<"Subtarget->hasAlternativeNZCV()">,
                                 AssemblerPredicateWithAll<(all_of FeatureAltFPCmp), "altnzcv">;
def HasFRInt3264     : Predicate<"Subtarget->hasFRInt3264()">,
                                 AssemblerPredicateWithAll<(all_of FeatureFRInt3264), "frint3264">;
def HasSB            : Predicate<"Subtarget->hasSB()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSB), "sb">;
def HasPredRes       : Predicate<"Subtarget->hasPredRes()">,
                                 AssemblerPredicateWithAll<(all_of FeaturePredRes), "predres">;
def HasCCDP          : Predicate<"Subtarget->hasCCDP()">,
                                 AssemblerPredicateWithAll<(all_of FeatureCacheDeepPersist), "ccdp">;
def HasBTI           : Predicate<"Subtarget->hasBTI()">,
                                 AssemblerPredicateWithAll<(all_of FeatureBranchTargetId), "bti">;
def HasBTIE          : Predicate<"Subtarget->hasBTIE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureBTIE), "btie">;
def HasMTE           : Predicate<"Subtarget->hasMTE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureMTE), "mte">;
def HasETE           : Predicate<"Subtarget->hasETE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureETE), "ete">;
def HasTRBE          : Predicate<"Subtarget->hasTRBE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureTRBE), "trbe">;
def HasBF16          : Predicate<"Subtarget->hasBF16()">,
                                 AssemblerPredicateWithAll<(all_of FeatureBF16), "bf16">;
def HasNoBF16        : Predicate<"!Subtarget->hasBF16()">;
def HasMatMulInt8    : Predicate<"Subtarget->hasMatMulInt8()">,
                                 AssemblerPredicateWithAll<(all_of FeatureMatMulInt8), "i8mm">;
def HasMatMulFP32    : Predicate<"Subtarget->hasMatMulFP32()">,
                                 AssemblerPredicateWithAll<(all_of FeatureMatMulFP32), "f32mm">;
def HasMatMulFP64    : Predicate<"Subtarget->hasMatMulFP64()">,
                                 AssemblerPredicateWithAll<(all_of FeatureMatMulFP64), "f64mm">;
def HasXS            : Predicate<"Subtarget->hasXS()">,
                                 AssemblerPredicateWithAll<(all_of FeatureXS), "xs">;
def HasWFxT          : Predicate<"Subtarget->hasWFxT()">,
                                 AssemblerPredicateWithAll<(all_of FeatureWFxT), "wfxt">;
def HasLS64          : Predicate<"Subtarget->hasLS64()">,
                                 AssemblerPredicateWithAll<(all_of FeatureLS64), "ls64">;
def HasBRBE          : Predicate<"Subtarget->hasBRBE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureBRBE), "brbe">;
def HasSPE_EEF       : Predicate<"Subtarget->hasSPE_EEF()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSPE_EEF), "spe-eef">;
def HasHBC           : Predicate<"Subtarget->hasHBC()">,
                                 AssemblerPredicateWithAll<(all_of FeatureHBC), "hbc">;
def HasMOPS          : Predicate<"Subtarget->hasMOPS()">,
                                 AssemblerPredicateWithAll<(all_of FeatureMOPS), "mops">;
def HasCLRBHB        : Predicate<"Subtarget->hasCLRBHB()">,
                                 AssemblerPredicateWithAll<(all_of FeatureCLRBHB), "clrbhb">;
def HasSPECRES2      : Predicate<"Subtarget->hasSPECRES2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureSPECRES2), "specres2">;
def HasITE           : Predicate<"Subtarget->hasITE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureITE), "ite">;
def HasTHE           : Predicate<"Subtarget->hasTHE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureTHE), "the">;
def HasRCPC3         : Predicate<"Subtarget->hasRCPC3()">,
                                 AssemblerPredicateWithAll<(all_of FeatureRCPC3), "rcpc3">;
def HasLSE128        : Predicate<"Subtarget->hasLSE128()">,
                                 AssemblerPredicateWithAll<(all_of FeatureLSE128), "lse128">;
def HasD128          : Predicate<"Subtarget->hasD128()">,
                                 AssemblerPredicateWithAll<(all_of FeatureD128), "d128">;
def HasCHK           : Predicate<"Subtarget->hasCHK()">,
                                 AssemblerPredicateWithAll<(all_of FeatureCHK), "chk">;
def HasGCS           : Predicate<"Subtarget->hasGCS()">,
                                 AssemblerPredicateWithAll<(all_of FeatureGCS), "gcs">;
def HasCPA           : Predicate<"Subtarget->hasCPA()">,
                                 AssemblerPredicateWithAll<(all_of FeatureCPA), "cpa">;
def HasTLBID         : Predicate<"Subtarget->hasTLBID()">,
                                 AssemblerPredicateWithAll<(all_of FeatureTLBID), "tlbid">;
def HasMTETC         : Predicate<"Subtarget->hasMTETC()">,
                                 AssemblerPredicateWithAll<(all_of FeatureMTETC), "mtetc">;
def HasGCIE          : Predicate<"Subtarget->hasGCIE()">,
                                 AssemblerPredicateWithAll<(all_of FeatureGCIE), "gcie">;
def HasMOPS_GO       : Predicate<"Subtarget->hasMOPS_GO()">,
                                 AssemblerPredicateWithAll<(all_of FeatureMOPS_GO), "mops-go">;
def HasS1POE2        : Predicate<"Subtarget->hasS1POE2()">,
                                 AssemblerPredicateWithAll<(all_of FeatureS1POE2), "poe2">;
def HasTEV           : Predicate<"Subtarget->hasTEV()">,
                                 AssemblerPredicateWithAll<(all_of FeatureTEV), "tev">;
def IsLE             : Predicate<"Subtarget->isLittleEndian()">;
def IsBE             : Predicate<"!Subtarget->isLittleEndian()">;
def IsWindows        : Predicate<"Subtarget->isTargetWindows()">;
def UseExperimentalZeroingPseudos
                     : Predicate<"Subtarget->useExperimentalZeroingPseudos()">;
def UseAlternateSExtLoadCVTF32
                     : Predicate<"Subtarget->useAlternateSExtLoadCVTF32Pattern()">;

def UseNegativeImmediates
                     : Predicate<"false">,
                                 AssemblerPredicate<(all_of (not FeatureNoNegativeImmediates)),
                                 "NegativeImmediates">;

def UseScalarIncVL   : Predicate<"Subtarget->useScalarIncVL()">;

def NoUseScalarIncVL : Predicate<"!Subtarget->useScalarIncVL()">;

def HasFastIncVL     : Predicate<"!Subtarget->hasDisableFastIncVL()">;

def UseSVEFPLD1R     : Predicate<"!Subtarget->noSVEFPLD1R()">;

def UseLDAPUR        : Predicate<"!Subtarget->avoidLDAPUR()">;
