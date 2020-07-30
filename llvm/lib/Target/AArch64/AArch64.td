//=- AArch64.td - Describe the AArch64 Target Machine --------*- tablegen -*-=//
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
// Target-independent interfaces which we are implementing.
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// AArch64 Subtarget features.
//

def FeatureFPARMv8 : SubtargetFeature<"fp-armv8", "HasFPARMv8", "true",
                                       "Enable ARMv8 FP">;

def FeatureNEON : SubtargetFeature<"neon", "HasNEON", "true",
  "Enable Advanced SIMD instructions", [FeatureFPARMv8]>;

def FeatureSM4 : SubtargetFeature<
    "sm4", "HasSM4", "true",
    "Enable SM3 and SM4 support", [FeatureNEON]>;

def FeatureSHA2 : SubtargetFeature<
    "sha2", "HasSHA2", "true",
    "Enable SHA1 and SHA256 support", [FeatureNEON]>;

def FeatureSHA3 : SubtargetFeature<
    "sha3", "HasSHA3", "true",
    "Enable SHA512 and SHA3 support", [FeatureNEON, FeatureSHA2]>;

def FeatureAES : SubtargetFeature<
    "aes", "HasAES", "true",
    "Enable AES support", [FeatureNEON]>;

// Crypto has been split up and any combination is now valid (see the
// crypto definitions above). Also, crypto is now context sensitive:
// it has a different meaning for e.g. Armv8.4 than it has for Armv8.2.
// Therefore, we rely on Clang, the user interacing tool, to pass on the
// appropriate crypto options. But here in the backend, crypto has very little
// meaning anymore. We kept the Crypto definition here for backward
// compatibility, and now imply features SHA2 and AES, which was the
// "traditional" meaning of Crypto.
def FeatureCrypto : SubtargetFeature<"crypto", "HasCrypto", "true",
  "Enable cryptographic instructions", [FeatureNEON, FeatureSHA2, FeatureAES]>;

def FeatureCRC : SubtargetFeature<"crc", "HasCRC", "true",
  "Enable ARMv8 CRC-32 checksum instructions">;

def FeatureRAS : SubtargetFeature<"ras", "HasRAS", "true",
  "Enable ARMv8 Reliability, Availability and Serviceability Extensions">;

def FeatureLSE : SubtargetFeature<"lse", "HasLSE", "true",
  "Enable ARMv8.1 Large System Extension (LSE) atomic instructions">;

def FeatureRDM : SubtargetFeature<"rdm", "HasRDM", "true",
  "Enable ARMv8.1 Rounding Double Multiply Add/Subtract instructions">;

def FeaturePAN : SubtargetFeature<
    "pan", "HasPAN", "true",
    "Enables ARM v8.1 Privileged Access-Never extension">;

def FeatureLOR : SubtargetFeature<
    "lor", "HasLOR", "true",
    "Enables ARM v8.1 Limited Ordering Regions extension">;

def FeatureVH : SubtargetFeature<
    "vh", "HasVH", "true",
    "Enables ARM v8.1 Virtual Host extension">;

def FeaturePerfMon : SubtargetFeature<"perfmon", "HasPerfMon", "true",
  "Enable ARMv8 PMUv3 Performance Monitors extension">;

def FeatureFullFP16 : SubtargetFeature<"fullfp16", "HasFullFP16", "true",
  "Full FP16", [FeatureFPARMv8]>;

def FeatureFP16FML : SubtargetFeature<"fp16fml", "HasFP16FML", "true",
  "Enable FP16 FML instructions", [FeatureFullFP16]>;

def FeatureSPE : SubtargetFeature<"spe", "HasSPE", "true",
  "Enable Statistical Profiling extension">;

def FeaturePAN_RWV : SubtargetFeature<
    "pan-rwv", "HasPAN_RWV", "true",
    "Enable v8.2 PAN s1e1R and s1e1W Variants",
    [FeaturePAN]>;

// UAO PState
def FeaturePsUAO : SubtargetFeature< "uaops", "HasPsUAO", "true",
    "Enable v8.2 UAO PState">;

def FeatureCCPP : SubtargetFeature<"ccpp", "HasCCPP",
    "true", "Enable v8.2 data Cache Clean to Point of Persistence" >;

def FeatureSVE : SubtargetFeature<"sve", "HasSVE", "true",
  "Enable Scalable Vector Extension (SVE) instructions", [FeatureFullFP16]>;

// This flag is currently still labeled as Experimental, but when fully
// implemented this should tell the compiler to use the zeroing pseudos to
// benefit from the reverse instructions (e.g. SUB vs SUBR) if the inactive
// lanes are known to be zero. The pseudos will then be expanded using the
// MOVPRFX instruction to zero the inactive lanes. This feature should only be
// enabled if MOVPRFX instructions are known to merge with the destructive
// operations they prefix.
//
// This feature could similarly be extended to support cheap merging of _any_
// value into the inactive lanes using the MOVPRFX instruction that uses
// merging-predication.
def FeatureExperimentalZeroingPseudos
    : SubtargetFeature<"use-experimental-zeroing-pseudos",
                       "UseExperimentalZeroingPseudos", "true",
                       "Hint to the compiler that the MOVPRFX instruction is "
                       "merged with destructive operations",
                       []>;

def FeatureSVE2 : SubtargetFeature<"sve2", "HasSVE2", "true",
  "Enable Scalable Vector Extension 2 (SVE2) instructions", [FeatureSVE]>;

def FeatureSVE2AES : SubtargetFeature<"sve2-aes", "HasSVE2AES", "true",
  "Enable AES SVE2 instructions", [FeatureSVE2, FeatureAES]>;

def FeatureSVE2SM4 : SubtargetFeature<"sve2-sm4", "HasSVE2SM4", "true",
  "Enable SM4 SVE2 instructions", [FeatureSVE2, FeatureSM4]>;

def FeatureSVE2SHA3 : SubtargetFeature<"sve2-sha3", "HasSVE2SHA3", "true",
  "Enable SHA3 SVE2 instructions", [FeatureSVE2, FeatureSHA3]>;

def FeatureSVE2BitPerm : SubtargetFeature<"sve2-bitperm", "HasSVE2BitPerm", "true",
  "Enable bit permutation SVE2 instructions", [FeatureSVE2]>;

def FeatureZCRegMove : SubtargetFeature<"zcm", "HasZeroCycleRegMove", "true",
                                        "Has zero-cycle register moves">;

def FeatureZCZeroingGP : SubtargetFeature<"zcz-gp", "HasZeroCycleZeroingGP", "true",
                                        "Has zero-cycle zeroing instructions for generic registers">;

def FeatureZCZeroingFP : SubtargetFeature<"zcz-fp", "HasZeroCycleZeroingFP", "true",
                                        "Has zero-cycle zeroing instructions for FP registers">;

def FeatureZCZeroing : SubtargetFeature<"zcz", "HasZeroCycleZeroing", "true",
                                        "Has zero-cycle zeroing instructions",
                                        [FeatureZCZeroingGP, FeatureZCZeroingFP]>;

/// ... but the floating-point version doesn't quite work in rare cases on older
/// CPUs.
def FeatureZCZeroingFPWorkaround : SubtargetFeature<"zcz-fp-workaround",
    "HasZeroCycleZeroingFPWorkaround", "true",
    "The zero-cycle floating-point zeroing instruction has a bug">;

def FeatureStrictAlign : SubtargetFeature<"strict-align",
                                          "StrictAlign", "true",
                                          "Disallow all unaligned memory "
                                          "access">;

foreach i = {1-7,9-15,18,20-28,30} in
    def FeatureReserveX#i : SubtargetFeature<"reserve-x"#i, "ReserveXRegister["#i#"]", "true",
                                             "Reserve X"#i#", making it unavailable "
                                             "as a GPR">;

foreach i = {8-15,18} in
    def FeatureCallSavedX#i : SubtargetFeature<"call-saved-x"#i,
         "CustomCallSavedXRegs["#i#"]", "true", "Make X"#i#" callee saved.">;

def FeatureUseAA : SubtargetFeature<"use-aa", "UseAA", "true",
                                    "Use alias analysis during codegen">;

def FeatureBalanceFPOps : SubtargetFeature<"balance-fp-ops", "BalanceFPOps",
    "true",
    "balance mix of odd and even D-registers for fp multiply(-accumulate) ops">;

def FeaturePredictableSelectIsExpensive : SubtargetFeature<
    "predictable-select-expensive", "PredictableSelectIsExpensive", "true",
    "Prefer likely predicted branches over selects">;

def FeatureCustomCheapAsMoveHandling : SubtargetFeature<"custom-cheap-as-move",
    "CustomAsCheapAsMove", "true",
    "Use custom handling of cheap instructions">;

def FeatureExynosCheapAsMoveHandling : SubtargetFeature<"exynos-cheap-as-move",
    "ExynosAsCheapAsMove", "true",
    "Use Exynos specific handling of cheap instructions",
    [FeatureCustomCheapAsMoveHandling]>;

def FeaturePostRAScheduler : SubtargetFeature<"use-postra-scheduler",
    "UsePostRAScheduler", "true", "Schedule again after register allocation">;

def FeatureSlowMisaligned128Store : SubtargetFeature<"slow-misaligned-128store",
    "Misaligned128StoreIsSlow", "true", "Misaligned 128 bit stores are slow">;

def FeatureSlowPaired128 : SubtargetFeature<"slow-paired-128",
    "Paired128IsSlow", "true", "Paired 128 bit loads and stores are slow">;

def FeatureSlowSTRQro : SubtargetFeature<"slow-strqro-store", "STRQroIsSlow",
    "true", "STR of Q register with register offset is slow">;

def FeatureAlternateSExtLoadCVTF32Pattern : SubtargetFeature<
    "alternate-sextload-cvt-f32-pattern", "UseAlternateSExtLoadCVTF32Pattern",
    "true", "Use alternative pattern for sextload convert to f32">;

def FeatureArithmeticBccFusion : SubtargetFeature<
    "arith-bcc-fusion", "HasArithmeticBccFusion", "true",
    "CPU fuses arithmetic+bcc operations">;

def FeatureArithmeticCbzFusion : SubtargetFeature<
    "arith-cbz-fusion", "HasArithmeticCbzFusion", "true",
    "CPU fuses arithmetic + cbz/cbnz operations">;

def FeatureFuseAddress : SubtargetFeature<
    "fuse-address", "HasFuseAddress", "true",
    "CPU fuses address generation and memory operations">;

def FeatureFuseAES : SubtargetFeature<
    "fuse-aes", "HasFuseAES", "true",
    "CPU fuses AES crypto operations">;

def FeatureFuseArithmeticLogic : SubtargetFeature<
    "fuse-arith-logic", "HasFuseArithmeticLogic", "true",
    "CPU fuses arithmetic and logic operations">;

def FeatureFuseCCSelect : SubtargetFeature<
    "fuse-csel", "HasFuseCCSelect", "true",
    "CPU fuses conditional select operations">;

def FeatureFuseCryptoEOR : SubtargetFeature<
    "fuse-crypto-eor", "HasFuseCryptoEOR", "true",
    "CPU fuses AES/PMULL and EOR operations">;

def FeatureFuseLiterals : SubtargetFeature<
    "fuse-literals", "HasFuseLiterals", "true",
    "CPU fuses literal generation operations">;

def FeatureDisableLatencySchedHeuristic : SubtargetFeature<
    "disable-latency-sched-heuristic", "DisableLatencySchedHeuristic", "true",
    "Disable latency scheduling heuristic">;

def FeatureForce32BitJumpTables
   : SubtargetFeature<"force-32bit-jump-tables", "Force32BitJumpTables", "true",
                      "Force jump table entries to be 32-bits wide except at MinSize">;

def FeatureRCPC : SubtargetFeature<"rcpc", "HasRCPC", "true",
                                   "Enable support for RCPC extension">;

def FeatureUseRSqrt : SubtargetFeature<
    "use-reciprocal-square-root", "UseRSqrt", "true",
    "Use the reciprocal square root approximation">;

def FeatureDotProd : SubtargetFeature<
    "dotprod", "HasDotProd", "true",
    "Enable dot product support">;

def FeaturePA : SubtargetFeature<
    "pa", "HasPA", "true",
    "Enable v8.3-A Pointer Authentication extension">;

def FeatureJS : SubtargetFeature<
    "jsconv", "HasJS", "true",
    "Enable v8.3-A JavaScript FP conversion instructions",
    [FeatureFPARMv8]>;

def FeatureCCIDX : SubtargetFeature<
    "ccidx", "HasCCIDX", "true",
    "Enable v8.3-A Extend of the CCSIDR number of sets">;

def FeatureComplxNum : SubtargetFeature<
    "complxnum", "HasComplxNum", "true",
    "Enable v8.3-A Floating-point complex number support",
    [FeatureNEON]>;

def FeatureNV : SubtargetFeature<
    "nv", "HasNV", "true",
    "Enable v8.4-A Nested Virtualization Enchancement">;

def FeatureRASv8_4 : SubtargetFeature<
    "rasv8_4", "HasRASv8_4", "true",
    "Enable v8.4-A Reliability, Availability and Serviceability extension",
    [FeatureRAS]>;

def FeatureMPAM : SubtargetFeature<
    "mpam", "HasMPAM", "true",
    "Enable v8.4-A Memory system Partitioning and Monitoring extension">;

def FeatureDIT : SubtargetFeature<
    "dit", "HasDIT", "true",
    "Enable v8.4-A Data Independent Timing instructions">;

def FeatureTRACEV8_4 : SubtargetFeature<
    "tracev8.4", "HasTRACEV8_4", "true",
    "Enable v8.4-A Trace extension">;

def FeatureAM : SubtargetFeature<
    "am", "HasAM", "true",
    "Enable v8.4-A Activity Monitors extension">;

def FeatureAMVS : SubtargetFeature<
    "amvs", "HasAMVS", "true",
    "Enable v8.6-A Activity Monitors Virtualization support",
    [FeatureAM]>;

def FeatureSEL2 : SubtargetFeature<
    "sel2", "HasSEL2", "true",
    "Enable v8.4-A Secure Exception Level 2 extension">;

def FeaturePMU : SubtargetFeature<
    "pmu", "HasPMU", "true",
    "Enable v8.4-A PMU extension">;

def FeatureTLB_RMI : SubtargetFeature<
    "tlb-rmi", "HasTLB_RMI", "true",
    "Enable v8.4-A TLB Range and Maintenance Instructions">;

def FeatureFMI : SubtargetFeature<
    "fmi", "HasFMI", "true",
    "Enable v8.4-A Flag Manipulation Instructions">;

// 8.4 RCPC enchancements: LDAPR & STLR instructions with Immediate Offset
def FeatureRCPC_IMMO : SubtargetFeature<"rcpc-immo", "HasRCPC_IMMO", "true",
    "Enable v8.4-A RCPC instructions with Immediate Offsets",
    [FeatureRCPC]>;

def FeatureNoNegativeImmediates : SubtargetFeature<"no-neg-immediates",
                                        "NegativeImmediates", "false",
                                        "Convert immediates and instructions "
                                        "to their negated or complemented "
                                        "equivalent when the immediate does "
                                        "not fit in the encoding.">;

def FeatureLSLFast : SubtargetFeature<
    "lsl-fast", "HasLSLFast", "true",
    "CPU has a fastpath logical shift of up to 3 places">;

def FeatureAggressiveFMA :
  SubtargetFeature<"aggressive-fma",
                   "HasAggressiveFMA",
                   "true",
                   "Enable Aggressive FMA for floating-point.">;

def FeatureAltFPCmp : SubtargetFeature<"altnzcv", "HasAlternativeNZCV", "true",
  "Enable alternative NZCV format for floating point comparisons">;

def FeatureFRInt3264 : SubtargetFeature<"fptoint", "HasFRInt3264", "true",
  "Enable FRInt[32|64][Z|X] instructions that round a floating-point number to "
  "an integer (in FP format) forcing it to fit into a 32- or 64-bit int" >;

def FeatureSpecRestrict : SubtargetFeature<"specrestrict", "HasSpecRestrict",
  "true", "Enable architectural speculation restriction" >;

def FeatureSB : SubtargetFeature<"sb", "HasSB",
  "true", "Enable v8.5 Speculation Barrier" >;

def FeatureSSBS : SubtargetFeature<"ssbs", "HasSSBS",
  "true", "Enable Speculative Store Bypass Safe bit" >;

def FeaturePredRes : SubtargetFeature<"predres", "HasPredRes", "true",
  "Enable v8.5a execution and data prediction invalidation instructions" >;

def FeatureCacheDeepPersist : SubtargetFeature<"ccdp", "HasCCDP",
    "true", "Enable v8.5 Cache Clean to Point of Deep Persistence" >;

def FeatureBranchTargetId : SubtargetFeature<"bti", "HasBTI",
    "true", "Enable Branch Target Identification" >;

def FeatureRandGen : SubtargetFeature<"rand", "HasRandGen",
    "true", "Enable Random Number generation instructions" >;

def FeatureMTE : SubtargetFeature<"mte", "HasMTE",
    "true", "Enable Memory Tagging Extension" >;

def FeatureTRBE : SubtargetFeature<"trbe", "HasTRBE",
    "true", "Enable Trace Buffer Extension">;

def FeatureETE : SubtargetFeature<"ete", "HasETE",
    "true", "Enable Embedded Trace Extension",
    [FeatureTRBE]>;

def FeatureTME : SubtargetFeature<"tme", "HasTME",
    "true", "Enable Transactional Memory Extension" >;

def FeatureTaggedGlobals : SubtargetFeature<"tagged-globals",
    "AllowTaggedGlobals",
    "true", "Use an instruction sequence for taking the address of a global "
    "that allows a memory tag in the upper address bits">;

def FeatureBF16 : SubtargetFeature<"bf16", "HasBF16",
    "true", "Enable BFloat16 Extension" >;

def FeatureMatMulInt8 : SubtargetFeature<"i8mm", "HasMatMulInt8",
    "true", "Enable Matrix Multiply Int8 Extension">;

def FeatureMatMulFP32 : SubtargetFeature<"f32mm", "HasMatMulFP32",
    "true", "Enable Matrix Multiply FP32 Extension", [FeatureSVE]>;

def FeatureMatMulFP64 : SubtargetFeature<"f64mm", "HasMatMulFP64",
    "true", "Enable Matrix Multiply FP64 Extension", [FeatureSVE]>;

def FeatureFineGrainedTraps : SubtargetFeature<"fgt", "HasFineGrainedTraps",
    "true", "Enable fine grained virtualization traps extension">;

def FeatureEnhancedCounterVirtualization :
      SubtargetFeature<"ecv", "HasEnhancedCounterVirtualization",
      "true", "Enable enhanced counter virtualization extension">;

//===----------------------------------------------------------------------===//
// Architectures.
//

def HasV8_1aOps : SubtargetFeature<"v8.1a", "HasV8_1aOps", "true",
  "Support ARM v8.1a instructions", [FeatureCRC, FeatureLSE, FeatureRDM,
  FeaturePAN, FeatureLOR, FeatureVH]>;

def HasV8_2aOps : SubtargetFeature<"v8.2a", "HasV8_2aOps", "true",
  "Support ARM v8.2a instructions", [HasV8_1aOps, FeaturePsUAO,
  FeaturePAN_RWV, FeatureRAS, FeatureCCPP]>;

def HasV8_3aOps : SubtargetFeature<"v8.3a", "HasV8_3aOps", "true",
  "Support ARM v8.3a instructions", [HasV8_2aOps, FeatureRCPC, FeaturePA,
  FeatureJS, FeatureCCIDX, FeatureComplxNum]>;

def HasV8_4aOps : SubtargetFeature<"v8.4a", "HasV8_4aOps", "true",
  "Support ARM v8.4a instructions", [HasV8_3aOps, FeatureDotProd,
  FeatureNV, FeatureRASv8_4, FeatureMPAM, FeatureDIT,
  FeatureTRACEV8_4, FeatureAM, FeatureSEL2, FeaturePMU, FeatureTLB_RMI,
  FeatureFMI, FeatureRCPC_IMMO]>;

def HasV8_5aOps : SubtargetFeature<
  "v8.5a", "HasV8_5aOps", "true", "Support ARM v8.5a instructions",
  [HasV8_4aOps, FeatureAltFPCmp, FeatureFRInt3264, FeatureSpecRestrict,
   FeatureSSBS, FeatureSB, FeaturePredRes, FeatureCacheDeepPersist,
   FeatureBranchTargetId]>;

def HasV8_6aOps : SubtargetFeature<
  "v8.6a", "HasV8_6aOps", "true", "Support ARM v8.6a instructions",

  [HasV8_5aOps, FeatureAMVS, FeatureBF16, FeatureFineGrainedTraps,
   FeatureEnhancedCounterVirtualization, FeatureMatMulInt8]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "AArch64RegisterInfo.td"
include "AArch64RegisterBanks.td"
include "AArch64CallingConvention.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "AArch64Schedule.td"
include "AArch64InstrInfo.td"
include "AArch64SchedPredicates.td"
include "AArch64SchedPredExynos.td"
include "AArch64Combine.td"

def AArch64InstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Named operands for MRS/MSR/TLBI/...
//===----------------------------------------------------------------------===//

include "AArch64SystemOperands.td"

//===----------------------------------------------------------------------===//
// Access to privileged registers
//===----------------------------------------------------------------------===//

foreach i = 1-3 in
def FeatureUseEL#i#ForTP : SubtargetFeature<"tpidr-el"#i, "UseEL"#i#"ForTP",
  "true", "Permit use of TPIDR_EL"#i#" for the TLS base">;

//===----------------------------------------------------------------------===//
// Control codegen mitigation against Straight Line Speculation vulnerability.
//===----------------------------------------------------------------------===//

def FeatureHardenSlsRetBr : SubtargetFeature<"harden-sls-retbr",
  "HardenSlsRetBr", "true",
  "Harden against straight line speculation across RET and BR instructions">;
def FeatureHardenSlsBlr : SubtargetFeature<"harden-sls-blr",
  "HardenSlsBlr", "true",
  "Harden against straight line speculation across BLR instructions">;

//===----------------------------------------------------------------------===//
// AArch64 Processors supported.
//

//===----------------------------------------------------------------------===//
// Unsupported features to disable for scheduling models
//===----------------------------------------------------------------------===//

class AArch64Unsupported { list<Predicate> F; }

def SVEUnsupported : AArch64Unsupported {
  let F = [HasSVE, HasSVE2, HasSVE2AES, HasSVE2SM4, HasSVE2SHA3,
           HasSVE2BitPerm];
}

def PAUnsupported : AArch64Unsupported {
  let F = [HasPA];
}

include "AArch64SchedA53.td"
include "AArch64SchedA57.td"
include "AArch64SchedCyclone.td"
include "AArch64SchedFalkor.td"
include "AArch64SchedKryo.td"
include "AArch64SchedExynosM3.td"
include "AArch64SchedExynosM4.td"
include "AArch64SchedExynosM5.td"
include "AArch64SchedThunderX.td"
include "AArch64SchedThunderX2T99.td"
include "AArch64SchedThunderX3T110.td"

def ProcA35     : SubtargetFeature<"a35", "ARMProcFamily", "CortexA35",
                                   "Cortex-A35 ARM processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureFPARMv8,
                                   FeatureNEON,
                                   FeaturePerfMon
                                   ]>;

def ProcA53     : SubtargetFeature<"a53", "ARMProcFamily", "CortexA53",
                                   "Cortex-A53 ARM processors", [
                                   FeatureBalanceFPOps,
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureCustomCheapAsMoveHandling,
                                   FeatureFPARMv8,
                                   FeatureFuseAES,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeaturePostRAScheduler,
                                   FeatureUseAA
                                   ]>;

def ProcA55     : SubtargetFeature<"a55", "ARMProcFamily", "CortexA55",
                                   "Cortex-A55 ARM processors", [
                                   HasV8_2aOps,
                                   FeatureCrypto,
                                   FeatureFPARMv8,
                                   FeatureFuseAES,
                                   FeatureNEON,
                                   FeatureFullFP16,
                                   FeatureDotProd,
                                   FeatureRCPC,
                                   FeaturePerfMon
                                   ]>;

def ProcA57     : SubtargetFeature<"a57", "ARMProcFamily", "CortexA57",
                                   "Cortex-A57 ARM processors", [
                                   FeatureBalanceFPOps,
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureCustomCheapAsMoveHandling,
                                   FeatureFPARMv8,
                                   FeatureFuseAES,
                                   FeatureFuseLiterals,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive
                                   ]>;

def ProcA65     : SubtargetFeature<"a65", "ARMProcFamily", "CortexA65",
                                   "Cortex-A65 ARM processors", [
                                   HasV8_2aOps,
                                   FeatureCrypto,
                                   FeatureDotProd,
                                   FeatureFPARMv8,
                                   FeatureFullFP16,
                                   FeatureNEON,
                                   FeatureRAS,
                                   FeatureRCPC,
                                   FeatureSSBS,
                                   ]>;

def ProcA72     : SubtargetFeature<"a72", "ARMProcFamily", "CortexA72",
                                   "Cortex-A72 ARM processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureFPARMv8,
                                   FeatureFuseAES,
                                   FeatureNEON,
                                   FeaturePerfMon
                                   ]>;

def ProcA73     : SubtargetFeature<"a73", "ARMProcFamily", "CortexA73",
                                   "Cortex-A73 ARM processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureFPARMv8,
                                   FeatureFuseAES,
                                   FeatureNEON,
                                   FeaturePerfMon
                                   ]>;

def ProcA75     : SubtargetFeature<"a75", "ARMProcFamily", "CortexA75",
                                   "Cortex-A75 ARM processors", [
                                   HasV8_2aOps,
                                   FeatureCrypto,
                                   FeatureFPARMv8,
                                   FeatureFuseAES,
                                   FeatureNEON,
                                   FeatureFullFP16,
                                   FeatureDotProd,
                                   FeatureRCPC,
                                   FeaturePerfMon
                                   ]>;

def ProcA76     : SubtargetFeature<"a76", "ARMProcFamily", "CortexA76",
                                   "Cortex-A76 ARM processors", [
                                    HasV8_2aOps,
                                    FeatureFPARMv8,
                                    FeatureNEON,
                                    FeatureRCPC,
                                    FeatureCrypto,
                                    FeatureFullFP16,
                                    FeatureDotProd,
                                    FeatureSSBS
                                    ]>;

def ProcA77     : SubtargetFeature<"a77", "ARMProcFamily", "CortexA77",
                                   "Cortex-A77 ARM processors", [
                                    HasV8_2aOps,
                                    FeatureFPARMv8,
                                    FeatureNEON, FeatureRCPC,
                                    FeatureCrypto,
                                    FeatureFullFP16,
                                    FeatureDotProd
                                    ]>;

def ProcA78 : SubtargetFeature<"cortex-a78", "ARMProcFamily",
                               "CortexA78",
                               "Cortex-A78 ARM processors", [
                               HasV8_2aOps,
                               FeatureCrypto,
                               FeatureFPARMv8,
                               FeatureFuseAES,
                               FeatureNEON,
                               FeatureRCPC,
                               FeaturePerfMon,
                               FeaturePostRAScheduler,
                               FeatureSPE,
                               FeatureFullFP16,
                               FeatureSSBS,
                               FeatureDotProd]>;

def ProcX1 : SubtargetFeature<"cortex-x1", "ARMProcFamily", "CortexX1",
                                  "Cortex-X1 ARM processors", [
                                  HasV8_2aOps,
                                  FeatureCrypto,
                                  FeatureFPARMv8,
                                  FeatureFuseAES,
                                  FeatureNEON,
                                  FeatureRCPC,
                                  FeaturePerfMon,
                                  FeaturePostRAScheduler,
                                  FeatureSPE,
                                  FeatureFullFP16,
                                  FeatureDotProd]>;

def ProcA64FX : SubtargetFeature<"a64fx", "ARMProcFamily", "A64FX",
                                 "Fujitsu A64FX processors", [
                                  HasV8_2aOps,
                                  FeatureFPARMv8,
                                  FeatureNEON,
                                  FeatureSHA2,
                                  FeaturePerfMon,
                                  FeatureFullFP16,
                                  FeatureSVE,
                                  FeaturePostRAScheduler,
                                  FeatureComplxNum
                                  ]>;

def ProcCarmel : SubtargetFeature<"carmel", "ARMProcFamily", "Carmel",
                                  "Nvidia Carmel processors", [
                                   HasV8_2aOps,
                                   FeatureNEON,
                                   FeatureCrypto,
                                   FeatureFullFP16
                                   ]>;

// Note that cyclone does not fuse AES instructions, but newer apple chips do
// perform the fusion and cyclone is used by default when targetting apple OSes.
def ProcAppleA7 : SubtargetFeature<"apple-a7", "ARMProcFamily", "AppleA7",
                                   "Apple A7 (the CPU formerly known as Cyclone)", [
                                   FeatureAlternateSExtLoadCVTF32Pattern,
                                   FeatureArithmeticBccFusion,
                                   FeatureArithmeticCbzFusion,
                                   FeatureCrypto,
                                   FeatureDisableLatencySchedHeuristic,
                                   FeatureFPARMv8,
                                   FeatureFuseAES,
                                   FeatureFuseCryptoEOR,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeatureZCRegMove,
                                   FeatureZCZeroing,
                                   FeatureZCZeroingFPWorkaround
                                   ]>;

def ProcAppleA10 : SubtargetFeature<"apple-a10", "ARMProcFamily", "AppleA10",
                                    "Apple A10", [
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureCrypto,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFPARMv8,
                                    FeatureFuseAES,
                                    FeatureFuseCryptoEOR,
                                    FeatureNEON,
                                    FeaturePerfMon,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing,
                                    FeatureCRC,
                                    FeatureRDM,
                                    FeaturePAN,
                                    FeatureLOR,
                                    FeatureVH,
                                    ]>;

def ProcAppleA11 : SubtargetFeature<"apple-a11", "ARMProcFamily", "AppleA11",
                                    "Apple A11", [
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureCrypto,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFPARMv8,
                                    FeatureFuseAES,
                                    FeatureFuseCryptoEOR,
                                    FeatureNEON,
                                    FeaturePerfMon,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing,
                                    FeatureFullFP16,
                                    HasV8_2aOps
                                    ]>;

def ProcAppleA12 : SubtargetFeature<"apple-a12", "ARMProcFamily", "AppleA12",
                                    "Apple A12", [
                                    FeatureAlternateSExtLoadCVTF32Pattern,
                                    FeatureArithmeticBccFusion,
                                    FeatureArithmeticCbzFusion,
                                    FeatureCrypto,
                                    FeatureDisableLatencySchedHeuristic,
                                    FeatureFPARMv8,
                                    FeatureFuseAES,
                                    FeatureFuseCryptoEOR,
                                    FeatureNEON,
                                    FeaturePerfMon,
                                    FeatureZCRegMove,
                                    FeatureZCZeroing,
                                    FeatureFullFP16,
                                    HasV8_3aOps
                                    ]>;

def ProcAppleA13 : SubtargetFeature<"apple-a13", "ARMProcFamily", "AppleA13",
                                     "Apple A13", [
                                     FeatureAlternateSExtLoadCVTF32Pattern,
                                     FeatureArithmeticBccFusion,
                                     FeatureArithmeticCbzFusion,
                                     FeatureCrypto,
                                     FeatureDisableLatencySchedHeuristic,
                                     FeatureFPARMv8,
                                     FeatureFuseAES,
                                     FeatureFuseCryptoEOR,
                                     FeatureNEON,
                                     FeaturePerfMon,
                                     FeatureZCRegMove,
                                     FeatureZCZeroing,
                                     FeatureFullFP16,
                                     FeatureFP16FML,
                                     FeatureSHA3,
                                     HasV8_4aOps
                                     ]>;

def ProcExynosM3 : SubtargetFeature<"exynosm3", "ARMProcFamily", "ExynosM3",
                                    "Samsung Exynos-M3 processors",
                                    [FeatureCRC,
                                     FeatureCrypto,
                                     FeatureExynosCheapAsMoveHandling,
                                     FeatureForce32BitJumpTables,
                                     FeatureFuseAddress,
                                     FeatureFuseAES,
                                     FeatureFuseCCSelect,
                                     FeatureFuseLiterals,
                                     FeatureLSLFast,
                                     FeaturePerfMon,
                                     FeaturePostRAScheduler,
                                     FeaturePredictableSelectIsExpensive,
                                     FeatureZCZeroingFP]>;

def ProcExynosM4 : SubtargetFeature<"exynosm4", "ARMProcFamily", "ExynosM3",
                                    "Samsung Exynos-M4 processors",
                                    [HasV8_2aOps,
                                     FeatureArithmeticBccFusion,
                                     FeatureArithmeticCbzFusion,
                                     FeatureCrypto,
                                     FeatureDotProd,
                                     FeatureExynosCheapAsMoveHandling,
                                     FeatureForce32BitJumpTables,
                                     FeatureFullFP16,
                                     FeatureFuseAddress,
                                     FeatureFuseAES,
                                     FeatureFuseArithmeticLogic,
                                     FeatureFuseCCSelect,
                                     FeatureFuseLiterals,
                                     FeatureLSLFast,
                                     FeaturePerfMon,
                                     FeaturePostRAScheduler,
                                     FeatureZCZeroing]>;

def ProcKryo    : SubtargetFeature<"kryo", "ARMProcFamily", "Kryo",
                                   "Qualcomm Kryo processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureCustomCheapAsMoveHandling,
                                   FeatureFPARMv8,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroing,
                                   FeatureLSLFast
                                   ]>;

def ProcFalkor  : SubtargetFeature<"falkor", "ARMProcFamily", "Falkor",
                                   "Qualcomm Falkor processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureCustomCheapAsMoveHandling,
                                   FeatureFPARMv8,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureRDM,
                                   FeatureZCZeroing,
                                   FeatureLSLFast,
                                   FeatureSlowSTRQro
                                   ]>;

def ProcNeoverseE1 : SubtargetFeature<"neoversee1", "ARMProcFamily",
                                      "NeoverseE1",
                                      "Neoverse E1 ARM processors", [
                                      HasV8_2aOps,
                                      FeatureCrypto,
                                      FeatureDotProd,
                                      FeatureFPARMv8,
                                      FeatureFullFP16,
                                      FeatureNEON,
                                      FeatureRCPC,
                                      FeatureSSBS,
                                      ]>;

def ProcNeoverseN1 : SubtargetFeature<"neoversen1", "ARMProcFamily",
                                      "NeoverseN1",
                                      "Neoverse N1 ARM processors", [
                                      HasV8_2aOps,
                                      FeatureCrypto,
                                      FeatureDotProd,
                                      FeatureFPARMv8,
                                      FeatureFullFP16,
                                      FeatureNEON,
                                      FeatureRCPC,
                                      FeatureSPE,
                                      FeatureSSBS,
                                      ]>;

def ProcSaphira  : SubtargetFeature<"saphira", "ARMProcFamily", "Saphira",
                                   "Qualcomm Saphira processors", [
                                   FeatureCrypto,
                                   FeatureCustomCheapAsMoveHandling,
                                   FeatureFPARMv8,
                                   FeatureNEON,
                                   FeatureSPE,
                                   FeaturePerfMon,
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroing,
                                   FeatureLSLFast,
                                   HasV8_4aOps]>;

def ProcThunderX2T99  : SubtargetFeature<"thunderx2t99", "ARMProcFamily",
                                         "ThunderX2T99",
                                         "Cavium ThunderX2 processors", [
                                          FeatureAggressiveFMA,
                                          FeatureCRC,
                                          FeatureCrypto,
                                          FeatureFPARMv8,
                                          FeatureArithmeticBccFusion,
                                          FeatureNEON,
                                          FeaturePostRAScheduler,
                                          FeaturePredictableSelectIsExpensive,
                                          FeatureLSE,
                                          HasV8_1aOps]>;

def ProcThunderX3T110  : SubtargetFeature<"thunderx3t110", "ARMProcFamily",
                                         "ThunderX3T110",
                                         "Marvell ThunderX3 processors", [
                                          FeatureAggressiveFMA,
                                          FeatureCRC,
                                          FeatureCrypto,
                                          FeatureFPARMv8,
                                          FeatureArithmeticBccFusion,
                                          FeatureNEON,
                                          FeaturePostRAScheduler,
                                          FeaturePredictableSelectIsExpensive,
                                          FeatureLSE,
                                          FeaturePA,
                                          FeatureUseAA,
                                          FeatureBalanceFPOps,
                                          FeaturePerfMon,
                                          FeatureStrictAlign,
                                          HasV8_3aOps]>;

def ProcThunderX : SubtargetFeature<"thunderx", "ARMProcFamily", "ThunderX",
                                    "Cavium ThunderX processors", [
                                    FeatureCRC,
                                    FeatureCrypto,
                                    FeatureFPARMv8,
                                    FeaturePerfMon,
                                    FeaturePostRAScheduler,
                                    FeaturePredictableSelectIsExpensive,
                                    FeatureNEON]>;

def ProcThunderXT88 : SubtargetFeature<"thunderxt88", "ARMProcFamily",
                                       "ThunderXT88",
                                       "Cavium ThunderX processors", [
                                       FeatureCRC,
                                       FeatureCrypto,
                                       FeatureFPARMv8,
                                       FeaturePerfMon,
                                       FeaturePostRAScheduler,
                                       FeaturePredictableSelectIsExpensive,
                                       FeatureNEON]>;

def ProcThunderXT81 : SubtargetFeature<"thunderxt81", "ARMProcFamily",
                                       "ThunderXT81",
                                       "Cavium ThunderX processors", [
                                       FeatureCRC,
                                       FeatureCrypto,
                                       FeatureFPARMv8,
                                       FeaturePerfMon,
                                       FeaturePostRAScheduler,
                                       FeaturePredictableSelectIsExpensive,
                                       FeatureNEON]>;

def ProcThunderXT83 : SubtargetFeature<"thunderxt83", "ARMProcFamily",
                                       "ThunderXT83",
                                       "Cavium ThunderX processors", [
                                       FeatureCRC,
                                       FeatureCrypto,
                                       FeatureFPARMv8,
                                       FeaturePerfMon,
                                       FeaturePostRAScheduler,
                                       FeaturePredictableSelectIsExpensive,
                                       FeatureNEON]>;

def ProcTSV110 : SubtargetFeature<"tsv110", "ARMProcFamily", "TSV110",
                                  "HiSilicon TS-V110 processors", [
                                  HasV8_2aOps,
                                  FeatureCrypto,
                                  FeatureCustomCheapAsMoveHandling,
                                  FeatureFPARMv8,
                                  FeatureFuseAES,
                                  FeatureNEON,
                                  FeaturePerfMon,
                                  FeaturePostRAScheduler,
                                  FeatureSPE,
                                  FeatureFullFP16,
                                  FeatureFP16FML,
                                  FeatureDotProd]>;

def : ProcessorModel<"generic", NoSchedModel, [
                     FeatureFPARMv8,
                     FeatureFuseAES,
                     FeatureNEON,
                     FeaturePerfMon,
                     FeaturePostRAScheduler,
// ETE and TRBE are future architecture extensions. We temporarily enable them
// by default for users targeting generic AArch64, until it is decided in which
// armv8.x-a architecture revision they will end up. The extensions do not
// affect code generated by the compiler and can be used only by explicitly
// mentioning the new system register names in assembly.
                     FeatureETE
                     ]>;

def : ProcessorModel<"cortex-a35", CortexA53Model, [ProcA35]>;
def : ProcessorModel<"cortex-a34", CortexA53Model, [ProcA35]>;
def : ProcessorModel<"cortex-a53", CortexA53Model, [ProcA53]>;
def : ProcessorModel<"cortex-a55", CortexA53Model, [ProcA55]>;
def : ProcessorModel<"cortex-a57", CortexA57Model, [ProcA57]>;
def : ProcessorModel<"cortex-a65", CortexA53Model, [ProcA65]>;
def : ProcessorModel<"cortex-a65ae", CortexA53Model, [ProcA65]>;
def : ProcessorModel<"cortex-a72", CortexA57Model, [ProcA72]>;
def : ProcessorModel<"cortex-a73", CortexA57Model, [ProcA73]>;
def : ProcessorModel<"cortex-a75", CortexA57Model, [ProcA75]>;
def : ProcessorModel<"cortex-a76", CortexA57Model, [ProcA76]>;
def : ProcessorModel<"cortex-a76ae", CortexA57Model, [ProcA76]>;
def : ProcessorModel<"cortex-a77", CortexA57Model, [ProcA77]>;
def : ProcessorModel<"cortex-a78", CortexA57Model, [ProcA78]>;
def : ProcessorModel<"cortex-x1", CortexA57Model, [ProcX1]>;
def : ProcessorModel<"neoverse-e1", CortexA53Model, [ProcNeoverseE1]>;
def : ProcessorModel<"neoverse-n1", CortexA57Model, [ProcNeoverseN1]>;
def : ProcessorModel<"exynos-m3", ExynosM3Model, [ProcExynosM3]>;
def : ProcessorModel<"exynos-m4", ExynosM4Model, [ProcExynosM4]>;
def : ProcessorModel<"exynos-m5", ExynosM5Model, [ProcExynosM4]>;
def : ProcessorModel<"falkor", FalkorModel, [ProcFalkor]>;
def : ProcessorModel<"saphira", FalkorModel, [ProcSaphira]>;
def : ProcessorModel<"kryo", KryoModel, [ProcKryo]>;
// Cavium ThunderX/ThunderX T8X  Processors
def : ProcessorModel<"thunderx", ThunderXT8XModel,  [ProcThunderX]>;
def : ProcessorModel<"thunderxt88", ThunderXT8XModel,  [ProcThunderXT88]>;
def : ProcessorModel<"thunderxt81", ThunderXT8XModel,  [ProcThunderXT81]>;
def : ProcessorModel<"thunderxt83", ThunderXT8XModel,  [ProcThunderXT83]>;
// Cavium ThunderX2T9X  Processors. Formerly Broadcom Vulcan.
def : ProcessorModel<"thunderx2t99", ThunderX2T99Model, [ProcThunderX2T99]>;
// Marvell ThunderX3T110 Processors.
def : ProcessorModel<"thunderx3t110", ThunderX3T110Model, [ProcThunderX3T110]>;
// FIXME: HiSilicon TSV110 is currently modeled as a Cortex-A57.
def : ProcessorModel<"tsv110", CortexA57Model, [ProcTSV110]>;

// Support cyclone as an alias for apple-a7 so we can still LTO old bitcode.
def : ProcessorModel<"cyclone", CycloneModel, [ProcAppleA7]>;

// iPhone and iPad CPUs
def : ProcessorModel<"apple-a7", CycloneModel, [ProcAppleA7]>;
def : ProcessorModel<"apple-a8", CycloneModel, [ProcAppleA7]>;
def : ProcessorModel<"apple-a9", CycloneModel, [ProcAppleA7]>;
def : ProcessorModel<"apple-a10", CycloneModel, [ProcAppleA10]>;
def : ProcessorModel<"apple-a11", CycloneModel, [ProcAppleA11]>;
def : ProcessorModel<"apple-a12", CycloneModel, [ProcAppleA12]>;
def : ProcessorModel<"apple-a13", CycloneModel, [ProcAppleA13]>;

// watch CPUs.
def : ProcessorModel<"apple-s4", CycloneModel, [ProcAppleA12]>;
def : ProcessorModel<"apple-s5", CycloneModel, [ProcAppleA12]>;

// Alias for the latest Apple processor model supported by LLVM.
def : ProcessorModel<"apple-latest", CycloneModel, [ProcAppleA13]>;

// Fujitsu A64FX
// FIXME: Scheduling model is not implemented yet.
def : ProcessorModel<"a64fx", NoSchedModel, [ProcA64FX]>;

// Nvidia Carmel
def : ProcessorModel<"carmel", NoSchedModel, [ProcCarmel]>;

//===----------------------------------------------------------------------===//
// Assembly parser
//===----------------------------------------------------------------------===//

def GenericAsmParserVariant : AsmParserVariant {
  int Variant = 0;
  string Name = "generic";
  string BreakCharacters = ".";
  string TokenizingCharacters = "[]*!/";
}

def AppleAsmParserVariant : AsmParserVariant {
  int Variant = 1;
  string Name = "apple-neon";
  string BreakCharacters = ".";
  string TokenizingCharacters = "[]*!/";
}

//===----------------------------------------------------------------------===//
// Assembly printer
//===----------------------------------------------------------------------===//
// AArch64 Uses the MC printer for asm output, so make sure the TableGen
// AsmWriter bits get associated with the correct class.
def GenericAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  int PassSubtarget = 1;
  int Variant = 0;
  bit isMCAsmWriter = 1;
}

def AppleAsmWriter : AsmWriter {
  let AsmWriterClassName = "AppleInstPrinter";
  int PassSubtarget = 1;
  int Variant = 1;
  int isMCAsmWriter = 1;
}

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def AArch64 : Target {
  let InstructionSet = AArch64InstrInfo;
  let AssemblyParserVariants = [GenericAsmParserVariant, AppleAsmParserVariant];
  let AssemblyWriters = [GenericAsmWriter, AppleAsmWriter];
  let AllowRegisterRenaming = 1;
}

//===----------------------------------------------------------------------===//
// Pfm Counters
//===----------------------------------------------------------------------===//

include "AArch64PfmCounters.td"
