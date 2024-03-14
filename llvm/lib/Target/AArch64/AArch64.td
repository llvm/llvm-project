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

// Each SubtargetFeature which corresponds to an Arm Architecture feature should
// be annotated with the respective FEAT_ feature name from the Architecture
// Reference Manual. If a SubtargetFeature enables instructions from multiple
// Arm Architecture Features, it should list all the relevant features. Not all
// FEAT_ features have a corresponding SubtargetFeature.

def FeatureFPARMv8 : SubtargetFeature<"fp-armv8", "HasFPARMv8", "true",
                                       "Enable ARMv8 FP (FEAT_FP)">;

def FeatureNEON : SubtargetFeature<"neon", "HasNEON", "true",
  "Enable Advanced SIMD instructions (FEAT_AdvSIMD)", [FeatureFPARMv8]>;

def FeatureSM4 : SubtargetFeature<
    "sm4", "HasSM4", "true",
    "Enable SM3 and SM4 support (FEAT_SM4, FEAT_SM3)", [FeatureNEON]>;

def FeatureSHA2 : SubtargetFeature<
    "sha2", "HasSHA2", "true",
    "Enable SHA1 and SHA256 support (FEAT_SHA1, FEAT_SHA256)", [FeatureNEON]>;

def FeatureSHA3 : SubtargetFeature<
    "sha3", "HasSHA3", "true",
    "Enable SHA512 and SHA3 support (FEAT_SHA3, FEAT_SHA512)", [FeatureNEON, FeatureSHA2]>;

def FeatureAES : SubtargetFeature<
    "aes", "HasAES", "true",
    "Enable AES support (FEAT_AES, FEAT_PMULL)", [FeatureNEON]>;

// Crypto has been split up and any combination is now valid (see the
// crypto definitions above). Also, crypto is now context sensitive:
// it has a different meaning for e.g. Armv8.4 than it has for Armv8.2.
// Therefore, we rely on Clang, the user interfacing tool, to pass on the
// appropriate crypto options. But here in the backend, crypto has very little
// meaning anymore. We kept the Crypto definition here for backward
// compatibility, and now imply features SHA2 and AES, which was the
// "traditional" meaning of Crypto.
def FeatureCrypto : SubtargetFeature<"crypto", "HasCrypto", "true",
  "Enable cryptographic instructions", [FeatureNEON, FeatureSHA2, FeatureAES]>;

def FeatureCRC : SubtargetFeature<"crc", "HasCRC", "true",
  "Enable ARMv8 CRC-32 checksum instructions (FEAT_CRC32)">;

def FeatureRAS : SubtargetFeature<"ras", "HasRAS", "true",
  "Enable ARMv8 Reliability, Availability and Serviceability Extensions (FEAT_RAS, FEAT_RASv1p1)">;

def FeatureRASv2 : SubtargetFeature<"rasv2", "HasRASv2", "true",
  "Enable ARMv8.9-A Reliability, Availability and Serviceability Extensions (FEAT_RASv2)",
  [FeatureRAS]>;

def FeatureLSE : SubtargetFeature<"lse", "HasLSE", "true",
  "Enable ARMv8.1 Large System Extension (LSE) atomic instructions (FEAT_LSE)">;

def FeatureLSE2 : SubtargetFeature<"lse2", "HasLSE2", "true",
  "Enable ARMv8.4 Large System Extension 2 (LSE2) atomicity rules (FEAT_LSE2)">;

def FeatureOutlineAtomics : SubtargetFeature<"outline-atomics", "OutlineAtomics", "true",
  "Enable out of line atomics to support LSE instructions">;

def FeatureFMV : SubtargetFeature<"fmv", "HasFMV", "true",
  "Enable Function Multi Versioning support.">;

def FeatureRDM : SubtargetFeature<"rdm", "HasRDM", "true",
  "Enable ARMv8.1 Rounding Double Multiply Add/Subtract instructions (FEAT_RDM)",
  [FeatureNEON]>;

def FeaturePAN : SubtargetFeature<
    "pan", "HasPAN", "true",
    "Enables ARM v8.1 Privileged Access-Never extension (FEAT_PAN)">;

def FeatureLOR : SubtargetFeature<
    "lor", "HasLOR", "true",
    "Enables ARM v8.1 Limited Ordering Regions extension (FEAT_LOR)">;

def FeatureCONTEXTIDREL2 : SubtargetFeature<"CONTEXTIDREL2", "HasCONTEXTIDREL2",
    "true", "Enable RW operand CONTEXTIDR_EL2" >;

def FeatureVH : SubtargetFeature<"vh", "HasVH", "true",
    "Enables ARM v8.1 Virtual Host extension (FEAT_VHE)", [FeatureCONTEXTIDREL2] >;

// This SubtargetFeature is special. It controls only whether codegen will turn
// `llvm.readcyclecounter()` into an access to a PMUv3 System Register. The
// `FEAT_PMUv3*` system registers are always available for assembly/disassembly.
def FeaturePerfMon : SubtargetFeature<"perfmon", "HasPerfMon", "true",
  "Enable Code Generation for ARMv8 PMUv3 Performance Monitors extension (FEAT_PMUv3)">;

def FeatureFullFP16 : SubtargetFeature<"fullfp16", "HasFullFP16", "true",
  "Full FP16 (FEAT_FP16)", [FeatureFPARMv8]>;

def FeatureFP16FML : SubtargetFeature<"fp16fml", "HasFP16FML", "true",
  "Enable FP16 FML instructions (FEAT_FHM)", [FeatureFullFP16]>;

def FeatureSPE : SubtargetFeature<"spe", "HasSPE", "true",
  "Enable Statistical Profiling extension (FEAT_SPE)">;

def FeaturePAN_RWV : SubtargetFeature<
    "pan-rwv", "HasPAN_RWV", "true",
    "Enable v8.2 PAN s1e1R and s1e1W Variants (FEAT_PAN2)",
    [FeaturePAN]>;

// UAO PState
def FeaturePsUAO : SubtargetFeature< "uaops", "HasPsUAO", "true",
    "Enable v8.2 UAO PState (FEAT_UAO)">;

def FeatureCCPP : SubtargetFeature<"ccpp", "HasCCPP",
    "true", "Enable v8.2 data Cache Clean to Point of Persistence (FEAT_DPB)" >;

def FeatureSVE : SubtargetFeature<"sve", "HasSVE", "true",
  "Enable Scalable Vector Extension (SVE) instructions (FEAT_SVE)", [FeatureFullFP16]>;

def FeatureFPMR : SubtargetFeature<"fpmr", "HasFPMR", "true",
  "Enable FPMR Register (FEAT_FPMR)">;

def FeatureFP8 : SubtargetFeature<"fp8", "HasFP8", "true",
  "Enable FP8 instructions (FEAT_FP8)">;

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

def FeatureUseScalarIncVL : SubtargetFeature<"use-scalar-inc-vl",
  "UseScalarIncVL", "true", "Prefer inc/dec over add+cnt">;

def FeatureBF16 : SubtargetFeature<"bf16", "HasBF16",
    "true", "Enable BFloat16 Extension (FEAT_BF16)" >;

def FeatureNoSVEFPLD1R : SubtargetFeature<"no-sve-fp-ld1r",
  "NoSVEFPLD1R", "true", "Avoid using LD1RX instructions for FP">;

def FeatureSVE2 : SubtargetFeature<"sve2", "HasSVE2", "true",
  "Enable Scalable Vector Extension 2 (SVE2) instructions (FEAT_SVE2)",
  [FeatureSVE, FeatureUseScalarIncVL]>;

def FeatureSVE2AES : SubtargetFeature<"sve2-aes", "HasSVE2AES", "true",
  "Enable AES SVE2 instructions (FEAT_SVE_AES, FEAT_SVE_PMULL128)",
  [FeatureSVE2, FeatureAES]>;

def FeatureSVE2SM4 : SubtargetFeature<"sve2-sm4", "HasSVE2SM4", "true",
  "Enable SM4 SVE2 instructions (FEAT_SVE_SM4)", [FeatureSVE2, FeatureSM4]>;

def FeatureSVE2SHA3 : SubtargetFeature<"sve2-sha3", "HasSVE2SHA3", "true",
  "Enable SHA3 SVE2 instructions (FEAT_SVE_SHA3)", [FeatureSVE2, FeatureSHA3]>;

def FeatureSVE2BitPerm : SubtargetFeature<"sve2-bitperm", "HasSVE2BitPerm", "true",
  "Enable bit permutation SVE2 instructions (FEAT_SVE_BitPerm)", [FeatureSVE2]>;

def FeatureSVE2p1: SubtargetFeature<"sve2p1", "HasSVE2p1", "true",
  "Enable Scalable Vector Extension 2.1 instructions", [FeatureSVE2]>;

def FeatureB16B16 : SubtargetFeature<"b16b16", "HasB16B16", "true",
  "Enable SVE2.1 or SME2.1 non-widening BFloat16 to BFloat16 instructions (FEAT_B16B16)", [FeatureBF16]>;

def FeatureZCRegMove : SubtargetFeature<"zcm", "HasZeroCycleRegMove", "true",
                                        "Has zero-cycle register moves">;

def FeatureZCZeroingGP : SubtargetFeature<"zcz-gp", "HasZeroCycleZeroingGP", "true",
                                        "Has zero-cycle zeroing instructions for generic registers">;

// It is generally beneficial to rewrite "fmov s0, wzr" to "movi d0, #0".
// as movi is more efficient across all cores. Newer cores can eliminate
// fmovs early and there is no difference with movi, but this not true for
// all implementations.
def FeatureNoZCZeroingFP : SubtargetFeature<"no-zcz-fp", "HasZeroCycleZeroingFP", "false",
                                        "Has no zero-cycle zeroing instructions for FP registers">;

def FeatureZCZeroing : SubtargetFeature<"zcz", "HasZeroCycleZeroing", "true",
                                        "Has zero-cycle zeroing instructions",
                                        [FeatureZCZeroingGP]>;

/// ... but the floating-point version doesn't quite work in rare cases on older
/// CPUs.
def FeatureZCZeroingFPWorkaround : SubtargetFeature<"zcz-fp-workaround",
    "HasZeroCycleZeroingFPWorkaround", "true",
    "The zero-cycle floating-point zeroing instruction has a bug">;

def FeatureStrictAlign : SubtargetFeature<"strict-align",
                                          "RequiresStrictAlign", "true",
                                          "Disallow all unaligned memory "
                                          "access">;

foreach i = {1-7,9-15,18,20-28,30} in
    def FeatureReserveX#i : SubtargetFeature<"reserve-x"#i, "ReserveXRegister["#i#"]", "true",
                                             "Reserve X"#i#", making it unavailable "
                                             "as a GPR">;

foreach i = {8-15,18} in
    def FeatureCallSavedX#i : SubtargetFeature<"call-saved-x"#i,
         "CustomCallSavedXRegs["#i#"]", "true", "Make X"#i#" callee saved.">;

def FeatureBalanceFPOps : SubtargetFeature<"balance-fp-ops", "BalanceFPOps",
    "true",
    "balance mix of odd and even D-registers for fp multiply(-accumulate) ops">;

def FeaturePredictableSelectIsExpensive : SubtargetFeature<
    "predictable-select-expensive", "PredictableSelectIsExpensive", "true",
    "Prefer likely predicted branches over selects">;

def FeatureEnableSelectOptimize : SubtargetFeature<
    "enable-select-opt", "EnableSelectOptimize", "true",
    "Enable the select optimize pass for select loop heuristics">;

def FeatureExynosCheapAsMoveHandling : SubtargetFeature<"exynos-cheap-as-move",
    "HasExynosCheapAsMoveHandling", "true",
    "Use Exynos specific handling of cheap instructions">;

def FeaturePostRAScheduler : SubtargetFeature<"use-postra-scheduler",
    "UsePostRAScheduler", "true", "Schedule again after register allocation">;

def FeatureSlowMisaligned128Store : SubtargetFeature<"slow-misaligned-128store",
    "IsMisaligned128StoreSlow", "true", "Misaligned 128 bit stores are slow">;

def FeatureSlowPaired128 : SubtargetFeature<"slow-paired-128",
    "IsPaired128Slow", "true", "Paired 128 bit loads and stores are slow">;

def FeatureAscendStoreAddress : SubtargetFeature<"ascend-store-address",
    "IsStoreAddressAscend", "true",
    "Schedule vector stores by ascending address">;

def FeatureSlowSTRQro : SubtargetFeature<"slow-strqro-store", "IsSTRQroSlow",
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

def FeatureCmpBccFusion : SubtargetFeature<
    "cmp-bcc-fusion", "HasCmpBccFusion", "true",
    "CPU fuses cmp+bcc operations">;

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

def FeatureFuseAdrpAdd : SubtargetFeature<
    "fuse-adrp-add", "HasFuseAdrpAdd", "true",
    "CPU fuses adrp+add operations">;

def FeatureFuseLiterals : SubtargetFeature<
    "fuse-literals", "HasFuseLiterals", "true",
    "CPU fuses literal generation operations">;

def FeatureFuseAddSub2RegAndConstOne : SubtargetFeature<
   "fuse-addsub-2reg-const1", "HasFuseAddSub2RegAndConstOne", "true",
   "CPU fuses (a + b + 1) and (a - b - 1)">;

def FeatureDisableLatencySchedHeuristic : SubtargetFeature<
    "disable-latency-sched-heuristic", "DisableLatencySchedHeuristic", "true",
    "Disable latency scheduling heuristic">;

def FeatureStorePairSuppress : SubtargetFeature<
    "store-pair-suppress", "EnableStorePairSuppress", "true",
    "Enable Store Pair Suppression heuristics">;

def FeatureForce32BitJumpTables
   : SubtargetFeature<"force-32bit-jump-tables", "Force32BitJumpTables", "true",
                      "Force jump table entries to be 32-bits wide except at MinSize">;

def FeatureRCPC : SubtargetFeature<"rcpc", "HasRCPC", "true",
                                   "Enable support for RCPC extension (FEAT_LRCPC)">;

def FeatureUseRSqrt : SubtargetFeature<
    "use-reciprocal-square-root", "UseRSqrt", "true",
    "Use the reciprocal square root approximation">;

def FeatureDotProd : SubtargetFeature<
    "dotprod", "HasDotProd", "true",
    "Enable dot product support (FEAT_DotProd)", [FeatureNEON]>;

def FeaturePAuth : SubtargetFeature<
    "pauth", "HasPAuth", "true",
    "Enable v8.3-A Pointer Authentication extension (FEAT_PAuth)">;

def FeatureJS : SubtargetFeature<
    "jsconv", "HasJS", "true",
    "Enable v8.3-A JavaScript FP conversion instructions (FEAT_JSCVT)",
    [FeatureFPARMv8]>;

def FeatureCCIDX : SubtargetFeature<
    "ccidx", "HasCCIDX", "true",
    "Enable v8.3-A Extend of the CCSIDR number of sets (FEAT_CCIDX)">;

def FeatureComplxNum : SubtargetFeature<
    "complxnum", "HasComplxNum", "true",
    "Enable v8.3-A Floating-point complex number support (FEAT_FCMA)",
    [FeatureNEON]>;

def FeatureNV : SubtargetFeature<
    "nv", "HasNV", "true",
    "Enable v8.4-A Nested Virtualization Enchancement (FEAT_NV, FEAT_NV2)">;

def FeatureMPAM : SubtargetFeature<
    "mpam", "HasMPAM", "true",
    "Enable v8.4-A Memory system Partitioning and Monitoring extension (FEAT_MPAM)">;

def FeatureDIT : SubtargetFeature<
    "dit", "HasDIT", "true",
    "Enable v8.4-A Data Independent Timing instructions (FEAT_DIT)">;

def FeatureTRACEV8_4 : SubtargetFeature<
    "tracev8.4", "HasTRACEV8_4", "true",
    "Enable v8.4-A Trace extension (FEAT_TRF)">;

def FeatureAM : SubtargetFeature<
    "am", "HasAM", "true",
    "Enable v8.4-A Activity Monitors extension (FEAT_AMUv1)">;

def FeatureAMVS : SubtargetFeature<
    "amvs", "HasAMVS", "true",
    "Enable v8.6-A Activity Monitors Virtualization support (FEAT_AMUv1p1)",
    [FeatureAM]>;

def FeatureSEL2 : SubtargetFeature<
    "sel2", "HasSEL2", "true",
    "Enable v8.4-A Secure Exception Level 2 extension (FEAT_SEL2)">;

def FeatureTLB_RMI : SubtargetFeature<
    "tlb-rmi", "HasTLB_RMI", "true",
    "Enable v8.4-A TLB Range and Maintenance Instructions (FEAT_TLBIOS, FEAT_TLBIRANGE)">;

def FeatureFlagM : SubtargetFeature<
    "flagm", "HasFlagM", "true",
    "Enable v8.4-A Flag Manipulation Instructions (FEAT_FlagM)">;

// 8.4 RCPC enchancements: LDAPR & STLR instructions with Immediate Offset
def FeatureRCPC_IMMO : SubtargetFeature<"rcpc-immo", "HasRCPC_IMMO", "true",
    "Enable v8.4-A RCPC instructions with Immediate Offsets (FEAT_LRCPC2)",
    [FeatureRCPC]>;

def FeatureNoNegativeImmediates : SubtargetFeature<"no-neg-immediates",
                                        "NegativeImmediates", "false",
                                        "Convert immediates and instructions "
                                        "to their negated or complemented "
                                        "equivalent when the immediate does "
                                        "not fit in the encoding.">;

def FeatureAddrLSLFast : SubtargetFeature<
    "addr-lsl-fast", "HasAddrLSLFast", "true",
    "Address operands with logical shift of up to 3 places are cheap">;

def FeatureALULSLFast : SubtargetFeature<
    "alu-lsl-fast", "HasALULSLFast", "true",
    "Add/Sub operations with lsl shift <= 4 are cheap">;

def FeatureAggressiveFMA :
  SubtargetFeature<"aggressive-fma",
                   "HasAggressiveFMA",
                   "true",
                   "Enable Aggressive FMA for floating-point.">;

def FeatureAltFPCmp : SubtargetFeature<"altnzcv", "HasAlternativeNZCV", "true",
  "Enable alternative NZCV format for floating point comparisons (FEAT_FlagM2)">;

def FeatureFRInt3264 : SubtargetFeature<"fptoint", "HasFRInt3264", "true",
  "Enable FRInt[32|64][Z|X] instructions that round a floating-point number to "
  "an integer (in FP format) forcing it to fit into a 32- or 64-bit int (FEAT_FRINTTS)" >;

def FeatureSpecRestrict : SubtargetFeature<"specrestrict", "HasSpecRestrict",
  "true", "Enable architectural speculation restriction (FEAT_CSV2_2)">;

def FeatureSB : SubtargetFeature<"sb", "HasSB",
  "true", "Enable v8.5 Speculation Barrier (FEAT_SB)" >;

def FeatureSSBS : SubtargetFeature<"ssbs", "HasSSBS",
  "true", "Enable Speculative Store Bypass Safe bit (FEAT_SSBS, FEAT_SSBS2)" >;

def FeaturePredRes : SubtargetFeature<"predres", "HasPredRes", "true",
  "Enable v8.5a execution and data prediction invalidation instructions (FEAT_SPECRES)" >;

def FeatureCacheDeepPersist : SubtargetFeature<"ccdp", "HasCCDP",
    "true", "Enable v8.5 Cache Clean to Point of Deep Persistence (FEAT_DPB2)" >;

def FeatureBranchTargetId : SubtargetFeature<"bti", "HasBTI",
    "true", "Enable Branch Target Identification (FEAT_BTI)" >;

def FeatureRandGen : SubtargetFeature<"rand", "HasRandGen",
    "true", "Enable Random Number generation instructions (FEAT_RNG)" >;

def FeatureMTE : SubtargetFeature<"mte", "HasMTE",
    "true", "Enable Memory Tagging Extension (FEAT_MTE, FEAT_MTE2)" >;

def FeatureTRBE : SubtargetFeature<"trbe", "HasTRBE",
    "true", "Enable Trace Buffer Extension (FEAT_TRBE)">;

def FeatureETE : SubtargetFeature<"ete", "HasETE",
    "true", "Enable Embedded Trace Extension (FEAT_ETE)",
    [FeatureTRBE]>;

def FeatureTME : SubtargetFeature<"tme", "HasTME",
    "true", "Enable Transactional Memory Extension (FEAT_TME)" >;

def FeatureTaggedGlobals : SubtargetFeature<"tagged-globals",
    "AllowTaggedGlobals",
    "true", "Use an instruction sequence for taking the address of a global "
    "that allows a memory tag in the upper address bits">;

def FeatureMatMulInt8 : SubtargetFeature<"i8mm", "HasMatMulInt8",
    "true", "Enable Matrix Multiply Int8 Extension (FEAT_I8MM)">;

def FeatureMatMulFP32 : SubtargetFeature<"f32mm", "HasMatMulFP32",
    "true", "Enable Matrix Multiply FP32 Extension (FEAT_F32MM)", [FeatureSVE]>;

def FeatureMatMulFP64 : SubtargetFeature<"f64mm", "HasMatMulFP64",
    "true", "Enable Matrix Multiply FP64 Extension (FEAT_F64MM)", [FeatureSVE]>;

def FeatureXS : SubtargetFeature<"xs", "HasXS",
    "true", "Enable Armv8.7-A limited-TLB-maintenance instruction (FEAT_XS)">;

def FeatureWFxT : SubtargetFeature<"wfxt", "HasWFxT",
    "true", "Enable Armv8.7-A WFET and WFIT instruction (FEAT_WFxT)">;

def FeatureHCX : SubtargetFeature<
    "hcx", "HasHCX", "true", "Enable Armv8.7-A HCRX_EL2 system register (FEAT_HCX)">;

def FeatureLS64 : SubtargetFeature<"ls64", "HasLS64",
    "true", "Enable Armv8.7-A LD64B/ST64B Accelerator Extension (FEAT_LS64, FEAT_LS64_V, FEAT_LS64_ACCDATA)">;

def FeatureHBC : SubtargetFeature<"hbc", "HasHBC",
    "true", "Enable Armv8.8-A Hinted Conditional Branches Extension (FEAT_HBC)">;

def FeatureMOPS : SubtargetFeature<"mops", "HasMOPS",
    "true", "Enable Armv8.8-A memcpy and memset acceleration instructions (FEAT_MOPS)">;

def FeatureNMI : SubtargetFeature<"nmi", "HasNMI",
    "true", "Enable Armv8.8-A Non-maskable Interrupts (FEAT_NMI, FEAT_GICv3_NMI)">;

def FeatureBRBE : SubtargetFeature<"brbe", "HasBRBE",
    "true", "Enable Branch Record Buffer Extension (FEAT_BRBE)">;

def FeatureSPE_EEF : SubtargetFeature<"spe-eef", "HasSPE_EEF",
    "true", "Enable extra register in the Statistical Profiling Extension (FEAT_SPEv1p2)">;

def FeatureFineGrainedTraps : SubtargetFeature<"fgt", "HasFineGrainedTraps",
    "true", "Enable fine grained virtualization traps extension (FEAT_FGT)">;

def FeatureEnhancedCounterVirtualization :
      SubtargetFeature<"ecv", "HasEnhancedCounterVirtualization",
      "true", "Enable enhanced counter virtualization extension (FEAT_ECV)">;

def FeatureRME : SubtargetFeature<"rme", "HasRME",
    "true", "Enable Realm Management Extension (FEAT_RME)">;

def FeatureSME : SubtargetFeature<"sme", "HasSME", "true",
  "Enable Scalable Matrix Extension (SME) (FEAT_SME)", [FeatureBF16, FeatureUseScalarIncVL]>;

def FeatureSMEF64F64 : SubtargetFeature<"sme-f64f64", "HasSMEF64F64", "true",
  "Enable Scalable Matrix Extension (SME) F64F64 instructions (FEAT_SME_F64F64)", [FeatureSME]>;

def FeatureSMEI16I64 : SubtargetFeature<"sme-i16i64", "HasSMEI16I64", "true",
  "Enable Scalable Matrix Extension (SME) I16I64 instructions (FEAT_SME_I16I64)", [FeatureSME]>;

def FeatureSMEF16F16 : SubtargetFeature<"sme-f16f16", "HasSMEF16F16", "true",
  "Enable SME2.1 non-widening Float16 instructions (FEAT_SME_F16F16)", []>;

def FeatureSMEFA64 : SubtargetFeature<"sme-fa64", "HasSMEFA64", "true",
  "Enable the full A64 instruction set in streaming SVE mode (FEAT_SME_FA64)", [FeatureSME, FeatureSVE2]>;

def FeatureSME2 : SubtargetFeature<"sme2", "HasSME2", "true",
  "Enable Scalable Matrix Extension 2 (SME2) instructions", [FeatureSME]>;

def FeatureSME2p1 : SubtargetFeature<"sme2p1", "HasSME2p1", "true",
  "Enable Scalable Matrix Extension 2.1 (FEAT_SME2p1) instructions", [FeatureSME2]>;

def FeatureFAMINMAX: SubtargetFeature<"faminmax", "HasFAMINMAX", "true",
   "Enable FAMIN and FAMAX instructions (FEAT_FAMINMAX)">;

def FeatureFP8FMA : SubtargetFeature<"fp8fma", "HasFP8FMA", "true",
  "Enable fp8 multiply-add instructions (FEAT_FP8FMA)">;

def FeatureSSVE_FP8FMA : SubtargetFeature<"ssve-fp8fma", "HasSSVE_FP8FMA", "true",
  "Enable SVE2 fp8 multiply-add instructions (FEAT_SSVE_FP8FMA)", [FeatureSME2]>;

def FeatureFP8DOT2: SubtargetFeature<"fp8dot2", "HasFP8DOT2", "true",
   "Enable fp8 2-way dot instructions (FEAT_FP8DOT2)">;

def FeatureSSVE_FP8DOT2 : SubtargetFeature<"ssve-fp8dot2", "HasSSVE_FP8DOT2", "true",
  "Enable SVE2 fp8 2-way dot product instructions (FEAT_SSVE_FP8DOT2)", [FeatureSME2]>;

def FeatureFP8DOT4: SubtargetFeature<"fp8dot4", "HasFP8DOT4", "true",
   "Enable fp8 4-way dot instructions (FEAT_FP8DOT4)">;

def FeatureSSVE_FP8DOT4 : SubtargetFeature<"ssve-fp8dot4", "HasSSVE_FP8DOT4", "true",
  "Enable SVE2 fp8 4-way dot product instructions (FEAT_SSVE_FP8DOT4)", [FeatureSME2]>;
def FeatureLUT: SubtargetFeature<"lut", "HasLUT", "true",
   "Enable Lookup Table instructions (FEAT_LUT)">;

def FeatureSME_LUTv2 : SubtargetFeature<"sme-lutv2", "HasSME_LUTv2", "true",
  "Enable Scalable Matrix Extension (SME) LUTv2 instructions (FEAT_SME_LUTv2)">;

def FeatureSMEF8F16 : SubtargetFeature<"sme-f8f16", "HasSMEF8F16", "true",
  "Enable Scalable Matrix Extension (SME) F8F16 instructions(FEAT_SME_F8F16)", [FeatureSME2, FeatureFP8]>;

def FeatureSMEF8F32 : SubtargetFeature<"sme-f8f32", "HasSMEF8F32", "true",
  "Enable Scalable Matrix Extension (SME) F8F32 instructions (FEAT_SME_F8F32)", [FeatureSME2, FeatureFP8]>;

def FeatureAppleA7SysReg  : SubtargetFeature<"apple-a7-sysreg", "HasAppleA7SysReg", "true",
  "Apple A7 (the CPU formerly known as Cyclone)">;

def FeatureEL2VMSA : SubtargetFeature<"el2vmsa", "HasEL2VMSA", "true",
  "Enable Exception Level 2 Virtual Memory System Architecture">;

def FeatureEL3 : SubtargetFeature<"el3", "HasEL3", "true",
  "Enable Exception Level 3">;

def FeatureCSSC : SubtargetFeature<"cssc", "HasCSSC", "true",
  "Enable Common Short Sequence Compression (CSSC) instructions (FEAT_CSSC)">;

def FeatureFixCortexA53_835769 : SubtargetFeature<"fix-cortex-a53-835769",
  "FixCortexA53_835769", "true", "Mitigate Cortex-A53 Erratum 835769">;

def FeatureNoBTIAtReturnTwice : SubtargetFeature<"no-bti-at-return-twice",
                                                 "NoBTIAtReturnTwice", "true",
                                                 "Don't place a BTI instruction "
                                                 "after a return-twice">;

def FeatureCHK : SubtargetFeature<"chk", "HasCHK",
    "true", "Enable Armv8.0-A Check Feature Status Extension (FEAT_CHK)">;

def FeatureGCS : SubtargetFeature<"gcs", "HasGCS",
    "true", "Enable Armv9.4-A Guarded Call Stack Extension", [FeatureCHK]>;

def FeatureCLRBHB : SubtargetFeature<"clrbhb", "HasCLRBHB",
    "true", "Enable Clear BHB instruction (FEAT_CLRBHB)">;

def FeaturePRFM_SLC : SubtargetFeature<"prfm-slc-target", "HasPRFM_SLC",
    "true", "Enable SLC target for PRFM instruction">;

def FeatureSPECRES2 : SubtargetFeature<"specres2", "HasSPECRES2",
    "true", "Enable Speculation Restriction Instruction (FEAT_SPECRES2)",
    [FeaturePredRes]>;

def FeatureMEC : SubtargetFeature<"mec", "HasMEC",
    "true", "Enable Memory Encryption Contexts Extension", [FeatureRME]>;

def FeatureITE : SubtargetFeature<"ite", "HasITE",
    "true", "Enable Armv9.4-A Instrumentation Extension FEAT_ITE", [FeatureETE,
    FeatureTRBE]>;

def FeatureRCPC3 : SubtargetFeature<"rcpc3", "HasRCPC3",
    "true", "Enable Armv8.9-A RCPC instructions for A64 and Advanced SIMD and floating-point instruction set (FEAT_LRCPC3)",
    [FeatureRCPC_IMMO]>;

def FeatureTHE : SubtargetFeature<"the", "HasTHE",
    "true", "Enable Armv8.9-A Translation Hardening Extension (FEAT_THE)">;

def FeatureLSE128 : SubtargetFeature<"lse128", "HasLSE128",
    "true", "Enable Armv9.4-A 128-bit Atomic Instructions (FEAT_LSE128)",
    [FeatureLSE]>;

// FEAT_D128, FEAT_LVA3, FEAT_SYSREG128, and FEAT_SYSINSTR128 are mutually implicit.
// Therefore group them all under a single feature flag, d128:
def FeatureD128 : SubtargetFeature<"d128", "HasD128",
    "true", "Enable Armv9.4-A 128-bit Page Table Descriptors, System Registers "
    "and Instructions (FEAT_D128, FEAT_LVA3, FEAT_SYSREG128, FEAT_SYSINSTR128)",
    [FeatureLSE128]>;

def FeatureDisableLdp : SubtargetFeature<"disable-ldp", "HasDisableLdp",
    "true", "Do not emit ldp">;

def FeatureDisableStp : SubtargetFeature<"disable-stp", "HasDisableStp",
    "true", "Do not emit stp">;

def FeatureLdpAlignedOnly : SubtargetFeature<"ldp-aligned-only", "HasLdpAlignedOnly",
    "true", "In order to emit ldp, first check if the load will be aligned to 2 * element_size">;

def FeatureStpAlignedOnly : SubtargetFeature<"stp-aligned-only", "HasStpAlignedOnly",
    "true", "In order to emit stp, first check if the store will be aligned to 2 * element_size">;

// AArch64 2023 Architecture Extensions (v9.5-A)

def FeatureCPA : SubtargetFeature<"cpa", "HasCPA", "true",
    "Enable Armv9.5-A Checked Pointer Arithmetic (FEAT_CPA)">;

def FeaturePAuthLR : SubtargetFeature<"pauth-lr", "HasPAuthLR",
    "true", "Enable Armv9.5-A PAC enhancements (FEAT_PAuth_LR)">;

def FeatureTLBIW : SubtargetFeature<"tlbiw", "HasTLBIW", "true",
  "Enable ARMv9.5-A TLBI VMALL for Dirty State (FEAT_TLBIW)">;

//===----------------------------------------------------------------------===//
// Architectures.
//
def HasV8_0aOps : SubtargetFeature<"v8a", "HasV8_0aOps", "true",
  "Support ARM v8.0a instructions", [FeatureEL2VMSA, FeatureEL3]>;

def HasV8_1aOps : SubtargetFeature<"v8.1a", "HasV8_1aOps", "true",
  "Support ARM v8.1a instructions", [HasV8_0aOps, FeatureCRC, FeatureLSE,
  FeatureRDM, FeaturePAN, FeatureLOR, FeatureVH]>;

def HasV8_2aOps : SubtargetFeature<"v8.2a", "HasV8_2aOps", "true",
  "Support ARM v8.2a instructions", [HasV8_1aOps, FeaturePsUAO,
  FeaturePAN_RWV, FeatureRAS, FeatureCCPP]>;

def HasV8_3aOps : SubtargetFeature<"v8.3a", "HasV8_3aOps", "true",
  "Support ARM v8.3a instructions", [HasV8_2aOps, FeatureRCPC, FeaturePAuth,
  FeatureJS, FeatureCCIDX, FeatureComplxNum]>;

def HasV8_4aOps : SubtargetFeature<"v8.4a", "HasV8_4aOps", "true",
  "Support ARM v8.4a instructions", [HasV8_3aOps, FeatureDotProd,
  FeatureNV, FeatureMPAM, FeatureDIT,
  FeatureTRACEV8_4, FeatureAM, FeatureSEL2, FeatureTLB_RMI,
  FeatureFlagM, FeatureRCPC_IMMO, FeatureLSE2]>;

def HasV8_5aOps : SubtargetFeature<
  "v8.5a", "HasV8_5aOps", "true", "Support ARM v8.5a instructions",
  [HasV8_4aOps, FeatureAltFPCmp, FeatureFRInt3264, FeatureSpecRestrict,
   FeatureSSBS, FeatureSB, FeaturePredRes, FeatureCacheDeepPersist,
   FeatureBranchTargetId]>;

def HasV8_6aOps : SubtargetFeature<
  "v8.6a", "HasV8_6aOps", "true", "Support ARM v8.6a instructions",
  [HasV8_5aOps, FeatureAMVS, FeatureBF16, FeatureFineGrainedTraps,
   FeatureEnhancedCounterVirtualization, FeatureMatMulInt8]>;

def HasV8_7aOps : SubtargetFeature<
  "v8.7a", "HasV8_7aOps", "true", "Support ARM v8.7a instructions",
  [HasV8_6aOps, FeatureXS, FeatureWFxT, FeatureHCX]>;

def HasV8_8aOps : SubtargetFeature<
  "v8.8a", "HasV8_8aOps", "true", "Support ARM v8.8a instructions",
  [HasV8_7aOps, FeatureHBC, FeatureMOPS, FeatureNMI]>;

def HasV8_9aOps : SubtargetFeature<
  "v8.9a", "HasV8_9aOps", "true", "Support ARM v8.9a instructions",
  [HasV8_8aOps, FeatureCLRBHB, FeaturePRFM_SLC, FeatureSPECRES2,
   FeatureCSSC, FeatureRASv2, FeatureCHK]>;

def HasV9_0aOps : SubtargetFeature<
  "v9a", "HasV9_0aOps", "true", "Support ARM v9a instructions",
  [HasV8_5aOps, FeatureMEC, FeatureSVE2]>;

def HasV9_1aOps : SubtargetFeature<
  "v9.1a", "HasV9_1aOps", "true", "Support ARM v9.1a instructions",
  [HasV8_6aOps, HasV9_0aOps]>;

def HasV9_2aOps : SubtargetFeature<
  "v9.2a", "HasV9_2aOps", "true", "Support ARM v9.2a instructions",
  [HasV8_7aOps, HasV9_1aOps]>;

def HasV9_3aOps : SubtargetFeature<
  "v9.3a", "HasV9_3aOps", "true", "Support ARM v9.3a instructions",
  [HasV8_8aOps, HasV9_2aOps]>;

def HasV9_4aOps : SubtargetFeature<
  "v9.4a", "HasV9_4aOps", "true", "Support ARM v9.4a instructions",
  [HasV8_9aOps, HasV9_3aOps]>;

def HasV9_5aOps : SubtargetFeature<
  "v9.5a", "HasV9_5aOps", "true", "Support ARM v9.5a instructions",
  [HasV9_4aOps, FeatureCPA]>;

def HasV8_0rOps : SubtargetFeature<
  "v8r", "HasV8_0rOps", "true", "Support ARM v8r instructions",
  [//v8.1
  FeatureCRC, FeaturePAN, FeatureLSE, FeatureCONTEXTIDREL2,
  //v8.2
  FeatureRAS, FeaturePsUAO, FeatureCCPP, FeaturePAN_RWV,
  //v8.3
  FeatureCCIDX, FeaturePAuth, FeatureRCPC,
  //v8.4
  FeatureTRACEV8_4, FeatureTLB_RMI, FeatureFlagM, FeatureDIT, FeatureSEL2,
  FeatureRCPC_IMMO,
  // Not mandatory in v8.0-R, but included here on the grounds that it
  // only enables names of system registers
  FeatureSpecRestrict
  ]>;

// Only intended to be used by disassemblers.
def FeatureAll
    : SubtargetFeature<"all", "IsAll", "true", "Enable all instructions", []>;

class AssemblerPredicateWithAll<dag cond, string name="">
    : AssemblerPredicate<(any_of FeatureAll, cond), name>;

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
include "AArch64SchedPredNeoverse.td"
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
def FeatureUseROEL0ForTP : SubtargetFeature<"tpidrro-el0", "UseROEL0ForTP",
  "true", "Permit use of TPIDRRO_EL0 for the TLS base">;

//===----------------------------------------------------------------------===//
// Control codegen mitigation against Straight Line Speculation vulnerability.
//===----------------------------------------------------------------------===//

def FeatureHardenSlsRetBr : SubtargetFeature<"harden-sls-retbr",
  "HardenSlsRetBr", "true",
  "Harden against straight line speculation across RET and BR instructions">;
def FeatureHardenSlsBlr : SubtargetFeature<"harden-sls-blr",
  "HardenSlsBlr", "true",
  "Harden against straight line speculation across BLR instructions">;
def FeatureHardenSlsNoComdat : SubtargetFeature<"harden-sls-nocomdat",
  "HardenSlsNoComdat", "true",
  "Generate thunk code for SLS mitigation in the normal text section">;

//===----------------------------------------------------------------------===//
// AArch64 Processors supported.
//

//===----------------------------------------------------------------------===//
// Unsupported features to disable for scheduling models
//===----------------------------------------------------------------------===//

class AArch64Unsupported { list<Predicate> F; }

let F = [HasSVE2p1, HasSVE2p1_or_HasSME2, HasSVE2p1_or_HasSME2p1] in
def SVE2p1Unsupported : AArch64Unsupported;

def SVE2Unsupported : AArch64Unsupported {
  let F = !listconcat([HasSVE2, HasSVE2orSME, HasSVE2orSME2, HasSSVE_FP8FMA, HasSMEF8F16,
                       HasSMEF8F32, HasSVE2AES, HasSVE2SHA3, HasSVE2SM4, HasSVE2BitPerm],
                       SVE2p1Unsupported.F);
}

def SVEUnsupported : AArch64Unsupported {
  let F = !listconcat([HasSVE, HasSVEorSME],
                      SVE2Unsupported.F);
}

let F = [HasSME2p1, HasSVE2p1_or_HasSME2p1] in
def SME2p1Unsupported : AArch64Unsupported;

def SME2Unsupported : AArch64Unsupported {
  let F = !listconcat([HasSME2, HasSVE2orSME2, HasSVE2p1_or_HasSME2, HasSSVE_FP8FMA,
                      HasSMEF8F16, HasSMEF8F32],
                      SME2p1Unsupported.F);
}

def SMEUnsupported : AArch64Unsupported {
  let F = !listconcat([HasSME, HasSMEI16I64, HasSMEF16F16, HasSMEF64F64, HasSMEFA64],
                      SME2Unsupported.F);
}

let F = [HasPAuth, HasPAuthLR] in
def PAUnsupported : AArch64Unsupported;

include "AArch64SchedA53.td"
include "AArch64SchedA55.td"
include "AArch64SchedA510.td"
include "AArch64SchedA57.td"
include "AArch64SchedCyclone.td"
include "AArch64SchedFalkor.td"
include "AArch64SchedKryo.td"
include "AArch64SchedExynosM3.td"
include "AArch64SchedExynosM4.td"
include "AArch64SchedExynosM5.td"
include "AArch64SchedThunderX.td"
include "AArch64SchedThunderX2T99.td"
include "AArch64SchedA64FX.td"
include "AArch64SchedThunderX3T110.td"
include "AArch64SchedTSV110.td"
include "AArch64SchedAmpere1.td"
include "AArch64SchedAmpere1B.td"
include "AArch64SchedNeoverseN1.td"
include "AArch64SchedNeoverseN2.td"
include "AArch64SchedNeoverseV1.td"
include "AArch64SchedNeoverseV2.td"

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

def TuneA520AE  : SubtargetFeature<"a520ae", "ARMProcFamily", "CortexA520AE",
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
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA73     : SubtargetFeature<"a73", "ARMProcFamily", "CortexA73",
                                   "Cortex-A73 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA75     : SubtargetFeature<"a75", "ARMProcFamily", "CortexA75",
                                   "Cortex-A75 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA76     : SubtargetFeature<"a76", "ARMProcFamily", "CortexA76",
                                   "Cortex-A76 ARM processors", [
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureAddrLSLFast,
                                   FeatureALULSLFast,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA77     : SubtargetFeature<"a77", "ARMProcFamily", "CortexA77",
                                   "Cortex-A77 ARM processors", [
                                   FeatureCmpBccFusion,
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureAddrLSLFast,
                                   FeatureALULSLFast,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA78 : SubtargetFeature<"a78", "ARMProcFamily", "CortexA78",
                               "Cortex-A78 ARM processors", [
                               FeatureCmpBccFusion,
                               FeatureFuseAES,
                               FeatureFuseAdrpAdd,
                               FeatureAddrLSLFast,
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
                                 FeatureAddrLSLFast,
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
                                FeatureAddrLSLFast,
                                FeatureALULSLFast,
                                FeaturePostRAScheduler,
                                FeatureEnableSelectOptimize,
                                FeaturePredictableSelectIsExpensive]>;

def TuneA710    : SubtargetFeature<"a710", "ARMProcFamily", "CortexA710",
                                   "Cortex-A710 ARM processors", [
                                   FeatureCmpBccFusion,
                                   FeatureFuseAES,
                                   FeatureFuseAdrpAdd,
                                   FeatureAddrLSLFast,
                                   FeatureALULSLFast,
                                   FeaturePostRAScheduler,
                                   FeatureEnableSelectOptimize,
                                   FeaturePredictableSelectIsExpensive]>;

def TuneA715 : SubtargetFeature<"a715", "ARMProcFamily", "CortexA715",
                                 "Cortex-A715 ARM processors", [
                                 FeatureFuseAES,
                                 FeaturePostRAScheduler,
                                 FeatureCmpBccFusion,
                                 FeatureAddrLSLFast,
                                 FeatureALULSLFast,
                                 FeatureFuseAdrpAdd,
                                 FeatureEnableSelectOptimize,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneA720 : SubtargetFeature<"a720", "ARMProcFamily", "CortexA720",
                                "Cortex-A720 ARM processors", [
                                 FeatureFuseAES,
                                 FeaturePostRAScheduler,
                                 FeatureCmpBccFusion,
                                 FeatureAddrLSLFast,
                                 FeatureALULSLFast,
                                 FeatureFuseAdrpAdd,
                                 FeatureEnableSelectOptimize,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneA720AE : SubtargetFeature<"a720ae", "ARMProcFamily", "CortexA720AE",
                                "Cortex-A720AE ARM processors", [
                                 FeatureFuseAES,
                                 FeaturePostRAScheduler,
                                 FeatureCmpBccFusion,
                                 FeatureAddrLSLFast,
                                 FeatureALULSLFast,
                                 FeatureFuseAdrpAdd,
                                 FeatureEnableSelectOptimize,
                                 FeaturePredictableSelectIsExpensive]>;

def TuneR82 : SubtargetFeature<"cortex-r82", "ARMProcFamily",
                               "CortexR82",
                               "Cortex-R82 ARM processors", [
                               FeaturePostRAScheduler]>;

def TuneX1 : SubtargetFeature<"cortex-x1", "ARMProcFamily", "CortexX1",
                                  "Cortex-X1 ARM processors", [
                                  FeatureCmpBccFusion,
                                  FeatureFuseAES,
                                  FeatureFuseAdrpAdd,
                                  FeatureAddrLSLFast,
                                  FeatureALULSLFast,
                                  FeaturePostRAScheduler,
                                  FeatureEnableSelectOptimize,
                                  FeaturePredictableSelectIsExpensive]>;

def TuneX2 : SubtargetFeature<"cortex-x2", "ARMProcFamily", "CortexX2",
                                  "Cortex-X2 ARM processors", [
                                  FeatureCmpBccFusion,
                                  FeatureFuseAES,
                                  FeatureFuseAdrpAdd,
                                  FeatureAddrLSLFast,
                                  FeatureALULSLFast,
                                  FeaturePostRAScheduler,
                                  FeatureEnableSelectOptimize,
                                  FeaturePredictableSelectIsExpensive]>;

def TuneX3 : SubtargetFeature<"cortex-x3", "ARMProcFamily", "CortexX3",
                              "Cortex-X3 ARM processors", [
                               FeatureAddrLSLFast,
                               FeatureALULSLFast,
                               FeatureFuseAdrpAdd,
                               FeatureFuseAES,
                               FeaturePostRAScheduler,
                               FeatureEnableSelectOptimize,
                               FeaturePredictableSelectIsExpensive]>;

def TuneX4 : SubtargetFeature<"cortex-x4", "ARMProcFamily", "CortexX4",
                              "Cortex-X4 ARM processors", [
                               FeatureAddrLSLFast,
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
                                     FeatureAddrLSLFast,
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
                                     FeatureAddrLSLFast,
                                     FeatureALULSLFast,
                                     FeaturePostRAScheduler,
                                     FeatureZCZeroing]>;

def TuneKryo    : SubtargetFeature<"kryo", "ARMProcFamily", "Kryo",
                                   "Qualcomm Kryo processors", [
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroing,
                                   FeatureAddrLSLFast,
                                   FeatureALULSLFast,
                                   FeatureStorePairSuppress]>;

def TuneFalkor  : SubtargetFeature<"falkor", "ARMProcFamily", "Falkor",
                                   "Qualcomm Falkor processors", [
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroing,
                                   FeatureStorePairSuppress,
                                   FeatureAddrLSLFast,
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
                                      FeatureAddrLSLFast,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneNeoverseN2 : SubtargetFeature<"neoversen2", "ARMProcFamily", "NeoverseN2",
                                      "Neoverse N2 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureAddrLSLFast,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneNeoverse512TVB : SubtargetFeature<"neoverse512tvb", "ARMProcFamily", "Neoverse512TVB",
                                      "Neoverse 512-TVB ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureAddrLSLFast,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneNeoverseV1 : SubtargetFeature<"neoversev1", "ARMProcFamily", "NeoverseV1",
                                      "Neoverse V1 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureAddrLSLFast,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive,
                                      FeatureNoSVEFPLD1R]>;

def TuneNeoverseV2 : SubtargetFeature<"neoversev2", "ARMProcFamily", "NeoverseV2",
                                      "Neoverse V2 ARM processors", [
                                      FeatureFuseAES,
                                      FeatureFuseAdrpAdd,
                                      FeatureAddrLSLFast,
                                      FeatureALULSLFast,
                                      FeaturePostRAScheduler,
                                      FeatureEnableSelectOptimize,
                                      FeaturePredictableSelectIsExpensive]>;

def TuneSaphira  : SubtargetFeature<"saphira", "ARMProcFamily", "Saphira",
                                   "Qualcomm Saphira processors", [
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroing,
                                   FeatureStorePairSuppress,
                                   FeatureAddrLSLFast,
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
                                   FeatureAddrLSLFast,
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
                                    FeatureAddrLSLFast,
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
                                    FeatureAddrLSLFast,
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
                                 FeatureComplxNum, FeatureJS];
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

// FeatureFuseAdrpAdd is enabled under Generic to allow linker merging
// optimizations.
def : ProcessorModel<"generic", CortexA510Model, ProcessorFeatures.Generic,
                     [FeatureFuseAES, FeatureFuseAdrpAdd, FeaturePostRAScheduler,
                      FeatureEnableSelectOptimize]>;
def : ProcessorModel<"cortex-a35", CortexA53Model, ProcessorFeatures.A53,
                     [TuneA35]>;
def : ProcessorModel<"cortex-a34", CortexA53Model, ProcessorFeatures.A53,
                     [TuneA35]>;
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
def : ProcessorModel<"cortex-a76", CortexA57Model, ProcessorFeatures.A76,
                     [TuneA76]>;
def : ProcessorModel<"cortex-a76ae", CortexA57Model, ProcessorFeatures.A76,
                     [TuneA76]>;
def : ProcessorModel<"cortex-a77", CortexA57Model, ProcessorFeatures.A77,
                     [TuneA77]>;
def : ProcessorModel<"cortex-a78", CortexA57Model, ProcessorFeatures.A78,
                     [TuneA78]>;
def : ProcessorModel<"cortex-a78ae", CortexA57Model, ProcessorFeatures.A78AE,
                     [TuneA78AE]>;
def : ProcessorModel<"cortex-a78c", CortexA57Model, ProcessorFeatures.A78C,
                     [TuneA78C]>;
def : ProcessorModel<"cortex-a710", NeoverseN2Model, ProcessorFeatures.A710,
                     [TuneA710]>;
def : ProcessorModel<"cortex-a715", NeoverseN2Model, ProcessorFeatures.A715,
                     [TuneA715]>;
def : ProcessorModel<"cortex-a720", NeoverseN2Model, ProcessorFeatures.A720,
                     [TuneA720]>;
def : ProcessorModel<"cortex-a720ae", NeoverseN2Model, ProcessorFeatures.A720AE,
                     [TuneA720AE]>;
def : ProcessorModel<"cortex-r82", CortexA55Model, ProcessorFeatures.R82,
                     [TuneR82]>;
def : ProcessorModel<"cortex-x1", CortexA57Model, ProcessorFeatures.X1,
                     [TuneX1]>;
def : ProcessorModel<"cortex-x1c", CortexA57Model, ProcessorFeatures.X1C,
                     [TuneX1]>;
def : ProcessorModel<"cortex-x2", NeoverseN2Model, ProcessorFeatures.X2,
                     [TuneX2]>;
def : ProcessorModel<"cortex-x3", NeoverseN2Model, ProcessorFeatures.X3,
                     [TuneX3]>;
def : ProcessorModel<"cortex-x4", NeoverseN2Model, ProcessorFeatures.X4,
                     [TuneX4]>;
def : ProcessorModel<"neoverse-e1", CortexA53Model,
                     ProcessorFeatures.NeoverseE1, [TuneNeoverseE1]>;
def : ProcessorModel<"neoverse-n1", NeoverseN1Model,
                     ProcessorFeatures.NeoverseN1, [TuneNeoverseN1]>;
def : ProcessorModel<"neoverse-n2", NeoverseN2Model,
                     ProcessorFeatures.NeoverseN2, [TuneNeoverseN2]>;
def : ProcessorModel<"neoverse-512tvb", NeoverseV1Model,
                     ProcessorFeatures.Neoverse512TVB, [TuneNeoverse512TVB]>;
def : ProcessorModel<"neoverse-v1", NeoverseV1Model,
                     ProcessorFeatures.NeoverseV1, [TuneNeoverseV1]>;
def : ProcessorModel<"neoverse-v2", NeoverseV2Model,
                     ProcessorFeatures.NeoverseV2, [TuneNeoverseV2]>;
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
def : ProcessorModel<"tsv110", TSV110Model, ProcessorFeatures.TSV110,
                     [TuneTSV110]>;

// Support cyclone as an alias for apple-a7 so we can still LTO old bitcode.
def : ProcessorModel<"cyclone", CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7]>;

// iPhone and iPad CPUs
def : ProcessorModel<"apple-a7", CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7]>;
def : ProcessorModel<"apple-a8", CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7]>;
def : ProcessorModel<"apple-a9", CycloneModel, ProcessorFeatures.AppleA7,
                     [TuneAppleA7]>;
def : ProcessorModel<"apple-a10", CycloneModel, ProcessorFeatures.AppleA10,
                     [TuneAppleA10]>;
def : ProcessorModel<"apple-a11", CycloneModel, ProcessorFeatures.AppleA11,
                     [TuneAppleA11]>;
def : ProcessorModel<"apple-a12", CycloneModel, ProcessorFeatures.AppleA12,
                     [TuneAppleA12]>;
def : ProcessorModel<"apple-a13", CycloneModel, ProcessorFeatures.AppleA13,
                     [TuneAppleA13]>;
def : ProcessorModel<"apple-a14", CycloneModel, ProcessorFeatures.AppleA14,
                     [TuneAppleA14]>;
def : ProcessorModel<"apple-a15", CycloneModel, ProcessorFeatures.AppleA15,
                     [TuneAppleA15]>;
def : ProcessorModel<"apple-a16", CycloneModel, ProcessorFeatures.AppleA16,
                     [TuneAppleA16]>;
def : ProcessorModel<"apple-a17", CycloneModel, ProcessorFeatures.AppleA17,
                     [TuneAppleA17]>;
// Mac CPUs
def : ProcessorModel<"apple-m1", CycloneModel, ProcessorFeatures.AppleA14,
                     [TuneAppleA14]>;
def : ProcessorModel<"apple-m2", CycloneModel, ProcessorFeatures.AppleA15,
                     [TuneAppleA15]>;
def : ProcessorModel<"apple-m3", CycloneModel, ProcessorFeatures.AppleA16,
                     [TuneAppleA16]>;

// watch CPUs.
def : ProcessorModel<"apple-s4", CycloneModel, ProcessorFeatures.AppleA12,
                     [TuneAppleA12]>;
def : ProcessorModel<"apple-s5", CycloneModel, ProcessorFeatures.AppleA12,
                     [TuneAppleA12]>;

// Alias for the latest Apple processor model supported by LLVM.
def : ProcessorModel<"apple-latest", CycloneModel, ProcessorFeatures.AppleA16,
                     [TuneAppleA16]>;

// Fujitsu A64FX
def : ProcessorModel<"a64fx", A64FXModel, ProcessorFeatures.A64FX,
                     [TuneA64FX]>;

// Nvidia Carmel
def : ProcessorModel<"carmel", NoSchedModel, ProcessorFeatures.Carmel,
                     [TuneCarmel]>;

// Ampere Computing
def : ProcessorModel<"ampere1", Ampere1Model, ProcessorFeatures.Ampere1,
                     [TuneAmpere1]>;

def : ProcessorModel<"ampere1a", Ampere1Model, ProcessorFeatures.Ampere1A,
                     [TuneAmpere1A]>;

def : ProcessorModel<"ampere1b", Ampere1BModel, ProcessorFeatures.Ampere1B,
                     [TuneAmpere1B]>;

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
