//=- AArch64Features.td - Describe AArch64 SubtargetFeatures -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

// A SubtargetFeature that can be toggled from the command line, and therefore
// has an AEK_* entry in ArmExtKind.
class Extension<
  string TargetFeatureName,            // String used for -target-feature and -march, unless overridden.
  string Spelling,                     // The XYZ in HasXYZ and AEK_XYZ.
  string Desc,                         // Description.
  list<SubtargetFeature> Implies = []  // List of dependent features.
> : SubtargetFeature<TargetFeatureName, "Has" # Spelling, "true", Desc, Implies>
{
    string ArchExtKindSpelling = "AEK_" # Spelling; // ArchExtKind enum name.

    // In general, the name written on the command line should match the name
    // used for -target-feature. However, there are exceptions. Therefore we
    // add a separate field for this, to allow overriding it. Strongly prefer
    // not doing so.
    string MArchName = TargetFeatureName;

    // An alias that can be used on the command line, if the extension has one.
    // Used for correcting historical names while remaining backwards compatible.
    string MArchAlias = "";
}




// Each SubtargetFeature which corresponds to an Arm Architecture feature should
// be annotated with the respective FEAT_ feature name from the Architecture
// Reference Manual. If a SubtargetFeature enables instructions from multiple
// Arm Architecture Features, it should list all the relevant features. Not all
// FEAT_ features have a corresponding SubtargetFeature.


//===----------------------------------------------------------------------===//
//  Armv8.0 Architecture Extensions
//===----------------------------------------------------------------------===//

let ArchExtKindSpelling = "AEK_FP", MArchName = "fp" in
def FeatureFPARMv8 : Extension<"fp-armv8", "FPARMv8",
  "Enable ARMv8 (FEAT_FP)">;

let ArchExtKindSpelling = "AEK_SIMD", MArchName = "simd" in
def FeatureNEON : Extension<"neon", "NEON",
  "Enable Advanced SIMD instructions (FEAT_AdvSIMD)", [FeatureFPARMv8]>;

def FeatureSHA2 : Extension<
    "sha2", "SHA2",
    "Enable SHA1 and SHA256 support (FEAT_SHA1, FEAT_SHA256)", [FeatureNEON]>;

def FeatureAES : Extension<
    "aes", "AES",
    "Enable AES support (FEAT_AES, FEAT_PMULL)", [FeatureNEON]>;

// Crypto has been split up and any combination is now valid (see the
// crypto definitions above). Also, crypto is now context sensitive:
// it has a different meaning for e.g. Armv8.4 than it has for Armv8.2.
// Therefore, we rely on Clang, the user interfacing tool, to pass on the
// appropriate crypto options. But here in the backend, crypto has very little
// meaning anymore. We kept the Crypto definition here for backward
// compatibility, and now imply features SHA2 and AES, which was the
// "traditional" meaning of Crypto.
def FeatureCrypto : Extension<"crypto", "Crypto",
  "Enable cryptographic instructions", [FeatureNEON, FeatureSHA2, FeatureAES]>;

def FeatureCRC : Extension<"crc", "CRC",
  "Enable ARMv8 CRC-32 checksum instructions (FEAT_CRC32)">;

// This SubtargetFeature is special. It controls only whether codegen will turn
// `llvm.readcyclecounter()` into an access to a PMUv3 System Register. The
// `FEAT_PMUv3*` system registers are always available for assembly/disassembly.
let MArchName = "pmuv3" in
def FeaturePerfMon : Extension<"perfmon", "PerfMon",
  "Enable Code Generation for ARMv8 PMUv3 Performance Monitors extension (FEAT_PMUv3)">;

def FeatureSpecRestrict : SubtargetFeature<"specrestrict", "HasSpecRestrict",
  "true", "Enable architectural speculation restriction (FEAT_CSV2_2)">;

//===----------------------------------------------------------------------===//
//  Armv8.1 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureLSE : Extension<"lse", "LSE",
  "Enable ARMv8.1 Large System Extension (LSE) atomic instructions (FEAT_LSE)">;

let MArchAlias = "rdma" in
def FeatureRDM : Extension<"rdm", "RDM",
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

//===----------------------------------------------------------------------===//
//  Armv8.2 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureSM4 : Extension<
    "sm4", "SM4",
    "Enable SM3 and SM4 support (FEAT_SM4, FEAT_SM3)", [FeatureNEON]>;

def FeatureSHA3 : Extension<
    "sha3", "SHA3",
    "Enable SHA512 and SHA3 support (FEAT_SHA3, FEAT_SHA512)", [FeatureNEON, FeatureSHA2]>;

def FeatureRAS : Extension<"ras", "RAS",
  "Enable ARMv8 Reliability, Availability and Serviceability Extensions (FEAT_RAS, FEAT_RASv1p1)">;

let ArchExtKindSpelling = "AEK_FP16", MArchName = "fp16" in
def FeatureFullFP16 : Extension<"fullfp16", "FullFP16",
  "Full FP16 (FEAT_FP16)", [FeatureFPARMv8]>;

let ArchExtKindSpelling = "AEK_PROFILE", MArchName = "profile" in
def FeatureSPE : Extension<"spe", "SPE",
  "Enable Statistical Profiling extension (FEAT_SPE)">;

def FeaturePAN_RWV : SubtargetFeature<
    "pan-rwv", "HasPAN_RWV", "true",
    "Enable v8.2 PAN s1e1R and s1e1W Variants (FEAT_PAN2)",
    [FeaturePAN]>;

def FeaturePsUAO : SubtargetFeature< "uaops", "HasPsUAO", "true",
    "Enable v8.2 UAO PState (FEAT_UAO)">;

def FeatureCCPP : SubtargetFeature<"ccpp", "HasCCPP",
    "true", "Enable v8.2 data Cache Clean to Point of Persistence (FEAT_DPB)" >;

def FeatureSVE : Extension<"sve", "SVE",
  "Enable Scalable Vector Extension (SVE) instructions (FEAT_SVE)", [FeatureFullFP16]>;

let ArchExtKindSpelling = "AEK_I8MM" in
def FeatureMatMulInt8 : Extension<"i8mm", "MatMulInt8",
    "Enable Matrix Multiply Int8 Extension (FEAT_I8MM)", []>;

let ArchExtKindSpelling = "AEK_F32MM" in
def FeatureMatMulFP32 : Extension<"f32mm", "MatMulFP32",
    "Enable Matrix Multiply FP32 Extension (FEAT_F32MM)", [FeatureSVE]>;

let ArchExtKindSpelling = "AEK_F64MM" in
def FeatureMatMulFP64 : Extension<"f64mm", "MatMulFP64",
    "Enable Matrix Multiply FP64 Extension (FEAT_F64MM)", [FeatureSVE]>;

//===----------------------------------------------------------------------===//
//  Armv8.3 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureRCPC : Extension<"rcpc", "RCPC",
    "Enable support for RCPC extension (FEAT_LRCPC)", []>;

def FeaturePAuth : Extension<
    "pauth", "PAuth",
    "Enable v8.3-A Pointer Authentication extension (FEAT_PAuth)">;

let ArchExtKindSpelling = "AEK_JSCVT", MArchName = "jscvt" in
def FeatureJS : Extension<
    "jsconv", "JS",
    "Enable v8.3-A JavaScript FP conversion instructions (FEAT_JSCVT)",
    [FeatureFPARMv8]>;

def FeatureCCIDX : SubtargetFeature<
    "ccidx", "HasCCIDX", "true",
    "Enable v8.3-A Extend of the CCSIDR number of sets (FEAT_CCIDX)">;

let ArchExtKindSpelling = "AEK_FCMA", MArchName = "fcma" in
def FeatureComplxNum : Extension<
    "complxnum", "ComplxNum",
    "Enable v8.3-A Floating-point complex number support (FEAT_FCMA)",
    [FeatureNEON]>;

def FeatureNV : SubtargetFeature<
    "nv", "HasNV", "true",
    "Enable v8.4-A Nested Virtualization Enchancement (FEAT_NV, FEAT_NV2)">;

//===----------------------------------------------------------------------===//
//  Armv8.4 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureLSE2 : SubtargetFeature<"lse2", "HasLSE2", "true",
  "Enable ARMv8.4 Large System Extension 2 (LSE2) atomicity rules (FEAT_LSE2)">;

def FeatureFP16FML : Extension<"fp16fml", "FP16FML",
  "Enable FP16 FML instructions (FEAT_FHM)", [FeatureFullFP16]>;

def FeatureDotProd : Extension<
    "dotprod", "DotProd",
    "Enable dot product support (FEAT_DotProd)", [FeatureNEON]>;

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

def FeatureSEL2 : SubtargetFeature<
    "sel2", "HasSEL2", "true",
    "Enable v8.4-A Secure Exception Level 2 extension (FEAT_SEL2)">;

def FeatureTLB_RMI : SubtargetFeature<
    "tlb-rmi", "HasTLB_RMI", "true",
    "Enable v8.4-A TLB Range and Maintenance Instructions (FEAT_TLBIOS, FEAT_TLBIRANGE)">;

def FeatureFlagM : Extension<
    "flagm", "FlagM",
    "Enable v8.4-A Flag Manipulation Instructions (FEAT_FlagM)", []>;

def FeatureRCPC_IMMO : SubtargetFeature<"rcpc-immo", "HasRCPC_IMMO", "true",
    "Enable v8.4-A RCPC instructions with Immediate Offsets (FEAT_LRCPC2)",
    [FeatureRCPC]>;

//===----------------------------------------------------------------------===//
//  Armv8.5 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureAltFPCmp : SubtargetFeature<"altnzcv", "HasAlternativeNZCV", "true",
  "Enable alternative NZCV format for floating point comparisons (FEAT_FlagM2)">;

def FeatureFRInt3264 : SubtargetFeature<"fptoint", "HasFRInt3264", "true",
  "Enable FRInt[32|64][Z|X] instructions that round a floating-point number to "
  "an integer (in FP format) forcing it to fit into a 32- or 64-bit int (FEAT_FRINTTS)" >;

def FeatureSB : Extension<"sb", "SB",
  "Enable v8.5 Speculation Barrier (FEAT_SB)", []>;

def FeatureSSBS : Extension<"ssbs", "SSBS",
  "Enable Speculative Store Bypass Safe bit (FEAT_SSBS, FEAT_SSBS2)", []>;

def FeaturePredRes : Extension<"predres", "PredRes",
  "Enable v8.5a execution and data prediction invalidation instructions (FEAT_SPECRES)", []>;

def FeatureCacheDeepPersist : SubtargetFeature<"ccdp", "CCDP", "true",
    "Enable v8.5 Cache Clean to Point of Deep Persistence (FEAT_DPB2)" >;

def FeatureBranchTargetId : SubtargetFeature<"bti", "BTI", "true",
    "Enable Branch Target Identification (FEAT_BTI)">;

let ArchExtKindSpelling = "AEK_RAND", MArchName = "rng" in
def FeatureRandGen : Extension<"rand", "RandGen",
    "Enable Random Number generation instructions (FEAT_RNG)", []>;

// NOTE: "memtag" means FEAT_MTE + FEAT_MTE2 for -march or
// __attribute((target(...))), but only FEAT_MTE for FMV.
let MArchName = "memtag" in
def FeatureMTE : Extension<"mte", "MTE",
    "Enable Memory Tagging Extension (FEAT_MTE, FEAT_MTE2)", []>;

//===----------------------------------------------------------------------===//
//  Armv8.6 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureBF16 : Extension<"bf16", "BF16",
    "Enable BFloat16 Extension (FEAT_BF16)">;

def FeatureAMVS : SubtargetFeature<
    "amvs", "HasAMVS", "true",
    "Enable v8.6-A Activity Monitors Virtualization support (FEAT_AMUv1p1)",
    [FeatureAM]>;

def FeatureFineGrainedTraps : SubtargetFeature<"fgt", "HasFineGrainedTraps",
    "true", "Enable fine grained virtualization traps extension (FEAT_FGT)">;

def FeatureEnhancedCounterVirtualization :
      SubtargetFeature<"ecv", "HasEnhancedCounterVirtualization",
      "true", "Enable enhanced counter virtualization extension (FEAT_ECV)">;

//===----------------------------------------------------------------------===//
//  Armv8.7 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureXS : SubtargetFeature<"xs", "HasXS",
    "true", "Enable Armv8.7-A limited-TLB-maintenance instruction (FEAT_XS)">;

def FeatureWFxT : SubtargetFeature<"wfxt", "WFxT", "true",
    "Enable Armv8.7-A WFET and WFIT instruction (FEAT_WFxT)">;

def FeatureHCX : SubtargetFeature<
    "hcx", "HasHCX", "true", "Enable Armv8.7-A HCRX_EL2 system register (FEAT_HCX)">;

def FeatureLS64 : Extension<"ls64", "LS64",
    "Enable Armv8.7-A LD64B/ST64B Accelerator Extension (FEAT_LS64, FEAT_LS64_V, FEAT_LS64_ACCDATA)", []>;

def FeatureSPE_EEF : SubtargetFeature<"spe-eef", "HasSPE_EEF",
    "true", "Enable extra register in the Statistical Profiling Extension (FEAT_SPEv1p2)">;

//===----------------------------------------------------------------------===//
//  Armv8.8 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureHBC : Extension<"hbc", "HBC",
    "Enable Armv8.8-A Hinted Conditional Branches Extension (FEAT_HBC)">;

def FeatureMOPS : Extension<"mops", "MOPS",
    "Enable Armv8.8-A memcpy and memset acceleration instructions (FEAT_MOPS)", []>;

def FeatureNMI : SubtargetFeature<"nmi", "HasNMI",
    "true", "Enable Armv8.8-A Non-maskable Interrupts (FEAT_NMI, FEAT_GICv3_NMI)">;

//===----------------------------------------------------------------------===//
//  Armv8.9 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureRASv2 : Extension<"rasv2", "RASv2",
  "Enable ARMv8.9-A Reliability, Availability and Serviceability Extensions (FEAT_RASv2)",
  [FeatureRAS]>;

def FeatureCSSC : Extension<"cssc", "CSSC",
  "Enable Common Short Sequence Compression (CSSC) instructions (FEAT_CSSC)">;

def FeatureCLRBHB : SubtargetFeature<"clrbhb", "HasCLRBHB",
    "true", "Enable Clear BHB instruction (FEAT_CLRBHB)">;

def FeaturePRFM_SLC : SubtargetFeature<"prfm-slc-target", "HasPRFM_SLC",
    "true", "Enable SLC target for PRFM instruction">;

let MArchName = "predres2" in
def FeatureSPECRES2 : Extension<"specres2", "SPECRES2",
    "Enable Speculation Restriction Instruction (FEAT_SPECRES2)",
    [FeaturePredRes]>;

def FeatureRCPC3 : Extension<"rcpc3", "RCPC3",
    "Enable Armv8.9-A RCPC instructions for A64 and Advanced SIMD and floating-point instruction set (FEAT_LRCPC3)",
    [FeatureRCPC_IMMO]>;

def FeatureTHE : Extension<"the", "THE",
    "Enable Armv8.9-A Translation Hardening Extension (FEAT_THE)">;

//===----------------------------------------------------------------------===//
//  Armv9.0 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureUseScalarIncVL : SubtargetFeature<"use-scalar-inc-vl",
  "UseScalarIncVL", "true", "Prefer inc/dec over add+cnt">;

def FeatureSVE2 : Extension<"sve2", "SVE2",
  "Enable Scalable Vector Extension 2 (SVE2) instructions (FEAT_SVE2)",
  [FeatureSVE, FeatureUseScalarIncVL]>;

def FeatureSVE2AES : Extension<"sve2-aes", "SVE2AES",
  "Enable AES SVE2 instructions (FEAT_SVE_AES, FEAT_SVE_PMULL128)",
  [FeatureSVE2, FeatureAES]>;

def FeatureSVE2SM4 : Extension<"sve2-sm4", "SVE2SM4",
  "Enable SM4 SVE2 instructions (FEAT_SVE_SM4)", [FeatureSVE2, FeatureSM4]>;

def FeatureSVE2SHA3 : Extension<"sve2-sha3", "SVE2SHA3",
  "Enable SHA3 SVE2 instructions (FEAT_SVE_SHA3)", [FeatureSVE2, FeatureSHA3]>;

def FeatureSVE2BitPerm : Extension<"sve2-bitperm", "SVE2BitPerm",
  "Enable bit permutation SVE2 instructions (FEAT_SVE_BitPerm)", [FeatureSVE2]>;

def FeatureTRBE : SubtargetFeature<"trbe", "TRBE", "true",
    "Enable Trace Buffer Extension (FEAT_TRBE)">;

def FeatureETE : SubtargetFeature<"ete", "ETE", "true",
    "Enable Embedded Trace Extension (FEAT_ETE)",
    [FeatureTRBE]>;

def FeatureTME : Extension<"tme", "TME",
    "Enable Transactional Memory Extension (FEAT_TME)" >;

//===----------------------------------------------------------------------===//
//  Armv9.1 Architecture Extensions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Armv9.2 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureBRBE : Extension<"brbe", "BRBE",
    "Enable Branch Record Buffer Extension (FEAT_BRBE)">;

def FeatureRME : SubtargetFeature<"rme", "HasRME",
    "true", "Enable Realm Management Extension (FEAT_RME)">;

def FeatureSME : Extension<"sme", "SME",
  "Enable Scalable Matrix Extension (SME) (FEAT_SME)", [FeatureBF16, FeatureUseScalarIncVL]>;

def FeatureSMEF64F64 : Extension<"sme-f64f64", "SMEF64F64",
  "Enable Scalable Matrix Extension (SME) F64F64 instructions (FEAT_SME_F64F64)", [FeatureSME]>;

def FeatureSMEI16I64 : Extension<"sme-i16i64", "SMEI16I64",
  "Enable Scalable Matrix Extension (SME) I16I64 instructions (FEAT_SME_I16I64)", [FeatureSME]>;

def FeatureSMEFA64 : Extension<"sme-fa64", "SMEFA64",
  "Enable the full A64 instruction set in streaming SVE mode (FEAT_SME_FA64)", [FeatureSME, FeatureSVE2]>;

//===----------------------------------------------------------------------===//
//  Armv9.3 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureSME2 : Extension<"sme2", "SME2",
  "Enable Scalable Matrix Extension 2 (SME2) instructions", [FeatureSME]>;

def FeatureMEC : SubtargetFeature<"mec", "HasMEC",
    "true", "Enable Memory Encryption Contexts Extension", [FeatureRME]>;

//===----------------------------------------------------------------------===//
//  Armv9.4 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureSVE2p1: Extension<"sve2p1", "SVE2p1",
  "Enable Scalable Vector Extension 2.1 instructions", [FeatureSVE2]>;

def FeatureB16B16 : Extension<"b16b16", "B16B16",
  "Enable SVE2.1 or SME2.1 non-widening BFloat16 to BFloat16 instructions (FEAT_B16B16)", [FeatureBF16]>;

def FeatureSMEF16F16 : Extension<"sme-f16f16", "SMEF16F16",
  "Enable SME non-widening Float16 instructions (FEAT_SME_F16F16)", [FeatureSME2]>;

def FeatureSME2p1 : Extension<"sme2p1", "SME2p1",
  "Enable Scalable Matrix Extension 2.1 (FEAT_SME2p1) instructions", [FeatureSME2]>;

def FeatureCHK : SubtargetFeature<"chk", "HasCHK",
    "true", "Enable Armv8.0-A Check Feature Status Extension (FEAT_CHK)">;

def FeatureGCS : Extension<"gcs", "GCS",
    "Enable Armv9.4-A Guarded Call Stack Extension", [FeatureCHK]>;

def FeatureITE : Extension<"ite", "ITE",
    "Enable Armv9.4-A Instrumentation Extension FEAT_ITE", [FeatureETE,
    FeatureTRBE]>;

def FeatureLSE128 : Extension<"lse128", "LSE128",
    "Enable Armv9.4-A 128-bit Atomic Instructions (FEAT_LSE128)",
    [FeatureLSE]>;

// FEAT_D128, FEAT_LVA3, FEAT_SYSREG128, and FEAT_SYSINSTR128 are mutually implicit.
// Therefore group them all under a single feature flag, d128:
def FeatureD128 : Extension<"d128", "D128",
    "Enable Armv9.4-A 128-bit Page Table Descriptors, System Registers "
    "and Instructions (FEAT_D128, FEAT_LVA3, FEAT_SYSREG128, FEAT_SYSINSTR128)",
    [FeatureLSE128]>;

//===----------------------------------------------------------------------===//
//  Armv9.5 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureFAMINMAX: Extension<"faminmax", "FAMINMAX",
   "Enable FAMIN and FAMAX instructions (FEAT_FAMINMAX)">;

def FeatureLUT: Extension<"lut", "LUT",
   "Enable Lookup Table instructions (FEAT_LUT)">;
   
def FeatureFP8 : Extension<"fp8", "FP8",
  "Enable FP8 instructions (FEAT_FP8)", [FeatureFAMINMAX, FeatureLUT, FeatureBF16]>;
  
def FeatureFP8FMA : Extension<"fp8fma", "FP8FMA",
  "Enable fp8 multiply-add instructions (FEAT_FP8FMA)", [FeatureFP8]>;

def FeatureSSVE_FP8FMA : Extension<"ssve-fp8fma", "SSVE_FP8FMA",
  "Enable SVE2 fp8 multiply-add instructions (FEAT_SSVE_FP8FMA)", [FeatureSME2, FeatureFP8]>;

def FeatureFP8DOT4: Extension<"fp8dot4", "FP8DOT4",
   "Enable fp8 4-way dot instructions (FEAT_FP8DOT4)", [FeatureFP8FMA]>;
  
def FeatureFP8DOT2: Extension<"fp8dot2", "FP8DOT2",
   "Enable fp8 2-way dot instructions (FEAT_FP8DOT2)", [FeatureFP8DOT4]>;

def FeatureSSVE_FP8DOT4 : Extension<"ssve-fp8dot4", "SSVE_FP8DOT4",
  "Enable SVE2 fp8 4-way dot product instructions (FEAT_SSVE_FP8DOT4)", [FeatureSSVE_FP8FMA]>;

def FeatureSSVE_FP8DOT2 : Extension<"ssve-fp8dot2", "SSVE_FP8DOT2",
  "Enable SVE2 fp8 2-way dot product instructions (FEAT_SSVE_FP8DOT2)", [FeatureSSVE_FP8DOT4]>;

def FeatureSME_LUTv2 : Extension<"sme-lutv2", "SME_LUTv2",
  "Enable Scalable Matrix Extension (SME) LUTv2 instructions (FEAT_SME_LUTv2)">;

def FeatureSMEF8F32 : Extension<"sme-f8f32", "SMEF8F32",
  "Enable Scalable Matrix Extension (SME) F8F32 instructions (FEAT_SME_F8F32)", [FeatureSME2, FeatureFP8]>;

def FeatureSMEF8F16 : Extension<"sme-f8f16", "SMEF8F16",
  "Enable Scalable Matrix Extension (SME) F8F16 instructions(FEAT_SME_F8F16)", [FeatureSMEF8F32]>;

def FeatureCPA : Extension<"cpa", "CPA",
    "Enable Armv9.5-A Checked Pointer Arithmetic (FEAT_CPA)">;

def FeaturePAuthLR : Extension<"pauth-lr", "PAuthLR",
    "Enable Armv9.5-A PAC enhancements (FEAT_PAuth_LR)">;

def FeatureTLBIW : Extension<"tlbiw", "TLBIW",
  "Enable ARMv9.5-A TLBI VMALL for Dirty State (FEAT_TLBIW)">;

//===----------------------------------------------------------------------===//
//  Other Features
//===----------------------------------------------------------------------===//

def FeatureOutlineAtomics : SubtargetFeature<"outline-atomics", "OutlineAtomics", "true",
  "Enable out of line atomics to support LSE instructions">;

def FeatureFMV : SubtargetFeature<"fmv", "HasFMV", "true",
  "Enable Function Multi Versioning support.">;

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

def FeatureNoSVEFPLD1R : SubtargetFeature<"no-sve-fp-ld1r",
  "NoSVEFPLD1R", "true", "Avoid using LD1RX instructions for FP">;

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

foreach i = {1-7,9-15,18,20-28} in
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

def FeatureUseRSqrt : SubtargetFeature<
    "use-reciprocal-square-root", "UseRSqrt", "true",
    "Use the reciprocal square root approximation">;

def FeatureNoNegativeImmediates : SubtargetFeature<"no-neg-immediates",
                                        "NegativeImmediates", "false",
                                        "Convert immediates and instructions "
                                        "to their negated or complemented "
                                        "equivalent when the immediate does "
                                        "not fit in the encoding.">;

// Address operands with shift amount 2 or 3 are fast on all Arm chips except
// some old Apple cores (A7-A10?) which handle all shifts slowly. Cortex-A57
// and derived designs through Cortex-X1 take an extra micro-op for shifts
// of 1 or 4. Other Arm chips handle all shifted operands at the same speed
// as unshifted operands.
//
// We don't try to model the behavior of the old Apple cores because new code
// targeting A7 is very unlikely to actually run on an A7. The Cortex cores
// are modeled by FeatureAddrLSLSlow14.
def FeatureAddrLSLSlow14 : SubtargetFeature<
    "addr-lsl-slow-14", "HasAddrLSLSlow14", "true",
    "Address operands with shift amount of 1 or 4 are slow">;

def FeatureALULSLFast : SubtargetFeature<
    "alu-lsl-fast", "HasALULSLFast", "true",
    "Add/Sub operations with lsl shift <= 4 are cheap">;

def FeatureAggressiveFMA :
  SubtargetFeature<"aggressive-fma",
                   "HasAggressiveFMA",
                   "true",
                   "Enable Aggressive FMA for floating-point.">;

def FeatureTaggedGlobals : SubtargetFeature<"tagged-globals",
    "AllowTaggedGlobals",
    "true", "Use an instruction sequence for taking the address of a global "
    "that allows a memory tag in the upper address bits">;

def FeatureAppleA7SysReg  : SubtargetFeature<"apple-a7-sysreg", "HasAppleA7SysReg", "true",
  "Apple A7 (the CPU formerly known as Cyclone)">;

def FeatureEL2VMSA : SubtargetFeature<"el2vmsa", "HasEL2VMSA", "true",
  "Enable Exception Level 2 Virtual Memory System Architecture">;

def FeatureEL3 : SubtargetFeature<"el3", "HasEL3", "true",
  "Enable Exception Level 3">;

def FeatureFixCortexA53_835769 : SubtargetFeature<"fix-cortex-a53-835769",
  "FixCortexA53_835769", "true", "Mitigate Cortex-A53 Erratum 835769">;

def FeatureNoBTIAtReturnTwice : SubtargetFeature<"no-bti-at-return-twice",
                                                 "NoBTIAtReturnTwice", "true",
                                                 "Don't place a BTI instruction "
                                                 "after a return-twice">;

def FeatureDisableLdp : SubtargetFeature<"disable-ldp", "HasDisableLdp",
    "true", "Do not emit ldp">;

def FeatureDisableStp : SubtargetFeature<"disable-stp", "HasDisableStp",
    "true", "Do not emit stp">;

def FeatureLdpAlignedOnly : SubtargetFeature<"ldp-aligned-only", "HasLdpAlignedOnly",
    "true", "In order to emit ldp, first check if the load will be aligned to 2 * element_size">;

def FeatureStpAlignedOnly : SubtargetFeature<"stp-aligned-only", "HasStpAlignedOnly",
    "true", "In order to emit stp, first check if the store will be aligned to 2 * element_size">;

//===----------------------------------------------------------------------===//
// Architectures.
//
class Architecture64<
  int major, int minor, string profile,
  string target_feature_name,
  list<SubtargetFeature> implied_features,
  list<Extension> default_extensions
> : SubtargetFeature<target_feature_name,
    "HasV" # major # "_" # minor # profile # "Ops", "true",
    "Support ARM " # target_feature_name # " architecture",
    implied_features
> {
  int Major = major;
  int Minor = minor;
  string Profile = profile;

  // Extensions enabled by default. Not the same as implied SubtargetFeatures.
  list<Extension> DefaultExts = default_extensions;
}

def HasV8_0aOps : Architecture64<8, 0, "a", "v8a",
  [FeatureEL2VMSA, FeatureEL3],
  [FeatureFPARMv8, FeatureNEON]>;
def HasV8_1aOps : Architecture64<8, 1, "a", "v8.1a",
  [HasV8_0aOps, FeatureCRC, FeatureLSE, FeatureRDM, FeaturePAN, FeatureLOR,
   FeatureVH],
  !listconcat(HasV8_0aOps.DefaultExts, [FeatureCRC, FeatureLSE, FeatureRDM])>;
def HasV8_2aOps : Architecture64<8, 2, "a", "v8.2a",
  [HasV8_1aOps, FeaturePsUAO, FeaturePAN_RWV, FeatureRAS, FeatureCCPP],
  !listconcat(HasV8_1aOps.DefaultExts, [FeatureRAS])>;
def HasV8_3aOps : Architecture64<8, 3, "a", "v8.3a",
  [HasV8_2aOps, FeatureRCPC, FeaturePAuth, FeatureJS, FeatureCCIDX,
    FeatureComplxNum],
  !listconcat(HasV8_2aOps.DefaultExts, [FeatureComplxNum, FeatureJS,
    FeaturePAuth, FeatureRCPC])>;
def HasV8_4aOps : Architecture64<8, 4, "a", "v8.4a",
  [HasV8_3aOps, FeatureDotProd, FeatureNV, FeatureMPAM, FeatureDIT,
    FeatureTRACEV8_4, FeatureAM, FeatureSEL2, FeatureTLB_RMI, FeatureFlagM,
    FeatureRCPC_IMMO, FeatureLSE2],
  !listconcat(HasV8_3aOps.DefaultExts, [FeatureDotProd])>;
def HasV8_5aOps : Architecture64<8, 5, "a", "v8.5a",
  [HasV8_4aOps, FeatureAltFPCmp, FeatureFRInt3264, FeatureSpecRestrict,
    FeatureSSBS, FeatureSB, FeaturePredRes, FeatureCacheDeepPersist,
    FeatureBranchTargetId],
  !listconcat(HasV8_4aOps.DefaultExts, [])>;
def HasV8_6aOps : Architecture64<8, 6, "a", "v8.6a",
  [HasV8_5aOps, FeatureAMVS, FeatureBF16, FeatureFineGrainedTraps,
    FeatureEnhancedCounterVirtualization, FeatureMatMulInt8],
  !listconcat(HasV8_5aOps.DefaultExts, [FeatureBF16, FeatureMatMulInt8])>;
def HasV8_7aOps : Architecture64<8, 7, "a", "v8.7a",
  [HasV8_6aOps, FeatureXS, FeatureWFxT, FeatureHCX],
  !listconcat(HasV8_6aOps.DefaultExts, [])>;
def HasV8_8aOps : Architecture64<8, 8, "a", "v8.8a",
  [HasV8_7aOps, FeatureHBC, FeatureMOPS, FeatureNMI],
  !listconcat(HasV8_7aOps.DefaultExts, [FeatureMOPS, FeatureHBC])>;
def HasV8_9aOps : Architecture64<8, 9, "a", "v8.9a",
  [HasV8_8aOps, FeatureCLRBHB, FeaturePRFM_SLC, FeatureSPECRES2,
   FeatureCSSC, FeatureRASv2, FeatureCHK],
  !listconcat(HasV8_8aOps.DefaultExts, [FeatureSPECRES2, FeatureCSSC,
    FeatureRASv2])>;
def HasV9_0aOps : Architecture64<9, 0, "a", "v9a",
  [HasV8_5aOps, FeatureMEC],
  !listconcat(HasV8_5aOps.DefaultExts, [FeatureFullFP16, FeatureSVE,
    FeatureSVE2])>;
def HasV9_1aOps : Architecture64<9, 1, "a", "v9.1a",
  [HasV8_6aOps, HasV9_0aOps],
  !listconcat(HasV9_0aOps.DefaultExts, [FeatureBF16, FeatureMatMulInt8])>;
def HasV9_2aOps : Architecture64<9, 2, "a", "v9.2a",
  [HasV8_7aOps, HasV9_1aOps],
  !listconcat(HasV9_1aOps.DefaultExts, [])>;
def HasV9_3aOps : Architecture64<9, 3, "a", "v9.3a",
  [HasV8_8aOps, HasV9_2aOps],
  !listconcat(HasV9_2aOps.DefaultExts, [FeatureMOPS, FeatureHBC])>;
def HasV9_4aOps : Architecture64<9, 4, "a", "v9.4a",
  [HasV8_9aOps, HasV9_3aOps],
  !listconcat(HasV9_3aOps.DefaultExts, [FeatureSPECRES2, FeatureCSSC,
    FeatureRASv2])>;
def HasV9_5aOps : Architecture64<9, 5, "a", "v9.5a",
  [HasV9_4aOps, FeatureCPA],
  !listconcat(HasV9_4aOps.DefaultExts, [FeatureCPA,  FeatureLUT, FeatureFAMINMAX])>;
def HasV8_0rOps : Architecture64<8, 0, "r", "v8r",
  [ //v8.1
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
  ],
  // For v8-R, we do not enable crypto and align with GCC that enables a more
  // minimal set of optional architecture extensions.
  !listconcat(
    !listremove(HasV8_5aOps.DefaultExts, [FeatureLSE]),
    [FeatureSSBS, FeatureFullFP16, FeatureFP16FML, FeatureSB]
  )>;

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


// Only intended to be used by disassemblers.
def FeatureAll
    : SubtargetFeature<"all", "IsAll", "true", "Enable all instructions">;
