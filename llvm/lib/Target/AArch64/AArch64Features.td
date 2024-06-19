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
//
// If Function MultiVersioning (FMV) properties are left at their defaults
// (FEAT_INIT, no dependencies, priority 0) it indiates that this extension is
// not an FMV feature, but can be enabled via the command line (-march, -mcpu,
// etc).
//
// Conversely if the ArchExtKindSpelling is set to AEK_NONE, this indicates
// that a feature is FMV-only, and can not be selected on the command line.
// Such extensions should be added via FMVOnlyExtension.
class Extension<
  string TargetFeatureName,            // String used for -target-feature and -march, unless overridden.
  string Spelling,                     // The XYZ in HasXYZ and AEK_XYZ.
  string Desc,                         // Description.
  list<SubtargetFeature> Implies = [], // List of dependent features.
  // FMV properties
  string _FMVBit = "FEAT_INIT",        // FEAT_INIT is repurposed to indicate "not an FMV feature"
  string _FMVDependencies = "",
  int _FMVPriority = 0
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

    // Function MultiVersioning (FMV) properties

    // A C++ expression giving the number of the bit in the FMV ABI.
    // Currently this is given as a value from the enum "CPUFeatures".
    // If this is not set, it indicates that this is not an FMV extension.
    string FMVBit = _FMVBit;

    // List of features that this feature depends on.
    // FIXME generate this from Implies.
    string FMVDependencies = _FMVDependencies;

    // The FMV priority
    int FMVPriority = _FMVPriority;
}

// Some extensions are available for FMV but can not be controlled via the
// command line. These entries:
//  - are SubtargetFeatures, so they have (unused) FieldNames on the subtarget
//    e.g. HasFMVOnlyFEAT_XYZ
//  - have incorrect (empty) Implies fields, because the code that handles FMV
//    ignores these dependencies and looks only at FMVDependencies.
//  - have no description.
// 
// In the generated data structures for extensions (ExtensionInfo), AEK_NONE is
// used to indicate that a feature is FMV only. Therefore ArchExtKindSpelling is
// manually overridden here.
class FMVOnlyExtension<string FMVBit, string Name, string Deps, int Priority>
  : Extension<Name, "FMVOnly"#FMVBit, "", [], FMVBit, Deps, Priority> {
    let ArchExtKindSpelling = "AEK_NONE"; // AEK_NONE indicates FMV-only feature
}

def : FMVOnlyExtension<"FEAT_DGH", "dgh", "", 260>;
def : FMVOnlyExtension<"FEAT_DPB", "dpb", "+ccpp", 190>;
def : FMVOnlyExtension<"FEAT_DPB2", "dpb2", "+ccpp,+ccdp", 200>;
def : FMVOnlyExtension<"FEAT_EBF16", "ebf16", "+bf16", 290>;
def : FMVOnlyExtension<"FEAT_FLAGM2", "flagm2", "+flagm,+altnzcv", 30>;
def : FMVOnlyExtension<"FEAT_FRINTTS", "frintts", "+fptoint", 250>;
def : FMVOnlyExtension<"FEAT_LS64_ACCDATA", "ls64_accdata", "+ls64", 540>;
def : FMVOnlyExtension<"FEAT_LS64_V", "ls64_v", "", 530>;
def : FMVOnlyExtension<"FEAT_MEMTAG2", "memtag2", "+mte", 450>;
def : FMVOnlyExtension<"FEAT_MEMTAG3", "memtag3", "+mte", 460>;
def : FMVOnlyExtension<"FEAT_PMULL", "pmull", "+aes,+fp-armv8,+neon", 160>;
def : FMVOnlyExtension<"FEAT_RCPC2", "rcpc2", "+rcpc", 240>;
def : FMVOnlyExtension<"FEAT_RPRES", "rpres", "", 300>;
def : FMVOnlyExtension<"FEAT_SHA1", "sha1", "+fp-armv8,+neon", 120>;
def : FMVOnlyExtension<"FEAT_SSBS2", "ssbs2", "+ssbs", 500>;
def : FMVOnlyExtension<"FEAT_SVE_BF16", "sve-bf16", "+sve,+bf16,+fullfp16,+fp-armv8,+neon", 320>;
def : FMVOnlyExtension<"FEAT_SVE_EBF16", "sve-ebf16", "+sve,+bf16,+fullfp16,+fp-armv8,+neon", 330>;
def : FMVOnlyExtension<"FEAT_SVE_I8MM", "sve-i8mm", "+sve,+i8mm,+fullfp16,+fp-armv8,+neon", 340>;
def : FMVOnlyExtension<"FEAT_SVE_PMULL128", "sve2-pmull128", "+sve2,+sve,+sve2-aes,+fullfp16,+fp-armv8,+neon", 390>;


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
  "Enable ARMv8 (FEAT_FP)", [],
  "FEAT_FP", "+fp-armv8,+neon", 90>;

let ArchExtKindSpelling = "AEK_SIMD", MArchName = "simd" in
def FeatureNEON : Extension<"neon", "NEON",
  "Enable Advanced SIMD instructions (FEAT_AdvSIMD)", [FeatureFPARMv8],
  "FEAT_SIMD", "+fp-armv8,+neon", 100>;

def FeatureSHA2 : Extension<
    "sha2", "SHA2",
    "Enable SHA1 and SHA256 support (FEAT_SHA1, FEAT_SHA256)", [FeatureNEON],
    "FEAT_SHA2", "+sha2,+fp-armv8,+neon", 130>;

def FeatureAES : Extension<
    "aes", "AES",
    "Enable AES support (FEAT_AES, FEAT_PMULL)", [FeatureNEON],
    "FEAT_AES", "+fp-armv8,+neon", 150>;

// Crypto has been split up and any combination is now valid (see the
// crypto definitions above). Also, crypto is now context sensitive:
// it has a different meaning for e.g. Armv8.4 than it has for Armv8.2.
// Therefore, we rely on Clang, the user interfacing tool, to pass on the
// appropriate crypto options. But here in the backend, crypto has very little
// meaning anymore. We kept the Crypto definition here for backward
// compatibility, and now imply features SHA2 and AES, which was the
// "traditional" meaning of Crypto.
let FMVDependencies = "+aes,+sha2" in
def FeatureCrypto : Extension<"crypto", "Crypto",
  "Enable cryptographic instructions", [FeatureNEON, FeatureSHA2, FeatureAES]>;

def FeatureCRC : Extension<"crc", "CRC",
  "Enable ARMv8 CRC-32 checksum instructions (FEAT_CRC32)", [],
  "FEAT_CRC", "+crc", 110>;

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
  "Enable ARMv8.1 Large System Extension (LSE) atomic instructions (FEAT_LSE)", [],
  "FEAT_LSE", "+lse", 80>;

let MArchAlias = "rdma" in
def FeatureRDM : Extension<"rdm", "RDM",
  "Enable ARMv8.1 Rounding Double Multiply Add/Subtract instructions (FEAT_RDM)",
  [FeatureNEON],
  "FEAT_RDM", "+rdm,+fp-armv8,+neon", 108>;

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
    "Enable SM3 and SM4 support (FEAT_SM4, FEAT_SM3)", [FeatureNEON],
    "FEAT_SM4", "+sm4,+fp-armv8,+neon", 106>;

def FeatureSHA3 : Extension<
    "sha3", "SHA3",
    "Enable SHA512 and SHA3 support (FEAT_SHA3, FEAT_SHA512)", [FeatureNEON, FeatureSHA2],
    "FEAT_SHA3", "+sha3,+sha2,+fp-armv8,+neon", 140>;

def FeatureRAS : Extension<"ras", "RAS",
  "Enable ARMv8 Reliability, Availability and Serviceability Extensions (FEAT_RAS, FEAT_RASv1p1)">;

let ArchExtKindSpelling = "AEK_FP16", MArchName = "fp16" in
def FeatureFullFP16 : Extension<"fullfp16", "FullFP16",
  "Full FP16 (FEAT_FP16)", [FeatureFPARMv8],
  "FEAT_FP16", "+fullfp16,+fp-armv8,+neon", 170>;

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
  "Enable Scalable Vector Extension (SVE) instructions (FEAT_SVE)", [FeatureFullFP16],
  "FEAT_SVE", "+sve,+fullfp16,+fp-armv8,+neon", 310>;

let ArchExtKindSpelling = "AEK_I8MM" in
def FeatureMatMulInt8 : Extension<"i8mm", "MatMulInt8",
    "Enable Matrix Multiply Int8 Extension (FEAT_I8MM)", [],
    "FEAT_I8MM", "+i8mm", 270>;

let ArchExtKindSpelling = "AEK_F32MM" in
def FeatureMatMulFP32 : Extension<"f32mm", "MatMulFP32",
    "Enable Matrix Multiply FP32 Extension (FEAT_F32MM)", [FeatureSVE],
    "FEAT_SVE_F32MM", "+sve,+f32mm,+fullfp16,+fp-armv8,+neon", 350>;

let ArchExtKindSpelling = "AEK_F64MM" in
def FeatureMatMulFP64 : Extension<"f64mm", "MatMulFP64",
    "Enable Matrix Multiply FP64 Extension (FEAT_F64MM)", [FeatureSVE],
    "FEAT_SVE_F64MM", "+sve,+f64mm,+fullfp16,+fp-armv8,+neon", 360>;

//===----------------------------------------------------------------------===//
//  Armv8.3 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureRCPC : Extension<"rcpc", "RCPC",
    "Enable support for RCPC extension (FEAT_LRCPC)", [],
    "FEAT_RCPC", "+rcpc", 230>;

def FeaturePAuth : Extension<
    "pauth", "PAuth",
    "Enable v8.3-A Pointer Authentication extension (FEAT_PAuth)">;

let ArchExtKindSpelling = "AEK_JSCVT", MArchName = "jscvt" in
def FeatureJS : Extension<
    "jsconv", "JS",
    "Enable v8.3-A JavaScript FP conversion instructions (FEAT_JSCVT)",
    [FeatureFPARMv8],
    "FEAT_JSCVT", "+fp-armv8,+neon,+jsconv", 210>;

def FeatureCCIDX : SubtargetFeature<
    "ccidx", "HasCCIDX", "true",
    "Enable v8.3-A Extend of the CCSIDR number of sets (FEAT_CCIDX)">;

let ArchExtKindSpelling = "AEK_FCMA", MArchName = "fcma" in
def FeatureComplxNum : Extension<
    "complxnum", "ComplxNum",
    "Enable v8.3-A Floating-point complex number support (FEAT_FCMA)",
    [FeatureNEON],
    "FEAT_FCMA", "+fp-armv8,+neon,+complxnum", 220>;

def FeatureNV : SubtargetFeature<
    "nv", "HasNV", "true",
    "Enable v8.4-A Nested Virtualization Enchancement (FEAT_NV, FEAT_NV2)">;

//===----------------------------------------------------------------------===//
//  Armv8.4 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureLSE2 : SubtargetFeature<"lse2", "HasLSE2", "true",
  "Enable ARMv8.4 Large System Extension 2 (LSE2) atomicity rules (FEAT_LSE2)">;

def FeatureFP16FML : Extension<"fp16fml", "FP16FML",
  "Enable FP16 FML instructions (FEAT_FHM)", [FeatureFullFP16],
  "FEAT_FP16FML", "+fp16fml,+fullfp16,+fp-armv8,+neon", 175>;

def FeatureDotProd : Extension<
    "dotprod", "DotProd",
    "Enable dot product support (FEAT_DotProd)", [FeatureNEON],
    "FEAT_DOTPROD", "+dotprod,+fp-armv8,+neon", 104>;

def FeatureMPAM : SubtargetFeature<
    "mpam", "HasMPAM", "true",
    "Enable v8.4-A Memory system Partitioning and Monitoring extension (FEAT_MPAM)">;

def FeatureDIT : Extension<
    "dit", "DIT",
    "Enable v8.4-A Data Independent Timing instructions (FEAT_DIT)", [],
    "FEAT_DIT", "+dit", 180>;

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
    "Enable v8.4-A Flag Manipulation Instructions (FEAT_FlagM)", [],
    "FEAT_FLAGM", "+flagm", 20>;

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
  "Enable v8.5 Speculation Barrier (FEAT_SB)", [],
  "FEAT_SB", "+sb", 470>;

def FeatureSSBS : Extension<"ssbs", "SSBS",
  "Enable Speculative Store Bypass Safe bit (FEAT_SSBS, FEAT_SSBS2)", [],
  "FEAT_SSBS", "", 490>;

def FeaturePredRes : Extension<"predres", "PredRes",
  "Enable v8.5a execution and data prediction invalidation instructions (FEAT_SPECRES)", [],
  "FEAT_PREDRES", "+predres", 480>;

def FeatureCacheDeepPersist : SubtargetFeature<"ccdp", "CCDP", "true",
    "Enable v8.5 Cache Clean to Point of Deep Persistence (FEAT_DPB2)" >;

let ArchExtKindSpelling = "AEK_NONE" in
def FeatureBranchTargetId : Extension<"bti", "BTI",
    "Enable Branch Target Identification (FEAT_BTI)", [],
    "FEAT_BTI", "+bti", 510>;

let ArchExtKindSpelling = "AEK_RAND", MArchName = "rng" in
def FeatureRandGen : Extension<"rand", "RandGen",
    "Enable Random Number generation instructions (FEAT_RNG)", [],
    "FEAT_RNG", "+rand", 10>;

// NOTE: "memtag" means FEAT_MTE + FEAT_MTE2 for -march or
// __attribute((target(...))), but only FEAT_MTE for FMV.
let MArchName = "memtag" in
def FeatureMTE : Extension<"mte", "MTE",
    "Enable Memory Tagging Extension (FEAT_MTE, FEAT_MTE2)", [],
    "FEAT_MEMTAG", "", 440>;

//===----------------------------------------------------------------------===//
//  Armv8.6 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureBF16 : Extension<"bf16", "BF16",
    "Enable BFloat16 Extension (FEAT_BF16)", [],
    "FEAT_BF16", "+bf16", 280>;

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

def FeatureWFxT : Extension<"wfxt", "WFxT",
    "Enable Armv8.7-A WFET and WFIT instruction (FEAT_WFxT)", [],
    "FEAT_WFXT", "+wfxt", 550>;

def FeatureHCX : SubtargetFeature<
    "hcx", "HasHCX", "true", "Enable Armv8.7-A HCRX_EL2 system register (FEAT_HCX)">;

def FeatureLS64 : Extension<"ls64", "LS64",
    "Enable Armv8.7-A LD64B/ST64B Accelerator Extension (FEAT_LS64, FEAT_LS64_V, FEAT_LS64_ACCDATA)", [],
    "FEAT_LS64", "", 520>;

def FeatureSPE_EEF : SubtargetFeature<"spe-eef", "HasSPE_EEF",
    "true", "Enable extra register in the Statistical Profiling Extension (FEAT_SPEv1p2)">;

//===----------------------------------------------------------------------===//
//  Armv8.8 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureHBC : Extension<"hbc", "HBC",
    "Enable Armv8.8-A Hinted Conditional Branches Extension (FEAT_HBC)">;

def FeatureMOPS : Extension<"mops", "MOPS",
    "Enable Armv8.8-A memcpy and memset acceleration instructions (FEAT_MOPS)", [],
    "FEAT_MOPS", "+mops", 650>;

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
    [FeatureRCPC_IMMO],
    "FEAT_RCPC3", "+rcpc,+rcpc3", 241>;

def FeatureTHE : Extension<"the", "THE",
    "Enable Armv8.9-A Translation Hardening Extension (FEAT_THE)">;

//===----------------------------------------------------------------------===//
//  Armv9.0 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureUseScalarIncVL : SubtargetFeature<"use-scalar-inc-vl",
  "UseScalarIncVL", "true", "Prefer inc/dec over add+cnt">;

def FeatureSVE2 : Extension<"sve2", "SVE2",
  "Enable Scalable Vector Extension 2 (SVE2) instructions (FEAT_SVE2)",
  [FeatureSVE, FeatureUseScalarIncVL],
  "FEAT_SVE2", "+sve2,+sve,+fullfp16,+fp-armv8,+neon", 370>;

def FeatureSVE2AES : Extension<"sve2-aes", "SVE2AES",
  "Enable AES SVE2 instructions (FEAT_SVE_AES, FEAT_SVE_PMULL128)",
  [FeatureSVE2, FeatureAES],
  "FEAT_SVE_AES", "+sve2,+sve,+sve2-aes,+fullfp16,+fp-armv8,+neon", 380>;

def FeatureSVE2SM4 : Extension<"sve2-sm4", "SVE2SM4",
  "Enable SM4 SVE2 instructions (FEAT_SVE_SM4)", [FeatureSVE2, FeatureSM4],
  "FEAT_SVE_SM4", "+sve2,+sve,+sve2-sm4,+fullfp16,+fp-armv8,+neon", 420>;

def FeatureSVE2SHA3 : Extension<"sve2-sha3", "SVE2SHA3",
  "Enable SHA3 SVE2 instructions (FEAT_SVE_SHA3)", [FeatureSVE2, FeatureSHA3],
  "FEAT_SVE_SHA3", "+sve2,+sve,+sve2-sha3,+fullfp16,+fp-armv8,+neon", 410>;

def FeatureSVE2BitPerm : Extension<"sve2-bitperm", "SVE2BitPerm",
  "Enable bit permutation SVE2 instructions (FEAT_SVE_BitPerm)", [FeatureSVE2],
  "FEAT_SVE_BITPERM", "+sve2,+sve,+sve2-bitperm,+fullfp16,+fp-armv8,+neon", 400>;

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
  "Enable Scalable Matrix Extension (SME) (FEAT_SME)", [FeatureBF16, FeatureUseScalarIncVL],
  "FEAT_SME", "+sme,+bf16", 430>;

def FeatureSMEF64F64 : Extension<"sme-f64f64", "SMEF64F64",
  "Enable Scalable Matrix Extension (SME) F64F64 instructions (FEAT_SME_F64F64)", [FeatureSME],
  "FEAT_SME_F64", "+sme,+sme-f64f64,+bf16", 560>;

def FeatureSMEI16I64 : Extension<"sme-i16i64", "SMEI16I64",
  "Enable Scalable Matrix Extension (SME) I16I64 instructions (FEAT_SME_I16I64)", [FeatureSME],
  "FEAT_SME_I64", "+sme,+sme-i16i64,+bf16", 570>;

def FeatureSMEFA64 : Extension<"sme-fa64", "SMEFA64",
  "Enable the full A64 instruction set in streaming SVE mode (FEAT_SME_FA64)", [FeatureSME, FeatureSVE2]>;

//===----------------------------------------------------------------------===//
//  Armv9.3 Architecture Extensions
//===----------------------------------------------------------------------===//

def FeatureSME2 : Extension<"sme2", "SME2",
  "Enable Scalable Matrix Extension 2 (SME2) instructions", [FeatureSME],
  "FEAT_SME2", "+sme2,+sme,+bf16", 580>;

def FeatureMEC : SubtargetFeature<"mec", "HasMEC",
    "true", "Enable Memory Encryption Contexts Extension", [FeatureRME]>;

//===----------------------------------------------------------------------===//
//  Armv9.4 Architecture Extensions
//===----------------------------------------------------------------------===//

let FMVDependencies = "+sve2p1,+sve2,+sve,+fullfp16,+fp-armv8,+neon" in
def FeatureSVE2p1: Extension<"sve2p1", "SVE2p1",
  "Enable Scalable Vector Extension 2.1 instructions", [FeatureSVE2]>;

def FeatureB16B16 : Extension<"b16b16", "B16B16",
  "Enable SVE2.1 or SME2.1 non-widening BFloat16 to BFloat16 instructions (FEAT_B16B16)", [FeatureBF16]>;

let FMVDependencies = "+sme2,+sme-f16f16" in
def FeatureSMEF16F16 : Extension<"sme-f16f16", "SMEF16F16",
  "Enable SME non-widening Float16 instructions (FEAT_SME_F16F16)", [FeatureSME2]>;

let FMVDependencies = "+sme2p1,+sme2,+sme,+bf16" in
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

let FMVDependencies = "+sme2" in
def FeatureSSVE_FP8FMA : Extension<"ssve-fp8fma", "SSVE_FP8FMA",
  "Enable SVE2 fp8 multiply-add instructions (FEAT_SSVE_FP8FMA)", [FeatureSME2, FeatureFP8]>;

def FeatureFP8DOT4: Extension<"fp8dot4", "FP8DOT4",
   "Enable fp8 4-way dot instructions (FEAT_FP8DOT4)", [FeatureFP8FMA]>;
  
def FeatureFP8DOT2: Extension<"fp8dot2", "FP8DOT2",
   "Enable fp8 2-way dot instructions (FEAT_FP8DOT2)", [FeatureFP8DOT4]>;

let FMVDependencies = "+sme2" in
def FeatureSSVE_FP8DOT4 : Extension<"ssve-fp8dot4", "SSVE_FP8DOT4",
  "Enable SVE2 fp8 4-way dot product instructions (FEAT_SSVE_FP8DOT4)", [FeatureSSVE_FP8FMA]>;

let FMVDependencies = "+sme2" in
def FeatureSSVE_FP8DOT2 : Extension<"ssve-fp8dot2", "SSVE_FP8DOT2",
  "Enable SVE2 fp8 2-way dot product instructions (FEAT_SSVE_FP8DOT2)", [FeatureSSVE_FP8DOT4]>;

def FeatureSME_LUTv2 : Extension<"sme-lutv2", "SME_LUTv2",
  "Enable Scalable Matrix Extension (SME) LUTv2 instructions (FEAT_SME_LUTv2)">;

let FMVDependencies = "+sme2,+fp8" in
def FeatureSMEF8F32 : Extension<"sme-f8f32", "SMEF8F32",
  "Enable Scalable Matrix Extension (SME) F8F32 instructions (FEAT_SME_F8F32)", [FeatureSME2, FeatureFP8]>;

let FMVDependencies = "+fp8,+sme2" in
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
  [HasV8_5aOps, FeatureMEC, FeatureSVE2],
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
    : SubtargetFeature<"all", "IsAll", "true", "Enable all instructions", []>;
