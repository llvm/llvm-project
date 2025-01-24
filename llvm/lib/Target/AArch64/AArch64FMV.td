//=------ AArch64FMV.td - Describe AArch64 FMV Features ------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Function MultiVersioning (FMV) properties. FMV features are accepted by the
// attributes target_version and target_clones, and they correspond to a mapping
// from the FMV feature name to:
//  - A bit in the FMV ABI, as defined by the ACLE.
//  - The FMV priority, as defined by the ACLE.
//  - A list of backend features.
// 
// The list of backend features is not a set of dependencies; it is specific to
// LLVM and indicates how to do codegen when the FMV feature is present.
// 
// Therefore FMVExtensions are separated from regular AArch64 Extensions, which
// encode dependencies between themselves and other SubtargetFeatures.
//===----------------------------------------------------------------------===//


// Something you can add to target_version or target_clones.
class FMVExtension<string name, string enumeration> {
    // Name, as spelled in target_version or target_clones. e.g. "memtag".
    string Name = name;

    // A C++ expression giving the number of the bit in the FMV ABI.
    // Currently this is given as a value from the enum "CPUFeatures".
    string FeatureBit = "FEAT_" # enumeration;

    // SubtargetFeature enabled for codegen when this FMV feature is present.
    string BackendFeature = name;

    // A C++ expression giving the number of the priority bit.
    // Currently this is given as a value from the enum "FeatPriorities".
    string PriorityBit = "PRIOR_" # enumeration;
}

def : FMVExtension<"aes", "PMULL">;
def : FMVExtension<"bf16", "BF16">;
def : FMVExtension<"bti", "BTI">;
def : FMVExtension<"crc", "CRC">;
def : FMVExtension<"dit", "DIT">;
def : FMVExtension<"dotprod", "DOTPROD">;
let BackendFeature = "ccpp" in def : FMVExtension<"dpb", "DPB">;
let BackendFeature = "ccdp" in def : FMVExtension<"dpb2", "DPB2">;
def : FMVExtension<"f32mm", "SVE_F32MM">;
def : FMVExtension<"f64mm", "SVE_F64MM">;
def : FMVExtension<"fcma", "FCMA">;
def : FMVExtension<"flagm", "FLAGM">;
let BackendFeature = "altnzcv" in def : FMVExtension<"flagm2", "FLAGM2">;
def : FMVExtension<"fp", "FP">;
def : FMVExtension<"fp16", "FP16">;
def : FMVExtension<"fp16fml", "FP16FML">;
let BackendFeature = "fptoint" in def : FMVExtension<"frintts", "FRINTTS">;
def : FMVExtension<"i8mm", "I8MM">;
def : FMVExtension<"jscvt", "JSCVT">;
def : FMVExtension<"ls64", "LS64_ACCDATA">;
def : FMVExtension<"lse", "LSE">;
def : FMVExtension<"memtag", "MEMTAG2">;
def : FMVExtension<"mops", "MOPS">;
def : FMVExtension<"predres", "PREDRES">;
def : FMVExtension<"rcpc", "RCPC">;
let BackendFeature = "rcpc-immo" in def : FMVExtension<"rcpc2", "RCPC2">;
def : FMVExtension<"rcpc3", "RCPC3">;
def : FMVExtension<"rdm", "RDM">;
def : FMVExtension<"rng", "RNG">;
def : FMVExtension<"sb", "SB">;
def : FMVExtension<"sha2", "SHA2">;
def : FMVExtension<"sha3", "SHA3">;
def : FMVExtension<"simd", "SIMD">;
def : FMVExtension<"sm4", "SM4">;
def : FMVExtension<"sme", "SME">;
def : FMVExtension<"sme-f64f64", "SME_F64">;
def : FMVExtension<"sme-i16i64", "SME_I64">;
def : FMVExtension<"sme2", "SME2">;
def : FMVExtension<"ssbs", "SSBS2">;
def : FMVExtension<"sve", "SVE">;
def : FMVExtension<"sve2", "SVE2">;
def : FMVExtension<"sve2-aes", "SVE_PMULL128">;
def : FMVExtension<"sve2-bitperm", "SVE_BITPERM">;
def : FMVExtension<"sve2-sha3", "SVE_SHA3">;
def : FMVExtension<"sve2-sm4", "SVE_SM4">;
def : FMVExtension<"wfxt", "WFXT">;
