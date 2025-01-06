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
class FMVExtension<string n, string b, int p> {
    // Name, as spelled in target_version or target_clones. e.g. "memtag".
    string Name = n;

    // A C++ expression giving the number of the bit in the FMV ABI.
    // Currently this is given as a value from the enum "CPUFeatures".
    string Bit = b;

    // SubtargetFeature enabled for codegen when this FMV feature is present.
    string BackendFeature = n;

    // The FMV priority.
    int Priority = p;
}

def : FMVExtension<"aes", "FEAT_PMULL", 150>;
def : FMVExtension<"bf16", "FEAT_BF16", 280>;
def : FMVExtension<"bti", "FEAT_BTI", 510>;
def : FMVExtension<"crc", "FEAT_CRC", 110>;
def : FMVExtension<"dit", "FEAT_DIT", 180>;
def : FMVExtension<"dotprod", "FEAT_DOTPROD", 104>;
let BackendFeature = "ccpp" in def : FMVExtension<"dpb", "FEAT_DPB", 190>;
let BackendFeature = "ccdp" in def : FMVExtension<"dpb2", "FEAT_DPB2", 200>;
def : FMVExtension<"f32mm", "FEAT_SVE_F32MM", 350>;
def : FMVExtension<"f64mm", "FEAT_SVE_F64MM", 360>;
def : FMVExtension<"fcma", "FEAT_FCMA", 220>;
def : FMVExtension<"flagm", "FEAT_FLAGM", 20>;
let BackendFeature = "altnzcv" in def : FMVExtension<"flagm2", "FEAT_FLAGM2", 30>;
def : FMVExtension<"fp", "FEAT_FP", 90>;
def : FMVExtension<"fp16", "FEAT_FP16", 170>;
def : FMVExtension<"fp16fml", "FEAT_FP16FML", 175>;
let BackendFeature = "fptoint" in def : FMVExtension<"frintts", "FEAT_FRINTTS", 250>;
def : FMVExtension<"i8mm", "FEAT_I8MM", 270>;
def : FMVExtension<"jscvt", "FEAT_JSCVT", 210>;
def : FMVExtension<"ls64", "FEAT_LS64_ACCDATA", 520>;
def : FMVExtension<"lse", "FEAT_LSE", 80>;
def : FMVExtension<"memtag", "FEAT_MEMTAG2", 440>;
def : FMVExtension<"mops", "FEAT_MOPS", 650>;
def : FMVExtension<"predres", "FEAT_PREDRES", 480>;
def : FMVExtension<"rcpc", "FEAT_RCPC", 230>;
let BackendFeature = "rcpc-immo" in def : FMVExtension<"rcpc2", "FEAT_RCPC2", 240>;
def : FMVExtension<"rcpc3", "FEAT_RCPC3", 241>;
def : FMVExtension<"rdm", "FEAT_RDM", 108>;
def : FMVExtension<"rng", "FEAT_RNG", 10>;
def : FMVExtension<"sb", "FEAT_SB", 470>;
def : FMVExtension<"sha2", "FEAT_SHA2", 130>;
def : FMVExtension<"sha3", "FEAT_SHA3", 140>;
def : FMVExtension<"simd", "FEAT_SIMD", 100>;
def : FMVExtension<"sm4", "FEAT_SM4", 106>;
def : FMVExtension<"sme", "FEAT_SME", 430>;
def : FMVExtension<"sme-f64f64", "FEAT_SME_F64", 560>;
def : FMVExtension<"sme-i16i64", "FEAT_SME_I64", 570>;
def : FMVExtension<"sme2", "FEAT_SME2", 580>;
def : FMVExtension<"ssbs", "FEAT_SSBS2", 490>;
def : FMVExtension<"sve", "FEAT_SVE", 310>;
def : FMVExtension<"sve2", "FEAT_SVE2", 370>;
def : FMVExtension<"sve2-aes", "FEAT_SVE_PMULL128", 380>;
def : FMVExtension<"sve2-bitperm", "FEAT_SVE_BITPERM", 400>;
def : FMVExtension<"sve2-sha3", "FEAT_SVE_SHA3", 410>;
def : FMVExtension<"sve2-sm4", "FEAT_SVE_SM4", 420>;
def : FMVExtension<"wfxt", "FEAT_WFXT", 550>;
