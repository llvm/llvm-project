//===-- ROCDLEnums.td - ROCDL IR dialect enums ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCDLIR_ENUMS
#define ROCDLIR_ENUMS

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// ROCDL enum definitions
//===----------------------------------------------------------------------===//

class ROCDL_I32Enum<string name, string summary, list<EnumCase> cases>
    : I32Enum<name, summary, cases> {
  let cppNamespace = "::mlir::ROCDL";
}

class ROCDL_I16BitEnum<string name, string summary,
                       list<BitEnumCaseBase> cases>
    : I16BitEnum<name, summary, cases> {
  let cppNamespace = "::mlir::ROCDL";
}

class ROCDL_I32BitEnum<string name, string summary,
                       list<BitEnumCaseBase> cases>
    : I32BitEnum<name, summary, cases> {
  let cppNamespace = "::mlir::ROCDL";
}

//===----------------------------------------------------------------------===//
// Buffer out-of-bounds mode enum.
//===----------------------------------------------------------------------===//

// Keep these values in sync with OOBFlagValue in AMDGPUTargetMachine.cpp and
// the LLVM IR verifier's AMDGPU module flag checks.
def ROCDL_BufferOOBModeAny : I32EnumCase<"Any", 0, "any">;
def ROCDL_BufferOOBModeRelaxed : I32EnumCase<"Relaxed", 1, "relaxed">;
def ROCDL_BufferOOBModeStrict : I32EnumCase<"Strict", 2, "strict">;

def ROCDL_BufferOOBMode :
    ROCDL_I32Enum<"BufferOOBMode", "ROCDL buffer out-of-bounds mode",
                  [ROCDL_BufferOOBModeAny, ROCDL_BufferOOBModeRelaxed,
                   ROCDL_BufferOOBModeStrict]>;

def ROCDL_MFMAPermB : ROCDL_I32Enum<"MFMAPermB",
    "permutations of the lanes storing B in an MFMA",
    [
      I32EnumCase<"none",            0>,
      I32EnumCase<"bcast_first_32",  1>,
      I32EnumCase<"bcast_second_32", 2>,
      I32EnumCase<"rotate_16_right", 3>,
      I32EnumCase<"bcast_first_16",  4>,
      I32EnumCase<"bcast_second_16", 5>,
      I32EnumCase<"bcast_third_16",  6>,
      I32EnumCase<"bcast_fourth_16", 7>
    ]>;

def ROCDL_MFMANegModifierNone : I32BitEnumCaseNone<"none">;
def ROCDL_MFMANegModifierA : I32BitEnumCaseBit<"neg_a", 0>;
def ROCDL_MFMANegModifierB : I32BitEnumCaseBit<"neg_b", 1>;
def ROCDL_MFMANegModifierC : I32BitEnumCaseBit<"neg_c", 2>;
def ROCDL_MFMANegModifier : ROCDL_I32BitEnum<"MFMANegModifier",
    "negation modifier bitfield for gfx94x double-precision MFMA",
    [
      ROCDL_MFMANegModifierNone,
      ROCDL_MFMANegModifierA,
      ROCDL_MFMANegModifierB,
      ROCDL_MFMANegModifierC
    ]> {
  let printBitEnumQuoted = 0;
}

def ROCDL_MatrixFormat : ROCDL_I32Enum<"MatrixFormat",
    "matrix operand formats selected by scaled MFMA/WMMA format fields",
    [
      I32EnumCase<"fp8_e4m3", 0>,
      I32EnumCase<"fp8_e5m2", 1>,
      I32EnumCase<"fp6_e2m3", 2>,
      I32EnumCase<"fp6_e3m2", 3>,
      I32EnumCase<"fp4_e2m1", 4>
    ]>;

def ROCDL_WMMAMatrixScale : ROCDL_I32Enum<"WMMAMatrixScale",
    "matrix scale row selector",
    [
      I32EnumCase<"row0", 0>,
      I32EnumCase<"row1", 1>
    ]>;

def ROCDL_WMMAMatrixScaleFormat : ROCDL_I32Enum<"WMMAMatrixScaleFormat",
    "matrix scale exponent formats",
    [
      I32EnumCase<"e8",   0>,
      I32EnumCase<"e5m3", 1>,
      I32EnumCase<"e4m3", 2>
    ]>;

def ROCDL_WMMACModifierNone : I16BitEnumCaseNone<"none">;
def ROCDL_WMMACModifierNeg : I16BitEnumCaseBit<"neg", 0>;
def ROCDL_WMMACModifierAbs : I16BitEnumCaseBit<"abs", 1>;
def ROCDL_WMMACModifierNegAbs : BitEnumCaseGroup<"neg_abs",
    [ROCDL_WMMACModifierNeg, ROCDL_WMMACModifierAbs]>;
def ROCDL_WMMACModifier : ROCDL_I16BitEnum<"WMMACModifier",
    "WMMA C operand modifiers",
    [
      ROCDL_WMMACModifierNone,
      ROCDL_WMMACModifierNeg,
      ROCDL_WMMACModifierAbs,
      ROCDL_WMMACModifierNegAbs
    ]> {
  let printBitEnumPrimaryGroups = 1;
  let printBitEnumQuoted = 0;
}

def ROCDL_SchedGroupMaskNone : I32BitEnumCaseNone<"none">;
def ROCDL_SchedGroupMaskNonMemNonSideEffect
    : I32BitEnumCaseBit<"non_mem_non_sideeffect", 0>;
def ROCDL_SchedGroupMaskVALU : I32BitEnumCaseBit<"valu", 1>;
def ROCDL_SchedGroupMaskSALU : I32BitEnumCaseBit<"salu", 2>;
def ROCDL_SchedGroupMaskMFMAWMMA : I32BitEnumCaseBit<"mfma_wmma", 3>;
def ROCDL_SchedGroupMaskAllVMEM : I32BitEnumCaseBit<"all_vmem", 4>;
def ROCDL_SchedGroupMaskVMEMRead : I32BitEnumCaseBit<"vmem_read", 5>;
def ROCDL_SchedGroupMaskVMEMWrite : I32BitEnumCaseBit<"vmem_write", 6>;
def ROCDL_SchedGroupMaskAllDS : I32BitEnumCaseBit<"all_ds", 7>;
def ROCDL_SchedGroupMaskDSRead : I32BitEnumCaseBit<"ds_read", 8>;
def ROCDL_SchedGroupMaskDSWrite : I32BitEnumCaseBit<"ds_write", 9>;
def ROCDL_SchedGroupMaskTranscendental
    : I32BitEnumCaseBit<"transcendental", 10>;
def ROCDL_SchedGroupMaskLDSDMA : I32BitEnumCaseBit<"ldsdma", 11>;
def ROCDL_SchedGroupMaskAll : BitEnumCaseGroup<"all",
    [ROCDL_SchedGroupMaskNonMemNonSideEffect, ROCDL_SchedGroupMaskVALU,
     ROCDL_SchedGroupMaskSALU, ROCDL_SchedGroupMaskMFMAWMMA,
     ROCDL_SchedGroupMaskAllVMEM, ROCDL_SchedGroupMaskVMEMRead,
     ROCDL_SchedGroupMaskVMEMWrite, ROCDL_SchedGroupMaskAllDS,
     ROCDL_SchedGroupMaskDSRead, ROCDL_SchedGroupMaskDSWrite,
     ROCDL_SchedGroupMaskTranscendental, ROCDL_SchedGroupMaskLDSDMA]>;
def ROCDL_SchedGroupMask : ROCDL_I32BitEnum<"SchedGroupMask",
    "instruction type mask for scheduling barriers",
    [
      ROCDL_SchedGroupMaskNone,
      ROCDL_SchedGroupMaskNonMemNonSideEffect,
      ROCDL_SchedGroupMaskVALU,
      ROCDL_SchedGroupMaskSALU,
      ROCDL_SchedGroupMaskMFMAWMMA,
      ROCDL_SchedGroupMaskAllVMEM,
      ROCDL_SchedGroupMaskVMEMRead,
      ROCDL_SchedGroupMaskVMEMWrite,
      ROCDL_SchedGroupMaskAllDS,
      ROCDL_SchedGroupMaskDSRead,
      ROCDL_SchedGroupMaskDSWrite,
      ROCDL_SchedGroupMaskTranscendental,
      ROCDL_SchedGroupMaskLDSDMA,
      ROCDL_SchedGroupMaskAll
    ]> {
  let printBitEnumPrimaryGroups = 1;
  let printBitEnumQuoted = 0;
}

def ROCDL_PreGfx12CachePolicyNone : I32BitEnumCaseNone<"none">;
def ROCDL_PreGfx12CachePolicyGLC : I32BitEnumCaseBit<"glc", 0>;
def ROCDL_PreGfx12CachePolicySLC : I32BitEnumCaseBit<"slc", 1>;
def ROCDL_PreGfx12CachePolicyDLC : I32BitEnumCaseBit<"dlc", 2>;
def ROCDL_PreGfx12CachePolicySWZ : I32BitEnumCaseBit<"swz", 3>;
def ROCDL_PreGfx12CachePolicySCC : I32BitEnumCaseBit<"scc", 4>;
def ROCDL_PreGfx12CachePolicyAll : BitEnumCaseGroup<"all",
    [ROCDL_PreGfx12CachePolicyGLC, ROCDL_PreGfx12CachePolicySLC,
     ROCDL_PreGfx12CachePolicyDLC, ROCDL_PreGfx12CachePolicySCC]>;
def ROCDL_PreGfx12CachePolicyVolatile
    : I32BitEnumCaseBit<"volatile_op", 31, "volatile">;
def ROCDL_PreGfx12CachePolicy : ROCDL_I32BitEnum<"PreGfx12CachePolicy",
    "pre-gfx12 buffer cache policy bits",
    [
      ROCDL_PreGfx12CachePolicyNone,
      ROCDL_PreGfx12CachePolicyGLC,
      ROCDL_PreGfx12CachePolicySLC,
      ROCDL_PreGfx12CachePolicyDLC,
      ROCDL_PreGfx12CachePolicySWZ,
      ROCDL_PreGfx12CachePolicySCC,
      ROCDL_PreGfx12CachePolicyAll,
      ROCDL_PreGfx12CachePolicyVolatile
    ]> {
  let printBitEnumPrimaryGroups = 1;
  let printBitEnumQuoted = 0;
}

def ROCDL_Gfx942CachePolicyNone : I32BitEnumCaseNone<"none">;
def ROCDL_Gfx942CachePolicySC0 : I32BitEnumCaseBit<"sc0", 0>;
def ROCDL_Gfx942CachePolicyNT : I32BitEnumCaseBit<"nt", 1>;
def ROCDL_Gfx942CachePolicySWZ : I32BitEnumCaseBit<"swz", 3>;
def ROCDL_Gfx942CachePolicySC1 : I32BitEnumCaseBit<"sc1", 4>;
def ROCDL_Gfx942CachePolicyVolatile
    : I32BitEnumCaseBit<"volatile_op", 31, "volatile">;
def ROCDL_Gfx942CachePolicy : ROCDL_I32BitEnum<"Gfx942CachePolicy",
    "gfx942 buffer cache policy bits",
    [
      ROCDL_Gfx942CachePolicyNone,
      ROCDL_Gfx942CachePolicySC0,
      ROCDL_Gfx942CachePolicyNT,
      ROCDL_Gfx942CachePolicySWZ,
      ROCDL_Gfx942CachePolicySC1,
      ROCDL_Gfx942CachePolicyVolatile
    ]> {
  let printBitEnumQuoted = 0;
}

def ROCDL_Gfx12CachePolicyNone : I32BitEnumCaseNone<"none">;
def ROCDL_Gfx12CachePolicyNT : I32BitEnumCaseBit<"nt", 0>;
def ROCDL_Gfx12CachePolicyHT : I32BitEnumCaseBit<"ht", 1>;
// Helper bit for temporal encodings. Keep this out of the enum case list: if
// listed there, the generated printer may print this bit directly instead of
// using the named nt_rt/rt_nt/nt_ht/nt_wb groups below.
def ROCDL_Gfx12CachePolicyTemporalBit2 : I32BitEnumCaseBit<"temporal_bit2", 2>;
def ROCDL_Gfx12CachePolicyLU : BitEnumCaseGroup<"lu",
    [ROCDL_Gfx12CachePolicyNT, ROCDL_Gfx12CachePolicyHT]>;
def ROCDL_Gfx12CachePolicyNTRT : BitEnumCaseGroup<"nt_rt",
    [ROCDL_Gfx12CachePolicyTemporalBit2]>;
def ROCDL_Gfx12CachePolicyRTNT : BitEnumCaseGroup<"rt_nt",
    [ROCDL_Gfx12CachePolicyNT, ROCDL_Gfx12CachePolicyTemporalBit2]>;
def ROCDL_Gfx12CachePolicyNTHT : BitEnumCaseGroup<"nt_ht",
    [ROCDL_Gfx12CachePolicyHT, ROCDL_Gfx12CachePolicyTemporalBit2]>;
def ROCDL_Gfx12CachePolicyNTWB : BitEnumCaseGroup<"nt_wb",
    [ROCDL_Gfx12CachePolicyNT, ROCDL_Gfx12CachePolicyHT,
     ROCDL_Gfx12CachePolicyTemporalBit2]>;
def ROCDL_Gfx12CachePolicyScopeSE : I32BitEnumCaseBit<"scope_se", 3>;
def ROCDL_Gfx12CachePolicyScopeDev : I32BitEnumCaseBit<"scope_dev", 4>;
def ROCDL_Gfx12CachePolicyScopeSys : BitEnumCaseGroup<"scope_sys",
    [ROCDL_Gfx12CachePolicyScopeSE, ROCDL_Gfx12CachePolicyScopeDev]>;
def ROCDL_Gfx12CachePolicyNV : I32BitEnumCaseBit<"nv", 5>;
def ROCDL_Gfx12CachePolicySWZ : I32BitEnumCaseBit<"swz", 6>;
def ROCDL_Gfx12CachePolicySCAL : I32BitEnumCaseBit<"scal", 11>;
def ROCDL_Gfx12CachePolicyVolatile
    : I32BitEnumCaseBit<"volatile_op", 31, "volatile">;
def ROCDL_Gfx12CachePolicy : ROCDL_I32BitEnum<"Gfx12CachePolicy",
    "gfx12 cache policy bits",
    [
      ROCDL_Gfx12CachePolicyNone,
      ROCDL_Gfx12CachePolicyNT,
      ROCDL_Gfx12CachePolicyHT,
      ROCDL_Gfx12CachePolicyLU,
      ROCDL_Gfx12CachePolicyNTRT,
      ROCDL_Gfx12CachePolicyRTNT,
      ROCDL_Gfx12CachePolicyNTHT,
      ROCDL_Gfx12CachePolicyNTWB,
      ROCDL_Gfx12CachePolicyScopeSE,
      ROCDL_Gfx12CachePolicyScopeDev,
      ROCDL_Gfx12CachePolicyScopeSys,
      ROCDL_Gfx12CachePolicyNV,
      ROCDL_Gfx12CachePolicySWZ,
      ROCDL_Gfx12CachePolicySCAL,
      ROCDL_Gfx12CachePolicyVolatile
    ]> {
  let printBitEnumPrimaryGroups = 1;
  let printBitEnumQuoted = 0;
}

def ROCDL_Gfx12AtomicCachePolicyNone : I32BitEnumCaseNone<"none">;
// Note: this bit **will** be set automatically by the backend, you do
// **not** need to include it in the policy, it's included for readability.
def ROCDL_Gfx12AtomicCachePolicyReturn
    : I32BitEnumCaseBit<"keep_return", 0, "return">;
def ROCDL_Gfx12AtomicCachePolicyNT : I32BitEnumCaseBit<"nt", 1>;
def ROCDL_Gfx12AtomicCachePolicyCascade
    : I32BitEnumCaseBit<"cascade", 2>;
def ROCDL_Gfx12AtomicCachePolicyScopeSE
    : I32BitEnumCaseBit<"scope_se", 3>;
def ROCDL_Gfx12AtomicCachePolicyScopeDev
    : I32BitEnumCaseBit<"scope_dev", 4>;
def ROCDL_Gfx12AtomicCachePolicyScopeSys
    : BitEnumCaseGroup<"scope_sys",
        [ROCDL_Gfx12AtomicCachePolicyScopeSE,
         ROCDL_Gfx12AtomicCachePolicyScopeDev]>;
def ROCDL_Gfx12AtomicCachePolicyNV : I32BitEnumCaseBit<"nv", 5>;
def ROCDL_Gfx12AtomicCachePolicyVolatile
    : I32BitEnumCaseBit<"volatile_op", 31, "volatile">;
def ROCDL_Gfx12AtomicCachePolicy : ROCDL_I32BitEnum<"Gfx12AtomicCachePolicy",
    "gfx12 atomic cache policy bits",
    [
      ROCDL_Gfx12AtomicCachePolicyNone,
      ROCDL_Gfx12AtomicCachePolicyReturn,
      ROCDL_Gfx12AtomicCachePolicyNT,
      ROCDL_Gfx12AtomicCachePolicyCascade,
      ROCDL_Gfx12AtomicCachePolicyScopeSE,
      ROCDL_Gfx12AtomicCachePolicyScopeDev,
      ROCDL_Gfx12AtomicCachePolicyScopeSys,
      ROCDL_Gfx12AtomicCachePolicyNV,
      ROCDL_Gfx12AtomicCachePolicyVolatile
    ]> {
  let printBitEnumPrimaryGroups = 1;
  let printBitEnumQuoted = 0;
}

#endif // ROCDLIR_ENUMS
