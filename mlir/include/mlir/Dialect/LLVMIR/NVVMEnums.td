//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the NVVM IR enum attributes.
///
//===----------------------------------------------------------------------===//

#ifndef NVVMIR_ENUMS
#define NVVMIR_ENUMS

include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"
include "mlir/Dialect/LLVMIR/NVVMDialect.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Floating-point rounding and saturation enums
//===----------------------------------------------------------------------===//

// Attributes for the floating point rounding modes supported by PTX
def FPRoundingModeNone : I32EnumAttrCase<"NONE", 0, "none">;
def FPRoundingModeRN   : I32EnumAttrCase<"RN",   1, "rn">;
def FPRoundingModeRM   : I32EnumAttrCase<"RM",   2, "rm">;
def FPRoundingModeRP   : I32EnumAttrCase<"RP",   3, "rp">;
def FPRoundingModeRZ   : I32EnumAttrCase<"RZ",   4, "rz">;
def FPRoundingModeRNA  : I32EnumAttrCase<"RNA",  5, "rna">;
def FPRoundingModeRS   : I32EnumAttrCase<"RS",   6, "rs">;

def FPRoundingMode : I32EnumAttr<"FPRoundingMode", "NVVM FPRoundingMode kind",
  [FPRoundingModeNone, FPRoundingModeRN, FPRoundingModeRM,
    FPRoundingModeRP, FPRoundingModeRZ, FPRoundingModeRNA, FPRoundingModeRS]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def FPRoundingModeAttr : EnumAttr<NVVM_Dialect, FPRoundingMode, "fp_rnd_mode">;

def SaturationModeNone   : I32EnumAttrCase<"NONE", 0, "none">;
def SaturationModeFinite : I32EnumAttrCase<"SATFINITE", 1, "satfinite">;
def SaturationModeSat    : I32EnumAttrCase<"SAT", 2, "sat">;

def SaturationMode : I32EnumAttr<"SaturationMode", "NVVM SaturationMode kind",
  [SaturationModeNone, SaturationModeFinite, SaturationModeSat]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def SaturationModeAttr : EnumAttr<NVVM_Dialect, SaturationMode, "sat_mode"> {
  let summary = "Describes the saturation mode";
  let description = [{
    A `nvvm.sat_mode` attribute specifies the saturation mode for instructions
    involving floating points or integers. It can be one of the following
    values:
    - `none`: No saturation is applied.
    - `satfinite`: If the absolute value of input (ignoring sign) is greater
      than the `MAX_NORM` of the specified destination format, then the result
      is the sign-preserved `MAX_NORM` of the destination format and a positive
      `MAX_NORM` in unsigned datatypes for which the destination sign is not
      supported. If the input is `NaN`, then the result can be `NaN` or the
      `MAX_NORM` of the destination format, depending on the format.
    - `sat`: For integer destination types, this limits the value to `MININT..
      MAXINT` and applies to both signed and unsigned integer datatypes. For
      floating point destination types (applies to only `F16`, `F32`, and `F64`
      types), this limits the value to the range `[0.0, 1.0]` and flushes NaN
      results to positive zero.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)
}];
}

//===----------------------------------------------------------------------===//
// Memory space, scope, ordering and proxy enums
//===----------------------------------------------------------------------===//

def MemSpaceGeneric : I32EnumCase<"Generic", 0, "generic">;
def MemSpaceGlobal : I32EnumCase<"Global", 1, "global">;
def MemSpaceShared : I32EnumCase<"Shared", 3, "shared">;
def MemSpaceConstant : I32EnumCase<"Constant", 4, "constant">;
def MemSpaceLocal : I32EnumCase<"Local", 5, "local">;
/// Tensor memory space identifier.
/// Tensor memory is available only in arch-accelerated
/// variants from sm100 onwards.
def MemSpaceTensor : I32EnumCase<"Tensor", 6, "tensor">;
def MemSpaceSharedCluster : I32EnumCase<"SharedCluster", 7, "shared_cluster">;

def NVVMMemorySpace : I32Enum<"NVVMMemorySpace", "NVVM Memory Space",
                  [MemSpaceGeneric, MemSpaceGlobal, MemSpaceShared,   
                   MemSpaceConstant, MemSpaceLocal, MemSpaceTensor, 
                   MemSpaceSharedCluster]> {
  let cppNamespace = "::mlir::NVVM";
}
def NVVMMemorySpaceAttr : 
  EnumAttr<NVVM_Dialect, NVVMMemorySpace, "memory_space", [
    DeclareAttrInterfaceMethods<LLVM_LLVMAddrSpaceAttrInterface>,
    DeclareAttrInterfaceMethods<MemorySpaceAttrInterface>
  ]>;

// Attrs describing the scope of the Memory Operation
def MemScopeKindCTA      : I32EnumAttrCase<"CTA", 0, "cta">;
def MemScopeKindCluster  : I32EnumAttrCase<"CLUSTER", 1, "cluster">;
def MemScopeKindGPU      : I32EnumAttrCase<"GPU", 2, "gpu">;
def MemScopeKindSYS      : I32EnumAttrCase<"SYS", 3, "sys">;

def MemScopeKind : I32EnumAttr<"MemScopeKind", "NVVM Memory Scope kind",
  [MemScopeKindCTA, MemScopeKindCluster, MemScopeKindGPU, MemScopeKindSYS]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MemScopeKindAttr : EnumAttr<NVVM_Dialect, MemScopeKind, "mem_scope">;

// Attrs to disambiguate the cta or cluster space within shared memory
def SharedSpaceCTA     : I32EnumAttrCase<"shared_cta", 0, "cta">;
def SharedSpaceCluster : I32EnumAttrCase<"shared_cluster", 1, "cluster">;
def SharedSpace : I32EnumAttr<"SharedSpace", "Shared memory space",
  [SharedSpaceCTA, SharedSpaceCluster]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def SharedSpaceAttr : EnumAttr<NVVM_Dialect, SharedSpace, "shared_space">;

// Attrs describing the Memory Ordering Semantics
def MemOrderKindWeak     : I32EnumAttrCase<"WEAK", 0, "weak">;
def MemOrderKindRelaxed  : I32EnumAttrCase<"RELAXED", 1, "relaxed">;
def MemOrderKindAcquire  : I32EnumAttrCase<"ACQUIRE", 2, "acquire">;
def MemOrderKindRelease  : I32EnumAttrCase<"RELEASE", 3, "release">;
def MemOrderKindAcqRel   : I32EnumAttrCase<"ACQ_REL", 4, "acq_rel">;
def MemOrderKindSC       : I32EnumAttrCase<"SC", 5, "sc">;
def MemOrderKindMMIO     : I32EnumAttrCase<"MMIO", 6, "mmio">;
def MemOrderKindVolatile : I32EnumAttrCase<"VOLATILE", 7, "volatile">;

def MemOrderKind : I32EnumAttr<"MemOrderKind", "NVVM Memory Ordering kind",
  [MemOrderKindWeak, MemOrderKindRelaxed, MemOrderKindAcquire,
    MemOrderKindRelease, MemOrderKindAcqRel, MemOrderKindSC,
    MemOrderKindMMIO, MemOrderKindVolatile]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MemOrderKindAttr : EnumAttr<NVVM_Dialect, MemOrderKind, "mem_order">;

def ProxyAlias       : I32EnumAttrCase<"alias", 0, "alias">;
def ProxyAsync       : I32EnumAttrCase<"async", 1, "async">;
def ProxyAsyncGlobal : I32EnumAttrCase<"async_global", 2, "async.global">;
def ProxyAsyncShared : I32EnumAttrCase<"async_shared", 3, "async.shared">;
def ProxyTensorMap   : I32EnumAttrCase<"TENSORMAP", 4, "tensormap">;
def ProxyGeneric     : I32EnumAttrCase<"GENERIC", 5, "generic">;
def ProxyKind : I32EnumAttr<"ProxyKind", "Proxy kind",
  [ProxyAlias, ProxyAsync, ProxyAsyncGlobal, ProxyAsyncShared, ProxyTensorMap, ProxyGeneric]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def ProxyKindAttr : EnumAttr<NVVM_Dialect, ProxyKind, "proxy_kind"> {
  let description = [{
    ProxyKind attribute represents a memory proxy which is an abstract label
    applied to a method of memory access. When two memory operations use distinct
    methods of memory access, they are said to be different proxies.
  }];
}

//===----------------------------------------------------------------------===//
// Cache control enums
//===----------------------------------------------------------------------===//

def EvictNormal : I32EnumCase<"EvictNormal", 0, "evict_normal">;
def EvictFirst : I32EnumCase<"EvictFirst", 1, "evict_first">;
def EvictLast : I32EnumCase<"EvictLast", 2, "evict_last">;
def EvictUnchanged : I32EnumCase<"EvictUnchanged", 3, "evict_unchanged">;
def NoAllocate : I32EnumCase<"NoAllocate", 4, "no_allocate">;

def CacheEvictionPriority : I32Enum<"CacheEvictionPriority",
                                    "NVVM Cache Eviction Priority",
                                    [EvictNormal, EvictFirst, EvictLast,   
                                     EvictUnchanged, NoAllocate]> {
  let cppNamespace = "::mlir::NVVM";
}
def CacheEvictionPriorityAttr : EnumAttr<NVVM_Dialect, CacheEvictionPriority, 
                                         "cache_eviction_priority">;

def LoadCacheModifierCA : I32EnumAttrCase<"CA", 0, "ca">;
def LoadCacheModifierCG : I32EnumAttrCase<"CG", 1, "cg">;
def LoadCacheModifierCS : I32EnumAttrCase<"CS", 2, "cs">;
def LoadCacheModifierLU : I32EnumAttrCase<"LU", 3, "lu">;
def LoadCacheModifierCV : I32EnumAttrCase<"CV", 4, "cv">;

def LoadCacheModifierKind : I32EnumAttr<"LoadCacheModifierKind", 
                                "NVVM load cache modifier kind",
  [LoadCacheModifierCA, LoadCacheModifierCG, LoadCacheModifierCS, 
    LoadCacheModifierLU, LoadCacheModifierCV]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
  let description = [{
    Enum attribute of the different kinds of cache operators for load instructions.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#id62)    
  }];
}
def LoadCacheModifierAttr : EnumAttr<NVVM_Dialect, LoadCacheModifierKind, "load_cache_modifier">;

def PrefetchCacheLevelL1 : I32EnumCase<"L1", 0, "L1">;
def PrefetchCacheLevelL2 : I32EnumCase<"L2", 1, "L2">;

def PrefetchCacheLevel : I32Enum<"PrefetchCacheLevel",
                                 "NVVM Prefetch Cache Level",
                                 [PrefetchCacheLevelL1, PrefetchCacheLevelL2]> {
  let cppNamespace = "::mlir::NVVM";
}
def PrefetchCacheLevelAttr : EnumAttr<NVVM_Dialect, PrefetchCacheLevel, "prefetch_cache_level">;

//===----------------------------------------------------------------------===//
// Synchronization, warp-level and control enums
//===----------------------------------------------------------------------===//

// Attrs describing the reduction operations for the barrier operation.
def BarrierReductionPopc : I32EnumAttrCase<"POPC", 0, "popc">;
def BarrierReductionAnd : I32EnumAttrCase<"AND", 1, "and">;
def BarrierReductionOr : I32EnumAttrCase<"OR", 2, "or">;

def BarrierReduction
    : I32EnumAttr<"BarrierReduction", "NVVM barrier reduction operation",
                  [BarrierReductionPopc, BarrierReductionAnd,
                   BarrierReductionOr]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def BarrierReductionAttr
    : EnumAttr<NVVM_Dialect, BarrierReduction, "reduction">;

// Attrs for supported Reduction operations
def ReductionKindNone : I32EnumAttrCase<"NONE", 0, "none">;
def ReductionKindAdd  : I32EnumAttrCase<"ADD", 1, "add">;
def ReductionKindAnd  : I32EnumAttrCase<"AND", 2, "and">;
def ReductionKindMax  : I32EnumAttrCase<"MAX", 3, "max">;
def ReductionKindMin  : I32EnumAttrCase<"MIN", 4, "min">;
def ReductionKindOr   : I32EnumAttrCase<"OR", 5, "or">;
def ReductionKindUmax : I32EnumAttrCase<"UMAX", 6, "umax">;
def ReductionKindUmin : I32EnumAttrCase<"UMIN", 7, "umin">;
def ReductionKindXor  : I32EnumAttrCase<"XOR", 8, "xor">;
def ReductionKindFmin : I32EnumAttrCase<"FMIN", 9, "fmin">;
def ReductionKindFmax : I32EnumAttrCase<"FMAX", 10, "fmax">;

def ReductionKind : I32EnumAttr<"ReductionKind", "NVVM Reduction Kind attribute",
  [ReductionKindAdd, ReductionKindAnd, ReductionKindMax, ReductionKindMin,
   ReductionKindOr, ReductionKindUmax, ReductionKindUmin, ReductionKindXor,
   ReductionKindFmin, ReductionKindFmax]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def ReductionKindAttr : EnumAttr<NVVM_Dialect, ReductionKind, "reduction_kind">;

def SetMaxRegisterActionIncrease : I32EnumAttrCase<"increase", 0>;
def SetMaxRegisterActionDecrease : I32EnumAttrCase<"decrease", 1>;
def SetMaxRegisterAction : I32EnumAttr<"SetMaxRegisterAction", "NVVM set max register action",
  [SetMaxRegisterActionDecrease, SetMaxRegisterActionIncrease]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def SetMaxRegisterActionAttr : EnumAttr<NVVM_Dialect, SetMaxRegisterAction, "action">;

def ShflKindBfly : I32EnumAttrCase<"bfly", 0>;
def ShflKindUp   : I32EnumAttrCase<"up", 1>;
def ShflKindDown : I32EnumAttrCase<"down", 2>;
def ShflKindIdx  : I32EnumAttrCase<"idx", 3>;

def ShflKind : I32EnumAttr<"ShflKind", "NVVM shuffle kind",
  [ShflKindBfly, ShflKindUp, ShflKindDown, ShflKindIdx]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def ShflKindAttr : EnumAttr<NVVM_Dialect, ShflKind, "shfl_kind">;

def VoteSyncKindAny : I32EnumAttrCase<"any", 0>;
def VoteSyncKindAll : I32EnumAttrCase<"all", 1>;
def VoteSyncKindBallot : I32EnumAttrCase<"ballot", 2>;
def VoteSyncKindUni : I32EnumAttrCase<"uni", 3>;

def VoteSyncKind : I32EnumAttr<"VoteSyncKind", "NVVM vote sync kind",
                               [VoteSyncKindAny, VoteSyncKindAll,
                                VoteSyncKindBallot, VoteSyncKindUni]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def VoteSyncKindAttr : EnumAttr<NVVM_Dialect, VoteSyncKind, "vote_sync_kind">;

def MatchSyncKindAny : I32EnumAttrCase<"any", 0>;
def MatchSyncKindAll : I32EnumAttrCase<"all", 1>;

def MatchSyncKind : I32EnumAttr<"MatchSyncKind", "NVVM match sync kind",
  [MatchSyncKindAny, MatchSyncKindAll]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MatchSyncKindAttr : EnumAttr<NVVM_Dialect, MatchSyncKind, "match_sync_kind">;

def GridDepActionWait : I32EnumCase<"wait", 0>;
def GridDepActionLaunchDependent : I32EnumCase<"launch_dependents", 1>;

def GridDepActionKind : I32Enum<"GridDepActionKind", "Action kind for grid dependency control",
  [GridDepActionWait, GridDepActionLaunchDependent]> {
  let cppNamespace = "::mlir::NVVM";
}
def GridDepActionAttr : EnumAttr<NVVM_Dialect, GridDepActionKind, "grid_dep_action">;

def ClusterLaunchControlIsCanceled
  : I32EnumCase<"IS_CANCELED", 0, "is_canceled">;
def ClusterLaunchControlGetFirstCTAIDX
  : I32EnumCase<"GET_FIRST_CTA_ID_X", 1, "get_first_cta_id_x">;
def ClusterLaunchControlGetFirstCTAIDY
  : I32EnumCase<"GET_FIRST_CTA_ID_Y", 2, "get_first_cta_id_y">;
def ClusterLaunchControlGetFirstCTAIDZ
  : I32EnumCase<"GET_FIRST_CTA_ID_Z", 3, "get_first_cta_id_z">;

def ClusterLaunchControlQueryType
  : I32Enum<"ClusterLaunchControlQueryType",
      "NVVM ClusterLaunchControlQueryType", 
      [ClusterLaunchControlIsCanceled, ClusterLaunchControlGetFirstCTAIDX,
      ClusterLaunchControlGetFirstCTAIDY, ClusterLaunchControlGetFirstCTAIDZ]> {
  let cppNamespace = "::mlir::NVVM";
}
def ClusterLaunchControlQueryTypeAttr
  : EnumAttr<NVVM_Dialect,
      ClusterLaunchControlQueryType, "cluster_launch_control_query_type">;

//===----------------------------------------------------------------------===//
// Integer arithmetic and byte permute enums
//===----------------------------------------------------------------------===//

// Attributes for the permute operation modes supported by PTX.
def PermuteModeDefault : I32EnumAttrCase<"DEFAULT", 0, "default">;
def PermuteModeF4E : I32EnumAttrCase<"F4E", 1, "f4e">;
def PermuteModeB4E : I32EnumAttrCase<"B4E", 2, "b4e">;
def PermuteModeRC8 : I32EnumAttrCase<"RC8", 3, "rc8">;
def PermuteModeECL : I32EnumAttrCase<"ECL", 4, "ecl">;
def PermuteModeECR : I32EnumAttrCase<"ECR", 5, "ecr">;
def PermuteModeRC16 : I32EnumAttrCase<"RC16", 6, "rc16">;

def PermuteMode : I32EnumAttr<"PermuteMode", "NVVM permute mode",
                              [PermuteModeDefault, PermuteModeF4E,
                               PermuteModeB4E, PermuteModeRC8, PermuteModeECL,
                               PermuteModeECR, PermuteModeRC16]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def PermuteModeAttr : EnumAttr<NVVM_Dialect, PermuteMode, "permute_mode">;

def DotAccumulateUnsigned : I32EnumAttrCase<"UNSIGNED", 0, "unsigned">;
def DotAccumulateSigned : I32EnumAttrCase<"SIGNED", 1, "signed">;

def DotAccumulateType : I32EnumAttr<"DotAccumulateType",
                              "NVVM DotAccumulateType",
                              [DotAccumulateSigned, DotAccumulateUnsigned]> {
  let cppNamespace = "::mlir::NVVM";
  let genSpecializedAttr = 0;
}
def DotAccumulateTypeAttr : EnumAttr<NVVM_Dialect, DotAccumulateType, "dot_accumulate_type">;

//===----------------------------------------------------------------------===//
// MMA, WMMA and ldmatrix/stmatrix enums
//===----------------------------------------------------------------------===//

/// Enum attribute for binary (b1) MMA operation type
def MMAB1OpNone : I32EnumAttrCase<"none", 0>;
def MMAB1OpXorPopc : I32EnumAttrCase<"xor_popc", 1>;
def MMAB1OpAndPopc : I32EnumAttrCase<"and_popc", 2>;
def MMAB1Op : I32EnumAttr<"MMAB1Op", "MMA binary operations",
  [MMAB1OpNone, MMAB1OpXorPopc, MMAB1OpAndPopc]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMAB1OpAttr : EnumAttr<NVVM_Dialect, MMAB1Op, "mma_b1op">;

/// Enum attribute type for the overflow behavior of MMA integer operations
def MMAIntOverflowWrap : I32EnumAttrCase<"wrapped", 0>;
def MMAIntOverflowSat : I32EnumAttrCase<"satfinite", 1>;
def MMAIntOverflow : I32EnumAttr<"MMAIntOverflow", "MMA overflow options",
  [MMAIntOverflowSat, MMAIntOverflowWrap]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMAIntOverflowAttr : EnumAttr<NVVM_Dialect, MMAIntOverflow, "mma_int_overflow">;

/// MMA kind types (for mixed-precision FP8 operations)
def MMAKindF8F6F4 : I32EnumAttrCase<"f8f6f4", 0>;
def MMAKind : I32EnumAttr<"MMAKind", "MMA operation kind",
  [MMAKindF8F6F4]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMAKindAttr : EnumAttr<NVVM_Dialect, MMAKind, "mma_kind">;

def MMALayoutRow : I32EnumAttrCase<"row", 0>;
def MMALayoutCol : I32EnumAttrCase<"col", 1>;

def MMALayout : I32EnumAttr<"MMALayout", "NVVM MMA layout",
  [MMALayoutRow, MMALayoutCol]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMALayoutAttr : EnumAttr<NVVM_Dialect, MMALayout, "mma_layout">;

/// Enum attribute of the different PTX element types used for MMA operands.
def MMATypeF16  : I32EnumAttrCase<"f16", 0>;
def MMATypeF32  : I32EnumAttrCase<"f32", 1>;
def MMATypeTF32 : I32EnumAttrCase<"tf32", 2>;
def MMATypeU8 : I32EnumAttrCase<"u8", 3>;
def MMATypeS8 : I32EnumAttrCase<"s8", 4>;
def MMATypeS32 : I32EnumAttrCase<"s32", 5>;
def MMATypeB1 : I32EnumAttrCase<"b1", 6>;
def MMATypeU4 : I32EnumAttrCase<"u4", 7>;
def MMATypeS4 : I32EnumAttrCase<"s4", 8>;
def MMATypeBF16 : I32EnumAttrCase<"bf16", 9>;
def MMATypeF64 : I32EnumAttrCase<"f64", 10>;
def MMATypeE4M3 : I32EnumAttrCase<"e4m3", 11>;
def MMATypeE5M2 : I32EnumAttrCase<"e5m2", 12>;
def MMATypeE3M2 : I32EnumAttrCase<"e3m2", 13>;
def MMATypeE2M3 : I32EnumAttrCase<"e2m3", 14>;
def MMATypeE2M1 : I32EnumAttrCase<"e2m1", 15>;

def MMATypes : I32EnumAttr<"MMATypes", "NVVM MMA types",
  [MMATypeF16, MMATypeF32, MMATypeTF32,
  MMATypeBF16, MMATypeS8, MMATypeU8,
  MMATypeS32, MMATypeS4, MMATypeU4,
  MMATypeB1, MMATypeF64,
  MMATypeE4M3, MMATypeE5M2, MMATypeE3M2, MMATypeE2M3, MMATypeE2M1]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMATypesAttr : EnumAttr<NVVM_Dialect, MMATypes, "mma_type">;

def MMAFragA : I32EnumAttrCase<"a", 0>;
def MMAFragB : I32EnumAttrCase<"b", 1>;
def MMAFragC : I32EnumAttrCase<"c", 2>;

def MMAFrag: I32EnumAttr<"MMAFrag", "NVVM MMA frag type",
  [MMAFragA, MMAFragB, MMAFragC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMAFragAttr : EnumAttr<NVVM_Dialect, MMAFrag, "mma_frag">;

def LdStMatrixEltTypeB16 : I32EnumAttrCase<"B16", 0, "b16">;
def LdStMatrixEltTypeB8 : I32EnumAttrCase<"B8", 1, "b8">;
def LdStMatrixEltTypeB8X16_B6X16_P32 : I32EnumAttrCase<"B8X16_B6X16_P32", 2, "b8x16.b6x16_p32">;
def LdStMatrixEltTypeB8X16_B4X16_P64 : I32EnumAttrCase<"B8X16_B4X16_P64", 3, "b8x16.b4x16_p64">;
def LdStMatrixEltTypeS8_S4 : I32EnumAttrCase<"S8_S4", 4, "s8.s4">;

def LdStMatrixEltType : I32EnumAttr<"LdStMatrixEltType", "Element type for ldmatrix and stmatrix",
  [LdStMatrixEltTypeB16, LdStMatrixEltTypeB8,
   LdStMatrixEltTypeB8X16_B6X16_P32, LdStMatrixEltTypeB8X16_B4X16_P64,
   LdStMatrixEltTypeS8_S4]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def LdStMatrixEltTypeAttr : EnumAttr<NVVM_Dialect, LdStMatrixEltType, "ld_st_matrix_elt_type">;

def ScaleVecSize1X  : I32EnumAttrCase<"X1", 0, "x1">;
def ScaleVecSize2X  : I32EnumAttrCase<"X2", 1, "x2">;
def ScaleVecSize4X  : I32EnumAttrCase<"X4", 2, "x4">;

def ScaleVecSize : I32EnumAttr<
  "ScaleVecSize",
  "MMA Scale Vector Sizes",
  [ScaleVecSize1X, ScaleVecSize2X, ScaleVecSize4X]> {
    let cppNamespace = "::mlir::NVVM";
    let genSpecializedAttr = 0;
}
def ScaleVecSizeAttr : EnumAttr<NVVM_Dialect, ScaleVecSize, "scale_vec_size">;

def UE8M0 : I32EnumAttrCase<"UE8M0", 0, "ue8m0">;
def UE4M3 : I32EnumAttrCase<"UE4M3", 1, "ue4m3">;

def BlockScaleFormat : I32EnumAttr<
  "BlockScaleFormat",
  "MMA Block Scale Format",
  [UE8M0, UE4M3]
> {
  let cppNamespace = "::mlir::NVVM";
  let genSpecializedAttr = 0;
}
def BlockScaleFormatAttr : EnumAttr<NVVM_Dialect, BlockScaleFormat, "block_scale_format">;

def MMABlockScaleKindMXF8F6F4 : I32EnumAttrCase<"MXF8F6F4", 0, "mxf8f6f4">;
def MMABlockScaleKindMXF4     : I32EnumAttrCase<"MXF4", 1, "mxf4">;
def MMABlockScaleKindMXF4NVF4 : I32EnumAttrCase<"MXF4NVF4", 2, "mxf4nvf4">;

def MMABlockScaleKind : I32EnumAttr<
  "MMABlockScaleKind",
  "Block Scale Kind",
  [MMABlockScaleKindMXF8F6F4, MMABlockScaleKindMXF4, MMABlockScaleKindMXF4NVF4]> {
    let cppNamespace = "::mlir::NVVM";
    let genSpecializedAttr = 0;
}
def MMABlockScaleKindAttr : EnumAttr<NVVM_Dialect, MMABlockScaleKind, "block_scale_kind"> {
  let description = [{
    The MMABlockScaleKind attribute describes the allowed set of types for matrix A and B in the *.mma.{sp}.block_scale Op. The following are supported types for each kind:

    ```
    +--------------+-------------------------------------------+
    | Matrix Kind  |      supported types for A / B            |
    +--------------+-------------------------------------------+
    | mxf8f6f4     | e4m3, e5m2, e2m3, e3m2, e2m1              |
    | mxf4         | e2m1                                      |
    | mxf4nvf4     | e2m1                                      |
    +--------------+-------------------------------------------+
    ```
  }];
}

//===----------------------------------------------------------------------===//
// WGMMA enums
//===----------------------------------------------------------------------===//

/// Enum attribute type for the negating of input operands
def WGMMAScaleInNeg : I32EnumAttrCase<"neg", -1>;
def WGMMAScaleInOne : I32EnumAttrCase<"one", 1>;
def WGMMAScaleIn : I32EnumAttr<"WGMMAScaleIn", "WGMMA overflow options",
  [WGMMAScaleInOne, WGMMAScaleInNeg]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def WGMMAScaleInAttr : EnumAttr<NVVM_Dialect, WGMMAScaleIn, "wgmma_scale_in">;

/// Enum attribute type for the output operand
def WGMMAScaleOutZero : I32EnumAttrCase<"zero", 0>;
def WGMMAScaleOutOne : I32EnumAttrCase<"one", 1>;
def WGMMAScaleOut : I32EnumAttr<"WGMMAScaleOut", "WGMMA input predicate",
  [WGMMAScaleOutZero, WGMMAScaleOutOne]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def WGMMAScaleOutAttr : EnumAttr<NVVM_Dialect, WGMMAScaleOut, "wgmma_scale_out">;

/// Enum attribute of the different PTX element types used for WGMMA operands.
def WGMMATypeF16  : I32EnumAttrCase<"f16", 0>;
def WGMMATypeTF32 : I32EnumAttrCase<"tf32", 1>;
def WGMMATypeU8 : I32EnumAttrCase<"u8", 2>;
def WGMMATypeS8 : I32EnumAttrCase<"s8", 3>;
def WGMMATypeB1 : I32EnumAttrCase<"b1", 4>;
def WGMMATypeBF16 : I32EnumAttrCase<"bf16", 5>;
def WGMMATypeF8E4M3 : I32EnumAttrCase<"e4m3", 6>;
def WGMMATypeF8E5M2 : I32EnumAttrCase<"e5m2", 7>;
def WGMMATypeF32 : I32EnumAttrCase<"f32", 8>;
def WGMMATypeS32 : I32EnumAttrCase<"s32", 9>;

def WGMMATypes : I32EnumAttr<"WGMMATypes", "NVVM WGMMA types",
  [WGMMATypeF16, WGMMATypeTF32,
    WGMMATypeU8, WGMMATypeS8,
    WGMMATypeB1, WGMMATypeBF16, WGMMATypeF8E4M3, 
    WGMMATypeF8E5M2, WGMMATypeF32, WGMMATypeS32]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def WGMMATypesAttr : EnumAttr<NVVM_Dialect, WGMMATypes, "wgmma_type">;

//===----------------------------------------------------------------------===//
// TMA enums
//===----------------------------------------------------------------------===//

// List of modes supported for TMA Load and Prefetch Ops
def TMALoadModeTile   : I32EnumAttrCase<"TILE", 0, "tile">;
def TMALoadModeIm2Col : I32EnumAttrCase<"IM2COL", 1, "im2col">;
def TMALoadModeIm2ColW : I32EnumAttrCase<"IM2COL_W", 2, "im2col_w">;
def TMALoadModeIm2ColW128 : I32EnumAttrCase<"IM2COL_W_128", 3, "im2col_w_128">;
def TMALoadModeTileGather4 : I32EnumAttrCase<"TILE_GATHER4", 4, "tile_gather4">;

def TMALoadMode : I32EnumAttr<"TMALoadMode", "NVVM TMA Load Mode",
    [TMALoadModeTile, TMALoadModeIm2Col,
     TMALoadModeIm2ColW, TMALoadModeIm2ColW128,
     TMALoadModeTileGather4]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def TMALoadModeAttr : EnumAttr<NVVM_Dialect, TMALoadMode, "tma_load_mode"> {
  let summary = "List of Load-Modes supported for TMA Tensor Ops";
  let description = [{
    TMA Tensor Ops support the following modes, when copying data from
    global memory to shared memory (i.e. load):

    Tile Mode: It's the default mode. The source multi-dimensional tensor
    layout is preserved at the destination.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode)

    Im2col Mode: This mode is used when `im2colOffsets` operands are present.
    The elements in the Bounding Box of the source tensor are rearranged into
    columns at the destination. In this mode, the tensor has to be at least
    3-dimensional. The number of `im2colOffsets` is `dims - 2` where `dims`
    is the dimension of the tensor.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-mode)

    Im2col_W Mode: This mode is similar to Im2Col mode with the restriction that
    elements are accessed across the W dimension only. The number of `im2colOffsets`
    are always two, referred as `wHalo` and `wOffset`.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-w-w128-modes)

    Im2col_W_128 Mode: This mode is similar to Im2Col_W mode with the number of
    elements accessed across the W dimension is always 128 only.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-w-w128-modes)

    Tile_Gather4 Mode: This mode is similar to Tile mode but works only on 2D tensor.
    In gather4 mode, four rows in the source 2D tensor are combined to form a single
    2D tensor at the destination. This mode requires five co-ordinates. The first one
    represents the column-index followed by four row indices.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-scatter4-gather4-modes)
  }];

}

// List of modes supported for TMA Store and Reduction Ops
def TMAStoreModeTile   : I32EnumAttrCase<"TILE", 0, "tile">;
def TMAStoreModeIm2Col : I32EnumAttrCase<"IM2COL", 1, "im2col">;
def TMAStoreModeTileScatter4 : I32EnumAttrCase<"TILE_SCATTER4", 2, "tile_scatter4">;
def TMAStoreModeIm2ColW : I32EnumAttrCase<"IM2COL_W", 3, "im2col_w">;

def TMAStoreMode : I32EnumAttr<"TMAStoreMode", "NVVM TMA Store Mode",
    [TMAStoreModeTile, TMAStoreModeIm2Col,
     TMAStoreModeTileScatter4, TMAStoreModeIm2ColW]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def TMAStoreModeAttr : EnumAttr<NVVM_Dialect, TMAStoreMode, "tma_store_mode">;

// List of Reduction Ops supported with TMA Store
def TMAReduxKindAdd : I32EnumAttrCase<"ADD", 0, "add">;
def TMAReduxKindMin : I32EnumAttrCase<"MIN", 1, "min">;
def TMAReduxKindMax : I32EnumAttrCase<"MAX", 2, "max">;
def TMAReduxKindInc : I32EnumAttrCase<"INC", 3, "inc">;
def TMAReduxKindDec : I32EnumAttrCase<"DEC", 4, "dec">;
def TMAReduxKindAnd : I32EnumAttrCase<"AND", 5, "and">;
def TMAReduxKindOr  : I32EnumAttrCase<"OR",  6, "or">;
def TMAReduxKindXor : I32EnumAttrCase<"XOR", 7, "xor">;

def TMAReduxKind : I32EnumAttr<"TMAReduxKind", "NVVM TMA redux kind",
    [TMAReduxKindAdd, TMAReduxKindMax, TMAReduxKindMin,
     TMAReduxKindInc, TMAReduxKindDec, TMAReduxKindAnd,
     TMAReduxKindOr,  TMAReduxKindXor]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def TMAReduxKindAttr : EnumAttr<NVVM_Dialect, TMAReduxKind, "tma_redux_kind">;

// Num CTAs in a group participating in the TMA/MMA operations.
// This corresponds to the "cta_group::1", "cta_group::2"
// modifiers in the PTX instructions.
def CTAGroup_1 : I32EnumAttrCase<"CTA_1", 0, "cta_1">;
def CTAGroup_2 : I32EnumAttrCase<"CTA_2", 1, "cta_2">;

def CTAGroupKind : I32EnumAttr<"CTAGroupKind",
                            "NVVM CTA group kind",
  [CTAGroup_1, CTAGroup_2]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def CTAGroupKindAttr :
  EnumAttr<NVVM_Dialect, CTAGroupKind, "cta_group">;

//===----------------------------------------------------------------------===//
// tcgen05 enums
//===----------------------------------------------------------------------===//

def Tcgen05FenceBefore : I32EnumAttrCase<"BEFORE_THREAD_SYNC", 0, "before">;
def Tcgen05FenceAfter  : I32EnumAttrCase<"AFTER_THREAD_SYNC",  1, "after">;
def Tcgen05FenceKind : I32EnumAttr<"Tcgen05FenceKind", "NVVM Tcgen05 fence kind",
  [Tcgen05FenceBefore, Tcgen05FenceAfter]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def Tcgen05FenceKindAttr :
  EnumAttr<NVVM_Dialect, Tcgen05FenceKind, "tcgen05_fence">;

def Tcgen05WaitLoad  : I32EnumAttrCase<"LOAD",  0, "load">;
def Tcgen05WaitStore : I32EnumAttrCase<"STORE", 1, "store">;
def Tcgen05WaitKind : I32EnumAttr<"Tcgen05WaitKind", "NVVM Tcgen05 wait kind",
  [Tcgen05WaitLoad, Tcgen05WaitStore]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def Tcgen05WaitKindAttr :
  EnumAttr<NVVM_Dialect, Tcgen05WaitKind, "tcgen05_wait">;

def Shape128x256b : I32EnumAttrCase<"SHAPE_128x256b", 0, "shape_128x256b">;
def Shape4x256b   : I32EnumAttrCase<"SHAPE_4x256b",   1, "shape_4x256b">;
def Shape128x128b : I32EnumAttrCase<"SHAPE_128x128b", 2, "shape_128x128b">;
def Shape64x128b  : I32EnumAttrCase<"SHAPE_64x128b",  3, "shape_64x128b">;
def Shape32x128b  : I32EnumAttrCase<"SHAPE_32x128b",  4, "shape_32x128b">;

def Tcgen05CpShape : I32EnumAttr<"Tcgen05CpShape", "tcgen05 cp shapes",
  [Shape128x256b, Shape4x256b, Shape128x128b, Shape64x128b, Shape32x128b]> {
    let cppNamespace = "::mlir::NVVM";
    let genSpecializedAttr = 0;
}
def Tcgen05CpShapeAttr : EnumAttr<NVVM_Dialect, Tcgen05CpShape, "tcgen05_cp_shape">;

def Tcgen05CpMulticastNone: I32EnumAttrCase<"NONE", 0, "none">;
def Tcgen05CpMulticastWarpx2_02_13: I32EnumAttrCase<"WARPX2_02_13", 1, "warpx2_02_13">;
def Tcgen05CpMulticastWarpx2_01_23: I32EnumAttrCase<"WARPX2_01_23", 2, "warpx2_01_23">;
def Tcgen05CpMulticastWarpx4: I32EnumAttrCase<"WARPX4", 3, "warpx4">;

def Tcgen05CpMulticast : I32EnumAttr<"Tcgen05CpMulticast", "tcgen05 cp multicast",
  [Tcgen05CpMulticastNone, Tcgen05CpMulticastWarpx2_02_13,
   Tcgen05CpMulticastWarpx2_01_23, Tcgen05CpMulticastWarpx4]> {
    let cppNamespace = "::mlir::NVVM";
    let genSpecializedAttr = 0;
}
def Tcgen05CpMulticastAttr : EnumAttr<NVVM_Dialect, Tcgen05CpMulticast, "tcgen05_cp_multicast">;

def FormatB6x16_P32: I32EnumAttrCase<"B6x16_P32", 0, "b6x16_p32">;
def FormatB4x16_P64: I32EnumAttrCase<"B4x16_P64", 1, "b4x16_p64">;

def Tcgen05CpSrcFormat : I32EnumAttr<"Tcgen05CpSrcFormat", "tcgen05 cp source format",
  [FormatB6x16_P32, FormatB4x16_P64]> {
    let cppNamespace = "::mlir::NVVM";
    let genSpecializedAttr = 0;
}
def Tcgen05CpSrcFormatAttr : EnumAttr<NVVM_Dialect, Tcgen05CpSrcFormat, "tcgen05_cp_src_fmt">;

def Tcgen05LdStShape16x64b: I32EnumAttrCase<"SHAPE_16X64B", 0, "shape_16x64b">;
def Tcgen05LdStShape16x128b: I32EnumAttrCase<"SHAPE_16X128B", 1, "shape_16x128b">;
def Tcgen05LdStShape16x256b: I32EnumAttrCase<"SHAPE_16X256B", 2, "shape_16x256b">;
def Tcgen05LdStShape32x32b: I32EnumAttrCase<"SHAPE_32X32B", 3, "shape_32x32b">;
def Tcgen05LdStShape16x32bx2: I32EnumAttrCase<"SHAPE_16X32BX2", 4, "shape_16x32bx2">;

def Tcgen05LdStShape: I32EnumAttr<
  "Tcgen05LdStShape",
  "",
  [Tcgen05LdStShape16x64b, Tcgen05LdStShape16x128b, Tcgen05LdStShape16x256b,
   Tcgen05LdStShape32x32b, Tcgen05LdStShape16x32bx2]
> {
  let cppNamespace = "::mlir::NVVM";
  let genSpecializedAttr = 0;
}
def Tcgen05LdStShapeAttr: EnumAttr<NVVM_Dialect, Tcgen05LdStShape, "tcgen05_ldst_shape">;

def Tcgen05MMAKindF16          : I32EnumAttrCase<"F16",      0, "f16">;
def Tcgen05MMAKindTF32         : I32EnumAttrCase<"TF32",     1, "tf32">;
def Tcgen05MMAKindF8F6F4       : I32EnumAttrCase<"F8F6F4",   2, "f8f6f4">;
def Tcgen05MMAKindI8           : I32EnumAttrCase<"I8",       3, "i8">;
def Tcgen05MMAKindMXF8F6F4     : I32EnumAttrCase<"MXF8F6F4", 4, "mxf8f6f4">;
def Tcgen05MMAKindMXF4         : I32EnumAttrCase<"MXF4",     5, "mxf4">;
def Tcgen05MMAKindMXF4NVF4     : I32EnumAttrCase<"MXF4NVF4", 6, "mxf4nvf4">;
def Tcgen05MMAKindTI16         : I32EnumAttrCase<"TI16",     7, "ti16">;

def Tcgen05MMAKind : I32EnumAttr<
  "Tcgen05MMAKind",
  "tcgen05 MMA Supported Types",
  [Tcgen05MMAKindF16, Tcgen05MMAKindTF32, Tcgen05MMAKindF8F6F4,
   Tcgen05MMAKindI8, Tcgen05MMAKindMXF8F6F4, Tcgen05MMAKindMXF4,
   Tcgen05MMAKindMXF4NVF4, Tcgen05MMAKindTI16
  ]> {
    let cppNamespace = "::mlir::NVVM";
    let genSpecializedAttr = 0;
}
def Tcgen05MMAKindAttr : EnumAttr<NVVM_Dialect, Tcgen05MMAKind, "tcgen05_mma_kind"> {
  let description = [{
    The Tcgen05MMAKind attribute describes the allowed set of types for matrix A and B in the tcgen05.mma Ops.
    The following are supported types for each kind:

    ```
    +-------------+--------------------------------------------+
    | Matrix Kind |      supported types for A / B             |
    +-------------+--------------------------------------------+
    | f16         | f16, bf16                                  |
    | tf32        | tf32                                       |
    | f8f6f4      | e4m3, e5m2, e2m3, e3m2, e2m1               |
    | i8          | unsigned 8b, signed 8b                     |
    | mxf8f6f4    | e4m3, e5m2, e2m3, e3m2, e2m1               |
    | mxf4        | e2m1                                       |
    | mxf4nvf4    | e2m1                                       |
    +-------------+--------------------------------------------+
    ```
  }];
}

def Tcgen05MMACollectorOpDiscard  : I32EnumAttrCase<"DISCARD", 0, "discard">;
def Tcgen05MMACollectorOpLastUse  : I32EnumAttrCase<"LASTUSE", 1, "lastuse">;
def Tcgen05MMACollectorOpFill     : I32EnumAttrCase<"FILL",    2, "fill">;
def Tcgen05MMACollectorOpUse      : I32EnumAttrCase<"USE",     3, "use">;

def Tcgen05MMACollectorOp : I32EnumAttr<
  "Tcgen05MMACollectorOp",
  "tcgen05.mma Collector Buffer Operation",
  [Tcgen05MMACollectorOpDiscard,
   Tcgen05MMACollectorOpLastUse,
   Tcgen05MMACollectorOpFill,
   Tcgen05MMACollectorOpUse]> {
    let cppNamespace = "::mlir::NVVM";
    let genSpecializedAttr = 0;
}
def Tcgen05MMACollectorOpAttr : EnumAttr<NVVM_Dialect, Tcgen05MMACollectorOp, "tcgen05_mma_collectorop"> {
  let description = [{
    Tcgen05MMACollectorOp attribute specifies the collector buffer operations.
    The following are the supported operations:
      * discard : Release buffer after use (default)
      * lastuse : Mark buffer for last use
      * fill    : Fill buffer
      * use     : Use buffer without modification
  }];
}

def Tcgen05MMABlockScaleDefault : I32EnumAttrCase<"DEFAULT", 0, "default">;
def Tcgen05MMABlockScaleBlock16 : I32EnumAttrCase<"BLOCK16", 1, "block16">;
def Tcgen05MMABlockScaleBlock32 : I32EnumAttrCase<"BLOCK32", 2, "block32">;

def Tcgen05MMABlockScale
    : I32EnumAttr<"Tcgen05MMABlockScale",
                  "tcgen05.mma block scale attribute",
                  [Tcgen05MMABlockScaleDefault, Tcgen05MMABlockScaleBlock16,
                   Tcgen05MMABlockScaleBlock32]> {
  let cppNamespace = "::mlir::NVVM";
  let genSpecializedAttr = 0;
}
def Tcgen05MMABlockScaleAttr : EnumAttr<NVVM_Dialect, Tcgen05MMABlockScale,
                                          "tcgen05_mma_block_scale">;

def Tcgen05MMACollectorBBuffer0  : I32EnumAttrCase<"B0", 0, "b0">;
def Tcgen05MMACollectorBBuffer1  : I32EnumAttrCase<"B1", 1, "b1">;
def Tcgen05MMACollectorBBuffer2  : I32EnumAttrCase<"B2", 2, "b2">;
def Tcgen05MMACollectorBBuffer3  : I32EnumAttrCase<"B3", 3, "b3">;

def Tcgen05MMACollectorBBuffer : I32EnumAttr<
  "Tcgen05MMACollectorBBuffer",
  "tcgen05 MMA Collector Buffer B Attribute",
  [Tcgen05MMACollectorBBuffer0, Tcgen05MMACollectorBBuffer1, Tcgen05MMACollectorBBuffer2,
   Tcgen05MMACollectorBBuffer3]> {
    let cppNamespace = "::mlir::NVVM";
    let genSpecializedAttr = 0;
}
def Tcgen05MMACollectorBBufferAttr : EnumAttr<NVVM_Dialect, Tcgen05MMACollectorBBuffer, "tcgen05_mma_collectorb">;

//===----------------------------------------------------------------------===//
// Tensormap enums
//===----------------------------------------------------------------------===//

def TensormapGlobalAddressField : I32EnumCase<"GLOBAL_ADDRESS", 0, "global_address">;
def TensormapRankField : I32EnumCase<"RANK", 1, "rank">;
def TensormapBoxDimField : I32EnumCase<"BOX_DIM", 2, "box_dim">;
def TensormapGlobalDimField : I32EnumCase<"GLOBAL_DIM", 3, "global_dim">;
def TensormapGlobalStrideField : I32EnumCase<"GLOBAL_STRIDE", 4, "global_stride">;
def TensormapElementStrideField : I32EnumCase<"ELEMENT_STRIDE", 5, "element_stride">;
def TensormapElemtypeField : I32EnumCase<"ELEMTYPE", 6, "elemtype">;
def TensormapInterleaveLayoutField : I32EnumCase<"INTERLEAVE_LAYOUT", 7, "interleave_layout">;
def TensormapSwizzleModeField : I32EnumCase<"SWIZZLE_MODE", 8, "swizzle_mode">;
def TensormapSwizzleAtomicityField : I32EnumCase<"SWIZZLE_ATOMICITY", 9, "swizzle_atomicity">;
def TensormapFillModeField : I32EnumCase<"FILL_MODE", 10, "fill_mode">;

def TensormapField : I32Enum<"TensormapField", "NVVM Tensormap Field Kind",
  [TensormapGlobalAddressField, TensormapRankField, TensormapBoxDimField, 
   TensormapGlobalDimField, TensormapGlobalStrideField, 
   TensormapElementStrideField, TensormapElemtypeField, 
   TensormapInterleaveLayoutField, TensormapSwizzleModeField, 
   TensormapSwizzleAtomicityField, TensormapFillModeField]> {
    let cppNamespace = "::mlir::NVVM";
}
def TensormapFieldAttr : EnumAttr<NVVM_Dialect, TensormapField, "tensormap_field">;

def TensormapElemtypeU8 : I32EnumCase<"U8", 0, "u8">;
def TensormapElemtypeU16 : I32EnumCase<"U16", 1, "u16">;
def TensormapElemtypeU32 : I32EnumCase<"U32", 2, "u32">;
def TensormapElemtypeS32 : I32EnumCase<"S32", 3, "s32">;
def TensormapElemtypeU64 : I32EnumCase<"U64", 4, "u64">;
def TensormapElemtypeS64 : I32EnumCase<"S64", 5, "s64">;
def TensormapElemtypeF16 : I32EnumCase<"F16", 6, "f16">;
def TensormapElemtypeF32 : I32EnumCase<"F32", 7, "f32">;
def TensormapElemtypeF32_Ftz : I32EnumCase<"F32_FTZ", 8, "f32.ftz">;
def TensormapElemtypeF64 : I32EnumCase<"F64", 9, "f64">;
def TensormapElemtypeBF16 : I32EnumCase<"BF16", 10, "bf16">;
def TensormapElemtypeTF32 : I32EnumCase<"TF32", 11, "tf32">;
def TensormapElemtypeTF32_Ftz : I32EnumCase<"TF32_FTZ", 12, "tf32.ftz">;
def TensormapElemtypeB4x16 : I32EnumCase<"B4x16", 13, "b4x16">;
def TensormapElemtypeB4x16P64 : I32EnumCase<"B4x16P64", 14, "b4x16_p64">;
def TensormapElemtypeB6x16P32 : I32EnumCase<"B6x16P32", 15, "b6x16_p32">;

def TensormapElemtype : I32Enum<"TensormapElemtype", "NVVM Tensormap Elemtype",
  [TensormapElemtypeU8, TensormapElemtypeU16, 
   TensormapElemtypeU32,TensormapElemtypeS32, 
   TensormapElemtypeU64, TensormapElemtypeS64,
   TensormapElemtypeF16, TensormapElemtypeF32, 
   TensormapElemtypeF32_Ftz,TensormapElemtypeF64, 
   TensormapElemtypeBF16, TensormapElemtypeTF32,
   TensormapElemtypeTF32_Ftz, TensormapElemtypeB4x16, 
   TensormapElemtypeB4x16P64, TensormapElemtypeB6x16P32]> {
    let cppNamespace = "::mlir::NVVM";
}
def TensormapElemtypeAttr : EnumAttr<NVVM_Dialect, TensormapElemtype, "tensormap_elemtype">;

def TensormapInterleaveLayoutNoInterleave : 
  I32EnumCase<"NO_INTERLEAVE", 0, "no_interleave">;
def TensormapInterleaveLayoutB16 : I32EnumCase<"B16", 1, "b16">;
def TensormapInterleaveLayoutB32 : I32EnumCase<"B32", 2, "b32">;

def TensormapInterleaveLayout : 
  I32Enum<"TensormapInterleaveLayout", "NVVM Tensormap Interleave Layout",
  [TensormapInterleaveLayoutNoInterleave, 
   TensormapInterleaveLayoutB16,
   TensormapInterleaveLayoutB32]> {
    let cppNamespace = "::mlir::NVVM";
}
def TensormapInterleaveLayoutAttr : EnumAttr<NVVM_Dialect,   
  TensormapInterleaveLayout, "tensormap_interleave_layout">;

def TensormapSwizzleModeNoSwizzling : I32EnumCase<"NO_SWIZZLING", 0, "no_swizzling">;
def TensormapSwizzleMode32B : I32EnumCase<"B32", 1, "b32">;
def TensormapSwizzleMode64B : I32EnumCase<"B64", 2, "b64">;
def TensormapSwizzleMode128B : I32EnumCase<"B128", 3, "b128">;
def TensormapSwizzleMode96B : I32EnumCase<"B96", 4, "b96">;

def TensormapSwizzleMode : I32Enum<"TensormapSwizzleMode", "NVVM Tensormap Swizzle Mode",
  [TensormapSwizzleModeNoSwizzling, 
   TensormapSwizzleMode32B,
   TensormapSwizzleMode64B, 
   TensormapSwizzleMode128B,
   TensormapSwizzleMode96B]> {
    let cppNamespace = "::mlir::NVVM";
}
def TensormapSwizzleModeAttr : EnumAttr<NVVM_Dialect,
  TensormapSwizzleMode, "tensormap_swizzle_mode">;

def TensormapSwizzleAtomicityB16 : I32EnumCase<"B16", 0, "b16">;
def TensormapSwizzleAtomicityB32 : I32EnumCase<"B32", 1, "b32">;
def TensormapSwizzleAtomicityB32FlipB8 : I32EnumCase<"B32_FLIP_8B", 2, "b32_flip_b8">;
def TensormapSwizzleAtomicityB64 : I32EnumCase<"B64", 3, "b64">;

def TensormapSwizzleAtomicity : 
  I32Enum<"TensormapSwizzleAtomicity", "NVVM Tensormap Swizzle Atomicity",
  [TensormapSwizzleAtomicityB16, TensormapSwizzleAtomicityB32,
   TensormapSwizzleAtomicityB32FlipB8, 
   TensormapSwizzleAtomicityB64]> {
    let cppNamespace = "::mlir::NVVM";
}
def TensormapSwizzleAtomicityAttr : EnumAttr<NVVM_Dialect,
  TensormapSwizzleAtomicity, "tensormap_swizzle_atomicity">;

def TensormapFillModeZeroFill : I32EnumCase<"ZERO", 0, "zero">;
def TensormapFillModeOOBNaNFill : I32EnumCase<"OOB_NAN", 1, "oob_nan">;

def TensormapFillMode : I32Enum<"TensormapFillMode", "NVVM Tensormap Fill Mode",
  [TensormapFillModeZeroFill, TensormapFillModeOOBNaNFill]> {
    let cppNamespace = "::mlir::NVVM";
}
def TensormapFillModeAttr : EnumAttr<NVVM_Dialect,
  TensormapFillMode, "tensormap_fill_mode">;

#endif // NVVMIR_ENUMS
