//===-- ROCDLAttrs.td - ROCDL IR dialect attributes -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCDLIR_ATTRS
#define ROCDLIR_ATTRS

include "mlir/Dialect/LLVMIR/ROCDLDialect.td"
include "mlir/Dialect/LLVMIR/ROCDLEnums.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// ROCDL attribute definitions
//===----------------------------------------------------------------------===//

class ROCDL_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ROCDL_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class ROCDL_IntrinsicIntegerEnumAttr<EnumInfo enumInfo, string name>
    : EnumAttr<ROCDL_Dialect, enumInfo, name,
               [LLVM_IntrinsicIntegerAttrInterface]>;

def ROCDL_MFMAPermBAttr
    : ROCDL_IntrinsicIntegerEnumAttr<ROCDL_MFMAPermB, "mfma_perm_b">;

def ROCDL_MFMANegModifierAttr
    : ROCDL_IntrinsicIntegerEnumAttr<ROCDL_MFMANegModifier,
                                     "mfma_neg_modifier">;

def ROCDL_MatrixFormatAttr
    : ROCDL_IntrinsicIntegerEnumAttr<ROCDL_MatrixFormat, "matrix_format">;

def ROCDL_WMMAMatrixScaleAttr
    : ROCDL_IntrinsicIntegerEnumAttr<ROCDL_WMMAMatrixScale,
                                     "wmma_matrix_scale">;

def ROCDL_WMMAMatrixScaleFormatAttr
    : ROCDL_IntrinsicIntegerEnumAttr<ROCDL_WMMAMatrixScaleFormat,
                                     "wmma_matrix_scale_format">;

def ROCDL_WMMACModifierAttr
    : ROCDL_IntrinsicIntegerEnumAttr<ROCDL_WMMACModifier,
                                     "wmma_c_modifier">;

def ROCDL_SchedGroupMaskAttr
    : ROCDL_IntrinsicIntegerEnumAttr<ROCDL_SchedGroupMask,
                                     "sched_group_mask">;

def ROCDL_PreGfx12CachePolicyAttr
    : ROCDL_IntrinsicIntegerEnumAttr<ROCDL_PreGfx12CachePolicy,
                                     "pre_gfx12_cache_policy">;

def ROCDL_Gfx942CachePolicyAttr
    : ROCDL_IntrinsicIntegerEnumAttr<ROCDL_Gfx942CachePolicy,
                                     "gfx942_cache_policy">;

def ROCDL_Gfx12CachePolicyAttr
    : ROCDL_IntrinsicIntegerEnumAttr<ROCDL_Gfx12CachePolicy,
                                     "gfx12_cache_policy">;

def ROCDL_Gfx12AtomicCachePolicyAttr
    : ROCDL_IntrinsicIntegerEnumAttr<ROCDL_Gfx12AtomicCachePolicy,
                                     "gfx12_atomic_cache_policy">;

def ROCDL_CachePolicyAttr : AnyAttrOf<[
      ROCDL_PreGfx12CachePolicyAttr, ROCDL_Gfx942CachePolicyAttr,
      ROCDL_Gfx12CachePolicyAttr, ROCDL_Gfx12AtomicCachePolicyAttr, I32Attr],
    "AMDGPU cache policy attribute"> {
  let storageType = "::mlir::Attribute";
  let returnType = "::mlir::Attribute";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

def ROCDL_DefaultCachePolicyAttr : DefaultValuedAttr<ROCDL_CachePolicyAttr,
    "$_builder.getI32IntegerAttr(0)">;

#endif // ROCDLIR_ATTRS
