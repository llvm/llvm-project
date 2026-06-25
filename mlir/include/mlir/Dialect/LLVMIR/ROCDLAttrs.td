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
include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"
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

def ROCDL_BufferOOBModeAttr :
    ROCDL_IntrinsicIntegerEnumAttr<ROCDL_BufferOOBMode, "buffer_oob_mode"> {
  let summary = "ROCDL buffer out-of-bounds mode";
  let description = [{
    Specifies the AMDGPU buffer out-of-bounds mode encoded in
    `amdgpu.buffer.oob.mode` or `amdgpu.tbuffer.oob.mode` LLVM module flags.
  }];
  let assemblyFormat = "`<` $value `>`";
}

class ROCDL_BufferOOBModeModuleFlagAttrBase<string attrName, string mnemonic>
    : ROCDL_Attr<attrName, mnemonic,
                 [DeclareAttrInterfaceMethods<LLVM_ModuleFlagAttrInterface>]> {
  let summary = "ROCDL buffer/tbuffer out-of-bounds mode LLVM module flag";
  let description = [{
    Specifies an AMDGPU buffer or tbuffer out-of-bounds mode LLVM module flag.
    The attribute supplies the required `max` merge behavior and AMDGPU module
    flag key. Prefer these typed attributes over hand-written generic
    `#llvm.mlir.module_flag` triples for AMDGPU OOB mode keys when MLIR-side
    behavior and enum validation are required.
  }];
  let parameters = (ins EnumParameter<ROCDL_BufferOOBMode>:$value);
  let assemblyFormat = "`<` $value `>`";
}

def ROCDL_BufferOOBModeModuleFlagAttr :
    ROCDL_BufferOOBModeModuleFlagAttrBase<"BufferOOBModeModuleFlag",
                                          "buffer_oob_mode_flag">;

def ROCDL_TBufferOOBModeModuleFlagAttr :
    ROCDL_BufferOOBModeModuleFlagAttrBase<"TBufferOOBModeModuleFlag",
                                          "tbuffer_oob_mode_flag">;

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

class ROCDL_CachePolicyAttrOf<list<Attr> policyAttrs, string summary>
    : AnyAttrOf<!listconcat(policyAttrs, [I32Attr]), summary> {
  let storageType = "::mlir::Attribute";
  let returnType = "::mlir::Attribute";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

class ROCDL_DefaultCachePolicyAttr<Attr policyAttr>
    : DefaultValuedAttr<policyAttr, "$_builder.getI32IntegerAttr(0)">;

def ROCDL_PreGfx12CachePolicyCompatAttr : ROCDL_CachePolicyAttrOf<[
      ROCDL_PreGfx12CachePolicyAttr, ROCDL_Gfx942CachePolicyAttr],
    "pre-gfx12 or gfx942 AMDGPU cache policy attribute">;

def ROCDL_Gfx12NonAtomicCachePolicyCompatAttr : ROCDL_CachePolicyAttrOf<[
      ROCDL_Gfx12CachePolicyAttr],
    "gfx12 non-atomic AMDGPU cache policy attribute">;

def ROCDL_NonAtomicBufferCachePolicyCompatAttr : ROCDL_CachePolicyAttrOf<[
      ROCDL_PreGfx12CachePolicyAttr, ROCDL_Gfx942CachePolicyAttr,
      ROCDL_Gfx12CachePolicyAttr],
    "non-atomic AMDGPU buffer cache policy attribute">;

def ROCDL_AtomicBufferCachePolicyCompatAttr : ROCDL_CachePolicyAttrOf<[
      ROCDL_PreGfx12CachePolicyAttr, ROCDL_Gfx942CachePolicyAttr,
      ROCDL_Gfx12AtomicCachePolicyAttr],
    "atomic AMDGPU buffer cache policy attribute">;

#endif // ROCDLIR_ATTRS
