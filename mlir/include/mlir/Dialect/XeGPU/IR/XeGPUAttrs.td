//===- XeGPUAttrs.td - XeGPU dialect attributes definition --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_IR_XEGPUATTRS_TD
#define MLIR_DIALECT_XEGPU_IR_XEGPUATTRS_TD

include "mlir/Dialect/XeGPU/IR/XeGPUDialect.td"
include "mlir/IR/EnumAttr.td"

class XeGPUAttr<string name, string attrMnemonic, list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<XeGPU_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def XeGPU_ScatteredAttr : XeGPUAttr<"Scattered", "scattered"> {
  let summary = "Scattered attribute for scattered read and write operation.";
  let description = [{An attribute represent scattered read and write operation.
    It does not (need to) have meaningful input values. The existence of itself
    implies scattered read/write.}];

  let assemblyFormat = "";
}

def XeGPU_SgMapAttr: XeGPUAttr<"SubGroupMap", "sg_map"> {
  let parameters = (ins
        "mlir::DenseI32ArrayAttr":$wi_layout,
        "mlir::DenseI32ArrayAttr":$wi_data
      );

  // In format of #xegpu.sg_map<{mma_block_size = [2, 4], wi_layout = [2, 4], wi_data = [2, 4]}>
  let assemblyFormat = "`<` struct(params) `>`";

  let genVerifyDecl = true;

  let builders = [
    AttrBuilder<(ins
      "llvm::ArrayRef<int32_t>":$wiLayout,
      "llvm::ArrayRef<int32_t>":$wiData
    )>
  ];
}

def XeGPU_TensorDescAttr: XeGPUAttr<"TensorDesc", "tdesc_attr"> {
  let parameters = (ins
        DefaultValuedParameter<"xegpu::MemoryScopeKind", "xegpu::MemoryScopeKind::GLOBAL">: $memory_scope,
        DefaultValuedParameter<"int", "1">: $array_length,
        DefaultValuedParameter<"bool", "true">: $boundary_check,
        OptionalParameter<"xegpu::ScatteredAttr">: $scattered,
        OptionalParameter<"xegpu::SubGroupMapAttr"> : $map
      );

  let builders = [
    AttrBuilder<(ins
      CArg<"xegpu::MemoryScopeKind", "xegpu::MemoryScopeKind::GLOBAL">:$memory_scope,
      CArg<"int", "1">:$array_length,
      CArg<"xegpu::ScatteredAttr", "{}">:$scattered,
      CArg<"xegpu::SubGroupMapAttr", "{}">:$map
    )>
  ];

  let extraClassDeclaration = [{
    bool hasNonDefaultAttrs();
  }];

  let hasCustomAssemblyFormat = true;
}

def ARG_TYPE_VECTOR : I32EnumAttrCase<"VECTOR", 0, "vector">; 
def ARG_TYPE_SCALAR : I32EnumAttrCase<"SCALAR", 1, "scalar">; 
def XeGPU_ArgTypeKind : I32EnumAttr<"ArgTypeKind", 
               "Argument type for Invoke_SIMD op", 
  [ARG_TYPE_VECTOR, ARG_TYPE_SCALAR]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}

def MODE_SIMT : I32EnumAttrCase<"SIMT", 0, "simt">;
def MODE_VC : I32EnumAttrCase<"VC", 1, "vc">;
def XeGPU_ModeKind : I32EnumAttr<"ModeKind", 
             "The Mode an operator runs on", 
  [MODE_SIMT, MODE_VC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}

def MEMORY_SCOPE_GLOBAL: I32EnumAttrCase<"GLOBAL", 0, "global">;
def MEMORY_SCOPE_SHARED: I32EnumAttrCase<"SLM", 1, "slm">;
def XeGPU_MemoryScopeKind: I32EnumAttr<"MemoryScopeKind", 
      "The scope of the memory the tensor descritor is created for", 
  [MEMORY_SCOPE_GLOBAL, MEMORY_SCOPE_SHARED]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}

def CACHE_KIND_CACHED:        I32EnumAttrCase<"CACHED", 0, "cached">;                    // valid for read and write
def CACHE_KIND_UNCACHED:      I32EnumAttrCase<"UNCACHED", 1, "uncached">;                // valid for read and write
def CACHE_KIND_STREAMING:     I32EnumAttrCase<"STREAMING", 2, "streaming">;              // valid for read only
def CACHE_KIND_INVALIDATE:    I32EnumAttrCase<"READ_INVALIDATE", 3, "read_invalidate">;  // valid for read only
def CACHE_KIND_WRITE_BACK:    I32EnumAttrCase<"WRITE_BACK", 4, "write_back">;            // valid for write only
def CACHE_KIND_WRITE_THROUGH: I32EnumAttrCase<"WRITE_THROUGH", 5, "write_through">;      // valid for write only



def XeGPU_CacheKind : I32EnumAttr<"CacheKind", "Cache kind", 
  [CACHE_KIND_CACHED, CACHE_KIND_UNCACHED, 
   CACHE_KIND_STREAMING, CACHE_KIND_INVALIDATE,
   CACHE_KIND_WRITE_BACK, CACHE_KIND_WRITE_THROUGH]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}

def XeGPU_ArgTypeAttr : EnumAttr<XeGPU_Dialect, XeGPU_ArgTypeKind, "arg_type_kind">;
def XeGPU_ModeAttr : EnumAttr<XeGPU_Dialect, XeGPU_ModeKind, "mode_kind">;
def XeGPU_MemoryScopeAttr : EnumAttr<XeGPU_Dialect, XeGPU_MemoryScopeKind, "memory_scope_kind">;
def XeGPU_CacheAttr : EnumAttr<XeGPU_Dialect, XeGPU_CacheKind, "cache_kind">;

// RMW kind attribute
def ATOMIC_RMW_KIND_ADDF    : I32EnumAttrCase<"addf", 0>;
def ATOMIC_RMW_KIND_ADDI    : I32EnumAttrCase<"addi", 1>;
def ATOMIC_RMW_KIND_ASSIGN  : I32EnumAttrCase<"assign", 2>;
def ATOMIC_RMW_KIND_MAXF    : I32EnumAttrCase<"maxf", 3>;
def ATOMIC_RMW_KIND_MAXS    : I32EnumAttrCase<"maxs", 4>;
def ATOMIC_RMW_KIND_MAXU    : I32EnumAttrCase<"maxu", 5>;
def ATOMIC_RMW_KIND_MINF    : I32EnumAttrCase<"minf", 6>;
def ATOMIC_RMW_KIND_MINS    : I32EnumAttrCase<"mins", 7>;
def ATOMIC_RMW_KIND_MINU    : I32EnumAttrCase<"minu", 8>;
def ATOMIC_RMW_KIND_MULF    : I32EnumAttrCase<"mulf", 9>;
def ATOMIC_RMW_KIND_MULI    : I32EnumAttrCase<"muli", 10>;
def ATOMIC_RMW_KIND_ORI     : I32EnumAttrCase<"ori", 11>;
def ATOMIC_RMW_KIND_ANDI    : I32EnumAttrCase<"andi", 12>;

def XeGPU_AtomicRMWKind : I32EnumAttr<"AtomicRMWKind", 
                       "Operation type for AtomicRMW",
  [ATOMIC_RMW_KIND_ADDF, ATOMIC_RMW_KIND_ADDI, ATOMIC_RMW_KIND_ASSIGN,
   ATOMIC_RMW_KIND_MAXF, ATOMIC_RMW_KIND_MAXS, ATOMIC_RMW_KIND_MAXU,
   ATOMIC_RMW_KIND_MINF, ATOMIC_RMW_KIND_MINS, ATOMIC_RMW_KIND_MINU,
   ATOMIC_RMW_KIND_MULF, ATOMIC_RMW_KIND_MULI, ATOMIC_RMW_KIND_ORI,
   ATOMIC_RMW_KIND_ANDI]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}
def XeGPU_AtomicRMWKindAttr : EnumAttr<XeGPU_Dialect, XeGPU_AtomicRMWKind, "atomic_rmw_kind">;

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUATTRS_TD
