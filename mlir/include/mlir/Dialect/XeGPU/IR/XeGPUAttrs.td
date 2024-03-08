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

def XeGPU_TensorDescAttr: XeGPUAttr<"TensorDesc", "tdesc_attr"> {
  let parameters = (ins
    OptionalParameter<"MemoryScopeKindAttr">: $memory_scope,
    OptionalParameter<"IntegerAttr", "1">: $array_length,
    OptionalParameter<"BoolAttr", "true">: $boundary_check
  );

  let builders = [
    AttrBuilder<(ins
      CArg<"xegpu::MemoryScopeKind", "xegpu::MemoryScopeKind::Global">:$memory_scope,
      CArg<"int", "1">:$array_length,
      CArg<"bool", "true">: $boundary_check
    )>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// XeGPU Memory Scope Enums.
//===----------------------------------------------------------------------===//
def XeGPU_MemoryScopeGlobal: I32EnumAttrCase<"Global", 0, "global">;
def XeGPU_MemoryScopeShared: I32EnumAttrCase<"SLM", 1, "slm">;
def XeGPU_MemoryScopeKind: I32EnumAttr<"MemoryScopeKind", 
      "The address space of the memory the tensor descritor is created for", 
      [XeGPU_MemoryScopeGlobal, XeGPU_MemoryScopeShared]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}

def XeGPU_MemoryScopeAttr: 
  EnumAttr<XeGPU_Dialect, XeGPU_MemoryScopeKind, "memory_scope"> {
    let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// XeGPU Operator Mode Enums.
//===----------------------------------------------------------------------===//
def XeGPU_OpModeSIMT : I32EnumAttrCase<"SIMT", 0, "simt">;
def XeGPU_OpModeVectorCompute : I32EnumAttrCase<"VectorCompute", 1, "vc">;
def XeGPU_ModeKind : I32EnumAttr<"ModeKind", 
             "The Mode an operator runs on", 
  [XeGPU_OpModeSIMT, XeGPU_OpModeVectorCompute]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}

def XeGPU_ModeAttr: 
  EnumAttr<XeGPU_Dialect, XeGPU_ModeKind, "mode"> {
    let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// XeGPU Cache Enums.
//===----------------------------------------------------------------------===//
def XeGPU_CacheKindCached:        I32EnumAttrCase<"CACHED", 0, "cached">;                    // valid for read and write
def XeGPU_CacheKindUncached:      I32EnumAttrCase<"UNCACHED", 1, "uncached">;                // valid for read and write
def XeGPU_CacheKindStreaming:     I32EnumAttrCase<"STREAMING", 2, "streaming">;              // valid for read only
def XeGPU_CacheKindInvalid:       I32EnumAttrCase<"READ_INVALIDATE", 3, "read_invalidate">;  // valid for read only
def XeGPU_CacheKindWriteBack:     I32EnumAttrCase<"WRITE_BACK", 4, "write_back">;            // valid for write only
def XeGPU_CacheKindWriteThrough:  I32EnumAttrCase<"WRITE_THROUGH", 5, "write_through">;      // valid for write only

def XeGPU_CacheKind : I32EnumAttr<"CacheKind", "Cache kind", 
  [XeGPU_CacheKindCached, XeGPU_CacheKindUncached, 
   XeGPU_CacheKindStreaming, XeGPU_CacheKindInvalid,
   XeGPU_CacheKindWriteBack, XeGPU_CacheKindWriteThrough]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}

def XeGPU_CacheAttr 
  : EnumAttr<XeGPU_Dialect, XeGPU_CacheKind, "cache_kind"> {
    let assemblyFormat = "`<` $value `>`";
}



#endif // MLIR_DIALECT_XEGPU_IR_XEGPUATTRS_TD
