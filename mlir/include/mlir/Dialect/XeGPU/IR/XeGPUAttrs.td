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
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class XeGPUAttr<string name, string attrMnemonic, list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<XeGPU_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

class XeGPU_TensorDescAttr<string name, string attrMnemonic, list<Trait> traits = [],
                         string baseCppClass = "::mlir::Attribute">
    : XeGPUAttr<name, attrMnemonic, traits, baseCppClass> {
  let assemblyFormat = "`<` struct(params) `>`";
}

def XeGPU_BlockTensorDescAttr: XeGPU_TensorDescAttr<"BlockTensorDesc", "block_tdesc_attr"> {
  let summary = [{a composite attribute for `TensorDescType`}];
  let description = [{`BlockTensorDesc` (or `block_tdesc_attr`) is a composite
    attribute defined for `TensorDescType` for describing following
    properties of a `TensorDesc`.
    1. `memory_space`: It describes where the data block described by the
        TensorDesc is located, `Global` device memory or `Shared` local memory.
        It is default to `Global`.
    2. `array_length`: It describes how many horizontally consecutive blocks
        will be loaded by a hardware load instruction. If the TensorDesc shape
        is 8x16, with array_length = 2. The loaded block shape will be acctually
        8x32. Its default value is 1.
    3. `boundary_check`: It is used to indicates the hardware whether to do
        out-of-boundary check. The default value is true.
  }];

  let parameters = (ins
    OptionalParameter<"MemorySpaceAttr">: $memory_space,
    OptionalParameter<"IntegerAttr", "1">: $array_length,
    OptionalParameter<"BoolAttr", "true">: $boundary_check
  );

  let builders = [
    AttrBuilder<(ins
      CArg<"xegpu::MemorySpace", "xegpu::MemorySpace::Global">:$memory_space,
      CArg<"int", "1">:$array_length,
      CArg<"bool", "true">: $boundary_check
    )>
  ];

}

def XeGPU_ScatterTensorDescAttr: XeGPU_TensorDescAttr<"ScatterTensorDesc", "scatter_tdesc_attr"> {
  let summary = [{a composite attribute for `TensorDescType`}];
  let description = [{`ScatterTensorDesc` (or `scatter_tdesc_attr`) is a composite
    attribute defined for `TensorDescType` for describing following
    properties of a `TensorDesc`.
    1. `memory_space`: It describes where the data block described by the
        TensorDesc is located, `Global` device memory or `Shared` local memory.
        It is default to `Global`.
    2.  `chunk_size`: indicates number of continious elements accessed for each
        offset, default is 1. It is used with `scattered` attr only.
  }];

  let parameters = (ins
    OptionalParameter<"MemorySpaceAttr">: $memory_space,
    OptionalParameter<"IntegerAttr", "1">: $chunk_size
  );

  let builders = [
    AttrBuilder<(ins
      CArg<"xegpu::MemorySpace", "xegpu::MemorySpace::Global">:$memory_space,
      CArg<"int", "1">: $chunk_size
    )>
  ];
 }

//===----------------------------------------------------------------------===//
// XeGPU Memory Scope Enums.
//===----------------------------------------------------------------------===//
def XeGPU_MemorySpaceGlobal: I32EnumAttrCase<"Global", 0, "global">;
def XeGPU_MemorySpaceShared: I32EnumAttrCase<"SLM", 3, "slm">;
def XeGPU_MemorySpace: I32EnumAttr<"MemorySpace",
      "The address space of the memory the tensor descritor is created for",
      [XeGPU_MemorySpaceGlobal, XeGPU_MemorySpaceShared]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}

def XeGPU_MemorySpaceAttr:
  EnumAttr<XeGPU_Dialect, XeGPU_MemorySpace, "memory_space"> {
    let summary = [{Describe the location of data described by a `TensorDesc`:
                 Global device memory (`Global`) or Shared local memory (`SLM`).}];
    let assemblyFormat = "$value";
}

//===----------------------------------------------------------------------===//
// XeGPU Cache Enums.
//===----------------------------------------------------------------------===//
def XeGPU_CachePolicyCached:        I32EnumAttrCase<"CACHED", 0, "cached">;                    // valid for read and write
def XeGPU_CachePolicyUncached:      I32EnumAttrCase<"UNCACHED", 1, "uncached">;                // valid for read and write
def XeGPU_CachePolicyStreaming:     I32EnumAttrCase<"STREAMING", 2, "streaming">;              // valid for read only
def XeGPU_CachePolicyInvalid:       I32EnumAttrCase<"READ_INVALIDATE", 3, "read_invalidate">;  // valid for read only
def XeGPU_CachePolicyWriteBack:     I32EnumAttrCase<"WRITE_BACK", 4, "write_back">;            // valid for write only
def XeGPU_CachePolicyWriteThrough:  I32EnumAttrCase<"WRITE_THROUGH", 5, "write_through">;      // valid for write only

def XeGPU_CachePolicyEnums : I32EnumAttr<"CachePolicy", "Cache policy",
  [XeGPU_CachePolicyCached, XeGPU_CachePolicyUncached,
   XeGPU_CachePolicyStreaming, XeGPU_CachePolicyInvalid,
   XeGPU_CachePolicyWriteBack, XeGPU_CachePolicyWriteThrough]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}

def XeGPU_CacheHintAttr
  : EnumAttr<XeGPU_Dialect, XeGPU_CachePolicyEnums, "cache_hint"> {
    let summary = [{Describe the cache settings for prefetch/load/store operators}];
    let assemblyFormat = "`<` $value `>`";
}

def XeGPU_FenceScopeWorkgroup: I32EnumAttrCase<"Workgroup", 0, "workgroup">;
def XeGPU_FenceScopeGPU: I32EnumAttrCase<"GPU", 1, "gpu">;
def XeGPU_FenceScope: I32EnumAttr<"FenceScope",
      "The enumeration for the scope of fence operation.",
      [XeGPU_FenceScopeWorkgroup, XeGPU_FenceScopeGPU]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}

def XeGPU_FenceScopeAttr:
  EnumAttr<XeGPU_Dialect, XeGPU_FenceScope, "fence_scope"> {
    let summary = [{Describes the scope of fence.
                    "workgroup" means that the scope is within each work group.
                    "gpu" means the scope is across work groups within the gpu.}];
    let assemblyFormat = "$value";
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUATTRS_TD
