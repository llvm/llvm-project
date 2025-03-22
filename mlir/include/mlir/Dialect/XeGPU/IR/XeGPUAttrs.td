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
  let description = [{
    `ScatterTensorDesc` is a composite attribute defined for `TensorDescType`
    for describing following properties of a `TensorDesc`:

    1. `memory_space`: It describes where the data block described by the
        TensorDesc is located, `Global` device memory or `Shared` local memory.
        It is default to `Global`.

    2.  `chunk_size`: indicates number of contiguous elements accessed for each
        offset, default is 1. It is used with `scattered` attr only.
  }];

  let parameters = (ins
    DefaultValuedParameter<
      "MemorySpaceAttr",
      "MemorySpaceAttr::get($_ctxt, xegpu::MemorySpace::Global)",
      "Data memory location"
    >: $memory_space,
    DefaultValuedParameter<
      "IntegerAttr",
      "IntegerAttr::get(IntegerType::get($_ctxt, 64), 1)",
      "Number of contiguous elements"
    >: $chunk_size
  );

  let builders = [
    AttrBuilder<(ins
      CArg<"xegpu::MemorySpace", "xegpu::MemorySpace::Global">:$memory_space,
      CArg<"int", "1">: $chunk_size
    )>
  ];

  let genVerifyDecl = 1;
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

def XeGPU_ScopeWG:   I32EnumAttrCase<"WG", 0, "wg">;      // workgroup level code
def XeGPU_ScopeSG:   I32EnumAttrCase<"SG", 1, "sg">;      // subgroup level code
def XeGPU_ScopeWI:   I32EnumAttrCase<"WI", 2, "wi">;      // simt level code

def XeGPU_ScopeEnums : I32EnumAttr<"Scope", "enumerate of scope",
  [XeGPU_ScopeWG,XeGPU_ScopeSG,XeGPU_ScopeWI]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xegpu";
}

def XeGPU_ScopeAttr
  : EnumAttr<XeGPU_Dialect,XeGPU_ScopeEnums, "Stage"> {
    let summary = [{Describe the stage of lowering progress}];
    let assemblyFormat = "``$value";
}

def XeGPU_LayoutAttr : XeGPUAttr<"Layout", "layout"> {
  let summary = [{
    Describes the mapping between work item (WI) and the 2D tensor specified by the tensor descriptor.
  }];
  let description = [{
    XeGPU operations leverages LayoutAttr to distribute data across work-item. It is specified in tensor_descs
    upon the tensor description creation. LayoutAttr contains the following parameters.

    * scope: specifies the scope of current code. It can be either wg (workgroup), sg (subgroup) or wi (workitem).
             it is hard required for subgroup, but optional for workgroup and wi. By default, if a LayoutAttr
             contains sg_layout and sg_data, it will be treated as workgroup code; and if it only contains
             wi_layout and wi_data, it will be considered as workitem level.
    * sg_layout: [optional] specifies the total number of subgroups and their layout in a workgroup.
    * sg_data: [optional] specifies the data size accessed per subgroup.
    * order: [optional] specifies the dimension order used to linearize n-d sbugroup ids to 1-d.
            The first dimension in the order list is the fastest-changing dimension.
    * wi_layout: [required] specifies the total number of work-items and their layout in a subgroup
    * wi_data: [required] specifies the data size accessed per work-item for a single distribution.

    `wi_data[0] * wi_data[1]` can be greater than 1, meaning that each work item operates on multiple elements,
    which is eventually lowered to "SIMT-flavor" vector, like SPIR-V vector or llvm vector, or packed to a storage data type.
    The multiple elements indicated by `wi_data` can only be from one dimension and must be contiguous in the memory along either dimension.

    E.g., #xegpu.layout<wi_layout = [1, 16], wi_data = [1, 1]>
    In this example, the subgroup has 16 work items in wi_layout=[1, 16], each accessing 1 element as specified by wi_data=[1, 1].

    E.g., #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], wi_layout = [1, 16], wi_data = [1, 1]>
    In this example, the layout representing a workgroup work distribution. A workgroup has 8 subgroups organized as 2x4 layout.
    and each subgroup accesses a 16x16 block per instruction, which is further disbributed to 16 work items as described above.

  }];
  let parameters = (ins
    OptionalParameter<"ScopeAttr">: $scope,
    OptionalParameter<"DenseI32ArrayAttr">: $sg_layout,
    OptionalParameter<"DenseI32ArrayAttr">: $sg_data,
    OptionalParameter<"DenseI32ArrayAttr">: $order,
    "DenseI32ArrayAttr": $wi_layout,
    "DenseI32ArrayAttr": $wi_data
  );

  let extraClassDeclaration = [{
    bool isForWorkgroupLevel() {
      if (!getScope())
        return getSgLayout() && getSgData();
      return getScope() == ScopeAttr::get(getContext(), Scope::WG);
    }

    bool isForSubgroupLevel() {
      return getScope() == ScopeAttr::get(getContext(), Scope::SG);
    }

    bool isForWorkItemLevel() {
      if (!getScope())
        return !getSgLayout() && !getSgData() && !getOrder();
      return getScope() == ScopeAttr::get(getContext(), Scope::WI);
    }
  }];

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUATTRS_TD
