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
        is 8x16, with array_length = 2. The loaded block shape will be actually
        8x32. Its default value is 1.
    3. `boundary_check`: It is used to indicates the hardware whether to do
        out-of-boundary check. The default value is true.
  }];

  let parameters = (ins
    DefaultValuedParameter<
      "MemorySpaceAttr",
      "MemorySpaceAttr::get($_ctxt, xegpu::MemorySpace::Global)",
      "Data memory location">: $memory_space,
    DefaultValuedParameter<
      "IntegerAttr",
      "IntegerAttr::get(IntegerType::get($_ctxt, 64), 1)",
      "Number of continuous blocks to load">: $array_length,
    DefaultValuedParameter<
      "BoolAttr",
      "BoolAttr::get($_ctxt, true)",
      "Checking the out of boundary access">: $boundary_check
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

    2. `chunk_size`: Specifies the number of contiguous elements accessed per offset.
      The default value is 1.
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

  let extraClassDeclaration = [{
    int64_t getChunkSizeAsInt() {
      return getChunkSize().getInt();
    }
  }];

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

def XeGPU_LayoutAttr : XeGPUAttr<"Layout", "layout"> {
  let summary = [{
    Describes the data distribution to subgroups and work-items for a tensor
    specified by the tensor descriptor.
  }];
  let description = [{
    XeGPU operations use `LayoutAttr` to define how data is distributed across subgroups and work-items.
    This attribute is specified in tensor descriptors during tensor description creation. `LayoutAttr`
    includes the following parameters:

    * `sg_layout`: Specifies the total number of subgroups and their layout within a workgroup.
      It is mandatory for workgroup-level programming. Its presence implies workgroup-level code.
    * `sg_data`: Defines the data size accessed per subgroup. It is optionally used with `sg_layout`
      for workgroup-level programming. When it is left empty, the size accessed per subgroup can be
      derived from the tensor shape and `sg_layout` using the formula:
      `sg_data[i] = tensor_shape[i] / sg_layout[i]`.
    * `inst_data`: Specifies the data size that is processed by an instruction. It is optionally
      used with lane_layout. When it is left empty, the data size per instruction is equivalent to
      the sg_data for workgroup-level programming or equivalent to tensor shape for subgroup-level
      programming.
    * `lane_layout` : Specifies the total number of work-items and their arrangement within a subgroup.
      It is mandatory for subgroup-level programming and optional for workgroup-level programming.
    * `lane_data` : Specifies the shape of the tensor fragment that each lane accesses. It defines a single,
      minimal distribution unit. Processing the entire tensor may require one or more distribution units per
      hardware instruction.
    * `order`: Specifies the dimension order used to linearize n-dimensional sg_layout and lane_layout to
      1-dimensional layout. The first dimension in the order list is the fastest-changing dimension. If it
      is not present, the default value is [1, 0].

    Examples:

    1. Subgroup level layout:
    ```mlir
    #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>
    ```
    In this example, there are 16 work-items per subgroup, and is organized as
    [[0, 1, 2, .., 7],[8, 9, .., 15]]. The distribution unit is 1x1.

    2. Subgroup level layout with order:
    ```mlir
    #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1], order = [0, 1]>
    ```
    In this example, there are 16 work-items per subgroup, and is organized as
    [[0, 2, 4, ..., 14], [1, 3, 5, ..., 15]]. The distribution unit is 1x1.

    3. Subgroup level layout with inst_data
    ```mlir
    #xegpu.layout<inst_data = [8, 16], lane_layout = [2, 8], lane_data = [2, 2]>
    ```
    In this example, the original problem size is partitioned into smaller subproblems of dimensions [8, 16],
    which are then distributed among 16 work-items arranged as [[0, 1, 2, ..., 7], [8, 9, ..., 15]]. Each
    work-item is assigned four 2x2 blocks in a round-robin manner.

    4. Workgroup level layout:
    ```mlir
    #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [2, 8], lane_data = [1, 1]>
    ```
    In this example, the layout represents a workgroup distribution. A workgroup consists of 8 subgroups
    arranged as [[0, 1, 2, 3], [4, 5, 6, 7]]. Each subgroup accesses a 16x16 block per instruction, which
    is further distributed to 16 work items which is organized as [[0, 1, 2, .., 7],[8, 9, .., 15]].

    5. Workgroup level layout with order:
    ```mlir
    #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [2, 8], lane_data = [1, 1], order = [0, 1]>
    ```
    In this example, the layout represents a workgroup distribution. A workgroup consists of 8 subgroups
    arranged as [[0, 2, 4, 6], [1, 3, 5, 7]]. Each subgroup accesses a 16x16 block per instruction, which
    is further distributed to 16 work items which is organized as [[0, 2, 4, ..., 14], [1, 3, 5, ..., 15]].

    6. Workgroup level layout with inst_data:
    ```mlir
    #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [2, 8], lane_data = [1, 1]>
    ```
    This example is similar to the previous ones, but the `inst_data` parameter divides `sg_data` into two instructions,
    each processing an 8x16 block. These blocks are further distributed across 16 work-items with a distribution unit of 1x1.
    Unlike the 2x2 distribution unit in example 3, which results in accessing contiguous 2x2 blocks, the 1x1 distribution
    unit may result in non-contiguous access.
  }];

  let parameters = (ins
    OptionalParameter<"DenseI32ArrayAttr">: $sg_layout,
    OptionalParameter<"DenseI32ArrayAttr">: $sg_data,
    OptionalParameter<"DenseI32ArrayAttr">: $inst_data,
    OptionalParameter<"DenseI32ArrayAttr">: $lane_layout,
    OptionalParameter<"DenseI32ArrayAttr">: $lane_data,
    OptionalParameter<"DenseI32ArrayAttr">: $order
  );

  let builders = [
    AttrBuilder<(ins "llvm::ArrayRef<int32_t>": $lane_layout,
                     "llvm::ArrayRef<int32_t>": $lane_data),
      [{
        auto sg_layout = DenseI32ArrayAttr();
        auto sg_data = DenseI32ArrayAttr();
        auto inst_data = DenseI32ArrayAttr();
        auto order = DenseI32ArrayAttr();
        return $_get($_ctxt, sg_layout, sg_data, inst_data,
                     DenseI32ArrayAttr::get($_ctxt, lane_layout),
                     DenseI32ArrayAttr::get($_ctxt, lane_data), order);
      }]>,
    AttrBuilder<(ins "llvm::ArrayRef<int32_t>": $lane_layout,
                     "llvm::ArrayRef<int32_t>": $lane_data,
                     "llvm::ArrayRef<int32_t>": $order),
      [{
        return $_get($_ctxt,
                     /*sg_layout =*/ nullptr,
                     /*sg_data   =*/ nullptr,
                     /*inst_data =*/ nullptr,
                     DenseI32ArrayAttr::get($_ctxt, lane_layout),
                     DenseI32ArrayAttr::get($_ctxt, lane_data),
                     DenseI32ArrayAttr::get($_ctxt, order));
      }]>,
    AttrBuilder<(ins "DenseI32ArrayAttr": $lane_layout,
                     "DenseI32ArrayAttr": $lane_data,
                     "DenseI32ArrayAttr": $order),
      [{
        return $_get($_ctxt, /*sg_layout =*/ nullptr, /*sg_data =*/ nullptr,
                  /*inst_data =*/ nullptr, lane_layout, lane_data, order);
      }]>
  ];

  let extraClassDeclaration = [{
    bool isWgLayout() {
      return getSgLayout() != nullptr;
    }

    bool isSgLayout() {
      return !isWgLayout();
    }

    int64_t getRank() {
      if (auto attr = getSgLayout())
        return attr.size();
      if (auto attr = getInstData())
        return attr.size();
      if (auto attr = getLaneLayout())
        return attr.size();
      return 0;
    }

    LayoutAttr dropSgLayoutAndData() {
      // avoid every field of the attribute is nullptr, which may lead to segment fault
      if (!getInstData() && !getLaneLayout())
        return nullptr;
      return LayoutAttr::get(getContext(), nullptr, nullptr, getInstData(),
                             getLaneLayout(), getLaneData(), getOrder());
    }

    LayoutAttr dropInstData() {
      // avoid every field of the attribute is nullptr, which may lead to segment fault
      if (!getSgLayout() && !getLaneLayout())
        return nullptr;
      return LayoutAttr::get(getContext(), getSgLayout(), getSgData(), nullptr,
                             getLaneLayout(), getLaneData(), getOrder());
    }
  }];

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUATTRS_TD
