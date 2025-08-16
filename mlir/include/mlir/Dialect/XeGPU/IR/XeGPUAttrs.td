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

  let extraClassDeclaration = [{
    // return true if all fields of the BlockTensorDescAttr are set with
    // default values.
    bool hasDefaultsOnly();
  }];

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

def LayoutTrait: AttrInterface<"LayoutTrait"> {
  let cppNamespace = "::mlir::xegpu";
  let description = [{
    Common trait for all XeGPU layouts.
  }];

  let methods = [
    InterfaceMethod<"Get the rank of attribute",
                    "int64_t",
                    "getRank">,
    InterfaceMethod<"Get the SgLayout field of the attribute as integer array",
                    "std::optional<SmallVector<int64_t>>",
                    "getSgLayoutAsInt">,
    InterfaceMethod<"Get the SgData field of the attribute as integer array",
                    "std::optional<SmallVector<int64_t>>",
                    "getSgDataAsInt">,
    InterfaceMethod<[{Delinearizes a linear subgroup ID into its multidimensional
                      indices based on the effective subgroup layout.}],
                    "FailureOr<SmallVector<Value>>",
                    "delinearizeSubgroupId",
                    (ins "OpBuilder &": $builder, "Location":$loc, "Value":$linearId)>,
    InterfaceMethod<[{Generates instructions to compute multidimensional offsets for blocks
                      assigned to a subgroup identified by linearId. The shape parameter
                      represents the workgroup-level problem size. Each subgroup may access
                      multiple blocks according to round-robin distribution rules.}],
                    "FailureOr<SmallVector<SmallVector<Value>>>",
                    "getOffsets",
                    (ins "OpBuilder &": $builder, "Location":$loc, "Value":$linearId, "ArrayRef<int64_t>":$shape)>
  ];
}

def XeGPU_LayoutAttr : XeGPUAttr<"Layout", "layout", [LayoutTrait]> {
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

    std::optional<SmallVector<int64_t>> getSgLayoutAsInt() const {
      if (DenseI32ArrayAttr layout = getSgLayout())
        return llvm::to_vector_of<int64_t>(layout.asArrayRef());
      return std::nullopt;
    }

    std::optional<SmallVector<int64_t>> getSgDataAsInt() const {
      if (DenseI32ArrayAttr data = getSgData())
        return llvm::to_vector_of<int64_t>(data.asArrayRef());
      return std::nullopt;
    }

    /// Delinearizes a linear subgroup ID into its multidimensional indices
    /// based on the effective subgroup layout.
    FailureOr<SmallVector<Value>>
    delinearizeSubgroupId(OpBuilder &builder, Location loc, Value linearId);

    /// Generates instructions to compute multidimensional offsets for blocks
    /// assigned to a subgroup identified by linearId. The shape parameter
    /// represents the workgroup-level problem size. Each subgroup may access
    /// multiple blocks according to round-robin distribution rules.
    FailureOr<SmallVector<SmallVector<Value>>>
    getOffsets(OpBuilder &builder, Location loc, Value linearId, ArrayRef<int64_t> shape);

  }];

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}


def XeGPU_SliceAttr : XeGPUAttr<"Slice", "slice", [LayoutTrait]> {
  let summary = [{Describes the data distribution and sharing among subgroups or work-items.}];

  let description = [{
    Like LayoutAttr, SliceAttr describes data distribution among subgroups or work-items.
    However, whereas LayoutAttr requires the data to have the same rank as the attribute,
    SliceAttr permits the data to have a lower rank. In this case, compute units in the
    specified dimensions (given by `$dims`) share the data, provided that the remaining
    ranks match the data rank. SliceAttr is commonly used by operations such as
    vector.multi_reduction and vector.broadcast.

    Example:
    ```
    #l = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>
    #r = #xegpu.slice<#l, dim = [0]>

    %exp = math.exp %input {layout_result_0 = #l}: vector<256x128xf32>
    %red = vector.multi_reduction<add>, %exp, %acc [0] {layout_result_0 = #r}: vector<256x128xf32> to vector<128xf32>
    %bcast = vector.broadcast %red {layout_result_0 = #l} : vector<128xf32> to vector<256x128xf32>
    ```
    In this example, %red is conceptually divided into 4 vectors of type vector<32xf32>, each assigned to
    a group of subgroups. Each group consists of 8 subgroups from the same column of sg_layout, sharing a
    single reduction result of type vector<32xf32>.

  }];

  let parameters = (ins
    "xegpu::LayoutTrait": $parent,
    "DenseI64ArrayAttr": $dims
  );

  let extraClassDeclaration = [{

    int64_t getRank() const {
      SliceAttr attr = flatten();
      auto parent = dyn_cast<LayoutAttr>(attr.getParent());
      return parent.getRank() - attr.getDims().size();
    }

    DenseI32ArrayAttr getOrder() const {
      SliceAttr attr = flatten();
      auto parent = dyn_cast<LayoutAttr>(attr.getParent());
      return parent.getOrder();
    }

    bool isWgLayout() const {
      SliceAttr attr = flatten();
      auto parent = dyn_cast<LayoutAttr>(attr.getParent());
      return parent.isWgLayout();
    }

    bool isSgLayout() const {
      SliceAttr attr = flatten();
      auto parent = dyn_cast<LayoutAttr>(attr.getParent());
      return parent.isSgLayout();
    }

    /// Returns the SgLayout of the attribute, computed by applying
    /// the slice dimensions to the underlying LayoutAttr.
    std::optional<SmallVector<int64_t>> getSgLayoutAsInt() const {
      SliceAttr attr = flatten();
      auto parent = dyn_cast<LayoutAttr>(attr.getParent());
      if (auto layout = parent.getSgLayoutAsInt()) {
        ArrayRef<int64_t> dims = attr.getDims().asArrayRef();
        return XeGPUDialect::slice(llvm::ArrayRef<int64_t>(*layout), dims);
      }
      return std::nullopt;
    }

    /// Returns the SgData of the attribute, computed by applying
    /// the slice dimensions to the underlying LayoutAttr.
    std::optional<SmallVector<int64_t>> getSgDataAsInt() const {
      SliceAttr attr = flatten();
      auto parent = dyn_cast<LayoutAttr>(attr.getParent());
      if (auto data = parent.getSgDataAsInt()) {
        ArrayRef<int64_t> dims = attr.getDims().asArrayRef();
        return XeGPUDialect::slice(llvm::ArrayRef<int64_t>(*data), dims);
      }
      return std::nullopt;
    }

    /// flatten a nested SliceAttr, e.g., for 2-level nested SliceAttr
    /// #xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 12]>, dims = [0]>, dims = [0]>
    /// it will coalese two slice operations and return a simplified SliceAttr
    /// #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 12]>, dims = [0, 1]>
    SliceAttr flatten() const;

    /// Delinearizes a linear subgroup ID into its multidimensional indices
    /// based on the effective subgroup layout.
    FailureOr<SmallVector<Value>>
    delinearizeSubgroupId(OpBuilder &builder, Location loc, Value linearId);

    /// Generates instructions to compute multidimensional offsets for blocks
    /// assigned to a subgroup identified by linearId. The shape parameter
    /// represents the workgroup-level problem size. Each subgroup may access
    /// multiple blocks according to round-robin distribution rules.
    FailureOr<SmallVector<SmallVector<Value>>>
    getOffsets(OpBuilder &builder, Location loc, Value linearId, ArrayRef<int64_t> shape);

  }];

  let assemblyFormat = "`<` qualified($parent) `,` `dims` `=` $dims `>`";
  let genVerifyDecl = 1;
}

def XeGPU_RangeAttr : XeGPUAttr<"Range", "range"> {
  let summary = [{Specifies a half-open range}];
  let description = [{
    `RangeAttr` is an attribute that defines a half-open range [start, end).
    The range is inclusive of the start value and exclusive of the end value.
    One usage of this attribute can be to specify the subgroup id range.
    The subgroup id range can be specified using this attribute,
    and it can be attached to a scf.if op like
    ```mlir
    scf.if %cond {
      // some operations
    } {sg_id_range = #xegpu.range<[2, 4]>}
    ```
    In this case, the scf.if op will only be executed for subgroup IDs 2 and 3.
  }];

  let parameters = (ins
    "IntegerAttr": $start,
    "IntegerAttr": $end
  );

  let builders = [
    AttrBuilder<(ins "int":$start, "int":$end)>
  ];

  let assemblyFormat = "`<` `[`$start `,` $end `]` `>`";
  let genVerifyDecl = 1;
}

def XeGPU_MemLayoutAttr : XeGPUAttr<"MemLayout", "mem_layout"> {
  let summary = [{Specifies memory layouts with named attributes.}];

  let description = [{
    This attribute stores a collection of named attributes that describe
    memory layout properties such as stride, block, etc.
  }];

  let parameters = (ins "DictionaryAttr": $attrs);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get a specific attribute by name
    Attribute getAttr(StringRef name) const {
      return getAttrs().get(name);
    }

    /// Check if a specific attribute exists
    bool hasAttr(StringRef name) const {
      return getAttrs().contains(name);
    }

    ArrayAttr getStrides() {
      return getAttrs().getAs<ArrayAttr>("stride");
    }

  }];

}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUATTRS_TD
