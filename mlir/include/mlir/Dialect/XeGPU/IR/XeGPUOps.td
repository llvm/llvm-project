//===- XeGPUOps.td - XeGPU dialect operations definition ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD
#define MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/XeGPU/IR/XeGPUAttrs.td"
include "mlir/Dialect/XeGPU/IR/XeGPUDialect.td"
include "mlir/Dialect/XeGPU/IR/XeGPUTypes.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class XeGPU_Op<string mnemonic, list<Trait> traits = []>:
          Op<XeGPU_Dialect, mnemonic, traits> {

  code extraBaseClassDeclaration = [{
    void printProperties(::mlir::MLIRContext *ctx,
            ::mlir::OpAsmPrinter &p, const Properties &prop,
            ::mlir::ArrayRef<::llvm::StringRef> elidedProps) {

      DictionaryAttr propAttr = dyn_cast_if_present<mlir::DictionaryAttr>(getPropertiesAsAttr(ctx, prop));

      // filter out the elidedProps from propAttr, and get the resultAttr
      mlir::SmallVector<mlir::NamedAttribute> filteredAttrs;
      if (propAttr) {
        for (auto namedAttr : propAttr.getValue()) {
          if (llvm::is_contained(elidedProps, namedAttr.getName().strref()))
            continue;
          filteredAttrs.push_back(namedAttr);
        }
      }

      if (!filteredAttrs.empty()) {
        p << "<" << DictionaryAttr::get(ctx, filteredAttrs) << ">";
      }
    }

    static ::mlir::ParseResult parseProperties(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result) {
      if (mlir::succeeded(parser.parseOptionalLess())) {
        if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater())
          return failure();
      }
      return success();
    }

  }];
}


def XeGPU_CreateNdDescOp: XeGPU_Op<"create_nd_tdesc", [Pure, ViewLikeOpInterface, AttrSizedOperandSegments]> {

  let summary = "Create nd-tensor descriptor operation";
  let description = [{
    The "create_nd_tdesc" operation creates a TensorDescType which represents
    a sub-view of a 1D/2D memory region inside the one or two innermost dimensions
    of the source. (It can be extended to support n-D memory region if needed in
    future). Elements in the subview continuous in each dimension. It encodes the
    following important information for supporting Intel hardware features:

    * source: an object representing (starting address/pointer of) a memory region.
       It can be either a memref object, or simply a pointer represented by uint64_t type.
       For the case of dynamic memrefs or pointer, the shape and layout information of the
       memory region should be explicitly passed via `shape` and `strides` parameters.

    * offsets: index values represents offsets from the "source" at the each dimension
        at which the subview of the target memory will be created. It is encoded via
        "offsets" and "const_offsets", such that it can accept various forms, such as,
        operands (e.g., [%c0, %c]) and attributes (e.g., [2, 4]).

    * shape: the shape information of the memory region pointed by the "source". It is
         typically encoded via the MemRefType of the source, e.g., memref<4096x4096xf16>.
        But if "source" is simply a pointer represented as uint64_t type, or a memref
        type without shape information e.g., memref<?x?xf16>, the shape information has
        to be explicitly passed via the "shape" and "const_shape" arguments.

    * strides: the strides of the memory region pointed by the "source". Similar to shape,
        it is typically encoded via the MemRefType of the source too. But if "source" is
        simply a pointer represented as uint64_t type, or a memref type without shape
        information e.g., memref<?x?xf16>, the strides information has to be explicitly
        passed via the "strides" and "const_strides" argument.

    Example 1 (suppose the tensor shape inferred by the compiler is 8x16):
    ```mlir
    %0 = memref.alloc() : memref<1024x1024xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c0]: memref<1024x1024xf32> -> TensorDesc<8x16xf32>
    ```

    Example 2 (suppose the tensor shape inferred by the compiler is 8x16):
    ```mlir
    %0 = memref.alloc(%h, %w) : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c0], [%h, %w], [%w, %c1]: memref<?x?xf32> -> TensorDesc<8x16xf32>
    ```

    Example 3 (suppose the tensor shape inferred by the compiler is 8x16):
    ```mlir
    %0 = ... : ui64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c0], [%h, %w], [%w, %c1]: ui64 -> TensorDesc<8x16xf32>
    ```
  }];

  let arguments = (ins
    XeGPU_BaseAddrType: $source,
    Variadic<Index>: $offsets,
    Variadic<Index>: $shape,
    Variadic<Index>: $strides,
    OptionalAttr<DenseI64ArrayAttr>: $const_offsets,
    OptionalAttr<DenseI64ArrayAttr>: $const_shape,
    OptionalAttr<DenseI64ArrayAttr>: $const_strides
  );

  let assemblyFormat = [{
    $source ``
    custom<OptionalDynamicIndexList>($offsets, $const_offsets)
    (`,` `shape` `:` custom<DynamicIndexList>($shape, $const_shape)^
     `,` `strides``:` custom<DynamicIndexList>($strides, $const_strides))?
    attr-dict `:` type($source) `->` qualified(type($TensorDesc))
  }];

  let results = (outs XeGPU_TensorDesc: $TensorDesc);

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Type": $tdesc, "TypedValue<MemRefType>": $source)>,

    OpBuilder<(ins "Type": $tdesc, "Value ": $source,
                   "llvm::ArrayRef<OpFoldResult>": $shape,
                   "llvm::ArrayRef<OpFoldResult>": $strides)>,

    OpBuilder<(ins "Type": $tdesc, "TypedValue<MemRefType>": $source,
                   "llvm::ArrayRef<OpFoldResult>": $offsets)>,

    OpBuilder<(ins "Type": $tdesc, "Value": $source,
                   "llvm::ArrayRef<OpFoldResult>": $offsets,
                   "llvm::ArrayRef<OpFoldResult>": $shape,
                   "llvm::ArrayRef<OpFoldResult>": $strides)>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the type of the source memref operand.
    Type getSourceType() {
      return getSource().getType();
    }

    /// Returns the type of the result TensorDesc.
    xegpu::TensorDescType getType() {
      return getTensorDesc().getType();
    }

    /// Return the element type of the TensorDesc
    Type getElementType() {
      return getType().getElementType();
    }

    /// Return the shape of the TensorDesc
    llvm::ArrayRef<int64_t> getTensorDescShape() {
      return getType().getShape();
    }

    SmallVector<OpFoldResult> getMixedOffsets() {
      auto statics = getConstOffsets().value_or(SmallVector<int64_t>());
      auto dynamics = getOffsets();
      if (statics.size() == 0 && dynamics.size() == 0)
        return {};
      return getMixedValues(statics, dynamics, getContext());
    }

    SmallVector<OpFoldResult> getMixedSizes() {
      SmallVector<int64_t> statics;

      /// Get the static sizes/shape, the value passed to const_shape
      /// will overide the value in memref shape.
      if (auto memrefTy = llvm::dyn_cast<MemRefType>(getSourceType()))
        statics = llvm::to_vector(memrefTy.getShape());
      if (auto attr = getConstShapeAttr())
        statics = llvm::to_vector(attr.asArrayRef());

      return getMixedValues(statics, getShape(), getContext());
    }

    SmallVector<OpFoldResult> getMixedStrides() {
      SmallVector<int64_t> statics;

      /// Get the static strides, the value passed to const_strides
      /// will overide the value in memref.
      if (auto memrefTy = llvm::dyn_cast<MemRefType>(getSourceType()))
        statics = memrefTy.getStridesAndOffset().first;
      if (auto attr = getConstStridesAttr())
        statics = llvm::to_vector(attr.asArrayRef());

      return getMixedValues(statics, getStrides(), getContext());
    }

    /// Return the number of leading operands before the `offsets`,
    /// `shape` and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    mlir::Value getViewSource() { return getSource(); }

    unsigned getSourceMemorySpace() {
      auto srcTy = getSourceType();
      if (auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(srcTy)) {
        auto attr = memrefTy.getMemorySpace();
        if (attr) {
          if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
            return static_cast<unsigned>(intAttr.getInt());
          }
          if (auto memSpaceAttr = llvm::dyn_cast<MemorySpaceAttr>(attr))
            return static_cast<unsigned>(memSpaceAttr.getValue());
        }
      }
      // take global as default memory scope.
      return static_cast<unsigned>(MemorySpace::Global);
    }

    xegpu::DistributeLayoutAttr getLayoutAttr() {
      return dyn_cast_if_present<xegpu::DistributeLayoutAttr>(getType().getLayout());
    }

    ArrayRef<int64_t> getDataShape() {
      return getTensorDescShape();
    }

  }];
}

def XeGPU_PrefetchNdOp : XeGPU_Op<"prefetch_nd", []> {
  let summary = "prefetches a n-D block to cache";
  let description = [{
    It issues an instruction to prefetch a block of data from continuous
    memory regions to each level of the cache based on their cache policy.

    Example:
    ```mlir
      xegpu.prefetch_nd %tdesc {l1_hint = #xegpu.cache_hint<cached>,
                                l2_hint = #xegpu.cache_hint<cached>,
                                l3_hint = #xegpu.cache_hint<cached>}
        : !xegpu.tensor_desc<8x16xf16>
    ```

  }];

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       Variadic<Index>: $offsets,
                       OptionalAttr<DenseI64ArrayAttr>: $const_offsets,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }

   SmallVector<OpFoldResult> getMixedOffsets() {
      auto statics = getConstOffsets().value_or(SmallVector<int64_t>());
      auto dynamics = getOffsets();
      if (statics.size() == 0 && dynamics.size() == 0)
        return {};
      return getMixedValues(statics, dynamics, getContext());
    }

    xegpu::DistributeLayoutAttr getLayoutAttr() {
      return dyn_cast_if_present<xegpu::DistributeLayoutAttr>(getTensorDescType().getLayout());
    }

    ArrayRef<int64_t> getDataShape() {
      return getTensorDescType().getShape();
    }

  }];

  let assemblyFormat = [{
    $TensorDesc ``
    custom<OptionalDynamicIndexList>($offsets, $const_offsets)
    prop-dict attr-dict `:` qualified(type($TensorDesc))
  }];

  let builders = [
    OpBuilder<(ins "Value": $TensorDesc,
                   "xegpu::CachePolicyAttr": $l1_hint,
                   "xegpu::CachePolicyAttr": $l2_hint,
                   "xegpu::CachePolicyAttr": $l3_hint)>,
    OpBuilder<(ins "Value": $TensorDesc,
                   "ArrayRef<OpFoldResult>": $offsets,
                   "xegpu::CachePolicyAttr": $l1_hint,
                   "xegpu::CachePolicyAttr": $l2_hint,
                   "xegpu::CachePolicyAttr": $l3_hint)>
  ];

  let hasVerifier = 1;
}


def XeGPU_LoadNdOp : XeGPU_Op<"load_nd", [
  AllElementTypesMatch<["value", "TensorDesc"]>, MemoryEffects<[MemRead]>
  ]> {
  let summary = "loads a n-D block from memory (represented by TensorDesc)"
                "to registers (represented by vector)";
  let description = [{
    LoadNdOp essentially mimics the hardware block read instruction to read
    a block of data from memory to register. It takes a set of optional cache
    hints for each level of cache, L1, L2 and L3. If hardware does not have a
    correspoding cache, Corresponding cache hint attribute will be masked.
    VNNI transformation is an hardware feature for Intel GPU, which is used to
    do data packing during the load for B operand of matrix operation, if
    the bit width of the data type is less then 32 bits, e.g., fp16. And
    transpose is another Intel hardware feature, which will do transpose
    operation when loading the data if the bit width of the data type is
    fp32 or fp64. It implies that vnni and transpose cannot exit at the
    same time. It is only available to 1D or 2D blocked tensor_desc.

    In SIMT mode, result vector represents the data to be loaded by each work-item.

    Example 1:
    ```mlir
      xegpu.load_nd %1 {transpose = [1, 0],
                        l1_hint = #xegpu.cache_hint<cached>,
                        l2_hint = #xegpu.cache_hint<uncached>,
                        l3_hint = #xegpu.cache_hint<streaming>}
              : !xegpu.tensor_desc<8x16xf32> -> vector<16x8xf32>
    ```
    Example 2 (SIMT mode):
    ```mlir
      xegpu.load_nd %1 {l1_hint = #xegpu.cache_hint<cached>,
                        l2_hint = #xegpu.cache_hint<uncached>}>
        : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>
    ```


  }];

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       Variadic<Index>: $offsets,
                       OptionalAttr<DenseI64ArrayAttr>: $const_offsets,
                       OptionalAttr<UnitAttr>: $packed,
                       OptionalAttr<DenseI64ArrayAttr>: $transpose,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);

  let results = (outs XeGPU_ValueType: $value);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    VectorType getType() {
      return llvm::dyn_cast<VectorType>(getValue().getType());
    }

    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }

    SmallVector<OpFoldResult> getMixedOffsets() {
      auto statics = getConstOffsets().value_or(SmallVector<int64_t>());
      auto dynamics = getOffsets();
      if (statics.size() == 0 && dynamics.size() == 0)
        return {};
      return getMixedValues(statics, dynamics, getContext());
    }

    xegpu::DistributeLayoutAttr getLayoutAttr() {
      return dyn_cast_if_present<xegpu::DistributeLayoutAttr>(getTensorDescType().getLayout());
    }

    ArrayRef<int64_t> getDataShape() {
      return getTensorDescType().getShape();
    }


  }];

  let assemblyFormat = [{
    $TensorDesc ``
    custom<OptionalDynamicIndexList>($offsets, $const_offsets)
    prop-dict attr-dict `:` qualified(type($TensorDesc)) `->` type($value)
  }];

  let builders = [
    OpBuilder<(ins "Type": $value, "Value": $TensorDesc,
                    "UnitAttr": $packed, "DenseI64ArrayAttr": $transpose,
                    "xegpu::CachePolicyAttr": $l1_hint,
                    "xegpu::CachePolicyAttr": $l2_hint,
                    "xegpu::CachePolicyAttr": $l3_hint)>,
    OpBuilder<(ins "Type": $value, "Value": $TensorDesc,
                    "ArrayRef<OpFoldResult>": $offsets,
                    "UnitAttr": $packed, "DenseI64ArrayAttr": $transpose,
                    "xegpu::CachePolicyAttr": $l1_hint,
                    "xegpu::CachePolicyAttr": $l2_hint,
                    "xegpu::CachePolicyAttr": $l3_hint)>
  ];

  let hasVerifier = 1;
}

def XeGPU_StoreNdOp : XeGPU_Op<"store_nd", [
  AllElementTypesMatch<["value", "TensorDesc"]>, MemoryEffects<[MemWrite]>
  ]> {
  let summary = "stores a n-D block register region back to memory, currently only supports 2D";

  let description = [{
    StoreNdOp essentially mimics the hardware block write instruction io
    write a block of data from register into the memory region as described
    by the TensorDesc. It takes a set of optional cache hints for each level
    of cache, L1, L2 and L3. If hardware does not have a correspoding cache,
    Corresponding cache hint attribute will be masked.
    It is only available to 1D or 2D blocked tensor_desc.

    In SIMT mode, the input vector represents the data to be stored by each work-item.

    Example 1:
    ```mlir
      xegpu.store_nd %3, %2 {l1_hint = #xegpu.cache_hint<uncached>,
                             l2_hint = #xegpu.cache_hint<write_back>,
                             l3_hint = #xegpu.cache_hint<write_through>}
                             : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
    ```
    Example 2 (SIMT mode):
    ```mlir
      xegpu.store_nd %3, %2 {l1_hint = #xegpu.cache_hint<uncached>,
                             l2_hint = #xegpu.cache_hint<write_back>,
                             l3_hint = #xegpu.cache_hint<write_through>}
                             : vector<8xf16>, !xegpu.tensor_desc<8x16xf16>
    ```


  }];

  let arguments = (ins XeGPU_ValueType: $value,
                       XeGPU_TensorDesc: $TensorDesc,
                       Variadic<Index>: $offsets,
                       OptionalAttr<DenseI64ArrayAttr>: $const_offsets,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    VectorType getValueType() {
      return llvm::dyn_cast<VectorType>(getValue().getType());
    }

    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }

    SmallVector<OpFoldResult> getMixedOffsets() {
      auto statics = getConstOffsets().value_or(SmallVector<int64_t>());
      auto dynamics = getOffsets();
      if (statics.size() == 0 && dynamics.size() == 0)
        return {};
      return getMixedValues(statics, dynamics, getContext());
    }

    xegpu::DistributeLayoutAttr getLayoutAttr() {
      return dyn_cast_if_present<xegpu::DistributeLayoutAttr>(getTensorDescType().getLayout());
    }

    ArrayRef<int64_t> getDataShape() {
      return getTensorDescType().getShape();
    }

  }];

   let assemblyFormat = [{
    $value `,`
    $TensorDesc ``
    custom<OptionalDynamicIndexList>($offsets, $const_offsets)
    prop-dict attr-dict `:`  type($value) `,` qualified(type($TensorDesc))
  }];

  let builders = [
    OpBuilder<(ins "Value": $value, "Value": $TensorDesc,
                   "xegpu::CachePolicyAttr": $l1_hint,
                   "xegpu::CachePolicyAttr": $l2_hint,
                   "xegpu::CachePolicyAttr": $l3_hint)>,
    OpBuilder<(ins "Value": $value, "Value": $TensorDesc,
                  "ArrayRef<OpFoldResult>": $offsets,
                  "xegpu::CachePolicyAttr": $l1_hint,
                  "xegpu::CachePolicyAttr": $l2_hint,
                  "xegpu::CachePolicyAttr": $l3_hint)>
  ];


  let hasVerifier = 1;
}

def XeGPU_UpdateNdOffsetOp : XeGPU_Op<"update_nd_offset",
                [Pure, AllTypesMatch<["TensorDesc", "result"]>]> {
  let summary = "It updates the offsets for the TensorDesc.";
  let description = [{The op updates the offset of the given TensorDesc.
    The offsets are relative offset to the current position in the number
    of elements. It will result in a same type TensorDesc as the input.

  Example:
  ```
    %2 = xegpu.update_nd_offset %1, [0, 16]: !xegpu.tensor_desc<8x16xf32>
  ```
  }];

  let arguments = (ins
    XeGPU_TensorDesc: $TensorDesc,
    Variadic<Index>: $offsets,
    DenseI64ArrayAttr: $const_offsets);

  let results = (outs XeGPU_TensorDesc: $result);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }

    SmallVector<OpFoldResult> getMixedOffsets() {
      Builder b(getContext());
      return getMixedValues(getConstOffsets(), getOffsets(), b);
    }

    size_t getNumOffsets() {
      return getMixedOffsets().size();
    }

    OpFoldResult getOffset(unsigned idx) {
      assert(idx < getNumOffsets() && "Invalid out of bound access.");
      return getMixedOffsets()[idx];
    }
  }];

  let assemblyFormat = [{
    $TensorDesc `,`
    custom<DynamicIndexList>($offsets, $const_offsets)
    attr-dict `:` qualified(type($result))
  }];

  let hasVerifier = 1;
}

def XeGPU_CreateDescOp: XeGPU_Op<"create_tdesc", [Pure, ViewLikeOpInterface]> {
  let summary = "create scattered tensor descriptors (TensorDesc).";
  let description = [{
    "create_tdesc" is similar to "create_nd_tdesc" in terms that it creates
    a Tensor Descriptor (TensorDescType) for a memory region. While "create_nd_tdesc"
    is for creating continuous subviews, "create_tdesc" is for creating non-continuous
    (scattered) subviews, allowing each work-item in a subgroup specifying their own offset.
    It accepts the following parameters:

    * source: a 1D memref or pointer (uint64_t) represents the flattened memory object.
    * offsets: a vector containing offsets of each access point. Its size
      is fixed to the hardware supportted subgroup size, e.g., 16 on PVC,
      implying each element in the vector corresponds to a work-item (SIMT lane)
      in the subgroup.

    The first dimension of the result TensorDesc corresponds to work-items, so it should
    match the dimension of offsets. It may also has a second dimension corresponding to
    the chunk_size if the chunk size is larger than 1.

    Example 1: It assumes subgroup size is 4, and accesses a[0], a[16], a[32], a[64]
    ```mlir
    %a = memref.alloc() : memref<1024xf32>
    %0 = arith.constant dense<[0, 16, 32, 64]> : vector<4xindex>
    %1 = xegpu.create_tdesc %a, %0: memref<1024xf32>, vector<4xindex> -> TensorDesc<4xf32>
    ```

    Example 2: It assumes subgroup size is 4, and each workitem access 8 elements.
               It will access totally 32 data elements: a[0:7], a[16:23], a[32:39], a[64:71]
    ```mlir
    %0 = memref.alloc() : memref<1024xf32>
    %off = arith.constant dense<[0, 16, 32, 64]> : vector<4xindex>
    %1 = xegpu.create_tdesc %0, %off : memref<1024xf32>, vector<4xindex>
          -> TensorDesc<4x8xf32, #xegpu.scattered_tdesc_attr<chunk_size = 8>>
    ```

    Example 3: It is similar to Example 2, but there is some overlaps among workitems.
               It accesses: a[0:7], a[4:11], a[8:15], a[12:19]
    ```mlir
    %0 = memref.alloc() : memref<1024xf32>
    %off = arith.constant dense<[0, 4, 8, 12]> : vector<4xindex>
    %1 = xegpu.create_tdesc %0, %off : memref<1024xf32>, vector<4xindex>
          -> TensorDesc<4x8xf32, #xegpu.scattered_tdesc_attr<chunk_size = 8>>
    ```
  }];

  let arguments = (ins XeGPU_BaseAddrType: $source,
                       XeGPU_OffsetType: $offsets);
  let results = (outs XeGPU_TensorDesc:$TensorDesc);

  let builders = [
    OpBuilder<(ins "xegpu::TensorDescType": $TensorDesc, "mlir::Value": $source,
                   "llvm::ArrayRef<OpFoldResult>": $offsets)>,
    OpBuilder<(ins "xegpu::TensorDescType": $TensorDesc, "mlir::Value": $source,
                   "llvm::ArrayRef<int64_t>": $offsets)>,
  ];

  let assemblyFormat = [{
    $source `,` $offsets attr-dict `:`  type($source) `,` type($offsets) `->` qualified(type($TensorDesc))
  }];

  let extraClassDeclaration = [{
    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }

    mlir::VectorType getOffsetsType() {
      return getOffsets().getType();
    }

    size_t getNumOffsets() {
      return getOffsetsType().getNumElements();
    }

    mlir::Value getViewSource() { return getSource(); }

    unsigned getSourceMemorySpace() {
      auto srcTy = getSource().getType();
      if (auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(srcTy)) {
        auto attr = memrefTy.getMemorySpace();
        if (attr) {
          if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
            return static_cast<unsigned>(intAttr.getInt());
          if (auto memSpaceAttr = llvm::dyn_cast<MemorySpaceAttr>(attr))
            return static_cast<unsigned>(memSpaceAttr.getValue());
        }
      }
      // take global as default memory scope.
      return static_cast<unsigned>(MemorySpace::Global);
    }

  }];

  let hasVerifier = 1;
}

def XeGPU_PrefetchOp : XeGPU_Op<"prefetch", []> {
  let summary = "prefetches a set of scattered data points to cache";

  let description = [{
    It issues instructions to prefetch a set of scattered data points
    from memory to each level of the cache based on their cache policy.
    As compared to prefetch_nd, which works on non-scattered TensorDesc,
    it works on scattered TensorDesc instead.

    Example 1:
    ```mlir
      xegpu.prefetch %tdesc {l1_hint = #xegpu.cache_hint<cached>,
                             l2_hint = #xegpu.cache_hint<cached>,
                             l3_hint = #xegpu.cache_hint<cached>}
        : !xegpu.tensor_desc<16xf16>
    ```

    Example 2:
    A variant accepts memref as base pointer and an offset instead of scattered TensorTdesc.
    It combines "create scattered TensorTdesc" and "prefetch with scattered TensorTdesc".
    The source operand could be a raw pointer (uint64_t).
    Please refer to create_tdesc for the restriction of memref.
    ```mlir
      %a = memref.alloc() : memref<1024xf32>
      %0 = arith.constant dense<[0, 16, 32, 64]> : vector<4xindex>
      xegpu.prefetch %a[%0] {l1_hint = #xegpu.cache_hint<cached>,
                             l2_hint = #xegpu.cache_hint<cached>,
                             l3_hint = #xegpu.cache_hint<cached>}
        : memref<1024xf32>, vector<4xindex>
    ```

  }];

  let arguments = (ins XeGPU_GatherScatterSourceType: $source,
                       Optional<XeGPU_OffsetType>: $offsets,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    Type getSourceType() {
      return getSource().getType();
    }

    TypedValue<xegpu::TensorDescType> getTensorDesc() {
      if (auto tdescType = getTensorDescType()) {
        return llvm::cast<TypedValue<xegpu::TensorDescType>>(getSource());
      }
      return TypedValue<xegpu::TensorDescType>();
    }

    xegpu::TensorDescType getTensorDescType() {
      return dyn_cast<xegpu::TensorDescType>(getSourceType());
    }

  }];

  let assemblyFormat = [{
    $source
    (`[` $offsets^ `]`)?
    prop-dict
    attr-dict `:` type(operands)
  }];

  let builders = [
    OpBuilder<(ins "Value": $source,
                    "xegpu::CachePolicyAttr": $l1_hint,
                    "xegpu::CachePolicyAttr": $l2_hint,
                    "xegpu::CachePolicyAttr": $l3_hint)>
  ];

  let hasVerifier = 1;
}

def XeGPU_LoadGatherOp : XeGPU_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "load a set of scattered data points from memory.";

  let description = [{ It (aka. load) load data per each work-item. The output
    describes the data being loaded at the subgroup level, so its size is
    consistent with the number of work-items in a subgroup. When the chunk size
    is larger than 2, the output vector is a 2D vector, with dim-0 correspoding
    to work-items, and dim-1 corresponding to the chunk size loaded by each work-item.
    The mask operand masks out memory access so that it is safe to pass out-of-boundary
    addresses/offsets as long as they are masked. It applies to slots of SIMD lanes.

    In SIMT mode, the result vector represents the data to be loaded by each work-item.
    Each work-item recieves a `chunk_size` number of elements.

  Example 1:
  ```mlir
    %2 = xegpu.load %1, %0 <{l1_hint = #xegpu.cache_hint<cached>,
                             l2_hint = #xegpu.cache_hint<uncached>,
                             l3_hint = #xegpu.cache_hint<uncached>}>
          : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space=global>>,
            vector<16xi1> -> vector<16xf32>
  ```

  Example 2:
  ```mlir
    %2 = xegpu.load %1, %0 <{l1_hint = #xegpu.cache_hint<cached>,
                             l2_hint = #xegpu.cache_hint<uncached>,
                             l3_hint = #xegpu.cache_hint<uncached>}>
          : !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<memory_space=global, chunk_size=8>>,
            vector<16xi1> -> vector<16x8xf32>
  ```

  Example 3 (SIMT mode):
  ```mlir
    %2 = xegpu.load %1, %0 <{l1_hint = #xegpu.cache_hint<cached>,
                             l2_hint = #xegpu.cache_hint<uncached>,
                             l3_hint = #xegpu.cache_hint<uncached>}>
          : !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<memory_space=global, chunk_size=8>>
            vector<16xi1> -> vector<8xf32>
  ```

  Example 4:
  A variant accepts memref as base pointer and an offset instead of scattered TensorTdesc.
  It combines "create scattered TensorTdesc" and "load with scattered TensorTdesc".
  The source operand could be a raw pointer (uint64_t). Please refer to create_tdesc
  for the restriction of memref.
  ```mlir
    %a = memref.alloc() : memref<1024xf32>
    %offsets = vector.step : vector<16xindex>
    %mask = vector.constant_mask [16]: vector<16xi1>
    %val = xegpu.load %a[%offsets], %mask {l1_hint = #xegpu.cache_hint<cached>,
                           l2_hint = #xegpu.cache_hint<cached>,
                           l3_hint = #xegpu.cache_hint<cached>}
      : memref<1024xf32>, vector<16xi1>, vector<16xindex> -> vector<16xf32>
  ```

  }];

  let arguments = (ins XeGPU_GatherScatterSourceType: $source,
                       Optional<XeGPU_OffsetType>: $offsets,
                       XeGPU_MaskType: $mask,
                       OptionalAttr<I64Attr>: $chunk_size,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);
  let results = (outs XeGPU_ValueType: $value);

  let extraClassDeclaration = extraBaseClassDeclaration # [{

    Type getSourceType() {
      return getSource().getType();
    }

    TypedValue<xegpu::TensorDescType> getTensorDesc() {
      if (auto tdescType = getTensorDescType()) {
        return llvm::cast<TypedValue<xegpu::TensorDescType>>(getSource());
      }
      return TypedValue<xegpu::TensorDescType>();
    }

    xegpu::TensorDescType getTensorDescType() {
       return dyn_cast<xegpu::TensorDescType>(getSourceType());
    }

    mlir::Type getElementType() {
      auto type = getValue().getType();
      return getElementTypeOrSelf(type);
    }

    VectorType getValueType() {
      return llvm::dyn_cast<VectorType>(getValue().getType());
    }

    Type getMaskType() {
      return getMask().getType();
    }

  }];

  let assemblyFormat = [{
    $source
    (`[` $offsets^ `]`)? `,`
    $mask prop-dict
    attr-dict `:` type(operands) `->` type($value)
  }];

  let builders = [
    OpBuilder<(ins "Type": $value, "Value": $source, "Value": $mask,
                    "xegpu::CachePolicyAttr": $l1_hint,
                    "xegpu::CachePolicyAttr": $l2_hint,
                    "xegpu::CachePolicyAttr": $l3_hint)>
   ];

  let hasVerifier = 1;
}

def XeGPU_StoreScatterOp : XeGPU_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "store data to scattered memory locations.";
  let description = [{ It (aka. store) stores data to scattered memory locations. The value is
  typically a 1D vector. But when the chunk size of the TensorDesc is larger than 1, it will be
  a 2D vector instead. For the later case, dim-1 of the value correspods to the simd lanes
  and the dim-0 of the value corresponds to the chunk size stored per lane. So `store_scatter`
  has transpose effect, which is similar to `load_gather`. Therefore, a transpose attribute is
  introduced on purpose, making sure users are aware of this implicit transformation.

  In SIMT mode, the input vector represents the data to be stored by each work-item.
  Each work-item stores a `chunk_size` number of elements.

  Example 1:
  ```mlir
    xegpu.store %0, %1, %2 <{l1_hint = #xegpu.cache_hint<uncached>,
                             l2_hint = #xegpu.cache_hint<write_back>,
                             l3_hint = #xegpu.cache_hint<write_through>}>
          : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scattered_tdesc_attr<>>, vector<16xi1>
  ```

  Example 2:
  ```mlir
    xegpu.store %0, %1, %2 <{l1_hint = #xegpu.cache_hint<uncached>,
                             l2_hint = #xegpu.cache_hint<write_back>,
                             l3_hint = #xegpu.cache_hint<write_through>}>
          : vector<16x8xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scattered_tdesc_attr<chunk_size=8>>, vector<16xi1>
  ```

  Example 3 (SIMT mode):
  ```mlir
    xegpu.store %0, %1, %2 <{l1_hint = #xegpu.cache_hint<uncached>,
                             l2_hint = #xegpu.cache_hint<write_back>,
                             l3_hint = #xegpu.cache_hint<write_through>}>
          : vector<8xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scattered_tdesc_attr<chunk_size=8>> vector<16xi1>
  ```

  Example 4:
  A variant accepts memref as base pointer and an offset instead of scattered TensorTdesc.
  It combines "create scattered TensorTdesc" and "store with scattered TensorTdesc".
  The dest operand could be a raw pointer (uint64_t).
  Please refer to create_tdesc for the restriction of memref.
  ```mlir
    %a = memref.alloc() : memref<1024xf32>
    %val = arith.constant dense<0.0> : vector<16xf32>
    %offsets = vector.step : vector<16xindex>
    %mask = vector.constant_mask [16]: vector<16xi1>
    xegpu.store %val, %a[%offsets], %mask {l1_hint = #xegpu.cache_hint<cached>,
                           l2_hint = #xegpu.cache_hint<cached>,
                           l3_hint = #xegpu.cache_hint<cached>}
      : memref<1024xf32>, vector<16xi1>, vector<16xindex> -> vector<16xf32>
  ```

  }];

  let arguments = (ins
    XeGPU_ValueType: $value,
    XeGPU_GatherScatterSourceType: $dest,
    Optional<XeGPU_OffsetType>: $offsets,
    XeGPU_MaskType: $mask,
    OptionalAttr<I64Attr>: $chunk_size,
    OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
    OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
    OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    Type getDestType() {
      return getDest().getType();
    }

    TypedValue<xegpu::TensorDescType> getTensorDesc() {
      if (auto tdescType = getTensorDescType()) {
        return llvm::cast<TypedValue<xegpu::TensorDescType>>(getDest());
      }
      return TypedValue<xegpu::TensorDescType>();
    }

    xegpu::TensorDescType getTensorDescType() {
      return dyn_cast<xegpu::TensorDescType>(getDestType());
    }

    VectorType getValueType() {
      return llvm::dyn_cast<VectorType>(getValue().getType());
    }

    Type getMaskType() {
      return getMask().getType();
    }
  }];

  let assemblyFormat = [{
    $value `,`
    $dest
    (`[` $offsets^ `]`)? `,`
    $mask
    prop-dict
    attr-dict `:`  type(operands)
  }];

  let builders = [
    OpBuilder<(ins "Value": $value, "Value": $dest, "Value": $mask,
                    "xegpu::CachePolicyAttr": $l1_hint,
                    "xegpu::CachePolicyAttr": $l2_hint,
                    "xegpu::CachePolicyAttr": $l3_hint)>
   ];

  let hasVerifier = 1;
}

def XeGPU_UpdateOffsetOp: XeGPU_Op<"update_offset",
          [AllTypesMatch<["TensorDesc", "result"]>]> {
  let summary = "It updates the offsets for the given tensor descriptor";

  let description = [{It behaves similar to `update_nd_offset` in terms that
    it updates offset of a TensorDesc, and the offsets are relative offset to
    the current position in the number of elements. However, `update_nd_offset`
    is to update the start point of a 2D block, so its offset constains two
    elements representing the shift in each dimension. `update_offset` is to
    update the offset per work-item, so its offsets contains values representing
    shifts for each work-item.

    Example:
    ```mlir
      %off = arith.constant dense<[32, 32, 32, 32]> : vector<4xindex>
      %2 = xegpu.update_offset %1, %off :
              !xegpu.tensor_desc<4x2xf32, #xegpu.scattered_tdesc_attr<chunk_size=2>>, vector<4xindex>
    ```

  }];

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       XeGPU_OffsetType: $offsets);
  let results = (outs XeGPU_TensorDesc: $result);

  let builders = [
    OpBuilder<(ins "mlir::Value": $TensorDesc,
                   "llvm::ArrayRef<OpFoldResult>": $offsets)>,
    OpBuilder<(ins "mlir::Value": $TensorDesc,
                   "llvm::ArrayRef<int64_t>": $offsets)>
  ];

  let extraClassDeclaration = [{
    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }

    mlir::VectorType getOffsetsType() {
      return getOffsets().getType();
    }

    size_t getNumOffsets() {
      return getOffsetsType().getNumElements();
    }
  }];

  let assemblyFormat = [{
    $TensorDesc `,` $offsets attr-dict `:` qualified(type($TensorDesc)) `,` type($offsets)
  }];

  let hasVerifier = 1;
}

def XeGPU_DpasOp : XeGPU_Op<"dpas", [Pure, AllElementTypesMatch<["lhs", "rhs"]>]> {
  let summary = "It performs mma computation";

  let description = [{DPAS performs matrix multiplication on matrix A of `mxk`
    size, B of `kxn` size, and accumulate on matrix C of `mxn` to the same size
    matrix , `m=8`, `n=16` and `k=8 * 32/bit_width_of_elem_type`. So for fp16
    data type, the matrices are `A: vector<8x16xf16>`, `B: vector<16x16xf16>`,
    and `C/D: vector<8x16xf32>`. Besides the matrix size requirements, DPAS
    also requires A and B to be loaded with the required data layout. Specially,
    VNNI layout is required for B operand. It is achieved via adding `packed`
    attribute to the `load_nd` operator.  Due to the VNNI transformation, B operands
    can be represented as a 3D vector, with the last dimension representing the VNNI
    factor, which is computed as `32/bit_width_of_elem_type`. Thus, `B: vector<16x16xf16>`
    can be represented as `B: vector<8x16x2xf16>`.

    In SIMT code, each work-item from a subgroup holds a data fragment for A, B, C and the result,
    which are represented as 1D vectors. Please refer to [OpenCL Intel extentions]
    (https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html)
    for more details about the fragment distribution.

    Note: on PVC, the hardware can perform load with VNNI transformation when data
          element type is 16-bit or lower precision, taking 2 or 4 elements from
          the first dimension and inserted into the newly added innermost dimension.
  }];

  let arguments = (ins
    XeGPU_DpasOprType : $lhs,
    XeGPU_DpasOprType : $rhs,
    Optional<XeGPU_DpasResType>: $acc);
  let results = (outs XeGPU_DpasResType: $result);

  let extraClassDeclaration = [{
    VectorType getLhsType() {
      return getLhs().getType();
    }

    VectorType getRhsType() {
      return getRhs().getType();
    }

    VectorType getAccType() {
      if (getAcc())
        return getAcc().getType();
      return {};
    }

    VectorType getResultType() {
      return getResult().getType();
    }

    bool hasAcc() {
      return getAcc() != nullptr;
    }
  }];

  let assemblyFormat = [{
    $lhs `,` $rhs (`,` $acc^)? attr-dict `:` type($lhs)`,` type($rhs) (`,` type($acc)^)?  `->` type($result)
  }];

  let hasVerifier = 1;
}

def XeGPU_AtomicRMWOp: XeGPU_Op<"atomic_rmw", [Pure,
      MemoryEffects<[MemRead, MemWrite]>,
      AllElementTypesMatch<["tensorDesc", "value", "result"]>,
      AllShapesMatch<["tensorDesc", "value", "result"]>]> {
  let summary = "Atomic read-modify-write operation on the TensorDesc. ";

  let description = [{
    The `xegpu.atomic_rmw` operation provides a way to perform a read-modify-write
    operation on the region described by the `TensorDesc` free from data races. The
    `kind` enumeration specifies the modification to be performed, The `mask` operand
    has the same shape with `TensorDesc`, and is used to enable or disable specific
    data points of the `TensorDesc`. The `value` operand represents the new value to
    be applied during the modification.
  }];

  let arguments = (ins
    AtomicRMWKindAttr:$kind,
    XeGPU_TensorDesc:$tensorDesc,
    XeGPU_MaskType:$mask,
    XeGPU_ValueType:$value);

  let results = (outs XeGPU_ValueType:$result);

  let assemblyFormat = [{
    $kind $tensorDesc `,` $mask `,` $value attr-dict `:`
    qualified(type($tensorDesc)) `,` type($mask) `,` type($value) `->` type($result)
  }];
}

def XeGPU_AllocNbarrierOp: XeGPU_Op<"alloc_nbarrier", []> {
  let summary = "It allocates a set of named barriers.";
  let description = [{AllocNbarrier is to create a set of named barriers as
  specified by `nbarrier_num`. Named barriers are workgroup level resources,
    and are shared by all threads in the workgroup. For example, there are
    up to 32 barriers (range 0-31) for each XeCore on PVC. A typical use case
    is that a workgroup is partitioned into N subgroups of threads (N <= 32),
    and each subgroup coordinating their work with a separate barrier with id
    range from 0 to N respectively.}];
  let arguments = (ins I64Attr: $nbarrier_num);
  let assemblyFormat = "$nbarrier_num attr-dict";
}

def XeGPU_InitNbarrierOp: XeGPU_Op<"init_nbarrier", []> {
  let summary = "It assigns a named barrier to the current thread.";
  let description = [{InitNbarrierOp assigns the named barrier with the specified
      barrier ID (0~31) to the current thread. Multiple threads may bind to the
      same named barrier, and the `participant_thread_num` specifies the total
      number of threads associated with the nbarrier. It returns an object of
      NbarrierType representing the barrier}];

  let arguments = (ins I8: $nbarrier_id,
                       I8: $participant_thread_num);
  let results = (outs XeGPU_Nbarrier: $result);
  let assemblyFormat = [{
    $nbarrier_id `,` $participant_thread_num attr-dict `:`
    type($nbarrier_id) `,` type($participant_thread_num) `->` qualified(type($result))
  }];
}

def XeGPU_NbarrierArriveOp: XeGPU_Op<"nbarrier_arrive", []> {
  let summary = "It signals the arrival at the named barrier.";
  let description = [{NbarrierArriveOp signals the hardware (or other threads)
    that the current thread has produced its data for the consumer threads. When
    the hardware signalled by `participant_thread_num` threads for the named barrier,
    it will notify the threads waiting for the named barrier to continue their work.}];

  let arguments = (ins XeGPU_Nbarrier: $nbarrier);
  let assemblyFormat = [{ $nbarrier attr-dict `:` qualified(type($nbarrier))}];
}

def XeGPU_NbarrierWaitOp: XeGPU_Op<"nbarrier_wait", []> {
  let summary = "It waits for a named barrier.";
  let description = [{NbarrierWaitOp signals the hardware which named barrier
    the current thread is waiting for, such that it can get notified when the
    named barrier is completed.}];
  let arguments = (ins XeGPU_Nbarrier: $nbarrier);
  let assemblyFormat = [{ $nbarrier attr-dict `:` qualified(type($nbarrier)) }];
}

def XeGPU_FenceOp: XeGPU_Op<"fence", []> {
  let summary = "It synchronizes memory accesses.";
  let description = [{It synchronizes the memory access between
    write and following read or write.
    1. `Memory_kind` describes the memory kind. "global" means the global memory,
        "slm" means the share local memory.
    2. `Fence_scope` describes the scope of fence. "Workgroup" means that the scope would be
        within each workgroup. "GPU" means the scope would be across workgroups within the GPU.
  }];
  let arguments = (ins XeGPU_MemorySpaceAttr: $memory_kind,
                       XeGPU_FenceScopeAttr: $fence_scope);
  let assemblyFormat = [{`memory_kind` `=` `` $memory_kind `,` `fence_scope` `=` `` $fence_scope attr-dict}];
  let extraClassDeclaration = extraBaseClassDeclaration;
}

def XeGPU_ConvertLayoutOp: XeGPU_Op<"convert_layout", [Pure, AllTypesMatch<["source", "result"]>]> {
    let summary = "Convert the layout of the input operand";
    let description = [{
      `convert_layout` redistribute data across subgroups and/or work-items from the `input_layout` to
      the `target_layout`. Both `input_layout` and `target_layout` must correspond to the same programming
      scope, such as workgroup-level (wg) or subgroup-level (sg) code. This operation is not valid once
      the IR is lowered to WI level because that is the end result of all distributions.
    }];
    let arguments = (ins XeGPU_VectorType: $source,
                         XeGPU_LayoutAttr: $input_layout,
                         XeGPU_LayoutAttr: $target_layout);
    let results = (outs XeGPU_VectorType: $result);
    let assemblyFormat = [{
        $source prop-dict attr-dict `:` type($source)
    }];

    let hasFolder = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def isSharedPred : CPred<"isSharedMemory(llvm::cast<mlir::MemRefType>($_self))">;
class StaticShared1DMemRefOf<list<Type> allowedTypes> :
  ConfinedType<MemRefRankOf<allowedTypes, [1]>, [HasStaticShapePred, isSharedPred],
     "statically shaped " # MemRefOf<allowedTypes>.summary # " for shared memory",
     "mlir::MemRefType">;

class SizeInBits<string name> :
  StrFunc<"llvm::cast<mlir::ShapedType>($" # name # ".getType()).getNumElements()"
          "*llvm::cast<mlir::ShapedType>($" # name # ".getType()).getElementTypeBitWidth()">;
class AllMemSizesMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, SizeInBits<"_self">.result,
                              "size in bits">;

def XeGPU_CreateMemDescOp: XeGPU_Op<"create_mem_desc", [Pure,
      AllMemSizesMatch<["source", "mem_desc"]>]>  {
  let summary = "Create a memory descriptor.";
  let description = [{
    Creates a memory descriptor from a shared local memory (SLM) buffer, and xegpu
    specific memory layout. The resulting memory descriptor has to have the same size
    as the underlying shared local memory.

    Arguments:
     - `source` : a 1D statically shaped memref with element type i8, representing the raw SLM buffer.
    Results:
     - `mem_desc` : the memory descriptor.
  }];
  let arguments = (ins StaticShared1DMemRefOf<[I8]>:$source);
  let results = (outs XeGPU_MemDesc:$mem_desc);
  let assemblyFormat = "$source prop-dict attr-dict `` `:` type($source) `->` qualified(type($mem_desc))";
}

def XeGPU_LoadMatrixOp: XeGPU_Op<"load_matrix", [MemoryEffects<[MemRead]>,
                              AllElementTypesMatch<["mem_desc", "res"]>,
                              AllRanksMatch<["mem_desc", "res"]>]>  {
  let arguments = (ins XeGPU_MemDesc:$mem_desc,
    Variadic<Index>: $offsets,
    DenseI64ArrayAttr: $const_offsets,
    OptionalAttr<DistributeLayoutAttr>:$layout
  );
  let results = (outs XeGPU_ValueType:$res);
  let assemblyFormat = [{
    $mem_desc `` custom<DynamicIndexList>($offsets, $const_offsets)
    prop-dict attr-dict `` `:` type(operands) `->` type(results)
  }];

  let description = [{
    This operation loads a 2D block of data from shared local memory (SLM) as specified
    by the provided 2D `mem_desc`. Only 2D memory descriptors are supported; use the
    subview operation to obtain a compatible 2D `mem_desc` from a higher-rank descriptor if needed.

    Arguments:
     - `mem_desc`: the memory descriptor identifying the SLM region.
     - `offsets`: the coordinates within the matrix to read from.
     - `layout`: [optional] An attribute for guiding distributions among
                 subgroups and/or work-items. It currently can accept either
                 LayoutAttr or SliceAttr.
    Results:
     - `res`: the matrix elements loaded from SLM.
  }];

  let builders = [
    OpBuilder<(ins "Type":$res, "TypedValue<MemDescType>": $mem_desc,
                    "llvm::ArrayRef<OpFoldResult>": $offsets, "DistributeLayoutAttr": $layout)>,
  ];
  let extraClassDeclaration = [{
    SmallVector<OpFoldResult> getMixedOffsets() {
      return getMixedValues(getConstOffsets(), getOffsets(), getContext());
    }

    ArrayRef<int64_t> getDataShape() {
      return getRes().getType().getShape();
    }
  }];

  let hasVerifier = 1;
}

def XeGPU_StoreMatrixOp: XeGPU_Op<"store_matrix", [MemoryEffects<[MemWrite]>,
                              AllElementTypesMatch<["mem_desc", "data"]>,
                              AllRanksMatch<["mem_desc", "data"]>]> {
  let arguments = (ins
    XeGPU_ValueType:$data,
    XeGPU_MemDesc:$mem_desc,
    Variadic<Index>: $offsets,
    DenseI64ArrayAttr: $const_offsets,
    OptionalAttr<DistributeLayoutAttr>:$layout
  );
  let assemblyFormat = [{ $data `,` $mem_desc `` custom<DynamicIndexList>($offsets, $const_offsets)
                          prop-dict attr-dict `` `:` type(operands)}];
  let description = [{
    This operation stores a 2D `data` fragment into the shared local memory region
    specified by a 2D `mem_desc`. Only 2D memory descriptors are supported; use the
    subview operation to obtain a 2D `mem_desc` from a higher-rank descriptor if needed.

    Arguments:
     - `mem_desc`: the memory descriptor specifying the SLM region.
     - `offsets`: the coordinates within the matrix where the data will be written.
     - `data`: the values to be stored in the matrix.
     - `layout`: [optional] An attribute for guiding distributions among
                 subgroups and/or work-items. It currently can accept either
                 LayoutAttr or SliceAttr.
  }];
  let builders = [
    OpBuilder<(ins "Value" : $data, "TypedValue<MemDescType>": $mem_desc,
                   "llvm::ArrayRef<OpFoldResult>": $offsets, "DistributeLayoutAttr": $layout)>,
  ];
  let extraClassDeclaration = [{
    SmallVector<OpFoldResult> getMixedOffsets() {
      return getMixedValues(getConstOffsets(), getOffsets(), getContext());
    }

    ArrayRef<int64_t> getDataShape() {
      return getData().getType().getShape();
    }

  }];

  let hasVerifier = 1;
}

def XeGPU_MemDescSubviewOp: XeGPU_Op<"mem_desc_subview",
          [Pure, ViewLikeOpInterface, AllElementTypesMatch<["src", "res"]>]> {
  let description = [{
    Creates a subview of a memory descriptor. The resulting memory descriptor can have
    a lower rank than the source; in this case, the result dimensions correspond to the
    higher-order dimensions of the source memory descriptor.

    Arguments:
     - `src` : a memory descriptor.
     - `offsets` : the coordinates within the matrix the subview will be created from.

    Results:
    - `res` : a memory descriptor with smaller size.

  }];
  let arguments = (ins XeGPU_MemDesc:$src,
                       Variadic<Index>:$offsets,
                       DenseI64ArrayAttr:$const_offsets);
  let results = (outs XeGPU_MemDesc:$res);
  let assemblyFormat = [{$src `` custom<DynamicIndexList>($offsets, $const_offsets) prop-dict
                         attr-dict `` `:` qualified(type($src)) `->` qualified(type($res))}];
  let builders = [
    OpBuilder<(ins "Type": $res, "Value":$src, "llvm::ArrayRef<OpFoldResult>": $offsets)>
  ];

  let extraClassDeclaration = [{
    mlir::Value getViewSource() { return getSrc(); }

    SmallVector<OpFoldResult> getMixedOffsets() {
      return getMixedValues(getConstOffsets(), getOffsets(), getContext());
    }
  }];

  let hasVerifier = 1;
}


#endif // MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD
