//===- XeGPUOps.td - XeGPU dialect operations definition ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD
#define MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD

include "mlir/IR/AttrTypeBase.td"
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
      Attribute propAttr = getPropertiesAsAttr(ctx, prop);
      if (propAttr)
        p << "<" << propAttr << ">";
    }

    static ::mlir::ParseResult parseProperties(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result) {
      if (mlir::succeeded(parser.parseLess())) {
        if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater())
          return failure();
      }
      return success();
    }

  }];
}


def XeGPU_CreateNdDescOp: XeGPU_Op<"create_nd_tdesc", [Pure, ViewLikeOpInterface,
                        AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface]> {

  let summary = "Create nd-tensor descriptor operation";
  let description = [{
    The "create_nd_tdesc" operation creates a TensorDescType which represents
    a sub-view of a 2D memory region (It can be extended to support n-D memory
    region if needed in future). Elements in the subview continuous in each
    dimension. It encodes the following important information for supporting
    Intel hardware features:

    * source: an object representing (starting address/pointer of) a 2D memory region.
        It can be either a 2D memref object, or simply a pointer represented by uint64_t type.
        for the later case, the shape and layout information of the 2D memory region should
        be explicitly passed via `shape` and `strides` parameters.
    * offsets: two index values represents offsets from the "source" at the each dimension
        at which the subview of the target memory will be created. It is encoded via two
        variables, including "offsets" and "const_offsets", such that it can
        accept various forms, such as, operands (e.g., [%c0, %c]) and attributes (e.g., [2, 4]).
    * shape: the shape information of the memory region pointed by the "source".  It is
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
    %0 = memref.alloc() : memref<1024x1024xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c0]: memref<1024x1024xf32> -> TensorDesc<8x16xf32>

    Example 2 (suppose the tensor shape inferred by the compiler is 8x16):
    %0 = memref.alloc(%h, %w) : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c0], [%h, %w], [%w, %c1]: memref<?x?xf32> -> TensorDesc<8x16xf32>

    Example 3 (suppose the tensor shape inferred by the compiler is 8x16):
    %0 = ... : ui64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c0], [%h, %w], [%w, %c1]: ui64 -> TensorDesc<8x16xf32>
  }];

  let arguments = (ins
    XeGPU_BaseAddrType: $source,
    Variadic<Index>: $offsets,
    Variadic<Index>: $shape,
    Variadic<Index>: $strides,
    DenseI64ArrayAttr: $const_offsets,
    OptionalAttr<DenseI64ArrayAttr>: $const_shape,
    OptionalAttr<DenseI64ArrayAttr>: $const_strides
  );
  let results = (outs XeGPU_TensorDesc: $TensorDesc);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $const_offsets)
    (`,` custom<DynamicIndexList>($shape, $const_shape)^
     `,` custom<DynamicIndexList>($strides, $const_strides))?
    attr-dict `:` type($source) `->` qualified(type($TensorDesc))
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Type": $tdesc, "TypedValue<MemRefType>": $source,
                   "llvm::ArrayRef<OpFoldResult>": $offsets)>,

    OpBuilder<(ins "Type": $tdesc, "TypedValue<IntegerType> ": $source,
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

    /// wrapper for matching with OffsetSizeAndStrideOpInterface
    OperandRange getSizes() {
      return getShape();
    }

    ArrayRef<int64_t> getStaticOffsets(){
      return getConstOffsets();
    }

    /// wrapper for matching with OffsetSizeAndStrideOpInterface
    /// If source is IntegerType or `const_shape` is filled,
    /// it will return `const_shape`, such that mixes of `shape`
    /// and `const_shape` will be used to represent the shape of
    /// source operand. They overide static shape from source memref type.
    ArrayRef<int64_t> getStaticSizes() {
      auto attr = getConstShapeAttr();
      if (getSourceType().isa<IntegerType>() || attr)
        return attr;

      auto memrefType = getSourceType().dyn_cast<MemRefType>();
      assert(memrefType && "Incorrect use of getStaticSizes");
      return memrefType.getShape();
    }

    /// wrapper for matching with OffsetSizeAndStrideOpInterface
    /// If source is IntegerType or `const_strides` is filled, it
    /// will return `const_strides`, such that mixes of `strides`
    /// and `const_strides` will be used to represent the strides of
    /// source operand. They overide static strides from source memref type.
    ArrayRef<int64_t> getStaticStrides() {
      auto attr = getConstStridesAttr();
      if (getSourceType().isa<IntegerType>() || attr)
        return attr;

      auto memrefType = getSourceType().dyn_cast<MemRefType>();
      assert(memrefType && "Incorrect use of getStaticStrides");
      auto [strides, offset] = getStridesAndOffset(memrefType);
      // reuse the storage of ConstStridesAttr since strides from
      // memref is not persistant
      setConstStrides(strides);
      attr = getConstStridesAttr();
      return attr;
    }

    /// Return the expected rank of each of the`static_offsets`,
    /// `static_shape` and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank;
      if (auto ty = getSourceType().dyn_cast<MemRefType>()) {
        rank = ty.getRank();
      } else {
        rank = (unsigned)getMixedOffsets().size();
      }
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`,
    /// `shape` and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    mlir::Value getViewSource() { return getSource(); }
  }];
}

def XeGPU_PrefetchNdOp : XeGPU_Op<"prefetch_nd", []> {
  let summary = "prefetches a n-D block to cache";
  let description = [{
    It issues an instruction to prefetch a block of data from continuous
    memory regions to each level of the cache based on their cache policy.

    Example:
    ```
      xegpu.prefetch_nd %tdesc {l1_hint = #xegpu.cache_hint<cached>,
                                l2_hint = #xegpu.cache_hint<cached>,
                                l3_hint = #xegpu.cache_hint<cached>}
        : !xegpu.tensor_desc<8x16xf16>
    ```

  }];

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }
  }];

  let assemblyFormat = "$TensorDesc prop-dict attr-dict `:` qualified(type($TensorDesc))";

  let hasVerifier = 1;
}


def XeGPU_LoadNdOp : XeGPU_Op<"load_nd", [AllElementTypesMatch<["value", "TensorDesc"]>,
                                         AllElementCountsMatch<["value", "TensorDesc"]>]> {
  let summary = "loads a n-D block from memory (represented by TensorDesc)"
                "to registers (represented by vector)";
  let description = [{
    LoadNdOp essentially mimics the hardware block read instruction to read
    a block of data from memory to register. It takes a set of optional cache
    hints for each level of cache, L1, L2 and L3. If hardware does not have a
    correspoding cache, Corresponding cache hint attribute will be masked.
    vnni transform is an hardware feature for Intel GPU, which is used to
    do data packing during the load for B operand of matrix operation, if
    the bit width of the data type is less then 32 bits, e.g., fp16. And
    transpose is another Intel hardware feature, which will do transpose
    operation when loading the data if the bit width of the data type is
    fp32 or fp64. It implies that vnni and transpose cannot exit at the
    same time.

    Example:
    ```
      xegpu.load_nd %1 {transpose = [1, 0],
                        l1_hint = #xegpu.cache_hint<cached>,
                        l2_hint = #xegpu.cache_hint<uncached>,
                        l3_hint = #xegpu.cache_hint<streaming>}
              : !xegpu.tensor_desc<8x16xf32> -> vector<16x8xf32>
    ```


  }];

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       OptionalAttr<I64Attr>: $vnni_axis,
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
  }];

  let assemblyFormat = "$TensorDesc prop-dict attr-dict `:` qualified(type($TensorDesc)) `->` type($value)";
  let hasVerifier = 1;
}

def XeGPU_StoreNdOp : XeGPU_Op<"store_nd", [AllShapesMatch<["value", "TensorDesc"]>,
                                       AllElementTypesMatch<["value", "TensorDesc"]>]> {
  let summary = "stores a n-D block register region back to memory, currently only supports 2D";

  let description = [{
    StoreNdOp essentially mimics the hardware block write instruction io
    write a block of data from register into the memory region as described
    by the TensorDesc. It takes a set of optional cache hints for each level
    of cache, L1, L2 and L3. If hardware does not have a correspoding cache,
    Corresponding cache hint attribute will be masked.

    Example:
    ```
      xegpu.store_nd %3, %2 {l1_hint = #xegpu.cache_hint<uncached>,
                             l2_hint = #xegpu.cache_hint<write_back>,
                             l3_hint = #xegpu.cache_hint<write_through>}
                             : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
    ```


  }];

  let arguments = (ins XeGPU_ValueType: $value,
                       XeGPU_TensorDesc: $TensorDesc,
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
  }];

  let assemblyFormat = [{$value `,` $TensorDesc prop-dict attr-dict
                        `:` type($value) `,` qualified(type($TensorDesc))}];
  let hasVerifier = 1;
}

def XeGPU_UpdateNdOffsetOp : XeGPU_Op<"update_nd_offset",
                [AllTypesMatch<["TensorDesc", "result"]>]> {
  let summary = "It updates the offsets for the TensorDesc.";
  let description = [{The op updates the offset of the given TensorDesc.
    The offsets are relative offset to the current position in the number
    of elements. It will result in a same type TensorDesc as the input.

  example:
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
    * offsets: a array containing offsets of each access point. Its size
      is fixed to the hardware supportted subgroup size, e.g., 16 on PVC,
      implying each element in the array corresponds to a work-item (SIMT lane)
      in the subgroup.
    * chunk_size: [optional attribute] indicates number of continious
      elements accessed for each offset, default is 1.

    Example 1. It assumes subgroup size is 4, and accesses a[0], a[16], a[32], a[64]
    ```
    %a = memref.alloc() : memref<1024xf32>
    %1 = xegpu.create_tdesc %a[0, 16, 32, 64]: memref<1024xf32> -> TensorDesc<4xf32>
    ```

    Example 2. It assumes subgroup size is 4, and each workitem access 8 elements.
               It will access totally 32 data elements: a[0:7], a[16:23], a[32:39], a[64:71]
    ```
    %0 = memref.alloc() : memref<1024xf32>
    %1 = xegpu.create_tdesc %0[0, 16, 32, 64] {chunk_size = 8}: memref<1024xf32> -> TensorDesc<4x8xf32>
    ```

    Example 3. It is similar to Example 2, but there is some overlaps among workitems.
               It accesses: a[0:7], a[4:11], a[8:15], a[12:19]
    ```
    %0 = memref.alloc() : memref<1024xf32>
    %1 = xegpu.create_tdesc %0[0, 4, 8, 12] {chunk_size = 8}: memref<1024xf32> -> TensorDesc<4x8xf32>
    ```




  }];

  let arguments = (ins XeGPU_BaseAddrType: $source,
                       Variadic<Index>: $offsets,
                       DenseI64ArrayAttr: $const_offsets,
                       DefaultValuedAttr<I64Attr, "1">: $chunk_size);
  let results = (outs XeGPU_TensorDesc:$TensorDesc);

  let builders = [
    OpBuilder<(ins "xegpu::TensorDescType": $TensorDesc, "Value": $source,
                   "llvm::ArrayRef<OpFoldResult>": $offsets,
                   CArg<"uint32_t", "1"> : $chunk_size)>,
  ];

  let assemblyFormat = [{
    $source
    custom<DynamicIndexList>($offsets, $const_offsets)
    attr-dict `:`  type($source) `->` qualified(type($TensorDesc))
  }];

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

    mlir::Value getViewSource() { return getSource(); }

    OpFoldResult getOffset(unsigned idx) {
      assert(idx < getNumOffsets() && "Invalid out of bound access.");
      return getMixedOffsets()[idx];
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

    Example:
    ```
      xegpu.prefetch %tdesc {l1_hint = #xegpu.cache_hint<cached>,
                             l2_hint = #xegpu.cache_hint<cached>,
                             l3_hint = #xegpu.cache_hint<cached>}
        : !xegpu.tensor_desc<16xf16>
    ```

  }];

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }
  }];

  let assemblyFormat = "$TensorDesc prop-dict attr-dict `:` qualified(type($TensorDesc))";

  let hasVerifier = 1;
}

def XeGPU_LoadGatherOp : XeGPU_Op<"load", [AllRanksMatch<["value", "TensorDesc"]>,
                                    AllElementTypesMatch<["value", "TensorDesc"]>,
                                   AllElementCountsMatch<["value", "TensorDesc"]>]> {
  let summary = "load a set of scattered data points from memory.";

  let description = [{ It (aka. load) load data per each work-item. The output
    describes the data being loaded at the subgroup level, so its size is
    consistent with the number of work-items in a subgroup. When `chunk_size_per_lane`
    attribute is larger than 1 in TensorDesc, the output vector will be 2D vector,
    with dim-1 correspoding to the chunk size.

    The mask operand masks out memory access so that it is safe to pass out-of-boundary
    addresses/offsets as long as they are masked. It applies to slots of SIMD lanes.

  Example:
  ```
    %2 = xegpu.load %1, %0 {transpose = [1, 0],
                            l1_hint = #xegpu.cache_hint<cached>,
                            l2_hint = #xegpu.cache_hint<uncached>,
                            l3_hint = #xegpu.cache_hint<uncached>}
          : !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<scattered=true>>, vector<16xi1>
            -> vector<16xf32>
  ```

  }];

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       XeGPU_MaskType: $mask,
                       OptionalAttr<DenseI64ArrayAttr>: $transpose,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);
  let results = (outs XeGPU_ValueType: $value);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }

    mlir::Type getElementType() {
      auto type = getValue().getType();
      return getElementTypeOrSelf(type);
    }

    Type getValueType() {
      return getValue().getType();
    }

    Type getMaskType() {
      return getMask().getType();
    }

  }];

  let assemblyFormat = [{$TensorDesc `,` $mask prop-dict attr-dict
      `:` qualified(type($TensorDesc)) `,` type($mask) `->` type($value)}];

  let hasVerifier = 1;
}

def XeGPU_StoreScatterOp : XeGPU_Op<"store", [AllShapesMatch<["value", "TensorDesc"]>,
                                        AllElementTypesMatch<["value", "TensorDesc"]>]> {
  let summary = "store data to scattered memory locations.";
  let description = [{ It (aka. store) stores data to scattered memory locations.
  It has similar semantic to `load_gather`.

  Example:
  ```
    %3 = xegpu.store %0, %1, %2 {l1_hint = #xegpu.cache_hint<uncached>,
                                 l2_hint = #xegpu.cache_hint<write_back>,
                                 l3_hint = #xegpu.cache_hint<write_through>}
          : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<scattered=true>>, vector<16xi1>
  ```
  }];

  let arguments = (ins
    XeGPU_ValueType: $value,
    XeGPU_TensorDesc: $TensorDesc,
    XeGPU_MaskType: $mask,
    OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
    OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
    OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }

    Type getValueType() {
      return getValue().getType();
    }

    Type getMaskType() {
      return getMask().getType();
    }
  }];

  let assemblyFormat = [{$value `,` $TensorDesc `,` $mask prop-dict attr-dict
            `:` type($value) `,` qualified(type($TensorDesc)) `,` type($mask)}];

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
    ```
      %2 = xegpu.update_offset %1, [32, 32, 32, 32]
            : !xegpu.tensor_desc<4x2xf32, #xegpu.tdesc_attr<scattered = true>>
    ```
  }];

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
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
    attr-dict `:` qualified(type($TensorDesc))
  }];
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD
