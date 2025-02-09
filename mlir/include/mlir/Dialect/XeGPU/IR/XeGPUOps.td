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
      Attribute propAttr = getPropertiesAsAttr(ctx, prop);
      if (propAttr)
        p << "<" << propAttr << ">";
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


def XeGPU_CreateNdDescOp: XeGPU_Op<"create_nd_tdesc", [Pure, ViewLikeOpInterface,
                        AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface]> {

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

    OpBuilder<(ins "Type": $tdesc, "TypedValue<MemRefType> ": $source,
                   "llvm::ArrayRef<OpFoldResult>": $offsets,
                   "llvm::ArrayRef<OpFoldResult>": $shape,
                   "llvm::ArrayRef<OpFoldResult>": $strides)>,

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
      if (llvm::isa<IntegerType>(getSourceType()) || attr)
        return attr;

      auto memrefType = llvm::dyn_cast<MemRefType>(getSourceType());
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
      if (llvm::isa<IntegerType>(getSourceType()) || attr)
        return attr;

      auto memrefType = llvm::dyn_cast<MemRefType>(getSourceType());
      assert(memrefType && "Incorrect use of getStaticStrides");
      auto [strides, offset] = memrefType.getStridesAndOffset();
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
      if (auto ty = llvm::dyn_cast<MemRefType>(getSourceType())) {
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
    same time.

    Example:
    ```mlir
      xegpu.load_nd %1 {transpose = [1, 0],
                        l1_hint = #xegpu.cache_hint<cached>,
                        l2_hint = #xegpu.cache_hint<uncached>,
                        l3_hint = #xegpu.cache_hint<streaming>}
              : !xegpu.tensor_desc<8x16xf32> -> vector<16x8xf32>
    ```


  }];

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
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
  }];

  let assemblyFormat = "$TensorDesc prop-dict attr-dict `:` qualified(type($TensorDesc)) `->` type($value)";
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

    Example:
    ```mlir
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
    * offsets: a vector containing offsets of each access point. Its size
      is fixed to the hardware supportted subgroup size, e.g., 16 on PVC,
      implying each element in the vector corresponds to a work-item (SIMT lane)
      in the subgroup.

    The first dimension of the result TensorDesc corresponds to work-items, so it should
    match the dimension of offsets. It may also has a second dimension corresponding to
    the chunk_size if the chunk size is larger than 1.

    Example 1. It assumes subgroup size is 4, and accesses a[0], a[16], a[32], a[64]
    ```mlir
    %a = memref.alloc() : memref<1024xf32>
    %0 = arith.constant dense<[0, 16, 32, 64]> : vector<4xindex>
    %1 = xegpu.create_tdesc %a, %0: memref<1024xf32>, vector<4xindex> -> TensorDesc<4xf32>
    ```

    Example 2. It assumes subgroup size is 4, and each workitem access 8 elements.
               It will access totally 32 data elements: a[0:7], a[16:23], a[32:39], a[64:71]
    ```mlir
    %0 = memref.alloc() : memref<1024xf32>
    %off = arith.constant dense<[0, 16, 32, 64]> : vector<4xindex>
    %1 = xegpu.create_tdesc %0, %off : memref<1024xf32>, vector<4xindex>
          -> TensorDesc<4x8xf32, #xegpu.scattered_tdesc_attr<chunk_size = 8>>
    ```

    Example 3. It is similar to Example 2, but there is some overlaps among workitems.
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

    Example:
    ```mlir
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

def XeGPU_LoadGatherOp : XeGPU_Op<"load", [
  AllElementTypesMatch<["value", "TensorDesc"]>, MemoryEffects<[MemRead]>
  ]> {
  let summary = "load a set of scattered data points from memory.";

  let description = [{ It (aka. load) load data per each work-item. The output
    describes the data being loaded at the subgroup level, so its size is
    consistent with the number of work-items in a subgroup. When the chunk size
    is larger than 2, the output vector is a 2D vector, with dim-1 correspoding
    to work-items, and dim-0 corresponding to the chunk size loaded by each work-item.
    Specially, there is a transpose effect on the result (as compared to the TensorDesc)
    due to the hardware implementation. Therefore, a transpose attribute is introduced
    on purpose, making sure users are aware of this implicit transformation.

    The mask operand masks out memory access so that it is safe to pass out-of-boundary
    addresses/offsets as long as they are masked. It applies to slots of SIMD lanes.

  Example 1:
  ```mlir
    %2 = xegpu.load %1, %0 {l1_hint = #xegpu.cache_hint<cached>,
                            l2_hint = #xegpu.cache_hint<uncached>,
                            l3_hint = #xegpu.cache_hint<uncached>}
          : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_space=global>>,
            vector<16xi1> -> vector<16xf32>
  ```

  Example 2:
  ```mlir
    %2 = xegpu.load %1, %0 {transpose,
                            l1_hint = #xegpu.cache_hint<cached>,
                            l2_hint = #xegpu.cache_hint<uncached>,
                            l3_hint = #xegpu.cache_hint<uncached>}
          : !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<memory_space=global, chunk_size=8>>,
            vector<16xi1> -> vector<8x16xf32>
  ```

  }];

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       XeGPU_MaskType: $mask,
                       OptionalAttr<UnitAttr>: $transpose,
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

def XeGPU_StoreScatterOp : XeGPU_Op<"store", [
  AllElementTypesMatch<["value", "TensorDesc"]>, MemoryEffects<[MemWrite]>
  ]> {
  let summary = "store data to scattered memory locations.";
  let description = [{ It (aka. store) stores data to scattered memory locations. The value is
  typically a 1D vector. But when the chunk size of the TensorDesc is larger than 1, it will be
  a 2D vector instead. For the later case, dim-1 of the value correspods to the simd lanes
  and the dim-0 of the value corresponds to the chunk size stored per lane. So `store_scatter`
  has transpose effect, which is similar to `load_gather`. Therefore, a transpose attribute is
  introduced on purpose, making sure users are aware of this implicit transformation.

  Example 1:
  ```mlir
    %3 = xegpu.store %0, %1, %2 {l1_hint = #xegpu.cache_hint<uncached>,
                                 l2_hint = #xegpu.cache_hint<write_back>,
                                 l3_hint = #xegpu.cache_hint<write_through>}
          : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scattered_tdesc_attr<>>, vector<16xi1>
  ```

  Example 2:
  ```mlir
    %3 = xegpu.store %0, %1, %2 {transpose,
                                 l1_hint = #xegpu.cache_hint<uncached>,
                                 l2_hint = #xegpu.cache_hint<write_back>,
                                 l3_hint = #xegpu.cache_hint<write_through>}
          : vector<8x16xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scattered_tdesc_attr<chunk_size=8>>, vector<16xi1>
  ```

  }];

  let arguments = (ins
    XeGPU_ValueType: $value,
    XeGPU_TensorDesc: $TensorDesc,
    XeGPU_MaskType: $mask,
    OptionalAttr<UnitAttr>: $transpose,
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
    ```mlir
      %off = arith.constant dense<[32, 32, 32, 32]> : vector<4xindex>
      %2 = xegpu.update_offset %1, %off :
              !xegpu.tensor_desc<4x2xf32, #xegpu.scattered_tdesc_attr<>>, vector<4xindex>
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

    Note: on PVC, the hardware can perform load with VNNI transformation when data
          element type is 16-bit or lower precision, taking 2 or 4 elements from
          the first dimension and inserted into the newly added innermost dimension.
  }];

  let arguments = (ins
    XeGPU_DpasOpType : $lhs,
    XeGPU_DpasOpType : $rhs,
    Optional<XeGPU_Vector2DType>: $acc);
  let results = (outs XeGPU_Vector2DType: $result);

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
  }];

  let assemblyFormat = [{
    $lhs `,` $rhs (`,` $acc^)? attr-dict `:` type($lhs)`,` type($rhs) (`,` type($acc)^)?  `->` type($result)
  }];

  let hasVerifier = 1;
}

def XeGPU_AtomicRMWOp: XeGPU_Op<"atomic_rmw", [Pure,
      AllElementTypesMatch<["tensorDesc", "value", "result"]>,
      AllShapesMatch<["tensorDesc", "value", "result"]>]> {
  let summary = "Atomic ready-modify-write operation on the TensorDesc. ";

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

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD
