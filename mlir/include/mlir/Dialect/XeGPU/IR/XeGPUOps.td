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
            ::mlir::OpAsmPrinter &p, const Properties &prop) {
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
    dimention. It encodes the following important information for supporting 
    Intel hardware features:

    * source: an object representing (starting address/pointer of) a 2D memory region. 
        It can be either a 2D memref object, or simply a pointer represented by uint64_t type.
        for the later case, the shape and layout information of the 2D memory region should 
        be explicitly passed via `dynamic_shape` and `dynamic_strides` parameters.
    * offsets: two index values represents offsets from the "source" at the each dimension 
        at which the subview of the target memory will be created. It is encoded via two
        variables, including "dynamic_offsets" and "static_offsets", such that it can
        accept various forms, such as, operands (e.g., [%c0, %c]) and attributes (e.g., [2, 4])).
    * shape: the shape information of the memory region pointed by the "source".  It is 
        typically encoded via the MemRefType of the source, e.g., memref<4096x4096xf16>. 
        But if "source" is simply a pointer represented as uint64_t type, or a memref 
        type without shape information e.g., memref<?x?xf16>, the shape information has 
        to be explicitly passed via the "dynamic_shape" argument. Currently "dynamic_shape" 
        only accepts operands(e.g., [%c4096, %c4096]), not attributes(e.g., [4096, 4096]).
    * strides: the strides of the memory region pointed by the "source". Similar to shape, 
        it is typically encoded via the MemRefType of the source too. But if "source" is 
        simply a pointer represented as uint64_t type, or a memref type without shape 
        information e.g., memref<?x?xf16>, the strides information has to be explicitly 
        passed via the "dynamic_strides" argument. And it currently only accepts operands two.

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
  let summary = "prefetches a nD block to cache";
  let description = [{
    It issues an instruction to prefetch the data from memory to each 
    level of the cache based on their cache policy.

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
                       
  let extraClassDeclaration = extraBaseClassDeclaration;

  let assemblyFormat = "$TensorDesc prop-dict attr-dict `:` qualified(type($TensorDesc))";
}


def XeGPU_LoadNdOp : XeGPU_Op<"load_nd"> {
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

def XeGPU_StoreNdOp : XeGPU_Op<"store_nd", []> {
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

  let extraClassDeclaration = extraBaseClassDeclaration;

  let assemblyFormat = [{$value `,` $TensorDesc prop-dict attr-dict 
                        `:` type($value) `,` qualified(type($TensorDesc))}];
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD
