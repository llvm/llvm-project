//===- XeGPUOps.td - XeGPU dialect operations definition ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD
#define MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD

include "mlir/Dialect/XeGPU/IR/XeGPUAttrs.td"
include "mlir/Dialect/XeGPU/IR/XeGPUDialect.td"
include "mlir/Dialect/XeGPU/IR/XeGPUTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"


include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"


// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class XeGPU_Op<string mnemonic, list<Trait> traits = []>:
          Op<XeGPU_Dialect, mnemonic, traits>;


def XeGPU_CreateNdDescOp: XeGPU_Op<"create_nd_tdesc", [Pure, AttrSizedOperandSegments]> {

  let summary = "create nd tensor descriptor operation";
  let description = [{
    The "create_nd_tdesc" operation creates a TensorDescType which represents
    a sub-view of a 2D memory region (It can be extended to support N-D memory
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
    Variadic<Index>: $dynamic_offsets, 
    Variadic<Index>: $dynamic_shape, 
    Variadic<Index>: $dynamic_strides,
    DenseI64ArrayAttr: $static_offsets
  );
  let results = (outs XeGPU_TensorDesc: $TensorDesc);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($dynamic_offsets, $static_offsets)
    (`,` `[` $dynamic_shape^ `]` `,` `[` $dynamic_strides `]`)?
    attr-dict `:` type($source) `->` type($TensorDesc)
  }];
  let skipDefaultBuilders = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Type": $TensorDesc, "Value": $source, "ValueRange": $offsets, 
                   "ValueRange": $shape, "ValueRange": $strides, 
                   "llvm::ArrayRef<int64_t>": $static_offsets)>,

    OpBuilder<(ins "Type": $tdesc, "Value": $source, 
                   "llvm::ArrayRef<OpFoldResult>": $offsets)>,

    OpBuilder<(ins "Type": $tdesc, "Value": $source, 
                   "llvm::ArrayRef<OpFoldResult>": $offsets,
                   "ValueRange": $shape, "ValueRange": $stride)>
  ];

  let extraClassDeclaration = [{
    /// Returns the type of the source memref operand.
    Type getSourceType() {
      return getSource().getType();
    }

    /// Returns the type of the result TensorDesc.
    xegpu::TensorDescType getType() {
      return getTensorDesc().getType();
    }

    /// Returns the offsets info to the source. It consolidates
    /// information from both dynamic_offsets and static_offsets
    /// parameters. static_offsets parameter always has the expected
    /// ranks with some dim could have ShapeType::kDynamic value
    /// indicating the corresponding value should be from dynamic_offsets.
    llvm::SmallVector<OpFoldResult> getOffsets();

    /// returns the shape info of the source. It is either from the
    /// memref type, if source is a memref with static shape
    /// information or from the dynamic_shape parameter. If both
    /// exists, the dynamic_shape parameter will be used and the
    /// shape information from  memref type will be ignored.
    llvm::SmallVector<OpFoldResult> getShape();

    /// returns the strides info of the source. It is either from the
    /// memref type, if source is a memref with static shape
    /// information or from the dynamic_stride parameter. If both
    /// exists, the dynamic_strides parameter will be used and the
    /// strides information from  memref type will be ignored.
    llvm::SmallVector<OpFoldResult> getStrides();

    /// Return the element type of the TensorDesc
    Type getElementType() {
      return getType().getElementType();
    }

    /// Return the shape of the TensorDesc
    llvm::ArrayRef<int64_t> getTensorDescShape() {
      return getType().getShape();
    }
  }];
}

def XeGPU_PrefetchNdOp : XeGPU_Op<"prefetch_nd", []> {
  let summary = "prefetches a nD block to cache";
  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);

  // Format: xegpu.prefetch_nd %tdesc {l1_hint = #xegpu.cache_hint<cached>, 
  //                                   l2_hint = #xegpu.cache_hint<cached>, 
  //                                   l3_hint = #xegpu.cache_hint<cached>}
  //         : !xegpu.tensor_desc<8x16xf16>
  let assemblyFormat = "$TensorDesc attr-dict `:` type($TensorDesc)";
}


def XeGPU_LoadNDOp : XeGPU_Op<"load_nd"> {
  let summary = "loads a n-D block from memory (represented by TensorDesc)" 
                "to registers (represented by vector)";
  let description = [{
    LoadNDOp essentially mimics the hardware block read instruction to read 
    a block of data from memory to register. It takes a set of cache hints 
    for each level of cache, L1, L2 and L3. If hardware does not have a 
    correspoding cache, Corresponding cache hint attribute will be masked.
    If both transpose and vnni_axis present at the same time. It assume to 
    perform transpose first and then vnni transform.
  }];

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       OptionalAttr<I64Attr>: $vnni_axis,
                       OptionalAttr<DenseI64ArrayAttr>: $transpose,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);

  let results = (outs XeGPU_ValueType: $value);

  let extraClassDeclaration = [{
    VectorType getType() {
      return llvm::dyn_cast<VectorType>(getValue().getType());
    }

    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }
  }];

  // Format: xegpu.load_nd %1 {transpose = [1, 0], 
  //                l1_hint = #xegpu.cache_hint<cached>, 
  //                l2_hint = #xegpu.cache_hint<uncached>, 
  //                l3_hint = #xegpu.cache_hint<streaming>}
  //         : !xegpu.tensor_desc<8x16xf32> -> vector<16x8xf32>
  let assemblyFormat = "$TensorDesc attr-dict `:` type($TensorDesc) `->` type($value)";
  let hasVerifier = 1;
}

def XeGPU_StoreNDOp : XeGPU_Op<"store_nd", []> {
  let summary = "stores a n-D block register region back to memory, currently only supports 2D";
  let arguments = (ins XeGPU_ValueType: $value,
                       XeGPU_TensorDesc: $TensorDesc,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheHintAttr>: $l3_hint);

  // Format: xegpu.store_nd %3, %2 {l1_hint = #xegpu.cache_hint<uncached>,
  //                                l2_hint = #xegpu.cache_hint<write_back>, 
  //                                l3_hint = #xegpu.cache_hint<write_through>}
  //                                : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  let assemblyFormat = "$value `,` $TensorDesc attr-dict `:` type($value) `,` type($TensorDesc)";
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD
