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

    * source: an object representing (starting address/pointer of) a 2D memory reagion. 
        It can be either a 2D memref object, or simply a pointer represented by uint64_t type.
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
    %0 = memref.alloc() : memref<32x24xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c1]: memref<32x24xf32> -> TensorDesc<8x16xf32>

    Example 2 (suppose the tensor shape inferred by the compiler is 8x16):
    %0 = memref.alloc(%h, %w) : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c1], [%h, %w], [%w, %c1]: memref<?x?xf32> -> TensorDesc<8x16xf32>

    Example 3 (suppose the tensor shape inferred by the compiler is 8x16):
    %0 = ... : ui64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c1], [%h, %w], [%w, %c1]: ui64 -> TensorDesc<8x16xf32>
  }];

  let arguments = (ins XeGPU_BaseAddrType: $source, 
                 Variadic<Index>: $dynamic_offsets, 
                 Variadic<Index>: $dynamic_shape, 
                 Variadic<Index>: $dynamic_strides,
                 DenseI64ArrayAttr: $static_offsets,
                 DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode);
  let results = (outs XeGPU_TensorDesc:$TensorDesc);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Type": $TensorDesc, "Value": $source, "ValueRange": $offsets, 
                   "ValueRange": $shape, "ValueRange": $strides, 
                   "llvm::ArrayRef<int64_t>": $static_offsets,
                    CArg<"xegpu::ModeKind", "xegpu::ModeKind::SIMT">: $mode)>,

    OpBuilder<(ins "Type": $tdesc, "Value": $source, 
                   "llvm::ArrayRef<OpFoldResult>": $offsets,
                   CArg<"xegpu::ModeKind", "xegpu::ModeKind::SIMT">: $mode)>,

    OpBuilder<(ins "Type": $tdesc, "Value": $source, 
                   "llvm::ArrayRef<OpFoldResult>": $offsets,
                   "ValueRange": $shape, "ValueRange": $stride,
                   CArg<"xegpu::ModeKind", "xegpu::ModeKind::SIMT">: $mode)>
  ];

  let extraClassDeclaration = [{
    /// Returns the type of the source memref operand.
    Type getSourceType() {
      return getSource().getType();
    }

    /// Returns the type of the result TensorDesc.
    xegpu::TensorDescType getTensorDescType();

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

    /// return the shape embeded in the memref type of the source.
    /// If source is not memref type. array of kDynamic will be returned.
    llvm::ArrayRef<int64_t> getStaticShape();

    /// return the strides embeded in the memref type of the source.
    /// If source is not memref type. array of kDynamic will be returned.
    llvm::ArrayRef<int64_t> getStaticStrides();

    /// Return the element type of the TensorDesc
    Type getElementType();

    /// Return the shape of the TensorDesc
    llvm::ArrayRef<int64_t> getTensorDescShape();
  }];

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
                       OptionalAttr<XeGPU_CacheAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheAttr>: $l3_hint,
                       OptionalAttr<DenseI64ArrayAttr>: $transpose,
                       DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode);
  let results = (outs XeGPU_ValueType: $value);

  let extraClassDeclaration = [{
    VectorType getValueType() {
      return llvm::dyn_cast<VectorType>(getValue().getType());
    }

    xegpu::TensorDescType getTensorDescType() {
      return getTensorDesc().getType();
    }
  }];

  // Format: xegpu.load_nd %1 {transpose = [1, 0], l1_hint = cached, l2_hint = uncached, l3_hint=streaming}
  //                          : !xegpu.tensor_desc<8x16xf32> -> vector<16x8xf32>
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_StoreNDOp : XeGPU_Op<"store_nd", []> {
  let summary = "stores a n-D block register region back to memory, currently only supports 2D";
  let arguments = (ins XeGPU_ValueType: $value,
                       XeGPU_TensorDesc: $TensorDesc,
                       OptionalAttr<XeGPU_CacheAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheAttr>: $l3_hint,
                       DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode);

  // Format: xegpu.store_nd %3, %2 {l1_hint = write_back, l2_hint = uncached}
  //                        : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_PrefetchNDOp : XeGPU_Op<"prefetch_nd", []> {
  let summary = "prefetches a nD block to cache";
  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       OptionalAttr<XeGPU_CacheAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheAttr>: $l3_hint,
                       DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode);

  // Format: xegpu.prefetch_nd %tdesc {l1_hint = cached, l2_hint = uncached}:
  //                                    !xegpu.tensor_desc<8x16xf16>
  let hasCustomAssemblyFormat = 1;
}

def XeGPU_UpdateNDOffsetOp : XeGPU_Op<"update_nd_offset", []> {
  let summary = "update the offsets for the given tensor descriptor";

  let arguments = (ins
    XeGPU_TensorDesc: $TensorDesc,
    Variadic<Index>: $offsets,
    DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode);

  let results = (outs XeGPU_TensorDesc: $result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_CreateDescOp: XeGPU_Op<"create_tdesc", [Pure]> {
  let summary = "create scattered tensor descriptors (TensorDesc).";
  let description = [{
    "create_tdesc" is similar to "create_nd_tdesc" in terms that it creates 
    a Tensor Descriptor (TensorDescType) for a memory region. While "create_nd_tdesc" 
    is for creating continious subviews, "create_tdesc" is for creating non-continious
    (scattered) subviews. It is designed only works with VectorCompute (VC) mode and 
    accepts the following parameters:

    * source: a 1D memref or pointer (uint64_t) represents the memory object.
    * offsets: It is a 1D vector containing offsets of each access point, the supportted 
          group size, e.g., vector<16xindex>. And each element in the vector corresponds 
          to a work item (SIMT lane) in the subgroup.
    * chunk_size_per_lane: [optional attribute] indicates number of continious elements 
          accessed for each offset, default is 1.

    Example 1. It assumes subgroup size is 4, and accesses a[0], a[16], a[32], a[64]
    %a = memref.alloc() : memref<1024xf32>
    %c0 = arith.constant dense<0, 16, 32, 64> : vector<4xindex>
    %1 = xegpu.create_tdesc %a, %c0: memref<1024xf32> -> TensorDesc<4xf32>

    Example 2. It assumes subgroup size is 4, and each workitem access 8 elements.
               It will access totally 32 data elements: a[0:7], a[16:23], a[32:39], a[64:71]
    %0 = memref.alloc() : memref<1024xf32>
    %c0 = arith.constant dense<0, 16, 32, 64> : vector<4xindex>
    %1 = xegpu.create_tdesc %0, %c0 {chunk_size_per_lane = 8}: memref<1024xf32> -> TensorDesc<4x8xf32>
  }];

  let arguments = (ins XeGPU_BaseAddrType: $source,
                       XeGPU_OffsetType: $offsets,
                       DefaultValuedAttr<I64Attr, "1">: $chunk_size_per_lane,
                       DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode);
  let results = (outs XeGPU_TensorDesc:$TensorDesc);

  let builders = [
    OpBuilder<(ins "xegpu::TensorDescType": $TensorDesc, "Value": $source,
                  "Value": $offsets, CArg<"uint32_t", "1"> : $chunk_size_per_lane)>,
    OpBuilder<(ins "xegpu::TensorDescType": $TensorDesc, "Value": $source,
                  "Value": $offsets, "IntegerAttr": $chunk_size_per_lane)>
  ];
  let skipDefaultBuilders = 1;

  // Format: xegpu.create_tdesc %src, %offsets {mode=simt, chunk_size_per_lane=1}
  //              : ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_LoadGatherOp : XeGPU_Op<"load"> {
  let summary = "load a scalar at source[offset].";

  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       XeGPU_MaskType: $mask,
                       OptionalAttr<I64Attr>: $vnni_axis,
                       OptionalAttr<DenseI64ArrayAttr>: $transpose,
                       OptionalAttr<XeGPU_CacheAttr>: $l1_hint,
                       OptionalAttr<XeGPU_CacheAttr>: $l2_hint,
                       OptionalAttr<XeGPU_CacheAttr>: $l3_hint,
                       DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode);
  let results = (outs XeGPU_ValueType: $value);

  let builders = [
    OpBuilder<(ins "mlir::Type": $value, "mlir::Value": $TensorDesc, 
                   "mlir::Value": $mask, "mlir::IntegerAttr": $vnni_axis,
                   CArg<"mlir::DenseI64ArrayAttr", "mlir::DenseI64ArrayAttr()">: $transpose,
                   CArg<"xegpu::CacheKindAttr", "xegpu::CacheKindAttr()">: $l1_hint,
                   CArg<"xegpu::CacheKindAttr", "xegpu::CacheKindAttr()">: $l2_hint,
                   CArg<"xegpu::CacheKindAttr", "xegpu::CacheKindAttr()">: $l3_hint)>,

    OpBuilder<(ins "mlir::Type": $value, "mlir::Value": $TensorDesc, 
                   "mlir::Value": $mask, "mlir::IntegerAttr": $vnni_axis,
                   CArg<"DenseI64ArrayAttr", "DenseI64ArrayAttr()">: $transpose,
                   CArg<"xegpu::CacheKind", "xegpu::CacheKind::CACHED">: $l1_hint,
                   CArg<"xegpu::CacheKind", "xegpu::CacheKind::CACHED">: $l2_hint,
                   CArg<"xegpu::CacheKind", "xegpu::CacheKind::CACHED">: $l3_hint)>
  ];
  let skipDefaultBuilders = 1;

  // Format: %2 = xegpu.load %1, %0 {transpose = [1, 0], l1_hint = cached, l2_hint = uncached}
  //                 : !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>, vector<16x8xi1> -> vector<8x16xf32>
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_StoreScatterOp : XeGPU_Op<"store", []> {
  let summary = "store a scalar to source[offset].";

  let arguments = (ins
    XeGPU_ValueType: $value,
    XeGPU_TensorDesc: $TensorDesc,
    XeGPU_MaskType: $mask,
    OptionalAttr<XeGPU_CacheAttr>: $l1_hint,
    OptionalAttr<XeGPU_CacheAttr>: $l2_hint,
    OptionalAttr<XeGPU_CacheAttr>: $l3_hint,
    DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode
  );

  let builders = [
    OpBuilder<(ins "Value": $value, "Value": $TensorDesc, "Value": $mask,
        CArg<"xegpu::CacheKindAttr", "xegpu::CacheKindAttr()">: $l1_hint,
        CArg<"xegpu::CacheKindAttr", "xegpu::CacheKindAttr()">: $l2_hint,
        CArg<"xegpu::CacheKindAttr", "xegpu::CacheKindAttr()">: $l3_hint)>,
    OpBuilder<(ins "Value": $value, "Value": $TensorDesc, "Value": $mask,
        CArg<"xegpu::CacheKind", "xegpu::CacheKind::WRITE_BACK">: $l1_hint,
        CArg<"xegpu::CacheKind", "xegpu::CacheKind::WRITE_BACK">: $l2_hint,
        CArg<"xegpu::CacheKind", "xegpu::CacheKind::WRITE_BACK">: $l3_hint)>
  ];
  let skipDefaultBuilders = 1;

  // Format: %3 = xegpu.load %1, %0 {l1_hint = cached, l2_hint = uncached}
  //                      : !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_PrefetchOp : XeGPU_Op<"prefetch", []> {
  let summary = "prefetches a nD block to cache";
  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
         OptionalAttr<XeGPU_CacheAttr>: $l1_hint,
         OptionalAttr<XeGPU_CacheAttr>: $l2_hint,
         OptionalAttr<XeGPU_CacheAttr>: $l3_hint,
         DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode);

  let builders = [
    OpBuilder<(ins "Value": $TensorDesc,
      CArg<"xegpu::CacheKindAttr", "xegpu::CacheKindAttr()">: $l1_hint,
      CArg<"xegpu::CacheKindAttr", "xegpu::CacheKindAttr()">: $l2_hint,
      CArg<"xegpu::CacheKindAttr", "xegpu::CacheKindAttr()">: $l3_hint)>,
    OpBuilder<(ins "Value": $TensorDesc,
      CArg<"xegpu::CacheKind", "xegpu::CacheKind::CACHED">: $l1_hint,
      CArg<"xegpu::CacheKind", "xegpu::CacheKind::CACHED">: $l2_hint,
      CArg<"xegpu::CacheKind", "xegpu::CacheKind::CACHED">: $l3_hint)>
  ];

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;

  // Format: xegpu.prefetch %tdesc {l1_hint = cached, l2_hint = uncached}:
  //                                    !xegpu.tensor_desc<8x16xf16>
  let hasCustomAssemblyFormat = 1;
}

def XeGPU_UpdateOffsetOp: XeGPU_Op<"update_offset", []> {
  let summary = "update the offsets for the given tensor descriptor";
  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
                       XeGPU_OffsetType: $offsets,
                       DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode);
  let results = (outs XeGPU_TensorDesc: $result);

  let builders = [
    OpBuilder<(ins "Type": $result, "Value": $TensorDesc, "Value": $offsets)>
  ];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_DpasOp : XeGPU_Op<"dpas"> {
  let summary = "performs dpas computation";
  let arguments = (ins
    XeGPU_DpasOpType : $lhs,
    XeGPU_DpasOpType : $rhs,
    Optional<XeGPU_Vector2DType>: $acc,
    DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode
  );
  let results = (outs XeGPU_Vector2DType: $result);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    VectorType getLhsType() {
      return ::llvm::cast<VectorType>(getLhs().getType());
    }

    VectorType getRhsType() {
      return ::llvm::cast<VectorType>(getRhs().getType());
    }

    VectorType getAccType() {
      return ::llvm::cast<VectorType>(getAcc().getType());
    }

    VectorType getResultType() { 
      return getResult().getType(); 
    }
  }];

  let hasVerifier = 1;
}

def XeGPU_InvokeSIMDOp : XeGPU_Op<"invoke_SIMD", []> {
    let summary = "Invoke_SIMD operation";
    let description = [{
      The `xegpu.invoke_SIMD` operation works similar to a direct call to a function. 
      But it is special to Intel GPU.
    }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyType>:$operands,
                       XeGPU_ArgTypeAttr: $argType);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      "xegpu::ArgTypeKindAttr":$argType, CArg<"ValueRange", "{}">:$operands)>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      "xegpu::ArgTypeKindAttr":$argType, CArg<"ValueRange", "{}">:$operands)>,
    OpBuilder<(ins "llvm::StringRef":$callee, "TypeRange":$results,
      "xegpu::ArgTypeKindAttr":$argType, CArg<"ValueRange", "{}">:$operands)>
  ];
}

def XeGPU_AtomicRMWOp: XeGPU_Op<"atomic_rmw", []> {
  let summary = "perform ready-modify-write operation that is free from data races.";
  let arguments = (ins
    XeGPU_AtomicRMWKindAttr:$kind,
    XeGPU_TensorDesc:$tensorDesc,
    XeGPU_MaskType:$mask,
    Optional<XeGPU_ValueType>:$value,
    DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode
  );

  let results = (outs XeGPU_ValueType:$result);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Type": $result, "xegpu::AtomicRMWKindAttr": $kind,
                   "Value": $tensorDesc, "Value": $mask, "Value": $value)>,
    OpBuilder<(ins "Type": $result, "xegpu::AtomicRMWKind": $kind,
                   "Value": $tensorDesc, "Value": $mask, "Value": $value)>
  ];

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
}

def XeGPU_AllocNbarrierOp: XeGPU_Op<"alloc_nbarrier", []> {
  let summary = "allocate a specific number of named barriers.";
  let arguments = (ins I64Attr: $nbarrierCount);
  let assemblyFormat = "$nbarrierCount attr-dict";
}


def XeGPU_CreateNbarrierOp: XeGPU_Op<"create_nbarrier", []> {
  let summary = "create a named barrier.";
  let arguments = (ins I8: $nbarrier_id,
          I8: $nbarrier_role,
          I8Attr: $num_producers,
          I8Attr: $num_consumers,
          DefaultValuedAttr<XeGPU_ModeAttr, "xegpu::ModeKind::SIMT">: $mode);
  let results = (outs XeGPU_Nbarrier: $result);
  let hasCustomAssemblyFormat = 1;
}

def XeGPU_NbarrierArriveOp: XeGPU_Op<"nbarrier_arrive", []> {
  let summary = "arrive at a named barrier.";
  let arguments = (ins XeGPU_Nbarrier: $payload);
  let assemblyFormat = [{ $payload attr-dict `:` qualified(type($payload))}];
}

def XeGPU_NbarrierWaitOp: XeGPU_Op<"nbarrier_wait", []> {
  let summary = "wait for a named barrier.";
  let arguments = (ins XeGPU_Nbarrier: $payload);
  let assemblyFormat = [{ $payload attr-dict `:` qualified(type($payload)) }];
}

def XeGPU_CompileHintOp: XeGPU_Op<"compile_hint", []> {
  let summary = "prevents the compiler from scheduling.";
  let assemblyFormat = [{ attr-dict }];
}

def XeGPU_MfenceOp: XeGPU_Op<"mfence", []> {
  let summary = "lsc fence.";
  let arguments = (ins StrAttr: $memory_kind,
                       StrAttr: $fence_op,
                       StrAttr: $fence_scope);
  let assemblyFormat = [{ attr-dict }];
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUOPS_TD
