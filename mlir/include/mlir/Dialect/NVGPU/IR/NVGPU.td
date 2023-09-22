//===-- NVGPU.td - NVGPU dialect operation definitions *- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the NVGPU dialect.
//
// This NVGPU provides a bridge between the target agnostic GPU and Vector
// dialects and lower level NVVM dialect. This allow representing PTX specific
// operations while using MLIR high level concepts like memref and 2-D vector.
//
// Ops semantic are going to be based on vendor specific PTX defintion:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
//
//===----------------------------------------------------------------------===//

#ifndef NVGPU
#define NVGPU

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def NVGPU_Dialect : Dialect {
  let name = "nvgpu";
  let cppNamespace = "::mlir::nvgpu";
  let description = [{
    The `NVGPU` dialect provides a bridge between higher-level target-agnostic
    dialects (GPU and Vector) and the lower-level target-specific dialect
    (LLVM IR based NVVM dialect) for NVIDIA GPUs. This allow representing PTX
    specific operations while using MLIR high level dialects such as Memref
    and Vector for memory and target-specific register operands, respectively.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  
  let extraClassDeclaration = [{
    /// Return true if the given MemRefType has an integer address
    /// space that matches the NVVM shared memory address space or
    /// is a gpu::AddressSpaceAttr attribute with value 'workgroup`.
    static bool hasSharedMemoryAddressSpace(MemRefType type);

    /// Return true if the given Attribute has an integer address
    /// space that matches the NVVM shared memory address space or
    /// is a gpu::AddressSpaceAttr attribute with value 'workgroup`.
    static bool isSharedMemoryAddressSpace(Attribute type);

    /// Defines the MemRef memory space attribute numeric value that indicates
    /// a memref is located in global memory. This should correspond to the
    /// value used in NVVM.
    static constexpr unsigned kGlobaldMemoryAddressSpace = 1;

    /// Defines the MemRef memory space attribute numeric value that indicates
    /// a memref is located in shared memory. This should correspond to the
    /// value used in NVVM.
    static constexpr unsigned kSharedMemoryAddressSpace = 3;
  }];
}

//===----------------------------------------------------------------------===//
// NVGPU Attribute Definitions
//===----------------------------------------------------------------------===//

def TensorMapSwizzleNone : I32EnumAttrCase<"SWIZZLE_NONE", 0, "none">;
def TensorMapSwizzle32B  : I32EnumAttrCase<"SWIZZLE_32B", 1, "swizzle_32b">;
def TensorMapSwizzle64B  : I32EnumAttrCase<"SWIZZLE_64B", 2, "swizzle_64b">;
def TensorMapSwizzle128B : I32EnumAttrCase<"SWIZZLE_128B", 3, "swizzle_128b">;
def TensorMapSwizzleKind : I32EnumAttr<"TensorMapSwizzleKind", 
                                "Tensor map swizzling mode of shared memory banks",
  [ TensorMapSwizzleNone, TensorMapSwizzle32B, TensorMapSwizzle64B, 
    TensorMapSwizzle128B]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::nvgpu";
}

def TensorMapL2PromoNone : I32EnumAttrCase<"L2PROMO_NONE", 0, "none">;
def TensorMapL2Promo64B  : I32EnumAttrCase<"L2PROMO_64B", 1, "l2promo_64b">;
def TensorMapL2Promo128B : I32EnumAttrCase<"L2PROMO_128B", 2, "l2promo_128b">;
def TensorMapL2Promo256B : I32EnumAttrCase<"L2PROMO_256B", 3, "l2promo_256b">;
def TensorMapL2PromoKind : I32EnumAttr<"TensorMapL2PromoKind", 
                                "Tensor map L2 promotion type",
  [ TensorMapL2PromoNone, TensorMapL2Promo64B, TensorMapL2Promo128B, 
    TensorMapL2Promo256B]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::nvgpu";
}

def TensorMapOOBZero : I32EnumAttrCase<"OOB_ZERO", 0, "zero">;
def TensorMapOOBNaN  : I32EnumAttrCase<"OOB_NAN", 1, "nan">;
def TensorMapOOBKind : I32EnumAttr<"TensorMapOOBKind", 
                                "Tensor map out-of-bounds fill type",
  [ TensorMapOOBZero, TensorMapOOBNaN]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::nvgpu";
}

def TensorMapInterleaveNone : I32EnumAttrCase<"INTERLEAVE_NONE", 0, "none">;
def TensorMapInterleave16B  : I32EnumAttrCase<"INTERLEAVE_16B", 1, "interleave_16b">;
def TensorMapInterleave32B  : I32EnumAttrCase<"INTERLEAVE_32B", 2, "interleave_32b">;
def TensorMapInterleaveKind : I32EnumAttr<"TensorMapInterleaveKind", 
                                "Tensor map interleave layout type",
  [ TensorMapInterleaveNone, TensorMapInterleave16B, TensorMapInterleave32B]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::nvgpu";
}

def TensorMapSwizzleAttr : EnumAttr<NVGPU_Dialect, TensorMapSwizzleKind, "swizzle">;
def TensorMapL2PromoAttr : EnumAttr<NVGPU_Dialect, TensorMapL2PromoKind, "l2promo">;
def TensorMapOOBAttr : EnumAttr<NVGPU_Dialect, TensorMapOOBKind, "oob">;
def TensorMapInterleaveAttr : EnumAttr<NVGPU_Dialect, TensorMapInterleaveKind, "interleave">;

//===----------------------------------------------------------------------===//
// NVGPU Type Definitions
//===----------------------------------------------------------------------===//

class NVGPU_Type<string name, string typeMnemonic,
        list<Trait> traits = []> : TypeDef<NVGPU_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def NVGPU_DeviceAsyncToken : NVGPU_Type<"DeviceAsyncToken",
                                        "device.async.token", []> {
  let summary = "device async token type";
  let description = [{
    `nvgpu.device.async.token` is a type returned by an asynchronous operation
    that runs on the GPU (device). It is used to establish an SSA-based link
    between the async operation (e.g. DeviceAsyncCopy) and operations that
    group or synchronize the async operations (e.g. DeviceAsyncCreateGroupOp,
    DeviceAsyncWaitOp).
  }];
}

def NVGPU_MBarrierGroup : NVGPU_Type<"MBarrierGroup", "mbarrier.group", []> {
  let summary = "mbarrier barrier type";
  let description = [{
    This is the type for one or more mbarrier object in shared memory that is 
    used to synchronize a variable number of threads.

    If `num_barriers` is not set, the number of mbarrier objects is 1.

    A mbarrier object is 64 bit with 8 byte alignment. The mbarrier object 
    can be initiated and invalidated.

    [See for more details in PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#size-and-alignment-of-mbarrier-object)
  }];    
  let parameters = (ins "Attribute":$memorySpace, DefaultValuedParameter<"unsigned", "1">:$num_barriers);
  let assemblyFormat = "`<` struct(params) `>`";
  let builders = [
    TypeBuilder<(ins "Attribute":$memorySpace), [{
      return $_get($_ctxt, memorySpace, 1);
    }]>
  ];
}

def NVGPU_MBarrierToken : NVGPU_Type<"MBarrierToken", "mbarrier.token", []> { }

// https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-map
def NVGPU_TensorMapDescriptor : NVGPU_Type<"TensorMapDescriptor", "tensormap.descriptor", []> {
  let summary = "TensorMap descriptor";
  let parameters = (ins "MemRefType":$tensor,
                        EnumParameter<TensorMapSwizzleKind>:$swizzle,
                        EnumParameter<TensorMapL2PromoKind>:$l2promo,
                        EnumParameter<TensorMapOOBKind>:$oob,
                        EnumParameter<TensorMapInterleaveKind>:$interleave);
  let description = [{
    `nvgpu.tma.descriptor` is a type that represents a TMA descriptor. It is 
    128-byte object either in constant space or kernel paramater.    
  }];
  let assemblyFormat = "`<` struct(params) `>`";
}

def NVGPU_WarpgroupMatrixDescriptor : NVGPU_Type<"WarpgroupMatrixDescriptor", "wgmma.descriptor", []> {
  let summary = "Warpgroup matrix descriptor type";
  let description = [{
  The descriptor specifies the properties of the matrix in shared memory that 
  is a multiplicand in the matrix multiply and accumulate operation. 
  
  The descriptor is a 64-bit value contained in a register with the following:
  ```
  +---------+-----+-----------+-----+-----------+-----+-----+-----------+-----+
  |   0-13  |14-15|   16-29   |30-31|   32-45   |46-48|49-51|   52-61   |62-63|
  +---------+-----+-----------+-----+-----------+-----+-----+-----------+-----+
  |  14bits |2bits|   14bits  |2bits|   14bits  |2bits|3bits|   10bits  |2bits|
  +---------+-----+-----------+-----+-----------+-----+-----+-----------+-----+
  | BaseAddr|  0  | LeadingDim|  0  |   Stride  |  0  |Offst|     0     |Swzle|
  +---------+-----+-----------+-----+-----------+-----+-----+-----------+-----+
  ```
   
  [See for more details in PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor) 
  
  }];  
  let parameters = (ins "MemRefType":$tensor);
  let assemblyFormat = "`<` struct(params) `>`";
}

def NVGPU_WarpgroupAccumulator : NVGPU_Type<"WarpgroupAccumulator", "warpgroup.accumulator", []> {
  let parameters = (ins "VectorType":$fragmented);
  let assemblyFormat = "`<` struct(params) `>`";
  let description = [{
    This type represents the result matrix obtained from `nvgpu.warpgroup.mma`. 
    The `$fragmented` type signifies the distributed or fragmented result 
    vector that is collectively owned by all the threads in the warp-group 
    that executed `nvgpu.warpgroup.mma`.
    [See the details of register fragment layout for accumulator matrix D]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d) 
  }];
}

//===----------------------------------------------------------------------===//
// NVGPU Op Definitions
//===----------------------------------------------------------------------===//

class NVGPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<NVGPU_Dialect, mnemonic, traits> {}

def NVGPU_LdMatrixOp : NVGPU_Op<"ldmatrix", [
                                MemoryEffects<[MemRead]>,
                                PredOpTrait<"srcMemref and res have same element type",
                                            TCresVTEtIsSameAsOp<0, 0>>]> {
  let description = [{
    The `nvgpu.ldmatrix` op represents loading a matrix fragment from
    memory to registers. The source and result type must be compatible
    with lowering to the `nvvm.ldmatrix` instruction. This op represents
    the distributed version of a `vector.transfer_read` as an intermediate
    step between lowering from `vector.transfer_read` to `nvvm.ldmatrix`.

    This operation is meant to follow the semantic of described here:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix

    Example:
    ```mlir
    %0 = nvgpu.ldmatrix %sm[%c0, %c0] {numTiles = 4 : i32, transpose = false} :
      memref<?x?xf16, 3> -> vector<4x2xf16>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$srcMemref,
                           Variadic<Index>:$indices, BoolAttr:$transpose,
                           I32Attr:$numTiles);
  let results = (outs AnyVector:$res);
  let assemblyFormat = [{
    $srcMemref`[` $indices `]` attr-dict `:` type($srcMemref) `->` type($res)
  }];

  let hasVerifier = 1;
}

class NVGPU_MmaSyncOp<string mnemonic> :
        NVGPU_Op<mnemonic,  [Pure,
                             PredOpTrait<"matrixA and matrixB have same element type",
                                         TCopVTEtIsSameAs<0, 1>>]> {
  code extraBaseClassDeclaration = [{
    std::array<int64_t, 3> getMmaShapeAsArray() {
      ArrayAttr mmaShape = this->getMmaShape();
      assert(mmaShape.size() == 3 && "mmaShape should be three integers");
      return {::llvm::cast<IntegerAttr>(mmaShape[0]).getInt(),
              ::llvm::cast<IntegerAttr>(mmaShape[1]).getInt(),
              ::llvm::cast<IntegerAttr>(mmaShape[2]).getInt()};
    }
  }];

  let hasVerifier = 1;
}

def NVGPU_MmaSyncOp : NVGPU_MmaSyncOp<"mma.sync"> {
  let description = [{
    The `nvgpu.mma.sync` op represents the warp-level matrix-multiply-and-
    accumulate (mma) operation that is compatible with `nvvm.mma.sync`.
    The operands and results vector sizes are thread-level onwership to
    the warp-level mma operation shape. `mmaShape` attribute holds the
    warp-level matrix-multiply shape.

    The `nvgpu.mma.sync` op serves as an intermediate point between lowering from
    `vector.contract` to `nvvm.mma.sync`.

    This operation is meant to follow the semantic of described here:
      https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma

    Example:

    ```mlir
    %res = nvgpu.mma.sync (%matrixA, %matrixB, %matrixC) {mmaShape = [16, 8, 16]} :
        (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
    ```
  }];
  let arguments = (ins AnyVector:$matrixA,
                       AnyVector:$matrixB,
                       AnyVector:$matrixC,
                       I64ArrayAttr:$mmaShape,
                       OptionalAttr<UnitAttr>:$tf32Enabled);

  let results = (outs AnyVector:$res);

  let builders = [
    OpBuilder<(ins "Value":$matrixA,
                   "Value":$matrixB,
                   "Value":$matrixC,
                   "ArrayAttr":$mmaShape)>,
    OpBuilder<(ins "Value":$matrixA,
                   "Value":$matrixB,
                   "Value":$matrixC,
                   "ArrayRef<int64_t>":$mmaShape,
                   CArg<"bool", "false">:$tf32Enabled)>
  ];

  let assemblyFormat = [{
    `(` $matrixA`,` $matrixB`,` $matrixC `)` attr-dict
    `:` `(` type($matrixA) `,` type($matrixB) `,` type($matrixC) `)` `->` type($res)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def NVGPU_MmaSparseSyncMetadataType : FixedVectorOfLengthAndType<[2], [I16]>,
                        BuildableType<"::mlir::VectorType::get("
                          "{2},$_builder.getI16Type())">;

def NVGPU_MmaSparseSyncOp : NVGPU_MmaSyncOp<"mma.sp.sync"> {
  let description = [{
  The `nvgu.mma.sp.sync` operation performs a warp-distributed MMA operation
  where operand A is "structured sparse". In this case, the `matrixA` operand
  represents the (warp-distributed) non-zero values of operand A, and the
  `sparse_metadata` operand provides the indices.

  The full description of the sparsity storage format and distribution scheme is
  described in the PTX docs. This operation is meant to follow the semantic
  described in the PTX documentation here:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma

  The way the indices are distributed among the threads in a warp is controlled
  by the optional `sparsity_selector` operand, which is `0` by default. For
  more information, please consult the PTX documentation linked above.

  Example (targetingthe f16 16x8x32 `mma.sp` PTX instruction):

  ```mlir
  nvgpu.mma.sp.sync (%a, %b, %c) metadata (%meta) {mmaShape = [16, 8, 32]} :
    (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  ```
  }];

  let arguments = (ins AnyVector:$matrixA,
                       AnyVector:$matrixB,
                       AnyVector:$matrixC,
                       NVGPU_MmaSparseSyncMetadataType:$sparseMetadata,
                       I64ArrayAttr:$mmaShape,
                       DefaultValuedAttr<I32Attr, "0">:$sparsitySelector,
                       OptionalAttr<UnitAttr>:$tf32Enabled
                       );

  let results = (outs AnyVector:$res);

  let builders = [
    OpBuilder<(ins "Value":$matrixA,
                   "Value":$matrixB,
                   "Value":$matrixC,
                   "Value":$sparseMetadata,
                   "ArrayRef<int64_t>":$mmaShape)>
  ];

  let assemblyFormat = [{
    `(` $matrixA`,` $matrixB`,` $matrixC `)` `metadata` `(` $sparseMetadata `)` attr-dict
    `:` `(` type($matrixA) `,` type($matrixB) `,` type($matrixC) `)` `->` type($res)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def NVGPU_DeviceAsyncCopyOp : NVGPU_Op<"device_async_copy", [
                                       AttrSizedOperandSegments]> {
  let summary = "device-side asynchronous copy";
  let description = [{
    The `nvgpu.device_async_copy` op initiates an asynchronous copy operation of
    elements from source (global memory) to the destination (shared memory)
    without blocking the thread. The async copy is added to a group.

    This op is meant to be used with `nvgpu.device_async_create_group` and
    `nvgpu.device_async_wait` to synchronize copies as explained in those ops
    descriptions.

    `bypassL1` attribute is hint to the hardware to bypass the L1 cache during
    async copy, this hint may be ignored by the hardware.

    `dstElements` attribute is the total number of elements written to
    destination (shared memory).

    `srcElements` argument is the total number of elements read from
    source (global memory).

    `srcElements` is an optional argument and when present the op only reads
    `srcElements` number of elements from the source (global memory) and zero fills
    the rest of the elements in the destination (shared memory).

    In order to do a copy and wait for the result we need the following
    combination:
    ```
    // copy 1.
    %cp1 = nvgpu.device_async_copy %A[%c0], %B[%c0], 4 :memref<16xf32> to memref<16xf32, 3>
    // copy 2.
    %cp2 = nvgpu.device_async_copy %C[%c0], %D[%c0], 4 : memref<16xf32> to memref<16xf32, 3>
    // group 1 contains copy 1 and copy 2.
    %token1 = nvgpu.device_async_create_group %cp1, %cp2
    // copy 3.
    %cp3 = nvgpu.device_async_copy %E[%c0], %F[%c0], 4 : memref<16xf32> to memref<16xf32, 3>
    // group 2 contains copy 3.
    %token2 = nvgpu.device_async_create_group %cp3
    // after the wait copy 1 and copy 2 are complete.
    nvgpu.device_async_wait %token1
    // after the wait copy 3 is complete.
    nvgpu.device_async_wait %token2
    ```

    Example:

    ```mlir
    %0 = nvgpu.device_async_copy %src[%c0, %c0], %dst[%c0, %c0, %c0], 4 :
      memref<4x5xf32> to memref<2x7x5xf32, 3>
    ```
  }];
  let results = (outs NVGPU_DeviceAsyncToken:$asyncToken);
  let arguments = (ins Arg<AnyMemRef, "", [MemWrite]>:$dst,
                       Variadic<Index>:$dstIndices,
                       Arg<AnyMemRef, "", [MemRead]>:$src,
                       Variadic<Index>:$srcIndices,
                       IndexAttr:$dstElements,
                       Optional<Index>:$srcElements,
                       OptionalAttr<UnitAttr>:$bypassL1);
  let assemblyFormat = [{
    $src `[` $srcIndices `]` `,` $dst `[` $dstIndices `]` `,` $dstElements (`,` $srcElements^)?
      attr-dict `:` type($src) `to` type($dst)
  }];
  let hasVerifier = 1;
}

def NVGPU_DeviceAsyncCreateGroupOp : NVGPU_Op<"device_async_create_group", []> {
  let summary = "device side asynchronous create group operation";
  let description = [{
    The `nvgpu.device_async_create_group` op creates a group of memory accesses
    containing all the pending `device_async_copy` operations associated with
    argument tokens. Each token can only be part of one group.

    It returns a token that can be use to wait until the group fully completes.

    This is meant to be used with `nvgpu.device_async_wait` to synchronize copies
    as explained in those ops descriptions.

    Groups are executed in the order they are created.

    Example:

    ```mlir
    %0 = nvgpu.device_async_create_group
  ```
  }];
  let results = (outs NVGPU_DeviceAsyncToken:$asyncToken);
  let arguments = (ins Variadic<NVGPU_DeviceAsyncToken>:$inputTokens);
  let assemblyFormat = [{
    $inputTokens attr-dict
  }];
}

def NVGPU_DeviceAsyncWaitOp : NVGPU_Op<"device_async_wait", []> {
  let summary = "Wait for async gpu ops to complete.";
  let description = [{
    The `nvgpu.device_async_wait` op will block the execution thread until the group
    associated with the source token is fully completed.

    The optional `$numGroups` attribute gives an upper bound of the number of
    groups uncompleted when the wait can unblock the thread. For example,  if
    16 async groups are pushe and `$numGroups` is set to 12, then the thread
    will unblock when 12 groups or fewer are in flight (4 groups have
    completed).

    Example:

    ```mlir
    nvgpu.device_async_wait %0
    ```
  }];
  let arguments = (ins NVGPU_DeviceAsyncToken:$asyncDependencies,
                       OptionalAttr<I32Attr>:$numGroups);
  let assemblyFormat = [{
    $asyncDependencies attr-dict
  }];
}

def NVGPU_MBarrierCreateOp : NVGPU_Op<"mbarrier.create", []> {
  let summary = "Creates a `nvgpu.mbarrier` object.";
  let description = [{
    The Op generates one or more `mbarrier` object, which is a barrier created in 
    shared memory and supports various synchronization behaviors for threads.

    The `mbarrier` object has the following type and alignment requirements:
      Type: .b64, Alignment: 8, Memory space: .shared
    
    Example:
    ```mlir
      %barrier = nvgpu.mbarrier.create -> !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
    ```
    }];
  let arguments = (ins);
  let results = (outs NVGPU_MBarrierGroup:$barriers);
  let assemblyFormat = [{
     attr-dict `->` type($barriers)
  }];
}

def NVGPU_MBarrierInitOp : NVGPU_Op<"mbarrier.init", []> {
  let summary = "Initialize the `nvgpu.mbarrier`.";
  let description = [{
    The Op initializes the `mbarrier` object with the given number of threads.

    Example:
    ```mlir
      %num_threads = gpu.block_dim x
      %barrier = nvgpu.mbarrier.create -> !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
      nvgpu.mbarrier.init %barrier, %num_threads : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
    ```
  }];
  let arguments = (ins NVGPU_MBarrierGroup:$barriers, Index:$count, Index:$mbarId);
  let assemblyFormat = "$barriers `[` $mbarId `]` `,` $count attr-dict `:` type($barriers)";
}

def NVGPU_MBarrierTestWaitOp : NVGPU_Op<"mbarrier.test.wait", []> {
  let summary = "Checks if the `nvgpu.mbarrier` has completed its current phase.";
  let description = [{
    Checks whether the mbarrier object has completed the phase. It is is a 
    non-blocking instruction which tests for the completion of the phase.

    Example:
    ```mlir
      %isComplete = nvgpu.mbarrier.test.wait %barrier, %token : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>, !nvgpu.mbarrier.token
    ```
  }];
  let arguments = (ins NVGPU_MBarrierGroup:$barriers, NVGPU_MBarrierToken:$token, Index:$mbarId);
  let results = (outs I1:$waitComplete);
  let assemblyFormat = "$barriers `[` $mbarId `]` `,` $token attr-dict `:` type($barriers) `,` type($token)";
}

def NVGPU_MBarrierArriveOp : NVGPU_Op<"mbarrier.arrive", []> {
  let summary = "Performs arrive operation on the `nvgpu.mbarrier.arrive`.";
  let description = [{
    The Op performs arrive-on operation on the `mbarrier` object and returns a 
    `nvgpu.mbarrier.token`.

    For more information, see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#arrive-on-operation-on-mbarrier-object

    Example:
    ```mlir
      %token = nvgpu.mbarrier.arrive %barrier : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>> -> !nvgpu.mbarrier.token
    ```
  }];
  let arguments = (ins NVGPU_MBarrierGroup:$barriers, Index:$mbarId);
  let results = (outs NVGPU_MBarrierToken:$token);
let assemblyFormat = "$barriers `[` $mbarId `]` attr-dict `:` type($barriers) `->` type($token)";
}

def NVGPU_MBarrierArriveNoCompleteOp : NVGPU_Op<"mbarrier.arrive.nocomplete", []> {
  let summary = "Performs arrive operation on the `nvgpu.mbarrier.arrive.nocomplete` as non-blocking.";
  let description = [{
    The Op performs arrive-on operation on the `mbarrier` object and returns a 
    `nvgpu.mbarrier.token`.

    The Op does not cause the `nvgpu.mbarrier` to complete its current phase.

    Example:
    ```mlir
      %token = nvgpu.mbarrier.arrive.noComplete %barrier, %count : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>> -> !nvgpu.mbarrier.token
    ```
  }];
  let arguments = (ins NVGPU_MBarrierGroup:$barriers, Index:$mbarId,
                       Index:$count);
  let results = (outs NVGPU_MBarrierToken:$token);
  let assemblyFormat = "$barriers `[` $mbarId `]` `,` $count attr-dict `:` type($barriers) `->` type($token)";
}

def NVGPU_MBarrierArriveExpectTxOp : NVGPU_Op<"mbarrier.arrive.expect_tx", []> {
  let summary = "Performs expect_tx operation on the `nvgpu.mbarrier.arrive`";
  let description = [{
    A thread executing the Op performs an expect-tx operation on the mbarrier 
    object at the location specified by the address operand $barrier. The 
    expect-tx operation, with an $txcount argument, increases the tx-count of 
    an mbarrier object by the value specified by $txcount. This makes the 
    current phase of the mbarrier object to expect and track the completion of 
    additional asynchronous transactions.
    
    The `$txCount` specifies the number of element to the expect-tx operation.

    Example:
    ```mlir
      nvgpu.mbarrier.arrive.expect_tx %barrier, %ic0 : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
    ```
  }];
  let arguments = (ins NVGPU_MBarrierGroup:$barriers, Index:$txcount, Index:$mbarId);
  let assemblyFormat = "$barriers `[` $mbarId `]` `,` $txcount  attr-dict `:` type($barriers)";
}

def NVGPU_MBarrierTryWaitParityOp : NVGPU_Op<"mbarrier.try_wait.parity", []> {
  let summary = "Waits for the `nvgpu.mbarrier` to complete its current phase.";
  let description = [{
    Checks whether the mbarrier object has completed the phase. It is is a 
    potentially blocking instruction which tests for the completion of the 
    phase. Suspended thread resumes execution when the specified phase completes 
    OR before the phase completes following a system-dependent time limit. 

    Example:
    ```mlir
      nvgpu.mbarrier.try_wait.parity %barrier, %phase, %ticks : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
    ```

  }];
  let arguments = (ins NVGPU_MBarrierGroup:$barriers, Index:$phase, Index:$ticks, Index:$mbarId);
  let assemblyFormat = "$barriers `[` $mbarId `]` `,` $phase `,` $ticks attr-dict `:` type($barriers)";  
}

def NVGPU_TmaAsyncLoadOp : NVGPU_Op<"tma.async.load", []> {
  let summary = "TMA asynchronous load";
  let description = [{
    The Op loads a tile memory region from global memory to shared memory by 
    Tensor Memory Access (TMA).
    
    `$tensorMapDescriptor` is tensor map descriptor which has information about
    tile shape. The descriptor is created by `nvgpu.tma.create.descriptor`

    The Op uses `$barrier` mbarrier based completion mechanism. 
  }];  
  let arguments = (ins Arg<AnyMemRef, "", [MemWrite]>:$dst,
                       NVGPU_MBarrierGroup:$barriers,
                       NVGPU_TensorMapDescriptor:$tensorMapDescriptor,
                       Variadic<Index>:$coordinates, 
                       Index:$mbarId);
  let assemblyFormat = [{
    $tensorMapDescriptor `[` $coordinates `]` `,` $barriers `[` $mbarId `]` `to` $dst
      attr-dict `:` type($tensorMapDescriptor) `,` type($barriers) `->` type($dst)
  }];
  let hasVerifier = 1;

}

def NVGPU_TmaCreateDescriptorOp : NVGPU_Op<"tma.create.descriptor", []> {
  let summary = "TMA create descriptor";
  let description = [{
    The Op creates a tensor map descriptor object representing tiled memory 
    region. To do that it calls CUDA Driver's `cuTensorMapEncodeTiled`. The 
    descriptor is used by Tensor Memory Access (TMA).

    The `tensor` is the source tensor to be tiled. 

    The `boxDimensions` is the size of the tiled memory region in each dimension.

    For more information see below:
    https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
  }];

  let arguments = (ins AnyUnrankedMemRef:$tensor,
                       Variadic<Index>:$boxDimensions);
  let results = (outs NVGPU_TensorMapDescriptor:$tensorMap);
  let assemblyFormat = [{
         $tensor `box` `[` $boxDimensions `]` attr-dict `:` type($tensor) `->` type($tensorMap)
  }];
  let hasVerifier = 1;
}

def NVGPU_GenerateGmmaDescriptorOp : NVGPU_Op<"wgmma.generate.descriptor", []> {
  let summary = "Generate a wgmma matrix descriptor";
  let description = [{
  This Op builds a `nvgpu.wgmma.descriptor` that is used by warpgroup-level 
  matrix multiply and accumulate.

  The descriptor specifies the properties of the matrix in shared memory that 
  is a multiplicand in the matrix multiply and accumulate operation. 
  }];  
  let results = (outs NVGPU_WarpgroupMatrixDescriptor:$descriptor);
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$tensor, 
                       NVGPU_TensorMapDescriptor:$tensorMap);
  let assemblyFormat = [{$tensor `,` $tensorMap attr-dict `:` type($tensor) `,` type($tensorMap) `->` type($descriptor)}];
  let hasVerifier = 1;
}

def NVGPU_WarpgroupMmaOp : NVGPU_Op<"warpgroup.mma"> {
  let description = [{
    The `nvgpu.warpgroup.mma` op performs the warpgroup-level (4 warps) 
    matrix-multiply-and-accumulate (mma) operation that results in 
    `nvvm.wgmma.mma_async`. 
    
    The operands are `descriptorA` and `descriptorB` that are wgmma matrix 
    descriptors that shows the properties of the matrix in shared memory. The 
    results are thread-level ownership to the warpgroup-level mma operation 
    shape. The shape is deduced from the descriptor types and output vector.

    The Op corresponds multiple `nvvm.wgmma.mma_async` operations to complete the 
    given shape. As the instruction `nvvm.wgmma.async` is an asynchronous, 
    this Op groups the `nvvm.wgmma.async` and surrounds them between 
    `wgmma.fence.aligned` and `wgmma.commit.group.sync.aligned`, 
    `wgmma.wait.group.sync.aligned` Ops.

    Example:
    ```mlir
      %r1,%r2 = nvgpu.warpgroup.mma %wgmmaDescA, %wgmmaDescB, %acc1, %acc2: 
                 !nvgpu.wgmma.descriptor<tensor = memref<128x64xf16, 3>>, 
                 !nvgpu.wgmma.descriptor<tensor = memref<64x128xf16, 3>>, 
                 !nvgpu.warpgroup.accumulator<fragmented = vector<64x128xf32>>,
                 !nvgpu.warpgroup.accumulator<fragmented = vector<64x128xf32>>
                 -> 
                 !nvgpu.warpgroup.accumulator<fragmented = vector<64x128xf32>>,
                 !nvgpu.warpgroup.accumulator<fragmented = vector<64x128xf32>>
    ```
  }];

  let arguments = (ins NVGPU_WarpgroupMatrixDescriptor:$descriptorA, 
                       NVGPU_WarpgroupMatrixDescriptor:$descriptorB,                                               
                       DefaultValuedOptionalAttr<I32Attr, "1">:$waitGroup,
                       OptionalAttr<UnitAttr>:$transposeA,
                       OptionalAttr<UnitAttr>:$transposeB,
                       Variadic<NVGPU_WarpgroupAccumulator>:$matrixC);
  let results = (outs Variadic<NVGPU_WarpgroupAccumulator>:$matrixD);
  let assemblyFormat = [{    
    $descriptorA`,` $descriptorB`,` $matrixC attr-dict
    `:` type($descriptorA) `,` type($descriptorB) `,` type($matrixC) `->` type($matrixD)
  }];
  let hasVerifier = 1;
}

#endif // NVGPU
