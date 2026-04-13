//===- NVGPUTypes.td - NVGPU types -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVGPU dialect types.
//
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_NVGPU_IR_NVGPUTYPES_TD
#define MLIR_DIALECT_NVGPU_IR_NVGPUTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/NVGPU/IR/NVGPU.td"

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

def NVGPU_WarpgroupMatrixDescriptor : NVGPU_Type<"WarpgroupMatrixDescriptor", "warpgroup.descriptor", []> {
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

#endif //MLIR_DIALECT_NVGPU_IR_NVGPUTYPES_TD
