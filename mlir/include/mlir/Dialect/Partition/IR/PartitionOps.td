//===- PartitionOps.td - Partition dialect ops ------*- tablegen -*-===//

 // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 // See https://llvm.org/LICENSE.txt for license information.
 // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 //
 //===----------------------------------------------------------------------===//
 //
 // The following operations bootstrap working with partition solely
 // within the Linalg dialect. They provide temporary bridges between a
 // future PartitionType (now an opaque pointer), the actual TensorType,
 // and MemRef arrays underlying an actual partition storage scheme in memory.
 //
 // Lacking a proper partition type, the 'partition' operation
 // provides a bridge between an opaque pointer and a regular tensor type
 // just to simplify feeding the value into a Linalg op. The operation
 // simply disappears during lowering.
 //
 // The other operations form the bridge between the opaque pointer and
 // the actual storage of pointers, indices, and values. These operations
 // resemble 'buffer_cast' in the sense that they map tensors to
 // their bufferized memrefs, but they lower into actual calls since
 // partition storage does not bufferize into a single memrefs, as dense
 // tensors do, but into a hierarchical storage scheme where pointers
 // access memrefs with indices and eventually into values.
 //
 // TODO: introduce PartitionType as first class citizen in MLIR
 //
 //===----------------------------------------------------------------------===//

 #ifndef PARTITION_OPS
 #define PARTITION_OPS

 include "mlir/Dialect/Linalg/IR/LinalgBase.td"
 include "mlir/Dialect/Partition/IR/PartitionBase.td"
 include "mlir/Interfaces/SideEffectInterfaces.td"

 // Base class.
 class Partition_Op<string mnemonic, list<Trait> traits = []>
   : Op<Partition_Dialect, mnemonic, traits>; 

 def Partition_FromPointerOp : Partition_Op<"fromPtr">,
     Arguments<(ins AnyType:$ptr)>,
     Results<(outs AnyTensor:$result)> {
   let summary = "Views an opaque partition pointer as a tensor";
  let description = [{
     Lacking a first class citizen type for partition, this operation
     forms the glue between a partition storage scheme (behind an opaque
     pointer) and the (dense) tensors used in the kernel definitions.
     This operation merely provides a way to assign a proper tensor
     type and shape to the incoming opaque pointer. It disappears
     completely during lowering.
     Example:
     ```mlir
      !Partition = type !llvm.ptr<i8>

      %0 = partition.fromPtr %arg0 : !Partition to tensor<64x64xf64>
     ```
   }];
   let assemblyFormat = "$ptr attr-dict `:` type($ptr) `to` type($result)";
 }


 #endif // PARTITION_OPS
