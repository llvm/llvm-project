//===- PartTensorOps.td - PartTensor dialect ops ------*- tablegen -*-===//

// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The following operations bootstrap working with partition solely
// within the Linalg dialect. They provide temporary bridges between a
// future PartTensorType (now an opaque pointer), the actual TensorType,
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
// TODO: introduce PartTensorType as first class citizen in MLIR
//
//===----------------------------------------------------------------------===//

#ifndef PARTTENSOR_OPS
#define PARTTENSOR_OPS

include "mlir/Dialect/PartTensor/IR/PartTensorBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class.
class PartTensor_Op<string mnemonic, list<Trait> traits = []>
    : Op<PartTensor_Dialect, mnemonic, traits>;

def PartTensor_GetPartitionOp : PartTensor_Op<"get_partitions">,
                                Arguments<(ins AnyTensor
                                           : $primary)>,
                                Results<(outs TensorOf<[AnySignlessIntegerOrIndex]> 
                                         : $tuples)> {
  let summary = "get all the partition specs";
  let description = [{
     This operation will take a tensor and give all the indices of a tensor in a hyper-rectangular form where 
	 tuples will give the start and end indices of the the partition tensor from the tensor.
     Example:
     ```mlir
      !Partition = type !llvm.ptr<i8>
	  arg0 = tensor<?x?xf32, #sp, # part>

      %0 = part_tensor.get_partitions %arg0 : <?x?xf32, #ar1, #ar2> -> tensor<?xindex>
     ```
   }];
  let assemblyFormat =
      "$primary attr-dict `:` type($primary) `->` type($tuples)";
}
 def PartTensor_GetSliceOp : PartTensor_Op<"get_slice">,
 	Arguments<(ins AnyTensor:$Parts, TensorOf<[AnySignlessIntegerOrIndex]>:$tupleList)>,
	Results<(outs AnyTensor:$result)> {
		let summary = "get slice";
		let description = [{
			This operation will take tuples and output one tensor
			}];
		//%slice = partition.get_slice(%total_iteration_space,%partition) -> tensor<2xindex>
		let assemblyFormat = " $Parts `,` $tupleList attr-dict `:` type($Parts) `,` type($tupleList) `->` type($result)";
		}

def PartTensor_NumPartitionOp : PartTensor_Op<"get_num_partitions">,
	Arguments<(ins AnyTensor:$tensor)>,
	Results<(outs AnySignlessIntegerOrIndex:$res)> {
		let summary = "get the size of partition";
		let description = [{
			This operation will represent the dimention
			}];
		// %partitions_num = partition.num_partitions %A: tensor<?x?xf32> -> index
		let assemblyFormat = " $tensor  attr-dict `:` type($tensor) `->` type($res)";
		}

#endif // PARTTENSOR_OPS
