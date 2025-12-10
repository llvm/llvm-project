//===- ParallelCombiningOpInterface.td - Parallel iface ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for ops that perform in parallel combining
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_PARALLELCOMBININGOPINTERFACE
#define MLIR_INTERFACES_PARALLELCOMBININGOPINTERFACE

include "mlir/IR/OpBase.td"

def InParallelOpInterface : OpInterface<"InParallelOpInterface"> {
  let description = [{
    An in parallel op is an op with a region.

    This is useful as a terminator to parallel operations that iterate over
    some set and return tensors while avoiding tight coupling between the
    iterating op, the combining op and the individual subtensor producing ops.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return `idx`^th result of the parent operation.
      }],
      /*retTy=*/"::mlir::OpResult",
      /*methodName=*/"getParentResult",
      /*args=*/(ins "int64_t":$idx),
      /*methodBody=*/[{
        return $_op.getParentResult(idx);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the contained ops that yield subvalues that this op combines to
        yield to its parent.
      }],
      /*retTy=*/"::llvm::iterator_range<Block::iterator>",
      /*methodName=*/"getYieldingOps",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_op.getYieldingOps();
      }]
    >,
  ];
  // TODO: Single region single block interface on interfaces ?
  let verify = [{
    return verifyInParallelOpInterface($_op);
  }];
}

def ParallelCombiningOpInterface : OpInterface<"ParallelCombiningOpInterface"> {
  let description = [{
    A parallel combining op is an operation that models parallel contributions
    to result tensors within the context of a parent iterating operation.
    
    This interface is designed for operations that need to coordinate parallel
    insertions or contributions to tensors that are being constructed across
    multiple parallel iterations. The destination refers to a tensor value that
    is assembled by aggregating results from parallel computations; each
    parallel iteration may contribute a slice, element, or region to the final
    result. No in-place mutation of tensors is implied.

    One significant use case for this interface is `tensor.parallel_insert_slice`
    which allows parallel insertion of slices that are aggregated into a
    destination tensor. With this interface, other operations that express
    similar parallel contributions can also be defined.

    This op works within an op implementing the `InParallelOpInterface` that
    specifies how the parallel results are combined.

    Key semantics:
    - The operation identifies destination tensors to which iterations
      contribute through the `getUpdatedDestinations` method
    - Each parallel iteration may produce elements or regions that are
      incorporated into the destination tensor
    - The parent iterating operation manages the coordination and ensures
      proper synchronization of these contributions
  
    Note: This interface does not verify itself, it is up to the implementing operation
    to verify the correctness of the op.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{ 
        Returns the list of destination values this op contributes to.
      }],
      /*retTy=*/"::mlir::MutableOperandRange",
      /*methodName=*/"getUpdatedDestinations",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the iterating parent for this op.
      }],
      /*retTy=*/"::mlir::Operation*",
      /*methodName=*/"getIteratingParent",
      /*args=*/(ins)
    >,
  ];
}

#endif // MLIR_INTERFACES_PARALLELCOMBININGOPINTERFACE
