//===-- SubsetInsertionOpInterface.td - Tensor Subsets -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSET_INSERTION_OP_INTERFACE
#define SUBSET_INSERTION_OP_INTERFACE

include "mlir/IR/OpBase.td"

def SubsetInsertionOpInterface : OpInterface<"SubsetInsertionOpInterface"> {
  let description = [{
    This interface can be implemented by ops that insert a source tensor into
    a destination tensor.

    The elements in the destination tensor that are overwritten by this
    insertion are called the "subset". How the subset is defined is up to the
    op. E.g., "tensor.insert_slice" defines the subset via a hyperrectangular
    slice. A scatter operation could define the subset via a list of indices.

    Ops that deal with tensor subsets come in two flavours:
    - Insertion flavor: Ops that insert a source tensor into a destination
      tensor at the specified subset. Such ops usually return a new destination
      tensor and implement the `DestinationStyleOpInterface`. Insertion ops can
      implement the `SubsetInsertionOpInterface`. Example: "tensor.insert_slice"
    - Extraction flavor: Ops that define a tensor subset. They extract a
      specified subset from a tensor. There is currently no op interface for
      such ops. Example: "tensor.extract_slice"

    This interface provides helper methods for efficient bufferization of
    subset-based tensor IR. Tensor subsets can bufferize to buffer "views"/
    "aliases" (in contrast to one or multiple less efficient buffer allocation).

    This interface is queried by One-Shot Bufferize to detect cases where a
    seeming read-after-write is not actually a conflict because the respective
    ops are operating on equivalent subsets. More details can be found in the
    documentation of One-Shot Analysis (see `areNonConflictingSubsets`).

    Note: This interface currently assumes that a subset op inserts a single
    tensor (source) into a destination tensor at a single subset.
  }];
  let cppNamespace = "::mlir::bufferization";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Return the source tensor operand.
        }],
        /*retType=*/"::mlir::OpOperand &",
        /*methodName=*/"getSourceOperand",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the destination tensor operand.
        }],
        /*retType=*/"::mlir::OpOperand &",
        /*methodName=*/"getDestinationOperand",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return ::mlir::bufferization::detail::defaultGetDestinationOperand(
              $_op.getOperation());
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return "true" if this operation inserts into a subset that is
          equivalent to the subset defined by `candidate`.

          Two subsets are "equivalent" and "same" if they can bufferize to the
          same buffer views/aliases. If they are "equivalent", the tensor IR
          may be expressed in terms of different SSA values (but they could
          bufferize to MemRef SSA values that can CSE without breaking
          correctness). `equivalenceFn` should return "true" if the two given
          values are equivalent.

          Example:
          ```
          // The subset of the SubsetInsertionOpInterface op %1 is equivalent to
          // the subset defined by %2 (but not "same"):
          %0 = arith.select %c, %t, %t : tensor<?xf32>
          %1 = tensor.insert_slice %x into %0[0][5][1]
              : tensor<5xf32> into tensor<?xf32>
          %2 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>

          // The subset of the SubsetInsertionOpInterface op %1 is equivalent to
          // and "same" as the subset defined by %2.
          %1 = tensor.insert_slice %x into %t[0][5][1]
              : tensor<5xf32> into tensor<?xf32>
          %2 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
          ```
        }],
        /*retType=*/"bool",
        /*methodName=*/"isEquivalentSubset",
        /*args=*/(ins
            "::mlir::Value":$candidate,
            "::llvm::function_ref<bool(Value, Value)>":$equivalenceFn)
      >,
  ];

  let extraClassDeclaration = [{
    /// Return "true" if this operation inserts into the same subset as defined
    /// by `candidate`.
    ///
    /// Note: This function is useful outside of bufferization, where no tensor
    /// equivalence information is available.
    bool isSameSubset(OpResult candidate) {
      auto subsetOp = cast<::mlir::bufferization::SubsetInsertionOpInterface>(
          getOperation());
      return subsetOp.isEquivalentSubset(
          candidate, [](Value v1, Value v2) { return v1 == v2; });
    }
  }];
}

#endif // SUBSET_INSERTION_OP_INTERFACE
