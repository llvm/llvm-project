//===-- SubsetOpInterface.td - Tensor Subsets --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSET_OP_INTERFACE
#define SUBSET_OP_INTERFACE

include "mlir/IR/OpBase.td"

def SubsetOpInterface : OpInterface<"SubsetOpInterface"> {
  let description = [{
    This interface can be implemented by ops that operate on tensor subsets. A
    "subset" is a part of a tensor. This interface describes the subset that
    an implementing op operates on. Only the specified subset may be accessed by
    the op.

    Subset ops come in two flavours and ops that implement the
    `SubsetOpInterface` must also implement one of the respective interfaces.
    - Insertion flavor: Ops that insert a source value into a destination
      tensor at the specified subset. Such ops return an updated destination
      tensor and usually implement the `DestinationStyleOpInterface`. Insertion
      ops must implement the `SubsetInsertionOpInterface`.
    - Extraction flavor: Ops that extract at a subset. Extraction ops must
      implement the `SubsetExtractionOpInterface`.

    How the subset is specified is up to the implementing op. E.g.:
    - `tensor.extract_slice/insert_slice` describe the subset as a
      hyperrectangular slice.
    - `tensor.gather/scatter` describe the subset as list of indices. (Not
      implemented yet.)
  }];

  let cppNamespace = "::mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Return "true" if this op and the given candidate subset op operate on
          equivalent subsets. Return "false" if the two subsets are disjoint
          or cannot be proven to be equivalent.

          This interface method does not have to be implemented if
          `getAccessedHyperrectangularSlice` is implemented.
        }],
        /*retType=*/"bool",
        /*methodName=*/"operatesOnEquivalentSubset",
        /*args=*/(ins
            "::mlir::SubsetOpInterface":$candidate,
            "::llvm::function_ref<bool(Value, Value)>":$equivalenceFn),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return ::mlir::detail::defaultOperatesOnEquivalentSubset(
              $_op, candidate, equivalenceFn);
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return "true" if this op and the given candidate subset op operate on
          disjoint subsets. Return "false" if the two subsets are equivalent,
          overlapping or cannot be proven to be disjoint.

          This interface method does not have to be implemented if
          `getAccessedHyperrectangularSlice` is implemented.
        }],
        /*retType=*/"bool",
        /*methodName=*/"operatesOnDisjointSubset",
        /*args=*/(ins
            "::mlir::SubsetOpInterface":$candidate,
            "::llvm::function_ref<bool(Value, Value)>":$equivalenceFn),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return ::mlir::detail::defaultOperatesOnDisjointSubset(
              $_op, candidate, equivalenceFn);
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          If this op operates on a hyperrectangular subset, return a
          description of the subset in terms of offsets, sizes and strides.
          Otherwise, return "failure".

          This interface method is a convenience method for the most common case
          of hyperrectangular subset ops. It is optional. If it is implemented,
          `operatesOnEquivalentSubset` and `operatesOnDisjointSubset` do not
          have to be implemented.
        }],
        /*retType=*/"::mlir::FailureOr<::mlir::HyperrectangularSlice>",
        /*methodName=*/"getAccessedHyperrectangularSlice",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return ::mlir::failure();
        }]
      >,
  ];

  let verify = [{
    return ::mlir::detail::verifySubsetOpInterface(
        ::mlir::cast<::mlir::SubsetOpInterface>($_op));
  }];

  let extraClassDeclaration = [{
    /// Return the container that this operation is operating on. In case of an
    /// extraction op, the container is the source tensor. In case of an
    /// insertion op, the container is the destination tensor.
    Value getTensorContainer() {
      return ::mlir::detail::getTensorContainer(getOperation());
    }
  }];
}

def SubsetExtractionOpInterface
    : OpInterface<"SubsetExtractionOpInterface", [SubsetOpInterface]> {
  let description = [{
    This interface can be implemented by ops that extract a value from
    a source tensor at a specified subset. The elements in the source tensor
    that are read by this extraction are called "subset".

    Extraction ops must have a single result value.
  }];

  let cppNamespace = "::mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Return the source tensor operand.
        }],
        /*retType=*/"::mlir::OpOperand &",
        /*methodName=*/"getSourceOperand",
        /*args=*/(ins)
      >,
  ];

  let verify = [{
    return ::mlir::detail::verifySubsetExtractionOpInterface(
        ::mlir::cast<::mlir::SubsetExtractionOpInterface>($_op));
  }];

  let extraClassDeclaration = [{
    /// Return the single result of this op.
    ::mlir::Value getResult() {
      return getOperation()->getResult(0);
    }
  }];
}

def SubsetInsertionOpInterface
    : OpInterface<"SubsetInsertionOpInterface", [SubsetOpInterface]> {
  let description = [{
    This interface can be implemented by ops that insert a source value into
    a destination tensor at a specified subset. The elements in the destination
    tensor that are overwritten by this insertion are called "subset". The
    updated destination tensor is returned.

    This interface provides helper methods for efficient bufferization of
    subset-based tensor IR. Tensor subsets can bufferize to buffer "views"/
    "aliases" (in contrast to one or multiple less efficient buffer allocation).

    This interface is queried by One-Shot Bufferize to detect cases where a
    seeming read-after-write is not actually a conflict because the respective
    ops are operating on equivalent subsets. More details can be found in the
    documentation of One-Shot Analysis (see `areNonConflictingSubsets`).
  }];

  let cppNamespace = "::mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Return the source operand. The source operand can have any type.
        }],
        /*retType=*/"::mlir::OpOperand &",
        /*methodName=*/"getSourceOperand",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the destination operand. The destination operand must be a
          tensor.

          This function does not have to be implemented for destination style
          ops that have exactly one "init" operand.
        }],
        /*retType=*/"::mlir::OpOperand &",
        /*methodName=*/"getDestinationOperand",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return ::mlir::detail::defaultGetDestinationOperand(
              $_op.getOperation());
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the updated destination result.

          This function does not have to be implemented for destination style
          ops.
        }],
        /*retType=*/"::mlir::OpResult",
        /*methodName=*/"getUpdatedDestination",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return ::mlir::detail::defaultGetUpdatedDestination(
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

          Note: By default, this function calls
          `SubsetOpInterface::operatesOnEquivalentSubset`.
        }],
        /*retType=*/"bool",
        /*methodName=*/"isEquivalentSubset",
        /*args=*/(ins
            "::mlir::Value":$candidate,
            "::llvm::function_ref<bool(Value, Value)>":$equivalenceFn),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return ::mlir::detail::defaultIsEquivalentSubset(
              $_op.getOperation(), candidate, equivalenceFn);
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the subset of the destination tensor that this operation
          inserts into.

          Example:
          ```
          // SubsetOpInterface op:
          %0 = tensor.insert_slice %t0 into %t1[%pos][5][1]
              : tensor<5xf32> into tensor<?xf32>
          // Subset (built by this function):
          %1 = tensor.extract_slice %t1[%pos][5][1]
              : tensor<?xf32> to tensor<5xf32>
          ```

          Note: Implementations do not necessarily have to build new IR. They
          may return existing SSA values.
        }],
        /*retType=*/"::mlir::Value",
        /*methodName=*/"buildSubsetExtraction",
        /*args=*/(ins "::mlir::OpBuilder &":$builder, "Location":$loc)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return all SSA values that are needed (i.e., must be in scope) at the
          insertion of the builder when calling `buildSubsetExtraction`. Users
          of `buildSubsetExtraction` can use this helper method to find a
          suitable insertion point.

          Example: The SSA values needed to build the subset in the example of
          `buildSubsetExtraction` are %t1 and %pos.
        }],
        /*retType=*/"::llvm::SmallVector<::mlir::Value>",
        /*methodName=*/"getValuesNeededToBuildSubsetExtraction",
        /*args=*/(ins)
      >,
  ];

  let extraClassDeclaration = [{
    /// Return "true" if this operation inserts into the same subset as defined
    /// by `candidate`.
    ///
    /// Note: This function is useful outside of bufferization, where no tensor
    /// equivalence information is available.
    bool isSameSubset(OpResult candidate) {
      auto subsetOp = cast<::mlir::SubsetInsertionOpInterface>(
          getOperation());
      return subsetOp.isEquivalentSubset(
          candidate, [](Value v1, Value v2) { return v1 == v2; });
    }
  }];
}

#endif // SUBSET_OP_INTERFACE
