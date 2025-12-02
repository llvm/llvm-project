//===- VectorInterfaces.td - Vector interfaces -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for operations on vectors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_VECTORINTERFACES
#define MLIR_INTERFACES_VECTORINTERFACES

include "mlir/IR/OpBase.td"

def VectorUnrollOpInterface : OpInterface<"VectorUnrollOpInterface"> {
  let description = [{
    Encodes properties of an operation on vectors that can be unrolled.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the shape ratio of unrolling to the target vector shape
        `targetShape`. Return `std::nullopt` if the op cannot be unrolled to the
        target vector shape.
      }],
      /*retTy=*/"::std::optional<::llvm::SmallVector<int64_t, 4>>",
      /*methodName=*/"getShapeForUnroll",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op->getNumResults() == 1);
        auto vt =
            ::llvm::dyn_cast<::mlir::VectorType>($_op.getResult().getType());
        if (!vt)
          return ::std::nullopt;
        ::llvm::SmallVector<int64_t, 4> res(
            vt.getShape().begin(), vt.getShape().end());
        return res;
      }]
    >,
  ];
}

def VectorTransferOpInterface : OpInterface<"VectorTransferOpInterface"> {
  let description = [{
    Encodes properties of a `vector.transfer_read` or `vector.transfer_write`
    operation. Vector transfer ops have:

    - A shaped value that the op reads from/writes to: a memref or a tensor.
    - A vector, either as a result or as an operand.
    - Indices that describe where the transfer from/to the shaped value starts.
    - An optional mask.
    - An optional in_bounds array to indicate transfer dimensions that are
      guaranteed to be in-bounds.
    - A permutation map to indicate transposes and broadcasts.

    The "vector rank" is the rank of the vector type. E.g.:
    ```
    // Transfer with shaped value rank 2 and vector (transfer) rank 1.
    %0 = vector.transfer_read %arg0[%c3, %c3], %f0
        {permutation_map = affine_map<(d0, d1) -> (d0)>}
        : memref<?x?xf32>, vector<128xf32>
    ```

    The "vector transfer rank" is the number of dimensions that participate in
    the transfer and broadcasts, and matches the number of results in the
    permutation map. In most cases, the vector rank matches the vector transfer
    rank; the only exception is when a vector is flattened as part of the
    transfer (see `getPermutationMap`).
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the `in_bounds` attribute name.
      }],
      /*retTy=*/"::mlir::StringRef",
      /*methodName=*/"getInBoundsAttrName",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `permutation_map` attribute name.
      }],
      /*retTy=*/"::mlir::StringRef",
      /*methodName=*/"getPermutationMapAttrName",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the optional in_bounds attribute that specifies for each vector
        dimension whether it is in-bounds or not. (Broadcast dimensions are
        always in-bounds).
      }],
      /*retTy=*/"::mlir::ArrayAttr",
      /*methodName=*/"getInBounds",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the memref or ranked tensor operand that this operation operates
        on. In case of a "read" operation, that's the source from which the
        operation reads. In case of a "write" operation, that's the destination
        into which the operation writes.
        TODO: Change name of operand, which is not accurate for xfer_write.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getBase",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the vector that this operation operates on. In case of a "read",
        that's the vector OpResult. In case of a "write", that's the vector
        operand value that is written by the op.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getVector",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the type of the vector that this operation operates on.
      }],
      /*retTy=*/"::mlir::VectorType",
      /*methodName=*/"getVectorType",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indices that specify the starting offsets into the source
        operand. The starting offsets are guaranteed to be in-bounds.
      }],
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getIndices",
      /*args=*/(ins)
    >,

    InterfaceMethod<
      /*desc=*/[{
        Return the permutation map that describes the mapping of vector
        dimensions to source dimensions, as well as broadcast dimensions.

        The permutation result has one result per vector transfer dimension.
        Each result is either a dim expression, indicating the corresponding
        dimension in the source operand, or a constant "0" expression,
        indicating a broadcast dimension.

        Note: Nested vector dimensions that are flattened by this op are not
        accounted for in the permutation map. E.g.:
        ```
        // Vector type has rank 4, but permutation map has only 2 results. That
        // is because there are only 2 transfer dimensions.
        %0 = vector.transfer_read %arg1[%c3, %c3], %vf0
            {permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
            : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
        ```
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"getPermutationMap",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the mask operand if the op has a mask. Otherwise, return an
        empty value.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getMask",
      /*args=*/(ins)
    >
  ];

  let extraSharedClassDeclaration = [{
    /// Return a vector of all in_bounds values as booleans (one per vector
    /// transfer dimension).
    ::llvm::SmallVector<bool> getInBoundsValues() {
      ::llvm::SmallVector<bool> inBounds;
      for (int64_t i = 0, e = $_op.getTransferRank(); i < e; ++i)
        inBounds.push_back($_op.isDimInBounds(i));
      return inBounds;
    }

    /// Return the number of leading shaped dimensions (of the "source" operand)
    /// that do not participate in the permutation map.
    unsigned getLeadingShapedRank() {
      return $_op.getShapedType().getRank() - $_op.getTransferRank();
    }

    /// Return the mask type if the op has a mask. Otherwise, return an empty
    /// VectorType.
    ::mlir::VectorType getMaskType() {
      return $_op.getMask()
        ? ::llvm::cast<::mlir::VectorType>($_op.getMask().getType())
        : ::mlir::VectorType();
    }

    /// Return the shaped type of the "source" operand value.
    ::mlir::ShapedType getShapedType() {
      return ::llvm::cast<::mlir::ShapedType>($_op.getBase().getType());
    }

    /// Return the number of dimensions that participate in the permutation map.
    unsigned getTransferRank() {
      return $_op.getPermutationMap().getNumResults();
    }

    /// Return "true" if at least one of the vector dimensions is a broadcasted
    /// dimension.
    bool hasBroadcastDim() {
      for (unsigned i = 0, e = $_op.getTransferRank(); i < e; ++i) {
        if ($_op.isBroadcastDim(i))
          return true;
      }
      return false;
    }

    /// Return "true" if at least one of the vector dimensions may be
    /// out-of-bounds.
    bool hasOutOfBoundsDim() {
      for (unsigned idx = 0, e = $_op.getTransferRank(); idx < e; ++idx)
        if (!$_op.isDimInBounds(idx))
          return true;
      return false;
    }

    /// Return "true" if the specified vector transfer dimension is a
    /// broadcasted dimension.
    bool isBroadcastDim(unsigned dim) {
      auto expr = $_op.getPermutationMap().getResult(dim);
      auto constExpr = ::llvm::dyn_cast<::mlir::AffineConstantExpr>(expr);
      return constExpr && constExpr.getValue() == 0;
    }

    /// Return "true" if the vector transfer dimension `dim` is in-bounds.
    /// Return "false" otherwise.
    bool isDimInBounds(unsigned dim) {
      auto inBounds = $_op.getInBounds();
      return ::llvm::cast<::mlir::BoolAttr>(inBounds[dim]).getValue();
    }

    /// Helper function to account for the fact that `permutationMap` results
    /// and `op.getIndices` sizes may not match and may not be aligned. The
    /// first `getLeadingShapedRank()` indices may just be indexed and not
    /// transferred from/into the vector.
    /// For example:
    /// ```
    /// vector.transfer %0[%i, %j, %k, %c0]
    ///     : memref<?x?x?x?xf32>, vector<2x4xf32>
    /// ```
    /// with `permutation_map = (d0, d1, d2, d3) -> (d2, d3)`.
    /// Provide a zip function to coiterate on 2 running indices: `resultIdx`
    /// and `indicesIdx` which accounts for this misalignment.
    void zipResultAndIndexing(
        ::llvm::function_ref<void(int64_t, int64_t)> fun) {
      for (int64_t resultIdx = 0,
                   indicesIdx = $_op.getLeadingShapedRank(),
                   eResult = $_op.getTransferRank();
          resultIdx < eResult;
          ++resultIdx, ++indicesIdx)
      fun(resultIdx, indicesIdx);
    }

    /// Return the shape of the hyperrectangular slice within the tensor/memref
    /// operand that is accessed by the transfer op.
    /// For example:
    /// ```
    /// vector.transfer %w0[%i, %j, %k] {
    ///     permutation_map = affine_map<(d0, d1, d2) -> (d1, d0, 0)>} :
    ///     : tensor<?x?x?xf32>, vector<4x2x6xf32>
    /// ```
    /// returns a shape [2, 4, 1].
    SmallVector<int64_t> getTransferChunkAccessed() {
      SmallVector<int64_t> dimSizes($_op.getPermutationMap().getNumDims(), 1);
      for (auto vecDims : llvm::zip($_op.getPermutationMap().getResults(),
                                    $_op.getVectorType().getShape())) {
        AffineExpr dim = std::get<0>(vecDims);
        int64_t size = std::get<1>(vecDims);
        // Skip broadcast.
        if (::llvm::isa<AffineConstantExpr>(dim))
          continue;
        dimSizes[::llvm::cast<AffineDimExpr>(dim).getPosition()] = size;
      }
      return dimSizes;
    }
  }];
}

#endif // MLIR_INTERFACES_VECTORINTERFACES
