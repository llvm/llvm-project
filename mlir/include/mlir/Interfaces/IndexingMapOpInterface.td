//===- IndexingMapOpInterface.td - Interface Declaration -*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the IndexingMapOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INDEXING_MAP_OP_INTERFACE
#define MLIR_INTERFACES_INDEXING_MAP_OP_INTERFACE

include "mlir/IR/OpBase.td"

def IndexingMapOpInterface : OpInterface<"IndexingMapOpInterface"> {
  let description = [{
    Interface for operations that connect an iteration domain to operands via
    affine maps. Provides methods to access indexing maps between iteration
    domain and operand index spaces.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps attribute within the current operation.
      }],
      /*retTy=*/"ArrayAttr",
      /*methodName=*/"getIndexingMaps"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps within the current operation.
      }],
      /*retTy=*/"SmallVector<AffineMap>",
      /*methodName=*/"getIndexingMapsArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto range = $_op.getIndexingMaps()
          .template getAsValueRange<AffineMapAttr>();
        return {range.begin(), range.end()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input or output indexing map for `opOperand`.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getMatchingIndexingMap",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        auto indexingMaps =
          $_op.getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + opOperand->getOperandNumber());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Hook to provide a custom AffineMap used to compute all the operand
        subshapes given loop bounds. This is used to answer the question: "given
        an iteration space over the codomain, what are the subshapes of the
        operands involved in the computation".
        The default behavior is to just concatenate all the indexing maps.
        A custom AffineMap allows providing a map that can be used to
        compute subshapes even in cases where the concatenation of indexing maps
        (i.e. the data traversal order) is not a simple permutation of the loop
        traversal order. It is then possible to define ops with skewed data
        traversal order for which we can still easily compute hyperrectangular
        loop bounds and subviews.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getLoopsToShapesMap",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto maps =  $_op.getIndexingMapsArray();
        return concatAffineMaps(maps, $_op.getContext());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Hook to provide a custom AffineMap used to construct the
        hyperrectangular loop iteration space given all the operand subshapes.
        This is used to answer the question:
        "Given a list of operand ranges, what is the subportion of the iteration
        space involved in the computation".
        This is the inverse problem of `getLoopsToShapesMap`.
        Return the empty AffineMap when such an AffineMap cannot be constructed.
        The default behavior is based on a very simple inference procedure that
        only works with permutation affine maps.
        A more advanced Tensor-Comprehension like inference is possible but has
        proven to be ambiguous in unfavorable case.
        A safer and more robust alternative is to allow each op to define
        its own AffineMap.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getShapesToLoopsMap",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return inversePermutation($_op.getLoopsToShapesMap());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the static shape of the underlying operand (note this is
        op-specific behavior).
        Returns ShapedType::kDynamic for non-statically-known loop ranges.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getStaticOperandShape",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<int64_t> res;
        llvm::append_range(res, $_op.getShape(opOperand));
        return res;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns loop ranges by composing `getShapesToLoopsMap()` with the
        flattened list of operand shapes.
        Returns ShapedType::kDynamic for non-statically-known loop ranges.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getStaticLoopRanges",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<int64_t> allShapesSizes;
        for (OpOperand &opOperand : this->getOperation()->getOpOperands())
          llvm::append_range(allShapesSizes, $_op.getShape(&opOperand));
        AffineMap invertedMap = $_op.getShapesToLoopsMap();
        assert(invertedMap && "expected a valid op");
        return invertedMap.compose(allShapesSizes);
      }]
    >
  ];
  let extraClassDeclaration = [{
    // Verifier implementation for IndexingMapOpInterface.
    // This must be called manually as part of other verifiers so that the
    // verification order, and meaningful error messages, are not preempted.
    LogicalResult verifyImpl();
  }];
}

#endif // MLIR_INTERFACES_INDEXING_MAP_OP_INTERFACE
