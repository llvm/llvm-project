//===- LinalgInterfaces.td - Linalg Interfaces Declaration -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the structured interface sfor Linalg ops.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_IR_LINALGINTERFACES
#define LINALG_IR_LINALGINTERFACES

include "mlir/IR/OpBase.td"

// The 'LinalgContractionOpInterface' provides access to the
// 'ContractionOpInterface'.
def LinalgContractionOpInterface : OpInterface<"ContractionOpInterface"> {
  let description = [{
   A Linalg contraction is defined in general terms:
     1. Has 2 input and 1 output shapes.
     2. Has at least one reduction dimension.
     3. Has only projected permutation indexing maps.
     4. its body computes `u5(u1(c) + u2(u3(a) * u4(b)))` on some field
     (AddOpType, MulOpType), where u1, u2, u3, u4 and u5 represent scalar unary
     operations that may change the type (e.g. for mixed-precision).
   As a consequence, when vectorization of such an op occurs, the only special
   behavior is that the (unique) MulOpType is vectorized into a
   `vector.contract`. All other ops are handled in a generic fashion.
   In the future, we may wish to allow more input arguments and elementwise and
   constant operations that do not involve the reduction dimension(s).
  }];
  let cppNamespace = "::mlir::linalg";
  let verify = [{ return detail::verifyContractionInterface($_op); }];
  let verifyWithRegions = 1;
  let methods = [
    InterfaceMethod<
    /*desc=*/"Returns the left-hand side operand.",
    /*retTy=*/"Value",
    /*methodName=*/"lhs",
    /*args=*/(ins),
    /*methodBody=*/[{
      return $_op.getOperation()->getOperand(0);
    }]>,
    InterfaceMethod<
    /*desc=*/"Returns the right-hand side operand.",
    /*retTy=*/"Value",
    /*methodName=*/"rhs",
    /*args=*/(ins),
    /*methodBody=*/[{
      return $_op.getOperation()->getOperand(1);
    }]>,
    InterfaceMethod<
    /*desc=*/[{
      Returns whether the given op has indexing maps that correspond to a
      row-major matmul operation.
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isRowMajorMatmul",
    /*args=*/(ins),
    /*methodBody=*/[{
        return mlir::isRowMajorMatmul($_op.getIndexingMaps());
    }]>,
    InterfaceMethod<
    /*desc=*/[{
      Returns whether the given op has indexing maps that correspond to a
      column-major matmul operation.
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isColumnMajorMatmul",
    /*args=*/(ins),
    /*methodBody=*/[{
        return mlir::isColumnMajorMatmul($_op.getIndexingMaps());
    }]>,
    InterfaceMethod<
    /*desc=*/[{
      Returns whether the given op has indexing maps that correspond to a
      row-major batch matmul operation.
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isRowMajorBatchMatmul",
    /*args=*/(ins),
    /*methodBody=*/[{
        return mlir::isRowMajorBatchMatmul($_op.getIndexingMaps());
    }]>,
  ];
}

def LinalgConvolutionOpInterface : OpInterface<"ConvolutionOpInterface"> {
  let description = [{
    A convolution is defined in general terms:
    1. Has an `image` and a `filter` operand.
    2. Has one `output` operand.
    3. The indexing maps of the input have expressions that satisfy
    ```
       AffineExpr ::== AffineDimExpr | ConvolvedExpr
       ConvolvedExpr ::== MulExpr (`+` MulExpr)+
       MulExpr ::== AffineDimExpr (`*` (AffineConstantExpr | AffineSymbolExpr))?
    ```
    4. The filter and the output have projected permutation maps.
    5. Each of the loops can be qualified as one of,
       - Loop over batch dimension,
       - Loop over output image dimensions,
       - Loop over output channel dimensions,
       - Loop over convolved filter dimensions,
       - Loop over input channel dimension.
  }];
  let cppNamespace = "::mlir::linalg";
  let verify = [{ return detail::verifyConvolutionInterface($_op); }];
  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the image operand.",
      /*retTy=*/"Value",
      /*methodName=*/"image",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOperation()->getOperand(0);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the filter operand.",
      /*retTy=*/"Value",
      /*methodName=*/"filter",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOperation()->getOperand(1);
      }]
    >,
  ];
}

def LinalgFillOpInterface : OpInterface<"FillOpInterface"> {
  let description = [{
    A fill operation is defined in general terms:
    1. Has a scalar `value` operand.
    2. Has one `output` operand.
  }];
  let cppNamespace = "::mlir::linalg";
  let verify = [{ return detail::verifyFillInterface($_op); }];
  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the fill value.",
      /*retTy=*/"Value",
      /*methodName=*/"value",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOperation()->getOperand(0);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the output operand.",
      /*retTy=*/"Value",
      /*methodName=*/"output",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOperation()->getOperand(1);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the result.",
      /*retTy=*/"Value",
      /*methodName=*/"result",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if ($_op.getOperation()->getResults().empty())
          return nullptr;
        return $_op.getOperation()->getResults().front();
      }]
    >,
  ];
}

// The 'LinalgStructuredInterface' provides access to the 'LinalgOp' interface.
def LinalgStructuredInterface : OpInterface<"LinalgOp"> {
  let cppNamespace = "::mlir::linalg";
  let methods = [
    //===------------------------------------------------------------------===//
    // Loop types handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the number of parallel loops.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumParallelLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumIterators(getParallelIteratorTypeName(),
                               $_op.iterator_types());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are parallel loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getParallelDims",
      /*args=*/(ins "SmallVectorImpl<unsigned> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return findPositionsOfType($_op.iterator_types(),
                                   getParallelIteratorTypeName(), res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of reduction loops.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumReductionLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumIterators(getReductionIteratorTypeName(),
                               $_op.iterator_types());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are reduction loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getReductionDims",
      /*args=*/(ins "SmallVectorImpl<unsigned> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return findPositionsOfType($_op.iterator_types(),
                                   getReductionIteratorTypeName(), res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of window loops.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumWindowLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumIterators(getWindowIteratorTypeName(),
                               $_op.iterator_types());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are window loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getWindowDims",
      /*args=*/(ins "SmallVectorImpl<unsigned> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return findPositionsOfType($_op.iterator_types(),
                                   getWindowIteratorTypeName(), res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the total number of loops within the current operation.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumIterators($_op.iterator_types());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the current operation has only one loop and it's a
        reduction loop.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasSingleReductionLoop",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto iters = $_op.iterator_types();
        return iters.size() == 1 &&
               getNumIterators(getReductionIteratorTypeName(), iters) == 1;
      }]>,
    //===------------------------------------------------------------------===//
    // Input and Output arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return true if the payload uses the value loaded from `opOperand`. This
        is useful to avoid loading from "write-only" memory that may be
        uninitialized, as well as properly cloning "read-write" operands.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"payloadUsesValueFromOperand",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        unsigned bbArgNumber = opOperand->getOperandNumber();
        // Init tensors have uses.
        return !getBlock()->getArgument(bbArgNumber).use_empty();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if `opOperand` is an init tensor. This is true when it is
        an output tensor operand whose value is used in the payload region.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isInitTensor",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (!$_op.isOutputTensor(opOperand))
          return false;
        return payloadUsesValueFromOperand(opOperand);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` rank or zero for scalars.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getRank",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getRank();
        return 0;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the output block arguments of the region.
      }],
      /*retTy=*/"Block::BlockArgListType",
      /*methodName=*/"getRegionOutputArgs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        return getBlock()->getArguments().take_back(
            cast<DestinationStyleOpInterface>(*this->getOperation())
                .getNumOutputs());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` shape or an empty vector for scalars.
      }],
      /*retTy=*/"ArrayRef<int64_t>",
      /*methodName=*/"getShape",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getShape();
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the block argument for an `opOperand`.
      }],
      /*retTy=*/"BlockArgument",
      /*methodName=*/"getTiedBlockArgument",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        return getBlock()->getArgument(opOperand->getOperandNumber());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the operand for a `blockArgument`.
      }],
      /*retTy=*/"OpOperand *",
      /*methodName=*/"getTiedOpOperand",
      /*args=*/(ins "BlockArgument":$blockArgument),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(blockArgument.getOwner() == getBlock());
        return &this->getOperation()->getOpOperand(
            blockArgument.getArgNumber());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input or output indexing map for `opOperand`.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getTiedIndexingMap",
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
        Return the indexing map for a `result`.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getTiedIndexingMapForResult",
      /*args=*/(ins "OpResult":$result),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(result.getOwner() == this->getOperation());
        auto indexingMaps =
          $_op.getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        // MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        return *(indexingMaps.begin() +
                 cast<DestinationStyleOpInterface>(*this->getOperation())
                     .getNumInputs() +
                 result.getResultNumber());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the value yielded by the region corresponding to an output
        `opOperand`.
      }],
      /*retTy=*/"OpOperand *",
      /*methodName=*/"getTiedYieldValue",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        // MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        int64_t resultIndex =
            opOperand->getOperandNumber() -
            cast<DestinationStyleOpInterface>(*this->getOperation())
                .getNumInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults());
        Operation *yieldOp = getBlock()->getTerminator();
        return &yieldOp->getOpOperand(resultIndex);
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the single block constituting the body of the operation by
        calling the getBody method on the concrete operation.
      }],
      /*retTy=*/"Block*",
      /*methodName=*/"getBlock",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Assume the concrete operation implements the
        // SingleBlockImplicitTerminator trait.
        return $_op.getBody();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the iterator types attribute within the current operation.
      }],
      /*retTy=*/"ArrayAttr",
      /*methodName=*/"iterator_types",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getIteratorTypes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return iterator types in the current operation.
      }],
      /*retTy=*/"SmallVector<StringRef>",
      /*methodName=*/"getIteratorTypesArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto range = $_op.iterator_types().template getAsValueRange<StringAttr>();
        return {range.begin(), range.end()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if the indexing map is depending on the current op instance.
        This means that the indexing map is dynamically synthesized by using the
        op instance's concrete attributes, instead of being static for all
        instances of the same op kind.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasDynamicIndexingMaps",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return false; }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Verify all attributes used by indexing maps are valid.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"verifyIndexingMapRequiredAttributes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return success(); }]
    >,
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
        Return true if any of the operands has a dynamic shape.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasDynamicShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::any_of(getStaticShape(), ShapedType::isDynamic);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the name registered for this op when lowering to an external
        library call.
      }],
      /*retTy=*/"std::string",
      /*methodName=*/"getLibraryCallName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getLibraryCallName();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
         Return whether the op accesses the iteration indices.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasIndexSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return op operands that have a corresponding argument in the basic block.
        By default, the block should have an argument for each operand, but there
        are expection. For example, in `map` output operand isn't used in
        the block.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getOpOperandsMatchingBBargs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getInputAndOutputOperands();
      }]
    >,
    //===------------------------------------------------------------------===//
    // Linalg generalization hooks.
    //===------------------------------------------------------------------===//
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
        return concatAffineMaps(maps);
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
        return inversePermutation(getLoopsToShapesMap());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Checks if the given operands can be dropped, and the remaining
        operands can still compute the bounds of the op.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"canOpOperandsBeDropped",
      /*args=*/(ins "ArrayRef<OpOperand *>":$droppedOperands),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return detail::canOpOperandsBeDroppedImpl($_op, droppedOperands);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Like `getShape`, but only returns statically-known information, without
        generating any new IR. For each shape dimension, returns >=0 if that
        dimension is statically known, or ShapeType::kDynamicSize otherwise.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getStaticShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<int64_t> res;
        // MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        auto iface = cast<DestinationStyleOpInterface>(*this->getOperation());
        for (OpOperand *opOperand : iface.getInputAndOutputOperands())
          llvm::append_range(res, getShape(opOperand));
        return res;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the statically-known loop ranges. Composes
        `getShapesToLoopsMap()` with the result of `getStaticShape`.
        Returns ShapeType::kDynamicSize for non-statically-known loop ranges.
        This is expected to be called by a valid Linalg op
      }],
      /*retTy=*/"SmallVector<int64_t, 4>",
      /*methodName=*/"getStaticLoopRanges",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<int64_t> viewSizes = getStaticShape();
        AffineMap invertedMap = getShapesToLoopsMap();
        assert(invertedMap && "expected a valid Linalg op to call the method");
        return invertedMap.compose(viewSizes);
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other static interface methods.
    //===------------------------------------------------------------------===//
    StaticInterfaceMethod<
      /*desc=*/[{
        Returns the region builder for constructing the body for linalg.generic.
        Returns a null function if this named op does not define a region
        builder.
      }],
      /*retTy=*/"std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)>",
      /*methodName=*/"getRegionBuilder",
      (ins),
      [{ return ConcreteOp::getRegionBuilder(); }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if all the indexing maps are projected permutations.
        Otherwise return false.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasOnlyProjectedPermutations",
      (ins),
      [{
        return llvm::all_of($_op.getIndexingMapsArray(),
                            [](AffineMap map) { return map.isProjectedPermutation(); });
      }]
    >
  ];

  let extraClassDeclaration = [{
    /// Return the flat list of all operand dimension sizes in the order they
    /// appear in the operands.
    SmallVector<OpFoldResult> createFlatListOfOperandDims(OpBuilder &, Location);

    /// Return the flat list of all operands' static dimension sizes in the
    /// order they appear in the operands. All operand dimension sizes have to
    /// be statically known.
    SmallVector<int64_t, 4> createFlatListOfOperandStaticDims();

    /// Create the loop ranges to materialize the computation over the current
    /// operands. This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandDims`.
    SmallVector<Range, 4> createLoopRanges(OpBuilder &b, Location loc);

    /// Compute the static loop sizes necessary to vectorize the computation.
    /// This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandStaticDims`.
    SmallVector<int64_t, 4> computeStaticLoopSizes();

    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible
    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes);

    // TODO: Remove once prefixing is flipped.
    ArrayAttr getIteratorTypes() { return iterator_types(); }

    SmallVector<StringRef> getIteratorTypeNames() {
      return llvm::to_vector(getIteratorTypes().getAsValueRange<StringAttr>());
    }

    //========================================================================//
    // Forwarding functions to access interface methods from the
    // DestinationStyleOpInterface.
    // MLIR currently does not support dependent interfaces or interface
    // inheritance. By construction all ops with StructuredOpInterface must
    // implement DestinationStyleOpInterface.
    // TODO: reevalute the need for a cast when a better mechanism exists.
    //========================================================================//

    ValueRange inputs() {
      return cast<DestinationStyleOpInterface>(*this->getOperation()).inputs();
    }

    int64_t getNumInputs() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getNumInputs();
    }

    ValueRange outputs() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputs();
    }

    int64_t getNumOutputs() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getNumOutputs();
    }

    int64_t getNumInputsAndOutputs() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getNumInputsAndOutputs();
    }

    OpOperandVector getInputOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getInputOperands();
    }

    OpOperand *getInputOperand(int64_t i) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getInputOperand(i);
    }

    OpOperandVector getInputBufferOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getInputBufferOperands();
    }

    OpOperandVector getInputTensorOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getInputTensorOperands();
    }

    OpOperandVector getOutputOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputOperands();
    }

    OpOperand *getOutputOperand(int64_t i) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputOperand(i);
    }

    void setOutputOperand(int64_t i, Value value) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .setOutputOperand(i, value);
    }

    OpOperandVector getOutputBufferOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputBufferOperands();
    }

    OpOperandVector getOutputTensorOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputTensorOperands();
    }

    SmallVector<MemRefType> getOutputBufferTypes() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputBufferTypes();
    }

    SmallVector<RankedTensorType> getOutputTensorTypes() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputTensorTypes();
    }

    OpOperandVector getInputAndOutputOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getInputAndOutputOperands();
    }

    bool isInputTensor(OpOperand *opOperand) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .isInputTensor(opOperand);
    }

    bool isOutputTensor(OpOperand *opOperand) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .isOutputTensor(opOperand);
    }

    bool isScalar(OpOperand *opOperand) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .isScalar(opOperand);
    }

    OpResult getTiedOpResult(OpOperand *opOperand) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getTiedOpResult(opOperand);
    }

    bool hasBufferSemantics() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .hasBufferSemantics();
    }

    bool hasTensorSemantics() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .hasTensorSemantics();
    }

    Operation *clone(OpBuilder & b, Location loc, TypeRange resultTypes,
                     ValueRange operands) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .clone(b, loc, resultTypes, operands);
    }

    Operation *cloneWithoutRegions(OpBuilder & b, Location loc,
                                   TypeRange resultTypes, ValueRange operands) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .cloneWithoutRegions(b, loc, resultTypes, operands);
    }

    //========================================================================//
    // Helper functions to mutate the `operand_segment_sizes` attribute.
    // These are useful when cloning and changing operand types.
    //========================================================================//
    void setNumInputs(unsigned num) { setOperandSegmentAt(0, num); }
    void setNumOutputBuffers(unsigned num) { setOperandSegmentAt(1, num); }

    private:
    void setOperandSegmentAt(unsigned idx, unsigned val) {
      auto attr = (*this)->getAttr("operand_segment_sizes")
        .cast<DenseIntElementsAttr>();
      unsigned i = 0;
      auto newAttr = attr.mapValues(IntegerType::get(getContext(), 32),
        [&](const APInt &v) { return (i++ == idx) ? APInt(32, val) : v; });
      getOperation()->setAttr("operand_segment_sizes", newAttr);
    }
  }];

  let verify = [{ return detail::verifyStructuredOpInterface($_op); }];
  let verifyWithRegions = 1;
}

// The 'DestinationStyleOpInterface' provides access to the methods relevant
// for destination-style ops. A destination-style operation has 'n' input
// arguments and 'm' output arguments. Each op that wants to implement
// DestinationStyleOpInterface needs to define inputs() and getOutputs()
// methods.
def DestinationStyleOpInterface : OpInterface<"DestinationStyleOpInterface"> {
  let cppNamespace = "::mlir::linalg";
  let methods = [
    //===------------------------------------------------------------------===//
    // Num input/output arguments handling.
    //===------------------------------------------------------------------===//
    // `inputs` must be defined by each op that wants to implement the
    // DestinationStyleOpInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the input shape operands.
      }],
      /*retTy=*/"ValueRange",
      /*methodName=*/"inputs",
      /*args=*/(ins)
    >,
    // These special methods rely on `inputs` and `outputs` being defined by
    // each op that wants to implement the DestinationStyleOpInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumInputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getInputs().size();
      }]
    >,
    // `getOutputs` must be defined by each op that wants to implement the
    // DestinationStyleOpInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the output shape operands.
      }],
      /*retTy=*/"ValueRange",
      /*methodName=*/"getOutputs",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of outputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOutputs().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs and outputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumInputsAndOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return this->getOperation()->getNumOperands();
      }]
    >,
    //===------------------------------------------------------------------===//
    // Input operands handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the input operands.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getInputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t numInputs = getNumInputs();
        OpOperandVector result;
        result.reserve(numInputs);
        llvm::transform(
          this->getOperation()->getOpOperands().take_front(numInputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th input operand.
      }],
      /*retTy=*/"OpOperand*",
      /*methodName=*/"getInputOperand",
      /*args=*/(ins "int64_t":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i >= 0 && i < getNumInputs());
        return &this->getOperation()->getOpOperand(i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of buffer type.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getInputBufferOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        OpOperandVector result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<MemRefType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of tensor type.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getInputTensorOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        OpOperandVector result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
        return result;
      }]
    >,
    //===------------------------------------------------------------------===//
    // Output operands handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the output operands.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getOutputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t numOutputs = getNumOutputs();
        OpOperandVector result;
        result.reserve(numOutputs);
        llvm::transform(
          this->getOperation()->getOpOperands()
            .take_back(numOutputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th output operand.
      }],
      /*retTy=*/"OpOperand*",
      /*methodName=*/"getOutputOperand",
      /*args=*/(ins "int64_t":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i >= 0 && i < getNumOutputs());
        return &this->getOperation()->getOpOperand(getNumInputs() + i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Set the `i`-th output operand.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setOutputOperand",
      /*args=*/(ins "int64_t":$i, "Value":$value),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i >= 0 && i < getNumOutputs());
        this->getOperation()->setOperand(getNumInputs() + i, value);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of buffer type.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getOutputBufferOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        OpOperandVector result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<MemRefType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of tensor type.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getOutputTensorOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        OpOperandVector result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the types of the subset of output operands that are of buffer type.
      }],
      /*retTy=*/"SmallVector<MemRefType>",
      /*methodName=*/"getOutputBufferTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<MemRefType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputBufferOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return opOperands->get().getType().cast<MemRefType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the types of the subset of output operands that are of tensor type.
      }],
      /*retTy=*/"SmallVector<RankedTensorType>",
      /*methodName=*/"getOutputTensorTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<RankedTensorType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputTensorOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return opOperands->get().getType().cast<RankedTensorType>();
          });
        return result;
      }]
    >,
    //===------------------------------------------------------------------===//
    // Input and Output arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the range over input and output operands.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getInputAndOutputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t numInputsAndOutputs = getNumInputsAndOutputs();
        OpOperandVector result;
        result.reserve(numInputsAndOutputs);
        llvm::transform(
          this->getOperation()->getOpOperands(),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if `opOperand` is an input tensor.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isInputTensor",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() < $_op.getNumInputs())
          return true;
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if `opOperand` is an output tensor.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isOutputTensor",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() >= $_op.getNumInputs())
          return true;
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if the `opOperand` is a scalar value.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isScalar",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        return !opOperand->get().getType().template isa<ShapedType>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the result tied to `opOperand`.
      }],
      /*retTy=*/"OpResult",
      /*methodName=*/"getTiedOpResult",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        int64_t resultIndex = opOperand->getOperandNumber() - getNumInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults() );
        return this->getOperation()->getResult(resultIndex);
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op has only MemRef input and outputs.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasBufferSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return this->getOperation()->getNumResults() == 0 &&
          llvm::all_of(this->getOperation()->getOpOperands(),
            [&](OpOperand &opOperand) {
              return isScalar(&opOperand) ||
                     opOperand.get().getType().template isa<MemRefType>();
            });
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op has only RankedTensor input and outputs.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasTensorSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::all_of(this->getOperation()->getOpOperands(),
          [&](OpOperand &opOperand) {
            return isScalar(&opOperand) ||
                   opOperand.get().getType().template isa<RankedTensorType>();
          });
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other static interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Clone the current operation with the given location and operands. This
        is used to abstract away the optional underlying region creation. This
        does not change the balance between input, output_buffer and
        init_tensors operands.
      }],
      /*retTy=*/"Operation *",
      /*methodName=*/"clone",
      (ins "OpBuilder &":$b, "Location":$loc, "TypeRange":$resultTypes,
           "ValueRange":$operands),
      [{
        BlockAndValueMapping bvm;
        OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          $_op->getAttrs());
        for (Region &r : $_op->getRegions())
          r.cloneInto(state.addRegion(), bvm);
        return b.create(state);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Clone the current operation with the given location, operands
        and BlockAndValueMapping but leave the regions empty. This is
        used to abstract away the optional underlying region creation.
        This does not change the balance between input, output_buffer
        and init_tensors operands.
      }],
      /*retTy=*/"Operation *",
      /*methodName=*/"cloneWithoutRegions",
      (ins "OpBuilder &":$b, "Location":$loc, "TypeRange":$resultTypes,
           "ValueRange":$operands),
      [{
        OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          $_op->getAttrs());
        for (size_t cnt = 0, e = $_op->getNumRegions(); cnt < e; ++cnt)
          state.addRegion();
        return b.create(state);
      }]
    >
  ];

  let verify = [{ return detail::verifyDestinationStyleOpInterface($_op); }];
  let verifyWithRegions = 1;
}

#endif // LINALG_IR_LINALGINTERFACES
