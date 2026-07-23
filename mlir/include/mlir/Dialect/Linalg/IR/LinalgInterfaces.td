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

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/IndexingMapOpInterface.td"
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
    InterfaceMethod<
    /*desc=*/[{
      Returns whether the given op has indexing maps that correspond to a
      vector-matrix multiplication.
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isVecmat",
    /*args=*/(ins),
    /*methodBody=*/[{
        return mlir::isVecmat($_op.getIndexingMaps());
    }]>,
    InterfaceMethod<
    /*desc=*/[{
      Returns whether the given op has indexing maps that correspond to a
      batched vector-matrix multiplication.
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isBatchVecmat",
    /*args=*/(ins),
    /*methodBody=*/[{
        return mlir::isBatchVecmat($_op.getIndexingMaps());
    }]>,
    InterfaceMethod<
    /*desc=*/[{
      Returns whether the given op has indexing maps that correspond to a
      matrix-vector multiplication.
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isMatvec",
    /*args=*/(ins),
    /*methodBody=*/[{
        return mlir::isMatvec($_op.getIndexingMaps());
    }]>,
    InterfaceMethod<
    /*desc=*/[{
      Returns whether the given op has indexing maps that correspond to a
      batched matrix-vector multiplication.
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isBatchMatvec",
    /*args=*/(ins),
    /*methodBody=*/[{
        return mlir::isBatchMatvec($_op.getIndexingMaps());
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
def LinalgStructuredInterface
    : OpInterface<"LinalgOp", 
      [DestinationStyleOpInterface, IndexingMapOpInterface]
  > {
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
        return llvm::count($_op.getIteratorTypesArray(),
                           utils::IteratorType::parallel);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if all loops are parallel.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isAllParallelLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumParallelLoops() ==  getNumLoops();
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
        return findPositionsOfType($_op.getIteratorTypesArray(),
                                   utils::IteratorType::parallel, res);
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
        return llvm::count($_op.getIteratorTypesArray(),
                           utils::IteratorType::reduction);
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
        return findPositionsOfType($_op.getIteratorTypesArray(),
                                   utils::IteratorType::reduction, res);
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
        return $_op.getIteratorTypesArray().size();
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
        auto iters = $_op.getIteratorTypesArray();
        return iters.size() == 1 &&
               llvm::count(iters, utils::IteratorType::reduction) == 1;
      }]>,
    //===------------------------------------------------------------------===//
    // Input and Init arguments handling.
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
        Returns true only if linalgOp takes one input and produces one result.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isSingleInputOutput",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getNumDpsInputs() == 1 && $_op.getNumDpsInits() == 1;
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
        if (!$_op.isDpsInit(opOperand))
          return false;
        return payloadUsesValueFromOperand(opOperand);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` rank or zero for scalars or vectors not wrapped within a tensor or a memref.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getRank",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        Type t = opOperand->get().getType();
        // A VectorType is an elemental type, do not consider its rank for the operand.
        if (isa<VectorType>(t))
          return 0;
        // Tensor and Memref container types have a rank.
        if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
          // Failsafe.
          assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
                 "expected a ranked tensor or memref in LinalgInterface::getRank");
          return shapedType.getRank();
        }
        return 0;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input block arguments of the region.
      }],
      /*retTy=*/"Block::BlockArgListType",
      /*methodName=*/"getRegionInputArgs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getBlock()->getArguments().take_front($_op.getNumDpsInputs());
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
        return getBlock()->getArguments().take_back($_op.getNumDpsInits());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the block argument for an `opOperand`.
      }],
      /*retTy=*/"BlockArgument",
      /*methodName=*/"getMatchingBlockArgument",
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
      /*methodName=*/"getMatchingOpOperand",
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
        Return the indexing map for a `result`.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getIndexingMapMatchingResult",
      /*args=*/(ins "OpResult":$result),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(result.getOwner() == this->getOperation());
        auto indexingMaps =
          $_op.getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + $_op.getNumDpsInputs() +
                 result.getResultNumber());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the value yielded by the region corresponding to an output
        `opOperand`.
      }],
      /*retTy=*/"OpOperand *",
      /*methodName=*/"getMatchingYieldValue",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        int64_t resultIndex =
            opOperand->getOperandNumber() - $_op.getNumDpsInputs();
        assert(resultIndex >= 0 &&
               resultIndex < $_op.getNumDpsInits());
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
        Return iterator types in the current operation.

        Default implementation assumes that the operation has an attribute
        `iterator_types`, but it's not always the case. Sometimes iterator types
        can be infered from other parameters and in such cases default
        getIteratorTypesArray should be overriden.
      }],
      /*retTy=*/"SmallVector<utils::IteratorType>",
      /*methodName=*/"getIteratorTypesArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto range = $_op.getIteratorTypes()
                         .template getAsValueRange<IteratorTypeAttr,
                                                   utils::IteratorType>();
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
        Return true if any of the operands has a dynamic shape.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasDynamicShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        for (OpOperand &opOperand : this->getOperation()->getOpOperands()) {
          if (auto shapedType = dyn_cast<ShapedType>(opOperand.get().getType())) {
            if (ShapedType::isDynamicShape(shapedType.getShape())) return true;
          }
        }
        return false;
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
      /*retTy=*/"::llvm::SmallVector<OpOperand *>",
      /*methodName=*/"getOpOperandsMatchingBBargs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::llvm::SmallVector<OpOperand *> result;
        result.reserve($_op->getNumOperands());
        llvm::transform(
          this->getOperation()->getOpOperands(),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Given a dimension of the iteration space of a Linalg operation, finds an
        operand in the operation that is defined on such dimension. Returns
        whether such operand was found or not. If found, also returns the
        operand value and the dimension position within the operand.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"mapIterationSpaceDimToOperandDim",
      /*args=*/(ins "unsigned":$dimPos,
                    "::mlir::Value &":$operand,
                    "unsigned &":$operandDimPos),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Retrieve the operand and its dimension position from the first
        // operand with a permutation map that is defined on such dimension.
        for (auto [i, idxMap] : llvm::enumerate($_op.getIndexingMapsArray())) {
          if (idxMap.isProjectedPermutation()) {
            if (auto mayOperandDim = idxMap.getResultPosition(
                getAffineDimExpr(dimPos, idxMap.getContext()))) {
              operand = $_op->getOperand(i);
              operandDimPos = *mayOperandDim;
              return success();
            }
          }
        }

        return failure();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Given a dimension of the iteration space of a Linalg operation, finds
        all the operands in the operation that are defined on such dimension.
        Returns all the operand values found and their dimension positions in
        `operandDimPairs`.
      }],
      /*retTy=*/"void",
      /*methodName=*/"mapIterationSpaceDimToAllOperandDims",
      /*args=*/(ins "unsigned":$dimPos,
                    "mlir::SmallVectorImpl<std::pair<Value, unsigned>>&":$operandDimPairs),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        for (auto [i, idxMap] : llvm::enumerate($_op.getIndexingMapsArray())) {
          if (idxMap.isProjectedPermutation()) {
            if (auto mayOperandDim = idxMap.getResultPosition(
                getAffineDimExpr(dimPos, idxMap.getContext()))) {
              operandDimPairs.push_back({$_op->getOperand(i), *mayOperandDim});
            }
          }
        }

        return;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the user has supplied explicit indexing maps that are
        different from default indexing maps for this op. Returns `false` otherwise.
        Note, if the user define maps that are identical to the default maps,
        this method returns `false`.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasUserDefinedMaps",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return false; }]
    >,
    //===------------------------------------------------------------------===//
    // Linalg generalization hooks.
    //===------------------------------------------------------------------===//
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
    //===------------------------------------------------------------------===//
    // IndexingMapOpInterface interface methods implementation.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` shape or an empty vector for scalars or vectors
        not wrapped within a tensor or a memref.
      }],
      /*retTy=*/"ArrayRef<int64_t>",
      /*methodName=*/"getShape",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Type t = opOperand->get().getType();
        // A VectorType is an elemental type, do not consider its rank for the operand.
        if (isa<VectorType>(t))
          return {};
        if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
          // Failsafe.
          assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
                 "expected a ranked tensor or memref in LinalgInterface::getRank");
          return shapedType.getShape();
        }
        return {};
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
      /*retTy=*/"std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>, function_ref<InFlightDiagnostic()>)>",
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

    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible
    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes);

    /// Return the index in the indexingMaps vector that corresponds to this `opOperand`
    int64_t getIndexingMapIndex(OpOperand *opOperand);
  }];

  let verify = [{ return detail::verifyStructuredOpInterface($_op); }];
  let verifyWithRegions = 1;
}

def AggregatedOpInterface : OpInterface<"AggregatedOpInterface"> {
  let description = [{
    Interface for decomposing aggregated operations into a sequence of simpler
    ops.
  }];
  let cppNamespace = "::mlir::linalg";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Method to decompose the operation into simpler operations.

          On success, this method returns one `Value` per result in the
          original operation.
          The order of the returned values must match the order of the
          original values.
          In other words, the returned vector can be used directly with
          `RewriterBase::replaceOp(this, returnedValues)`.
        }],
        /*retType=*/"FailureOr<SmallVector<Value>>",
        /*methodName=*/"decomposeOperation",
        /*args=*/(ins
            "OpBuilder &":$b),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return {};
        }]
      >
  ];
}

#endif // LINALG_IR_LINALGINTERFACES
