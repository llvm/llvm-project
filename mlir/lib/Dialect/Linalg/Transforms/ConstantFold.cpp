//===- ConstantFold.cpp - Implementation of constant folding on Linalg ops ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements constant folding on Linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
/// Base class for constant folding linalg.generic ops with N inputs, 1 output,
/// and permutation indexing maps.
///
/// `ConcreteType` should provide methods with signatures
///
/// ```c++
///   bool matchIndexingMaps(GenericOp genericOp) const;
///   RegionComputationFn getRegionComputeFn(GenericOp) const;
/// ```
///
/// The latter inspects the region and returns the computation inside as a
/// functor. The functor will be invoked with constant elements for all inputs
/// and should return the corresponding computed constant element for output.
template <typename ConcreteType>
class FoldConstantBase : public OpRewritePattern<GenericOp> {
public:
  struct APIntOrFloat {
    Optional<APInt> apInt;
    Optional<APFloat> apFloat;
  };
  struct APIntOrFloatArray {
    SmallVector<APInt> apInts;
    SmallVector<APFloat> apFloats;
  };
  using RegionComputationFn =
      std::function<APIntOrFloat(const APIntOrFloatArray &)>;

  FoldConstantBase(MLIRContext *context, const ControlFusionFn &controlFn,
                   PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit), controlFn(controlFn) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (genericOp.hasBufferSemantics())
      return failure();

    // Only support ops generating one output for now.
    if (genericOp.getNumOutputs() != 1)
      return failure();

    auto outputType = genericOp.getResultTypes().front().dyn_cast<ShapedType>();
    // Require the output types to be static given that we are generating
    // constants.
    if (!outputType || !outputType.hasStaticShape())
      return failure();

    if (!llvm::all_of(genericOp.getInputOperands(), [](OpOperand *operand) {
          return operand->get().getType().isa<ShapedType>();
        }))
      return failure();

    // Make sure all element types are the same.
    auto getOperandElementType = [](OpOperand *operand) {
      return operand->get().getType().cast<ShapedType>().getElementType();
    };
    if (!llvm::is_splat(llvm::map_range(genericOp.getInputAndOutputOperands(),
                                        getOperandElementType)))
      return failure();

    // We can only handle the case where we have int/float elements.
    auto elementType = outputType.getElementType();
    if (!elementType.isIntOrFloat())
      return failure();

    // Require all indexing maps to be permutations for now. This is common and
    // it simplifies input/output access greatly: we can do the data shuffling
    // entirely in the compiler, without needing to turn all indices into
    // Values, and then do affine apply on them, and then match back the
    // constant again.
    if (!llvm::all_of(genericOp.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isPermutation(); }))
      return failure();

    for (OpOperand *operand : genericOp.getOutputOperands()) {
      if (genericOp.payloadUsesValueFromOperand(operand))
        return failure();
    }

    // Further check the indexing maps are okay for the ConcreteType.
    if (!static_cast<const ConcreteType *>(this)->matchIndexingMaps(genericOp))
      return failure();

    // Defer to the concrete type to check the region and discover the
    // computation inside.
    RegionComputationFn computeFn =
        static_cast<const ConcreteType *>(this)->getRegionComputeFn(genericOp);
    if (!computeFn)
      return failure();

    // All inputs should be constants.
    int numInputs = genericOp.getNumInputs();
    SmallVector<DenseIntOrFPElementsAttr> inputValues(numInputs);
    for (const auto &operand : llvm::enumerate(genericOp.getInputOperands())) {
      if (!matchPattern(operand.value()->get(),
                        m_Constant(&inputValues[operand.index()])))
        return failure();
    }

    // Identified this as a potential candidate for folding. Now check the
    // policy to see whether we are allowed to proceed.
    for (int i = 0; i < numInputs; ++i) {
      OpOperand *consumer = genericOp.getInputOperand(i);
      OpResult producer = consumer->get().cast<OpResult>();
      if (!controlFn(producer, *consumer))
        return failure();
    }

    auto linalgOp = cast<LinalgOp>(genericOp.getOperation());
    SmallVector<int64_t, 4> loopBounds = linalgOp.computeStaticLoopSizes();
    int64_t numElements = outputType.getNumElements();

    // Use APInt/APFloat instead of Attribute here for constructing the output.
    // This helps to avoid blowing up compiler memory usage: Attributes would
    // unify the following cases but they have lifetime as the MLIRContext.
    SmallVector<APInt> intOutputValues;
    SmallVector<APFloat> fpOutputValues;
    if (elementType.template isa<FloatType>())
      fpOutputValues.resize(numElements, APFloat(0.f));
    else
      intOutputValues.resize(numElements);

    // Return the constant dim positions from the given permutation map.
    auto getDimPositions = [](AffineMap map) {
      SmallVector<unsigned> dims;
      dims.reserve(map.getNumResults());
      for (AffineExpr result : map.getResults()) {
        dims.push_back(result.cast<AffineDimExpr>().getPosition());
      }
      return dims;
    };

    SmallVector<SmallVector<unsigned>> inputDims;
    for (int i = 0; i < numInputs; ++i)
      inputDims.push_back(getDimPositions(genericOp.getIndexingMapsArray()[i]));
    auto outputDims = getDimPositions(genericOp.getIndexingMapsArray().back());
    auto outputShape = outputType.getShape();

    // Allocate small vectors for index delinearization. Initial values do not
    // matter here as they will be overwritten later.
    SmallVector<uint64_t> indices(loopBounds.size(), 0);
    SmallVector<uint64_t> dstIndices(loopBounds.size(), 0);
    SmallVector<SmallVector<uint64_t>> srcIndices(
        numInputs, SmallVector<uint64_t>(loopBounds.size(), 0));
    SmallVector<uint64_t> srcLinearIndices(numInputs, 0);
    uint64_t dstLinearIndex = 0;

    // Allocate spaces for compute function inputs. Initial values do not matter
    // here as they will be overwritten later.
    APIntOrFloatArray computeFnInputs;

    auto inputShapes = llvm::to_vector<4>(
        llvm::map_range(genericOp.getInputOperands(), [](OpOperand *operand) {
          return operand->get().getType().cast<ShapedType>().getShape();
        }));

    // Given a `linearIndex`, remap it to a linear index to access linalg op
    // inputs/ouputs. This mutates `indices`, `srcIndices`, `dstIndices`,
    // `srcLinearIndices`, `dstLinearIndex` in place.
    auto computeRemappedLinearIndex = [&](int linearIndex) {
      int totalCount = linearIndex;
      for (int dim = loopBounds.size() - 1; dim >= 0; --dim) {
        indices[dim] = totalCount % loopBounds[dim];
        totalCount /= loopBounds[dim];
      }

      for (int dim = loopBounds.size() - 1; dim >= 0; --dim) {
        for (int i = 0; i < numInputs; ++i)
          srcIndices[i][dim] = indices[inputDims[i][dim]];
        dstIndices[dim] = indices[outputDims[dim]];
      }

      dstLinearIndex = dstIndices.front();
      for (int i = 0; i < numInputs; ++i)
        srcLinearIndices[i] = srcIndices[i].front();

      for (int dim = 1; dim < outputType.getRank(); ++dim) {
        dstLinearIndex = dstLinearIndex * outputShape[dim] + dstIndices[dim];
        for (int i = 0; i < numInputs; ++i)
          srcLinearIndices[i] =
              srcLinearIndices[i] * inputShapes[i][dim] + srcIndices[i][dim];
      }
    };

    bool isFloat = elementType.isa<FloatType>();
    if (isFloat) {
      SmallVector<DenseElementsAttr::iterator_range<APFloat>> inFpRanges;
      for (int i = 0; i < numInputs; ++i)
        inFpRanges.push_back(inputValues[i].getValues<APFloat>());

      computeFnInputs.apFloats.resize(numInputs, APFloat(0.f));

      // Transpose the input constant. Because we don't know its rank in
      // advance, we need to loop over the range [0, element count) and
      // delinearize the index.
      for (int linearIndex = 0; linearIndex < numElements; ++linearIndex) {
        computeRemappedLinearIndex(linearIndex);

        // Collect constant elements for all inputs at this loop iteration.
        for (int i = 0; i < numInputs; ++i)
          computeFnInputs.apFloats[i] = inFpRanges[i][srcLinearIndices[i]];

        // Invoke the computation to get the corresponding constant output
        // element.
        fpOutputValues[dstLinearIndex] = *computeFn(computeFnInputs).apFloat;
      }
    } else {
      SmallVector<DenseElementsAttr::iterator_range<APInt>> inIntRanges;
      for (int i = 0; i < numInputs; ++i)
        inIntRanges.push_back(inputValues[i].getValues<APInt>());

      computeFnInputs.apInts.resize(numInputs);

      // Transpose the input constant. Because we don't know its rank in
      // advance, we need to loop over the range [0, element count) and
      // delinearize the index.
      for (int linearIndex = 0; linearIndex < numElements; ++linearIndex) {
        computeRemappedLinearIndex(linearIndex);

        // Collect constant elements for all inputs at this loop iteration.
        for (int i = 0; i < numInputs; ++i)
          computeFnInputs.apInts[i] = inIntRanges[i][srcLinearIndices[i]];

        // Invoke the computation to get the corresponding constant output
        // element.
        intOutputValues[dstLinearIndex] = *computeFn(computeFnInputs).apInt;
      }
    }

    DenseElementsAttr outputAttr =
        isFloat ? DenseElementsAttr::get(outputType, fpOutputValues)
                : DenseElementsAttr::get(outputType, intOutputValues);

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(genericOp, outputAttr);
    return success();
  }

private:
  ControlFusionFn controlFn;
};

// Folds linalg.generic ops that are actually transposes on constant values.
struct FoldConstantTranspose : public FoldConstantBase<FoldConstantTranspose> {
  using FoldConstantBase::FoldConstantBase;

  bool matchIndexingMaps(GenericOp genericOp) const {
    // We should have one input and one output.
    return genericOp.getIndexingMapsArray().size() == 2;
  }

  RegionComputationFn getRegionComputeFn(GenericOp genericOp) const {
    // Make sure the region only contains a yield op.
    Block &body = genericOp.region().front();
    if (!llvm::hasSingleElement(body))
      return nullptr;
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yieldOp)
      return nullptr;

    // The yield op should return the block argument corresponds to the input.
    for (Value yieldVal : yieldOp.values()) {
      auto yieldArg = yieldVal.dyn_cast<BlockArgument>();
      if (!yieldArg || yieldArg.getOwner() != &body)
        return nullptr;
      if (yieldArg.getArgNumber() != 0)
        return nullptr;
    }

    // No computation; just return the orginal value.
    return [](const APIntOrFloatArray &inputs) {
      if (inputs.apFloats.empty())
        return APIntOrFloat{inputs.apInts.front(), llvm::None};
      return APIntOrFloat{llvm::None, inputs.apFloats.front()};
    };
  }

  ControlFusionFn controlFn;
};
} // namespace

void mlir::linalg::populateConstantFoldLinalgOperations(
    RewritePatternSet &patterns, const ControlFusionFn &controlFn) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<FoldConstantTranspose>(context, controlFn);
}
