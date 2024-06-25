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
#include <optional>

using namespace mlir;
using namespace mlir::linalg;

namespace {
/// Base class for constant folding linalg structured ops with N inputs, 1
/// output, and permutation indexing maps.
///
/// `ConcreteType` should provide methods with signatures
///
/// ```c++
///   bool matchIndexingMaps(LinalgOp linalgOp) const;
///   RegionComputationFn getRegionComputeFn(LinalgOp) const;
/// ```
///
/// The latter inspects the region and returns the computation inside as a
/// functor. The functor will be invoked with constant elements for all inputs
/// and should return the corresponding computed constant element for output.
template <typename ConcreteType>
class FoldConstantBase : public OpInterfaceRewritePattern<LinalgOp> {
public:
  struct APIntOrFloat {
    std::optional<APInt> apInt;
    std::optional<APFloat> apFloat;
  };
  struct APIntOrFloatArray {
    SmallVector<APInt> apInts;
    SmallVector<APFloat> apFloats;
  };
  using RegionComputationFn =
      std::function<APIntOrFloat(const APIntOrFloatArray &)>;

  FoldConstantBase(MLIRContext *context, const ControlFusionFn &controlFn,
                   PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
        controlFn(controlFn) {}

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // Mixed and buffer sematics aren't supported.
    if (!linalgOp.hasPureTensorSemantics())
      return failure();

    // Only support ops generating one output for now.
    if (linalgOp.getNumDpsInits() != 1)
      return failure();

    auto outputType = dyn_cast<ShapedType>(linalgOp->getResultTypes().front());
    // Require the output types to be static given that we are generating
    // constants.
    if (!outputType || !outputType.hasStaticShape())
      return failure();

    if (!llvm::all_of(linalgOp.getDpsInputs(), [](Value input) {
          return isa<ShapedType>(input.getType());
        }))
      return failure();

    // Make sure all element types are the same.
    auto getOperandElementType = [](Value value) {
      return cast<ShapedType>(value.getType()).getElementType();
    };
    if (!llvm::all_equal(
            llvm::map_range(linalgOp->getOperands(), getOperandElementType)))
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
    if (!llvm::all_of(linalgOp.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isPermutation(); }))
      return failure();

    for (OpOperand &operand : linalgOp.getDpsInitsMutable()) {
      if (linalgOp.payloadUsesValueFromOperand(&operand))
        return failure();
    }

    // Further check the indexing maps are okay for the ConcreteType.
    if (!static_cast<const ConcreteType *>(this)->matchIndexingMaps(linalgOp))
      return failure();

    // Defer to the concrete type to check the region and discover the
    // computation inside.
    RegionComputationFn computeFn =
        static_cast<const ConcreteType *>(this)->getRegionComputeFn(linalgOp);
    if (!computeFn)
      return failure();

    // All inputs should be constants.
    int numInputs = linalgOp.getNumDpsInputs();
    SmallVector<DenseIntOrFPElementsAttr> inputValues(numInputs);
    for (const auto &en : llvm::enumerate(linalgOp.getDpsInputOperands())) {
      if (!matchPattern(en.value()->get(),
                        m_Constant(&inputValues[en.index()])))
        return failure();
    }

    // Identified this as a potential candidate for folding. Now check the
    // policy to see whether we are allowed to proceed.
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      if (!controlFn(operand))
        return failure();
    }

    SmallVector<int64_t, 4> loopBounds = linalgOp.computeStaticLoopSizes();
    int64_t numElements = outputType.getNumElements();

    // Use APInt/APFloat instead of Attribute here for constructing the output.
    // This helps to avoid blowing up compiler memory usage: Attributes would
    // unify the following cases but they have lifetime as the MLIRContext.
    SmallVector<APInt> intOutputValues;
    SmallVector<APFloat> fpOutputValues;
    if (isa<FloatType>(elementType))
      fpOutputValues.resize(numElements, APFloat(0.f));
    else
      intOutputValues.resize(numElements);

    // Return the constant dim positions from the given permutation map.
    auto getDimPositions = [](AffineMap map) {
      SmallVector<unsigned> dims;
      dims.reserve(map.getNumResults());
      for (AffineExpr result : map.getResults()) {
        dims.push_back(cast<AffineDimExpr>(result).getPosition());
      }
      return dims;
    };

    SmallVector<SmallVector<unsigned>> inputDims;
    for (int i = 0; i < numInputs; ++i)
      inputDims.push_back(getDimPositions(linalgOp.getIndexingMapsArray()[i]));
    auto outputDims = getDimPositions(linalgOp.getIndexingMapsArray().back());
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
        llvm::map_range(linalgOp.getDpsInputs(), [](Value value) {
          return cast<ShapedType>(value.getType()).getShape();
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

    bool isFloat = isa<FloatType>(elementType);
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

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(linalgOp, outputAttr);
    return success();
  }

private:
  ControlFusionFn controlFn;
};

// Folds linalg.transpose (and linalg.generic ops that are actually transposes)
// on constant values.
struct FoldConstantTranspose : public FoldConstantBase<FoldConstantTranspose> {

  using FoldConstantBase::FoldConstantBase;

  bool matchIndexingMaps(LinalgOp linalgOp) const {
    // We should have one input and one output.
    return linalgOp.getIndexingMapsArray().size() == 2;
  }

  RegionComputationFn getRegionComputeFn(LinalgOp linalgOp) const {
    // Make sure the region only contains a yield op.
    Block &body = linalgOp->getRegion(0).front();
    if (!llvm::hasSingleElement(body))
      return nullptr;
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yieldOp)
      return nullptr;

    // The yield op should return the block argument corresponds to the input.
    for (Value yieldVal : yieldOp.getValues()) {
      auto yieldArg = dyn_cast<BlockArgument>(yieldVal);
      if (!yieldArg || yieldArg.getOwner() != &body)
        return nullptr;
      if (yieldArg.getArgNumber() != 0)
        return nullptr;
    }

    // No computation; just return the orginal value.
    return [](const APIntOrFloatArray &inputs) {
      if (inputs.apFloats.empty())
        return APIntOrFloat{inputs.apInts.front(), std::nullopt};
      return APIntOrFloat{std::nullopt, inputs.apFloats.front()};
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
