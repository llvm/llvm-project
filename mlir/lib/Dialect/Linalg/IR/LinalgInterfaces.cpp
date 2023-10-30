//===- LinalgInterfaces.cpp - Linalg interfaces implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::linalg;

/// Include the definitions of the copy operation interface.
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Interface utility functions
//===----------------------------------------------------------------------===//
bool linalg::detail::canOpOperandsBeDroppedImpl(
    linalg::LinalgOp linalgOp, ArrayRef<OpOperand *> droppedOperands) {
  SmallVector<AffineMap> indexingMaps;
  for (auto &opOperand : linalgOp->getOpOperands()) {
    if (llvm::is_contained(droppedOperands, &opOperand))
      continue;
    indexingMaps.push_back(linalgOp.getMatchingIndexingMap(&opOperand));
  }
  if (indexingMaps.empty()) {
    // If there are no indexing maps, the operand can only be dropped
    // if the op has no loops.
    return linalgOp.getNumLoops() == 0;
  }
  return inversePermutation(concatAffineMaps(indexingMaps)) != AffineMap();
}

//===----------------------------------------------------------------------===//
// ContractionOpInterface implementation
//===----------------------------------------------------------------------===//

/// If the value is defined by a chain of unary side effect-free, go up the
/// use-def chain until the first value that isn't defined by such an op.
// TODO: relax to multi-operands with constants, which are technically unary ops
// as needed (e.g. add5).
static Value getSourceSkipUnary(Value value) {
  Operation *op = value.getDefiningOp();
  while (op && op->getNumOperands() == 1) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface || !iface.hasNoEffect())
      break;
    value = op->getOperand(0);
    op = value.getDefiningOp();
  }
  return value;
}

bool mlir::linalg::detail::isContractionBody(
    Block &block, function_ref<bool(Operation *, Operation *)> isaPair,
    llvm::raw_ostream &errs) {
  if (block.empty() || !block.back().mightHaveTrait<OpTrait::IsTerminator>()) {
    errs << "no terminator in the block";
    return false;
  }

  if (block.getNumArguments() != 3) {
    errs << "expected block with 3 arguments";
    return false;
  }

  Operation *terminator = block.getTerminator();
  if (terminator->getNumOperands() != 1) {
    errs << "expected terminator with 1 operand";
    return false;
  }

  Value yielded = getSourceSkipUnary(terminator->getOperand(0));
  Operation *reductionOp = yielded.getDefiningOp();
  if (reductionOp->getNumResults() != 1 || reductionOp->getNumOperands() != 2) {
    errs << "expected reduction op to be binary";
    return false;
  }

  Value reductionLHS = getSourceSkipUnary(reductionOp->getOperand(0));
  Value reductionRHS = getSourceSkipUnary(reductionOp->getOperand(1));

  if (reductionLHS != block.getArgument(2) &&
      reductionRHS != block.getArgument(2)) {
    errs << "expected reduction to take block argument #2 as one of the "
            "operands (modulo unary casts)";
    return false;
  }

  Value contributed = getSourceSkipUnary(
      isa<BlockArgument>(reductionLHS) ? reductionRHS : reductionLHS);
  Operation *elementwiseOp = contributed.getDefiningOp();
  if (elementwiseOp->getNumResults() != 1 ||
      elementwiseOp->getNumOperands() != 2) {
    errs << "expected elementwise op to be binary";
    return false;
  }

  if (!isaPair(elementwiseOp, reductionOp)) {
    errs << "expected reduction/elementwise op kind not satisfied";
    return false;
  }

  Value elementwiseLHS = getSourceSkipUnary(elementwiseOp->getOperand(0));
  Value elementwiseRHS = getSourceSkipUnary(elementwiseOp->getOperand(1));
  if ((elementwiseLHS == block.getArgument(0) &&
       elementwiseRHS == block.getArgument(1)) ||
      (elementwiseLHS == block.getArgument(1) &&
       elementwiseRHS == block.getArgument(0))) {
    return true;
  }

  errs << "expected elementwise op to apply to block arguments (modulo unary "
          "casts)";
  return false;
}

/// Returns true if the two operations are of the kinds specified by a pair of
/// consecutive template arguments.
template <typename AddOpTy, typename MulOpTy, typename... Args>
static bool isPairTemplateImpl(Operation *add, Operation *mul) {
  static_assert(sizeof...(Args) % 2 == 0,
                "expected an even number of template arguments");
  if (isa<AddOpTy>(add) && isa<MulOpTy>(mul))
    return true;

  if constexpr (sizeof...(Args) > 0)
    return isPairTemplateImpl<Args...>(add, mul);
  else
    return false;
}

/// Returns true if the block is a body of a contraction with the kinds of
/// operations given pairwise by template arguments.
template <typename... Args>
static bool isContractionBody(Block &block) {
  return linalg::detail::isContractionBody(block, &isPairTemplateImpl<Args...>);
}

/// Given a `linalgOp` and one of its `opOperand`, returns the positions of the
/// iterators of type `iter` that index the `opOperand` as a permutation.
/// This is useful to infer various subcomputations on a given `linalgOp`.
/// This is performed by looking up each result in the matching indexing map and
/// determining whether:
///   - It is a single AffineDimExpr.
///   - It is the only result involving this AffineDimExpr.
static llvm::SmallDenseSet<int64_t>
findPermutationsIndexingOperand(LinalgOp linalgOp, OpOperand *opOperand,
                                utils::IteratorType iter) {
  llvm::SmallDenseSet<int64_t> res;
  assert(linalgOp == opOperand->getOwner() && "expected linalgOp owner");
  AffineMap indexingMap = linalgOp.getMatchingIndexingMap(opOperand);
  for (AffineExpr e : indexingMap.getResults()) {
    if (auto d = e.dyn_cast<AffineDimExpr>()) {
      if (linalgOp.getIteratorTypesArray()[d.getPosition()] == iter &&
          llvm::count_if(indexingMap.getResults(), [d](AffineExpr e) {
            return e.isFunctionOfDim(d.getPosition());
          }) == 1)
        res.insert(d.getPosition());
    }
  }
  return res;
}

namespace {
auto par = utils::IteratorType::parallel;
auto red = utils::IteratorType::reduction;
} // namespace

/// Find 2 parallel (m and n) and 1 reduction (k) dimension candidates that form
/// a matmul subcomputation within `linalgOp`. These dimensions are such that:
///   1. The m dimension is involved in an outer-product along LHS
///      (i.e. it is a permutation on RES and LHS and does not appear in RHS).
///   2. The n dimension is involved in an outer-product along RHS
///      (i.e. it is a permutation on RES and RHS and does not appear in LHS).
///   3. The k dimension appears as a permutation on LHS and RHS.
///   4. m, n and k appear only once in any given indexing.
///   5. Optional batch dimensions that appear in all operands are captured.
/// This allows e.g. detecting that some contraction is embedded within
/// `linalgOp` with some orthogonal heuristic.
FailureOr<ContractionDimensions>
mlir::linalg::inferContractionDims(LinalgOp linalgOp) {
  if (linalgOp.getNumDpsInits() != 1 || linalgOp.getNumDpsInputs() != 2)
    return failure();

  llvm::SmallDenseSet<int64_t> a = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInputOperand(0), par);
  llvm::SmallDenseSet<int64_t> b = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInputOperand(1), par);
  llvm::SmallDenseSet<int64_t> c = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInitOperand(0), par);

  // A & C - B are the iterators involved in an outer-product along A (the LHS).
  llvm::SmallDenseSet<int64_t> ac = a;
  llvm::set_intersect(ac, c);
  llvm::set_subtract(ac, b);
  // B & C - A are the iterators involved in an outer-product along B (the RHS).
  llvm::SmallDenseSet<int64_t> bc = b;
  llvm::set_intersect(bc, c);
  llvm::set_subtract(bc, a);
  // A & B & C are the "batch" dimensions.
  llvm::SmallDenseSet<int64_t> batches = a;
  llvm::set_intersect(batches, b);
  llvm::set_intersect(batches, c);

  // A & B red are the reduction dimensions.
  llvm::SmallDenseSet<int64_t> ra = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInputOperand(0), red);
  llvm::SmallDenseSet<int64_t> rb = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInputOperand(1), red);
  llvm::set_intersect(ra, rb);

  // Return each set in sorted order.
  ContractionDimensions dimensions{
      SmallVector<unsigned, 2>(batches.begin(), batches.end()),
      SmallVector<unsigned, 2>(ac.begin(), ac.end()),
      SmallVector<unsigned, 2>(bc.begin(), bc.end()),
      SmallVector<unsigned, 2>(ra.begin(), ra.end())};
  llvm::sort(dimensions.batch.begin(), dimensions.batch.end());
  llvm::sort(dimensions.m.begin(), dimensions.m.end());
  llvm::sort(dimensions.n.begin(), dimensions.n.end());
  llvm::sort(dimensions.k.begin(), dimensions.k.end());
  return dimensions;
}

namespace mlir::linalg::detail {
enum class MatchContractionResult {
  Success = 0,
  NotLinalgOp,
  WrongNumOperands,
  NoReduction,
  NotProjectedPermutations,
  NotAddMul
};
} // namespace mlir::linalg::detail

mlir::linalg::detail::MatchContractionResult
mlir::linalg::detail::isContractionInterfaceImpl(
    Operation *op, mlir::linalg::ContractionDimensions *dimensions) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return MatchContractionResult::NotLinalgOp;
  if (linalgOp.getNumDpsInputs() != 2 || linalgOp.getNumDpsInits() != 1)
    return MatchContractionResult::WrongNumOperands;
  auto mapRange = linalgOp.getIndexingMapsArray();
  if (linalgOp.getNumReductionLoops() == 0)
    return MatchContractionResult::NoReduction;
  if (llvm::any_of(mapRange,
                   [](AffineMap m) { return !m.isProjectedPermutation(); }))
    return MatchContractionResult::NotProjectedPermutations;
  // TODO: more fields than add/mul.
  // clang-format off
  if (!::isContractionBody<
        arith::MulFOp, arith::AddFOp,
        arith::MulIOp, arith::AddIOp,
        complex::MulOp, complex::AddOp,
        arith::AndIOp, arith::OrIOp>(
      *linalgOp.getBlock())) {
    return MatchContractionResult::NotAddMul;
  }
  // clang-format on

  if (dimensions) {
    FailureOr<ContractionDimensions> res = inferContractionDims(linalgOp);
    assert(succeeded(res) && "unexpected failure to infer contraction dims");
    *dimensions = *res;
  }
  return MatchContractionResult::Success;
}

StringRef
mlir::linalg::detail::getMatchContractionMessage(MatchContractionResult res) {
  switch (res) {
  case MatchContractionResult::NotLinalgOp:
    return "expected a LinalgOp";
  case MatchContractionResult::WrongNumOperands:
    return "expected op with 2 inputs and 1 output";
  case MatchContractionResult::NoReduction:
    return "expected at least 1 reduction";
  case MatchContractionResult::NotProjectedPermutations:
    return "expected indexing maps to be projected permutations";
  case MatchContractionResult::NotAddMul:
    return "expected add/mul op in the body";
  case MatchContractionResult::Success:
    return "";
  }
  llvm_unreachable("unhandled MatchContractionResult case");
}

bool mlir::linalg::isaContractionOpInterface(LinalgOp linalgOp) {
  if (!linalgOp)
    return false;
  Operation *op = linalgOp.getOperation();
  return isa<ContractionOpInterface>(op) ||
         (mlir::linalg::detail::isContractionInterfaceImpl(op) ==
          mlir::linalg::detail::MatchContractionResult::Success);
}

/// Verify that a LinalgOp `op` is a contraction.
/// A Linalg contraction is defined in general terms:
///   1. Has 2 input and 1 output shapes.
///   2. Has at least one reduction dimension.
///   3. Has only projected permutation indexing maps.
///   4. its body computes `u5(u1(c) + u2(u3(a) * u4(b)))` on some field
///   (AddOpType, MulOpType), where u1, u2, u3, u4 and u5 represent scalar unary
///   operations that may change the type (e.g. for mixed-precision).
/// As a consequence, when vectorization of such an op occurs, the only special
/// behavior is that the (unique) MulOpType is vectorized into a
/// `vector.contract`. All other ops are handled in a generic fashion.
/// In the future, we may wish to allow more input arguments and elementwise and
/// constant operations that do not involve the reduction dimension(s).
LogicalResult mlir::linalg::detail::verifyContractionInterface(Operation *op) {
  auto res = isContractionInterfaceImpl(op);
  if (res != MatchContractionResult::Success)
    return op->emitError(getMatchContractionMessage(res));
  return success();
}

//===----------------------------------------------------------------------===//
// ConvolutionOpInterface implementation
//===----------------------------------------------------------------------===//

/// Of the given two expressions returns one that is of type T (`lhs` gets
/// preference over `rhs`)
template <typename T>
static T getAffineExprOfType(AffineExpr lhs, AffineExpr rhs) {
  return lhs.isa<T>() ? lhs.cast<T>()
                      : (rhs.isa<T>() ? rhs.cast<T>() : nullptr);
}

namespace {
/// Walk the indexing expressions for input of a convolution operation to verify
/// its of the right form, either
/// - AffineDimExpr
/// - AffineDimExpr (`*` (AffineSymbolExpr | AffineConstantExpr))?
///      (`+` AffineDimExpr (`*` (AffineSymbolExpr | AffineConstantExpr))?)*
///
/// classifies the AffineDimExpr as convolved dimensions or unconvolved
/// dimensions and verifies each dimension occurs only once.
struct ConvAccessExprWalker
    : public AffineExprVisitor<ConvAccessExprWalker, LogicalResult> {
  // Stores dimensions used in expressions of the above form.
  llvm::SmallDenseSet<int64_t> convolvedDims;
  // Stores the dual mapping between LHS and RHS of convolution exprs.
  llvm::SmallDenseMap<int64_t, int64_t> convolvedDimMapping;
  // Stores single use dimensions used by an AffineDimExpr.
  llvm::SmallDenseSet<int64_t> unConvolvedDims;
  // Stores a mapping from convolved dims to their coefficient.
  llvm::SmallDenseMap<int64_t, AffineExpr> strideAndDilationMapping;

  // Removes dims with multiple uses in the source input map from dimension
  // sets tracked by this walker.
  void clearMultiUseDims(AffineMap map) {
    for (int dimPos = 0, e = map.getNumDims(); dimPos < e; ++dimPos) {
      if (llvm::count_if(map.getResults(), [dimPos](AffineExpr e) {
            return e.isFunctionOfDim(dimPos);
          }) > 1) {
        convolvedDims.erase(dimPos);
        unConvolvedDims.erase(dimPos);
        // If a duplicate dim is marked as convolved, the pair of the duplicate
        // dim must be removed from the map as well.
        if (convolvedDimMapping.contains(dimPos)) {
          int64_t pairedDim = convolvedDimMapping[dimPos];
          convolvedDims.erase(pairedDim);
          unConvolvedDims.erase(pairedDim);
          strideAndDilationMapping.erase(pairedDim);
          convolvedDimMapping.erase(dimPos);
          convolvedDimMapping.erase(pairedDim);
        }
      }
    }
  }

  LogicalResult visitDimExpr(AffineDimExpr dimExpr) {
    unsigned position = dimExpr.getPosition();
    if (unConvolvedDims.count(position) || convolvedDims.count(position)) {
      return failure();
    }
    unConvolvedDims.insert(position);
    return success();
  }

  LogicalResult visitSymbolExpr(AffineSymbolExpr expr) { return failure(); }

  LogicalResult visitConstantExpr(AffineConstantExpr expr) { return failure(); }

  LogicalResult visitAffineBinaryOpExpr(AffineBinaryOpExpr binaryExpr) {
    // In pre-order visit, top level op has to be an add op.
    if (binaryExpr.getKind() != AffineExprKind::Add)
      return failure();
    auto lhsDimPos = getDimExprOrMulExprDimPos(binaryExpr.getLHS());
    auto rhsDimPos = getDimExprOrMulExprDimPos(binaryExpr.getRHS());
    if (failed(lhsDimPos) || failed(rhsDimPos))
      return failure();
    convolvedDimMapping[*lhsDimPos] = *rhsDimPos;
    convolvedDimMapping[*rhsDimPos] = *lhsDimPos;
    return success();
  }

  FailureOr<int64_t> getDimExprOrMulExprDimPos(AffineExpr expr) {
    if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
      int64_t dim = dimExpr.getPosition();
      if (convolvedDims.count(dim) || unConvolvedDims.count(dim))
        return failure();
      // Stride/dilation for this dim is implicitly 1.
      strideAndDilationMapping[dim] =
          getAffineConstantExpr(1, expr.getContext());
      convolvedDims.insert(dim);
      return dim;
    }
    if (auto symbolMulExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
      if (symbolMulExpr.getKind() != AffineExprKind::Mul)
        return failure();
      auto lhsExpr = symbolMulExpr.getLHS();
      auto rhsExpr = symbolMulExpr.getRHS();
      // Check for symbol expression.
      AffineExpr mulExpr =
          getAffineExprOfType<AffineSymbolExpr>(lhsExpr, rhsExpr);
      // If there was no symbol expr, check for constant expression.
      if (!mulExpr) {
        mulExpr = getAffineExprOfType<AffineConstantExpr>(lhsExpr, rhsExpr);
      }
      auto dimExpr = getAffineExprOfType<AffineDimExpr>(lhsExpr, rhsExpr);
      if (!mulExpr || !dimExpr)
        return failure();
      int64_t dim = dimExpr.getPosition();
      if (convolvedDims.count(dim) || unConvolvedDims.count(dim))
        return failure();
      strideAndDilationMapping[dim] = mulExpr;
      convolvedDims.insert(dim);
      return dim;
    }
    return failure();
  }
};
} // namespace

static llvm::SmallDenseSet<int64_t> getPreservedDims(AffineMap map) {
  assert(map.isProjectedPermutation() &&
         "expected map to have projected permutations");
  llvm::SmallDenseSet<int64_t> preservedDims;
  for (auto expr : map.getResults())
    preservedDims.insert(expr.cast<AffineDimExpr>().getPosition());
  return preservedDims;
}

static SmallVector<int64_t, 2>
getConstantsFromExprList(SmallVector<AffineExpr, 2> exprs) {
  SmallVector<int64_t, 2> vals;
  for (auto e : exprs) {
    auto constantExpr = e.dyn_cast<AffineConstantExpr>();
    assert(constantExpr && "Found non-constant stride/dilation");
    vals.push_back(constantExpr.getValue());
  }
  return vals;
}

/// Classifies dimensions in the `linalgOp` used by a convolution
/// subcomputation, as captured by `inputExprWalker`. If
/// `allowEmptyConvolvedDims` is not set this this will fail if there is not
/// at least convolved dimension pair (output image + filter loop). Convolution
/// dimensions are specified in sorted order, and strides match the order of
/// the filter loop dimensions, while the dilations match the order of the
/// output image dimensions.
static FailureOr<ConvolutionDimensions>
inferConvolutionDimsImpl(LinalgOp linalgOp,
                         ConvAccessExprWalker &inputExprWalker,
                         bool allowEmptyConvolvedDims) {
  llvm::SmallDenseSet<int64_t> filterDims = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInputOperand(1), par);
  llvm::SmallDenseSet<int64_t> outputDims = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInitOperand(0), par);

  // unConvolvedDims & outputDims - filterDims are the batch iterators.
  llvm::SmallDenseSet<int64_t> batch = inputExprWalker.unConvolvedDims;
  llvm::set_intersect(batch, outputDims);
  llvm::set_subtract(batch, filterDims);

  // convolvedDims & outputDims are the output image iterators.
  llvm::SmallDenseSet<int64_t> oi = inputExprWalker.convolvedDims;
  llvm::set_intersect(oi, outputDims);

  // filterDims & outputDims - unConvolvedDims are the output channel iterators.
  llvm::SmallDenseSet<int64_t> oc = filterDims;
  llvm::set_intersect(oc, outputDims);
  llvm::set_subtract(oc, inputExprWalker.unConvolvedDims);

  // filterDims & outputDims & unConvolvedDims are the depth iterators.
  llvm::SmallDenseSet<int64_t> depth = filterDims;
  llvm::set_intersect(depth, outputDims);
  llvm::set_intersect(depth, inputExprWalker.unConvolvedDims);

  llvm::SmallDenseSet<int64_t> filterReducedDims =
      findPermutationsIndexingOperand(linalgOp, linalgOp.getDpsInputOperand(1),
                                      red);

  // convolvedDims & filterReducedDims are the filter loop iterators.
  llvm::SmallDenseSet<int64_t> fl = inputExprWalker.convolvedDims;
  llvm::set_intersect(fl, filterReducedDims);

  // unConvolvedDims & filterReducedDims are the input channel iterators.
  llvm::SmallDenseSet<int64_t> ic = inputExprWalker.unConvolvedDims;
  llvm::set_intersect(ic, filterReducedDims);

  if (oi.empty() && !allowEmptyConvolvedDims)
    return failure();

  // Return each set in sorted order.
  ConvolutionDimensions dimensions{
      SmallVector<unsigned, 2>(batch.begin(), batch.end()),
      SmallVector<unsigned, 2>(oi.begin(), oi.end()),
      SmallVector<unsigned, 2>(oc.begin(), oc.end()),
      SmallVector<unsigned, 2>(fl.begin(), fl.end()),
      SmallVector<unsigned, 2>(ic.begin(), ic.end()),
      SmallVector<unsigned, 2>(depth.begin(), depth.end()),
      /*strides=*/SmallVector<int64_t, 2>{},
      /*dilations=*/SmallVector<int64_t, 2>{}};
  llvm::sort(dimensions.batch.begin(), dimensions.batch.end());
  llvm::sort(dimensions.outputImage.begin(), dimensions.outputImage.end());
  llvm::sort(dimensions.outputChannel.begin(), dimensions.outputChannel.end());
  llvm::sort(dimensions.filterLoop.begin(), dimensions.filterLoop.end());
  llvm::sort(dimensions.inputChannel.begin(), dimensions.inputChannel.end());
  llvm::sort(dimensions.depth.begin(), dimensions.depth.end());

  // Use the op carried strides/dilations attribute if present.
  auto nativeStrides = linalgOp->getAttrOfType<DenseIntElementsAttr>("strides");
  if (!nativeStrides) {
    SmallVector<AffineExpr, 2> strideExprs;
    for (unsigned oiDim : dimensions.outputImage)
      strideExprs.push_back(inputExprWalker.strideAndDilationMapping[oiDim]);
    dimensions.strides = getConstantsFromExprList(strideExprs);
  } else {
    dimensions.strides = llvm::to_vector<2>(nativeStrides.getValues<int64_t>());
  }
  auto nativeDilations =
      linalgOp->getAttrOfType<DenseIntElementsAttr>("dilations");
  if (!nativeDilations) {
    SmallVector<AffineExpr, 2> dilationExprs;
    for (unsigned flDim : dimensions.filterLoop)
      dilationExprs.push_back(inputExprWalker.strideAndDilationMapping[flDim]);
    dimensions.dilations = getConstantsFromExprList(dilationExprs);
  } else {
    dimensions.dilations =
        llvm::to_vector<2>(nativeDilations.getValues<int64_t>());
  }
  return dimensions;
}

/// Find at least 1 parallel (output_image) and reduction (filter_loop)
/// dimension candidates that form a convolution subcomputation within
/// `linalgOp`. The LHS is assumed to be the convolution input while the
/// RHS is assumed as the filter.
/// These dimensions are such that:
///   1. Optional batch dimensions that appear in the input and filter.
///   2. The output_image dimension is involved in a cross-correlation along LHS
///      (i.e. it is a permutation on RES and LHS and has an associated
///      filter_loop in RHS).
///   3. Optional output_channel dimension is involved in an outer-product along
///      RHS (i.e. it is a permutation on RES and RHS and does not appear in
///      LHS).
///   4. Optional input_channel dimension appears as a permutation on LHS and
///      RHS.
///   5. The filter_loop dimension appears as a permutation on the RHS and
///      represents the shape of the kernel cross-correlated along a
///      corresponding output_image dim.
///   6. The input_channel dimension appears as a permutation on LHS and RHS.
///   7. All dimensions appear only once in any given indexing map.
/// This allows e.g. detecting that some convolution is embedded within
/// `linalgOp` with some orthogonal heuristic.
/// When multiple dimension occurrences exist that match any classification
/// indices are returned in sorted order.
/// Returns a failure if `output_image` (and implicitly `filter_loop`) is empty.
FailureOr<ConvolutionDimensions>
mlir::linalg::inferConvolutionDims(LinalgOp linalgOp) {
  if (linalgOp.getNumDpsInits() != 1 || linalgOp.getNumDpsInputs() != 2)
    return failure();

  auto indexingMaps = linalgOp.getIndexingMapsArray();

  // Check the input indexing map has the right form.
  ConvAccessExprWalker inputExprWalker;
  for (AffineExpr expr : indexingMaps[0].getResults())
    (void)inputExprWalker.visit(expr);
  inputExprWalker.clearMultiUseDims(indexingMaps[0]);

  return inferConvolutionDimsImpl(linalgOp, inputExprWalker,
                                  /*allowEmptyConvolvedDims=*/false);
}

namespace mlir::linalg::detail {
enum class MatchConvolutionResult {
  Success = 0,
  NotLinalgOp,
  WrongNumOperands,
  WrongInputIndexingMap,
  NotProjectedPermutations,
  NonConvolutionLoop,
  OutputDimsNotParallel,
  NonOutputDimNotReduction
};
} // namespace mlir::linalg::detail

mlir::linalg::detail::MatchConvolutionResult
mlir::linalg::detail::isConvolutionInterfaceImpl(
    Operation *op, ConvolutionDimensions *dimensions) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return MatchConvolutionResult::NotLinalgOp;
  if (linalgOp.getNumDpsInputs() < 2 || linalgOp.getNumDpsInits() != 1)
    return MatchConvolutionResult::WrongNumOperands;

  auto indexingMaps = linalgOp.getIndexingMapsArray();

  // Check the input indexing map has the right form.
  ConvAccessExprWalker inputExprWalker;
  if (llvm::any_of(indexingMaps[0].getResults(),
                   [&inputExprWalker](AffineExpr expr) {
                     return failed(inputExprWalker.visit(expr));
                   })) {
    return MatchConvolutionResult::WrongInputIndexingMap;
  }

  // Filter and output maps must be projected permutation.
  if (!indexingMaps[1].isProjectedPermutation() ||
      !indexingMaps.back().isProjectedPermutation())
    return MatchConvolutionResult::NotProjectedPermutations;

  auto iteratorTypes = linalgOp.getIteratorTypesArray();

  llvm::SmallDenseSet<int64_t> outputDims =
      getPreservedDims(indexingMaps.back());
  llvm::SmallDenseSet<int64_t> filterDims = getPreservedDims(indexingMaps[1]);
  // Make sure all loops are characterized as one of:
  // - Batch loop : present in output, as non-convolved in input, not present in
  //   filter.
  // - Output image dimension : present in output, convolved dims in input, not
  //   present in filter.
  // - Output channel dimension : present in output, not present in input,
  //   present in filter.
  // - Filter loop dimension : present in filter, convolved in input, not
  //   present in output.
  // - Input channel dimension : unconvolved in input, not present in output,
  //   present in filter.
  // - Depth multiplier : unconvolved in input, present in output, present in
  //   filter.
  llvm::SmallDenseSet<int64_t> allLoopDims;
  for (auto outputExpr : indexingMaps.back().getResults()) {
    int64_t outputDim = outputExpr.cast<AffineDimExpr>().getPosition();
    if (inputExprWalker.unConvolvedDims.count(outputDim) &&
        !filterDims.count(outputDim)) {
      // Batch dimension.
      if (iteratorTypes[outputDim] != utils::IteratorType::parallel)
        return MatchConvolutionResult::OutputDimsNotParallel;
      allLoopDims.insert(outputDim);
      continue;
    }
    if (inputExprWalker.convolvedDims.count(outputDim) &&
        !filterDims.count(outputDim)) {
      // Output image Loop dimension.
      if (iteratorTypes[outputDim] != utils::IteratorType::parallel)
        return MatchConvolutionResult::OutputDimsNotParallel;
      allLoopDims.insert(outputDim);
      continue;
    }
    if (!inputExprWalker.convolvedDims.count(outputDim) &&
        !inputExprWalker.unConvolvedDims.count(outputDim) &&
        filterDims.count(outputDim)) {
      // Output channel dimension.
      if (iteratorTypes[outputDim] != utils::IteratorType::parallel)
        return MatchConvolutionResult::OutputDimsNotParallel;
      allLoopDims.insert(outputDim);
      continue;
    }
    if (inputExprWalker.unConvolvedDims.count(outputDim) &&
        filterDims.count(outputDim)) {
      // Depth multiplier.
      if (iteratorTypes[outputDim] != utils::IteratorType::parallel)
        return MatchConvolutionResult::OutputDimsNotParallel;
      allLoopDims.insert(outputDim);
      continue;
    }
    return MatchConvolutionResult::NonConvolutionLoop;
  }
  for (auto filterExpr : indexingMaps[1].getResults()) {
    int64_t filterDim = filterExpr.cast<AffineDimExpr>().getPosition();
    if (outputDims.count(filterDim) &&
        !inputExprWalker.unConvolvedDims.count(filterDim) &&
        !inputExprWalker.convolvedDims.count(filterDim)) {
      // Output channel dimension. This is already seen, continue;
      continue;
    }
    if (inputExprWalker.convolvedDims.count(filterDim) &&
        !outputDims.count(filterDim)) {
      // Filter loop dimension.
      if (iteratorTypes[filterDim] != utils::IteratorType::reduction)
        return MatchConvolutionResult::NonOutputDimNotReduction;
      if (allLoopDims.count(filterDim))
        return MatchConvolutionResult::NonConvolutionLoop;
      allLoopDims.insert(filterDim);
      continue;
    }
    if (inputExprWalker.unConvolvedDims.count(filterDim) &&
        !outputDims.count(filterDim)) {
      // Input channel dimension.
      if (iteratorTypes[filterDim] != utils::IteratorType::reduction)
        return MatchConvolutionResult::NonOutputDimNotReduction;
      if (allLoopDims.count(filterDim))
        return MatchConvolutionResult::NonConvolutionLoop;
      allLoopDims.insert(filterDim);
      continue;
    }
    if (inputExprWalker.unConvolvedDims.count(filterDim) &&
        outputDims.count(filterDim)) {
      // Depthwise loop. Already seen.
      continue;
    }
    return MatchConvolutionResult::NonConvolutionLoop;
  }
  // All loops must be covered now.
  if (allLoopDims.size() != linalgOp.getNumLoops())
    return MatchConvolutionResult::NonConvolutionLoop;

  if (dimensions) {
    FailureOr<ConvolutionDimensions> res =
        inferConvolutionDimsImpl(linalgOp, inputExprWalker,
                                 /*allowEmptyConvolvedDims=*/true);
    assert(succeeded(res) && "unexpected failure to infer convolution dims");
    *dimensions = *res;
  }

  return MatchConvolutionResult::Success;
}

StringRef
mlir::linalg::detail::getMatchConvolutionMessage(MatchConvolutionResult res) {
  switch (res) {
  case MatchConvolutionResult::NotLinalgOp:
    return "expected a LinalgOp";
  case MatchConvolutionResult::WrongNumOperands:
    return "expected op with 2 inputs and 1 output";
  case MatchConvolutionResult::WrongInputIndexingMap:
    return "unexpected input index map for convolutions";
  case MatchConvolutionResult::NotProjectedPermutations:
    return "expected output/filter indexing maps to be projected permutations";
  case MatchConvolutionResult::NonConvolutionLoop:
    return "unexpected loop dimension for convolution op";
  case MatchConvolutionResult::OutputDimsNotParallel:
    return "expected all iterators used to access outputs to be parallel";
  case MatchConvolutionResult::NonOutputDimNotReduction:
    return "expected all iterators not used to access outputs to be reduction";
  case MatchConvolutionResult::Success:
    return "";
  }
  llvm_unreachable("unhandled MatchConvolutionResult case");
}

bool mlir::linalg::isaConvolutionOpInterface(LinalgOp linalgOp) {
  return linalg::detail::isConvolutionInterfaceImpl(linalgOp.getOperation()) ==
         linalg::detail::MatchConvolutionResult::Success;
}

LogicalResult mlir::linalg::detail::verifyConvolutionInterface(Operation *op) {
  MatchConvolutionResult res = isConvolutionInterfaceImpl(op);
  if (res != MatchConvolutionResult::Success)
    return op->emitError(getMatchConvolutionMessage(res));
  return success();
}

//===----------------------------------------------------------------------===//
// FillOpInterface implementation
//===----------------------------------------------------------------------===//

enum class MatchFillResult {
  Success = 0,
  NotLinalgOp,
  WrongNumOperands,
  NotScalarInput
};

static MatchFillResult isFillInterfaceImpl(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return MatchFillResult::NotLinalgOp;
  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1)
    return MatchFillResult::WrongNumOperands;

  OpOperand *value = linalgOp.getDpsInputOperand(0);
  if (!linalgOp.isScalar(value))
    return MatchFillResult::NotScalarInput;

  return MatchFillResult::Success;
}

LogicalResult mlir::linalg::detail::verifyFillInterface(Operation *op) {
  auto res = isFillInterfaceImpl(op);
  if (res == MatchFillResult::NotLinalgOp)
    return op->emitError("expected a LinalgOp");
  if (res == MatchFillResult::WrongNumOperands)
    return op->emitError("expected op with 1 input and 1 output");
  if (res == MatchFillResult::NotScalarInput)
    return op->emitError("expected op with scalar input");

  return success();
}

//===----------------------------------------------------------------------===//
// StructuredOpInterface implementation
//===----------------------------------------------------------------------===//

SmallVector<OpFoldResult> LinalgOp::createFlatListOfOperandDims(OpBuilder &b,
                                                                Location loc) {
  SmallVector<OpFoldResult> res;
  for (OpOperand &opOperand : getOperation()->getOpOperands()) {
    for (int64_t i = 0, e = getRank(&opOperand); i < e; ++i)
      res.push_back(createFoldedDimOp(b, loc, opOperand.get(), i));
  }
  return res;
}

SmallVector<int64_t, 4> LinalgOp::createFlatListOfOperandStaticDims() {
  SmallVector<int64_t, 4> res;
  assert(!hasDynamicShape() && "expected operands to have static shapes");
  for (OpOperand &opOperand : getOperation()->getOpOperands())
    llvm::append_range(res, getShape(&opOperand));
  return res;
}

SmallVector<Range, 4> LinalgOp::createLoopRanges(OpBuilder &b, Location loc) {
  AffineMap map = getLoopsToShapesMap();
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  auto viewSizes = createFlatListOfOperandDims(b, loc);
  SmallVector<Range, 4> res(numDims);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = result.dyn_cast<AffineDimExpr>()) {
      if (res[d.getPosition()].offset)
        continue;
      res[d.getPosition()] =
          Range{b.getIndexAttr(0), viewSizes[idx], b.getIndexAttr(1)};
    }
  }
  return res;
}

SmallVector<int64_t, 4> LinalgOp::computeStaticLoopSizes() {
  AffineMap map = getLoopsToShapesMap();
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  SmallVector<int64_t, 4> allShapeSizes = createFlatListOfOperandStaticDims();
  SmallVector<int64_t, 4> res(numDims, 0);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = result.dyn_cast<AffineDimExpr>())
      res[d.getPosition()] = allShapeSizes[idx];
  }
  return res;
}

/// Visitor to check if any of the given set of positions from AffineDimExprs
/// are used within an AffineExpr.
struct HasAffineDimExprVisitor
    : public AffineExprVisitor<HasAffineDimExprVisitor, bool> {
  HasAffineDimExprVisitor(llvm::SmallBitVector positions)
      : positions(std::move(positions)) {}

  bool visitAffineBinaryOpExpr(AffineBinaryOpExpr binaryOpExpr) {
    return visit(binaryOpExpr.getLHS()) || visit(binaryOpExpr.getRHS());
  }

  bool visitDimExpr(AffineDimExpr dimExpr) {
    return positions.test(dimExpr.getPosition());
  }

  bool visitConstantExpr(AffineConstantExpr constExpr) { return false; }

  bool visitSymbolExpr(AffineSymbolExpr symbolExpr) { return false; }

private:
  llvm::SmallBitVector positions;
};

static std::pair<int64_t, int64_t>
getResultsPositionInLoopsToShapeMap(LinalgOp &op) {
  int64_t inputRankSum = 0;
  int64_t outputRankSum = 0;
  for (OpOperand *input : op.getDpsInputOperands())
    inputRankSum += op.getRank(input);
  for (OpOperand &output : op.getDpsInitsMutable())
    outputRankSum += op.getRank(&output);
  return {inputRankSum, inputRankSum + outputRankSum};
}

LogicalResult
LinalgOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  // An example that helps understand the logic below.
  // Consider the following expression O(i+j, j) += A(i,k) * B(k, j)
  // We want to express the shape of dim 0 of O in terms of shape of the inputs.
  // This is achieved as follows.
  //   loopsToShapesMap = (d0, d1, d2) -> (d0, d2, d2, d1, d0 + d1, d1)
  //   subMapOfResultShapes = (d0, d1, d2) -> (d0 + d1, d1)
  //   shapesToLoopsMap = (d0, d2, d2, d3, d4, d5) -> (d0, d3, d2)
  //   resultShapesFromInputShapes = subMapOfResultDim.compose(shapesToLoopMap)
  //     = (d0, d1, d2, d3, d4, d5) -> (d0 + d1, d1)
  AffineMap loopsToShapesMap = getLoopsToShapesMap();

  // Find the position in the above map that represents the shape of the
  // result:dim being inferred.
  auto resultShapesSubMapPos = getResultsPositionInLoopsToShapeMap(*this);

  /// From loopsToShapesMap extract the submap that represents the shape of the
  /// (resultIdx, dim) needed.
  AffineMap loopToResultsShapeMap = loopsToShapesMap.getSliceMap(
      resultShapesSubMapPos.first,
      resultShapesSubMapPos.second - resultShapesSubMapPos.first);
  AffineMap resultShapesFromInputShapesMap =
      loopToResultsShapeMap.compose(getShapesToLoopsMap());

  // Check that the result dim map does not contain the positions corresponding
  // to the outputs.
  llvm::SmallBitVector outputDims(resultShapesFromInputShapesMap.getNumDims());
  outputDims.set(resultShapesSubMapPos.first, resultShapesSubMapPos.second);
  HasAffineDimExprVisitor checkDimExpr(std::move(outputDims));
  Location loc = getOperation()->getLoc();
  IRRewriter rewriter(b);
  SmallVector<OpFoldResult> allResultDimValues =
      affine::makeComposedFoldedMultiResultAffineApply(
          rewriter, loc, resultShapesFromInputShapesMap,
          createFlatListOfOperandDims(b, loc));
  int64_t pos = 0;
  ArrayRef<AffineExpr> shapeExprs = resultShapesFromInputShapesMap.getResults();
  for (OpOperand &opOperand : getDpsInitsMutable()) {
    SmallVector<OpFoldResult> shapes;
    for (int64_t dim : llvm::seq<int64_t>(0, getRank(&opOperand))) {
      auto shapedType = llvm::cast<ShapedType>(opOperand.get().getType());
      if (!shapedType.isDynamicDim(dim)) {
        // Static dim: Return IntegerAttr.
        shapes.push_back(b.getIndexAttr(shapedType.getDimSize(dim)));
      } else {
        // Dynamic dim: Return Value.
        OpFoldResult ofr = checkDimExpr.visit(shapeExprs[pos])
                               ? createOrFoldDimOp(b, loc, opOperand.get(), dim)
                               : allResultDimValues[pos];
        shapes.push_back(getValueOrCreateConstantIndexOp(b, loc, ofr));
      }
      pos++;
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));
  }
  return success();
}

/// Return the index in the indexingMaps vector that corresponds to this
/// `opOperand`.
int64_t LinalgOp::getIndexingMapIndex(OpOperand *opOperand) {
  auto operandNumber = opOperand->getOperandNumber();
  auto dpsIface = cast<DestinationStyleOpInterface>(*this->getOperation());
  if (!dpsIface.isDpsInput(opOperand))
    return operandNumber;
  unsigned start = dpsIface.getDpsInits().getBeginOperandIndex();
  assert(!dpsIface.isDpsInit(opOperand));
  // Account for potential inputs that are not DPS and may not appear in
  // `indexingMaps`.
  return cast<DestinationStyleOpInterface>(*this->getOperation())
             .getNumDpsInputs() +
         operandNumber - start;
}

LogicalResult mlir::linalg::detail::verifyStructuredOpInterface(Operation *op) {
  LinalgOp linalgOp = cast<LinalgOp>(op);

  // Before checking indexing maps, we need to make sure the attributes
  // referenced by it are valid.
  if (linalgOp.hasDynamicIndexingMaps())
    if (failed(linalgOp.verifyIndexingMapRequiredAttributes()))
      return failure();

  // All input/output operands must be indexed.
  if (static_cast<int64_t>(linalgOp.getIndexingMapsArray().size()) !=
      linalgOp->getNumOperands())
    return op->emitOpError("expected the number of indexing_map (")
           << linalgOp.getIndexingMapsArray().size()
           << ") to be equal to the number of input/output operands ("
           << linalgOp->getNumOperands() << ")";

  for (OpOperand &opOperand : linalgOp->getOpOperands()) {
    AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&opOperand);

    // Symbols disallowed.
    if (indexingMap.getNumSymbols() != 0)
      return op->emitOpError("unexpected symbols in indexing_map #")
             << opOperand.getOperandNumber();

    // Domain must be consistent.
    unsigned numLoops = linalgOp.getNumLoops();
    if (indexingMap.getNumDims() != numLoops)
      return op->emitOpError("expected indexing_map #")
             << opOperand.getOperandNumber() << " to have " << numLoops
             << " dim(s) to match the number of loops";

    int64_t rank = linalgOp.getRank(&opOperand);
    if (indexingMap.getNumResults() != rank)
      return op->emitOpError("expected operand rank (")
             << rank << ") to match the result rank of indexing_map #"
             << opOperand.getOperandNumber() << " ("
             << indexingMap.getNumResults() << ")";
  }

  SmallVector<unsigned> redDims;
  linalgOp.getReductionDims(redDims);

  if (!linalgOp.getShapesToLoopsMap())
    return op->emitOpError("expected the shape-to-loops map to be non-null");

  // Check if given shapes match to inferred shapes.
  SmallVector<int64_t, 4> endLoopRangeValues = linalgOp.getStaticLoopRanges();
  SmallVector<int64_t, 4> startLoopRangeValues(endLoopRangeValues.size(), 0);

  // Verify only static cases since we can't get exact dimension sizes and loop
  // ranges for dynamic cases in this stage.
  if (llvm::none_of(endLoopRangeValues, ShapedType::isDynamic)) {
    for (int64_t &range : endLoopRangeValues)
      range -= 1;
    for (OpOperand &opOperand : linalgOp->getOpOperands()) {
      AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&opOperand);
      SmallVector<int64_t, 4> startIndices =
          indexingMap.compose(startLoopRangeValues);
      SmallVector<int64_t, 4> endIndices =
          indexingMap.compose(endLoopRangeValues);
      ArrayRef<int64_t> shape = linalgOp.getShape(&opOperand);
      for (auto dim : llvm::seq<int64_t>(0, shape.size())) {
        // Ignore dynamic dimension or the case that the dimension size is 0
        if (ShapedType::isDynamic(shape[dim]) || shape[dim] == 0)
          continue;

        // The first index or last index should be the maximum or the minimum in
        // the inferred index ranges since the range is increasing or
        // decreasing. The size of dimensions of input/output operands and the
        // maximum value + 1 in the inferred range should be the same. But, for
        // now we check if the inferred ranges are in boundary of input/output
        // operands' size or not in case that Affine Expressions are complicated
        // such as d0 * 3
        // + d1 since it is not easy to handle the issues.
        // Found the case that this solution can't check, for example, (d0, d1)
        // -> (d1 - d0)
        int64_t inferredDimSize =
            std::max(startIndices[dim], endIndices[dim]) + 1;
        if (std::min(startIndices[dim], endIndices[dim]) < 0) {
          std::string mapStr;
          {
            llvm::raw_string_ostream os(mapStr);
            os << indexingMap;
          }
          return op->emitOpError(
                     "unexpected result less than 0 at expression #")
                 << dim << " in " << mapStr;
        }
        if (indexingMap.getResult(dim).dyn_cast<AffineDimExpr>()) {
          if (inferredDimSize != shape[dim]) {
            return op->emitOpError("inferred input/output operand #")
                   << opOperand.getOperandNumber() << " has shape's dimension #"
                   << dim << " to be " << inferredDimSize << ", but found "
                   << shape[dim];
          }
        } else {
          if (inferredDimSize > shape[dim]) {
            return op->emitOpError("inferred input/output operand #")
                   << opOperand.getOperandNumber() << " has shape's dimension #"
                   << dim << " to be greater than or equal to "
                   << inferredDimSize << ", but found " << shape[dim];
          }
        }
      }
    }
  }

  // Check the region has exactly one block.
  if (linalgOp->getNumRegions() != 1 ||
      !llvm::hasSingleElement(linalgOp->getRegion(0)))
    return op->emitOpError("expects to have 1 region with 1 block");

  // Simplifying assumption: bbargs match 1-1 with shape operands elemental
  // types.
  // TODO: once ranked shape types are plugged in, we may want to drop the
  // corresponding bbargs, that can never be read from. This will be subject to
  // consistency discussions (i.e. what to do with output tensors whose bbarg is
  // not used).
  Block &block = linalgOp->getRegion(0).front();

  if (linalgOp.getOpOperandsMatchingBBargs().size() != block.getNumArguments())
    return op->emitOpError("expected as many non-induction variable region "
                           "arguments as the number of input/output operands");

  for (OpOperand *opOperand : linalgOp.getOpOperandsMatchingBBargs()) {
    Type elementType = getElementTypeOrSelf(opOperand->get());
    Type argType = block.getArgument(opOperand->getOperandNumber()).getType();
    if (elementType != argType)
      return op->emitOpError("expected type of bb argument #")
             << opOperand->getOperandNumber() << " (" << argType << ")"
             << " to match element or self type of the corresponding operand ("
             << elementType << ")";
  }

  return success();
}
