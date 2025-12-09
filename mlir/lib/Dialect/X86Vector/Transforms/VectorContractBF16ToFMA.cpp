//===- VectorContractBF16ToFMA.cpp
//--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/Utils/X86VectorUtils.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;

// This function retrives the source operation of the load or transfer
// reads and creates subviews for the BF16 packed-operations to
// broadcast or load BF16 elements as F32 packed elements.
//
// For example:
// ```
//   vector.load %arg0[%c0, %c0, %c0]:memref<4x1x2xbf16>,vector<1x1x2xbf16>
//   vector.load %arg0[%c0, %c0, %c0]:memref<1x32x2xbf16>,vector<1x8x2xbf16>
// ```
// to
// ```
//   memref.subview %arg0[%c0,%c0,%c1]:memref<4x1x2xbf16> to memref<1x1x1xbf16>
//   memref.subview %arg1[%c0,%c0,%c0]:memref<1x32x2xbf16> to memref<1x8x2xbf16>
//   memref.subview %arg0[%c0,%c0,%c0]:memref<4x1x2xbf16> to memref<1x1x1xbf16>
// ```
static FailureOr<llvm::SmallVector<mlir::memref::SubViewOp>>
getSubviewFromVectorInput(Location loc, PatternRewriter &rewriter,
                          mlir::Value prodOp, int64_t mnDim, int64_t vnniDim,
                          int64_t mnDimIndx) {

  llvm::SmallVector<mlir::memref::SubViewOp> subviews;

  Value srcOperation;
  SmallVector<OpFoldResult> indexVals;

  Operation *defOp = prodOp.getDefiningOp();
  if (!defOp)
    return failure();

  llvm::TypeSwitch<Operation *>(defOp)
      .Case<mlir::vector::TransferReadOp, mlir::vector::LoadOp>(
          [&](auto readOp) {
            srcOperation = readOp.getOperand(0);
            indexVals = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                                  readOp.getIndices().end());
          });

  if (!srcOperation)
    return failure();

  Type srcType = srcOperation.getType();
  if (!llvm::isa<mlir::MemRefType>(srcType))
    return failure();

  llvm::SmallVector<OpFoldResult> strides;
  llvm::SmallVector<OpFoldResult> sizes;

  auto nonVNNIDimSize = indexVals.size() - 1;
  // Create the size and stride offsets.
  for (unsigned int i = 0; i < nonVNNIDimSize; i++) {
    strides.push_back(rewriter.getIndexAttr(1));
    sizes.push_back(rewriter.getIndexAttr(1));
  }

  strides.push_back(rewriter.getIndexAttr(1));
  sizes.push_back(rewriter.getIndexAttr(vnniDim));

  // update the unit/nonUnit Dim size eiither it is A(LHS) or B(RHS).
  sizes[indexVals.size() - mnDimIndx] = rewriter.getIndexAttr(mnDim);

  // for unitDim, first broadcast odd element, so index is set to C1.
  if (mnDim == 1) {
    indexVals[indexVals.size() - 1] = rewriter.getIndexAttr(1);
  }

  auto subview = memref::SubViewOp::create(rewriter, loc, srcOperation,
                                           indexVals, sizes, strides);
  subviews.push_back(subview);

  // For unit-dims, two subviews should be created for the odd and even
  // indexed BF16 element because x86vector.avx.bcst_to_f32.packed op
  // loads and broadcast the first BF16 element into packed F32. It
  // cannot distinguish between even and odd BF16 elements within a
  // packed pair.
  if (mnDim == 1) {
    indexVals[indexVals.size() - 1] = rewriter.getIndexAttr(0);
    sizes[indexVals.size() - 1] = rewriter.getIndexAttr(1);

    auto unitDimEvenIndxSubview = memref::SubViewOp::create(
        rewriter, loc, srcOperation, indexVals, sizes, strides);
    subviews.push_back(unitDimEvenIndxSubview);
  }

  return subviews;
}

// Implements outer product contraction as a sequence of BF16-packed
// operation even/odd loads and FMA operations.
//
// For example:
// ```
//   %1 = vector.load from memref (%m1) -> vector<1x1x2xbf16>
//   %2 = vector.load from memref (%m2) -> vector<1x8x2xbf16>
//   return vector.contract %1, %2, %arg1
// ```
// to
// ```
//   %1 = x86vector.avx.bcst_to_f32.packed %m1[c1] -> vector<8xf32>
//   %2 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %m2 -> vector<8xf32>
//   %3 = vector.fma %1, %2, %arg1
//   %4 = x86vector.avx.bcst_to_f32.packed %m1[c0] -> vector<8xf32>
//   %5 = x86vector.avx.cvt.packed.even.indexed_to_f32 %m2 -> vector<8xf32>
//   return vector.fma %4, %5, %3
// ```
struct VectorContractBF16ToFMA
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind.");

    // TODO: Move this validation to a comon utility folder. Planned to
    // do once (code refactoring), all architecture specific nanokernel
    // passes are merged into the repo.
    VectorType lhsTy = contractOp.getLhsType();
    if (!lhsTy.getElementType().isBF16())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only BF16 lowering is supported.");

    if (!isInVnniLayout(contractOp.getOperation(),
                        contractOp.getIndexingMapsArray(), 2))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Input matrices not in VNNI format.");

    ArrayRef<int64_t> lhsShape = lhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimLhs;
    llvm::copy_if(lhsShape, std::back_inserter(nonUnitDimLhs),
                  [](int64_t dim) { return dim != 1; });

    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimRhs;
    llvm::copy_if(rhsShape, std::back_inserter(nonUnitDimRhs),
                  [](int64_t dim) { return dim != 1; });

    if ((nonUnitDimLhs.size() - 1) > 0 && (nonUnitDimRhs.size() - 1) > 0)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Excepts unit dimensions for either "
                                         "LHS or RHS shape other than VNNI.");

    if ((nonUnitDimLhs.size() - 1) != 1 && (nonUnitDimRhs.size() - 1) != 1)
      return rewriter.notifyMatchFailure(
          contractOp,
          "Excepts a one non-unit A/B dimension for either LHS or RHS shape.");

    VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
    if (!accTy)
      return rewriter.notifyMatchFailure(contractOp, "Wrong accmulator type.");

    if ((lhsTy.getElementType().isBF16() && !accTy.getElementType().isF32()))
      return rewriter.notifyMatchFailure(
          contractOp, "Only F32 acumulation supported for BF16 type.");

    ArrayRef<int64_t> accShape = accTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimAcc;
    llvm::copy_if(accShape, std::back_inserter(nonUnitDimAcc),
                  [](int64_t dim) { return dim != 1; });
    if (nonUnitDimAcc.size() != 1)
      return rewriter.notifyMatchFailure(
          contractOp, "A or B should be a non-unit dim in acc.");

    // Non-unit dimensions should match the vector length of BF16.
    unsigned int nonUnitDim = nonUnitDimLhs.size() == 2 ? nonUnitDimLhs.front()
                                                        : nonUnitDimRhs.front();
    if (nonUnitDim != 4 && nonUnitDim != 8 &&
        !(nonUnitDimAcc.front() == nonUnitDim))
      return rewriter.notifyMatchFailure(
          contractOp, "BF16 packed load operation expects non-unit (LHR or "
                      "RHS) dim and acc dim of size 4/8.");

    // Lower vector.contract to FMAs with help of BF16 packed ops.
    auto loc = contractOp.getLoc();

    // create the unit-dimension LHS or RHS subview and the
    // corresponding non-unit dimension LHS or RHS subview on the other-side.
    // For example, if LHS has type vector<1x1x2xbf16> and RHS has type
    // vector<1x8x2xbf16>, we create two subview for the LHS and one subview
    // for the RHS. In the opposite case (non-unit dimension on the LHS), we
    // do vice-versa.
    bool rhsHasMultipleNonUnitDims = (nonUnitDimRhs.size() - 1) > 0;
    // Select which operand is "unit" and which is "non-unit".
    Value unitSrc =
        rhsHasMultipleNonUnitDims ? contractOp.getLhs() : contractOp.getRhs();
    Value nonUnitSrc =
        rhsHasMultipleNonUnitDims ? contractOp.getRhs() : contractOp.getLhs();

    // mnDim index differs depending on the orientation.
    int unitMnDim = rhsHasMultipleNonUnitDims ? 2 : 2;    // same for both
    int nonUnitMnDim = rhsHasMultipleNonUnitDims ? 2 : 3; // A or B

    // VNNI factor: always 1 for unit dims, 2 for non-unit dims.
    int unitVnni = 1;
    int nonUnitVnni = 2;

    // Non-unit dim size.
    int nonUnitSize = nonUnitDimRhs.front();

    // Build subviews.
    auto unitSubview = getSubviewFromVectorInput(
        loc, rewriter, unitSrc, /*size=*/1, unitVnni, unitMnDim);

    auto nonUnitSubview = getSubviewFromVectorInput(loc, rewriter, nonUnitSrc,
                                                    /*size=*/nonUnitSize,
                                                    nonUnitVnni, nonUnitMnDim);

    // Check failures once.
    if (failed(unitSubview) || failed(nonUnitSubview))
      return rewriter.notifyMatchFailure(
          contractOp, "The input source is not MemRef Type.");

    llvm::SmallVector<mlir::memref::SubViewOp> unitDimSubview = *unitSubview;
    llvm::SmallVector<mlir::memref::SubViewOp> nonUnitDimSubview =
        *nonUnitSubview;

    auto castAcc = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get(nonUnitDimAcc.front(), accTy.getElementType()),
        contractOp.getAcc());
    mlir::VectorType dstType =
        mlir::VectorType::get(nonUnitDimAcc.front(), rewriter.getF32Type());

    // Load, broadcast, and do FMA for odd indexed BF16 elements.
    auto loadBcstOddIndxElementToF32 = x86vector::BcstToPackedF32Op::create(
        rewriter, loc, dstType, unitDimSubview[0]);
    auto loadOddIndxElementF32 = x86vector::CvtPackedOddIndexedToF32Op::create(
        rewriter, loc, dstType, nonUnitDimSubview[0]);
    auto oddIndxFMA =
        vector::FMAOp::create(rewriter, loc, loadBcstOddIndxElementToF32,
                              loadOddIndxElementF32, castAcc);

    llvm::SmallVector<Operation *> users;
    for (OpResult result : contractOp->getResults())
      for (Operation *user : result.getUsers())
        users.push_back(user);

    if (users.size() == 1) {
      rewriter.setInsertionPoint(users[0]);
    }

    // Load, broadcast, and do FMA for even indexed BF16 elements.
    auto loadBcstEvenIndxElementToF32 = x86vector::BcstToPackedF32Op::create(
        rewriter, loc, dstType, unitDimSubview[1]);
    auto loadEvenIndxElementF32 =
        x86vector::CvtPackedEvenIndexedToF32Op::create(rewriter, loc, dstType,
                                                       nonUnitDimSubview[0]);
    vector::FMAOp fma =
        vector::FMAOp::create(rewriter, loc, loadBcstEvenIndxElementToF32,
                              loadEvenIndxElementF32, oddIndxFMA);

    auto castFma = vector::ShapeCastOp::create(rewriter, loc, accTy, fma);
    rewriter.replaceOp(contractOp, castFma);
    return success();
  }
};

void x86vector::populateVectorContractBF16ToFMAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractBF16ToFMA>(patterns.getContext());
}
