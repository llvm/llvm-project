//===- LegalizeForLLVMExport.cpp - Prepare X86Vector for LLVM translation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/X86Vector/Transforms.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::x86vector;

namespace {

/// Replaces an operation with a call to an LLVM intrinsic with the specified
/// name and operands.
///
/// The rewrite performs a simple one-to-one matching between the op and LLVM
/// intrinsic. For example:
///
/// ```mlir
/// %res = x86vector.op %val : vector<16xf32>
/// ```
///
/// can be converted to
///
/// ```mlir
/// %res = llvm.call_intrinsic "intrinsic"(%val)
/// ```
///
/// The provided operands must be LLVM-compatible.
///
/// Upholds a convention that multi-result operations get converted into an
/// operation returning the LLVM IR structure type, in which case individual
/// values are first extracted before replacing the original results.
LogicalResult intrinsicRewrite(Operation *op, StringAttr intrinsic,
                               ValueRange operands,
                               const LLVMTypeConverter &typeConverter,
                               PatternRewriter &rewriter) {
  auto loc = op->getLoc();

  if (!llvm::all_of(operands, [](Value value) {
        return LLVM::isCompatibleType(value.getType());
      }))
    return rewriter.notifyMatchFailure(op, "Expects LLVM-compatible types.");

  unsigned numResults = op->getNumResults();
  Type resType;
  if (numResults != 0)
    resType = typeConverter.packOperationResults(op->getResultTypes());

  auto callIntrOp =
      rewriter.create<LLVM::CallIntrinsicOp>(loc, resType, intrinsic, operands);
  // Propagate attributes.
  callIntrOp->setAttrs(op->getAttrDictionary());

  if (numResults <= 1) {
    // Directly replace the original op.
    rewriter.replaceOp(op, callIntrOp);
    return success();
  }

  // Extract individual results from packed structure and use them as
  // replacements.
  SmallVector<Value, 4> results;
  results.reserve(numResults);
  Value intrRes = callIntrOp.getResults();
  for (unsigned i = 0; i < numResults; ++i) {
    results.push_back(rewriter.create<LLVM::ExtractValueOp>(loc, intrRes, i));
  }
  rewriter.replaceOp(op, results);

  return success();
}

/// Generic one-to-one conversion of simply mappable operations into calls
/// to their respective LLVM intrinsics.
struct OneToOneIntrinsicOpConversion
    : public OpInterfaceRewritePattern<x86vector::OneToOneIntrinsicOp> {
  using OpInterfaceRewritePattern<
      x86vector::OneToOneIntrinsicOp>::OpInterfaceRewritePattern;

  OneToOneIntrinsicOpConversion(const LLVMTypeConverter &typeConverter,
                                PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(&typeConverter.getContext(), benefit),
        typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(x86vector::OneToOneIntrinsicOp op,
                                PatternRewriter &rewriter) const override {
    return intrinsicRewrite(op, rewriter.getStringAttr(op.getIntrinsicName()),
                            op.getIntrinsicOperands(rewriter, typeConverter),
                            typeConverter, rewriter);
  }

private:
  const LLVMTypeConverter &typeConverter;
};

} // namespace

/// Populate the given list with patterns that convert from X86Vector to LLVM.
void mlir::populateX86VectorLegalizeForLLVMExportPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<OneToOneIntrinsicOpConversion>(converter);
}

void mlir::configureX86VectorLegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addIllegalOp<MaskCompressOp, MaskRndScaleOp, MaskScaleFOp,
                      Vp2IntersectOp, DotBF16Op, CvtPackedF32ToBF16Op,
                      CvtPackedEvenIndexedToF32Op, CvtPackedOddIndexedToF32Op,
                      BcstToPackedF32Op, RsqrtOp, DotOp>();
}
