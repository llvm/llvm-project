//===- LinalgToStandard.cpp - conversion from Linalg to Standard dialect --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::linalg;

/// Helper function to extract the operand types that are passed to the
/// generated CallOp. MemRefTypes have their layout canonicalized since the
/// information is not used in signature generation.
/// Note that static size information is not modified.
static SmallVector<Type, 4> extractOperandTypes(Operation *op) {
  SmallVector<Type, 4> result;
  result.reserve(op->getNumOperands());
  if (auto indexedGenericOp = dyn_cast<IndexedGenericOp>(op)) {
    auto *ctx = op->getContext();
    auto numLoops = indexedGenericOp.getNumLoops();
    result.reserve(op->getNumOperands() + numLoops);
    result.assign(numLoops, IndexType::get(ctx));
  }
  for (auto type : op->getOperandTypes()) {
    // The underlying descriptor type (e.g. LLVM) does not have layout
    // information. Canonicalizing the type at the level of std when going into
    // a library call avoids needing to introduce DialectCastOp.
    if (auto memrefType = type.dyn_cast<MemRefType>())
      result.push_back(eraseStridedLayout(memrefType));
    else
      result.push_back(type);
  }
  return result;
}

// Get a SymbolRefAttr containing the library function name for the LinalgOp.
// If the library function does not exist, insert a declaration.
static FlatSymbolRefAttr getLibraryCallSymbolRef(Operation *op,
                                                 PatternRewriter &rewriter) {
  auto linalgOp = cast<LinalgOp>(op);
  auto fnName = linalgOp.getLibraryCallName();
  if (fnName.empty()) {
    op->emitWarning("No library call defined for: ") << *op;
    return {};
  }

  // fnName is a dynamic std::string, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr = rewriter.getSymbolRefAttr(fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnName)) {
    return fnNameAttr;
  }

  SmallVector<Type, 4> inputTypes(extractOperandTypes(op));
  assert(op->getNumResults() == 0 &&
         "Library call for linalg operation can be generated only for ops that "
         "have void return types");
  auto libFnType = rewriter.getFunctionType(inputTypes, {});

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  FuncOp funcOp =
      rewriter.create<FuncOp>(op->getLoc(), fnNameAttr.getValue(), libFnType);
  // Insert a function attribute that will trigger the emission of the
  // corresponding `_mlir_ciface_xxx` interface so that external libraries see
  // a normalized ABI. This interface is added during std to llvm conversion.
  funcOp->setAttr("llvm.emit_c_interface", UnitAttr::get(op->getContext()));
  funcOp.setPrivate();
  return fnNameAttr;
}

static SmallVector<Value, 4>
createTypeCanonicalizedMemRefOperands(OpBuilder &b, Location loc,
                                      ValueRange operands) {
  SmallVector<Value, 4> res;
  res.reserve(operands.size());
  for (auto op : operands) {
    auto memrefType = op.getType().dyn_cast<MemRefType>();
    if (!memrefType) {
      res.push_back(op);
      continue;
    }
    Value cast =
        b.create<memref::CastOp>(loc, eraseStridedLayout(memrefType), op);
    res.push_back(cast);
  }
  return res;
}

LogicalResult mlir::linalg::LinalgOpToLibraryCallRewrite::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  // Only LinalgOp for which there is no specialized pattern go through this.
  if (!isa<LinalgOp>(op) || isa<CopyOp>(op) || isa<IndexedGenericOp>(op))
    return failure();

  auto libraryCallName = getLibraryCallSymbolRef(op, rewriter);
  if (!libraryCallName)
    return failure();

  rewriter.replaceOpWithNewOp<mlir::CallOp>(
      op, libraryCallName.getValue(), TypeRange(),
      createTypeCanonicalizedMemRefOperands(rewriter, op->getLoc(),
                                            op->getOperands()));
  return success();
}

LogicalResult mlir::linalg::CopyOpToLibraryCallRewrite::matchAndRewrite(
    CopyOp op, PatternRewriter &rewriter) const {
  auto inputPerm = op.inputPermutation();
  if (inputPerm.hasValue() && !inputPerm->isIdentity())
    return failure();
  auto outputPerm = op.outputPermutation();
  if (outputPerm.hasValue() && !outputPerm->isIdentity())
    return failure();

  auto libraryCallName = getLibraryCallSymbolRef(op, rewriter);
  if (!libraryCallName)
    return failure();

  rewriter.replaceOpWithNewOp<mlir::CallOp>(
      op, libraryCallName.getValue(), TypeRange(),
      createTypeCanonicalizedMemRefOperands(rewriter, op.getLoc(),
                                            op.getOperands()));
  return success();
}

LogicalResult mlir::linalg::CopyTransposeRewrite::matchAndRewrite(
    CopyOp op, PatternRewriter &rewriter) const {
  Value in = op.input(), out = op.output();

  // If either inputPerm or outputPerm are non-identities, insert transposes.
  auto inputPerm = op.inputPermutation();
  if (inputPerm.hasValue() && !inputPerm->isIdentity())
    in = rewriter.create<memref::TransposeOp>(op.getLoc(), in,
                                              AffineMapAttr::get(*inputPerm));
  auto outputPerm = op.outputPermutation();
  if (outputPerm.hasValue() && !outputPerm->isIdentity())
    out = rewriter.create<memref::TransposeOp>(op.getLoc(), out,
                                               AffineMapAttr::get(*outputPerm));

  // If nothing was transposed, fail and let the conversion kick in.
  if (in == op.input() && out == op.output())
    return failure();

  auto libraryCallName = getLibraryCallSymbolRef(op, rewriter);
  if (!libraryCallName)
    return failure();

  rewriter.replaceOpWithNewOp<mlir::CallOp>(
      op, libraryCallName.getValue(), TypeRange(),
      createTypeCanonicalizedMemRefOperands(rewriter, op.getLoc(), {in, out}));
  return success();
}

LogicalResult
mlir::linalg::IndexedGenericOpToLibraryCallRewrite::matchAndRewrite(
    IndexedGenericOp op, PatternRewriter &rewriter) const {
  auto libraryCallName = getLibraryCallSymbolRef(op, rewriter);
  if (!libraryCallName)
    return failure();

  // TODO: Use induction variables values instead of zeros, when
  // IndexedGenericOp is tiled.
  auto zero = rewriter.create<mlir::ConstantOp>(
      op.getLoc(), rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  auto indexedGenericOp = cast<IndexedGenericOp>(op);
  auto numLoops = indexedGenericOp.getNumLoops();
  SmallVector<Value, 4> operands;
  operands.reserve(numLoops + op.getNumOperands());
  for (unsigned i = 0; i < numLoops; ++i)
    operands.push_back(zero);
  for (auto operand : op.getOperands())
    operands.push_back(operand);
  rewriter.replaceOpWithNewOp<mlir::CallOp>(
      op, libraryCallName.getValue(), TypeRange(),
      createTypeCanonicalizedMemRefOperands(rewriter, op.getLoc(), operands));
  return success();
}

/// Populate the given list with patterns that convert from Linalg to Standard.
void mlir::linalg::populateLinalgToStandardConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  // TODO: ConvOp conversion needs to export a descriptor with relevant
  // attribute values such as kernel striding and dilation.
  // clang-format off
  patterns.insert<
      CopyOpToLibraryCallRewrite,
      CopyTransposeRewrite,
      IndexedGenericOpToLibraryCallRewrite>(ctx);
  patterns.insert<LinalgOpToLibraryCallRewrite>();
  // clang-format on
}

namespace {
struct ConvertLinalgToStandardPass
    : public ConvertLinalgToStandardBase<ConvertLinalgToStandardPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertLinalgToStandardPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, memref::MemRefDialect, scf::SCFDialect,
                         StandardOpsDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ModuleTerminatorOp, ReturnOp>();
  target.addLegalOp<linalg::ReshapeOp, linalg::RangeOp>();
  OwningRewritePatternList patterns;
  populateLinalgToStandardConversionPatterns(patterns, &getContext());
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgToStandardPass() {
  return std::make_unique<ConvertLinalgToStandardPass>();
}
