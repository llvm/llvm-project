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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::linalg;

/// Helper function to extract the operand types that are passed to the
/// generated CallOp. MemRefTypes have their layout canonicalized since the
/// information is not used in signature generation.
/// Note that static size information is not modified.
template <typename LinalgOp>
static SmallVector<Type, 4> extractOperandTypes(Operation *op) {
  SmallVector<Type, 4> result;
  result.reserve(op->getNumOperands());
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

template <>
SmallVector<Type, 4> extractOperandTypes<IndexedGenericOp>(Operation *op) {
  auto *ctx = op->getContext();
  auto indexedGenericOp = cast<IndexedGenericOp>(op);
  auto numLoops = indexedGenericOp.getNumLoops();

  SmallVector<Type, 4> result(numLoops, IndexType::get(ctx));
  auto canonicalizedOperands = extractOperandTypes<LinalgOp>(op);
  result.append(canonicalizedOperands.begin(), canonicalizedOperands.end());
  return result;
}

// Get a SymbolRefAttr containing the library function name for the LinalgOp.
// If the library function does not exist, insert a declaration.
template <typename LinalgOp>
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

  SmallVector<Type, 4> inputTypes(extractOperandTypes<LinalgOp>(op));
  assert(op->getNumResults() == 0 &&
         "Library call for linalg operation can be generated only for ops that "
         "have void return types");
  auto libFnType = FunctionType::get(inputTypes, {}, rewriter.getContext());

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  FuncOp funcOp =
      rewriter.create<FuncOp>(op->getLoc(), fnNameAttr.getValue(), libFnType,
                              ArrayRef<NamedAttribute>{});
  // Insert a function attribute that will trigger the emission of the
  // corresponding `_mlir_ciface_xxx` interface so that external libraries see
  // a normalized ABI. This interface is added during std to llvm conversion.
  funcOp.setAttr("llvm.emit_c_interface", UnitAttr::get(op->getContext()));
  return fnNameAttr;
}

namespace {

SmallVector<Value, 4>
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
        b.create<MemRefCastOp>(loc, eraseStridedLayout(memrefType), op);
    res.push_back(cast);
  }
  return res;
}

// LinalgOpConversion<LinalgOp> creates a new call to the type-canonicalized
// `LinalgOp::getLibraryCallName()` function.
// The implementation of the function can be either in the same module or in an
// externally linked library.
template <typename LinalgOp>
class LinalgOpConversion : public OpRewritePattern<LinalgOp> {
public:
  using OpRewritePattern<LinalgOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    auto libraryCallName = getLibraryCallSymbolRef<LinalgOp>(op, rewriter);
    if (!libraryCallName)
      return failure();

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, libraryCallName.getValue(), ArrayRef<Type>{},
        createTypeCanonicalizedMemRefOperands(rewriter, op.getLoc(),
                                              op.getOperands()));
    return success();
  }
};

/// Conversion pattern specialization for CopyOp. This kicks in when both input
/// and output permutations are left unspecified or are the identity.
template <>
class LinalgOpConversion<CopyOp> : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto inputPerm = op.inputPermutation();
    if (inputPerm.hasValue() && !inputPerm->isIdentity())
      return failure();
    auto outputPerm = op.outputPermutation();
    if (outputPerm.hasValue() && !outputPerm->isIdentity())
      return failure();

    auto libraryCallName = getLibraryCallSymbolRef<CopyOp>(op, rewriter);
    if (!libraryCallName)
      return failure();

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, libraryCallName.getValue(), ArrayRef<Type>{},
        createTypeCanonicalizedMemRefOperands(rewriter, op.getLoc(),
                                              op.getOperands()));
    return success();
  }
};

/// Conversion pattern specialization for IndexedGenericOp.
template <>
class LinalgOpConversion<IndexedGenericOp>
    : public OpRewritePattern<IndexedGenericOp> {
public:
  using OpRewritePattern<IndexedGenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexedGenericOp op,
                                PatternRewriter &rewriter) const override {
    auto libraryCallName =
        getLibraryCallSymbolRef<IndexedGenericOp>(op, rewriter);
    if (!libraryCallName)
      return failure();

    // TODO(pifon, ntv): Use induction variables values instead of zeros, when
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
        op, libraryCallName.getValue(), ArrayRef<Type>{},
        createTypeCanonicalizedMemRefOperands(rewriter, op.getLoc(), operands));
    return success();
  }
};

/// A non-conversion rewrite pattern kicks in to convert CopyOp with
/// permutations into a sequence of TransposeOp and permutation-free CopyOp.
/// This interplays together with TransposeOpConversion and
/// LinalgConversion<CopyOp> to create a path to the LLVM dialect.
class CopyTransposeConversion : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.input(), out = op.output();

    // If either inputPerm or outputPerm are non-identities, insert transposes.
    auto inputPerm = op.inputPermutation();
    if (inputPerm.hasValue() && !inputPerm->isIdentity())
      in = rewriter.create<linalg::TransposeOp>(op.getLoc(), in,
                                                AffineMapAttr::get(*inputPerm));
    auto outputPerm = op.outputPermutation();
    if (outputPerm.hasValue() && !outputPerm->isIdentity())
      out = rewriter.create<linalg::TransposeOp>(
          op.getLoc(), out, AffineMapAttr::get(*outputPerm));

    // If nothing was transposed, fail and let the conversion kick in.
    if (in == op.input() && out == op.output())
      return failure();

    rewriter.replaceOpWithNewOp<CopyOp>(op, in, out);
    return success();
  }
};
} // namespace

/// Populate the given list with patterns that convert from Linalg to Standard.
void mlir::populateLinalgToStandardConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  // TODO(ntv) ConvOp conversion needs to export a descriptor with relevant
  // attribute values such as kernel striding and dilation.
  // clang-format off
  patterns.insert<
      CopyTransposeConversion,
      LinalgOpConversion<ConvOp>,
      LinalgOpConversion<PoolingMaxOp>,
      LinalgOpConversion<PoolingMinOp>,
      LinalgOpConversion<PoolingSumOp>,
      LinalgOpConversion<CopyOp>,
      LinalgOpConversion<DotOp>,
      LinalgOpConversion<FillOp>,
      LinalgOpConversion<GenericOp>,
      LinalgOpConversion<IndexedGenericOp>>(ctx);
  // TODO: collect all auto-generated named ops with a tblgen directive.
  patterns.insert<
      LinalgOpConversion<BatchMatmulOp>,
      LinalgOpConversion<MatvecOp>,
      LinalgOpConversion<MatmulOp>>(ctx);
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
  target.addLegalDialect<AffineDialect, scf::SCFDialect, StandardOpsDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ModuleTerminatorOp, ReturnOp>();
  target.addLegalOp<linalg::TransposeOp, linalg::ReshapeOp, linalg::RangeOp>();
  OwningRewritePatternList patterns;
  populateLinalgToStandardConversionPatterns(patterns, &getContext());
  if (failed(applyFullConversion(module, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgToStandardPass() {
  return std::make_unique<ConvertLinalgToStandardPass>();
}
