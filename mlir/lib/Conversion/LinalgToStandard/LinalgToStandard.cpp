//===- LinalgToStandard.cpp - conversion from Linalg to Standard dialect --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTOSTANDARD
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

static MemRefType makeStridedLayoutDynamic(MemRefType type) {
  return MemRefType::Builder(type).setLayout(StridedLayoutAttr::get(
      type.getContext(), ShapedType::kDynamicStrideOrOffset,
      SmallVector<int64_t>(type.getRank(),
                           ShapedType::kDynamicStrideOrOffset)));
}

/// Helper function to extract the operand types that are passed to the
/// generated CallOp. MemRefTypes have their layout canonicalized since the
/// information is not used in signature generation.
/// Note that static size information is not modified.
static SmallVector<Type, 4> extractOperandTypes(Operation *op) {
  SmallVector<Type, 4> result;
  result.reserve(op->getNumOperands());
  for (auto type : op->getOperandTypes()) {
    // The underlying descriptor type (e.g. LLVM) does not have layout
    // information. Canonicalizing the type at the level of std when going into
    // a library call avoids needing to introduce DialectCastOp.
    if (auto memrefType = type.dyn_cast<MemRefType>())
      result.push_back(makeStridedLayoutDynamic(memrefType));
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
  FlatSymbolRefAttr fnNameAttr =
      SymbolRefAttr::get(rewriter.getContext(), fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnNameAttr.getAttr()))
    return fnNameAttr;

  SmallVector<Type, 4> inputTypes(extractOperandTypes(op));
  assert(op->getNumResults() == 0 &&
         "Library call for linalg operation can be generated only for ops that "
         "have void return types");
  auto libFnType = rewriter.getFunctionType(inputTypes, {});

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  func::FuncOp funcOp = rewriter.create<func::FuncOp>(
      op->getLoc(), fnNameAttr.getValue(), libFnType);
  // Insert a function attribute that will trigger the emission of the
  // corresponding `_mlir_ciface_xxx` interface so that external libraries see
  // a normalized ABI. This interface is added during std to llvm conversion.
  funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                  UnitAttr::get(op->getContext()));
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
        b.create<memref::CastOp>(loc, makeStridedLayoutDynamic(memrefType), op);
    res.push_back(cast);
  }
  return res;
}

LogicalResult mlir::linalg::LinalgOpToLibraryCallRewrite::matchAndRewrite(
    LinalgOp op, PatternRewriter &rewriter) const {
  auto libraryCallName = getLibraryCallSymbolRef(op, rewriter);
  if (!libraryCallName)
    return failure();

  // TODO: Add support for more complex library call signatures that include
  // indices or captured values.
  rewriter.replaceOpWithNewOp<func::CallOp>(
      op, libraryCallName.getValue(), TypeRange(),
      createTypeCanonicalizedMemRefOperands(rewriter, op->getLoc(),
                                            op->getOperands()));
  return success();
}

/// Populate the given list with patterns that convert from Linalg to Standard.
void mlir::linalg::populateLinalgToStandardConversionPatterns(
    RewritePatternSet &patterns) {
  // TODO: ConvOp conversion needs to export a descriptor with relevant
  // attribute values such as kernel striding and dilation.
  patterns.add<LinalgOpToLibraryCallRewrite>(patterns.getContext());
}

namespace {
struct ConvertLinalgToStandardPass
    : public impl::ConvertLinalgToStandardBase<ConvertLinalgToStandardPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertLinalgToStandardPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect, scf::SCFDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  RewritePatternSet patterns(&getContext());
  populateLinalgToStandardConversionPatterns(patterns);
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgToStandardPass() {
  return std::make_unique<ConvertLinalgToStandardPass>();
}
