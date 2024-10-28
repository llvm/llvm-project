

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_RAISEMEMREFDIALECT
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "raise-memref-to-affine"

using namespace mlir;
using namespace mlir::affine;

namespace {

static std::optional<size_t>
findInListOrAdd(Value value, llvm::SmallVectorImpl<Value> &dims,
                const std::function<bool(Value)> &isValidElement) {

  Value *loopIV = std::find(dims.begin(), dims.end(), value);
  if (loopIV != dims.end()) {
    // found an IV that already has an index
    return {std::distance(dims.begin(), loopIV)};
  }
  if (isValidElement(value)) {
    // push this IV in the parameters
    size_t idx = dims.size();
    dims.push_back(value);
    return idx;
  }
  return std::nullopt;
}

static LogicalResult toAffineExpr(Value value, AffineExpr &result,
                                  llvm::SmallVectorImpl<Value> &affineDims,
                                  llvm::SmallVectorImpl<Value> &affineSymbols) {
  using namespace matchers;
  IntegerAttr::ValueType cst;
  if (matchPattern(value, m_ConstantInt(&cst))) {
    result = getAffineConstantExpr(cst.getSExtValue(), value.getContext());
    return success();
  }
  Value lhs;
  Value rhs;
  if (matchPattern(value, m_Op<arith::AddIOp>(m_Any(&lhs), m_Any(&rhs))) ||
      matchPattern(value, m_Op<arith::MulIOp>(m_Any(&lhs), m_Any(&rhs)))) {
    AffineExpr lhsE;
    AffineExpr rhsE;
    if (succeeded(toAffineExpr(lhs, lhsE, affineDims, affineSymbols)) &&
        succeeded(toAffineExpr(rhs, rhsE, affineDims, affineSymbols))) {
      AffineExprKind kind;
      if (isa<arith::AddIOp>(value.getDefiningOp())) {
        kind = mlir::AffineExprKind::Add;
      } else {
        kind = mlir::AffineExprKind::Mul;
      }
      result = getAffineBinaryOpExpr(kind, lhsE, rhsE);
      return success();
    }
  }

  if (auto dimIx = findInListOrAdd(value, affineSymbols, [](Value v) {
        return affine::isValidSymbol(v);
      })) {
    result = getAffineSymbolExpr(*dimIx, value.getContext());
    return success();
  }

  if (auto dimIx = findInListOrAdd(
          value, affineDims, [](Value v) { return affine::isValidDim(v); })) {

    result = getAffineDimExpr(*dimIx, value.getContext());
    return success();
  }

  return failure();
}

static LogicalResult
computeAffineMapAndArgs(MLIRContext *ctx, ValueRange indices, AffineMap &map,
                        llvm::SmallVectorImpl<Value> &mapArgs) {
  llvm::SmallVector<AffineExpr> results;
  llvm::SmallVector<Value, 2> symbols;
  llvm::SmallVector<Value, 8> dims;

  for (auto indexExpr : indices) {
    if (failed(
            toAffineExpr(indexExpr, results.emplace_back(), dims, symbols))) {
      return failure();
    }
  }

  map = AffineMap::get(dims.size(), symbols.size(), results, ctx);

  dims.append(symbols);
  mapArgs.swap(dims);
  return success();
}

struct RaiseMemrefDialect
    : public affine::impl::RaiseMemrefDialectBase<RaiseMemrefDialect> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    Operation *op = getOperation();
    IRRewriter rewriter(ctx);
    AffineMap map;
    SmallVector<Value> mapArgs;
    op->walk([&](Operation *op) {
      rewriter.setInsertionPoint(op);
      if (auto store = llvm::dyn_cast_or_null<memref::StoreOp>(op)) {

        if (succeeded(computeAffineMapAndArgs(ctx, store.getIndices(), map,
                                              mapArgs))) {
          rewriter.replaceOpWithNewOp<AffineStoreOp>(
              op, store.getValueToStore(), store.getMemRef(), map, mapArgs);
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "[affine] Cannot raise memref op: " << op << "\n");
        }

      } else if (auto load = llvm::dyn_cast_or_null<memref::LoadOp>(op)) {

        if (succeeded(computeAffineMapAndArgs(ctx, load.getIndices(), map,
                                              mapArgs))) {
          rewriter.replaceOpWithNewOp<AffineLoadOp>(op, load.getMemRef(), map,
                                                    mapArgs);
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "[affine] Cannot raise memref op: " << op << "\n");
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createRaiseMemrefToAffine() {
  return std::make_unique<RaiseMemrefDialect>();
}
