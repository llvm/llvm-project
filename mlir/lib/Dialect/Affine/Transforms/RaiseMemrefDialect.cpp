//===- RaiseMemrefDialect.cpp - raise memref.store and load to affine ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functionality to convert memref load and store ops to
// the corresponding affine ops, inferring the affine map as needed.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

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

/// Find the index of the given value in the `dims` list,
/// and append it if it was not already in the list. The
/// dims list is a list of symbols or dimensions of the
/// affine map. Within the results of an affine map, they
/// are identified by their index, which is why we need
/// this function.
static std::optional<size_t>
findInListOrAdd(Value value, llvm::SmallVectorImpl<Value> &dims,
                function_ref<bool(Value)> isValidElement) {

  Value *loopIV = std::find(dims.begin(), dims.end(), value);
  if (loopIV != dims.end()) {
    // We found an IV that already has an index, return that index.
    return {std::distance(dims.begin(), loopIV)};
  }
  if (isValidElement(value)) {
    // This is a valid element for the dim/symbol list, push this as a
    // parameter.
    size_t idx = dims.size();
    dims.push_back(value);
    return idx;
  }
  return std::nullopt;
}

/// Convert a value to an affine expr if possible. Adds dims and symbols
/// if needed.
static AffineExpr toAffineExpr(Value value,
                               llvm::SmallVectorImpl<Value> &affineDims,
                               llvm::SmallVectorImpl<Value> &affineSymbols) {
  using namespace matchers;
  IntegerAttr::ValueType cst;
  if (matchPattern(value, m_ConstantInt(&cst))) {
    return getAffineConstantExpr(cst.getSExtValue(), value.getContext());
  }

  Operation *definingOp = value.getDefiningOp();
  if (llvm::isa_and_nonnull<arith::AddIOp>(definingOp) ||
      llvm::isa_and_nonnull<arith::MulIOp>(definingOp)) {
    // TODO: replace recursion with explicit stack.
    // For the moment this can be tolerated as we only recurse on
    // arith.addi and arith.muli, so there cannot be any infinite
    // recursion. The depth of these expressions should be in most
    // cases very manageable, as affine expressions should be as
    // simple as `a + b * c`.
    AffineExpr lhsE =
        toAffineExpr(definingOp->getOperand(0), affineDims, affineSymbols);
    AffineExpr rhsE =
        toAffineExpr(definingOp->getOperand(1), affineDims, affineSymbols);

    if (lhsE && rhsE) {
      AffineExprKind kind;
      if (isa<arith::AddIOp>(definingOp)) {
        kind = mlir::AffineExprKind::Add;
      } else {
        kind = mlir::AffineExprKind::Mul;

        if (!lhsE.isSymbolicOrConstant() && !rhsE.isSymbolicOrConstant()) {
          // This is not an affine expression, give up.
          return {};
        }
      }
      return getAffineBinaryOpExpr(kind, lhsE, rhsE);
    }
    return {};
  }

  if (auto dimIx = findInListOrAdd(value, affineSymbols, [](Value v) {
        return affine::isValidSymbol(v);
      })) {
    return getAffineSymbolExpr(*dimIx, value.getContext());
  }

  if (auto dimIx = findInListOrAdd(
          value, affineDims, [](Value v) { return affine::isValidDim(v); })) {

    return getAffineDimExpr(*dimIx, value.getContext());
  }

  return {};
}

static LogicalResult
computeAffineMapAndArgs(MLIRContext *ctx, ValueRange indices, AffineMap &map,
                        llvm::SmallVectorImpl<Value> &mapArgs) {
  SmallVector<AffineExpr> results;
  SmallVector<Value> symbols;
  SmallVector<Value> dims;

  for (Value indexExpr : indices) {
    AffineExpr res = toAffineExpr(indexExpr, dims, symbols);
    if (!res) {
      return failure();
    }
    results.push_back(res);
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
          return;
        }

        LLVM_DEBUG(llvm::dbgs()
                   << "[affine] Cannot raise memref op: " << op << "\n");

      } else if (auto load = llvm::dyn_cast_or_null<memref::LoadOp>(op)) {
        if (succeeded(computeAffineMapAndArgs(ctx, load.getIndices(), map,
                                              mapArgs))) {
          rewriter.replaceOpWithNewOp<AffineLoadOp>(op, load.getMemRef(), map,
                                                    mapArgs);
          return;
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "[affine] Cannot raise memref op: " << op << "\n");
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createRaiseMemrefToAffine() {
  return std::make_unique<RaiseMemrefDialect>();
}
