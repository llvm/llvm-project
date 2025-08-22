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

  Value *loopIV = llvm::find(dims, value);
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
static AffineExpr toAffineExpr(Value root,
                               llvm::SmallVectorImpl<Value> &affineDims,
                               llvm::SmallVectorImpl<Value> &affineSymbols) {
  using namespace matchers;

  // Table for already-built subexpressions.
  llvm::DenseMap<Value, AffineExpr> built;

  // Post-order traversal stack: (value, state)
  // state = 0 -> first-time seen (push the children).
  // state = 1 -> children have been processed (build the node).
  llvm::SmallVector<std::pair<Value, unsigned>, 16> stack;
  stack.emplace_back(root, 0); // push the root value onto the stack.

  auto makeLeaf = [&](Value v) -> AffineExpr {
    // Constant?
    IntegerAttr::ValueType cst;
    if (matchPattern(root, m_ConstantInt(&cst)))
      return getAffineConstantExpr(cst.getSExtValue(), v.getContext());

    // Symbol?
    if (auto symIx = findInListOrAdd(
            v, affineSymbols, [](Value x) { return isValidSymbol(x); })) {
      return getAffineSymbolExpr(*symIx, v.getContext());
    }

    // Dimension?
    if (auto dimIx = findInListOrAdd(v, affineDims,
                                     [](Value x) { return isValidDim(x); })) {
      return getAffineDimExpr(*dimIx, v.getContext());
    }

    // Not representable.
    return AffineExpr();
  };

  while (!stack.empty()) {
    auto [v, state] = stack.back();
    stack.pop_back();

    // If we already built the current value, nothing more to do.
    if (built.count(v))
      continue;

    if (state == 0) {
      Operation *def = v.getDefiningOp();

      // If not an addi/muli, we'll handle it as a leaf in state == 1,
      bool isAddOrMul = llvm::isa_and_nonnull<arith::AddIOp>(def) ||
                        llvm::isa_and_nonnull<arith::MulIOp>(def);

      if (!isAddOrMul) {
        // Defer to leaf handling.
        stack.emplace_back(v, 1);
        continue;
      }

      // For addi/muli, push the ops.
      stack.emplace_back(v, 1);
      Value lhs = def->getOperand(0);
      Value rhs = def->getOperand(1);

      if (!built.count(lhs))
        stack.emplace_back(lhs, 0);
      if (!built.count(rhs))
        stack.emplace_back(rhs, 0);
      continue;
    }
    // state == 1, time to build the node (either add/mul or a leaf).
    Operation *def = v.getDefiningOp();
    if (llvm::isa_and_nonnull<arith::AddIOp>(def) ||
        llvm::isa_and_nonnull<arith::MulIOp>(def)) {
      Value lhsV = def->getOperand(0);
      Value rhsV = def->getOperand(1);

      auto itL = built.find(lhsV);
      auto itR = built.find(rhsV);
      if (itL == built.end() || itR == built.end()) {
        // Child failed to build.
        return AffineExpr();
      }

      AffineExpr lhsE = itL->second;
      AffineExpr rhsE = itR->second;

      if (!lhsE || !rhsE)
        return AffineExpr();

      AffineExprKind kind;
      if (llvm::isa<arith::AddIOp>(def)) {
        kind = AffineExprKind::Add;
      } else {
        kind = AffineExprKind::Mul;
        // Enforce restriction: one side must by symbolic or constant.
        if (!lhsE.isSymbolicOrConstant() && !rhsE.isSymbolicOrConstant())
          return AffineExpr();
      }

      built[v] = getAffineBinaryOpExpr(kind, lhsE, rhsE);
      continue;
    }

    // Not addi/muli: treat as leaf.
    AffineExpr leaf = makeLeaf(v);
    if (!leaf)
      return AffineExpr();
    built[v] = leaf;
  }
  auto it = built.find(root);
  return it == built.end() ? AffineExpr()
                           : it->second; // Return the root expression.
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
