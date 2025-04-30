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

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Operation.h"
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

        if (succeeded(affine::convertValuesToAffineMapAndArgs(
                ctx, store.getIndices(), map, mapArgs))) {
          rewriter.replaceOpWithNewOp<AffineStoreOp>(
              op, store.getValueToStore(), store.getMemRef(), map, mapArgs);
          return;
        }

        LLVM_DEBUG(llvm::dbgs()
                   << "[affine] Cannot raise memref op: " << op << "\n");

      } else if (auto load = llvm::dyn_cast_or_null<memref::LoadOp>(op)) {
        if (succeeded(affine::convertValuesToAffineMapAndArgs(
                ctx, load.getIndices(), map, mapArgs))) {
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

std::unique_ptr<AffineScopePassBase> mlir::affine::createRaiseMemrefToAffine() {
  return std::make_unique<RaiseMemrefDialect>();
}
