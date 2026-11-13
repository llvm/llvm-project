//===- DeduplicateIncludes.cpp - Deduplicate standard includes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace emitc {
#define GEN_PASS_DEF_DEDUPLICATEINCLUDESPASS
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"
} // namespace emitc
} // namespace mlir

using namespace mlir;
using namespace mlir::emitc;

namespace {
void deduplicateIncludes(Operation *root) {
  llvm::DenseSet<StringAttr> headers;
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    // File bodies can be emitted separately, so they need their own set.
    if (op != root && isa<FileOp>(op)) {
      deduplicateIncludes(op);
      return WalkResult::skip();
    }

    // Visit each include once, in emission order. The set contains exactly the
    // standard header names already encountered in this output container.
    if (auto include = dyn_cast<IncludeOp>(op)) {
      if (include.getIsStandardInclude() &&
          !headers.insert(include.getIncludeAttr()).second) {
        include.erase();
        return WalkResult::skip();
      }
    }
    return WalkResult::advance();
  });
}

struct DeduplicateIncludesPass
    : emitc::impl::DeduplicateIncludesPassBase<DeduplicateIncludesPass> {
  void runOnOperation() override { deduplicateIncludes(getOperation()); }
};
} // namespace
