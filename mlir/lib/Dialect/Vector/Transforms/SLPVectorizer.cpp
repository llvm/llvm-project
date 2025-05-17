//===- SLPVectorizer.cpp - SLP Vectorizer Pass ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SLP vectorizer pass for MLIR. The pass attempts to
// combine similar independent operations into vector operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "slp-vectorizer"

namespace mlir {
namespace vector {
#define GEN_PASS_DEF_SLPVECTORIZER
#include "mlir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace vector
} // namespace mlir

using namespace mlir;
using namespace mlir::vector;

namespace {
/// A group of consecutive memory operations of the same type (load or store)
/// that can potentially be vectorized together.
struct MemoryOpGroup {
  enum class Type { Load, Store };
  Type type;
  SmallVector<Operation *> ops;

  MemoryOpGroup(Type t) : type(t) {}

  bool isLoadGroup() const { return type == Type::Load; }
  bool isStoreGroup() const { return type == Type::Store; }

  size_t size() const { return ops.size(); }
  bool empty() const { return ops.empty(); }
};

/// This pass implements the SLP vectorizer. It detects consecutive operations
/// that can be put together into vector operations. The pass works bottom-up,
/// across basic blocks, in search of scalars to combine.
struct SLPVectorizerPass
    : public mlir::vector::impl::SLPVectorizerBase<SLPVectorizerPass> {
  void runOnOperation() override;

private:
  /// Collect all memory operations in the block into groups.
  /// Each group contains either all loads or all stores, uninterrupted by
  /// operations of the other type.
  SmallVector<MemoryOpGroup> collectMemoryOpGroups(Block &block);
};

} // namespace

SmallVector<MemoryOpGroup>
SLPVectorizerPass::collectMemoryOpGroups(Block &block) {
  SmallVector<MemoryOpGroup> groups;
  MemoryOpGroup *currentGroup = nullptr;

  LLVM_DEBUG(llvm::dbgs() << "Scanning block for memory operations...\n");

  for (Operation &op : block) {
    LLVM_DEBUG(llvm::dbgs() << "Checking operation: " << op.getName() << "\n");

    // Skip non-memory operations
    if (!isa<memref::LoadOp, memref::StoreOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "  Not a memory operation\n");
      continue;
    }

    bool isLoad = isa<memref::LoadOp>(op);
    MemoryOpGroup::Type type =
        isLoad ? MemoryOpGroup::Type::Load : MemoryOpGroup::Type::Store;

    LLVM_DEBUG(llvm::dbgs()
               << "  Found " << (isLoad ? "load" : "store") << " operation\n");

    // Start a new group if:
    // 1. We don't have a current group, or
    // 2. The current operation is a different type than the current group
    if (!currentGroup || currentGroup->type != type) {
      LLVM_DEBUG(llvm::dbgs() << "  Starting new group\n");
      groups.emplace_back(type);
      currentGroup = &groups.back();
    }

    currentGroup->ops.push_back(&op);
  }

  // Remove empty groups
  groups.erase(std::remove_if(groups.begin(), groups.end(),
                              [](const MemoryOpGroup &g) { return g.empty(); }),
               groups.end());

  LLVM_DEBUG({
    llvm::dbgs() << "Found " << groups.size() << " memory operation groups:\n";
    for (const auto &group : groups) {
      llvm::dbgs() << "  Group type: "
                   << (group.isLoadGroup() ? "Load" : "Store")
                   << ", size: " << group.size() << "\n";
    }
  });

  return groups;
}

void SLPVectorizerPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = &getContext();

  LLVM_DEBUG(llvm::dbgs() << "Running SLP Vectorizer pass on operation: "
                          << op->getName() << "\n");

  // Process each function in the module
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        // If this is a function, process its body
        if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Processing function: " << funcOp.getName() << "\n");

          // Process each block in the function
          for (Block &funcBlock : funcOp.getBody()) {
            // Collect memory operation groups
            SmallVector<MemoryOpGroup> groups =
                collectMemoryOpGroups(funcBlock);

            LLVM_DEBUG({
              llvm::dbgs() << "Found " << groups.size()
                           << " memory operation groups:\n";
              for (const auto &group : groups) {
                llvm::dbgs() << "  Group type: "
                             << (group.isLoadGroup() ? "Load" : "Store")
                             << ", size: " << group.size() << "\n";
              }
            });
          }
        }
      }
    }
  }

  llvm::errs() << "Running SLP Vectorizer pass\n";
}

std::unique_ptr<Pass> mlir::vector::createSLPVectorizerPass() {
  return std::make_unique<SLPVectorizerPass>();
}
