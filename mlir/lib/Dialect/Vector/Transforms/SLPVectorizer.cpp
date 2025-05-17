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

// Helper function to extract base and index from a memory operation
std::optional<std::pair<Value, int64_t>> getBaseAndIndex(Operation *op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    if (auto value = getConstantIntValue(loadOp.getIndices().front()))
      return std::make_pair(loadOp.getMemRef(), *value);
  } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    if (auto value = getConstantIntValue(storeOp.getIndices().front()))
      return std::make_pair(storeOp.getMemRef(), *value);
  }
  return std::nullopt;
}

// Extract contiguous groups from a MemoryOpGroup
SmallVector<MemoryOpGroup> extractContiguousGroups(const MemoryOpGroup &group) {
  SmallVector<MemoryOpGroup> result;
  if (group.ops.empty())
    return result;

  // Keep track of which operations we've processed
  DenseSet<Operation *> processedOps;

  // Process each operation
  for (Operation *op : group.ops) {
    // Skip if we've already processed this operation
    if (processedOps.contains(op))
      continue;

    // Get base and index of current operation
    auto baseAndIndex = getBaseAndIndex(op);
    if (!baseAndIndex)
      continue;

    auto [base, index] = *baseAndIndex;

    // Start a new group with this operation
    result.emplace_back(group.type);
    MemoryOpGroup &currentGroup = result.back();
    currentGroup.ops.push_back(op);
    processedOps.insert(op);

    LLVM_DEBUG(llvm::dbgs() << "Starting new group at base " << base
                            << " index " << index << "\n");

    // Try to find operations with adjacent indices
    bool foundMore;
    do {
      foundMore = false;
      // Look for operations with index+1
      for (Operation *otherOp : group.ops) {
        if (processedOps.contains(otherOp))
          continue;

        auto otherBaseAndIndex = getBaseAndIndex(otherOp);
        if (!otherBaseAndIndex)
          continue;

        auto [otherBase, otherIndex] = *otherBaseAndIndex;

        // Check if this operation has the same base and adjacent index
        if (otherBase == base && otherIndex == currentGroup.ops.size()) {
          currentGroup.ops.push_back(otherOp);
          processedOps.insert(otherOp);
          foundMore = true;
          LLVM_DEBUG(llvm::dbgs()
                     << "Added operation with index " << otherIndex << "\n");
          break;
        }
      }
    } while (foundMore);
  }

  // Remove empty groups
  result.erase(std::remove_if(result.begin(), result.end(),
                              [](const MemoryOpGroup &g) { return g.empty(); }),
               result.end());

  return result;
}

/// A node in the SLP graph representing a vectorizable operation
struct SLPGraphNode {
  Operation *op;
  DenseSet<SLPGraphNode *> users;
  DenseSet<SLPGraphNode *> operands;
  bool isRoot = false;

  SLPGraphNode(Operation *op) : op(op) {}
};

/// A graph of vectorizable operations
class SLPGraph {
public:
  SLPGraph() = default;
  ~SLPGraph() {
    for (auto *node : nodes)
      delete node;
  }

  /// Add a new node to the graph
  SLPGraphNode *addNode(Operation *op) {
    nodes.push_back(new SLPGraphNode(op));
    return nodes.back();
  }

  /// Add a root node (memory operation)
  SLPGraphNode *addRoot(Operation *op) {
    auto *node = addNode(op);
    node->isRoot = true;
    return node;
  }

  /// Add a dependency edge between nodes
  void addEdge(SLPGraphNode *from, SLPGraphNode *to) {
    from->users.insert(to);
    to->operands.insert(from);
  }

  /// Get all root nodes
  SmallVector<SLPGraphNode *> getRoots() const {
    SmallVector<SLPGraphNode *> roots;
    for (auto *node : nodes)
      if (node->isRoot)
        roots.push_back(node);
    return roots;
  }

  /// Print the graph structure
  void print() const {
    llvm::dbgs() << "SLP Graph Structure:\n";
    llvm::dbgs() << "===================\n";

    // First print all roots
    llvm::dbgs() << "Roots:\n";
    for (auto *node : nodes) {
      if (!node->isRoot)
        continue;
      llvm::dbgs() << "  " << *node->op << "\n";
      llvm::dbgs() << "    Users: ";
      for (auto *user : node->users) {
        llvm::dbgs() << "\n      " << *user->op;
      }
      llvm::dbgs() << "\n";
    }

    // Then print all non-root nodes
    llvm::dbgs() << "\nNon-root nodes:\n";
    for (auto *node : nodes) {
      if (node->isRoot)
        continue;
      llvm::dbgs() << "  " << *node->op << "\n";
      llvm::dbgs() << "    Operands: ";
      for (auto *operand : node->operands) {
        llvm::dbgs() << "\n      " << *operand->op;
      }
      llvm::dbgs() << "\n    Users: ";
      for (auto *user : node->users) {
        llvm::dbgs() << "\n      " << *user->op;
      }
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "===================\n";
  }

private:
  SmallVector<SLPGraphNode *> nodes;
};

/// Build the SLP graph starting from memory operation roots
SLPGraph buildSLPGraph(const SmallVector<MemoryOpGroup> &rootGroups) {
  SLPGraph graph;
  DenseMap<Operation *, SLPGraphNode *> opToNode;

  // First, add all memory operations as roots
  for (const auto &group : rootGroups) {
    for (Operation *op : group.ops) {
      opToNode[op] = graph.addRoot(op);
    }
  }

  // Process each root group to build the graph
  for (const auto &group : rootGroups) {
    for (Operation *rootOp : group.ops) {
      // Get the value produced by this memory operation
      Value rootValue = group.isLoadGroup()
                            ? cast<memref::LoadOp>(rootOp).getResult()
                            : cast<memref::StoreOp>(rootOp).getValue();

      // Find all users of this value
      for (Operation *user : rootValue.getUsers()) {
        // Skip if we've already processed this operation
        if (opToNode.contains(user))
          continue;

        // Check if this is a vectorizable operation
        if (isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
                arith::MulFOp, arith::MulIOp>(user)) {
          // Check if at least one other operand is already in the graph
          bool hasGraphOperand = false;
          for (Value operand : user->getOperands()) {
            if (operand == rootValue)
              continue;
            if (auto *defOp = operand.getDefiningOp()) {
              if (opToNode.contains(defOp)) {
                hasGraphOperand = true;
                break;
              }
            }
          }

          // Only add the operation if it has at least one other operand in the
          // graph
          if (hasGraphOperand) {
            auto *node = graph.addNode(user);
            opToNode[user] = node;
            graph.addEdge(opToNode[rootOp], node);

            // Add edges from other operands that are in the graph
            for (Value operand : user->getOperands()) {
              if (auto *defOp = operand.getDefiningOp()) {
                if (opToNode.contains(defOp)) {
                  graph.addEdge(opToNode[defOp], node);
                }
              }
            }
          }
        }
      }
    }
  }

  return graph;
}

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

  for (Operation &op : block) {
    // Skip non-memory operations
    if (!isa<memref::LoadOp, memref::StoreOp>(op))
      continue;

    bool isLoad = isa<memref::LoadOp>(op);
    MemoryOpGroup::Type type =
        isLoad ? MemoryOpGroup::Type::Load : MemoryOpGroup::Type::Store;

    // Start a new group if:
    // 1. We don't have a current group, or
    // 2. The current operation is a different type than the current group
    if (!currentGroup || currentGroup->type != type) {
      groups.emplace_back(type);
      currentGroup = &groups.back();
    }

    currentGroup->ops.push_back(&op);
  }

  // Remove empty groups
  groups.erase(std::remove_if(groups.begin(), groups.end(),
                              [](const MemoryOpGroup &g) { return g.empty(); }),
               groups.end());

  return groups;
}

void SLPVectorizerPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = &getContext();

  // Walk all blocks recursively
  op->walk([&](Block *block) {
    LLVM_DEBUG(llvm::dbgs() << "Processing block in operation: "
                            << block->getParentOp()->getName() << "\n");

    // Collect memory operation groups
    SmallVector<MemoryOpGroup> groups = collectMemoryOpGroups(*block);

    // Process each group to find contiguous sequences
    SmallVector<MemoryOpGroup> rootGroups;
    for (const auto &group : groups) {
      SmallVector<MemoryOpGroup> contiguousGroups =
          extractContiguousGroups(group);
      LLVM_DEBUG({
        llvm::dbgs() << "Found " << contiguousGroups.size()
                     << " contiguous groups in "
                     << (group.isLoadGroup() ? "load" : "store") << " group\n";
        for (const auto &contigGroup : contiguousGroups) {
          llvm::dbgs() << "  Contiguous group with " << contigGroup.size()
                       << " operations\n";
        }
      });
      rootGroups.append(contiguousGroups.begin(), contiguousGroups.end());
    }

    // Build the SLP graph from root groups
    SLPGraph graph = buildSLPGraph(rootGroups);

    // Print the graph structure
    LLVM_DEBUG(graph.print());
  });
}

std::unique_ptr<Pass> mlir::vector::createSLPVectorizerPass() {
  return std::make_unique<SLPVectorizerPass>();
}
