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
#include "llvm/Support/SHA1.h"

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
static SmallVector<MemoryOpGroup>
extractContiguousGroups(const MemoryOpGroup &group) {
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

/// A node in the SLP graph representing a group of vectorizable operations
struct SLPGraphNode {
  SmallVector<Operation *> ops;
  SmallVector<SLPGraphNode *> users;
  SmallVector<SLPGraphNode *> operands;
  bool isRoot = false;

  SLPGraphNode() = default;
  SLPGraphNode(ArrayRef<Operation *> operations)
      : ops(operations.begin(), operations.end()) {}
};

/// A graph of vectorizable operations
class SLPGraph {
public:
  SLPGraph() = default;
  ~SLPGraph() = default;

  SLPGraph(const SLPGraph &) = delete;
  SLPGraph &operator=(const SLPGraph &) = delete;

  SLPGraph(SLPGraph &&) = default;
  SLPGraph &operator=(SLPGraph &&) = default;

  /// Add a new node to the graph
  SLPGraphNode *addNode(ArrayRef<Operation *> operations) {
    nodes.push_back(std::make_unique<SLPGraphNode>(operations));
    auto *node = nodes.back().get();
    for (Operation *op : operations)
      opToNode[op] = node;
    return node;
  }

  /// Add a root node (memory operation)
  SLPGraphNode *addRoot(ArrayRef<Operation *> operations) {
    auto *node = addNode(operations);
    node->isRoot = true;
    return node;
  }

  /// Add a dependency edge between nodes
  void addEdge(SLPGraphNode *from, SLPGraphNode *to) {
    from->users.push_back(to);
    to->operands.push_back(from);
  }

  /// Get all root nodes
  SmallVector<SLPGraphNode *> getRoots() const {
    SmallVector<SLPGraphNode *> roots;
    for (const auto &node : nodes)
      if (node->isRoot)
        roots.push_back(node.get());
    return roots;
  }

  /// Get the node associated with an operation
  SLPGraphNode *getNodeForOp(Operation *op) const {
    auto it = opToNode.find(op);
    return it != opToNode.end() ? it->second : nullptr;
  }

  /// Topologically sort the nodes in the graph
  SmallVector<SLPGraphNode *> topologicalSort() const {
    SmallVector<SLPGraphNode *> result;
    llvm::SmallDenseSet<SLPGraphNode *> visited;

    SmallVector<SLPGraphNode *> stack;

    // Process each node
    for (const auto &node : nodes) {
      if (visited.contains(node.get()))
        continue;

      stack.emplace_back(node.get());
      while (!stack.empty()) {
        SLPGraphNode *node = stack.pop_back_val();
        if (visited.contains(node))
          continue;

        stack.push_back(node);

        bool pushed = false;
        for (SLPGraphNode *operand : node->operands) {
          if (visited.contains(operand))
            continue;

          stack.push_back(operand);
          pushed = true;
        }

        if (!pushed) {
          visited.insert(node);
          result.push_back(node);
        }
      }
    }

    return result;
  }

  /// Vectorize the operations in the graph
  LogicalResult vectorize(IRRewriter &rewriter) {
    if (nodes.empty())
      return success();

    LLVM_DEBUG(llvm::dbgs()
               << "Vectorizing SLP graph with " << nodes.size() << " nodes\n");

    // Get topologically sorted nodes
    SmallVector<SLPGraphNode *> sortedNodes = topologicalSort();
    if (sortedNodes.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to topologically sort nodes\n");
      return failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Topologically sorted nodes:\n";
      for (auto *node : sortedNodes) {
        llvm::dbgs() << "  Node with " << node->ops.size()
                     << " operations: " << node->ops.front()->getName() << "\n";
      }
    });

    // TODO: Implement vectorization logic:
    // 1. Process nodes in topological order
    // 2. For each node:
    //    a. Check if all operands are vectorized
    //    b. Create vector operation
    //    c. Replace scalar operations with vector operation
    // 3. Handle memory operations (loads/stores) specially
    // 4. Update use-def chains

    return success();
  }

  /// Print the graph structure
  [[maybe_unused]] void print() const {
    llvm::dbgs() << "SLP Graph Structure:\n";
    llvm::dbgs() << "===================\n";

    // First print all roots
    llvm::dbgs() << "Roots:\n";
    for (const auto &node : nodes) {
      if (!node->isRoot)
        continue;
      llvm::dbgs() << "  "
                   << (isa<memref::LoadOp>(node->ops[0]) ? "LOAD" : "STORE")
                   << " group with " << node->ops.size() << " operations:\n";
      for (auto *op : node->ops) {
        llvm::dbgs() << "    " << *op << "\n";
      }
      llvm::dbgs() << "    Users: ";
      for (auto *user : node->users) {
        llvm::dbgs() << "\n      Group with " << user->ops.size()
                     << " operations:";
        for (auto *op : user->ops) {
          llvm::dbgs() << "\n        " << *op;
        }
      }
      llvm::dbgs() << "\n";
    }

    // Then print all non-root nodes
    llvm::dbgs() << "\nNon-root nodes:\n";
    for (const auto &node : nodes) {
      if (node->isRoot)
        continue;
      llvm::dbgs() << "  Group with " << node->ops.size() << " operations:\n";
      for (auto *op : node->ops) {
        llvm::dbgs() << "    " << *op << "\n";
      }
      llvm::dbgs() << "    Operands: ";
      for (auto *operand : node->operands) {
        llvm::dbgs() << "\n      Group with " << operand->ops.size()
                     << " operations:";
        for (auto *op : operand->ops) {
          llvm::dbgs() << "\n        " << *op;
        }
      }
      llvm::dbgs() << "\n    Users: ";
      for (auto *user : node->users) {
        llvm::dbgs() << "\n      Group with " << user->ops.size()
                     << " operations:";
        for (auto *op : user->ops) {
          llvm::dbgs() << "\n        " << *op;
        }
      }
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "===================\n";
  }

private:
  SmallVector<std::unique_ptr<SLPGraphNode>> nodes;
  llvm::SmallDenseMap<Operation *, SLPGraphNode *> opToNode;
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

static bool isVectorizable(Operation *op) {
  return OpTrait::hasElementwiseMappableTraits(op);
}

using Fingerprint = std::array<uint8_t, 20>;

template <typename T>
static void addDataToHash(llvm::SHA1 &hasher, const T &data) {
  hasher.update(
      ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&data), sizeof(T)));
}

struct OperationsFingerprint {
  OperationsFingerprint(const SLPGraph &graph) : graph(graph) {}

  Fingerprint getFingerprint(Operation *op) {
    auto it = fingerprints.find(op);
    if (it != fingerprints.end())
      return it->second;

    SmallVector<Operation *> worklist;
    SmallVector<Operation *> toposortedOps;
    worklist.emplace_back(op);
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      toposortedOps.emplace_back(op);
      if (graph.getNodeForOp(op))
        continue;

      for (Value operand : op->getOperands()) {
        auto *defOp = operand.getDefiningOp();
        if (!defOp || !isVectorizable(defOp))
          continue;

        toposortedOps.emplace_back(defOp);
        worklist.emplace_back(defOp);
      }
    }

    for (Operation *op : llvm::reverse(toposortedOps)) {
      llvm::SHA1 hasher;
      addDataToHash(hasher, op->getName().getTypeID());
      addDataToHash(hasher, op->getRawDictionaryAttrs());
      addDataToHash(hasher, op->hashProperties());
      for (Value operand : op->getOperands()) {
        auto *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;

        auto *node = graph.getNodeForOp(defOp);
        if (node) {
          addDataToHash(hasher, node);
          continue;
        }

        auto it2 = fingerprints.find(defOp);
        if (it2 != fingerprints.end()) {
          addDataToHash(hasher, it2->second);
          continue;
        }
      }
      fingerprints[op] = hasher.result();
    }

    return fingerprints[op];
  }

  void invalidate(Operation *op) {
    if (fingerprints.contains(op))
      fingerprints.clear();
  }

  const SLPGraph &graph;
  DenseMap<Operation *, Fingerprint> fingerprints;
};

static bool isEquivalent(Operation *op1, Operation *op2) {
  if (op1->getName() != op2->getName())
    return false;

  if (op1->getRawDictionaryAttrs() != op2->getRawDictionaryAttrs())
    return false;

  return true;
}

/// Build the SLP graph starting from memory operation groups
static SLPGraph buildSLPGraph(ArrayRef<MemoryOpGroup> rootGroups) {
  if (rootGroups.empty())
    return SLPGraph();

  LLVM_DEBUG(llvm::dbgs() << "=== Building SLP graph from " << rootGroups.size()
                          << " root groups ===\n");
  SLPGraph graph;

  SmallVector<SLPGraphNode *> worklist;

  // First, create nodes for each contiguous memory operation group
  for (const auto &group : rootGroups) {
    auto *node = graph.addRoot(group.ops);
    worklist.push_back(node);

    LLVM_DEBUG({
      llvm::dbgs() << "Created root group node with " << node->ops.size()
                   << " operations of type "
                   << (group.isLoadGroup() ? "Load" : "Store") << "\n";
    });
  }

  OperationsFingerprint fingerprints(graph);

  auto processUse = [&](SLPGraphNode *node, OpOperand &use) {
    Operation *user = use.getOwner();
    auto *existingNode = graph.getNodeForOp(user);
    if (existingNode) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Adding edge from " << node->ops.front()->getName()
                 << " to " << user->getName() << "\n");
      graph.addEdge(node, existingNode);
      return;
    }

    if (!isVectorizable(user))
      return;

    Fingerprint expectedFingerprint = fingerprints.getFingerprint(user);

    SmallVector<Operation *> currentOps;
    currentOps.emplace_back(user);
    for (Operation *op : ArrayRef(node->ops).drop_front()) {
      Operation *found = nullptr;
      for (OpOperand &opUse : op->getUses()) {
        if (opUse.getOperandNumber() != use.getOperandNumber())
          continue;

        Operation *useOwner = opUse.getOwner();
        if (!isEquivalent(useOwner, user) ||
            fingerprints.getFingerprint(useOwner) != expectedFingerprint)
          continue;

        found = useOwner;
        break;
      }
      if (!found)
        break;

      currentOps.push_back(found);
    }

    if (currentOps.size() == 1)
      return;

    auto *newNode = graph.addNode(currentOps);
    graph.addEdge(node, newNode);
    for (Operation *op : currentOps) {
      fingerprints.invalidate(op);
    }

    worklist.push_back(newNode);
  };

  while (!worklist.empty()) {
    SLPGraphNode *node = worklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs() << "Processing node with " << node->ops.size()
                            << " operations, first op: "
                            << node->ops.front()->getName() << "\n");

    Operation *op = node->ops.front();
    for (OpOperand &use : op->getUses()) {
      processUse(node, use);
      LLVM_DEBUG(llvm::dbgs() << "  Processing use in operation: "
                              << use.getOwner()->getName() << "\n");
    }
  }

  return graph;
}

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

    // Vectorize the graph
    IRRewriter rewriter(&getContext());
    if (failed(graph.vectorize(rewriter))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to vectorize graph\n");
      return signalPassFailure();
    }
  });
}

} // namespace

std::unique_ptr<Pass> mlir::vector::createSLPVectorizerPass() {
  return std::make_unique<SLPVectorizerPass>();
}
