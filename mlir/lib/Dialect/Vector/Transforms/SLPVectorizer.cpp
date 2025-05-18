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
#define GEN_PASS_DEF_GREEDYSLPVECTORIZER
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

static bool isReadOp(Operation *op) {
  auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectInterface)
    return true;

  return effectInterface.hasEffect<MemoryEffects::Read>();
}

static bool isWriteOp(Operation *op) {
  auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectInterface)
    return true;

  return effectInterface.hasEffect<MemoryEffects::Write>();
}

/// Collect all memory operations in the block into groups.
/// Each group contains either all loads or all stores, uninterrupted by
/// operations of the other type.
static SmallVector<MemoryOpGroup> collectMemoryOpGroups(Block &block) {
  SmallVector<MemoryOpGroup> groups;
  MemoryOpGroup *currentGroup = nullptr;

  for (Operation &op : block) {
    if (currentGroup) {
      if (currentGroup->isLoadGroup() && isWriteOp(&op)) {
        currentGroup = nullptr;
      } else if (currentGroup->isStoreGroup() && isReadOp(&op)) {
        currentGroup = nullptr;
      }
    }

    if (!isa<memref::LoadOp, memref::StoreOp>(op))
      continue;

    bool isLoad = isReadOp(&op);
    MemoryOpGroup::Type type =
        isLoad ? MemoryOpGroup::Type::Load : MemoryOpGroup::Type::Store;

    if (!currentGroup) {
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

static Value getBase(Operation *op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getMemRef();
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getMemRef();
  return {};
}

static bool isContiguousLastDim(Value val) {
  auto memrefType = dyn_cast<MemRefType>(val.getType());
  if (!memrefType)
    return false;

  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(memrefType.getStridesAndOffset(strides, offset)))
    return false;

  return !strides.empty() && strides.back() == 1;
}

static ValueRange getIndices(Operation *op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getIndices();
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getIndices();
  return {};
}

static Type getElementType(Operation *op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getResult().getType();
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getValueToStore().getType();
  return {};
}

static bool isAdjacentIndices(Value idx1, Value idx2) {
  if (auto c1 = getConstantIntValue(idx1)) {
    if (auto c2 = getConstantIntValue(idx2))
      return *c1 + 1 == *c2;
  }
  return false;
}

static bool isAdjacentIndices(ValueRange idx1, ValueRange idx2) {
  if (idx1.empty() || idx1.size() != idx2.size())
    return false;

  if (idx1.drop_back() != idx2.drop_back())
    return false;

  return isAdjacentIndices(idx1.back(), idx2.back());
}

static bool isAdjacentIndices(Operation *op1, Operation *op2) {
  Value base1 = getBase(op1);
  Value base2 = getBase(op2);
  if (base1 != base2)
    return false;

  if (!isContiguousLastDim(base1))
    return false;

  return getElementType(op1) == getElementType(op2) &&
         isAdjacentIndices(getIndices(op1), getIndices(op2));
}

// Extract contiguous groups from a MemoryOpGroup
static SmallVector<MemoryOpGroup>
extractContiguousGroups(const MemoryOpGroup &group) {
  SmallVector<MemoryOpGroup> result;
  if (group.ops.empty())
    return result;

  llvm::SmallDenseSet<Operation *> processedOps;

  for (Operation *op : group.ops) {
    if (processedOps.contains(op))
      continue;

    // Start a new group with this operation
    result.emplace_back(group.type);
    MemoryOpGroup &currentGroup = result.back();
    auto &currentOps = currentGroup.ops;
    currentOps.push_back(op);
    processedOps.insert(op);

    bool foundMore;
    do {
      foundMore = false;
      for (Operation *otherOp : group.ops) {
        if (processedOps.contains(otherOp))
          continue;

        Operation *firstOp = currentOps.front();
        Operation *lastOp = currentOps.back();
        if (isAdjacentIndices(otherOp, firstOp)) {
          currentOps.insert(currentOps.begin(), otherOp);
          processedOps.insert(otherOp);
          foundMore = true;
        } else if (isAdjacentIndices(lastOp, otherOp)) {
          currentOps.push_back(otherOp);
          processedOps.insert(otherOp);
          foundMore = true;
        }
      }
    } while (foundMore);

    if (currentOps.size() <= 1) {
      result.pop_back();
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Extracted contiguous group with "
                            << currentGroup.ops.size() << " operations\n");
  }
  return result;
}

static bool isVectorizable(Operation *op) {
  return OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1;
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

    auto isGoodNode = [&](SLPGraphNode *node) {
      return node->users.empty() && node->operands.empty();
    };

    IRMapping mapping;
    for (auto *node : sortedNodes) {
      if (isGoodNode(node))
        continue;

      int64_t numElements = node->ops.size();
      Operation *op = node->ops.front();
      rewriter.setInsertionPoint(op);
      Location loc = op->getLoc();

      auto handleNonVectorInputs = [&](ValueRange operands) {
        for (auto [i, operand] : llvm::enumerate(operands)) {
          if (getNodeForOp(operand.getDefiningOp()))
            continue;

          SmallVector<Value> args;
          for (Operation *defOp : node->ops)
            args.push_back(defOp->getOperand(i));

          auto vecType = VectorType::get(numElements, operand.getType());
          Value vector =
              rewriter.create<vector::FromElementsOp>(loc, vecType, args);
          mapping.map(operand, vector);
        }
      };

      auto handleNonVectorOutputs = [&](Value newResult) {
        for (auto [i, result] : llvm::enumerate(node->ops)) {
          for (OpOperand &use : result->getUses()) {
            Operation *useOwner = use.getOwner();
            if (getNodeForOp(useOwner))
              continue;

            Value elem = rewriter.create<vector::ExtractOp>(loc, newResult, i);
            use.set(elem);
          }
        }
      };

      auto handleVecSizeMismatch = [&](Value arg) -> Value {
        auto srcType = cast<VectorType>(arg.getType());
        assert(srcType.getRank() == 1);
        if (srcType.getDimSize(0) == numElements)
          return arg;

        return rewriter.create<vector::ExtractStridedSliceOp>(loc, arg, 0,
                                                              numElements, 1);
      };

      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        auto vecType =
            VectorType::get(numElements, load.getMemRefType().getElementType());
        Value result = rewriter.create<vector::LoadOp>(
            loc, vecType, load.getMemRef(), load.getIndices());
        mapping.map(load.getResult(), result);
        handleNonVectorOutputs(result);
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        handleNonVectorInputs(store.getValueToStore());
        Value val = mapping.lookupOrDefault(store.getValueToStore());
        val = handleVecSizeMismatch(val);
        rewriter.create<vector::StoreOp>(loc, val, store.getMemRef(),
                                         store.getIndices());
      } else if (isVectorizable(op)) {
        handleNonVectorInputs(op->getOperands());
        Operation *newOp = rewriter.clone(*op, mapping);
        auto resVectorType =
            VectorType::get(numElements, op->getResultTypes().front());

        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(newOp);
          for (OpOperand &operand : newOp->getOpOperands()) {
            Value newOperand = handleVecSizeMismatch(operand.get());
            operand.set(newOperand);
          }
        }
        newOp->getResult(0).setType(resVectorType);

        mapping.map(op->getResults(), newOp->getResults());
        handleNonVectorOutputs(newOp->getResult(0));
      } else {
        op->emitError("unsupported operation");
        return failure();
      }
    }

    for (auto *node : llvm::reverse(sortedNodes)) {
      if (isGoodNode(node))
        continue;

      for (Operation *op : node->ops) {
        rewriter.eraseOp(op);
      }
    }

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

struct GreedySLPVectorizerPass
    : public mlir::vector::impl::GreedySLPVectorizerBase<
          GreedySLPVectorizerPass> {
  void runOnOperation() override;
};

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
    for (Operation *op : currentOps)
      fingerprints.invalidate(op);

    worklist.push_back(newNode);
  };

  auto processOperands = [&](SLPGraphNode *node, Value operand, int64_t index) {
    Operation *srcOp = operand.getDefiningOp();
    if (!srcOp)
      return;

    auto *existingNode = graph.getNodeForOp(srcOp);
    if (existingNode) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Adding edge from " << srcOp->getName() << " to "
                 << node->ops.front()->getName() << "\n");
      graph.addEdge(existingNode, node);
      return;
    }

    if (!isVectorizable(srcOp))
      return;

    SmallVector<Operation *> currentOps;
    currentOps.emplace_back(srcOp);
    for (Operation *op : ArrayRef(node->ops).drop_front()) {
      Operation *otherOp = op->getOperand(index).getDefiningOp();
      if (!otherOp || !isEquivalent(otherOp, srcOp))
        break;

      currentOps.push_back(otherOp);
    }

    if (currentOps.size() == 1)
      return;

    auto *newNode = graph.addNode(currentOps);
    graph.addEdge(newNode, node);
    for (Operation *op : currentOps)
      fingerprints.invalidate(op);

    worklist.push_back(newNode);
  };

  while (!worklist.empty()) {
    SLPGraphNode *node = worklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs() << "Processing node with " << node->ops.size()
                            << " operations, first op: "
                            << node->ops.front()->getName() << "\n");

    Operation *op = node->ops.front();
    for (OpOperand &use : op->getUses())
      processUse(node, use);

    for (auto [i, operand] : llvm::enumerate(op->getOperands()))
      processOperands(node, operand, i);
  }

  return graph;
}

void GreedySLPVectorizerPass::runOnOperation() {
  Operation *op = getOperation();

  // Walk all blocks recursively
  op->walk([&](Block *block) -> WalkResult {
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
    LLVM_DEBUG(graph.print());

    // Vectorize the graph
    IRRewriter rewriter(&getContext());
    if (failed(graph.vectorize(rewriter))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to vectorize graph\n");
      signalPassFailure();
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
}

} // namespace
