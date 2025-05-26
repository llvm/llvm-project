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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
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
};

static bool maybeReadOp(Operation *op) {
  auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectInterface)
    return true;

  return effectInterface.hasEffect<MemoryEffects::Read>();
}

static bool maybeWriteOp(Operation *op) {
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
    // Check if current group is interrupted by a read or write op.
    if (currentGroup) {
      if (currentGroup->isLoadGroup() && maybeWriteOp(&op)) {
        currentGroup = nullptr;
      } else if (currentGroup->isStoreGroup() && maybeReadOp(&op)) {
        currentGroup = nullptr;
      }
    }

    if (!isa<memref::LoadOp, memref::StoreOp>(op))
      continue;

    bool isLoad = maybeReadOp(&op);
    MemoryOpGroup::Type type =
        isLoad ? MemoryOpGroup::Type::Load : MemoryOpGroup::Type::Store;

    if (!currentGroup) {
      groups.emplace_back(type);
      currentGroup = &groups.back();
    }

    currentGroup->ops.push_back(&op);
  }

  return groups;
}

static Value getBase(Operation *op) {
  assert(op && "null op");
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
  assert(op && "null op");
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getIndices();
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getIndices();
  return {};
}

static Type getElementType(Operation *op) {
  assert(op && "null op");
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getResult().getType();
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getValueToStore().getType();
  return {};
}

/// Check if two indices are consecutive, i.e index1 + 1 == index2.
static bool isAdjacentIndices(Value idx1, Value idx2) {
  if (auto c1 = getConstantIntValue(idx1)) {
    if (auto c2 = getConstantIntValue(idx2))
      return *c1 + 1 == *c2;
  }

  if (auto addOp2 = idx2.getDefiningOp<arith::AddIOp>()) {
    if (addOp2.getLhs() == idx1 && getConstantIntValue(addOp2.getRhs()) == 1)
      return true;

    if (auto addOp1 = idx1.getDefiningOp<arith::AddIOp>()) {
      if (addOp1.getLhs() == addOp2.getLhs() &&
          isAdjacentIndices(addOp1.getRhs(), addOp2.getRhs()))
        return true;
    }
  }

  // TODO: Handle affine.apply, etc
  return false;
}

/// Check if two ranges of indices are consecutive, i.e fastest index differs
/// by 1 and all other indices are the same.
static bool isAdjacentIndices(ValueRange idx1, ValueRange idx2) {
  if (idx1.empty() || idx1.size() != idx2.size())
    return false;

  if (idx1.drop_back() != idx2.drop_back())
    return false;

  return isAdjacentIndices(idx1.back(), idx2.back());
}

/// Check if two operations are adjacent and can be combined into a vector op.
/// This is done by checking if the base memrefs are the same, the last
/// dimension is contiguous, and the element types and indices are compatible
static bool isAdjacentOps(Operation *op1, Operation *op2) {
  assert(op1 && "null op1");
  assert(op2 && "null op2");

  Value base1 = getBase(op1);
  Value base2 = getBase(op2);
  if (base1 != base2)
    return false;

  if (!isContiguousLastDim(base1))
    return false;

  if (getElementType(op1) != getElementType(op2))
    return false;

  return isAdjacentIndices(getIndices(op1), getIndices(op2));
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

    // Keep adding ops to the beginning or end of the current group until no
    // more ops can be added.
    bool foundMore;
    do {
      foundMore = false;
      for (Operation *otherOp : group.ops) {
        if (processedOps.contains(otherOp))
          continue;

        Operation *firstOp = currentOps.front();
        Operation *lastOp = currentOps.back();
        if (isAdjacentOps(otherOp, firstOp)) {
          currentOps.insert(currentOps.begin(), otherOp);
          processedOps.insert(otherOp);
          foundMore = true;
        } else if (isAdjacentOps(lastOp, otherOp)) {
          currentOps.push_back(otherOp);
          processedOps.insert(otherOp);
          foundMore = true;
        }
      }
    } while (foundMore);

    if (currentOps.size() <= 1) {
      // Do not vectorize if there is only one op.
      result.pop_back();
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Extracted contiguous group with "
                            << currentGroup.size() << " operations\n");
  }
  return result;
}

static bool isVectorizable(Operation *op) {
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return false;

  if (op->getNumResults() != 1)
    return false;

  for (auto type :
       llvm::concat<Type>(op->getResultTypes(), op->getOperandTypes())) {
    if (!type.isIntOrIndexOrFloat())
      return false;
  }

  return true;
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

  size_t size() const { return ops.size(); }

  Operation *op() const {
    assert(!ops.empty() && "empty ops");
    return ops.front();
  }

  Operation *getInsertionPoint() const {
    // Find the toplogically first node, which is not nessesary the first in the
    // `ops` as `ops` are sorted by their position in vector.
    assert(!ops.empty() && "empty node");
    Operation *ret = op();
    for (Operation *op : ArrayRef(ops).drop_front()) {
      if (op->isBeforeInBlock(ret))
        ret = op;
    }
    return ret;
  }
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

  /// Vectorize the operations in the graph.
  /// Returns number of nodes vectorized or failure if failed.
  FailureOr<size_t> vectorize(IRRewriter &rewriter) {
    if (nodes.empty())
      return 0;

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
        llvm::dbgs() << "  Node with " << node->size()
                     << " operations: " << node->op()->getName() << "\n";
      }
    });

    auto isBadNode = [&](SLPGraphNode *node) {
      // Do not vectorize stray nodes which are not connected to any other
      // nodes.
      return node->users.empty() && node->operands.empty();
    };

    // Update node vec sizes if its inputs vec sizes are smaller.
    // This is nedeed to handle situations when we have 3->3->4 sizes in tree.
    // TODO: It maybe possible to reconstruct the larger vec size combining src
    // smaller vector and scalar arg.
    for (auto *node : sortedNodes) {
      size_t size = node->size();
      for (auto *operand : node->operands)
        size = std::min(size, operand->size());

      node->ops.resize(size);
    }

    llvm::erase_if(sortedNodes, isBadNode);

    IRMapping mapping;
    for (auto *node : sortedNodes) {
      LLVM_DEBUG({
        llvm::dbgs() << "Processing node with " << node->size()
                     << " operations\n";
        llvm::dbgs() << "  First op: " << *node->op() << "\n";
      });

      // `op` is the node with the smallest index in vector and not the
      // nessesarily the good insertion point.
      Operation *op = node->op();
      Operation *ip = node->getInsertionPoint();
      if (!ip)
        return op->emitError("no insertion point found for node");

      rewriter.setInsertionPoint(ip);
      int64_t numElements = node->size();
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
      } else if (auto extract = dyn_cast<vector::ExtractOp>(op)) {
        Value val = handleVecSizeMismatch(extract.getVector());
        mapping.map(extract.getResult(), val);
      } else {
        op->emitError("unsupported operation");
        return failure();
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Erasing original ops\n");

    // As all nodes were cloned, we need to erase the original ops in reverse
    // topo order to avoid invalidation users.
    for (auto *node : llvm::reverse(sortedNodes)) {
      for (Operation *op : node->ops) {
        rewriter.eraseOp(op);
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Vectorization completed successfully\n");
    return sortedNodes.size();
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
                   << (isa<memref::LoadOp>(node->op()) ? "LOAD" : "STORE")
                   << " group with " << node->size() << " operations:\n";
      for (auto *op : node->ops) {
        llvm::dbgs() << "    " << *op << "\n";
      }
      llvm::dbgs() << "    Users: ";
      for (auto *user : node->users) {
        llvm::dbgs() << "\n      Group with " << user->size() << " operations:";
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
      llvm::dbgs() << "  Group with " << node->size() << " operations:\n";
      for (auto *op : node->ops) {
        llvm::dbgs() << "    " << *op << "\n";
      }
      llvm::dbgs() << "    Operands: ";
      for (auto *operand : node->operands) {
        llvm::dbgs() << "\n      Group with " << operand->size()
                     << " operations:";
        for (auto *op : operand->ops) {
          llvm::dbgs() << "\n        " << *op;
        }
      }
      llvm::dbgs() << "\n    Users: ";
      for (auto *user : node->users) {
        llvm::dbgs() << "\n      Group with " << user->size() << " operations:";
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

/// This pass implements the greedy SLP vectorizer. It detects consecutive
/// operations that can be put together into vector operations. The pass works
/// bi-directionaly, starting from reads or stores, in search of scalars to
/// combine.
///
/// Pass is split into multiple steps:
/// 1. Collect memory operation groups within same block.
/// Group is either multiple loads uninterrupted by stores or multiple stores
/// uninterrupted by loads.
///
/// 2. Extract contiguous groups from memory operation groups, based on the
/// ops base memrefs, load/store element types, and indices.
///
/// 3. Build SLP graph from contiguous groups. This is done by going both
/// top-down and bottom-up through uses/operands respectively, starting from
/// contiguous memory operation groups.
///
/// 4. Vectorize SLP graph. This is done by topological sort of the graph and
/// vectorizing each node in the order of the sort.
///
/// Vectorization is done by cloning the operations and mapping the operands and
/// results.
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

/// SLP vectorizer is bi-directional, so when we go top-down we can can have
/// multiple users with the same immediate op type, this class tries to compute
/// fingerprint for such ops based on the entire ops graph to maximize further
/// scalar ops merging.
///
/// Example:
/// ```
///  %0 = memref.load %arg0[%c0] : memref<8xi32>
///  %1 = memref.load %arg0[%c1] : memref<8xi32>
///  %2 = memref.load %arg0[%c2] : memref<8xi32>
///  %3 = memref.load %arg0[%c3] : memref<8xi32>
///
///  %4 = memref.load %arg1[%c0] : memref<8xi32>
///  %5 = memref.load %arg1[%c1] : memref<8xi32>
///  %6 = memref.load %arg1[%c2] : memref<8xi32>
///  %7 = memref.load %arg1[%c3] : memref<8xi32>
///
///  %8 = arith.addi %0, %4 : i32
///  %12 = arith.addi %0, %arg2 : i32
///
///  %13 = arith.addi %1, %arg3 : i32
///  %9 = arith.addi %1, %5 : i32
///
///  %10 = arith.addi %2, %6 : i32
///  %14 = arith.addi %2, %arg4 : i32
///
///  %15 = arith.addi %3, %arg5 : i32
///  %11 = arith.addi %3, %7 : i32
/// ```
/// Here each load have multiple uses, in different order, and we want to merge
/// them in a way that maximizes the number of merged ops.
///
/// To achieve this, we compute fingerprint for each op including the other
/// operands, which will include the other loads in this example.
struct OperationsFingerprint {
  OperationsFingerprint(const SLPGraph &graph) : graph(graph) {}

  Fingerprint getFingerprint(Operation *op) {
    assert(op && "null op");
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

/// Check if two ops are equivalent for the purposes of SLP vectorization, i.e.
/// they can be merged into single vector op.
static bool isEquivalent(Operation *op1, Operation *op2) {
  assert(op1 && "null op1");
  assert(op2 && "null op2");
  if (op1 == op2)
    return true;

  if (op1->getName() != op2->getName())
    return false;

  if (op1->getAttrs() != op2->getAttrs())
    return false;

  if (op1->getBlock() != op2->getBlock())
    return false;

  return true;
}

/// Get static position of the extract op, if it is 1D and static.
static std::optional<int64_t> getExtractIndex(vector::ExtractOp extractOp) {
  if (extractOp.getNumIndices() != 1 || extractOp.hasDynamicPosition())
    return std::nullopt;

  return extractOp.getStaticPosition().front();
}

/// Build the SLP graph starting from memory operation groups and going both
/// top-down and bottom-up through uses/operands respectively.
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
      llvm::dbgs() << "Created root group node with " << node->size()
                   << " operations of type "
                   << (group.isLoadGroup() ? "Load" : "Store") << "\n";
    });
  }

  OperationsFingerprint fingerprints(graph);

  // Process node uses, going top-down.
  auto processUse = [&](SLPGraphNode *node, OpOperand &use) {
    Operation *user = use.getOwner();
    auto *existingNode = graph.getNodeForOp(user);
    if (existingNode) {
      LLVM_DEBUG(llvm::dbgs() << "  Adding edge from " << node->op()->getName()
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

  // Process node operands, going bottom-up.
  auto processOperands = [&](SLPGraphNode *node, Value operand, int64_t index) {
    Operation *srcOp = operand.getDefiningOp();
    if (!srcOp)
      return;

    auto *existingNode = graph.getNodeForOp(srcOp);
    if (existingNode) {
      LLVM_DEBUG(llvm::dbgs() << "  Adding edge from " << srcOp->getName()
                              << " to " << node->op()->getName() << "\n");
      graph.addEdge(existingNode, node);
      return;
    }

    SmallVector<Operation *> currentOps;
    if (auto extractOp = dyn_cast<vector::ExtractOp>(srcOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Processing vector.extract op with index "
                 << getExtractIndex(extractOp).value_or(-1) << "\n");
      currentOps.push_back(extractOp);

      std::optional<int64_t> extractIndex = getExtractIndex(extractOp);
      if (!extractIndex)
        return;

      Value vector = extractOp.getVector();
      int64_t currentIndex = *extractIndex;
      for (Operation *op : ArrayRef(node->ops).drop_front()) {
        auto otherOp = op->getOperand(index).getDefiningOp<vector::ExtractOp>();
        if (!otherOp || otherOp.getVector() != vector)
          break;

        std::optional<int64_t> otherExtractIndex = getExtractIndex(otherOp);
        if (!otherExtractIndex || *otherExtractIndex != (currentIndex + 1))
          break;

        currentOps.push_back(otherOp);
        ++currentIndex;
      }
    } else if (isVectorizable(srcOp)) {
      LLVM_DEBUG(llvm::dbgs() << "  Processing vectorizable op "
                              << srcOp->getName() << "\n");

      currentOps.emplace_back(srcOp);
      for (Operation *op : ArrayRef(node->ops).drop_front()) {
        Operation *otherOp = op->getOperand(index).getDefiningOp();
        if (!otherOp || !isEquivalent(otherOp, srcOp))
          break;

        currentOps.push_back(otherOp);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Unsupported op " << srcOp->getName() << "\n");
      return;
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
    LLVM_DEBUG(llvm::dbgs()
               << "Processing node with " << node->size()
               << " operations, first op: " << node->op()->getName() << "\n");

    Operation *op = node->op();
    for (OpOperand &use : op->getUses())
      processUse(node, use);

    for (auto [i, operand] : llvm::enumerate(op->getOperands()))
      processOperands(node, operand, i);
  }

  return graph;
}

/// Try to vectorize ops in a block.
/// Returns number of nodes vectorized or error flag if failed.
static FailureOr<size_t> tryToVectorizeInBlock(Block &block) {
  LLVM_DEBUG(llvm::dbgs() << "Processing block in operation: "
                          << block.getParentOp()->getName() << "\n");

  // Collect memory operation groups
  SmallVector<MemoryOpGroup> groups = collectMemoryOpGroups(block);

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
  IRRewriter rewriter(block.getParentOp()->getContext());
  FailureOr<size_t> numNodesVectorized = graph.vectorize(rewriter);
  if (failed(numNodesVectorized))
    LLVM_DEBUG(llvm::dbgs() << "Failed to vectorize graph\n");

  return numNodesVectorized;
}

void GreedySLPVectorizerPass::runOnOperation() {
  Operation *op = getOperation();

  // Run until fixed point is reached.
  bool changed;
  do {
    changed = false;
    auto visitor = [&](Block *block) -> WalkResult {
      FailureOr<size_t> numNodesVectorized = tryToVectorizeInBlock(*block);
      if (failed(numNodesVectorized))
        return WalkResult::interrupt();

      changed = changed || *numNodesVectorized > 0;
      return WalkResult::advance();
    };
    // Walk all blocks recursively
    if (op->walk(visitor).wasInterrupted())
      return signalPassFailure();

    // Run empty `applyPatternsGreedily` for simple DCE and folding.
    if (changed) {
      auto config = GreedyRewriteConfig().enableFolding().enableConstantCSE();
      (void)applyPatternsGreedily(op, {}, config);
    }
  } while (changed);
}

} // namespace
