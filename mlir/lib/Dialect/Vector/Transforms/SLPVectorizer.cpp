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

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
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
  int64_t elementsCount = 0;

  MemoryOpGroup(Type t) : type(t) {}

  bool isLoadGroup() const { return type == Type::Load; }
  bool isStoreGroup() const { return type == Type::Store; }

  size_t opsCount() const { return ops.size(); }
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

static std::optional<std::pair<Type, int64_t>>
getVectorElementTypeAndCount(VectorType vectorType) {
  if (vectorType.getRank() > 1 || vectorType.isScalable())
    return std::nullopt;

  return std::make_pair(vectorType.getElementType(),
                        vectorType.getNumElements());
}

static std::optional<std::pair<Type, int64_t>>
getElementTypeAndCount(Operation *op) {
  assert(op && "null op");
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return std::make_pair(loadOp.getResult().getType(), 1);
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return std::make_pair(storeOp.getValueToStore().getType(), 1);
  if (auto loadOp = dyn_cast<vector::LoadOp>(op))
    return getVectorElementTypeAndCount(loadOp.getVectorType());
  if (auto storeOp = dyn_cast<vector::StoreOp>(op))
    return getVectorElementTypeAndCount(storeOp.getVectorType());

  return std::nullopt;
}

static bool isSupportedMemOp(Operation *op) {
  assert(op && "null op");
  auto typeAndCount = getElementTypeAndCount(op);
  if (!typeAndCount)
    return false;

  return isa_and_present<IntegerType, FloatType, IndexType>(
      typeAndCount->first);
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

    if (!isSupportedMemOp(&op))
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
  if (auto loadOp = dyn_cast<vector::LoadOp>(op))
    return loadOp.getBase();
  if (auto storeOp = dyn_cast<vector::StoreOp>(op))
    return storeOp.getBase();

  llvm_unreachable("unsupported op");
}

static Value getValueToStore(Operation *op) {
  assert(op && "null op");
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getValueToStore();
  if (auto storeOp = dyn_cast<vector::StoreOp>(op))
    return storeOp.getValueToStore();

  llvm_unreachable("unsupported op");
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
  if (auto loadOp = dyn_cast<vector::LoadOp>(op))
    return loadOp.getIndices();
  if (auto storeOp = dyn_cast<vector::StoreOp>(op))
    return storeOp.getIndices();

  llvm_unreachable("unsupported op");
}

static bool isAdjacentAffineMapIndices(Value idx1, Value idx2, int64_t offset) {
  auto applyOp1 = idx1.getDefiningOp<affine::AffineApplyOp>();
  if (!applyOp1)
    return false;

  auto applyOp2 = idx2.getDefiningOp<affine::AffineApplyOp>();
  if (!applyOp2)
    return false;

  if (applyOp1.getOperands() != applyOp2.getOperands())
    return false;

  AffineExpr expr1 = applyOp1.getAffineMap().getResult(0);
  AffineExpr expr2 = applyOp2.getAffineMap().getResult(0);
  auto diff =
      simplifyAffineExpr(expr2 - expr1, 0, applyOp1.getOperands().size());

  auto diffConst = dyn_cast<AffineConstantExpr>(diff);
  return diffConst && diffConst.getValue() == offset;
}

/// Check if two indices are consecutive, i.e index1 + offset == index2.
static bool isAdjacentIndices(Value idx1, Value idx2, int64_t offset) {
  if (auto c1 = getConstantIntValue(idx1)) {
    if (auto c2 = getConstantIntValue(idx2))
      return *c1 + offset == *c2;
  }

  if (auto addOp2 = idx2.getDefiningOp<arith::AddIOp>()) {
    if (addOp2.getLhs() == idx1 &&
        getConstantIntValue(addOp2.getRhs()) == offset)
      return true;

    if (auto addOp1 = idx1.getDefiningOp<arith::AddIOp>()) {
      if (addOp1.getLhs() == addOp2.getLhs() &&
          isAdjacentIndices(addOp1.getRhs(), addOp2.getRhs(), offset))
        return true;
    }
  }

  if (isAdjacentAffineMapIndices(idx1, idx2, offset))
    return true;

  return false;
}

/// Check if two ranges of indices are consecutive, i.e fastest index differs
/// by `offset` and all other indices are the same.
static bool isAdjacentIndices(ValueRange idx1, ValueRange idx2,
                              int64_t offset) {
  if (idx1.empty() || idx1.size() != idx2.size())
    return false;

  if (idx1.drop_back() != idx2.drop_back())
    return false;

  return isAdjacentIndices(idx1.back(), idx2.back(), offset);
}

/// Check if two operations are adjacent and can be combined into a vector op.
/// This is done by checking if the base memrefs are the same, the last
/// dimension is contiguous, and the element types and indices are compatible.
/// If source read/write is already vectorized, only merge ops if vector
/// elements count is the same.
static bool isAdjacentOps(Operation *op1, Operation *op2) {
  assert(op1 && "null op1");
  assert(op2 && "null op2");

  Value base1 = getBase(op1);
  Value base2 = getBase(op2);
  if (base1 != base2)
    return false;

  if (!isContiguousLastDim(base1))
    return false;

  auto typeAndCount1 = getElementTypeAndCount(op1);
  if (!typeAndCount1)
    return false;

  auto typeAndCount2 = getElementTypeAndCount(op2);
  if (!typeAndCount2)
    return false;

  if (typeAndCount1 != typeAndCount2)
    return false;

  // For now we are only merging ops with same elements count.
  return isAdjacentIndices(getIndices(op1), getIndices(op2),
                           typeAndCount1->second);
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
    currentGroup.elementsCount = getElementTypeAndCount(op)->second;
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
                            << currentGroup.opsCount() << " operations\n");
  }
  return result;
}

/// Check if an operation is vectorizable.
/// If `expectedElementsCount` is provided, check if original op had the
/// specified number of elements.
static bool
isVectorizable(Operation *op,
               std::optional<int64_t> expectedElementsCount = std::nullopt) {
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return false;

  if (op->getNumResults() != 1)
    return false;

  for (auto type :
       llvm::concat<Type>(op->getResultTypes(), op->getOperandTypes())) {
    int64_t vectorElementsCount = 1;
    if (auto vectorType = dyn_cast<VectorType>(type)) {
      if (vectorType.getRank() > 1 || vectorType.isScalable())
        return false;

      type = vectorType.getElementType();
      vectorElementsCount = vectorType.getNumElements();
    }

    if (expectedElementsCount && vectorElementsCount != *expectedElementsCount)
      return false;

    if (!isa<IntegerType, FloatType, IndexType>(type))
      return false;
  }

  return true;
}

/// Get the next operation in the block, assuming `op` is not a terminator/last
/// operation in the block.
static Operation *nextOp(Operation *op) {
  assert(op && "null op");
  auto it = op->getIterator();
  return &*std::next(it);
}

/// A node in the SLP graph representing a group of vectorizable operations
struct SLPGraphNode {
  SmallVector<Operation *> ops;
  SmallVector<SLPGraphNode *> users;
  SmallVector<SLPGraphNode *> operands;
  Operation *insertionPoint = nullptr;
  int64_t elementsCount = 0;
  bool isRoot = false;

  SLPGraphNode() = default;
  SLPGraphNode(ArrayRef<Operation *> operations)
      : ops(operations.begin(), operations.end()) {}

  size_t opsCount() const { return ops.size(); }
  size_t vectorSize() const { return elementsCount * opsCount(); }

  Operation *op() const {
    assert(!ops.empty() && "empty ops");
    return ops.front();
  }

  /// Get the suitable insertion point for the new vectorized op.
  /// This method is trying to take into account operands insertions points too
  /// to satisfy dominance relations.
  Operation *getInsertionPoint() {
    assert(!ops.empty() && "empty node");
    if (insertionPoint)
      return insertionPoint;

    // Find the toplogically first node, which is not nessesary the first in the
    // `ops` as `ops` are sorted by their position in vector.
    Operation *ret = op();
    for (Operation *op : ArrayRef(ops).drop_front()) {
      if (op->isBeforeInBlock(ret))
        ret = op;
    }

    for (Operation *op : ops) {
      for (Value opOperand : op->getOperands()) {
        Operation *defOp = opOperand.getDefiningOp();
        if (!defOp || defOp->getBlock() != ret->getBlock())
          continue;

        Operation *next = nextOp(defOp);
        if (ret->isBeforeInBlock(next))
          ret = next;
      }
    }

    // Try to adjust insertion point to satisfy dominance relations with
    // operands.
    for (SLPGraphNode *operand : operands) {
      Operation *ip = operand->getInsertionPoint();
      if (!ip)
        return nullptr;

      Operation *next = nextOp(ip);
      if (next->getBlock() == ret->getBlock() && ret->isBeforeInBlock(next))
        ret = next;
    }

    insertionPoint = ret;
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
  SLPGraphNode *addNode(ArrayRef<Operation *> operations,
                        int64_t elementsCount) {
    nodes.push_back(std::make_unique<SLPGraphNode>(operations));
    auto *node = nodes.back().get();
    node->elementsCount = elementsCount;
    for (Operation *op : operations)
      opToNode[op] = node;
    return node;
  }

  /// Add a root node (memory operation)
  SLPGraphNode *addRoot(ArrayRef<Operation *> operations,
                        int64_t elementsCount) {
    auto *node = addNode(operations, elementsCount);
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
  FailureOr<size_t>
  vectorize(IRRewriter &rewriter,
            llvm::function_ref<bool(Type, size_t)> isValidVecType);

  /// Print the graph structure
  [[maybe_unused]] void print() const {
    llvm::dbgs() << "SLP Graph Structure:\n";
    llvm::dbgs() << "===================\n";

    // First print all roots
    llvm::dbgs() << "Roots:\n";
    for (const auto &node : nodes) {
      if (!node->isRoot)
        continue;
      llvm::dbgs() << "  " << (maybeReadOp(node->op()) ? "LOAD" : "STORE")
                   << " group with " << node->opsCount() << " operations:\n";
      for (auto *op : node->ops) {
        llvm::dbgs() << "    " << *op << "\n";
      }
      llvm::dbgs() << "    Users: ";
      for (auto *user : node->users) {
        llvm::dbgs() << "\n      Group with " << user->opsCount()
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
      llvm::dbgs() << "  Group with " << node->opsCount() << " operations:\n";
      for (auto *op : node->ops) {
        llvm::dbgs() << "    " << *op << "\n";
      }
      llvm::dbgs() << "    Operands: ";
      for (auto *operand : node->operands) {
        llvm::dbgs() << "\n      Group with " << operand->opsCount()
                     << " operations:";
        for (auto *op : operand->ops) {
          llvm::dbgs() << "\n        " << *op;
        }
      }
      llvm::dbgs() << "\n    Users: ";
      for (auto *user : node->users) {
        llvm::dbgs() << "\n      Group with " << user->opsCount()
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
  using GreedySLPVectorizerBase::GreedySLPVectorizerBase;

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

/// Check if op input/output types can be vectorized.
static bool
checkOpVecType(SLPGraphNode *node,
               llvm::function_ref<bool(Type, size_t)> isValidVecType) {
  Operation *op = node->op();
  size_t size = node->vectorSize();
  auto checkRes = [](bool res) -> bool {
    LLVM_DEBUG(llvm::dbgs() << (res ? "true" : "false") << "\n");
    return res;
  };

  if (auto typeAndCount = getElementTypeAndCount(op)) {
    Type elementType = typeAndCount->first;
    LLVM_DEBUG(llvm::dbgs() << "Checking if type " << elementType
                            << " with size " << size << " can be vectorized: ");
    return checkRes(isValidVecType(elementType, size));
  }

  if (isVectorizable(op)) {
    for (auto type :
         llvm::concat<Type>(op->getResultTypes(), op->getOperandTypes())) {
      Type elementType = getElementTypeOrSelf(type);
      LLVM_DEBUG(llvm::dbgs()
                 << "Checking if type " << elementType << " with size " << size
                 << " can be vectorized: ");
      if (!checkRes(isValidVecType(elementType, size)))
        return false;
    }
    return true;
  }

  if (auto extract = dyn_cast<vector::ExtractOp>(op)) {
    Type type = extract.getResult().getType();
    LLVM_DEBUG(llvm::dbgs() << "Checking if type " << type << " with size "
                            << size << " can be vectorized: ");
    return checkRes(isValidVecType(type, size));
  }

  LLVM_DEBUG(llvm::dbgs() << "Unsupported op " << op->getName() << "\n");
  return false;
}

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
    auto *node = graph.addRoot(group.ops, group.elementsCount);
    worklist.push_back(node);

    LLVM_DEBUG({
      llvm::dbgs() << "Created root group node with " << node->opsCount()
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

    if (!isVectorizable(user, node->elementsCount))
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

    auto *newNode = graph.addNode(currentOps, node->elementsCount);
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
    } else if (isVectorizable(srcOp, node->elementsCount)) {
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

    auto *newNode = graph.addNode(currentOps, node->elementsCount);
    graph.addEdge(newNode, node);
    for (Operation *op : currentOps)
      fingerprints.invalidate(op);

    worklist.push_back(newNode);
  };

  while (!worklist.empty()) {
    SLPGraphNode *node = worklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs()
               << "Processing node with " << node->opsCount()
               << " operations, first op: " << node->op()->getName() << "\n");

    Operation *op = node->op();
    for (OpOperand &use : op->getUses())
      processUse(node, use);

    for (auto [i, operand] : llvm::enumerate(op->getOperands()))
      processOperands(node, operand, i);
  }

  return graph;
}

FailureOr<size_t>
SLPGraph::vectorize(IRRewriter &rewriter,
                    llvm::function_ref<bool(Type, size_t)> isValidVecType) {
  if (nodes.empty())
    return 0;

  LLVM_DEBUG(llvm::dbgs() << "Vectorizing SLP graph with " << nodes.size()
                          << " nodes\n");

  // Get topologically sorted nodes
  SmallVector<SLPGraphNode *> sortedNodes = topologicalSort();
  if (sortedNodes.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to topologically sort nodes\n");
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Topologically sorted nodes:\n";
    for (auto *node : sortedNodes) {
      llvm::dbgs() << "  Node with " << node->opsCount()
                   << " operations: " << node->op()->getName() << "\n";
    }
  });

  auto isBadNode = [&](SLPGraphNode *node) {
    // Do not vectorize stray nodes which are not connected to any other
    // nodes.
    return (node->users.empty() && node->operands.empty()) ||
           node->opsCount() <= 1;
  };

  // Update node vec sizes if its inputs vec sizes are smaller.
  // This is nedeed to handle situations when we have 3->3->4 sizes in tree.
  // TODO: It maybe possible to reconstruct the larger vec size combining src
  // smaller vector and scalar arg.
  for (auto *node : sortedNodes) {
    size_t size = node->opsCount();
    for (auto *operand : node->operands)
      size = std::min(size, operand->opsCount());

    if (size < node->opsCount()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Size mismatch, resizing node with " << node->opsCount()
                 << " operations to " << size << "\n");
      node->ops.resize(size);
    }

    while (node->opsCount() > 1) {
      if (checkOpVecType(node, isValidVecType))
        break;

      LLVM_DEBUG(llvm::dbgs() << "No a valid vector type, popping back op: "
                              << node->ops.back()->getName() << "\n");
      node->ops.pop_back();
    }
  }

  llvm::erase_if(sortedNodes, isBadNode);

  IRMapping mapping;
  for (auto *node : sortedNodes) {
    LLVM_DEBUG({
      llvm::dbgs() << "Processing node with " << node->opsCount()
                   << " operations\n";
      llvm::dbgs() << "  First op: " << *node->op() << "\n";
    });

    // `op` is the node with the smallest index in vector and not the
    // nessesarily the good insertion point.
    Operation *op = node->op();
    Operation *ip = node->getInsertionPoint();
    if (!ip)
      return op->emitError("no insertion point found for node");

    LLVM_DEBUG(llvm::dbgs() << "  Insertion point: " << *ip << "\n");

    rewriter.setInsertionPoint(ip);
    int64_t numElements = node->vectorSize();
    Location loc = op->getLoc();

    auto handleNonVectorInputs = [&](ValueRange operands) {
      // Handle the case when op operands are not vectorized or have smaller
      // vector size, construct the vector from the scalar operands using
      // FromElementsOp.
      for (auto [i, operand] : llvm::enumerate(operands)) {
        if (getNodeForOp(operand.getDefiningOp()))
          continue;

        SmallVector<Value> args;
        for (Operation *defOp : node->ops) {
          Value arg = defOp->getOperand(i);
          if (auto vecType = dyn_cast<VectorType>(arg.getType())) {
            assert(vecType.getRank() == 1);
            for (auto j : llvm::seq(vecType.getNumElements()))
              args.push_back(rewriter.create<vector::ExtractOp>(loc, arg, j));

          } else {
            args.push_back(arg);
          }
        }

        auto vecType = VectorType::get(numElements,
                                       getElementTypeOrSelf(operand.getType()));
        Value vector =
            rewriter.create<vector::FromElementsOp>(loc, vecType, args);
        mapping.map(operand, vector);
      }
    };

    auto handleNonVectorOutputs = [&](Value newResult,
                                      Type originalResultType) {
      // Handle the case when op results are not vectorized or have smaller
      // vector size, extract the elements from the vector.
      for (auto [i, result] : llvm::enumerate(node->ops)) {
        for (OpOperand &use : result->getUses()) {
          Operation *useOwner = use.getOwner();
          if (getNodeForOp(useOwner))
            continue;

          int64_t offset = i * node->elementsCount;
          Value elem;

          if (auto vecType = dyn_cast<VectorType>(originalResultType)) {
            assert(vecType.getRank() <= 1);
            if (vecType.getRank() == 0) {
              elem = rewriter.create<vector::ExtractOp>(loc, newResult, offset);
              elem = rewriter.create<vector::SplatOp>(loc, vecType, elem);
            } else {
              elem = rewriter.create<vector::ExtractStridedSliceOp>(
                  loc, newResult, offset, vecType.getNumElements(), 1);
            }
          } else {
            elem = rewriter.create<vector::ExtractOp>(loc, newResult, offset);
          }

          use.set(elem);
        }
      }
    };

    auto handleVecSizeMismatch = [&](Value arg, int64_t offset = 0) -> Value {
      // Handle vector size misamatch between 2 vectorized nodes.
      auto srcType = cast<VectorType>(arg.getType());
      assert(srcType.getRank() == 1);
      if (srcType.getDimSize(0) == numElements)
        return arg;

      return rewriter.create<vector::ExtractStridedSliceOp>(loc, arg, offset,
                                                            numElements, 1);
    };

    if (maybeReadOp(op)) {
      auto vecType =
          VectorType::get(numElements, getElementTypeAndCount(op)->first);
      Value result = rewriter.create<vector::LoadOp>(loc, vecType, getBase(op),
                                                     getIndices(op));
      Value originalResult = op->getResult(0);
      mapping.map(originalResult, result);
      handleNonVectorOutputs(result, originalResult.getType());
    } else if (maybeWriteOp(op)) {
      handleNonVectorInputs(getValueToStore(op));
      Value val = mapping.lookupOrDefault(getValueToStore(op));
      val = handleVecSizeMismatch(val);
      rewriter.create<vector::StoreOp>(loc, val, getBase(op), getIndices(op));
    } else if (isVectorizable(op)) {
      handleNonVectorInputs(op->getOperands());
      Operation *newOp = rewriter.clone(*op, mapping);
      Type resType = getElementTypeOrSelf(op->getResultTypes().front());
      auto resVectorType = VectorType::get(numElements, resType);

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
      handleNonVectorOutputs(newOp->getResult(0), op->getResultTypes().front());
    } else if (auto extract = dyn_cast<vector::ExtractOp>(op)) {
      // We alredy verified index is valid during graph construction, so
      // do need to check `getExtractIndex` result.
      int64_t offset = *getExtractIndex(extract);
      Value val = handleVecSizeMismatch(extract.getVector(), offset);
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
      LLVM_DEBUG(llvm::dbgs() << "Erasing op: " << *op << "\n");
      rewriter.eraseOp(op);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Vectorization completed successfully\n");
  return sortedNodes.size();
}

/// Try to vectorize ops in a block.
/// Returns number of nodes vectorized or error flag if failed.
static FailureOr<size_t>
tryToVectorizeInBlock(Block &block,
                      llvm::function_ref<bool(Type, size_t)> isValidVecType) {
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
        llvm::dbgs() << "  Contiguous group with " << contigGroup.opsCount()
                     << " operations\n";
      }
    });
    rootGroups.append(contiguousGroups);
  }

  // Build the SLP graph from root groups
  SLPGraph graph = buildSLPGraph(rootGroups);
  LLVM_DEBUG(graph.print());

  // Vectorize the graph
  IRRewriter rewriter(block.getParentOp()->getContext());
  FailureOr<size_t> numNodesVectorized =
      graph.vectorize(rewriter, isValidVecType);
  if (failed(numNodesVectorized))
    LLVM_DEBUG(llvm::dbgs() << "Failed to vectorize graph\n");

  return numNodesVectorized;
}

static bool isPow2(size_t size) {
  assert(size > 0);
  return (size & (size - 1)) == 0;
}

void GreedySLPVectorizerPass::runOnOperation() {
  Operation *op = getOperation();

  const DataLayout *dataLayout = nullptr;
  auto isValidVecType = [&](Type type, size_t count) {
    if (!isPow2(count))
      return false;

    if (!dataLayout)
      dataLayout = &getAnalysis<DataLayoutAnalysis>().getAtOrAbove(op);

    auto sizeInBits = dataLayout->getTypeSizeInBits(type);

    return sizeInBits * count <= this->maxVectorBitwidth;
  };

  // Run until fixed point is reached.
  bool changed;
  do {
    changed = false;
    auto visitor = [&](Block *block) -> WalkResult {
      FailureOr<size_t> numNodesVectorized =
          tryToVectorizeInBlock(*block, isValidVecType);
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
