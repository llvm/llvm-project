//===- CommutativityUtils.cpp - Commutativity utilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a commutativity utility pattern and a function to
// populate this pattern. The function is intended to be used inside passes to
// simplify the matching of commutative operations by fixing the order of their
// operands.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/CommutativityUtils.h"

#include <queue>

using namespace mlir;

/// The possible "types" of ancestors. Here, an ancestor is an op or a block
/// argument present in the backward slice of a value.
enum AncestorType {
  /// Pertains to a block argument.
  BLOCK_ARGUMENT,

  /// Pertains to a non-constant-like op.
  NON_CONSTANT_OP,

  /// Pertains to a constant-like op.
  CONSTANT_OP
};

/// Stores the "key" associated with an ancestor.
struct AncestorKey {
  /// Holds `BLOCK_ARGUMENT`, `NON_CONSTANT_OP`, or `CONSTANT_OP`, depending on
  /// the ancestor.
  AncestorType type;

  /// Holds the op name of the ancestor if its `type` is `NON_CONSTANT_OP` or
  /// `CONSTANT_OP`. Else, holds "".
  StringRef opName;

  /// Constructor for `AncestorKey`.
  AncestorKey(Operation *op) {
    if (!op) {
      type = BLOCK_ARGUMENT;
    } else {
      type =
          op->hasTrait<OpTrait::ConstantLike>() ? CONSTANT_OP : NON_CONSTANT_OP;
      opName = op->getName().getStringRef();
    }
  }

  /// Overloaded operator `<` for `AncestorKey`.
  ///
  /// AncestorKeys of type `BLOCK_ARGUMENT` are considered the smallest, those
  /// of type `CONSTANT_OP`, the largest, and `NON_CONSTANT_OP` types come in
  /// between. Within the types `NON_CONSTANT_OP` and `CONSTANT_OP`, the smaller
  /// ones are the ones with smaller op names (lexicographically).
  ///
  /// TODO: Include other information like attributes, value type, etc., to
  /// enhance this comparison. For example, currently this comparison doesn't
  /// differentiate between `cmpi sle` and `cmpi sgt` or `addi (in i32)` and
  /// `addi (in i64)`. Such an enhancement should only be done if the need
  /// arises.
  bool operator<(const AncestorKey &key) const {
    return std::tie(type, opName) < std::tie(key.type, key.opName);
  }
};

/// Stores a commutative operand along with its BFS traversal information.
struct CommutativeOperand {
  /// Stores the operand.
  Value operand;

  /// Stores the queue of ancestors of the operand's BFS traversal at a
  /// particular point in time.
  std::queue<Operation *> ancestorQueue;

  /// Stores the list of ancestors that have been visited by the BFS traversal
  /// at a particular point in time.
  DenseSet<Operation *> visitedAncestors;

  /// Stores the operand's "key". This "key" is defined as a list of the
  /// "AncestorKeys" associated with the ancestors of this operand, in a
  /// breadth-first order.
  ///
  /// So, if an operand, say `A`, was produced as follows:
  ///
  /// `<block argument>`  `<block argument>`
  ///             \          /
  ///              \        /
  ///             `arith.subi`           `arith.constant`
  ///                       \            /
  ///                        `arith.addi`
  ///                              |
  ///                         returns `A`
  ///
  /// Then, the ancestors of `A`, in the breadth-first order are:
  /// `arith.addi`, `arith.subi`, `arith.constant`, `<block argument>`, and
  /// `<block argument>`.
  ///
  /// Thus, the "key" associated with operand `A` is:
  /// {
  ///  {type: `NON_CONSTANT_OP`, opName: "arith.addi"},
  ///  {type: `NON_CONSTANT_OP`, opName: "arith.subi"},
  ///  {type: `CONSTANT_OP`, opName: "arith.constant"},
  ///  {type: `BLOCK_ARGUMENT`, opName: ""},
  ///  {type: `BLOCK_ARGUMENT`, opName: ""}
  /// }
  SmallVector<AncestorKey, 4> key;

  /// Push an ancestor into the operand's BFS information structure. This
  /// entails it being pushed into the queue (always) and inserted into the
  /// "visited ancestors" list (iff it is an op rather than a block argument).
  void pushAncestor(Operation *op) {
    ancestorQueue.push(op);
    if (op)
      visitedAncestors.insert(op);
  }

  /// Refresh the key.
  ///
  /// Refreshing a key entails making it up-to-date with the operand's BFS
  /// traversal that has happened till that point in time, i.e, appending the
  /// existing key with the front ancestor's "AncestorKey". Note that a key
  /// directly reflects the BFS and thus needs to be refreshed during the
  /// progression of the traversal.
  void refreshKey() {
    if (ancestorQueue.empty())
      return;

    Operation *frontAncestor = ancestorQueue.front();
    AncestorKey frontAncestorKey(frontAncestor);
    key.push_back(frontAncestorKey);
  }

  /// Pop the front ancestor, if any, from the queue and then push its adjacent
  /// unvisited ancestors, if any, to the queue (this is the main body of the
  /// BFS algorithm).
  void popFrontAndPushAdjacentUnvisitedAncestors() {
    if (ancestorQueue.empty())
      return;
    Operation *frontAncestor = ancestorQueue.front();
    ancestorQueue.pop();
    if (!frontAncestor)
      return;
    for (Value operand : frontAncestor->getOperands()) {
      Operation *operandDefOp = operand.getDefiningOp();
      if (!operandDefOp || !visitedAncestors.contains(operandDefOp))
        pushAncestor(operandDefOp);
    }
  }
};

/// Sorts the operands of `op` in ascending order of the "key" associated with
/// each operand iff `op` is commutative. This is a stable sort.
///
/// After the application of this pattern, since the commutative operands now
/// have a deterministic order in which they occur in an op, the matching of
/// large DAGs becomes much simpler, i.e., requires much less number of checks
/// to be written by a user in her/his pattern matching function.
///
/// Some examples of such a sorting:
///
/// Assume that the sorting is being applied to `foo.commutative`, which is a
/// commutative op.
///
/// Example 1:
///
/// %1 = foo.const 0
/// %2 = foo.mul <block argument>, <block argument>
/// %3 = foo.commutative %1, %2
///
/// Here,
/// 1. The key associated with %1 is:
///     `{
///       {CONSTANT_OP, "foo.const"}
///      }`
/// 2. The key associated with %2 is:
///     `{
///       {NON_CONSTANT_OP, "foo.mul"},
///       {BLOCK_ARGUMENT, ""},
///       {BLOCK_ARGUMENT, ""}
///      }`
///
/// The key of %2 < the key of %1
/// Thus, the sorted `foo.commutative` is:
/// %3 = foo.commutative %2, %1
///
/// Example 2:
///
/// %1 = foo.const 0
/// %2 = foo.mul <block argument>, <block argument>
/// %3 = foo.mul %2, %1
/// %4 = foo.add %2, %1
/// %5 = foo.commutative %1, %2, %3, %4
///
/// Here,
/// 1. The key associated with %1 is:
///     `{
///       {CONSTANT_OP, "foo.const"}
///      }`
/// 2. The key associated with %2 is:
///     `{
///       {NON_CONSTANT_OP, "foo.mul"},
///       {BLOCK_ARGUMENT, ""}
///      }`
/// 3. The key associated with %3 is:
///     `{
///       {NON_CONSTANT_OP, "foo.mul"},
///       {NON_CONSTANT_OP, "foo.mul"},
///       {CONSTANT_OP, "foo.const"},
///       {BLOCK_ARGUMENT, ""},
///       {BLOCK_ARGUMENT, ""}
///      }`
/// 4. The key associated with %4 is:
///     `{
///       {NON_CONSTANT_OP, "foo.add"},
///       {NON_CONSTANT_OP, "foo.mul"},
///       {CONSTANT_OP, "foo.const"},
///       {BLOCK_ARGUMENT, ""},
///       {BLOCK_ARGUMENT, ""}
///      }`
///
/// Thus, the sorted `foo.commutative` is:
/// %5 = foo.commutative %4, %3, %2, %1
class SortCommutativeOperands : public RewritePattern {
public:
  SortCommutativeOperands(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/5, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Custom comparator for two commutative operands, which returns true iff
    // the "key" of `constCommOperandA` < the "key" of `constCommOperandB`,
    // i.e.,
    // 1. In the first unequal pair of corresponding AncestorKeys, the
    // AncestorKey in `constCommOperandA` is smaller, or,
    // 2. Both the AncestorKeys in every pair are the same and the size of
    // `constCommOperandA`'s "key" is smaller.
    auto commutativeOperandComparator =
        [](const std::unique_ptr<CommutativeOperand> &constCommOperandA,
           const std::unique_ptr<CommutativeOperand> &constCommOperandB) {
          if (constCommOperandA->operand == constCommOperandB->operand)
            return false;

          auto &commOperandA =
              const_cast<std::unique_ptr<CommutativeOperand> &>(
                  constCommOperandA);
          auto &commOperandB =
              const_cast<std::unique_ptr<CommutativeOperand> &>(
                  constCommOperandB);

          // Iteratively perform the BFS's of both operands until an order among
          // them can be determined.
          unsigned keyIndex = 0;
          while (true) {
            if (commOperandA->key.size() <= keyIndex) {
              if (commOperandA->ancestorQueue.empty())
                return true;
              commOperandA->popFrontAndPushAdjacentUnvisitedAncestors();
              commOperandA->refreshKey();
            }
            if (commOperandB->key.size() <= keyIndex) {
              if (commOperandB->ancestorQueue.empty())
                return false;
              commOperandB->popFrontAndPushAdjacentUnvisitedAncestors();
              commOperandB->refreshKey();
            }
            if (commOperandA->ancestorQueue.empty() ||
                commOperandB->ancestorQueue.empty())
              return commOperandA->key.size() < commOperandB->key.size();
            if (commOperandA->key[keyIndex] < commOperandB->key[keyIndex])
              return true;
            if (commOperandB->key[keyIndex] < commOperandA->key[keyIndex])
              return false;
            keyIndex++;
          }
        };

    // If `op` is not commutative, do nothing.
    if (!op->hasTrait<OpTrait::IsCommutative>())
      return failure();

    // Populate the list of commutative operands.
    SmallVector<Value, 2> operands = op->getOperands();
    SmallVector<std::unique_ptr<CommutativeOperand>, 2> commOperands;
    for (Value operand : operands) {
      std::unique_ptr<CommutativeOperand> commOperand =
          std::make_unique<CommutativeOperand>();
      commOperand->operand = operand;
      commOperand->pushAncestor(operand.getDefiningOp());
      commOperand->refreshKey();
      commOperands.push_back(std::move(commOperand));
    }

    // Sort the operands.
    std::stable_sort(commOperands.begin(), commOperands.end(),
                     commutativeOperandComparator);
    SmallVector<Value, 2> sortedOperands;
    for (const std::unique_ptr<CommutativeOperand> &commOperand : commOperands)
      sortedOperands.push_back(commOperand->operand);
    if (sortedOperands == operands)
      return failure();
    rewriter.updateRootInPlace(op, [&] { op->setOperands(sortedOperands); });
    return success();
  }
};

void mlir::populateCommutativityUtilsPatterns(RewritePatternSet &patterns) {
  patterns.add<SortCommutativeOperands>(patterns.getContext());
}
