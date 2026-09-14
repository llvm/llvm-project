//===- Normalize.cpp - Transforms IR into a normal form ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/CommutativityUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_NORMALIZEPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "normalize"

namespace {

class Normalize {
public:
  Normalize(IRRewriter &rewriter, DominanceInfo &domInfo,
            NormalizePassOptions &options)
      : rewriter(rewriter), domInfo(domInfo), options(options) {}
  /// Collect a vector of output ops within in \p root.
  void collectOutputs(Operation *root);

  /// Reorders ops by walking up the tree from each operand of an output op and
  /// reducing the def-use distance. This method assumes that output ops were
  /// collected top-down, otherwise the def-use chain may be broken. This method
  /// is a wrapper for recursive reorderOutput().
  void reorderOutputs();

  /// Assigns unique, sequential names (e.g., "a0", "a1") to all block arguments
  /// within \p root.
  void nameBlockArguments(Operation *root);

  /// Assigns unique, sequential names to all collected output operations.
  void nameOperations();

  /// Fold the operation name within \p root.
  void foldOperationsName(Operation *root);

  /// Greedily applies commutativity patterns using \p root to define the
  /// transformation scope.
  LogicalResult sortCommutativeOperands(Operation *root);

private:
  /// Reorders operations along the def-use chain from left to right, bottom to
  /// top, starting from \p producer.
  void reorderOutput(Operation *producer);

  /// Returns true if the \p is a terminator or contains memory/side effects.
  bool isOutput(Operation *op);

  /// Returns true if the \p op is an initial operation (has no operands or only
  /// constant-like operands).
  bool isInitialOperation(Operation *op);

  /// Assigns a unique name to \p op, using \p visited to track and skip already
  /// processed operations.
  void nameOperation(Operation *op, SmallPtrSet<Operation *, 32> &visited);

  /// Generates and assigns a stable, deterministic name to the initial \p op,
  /// while recursively resolving names for its upstream operands.
  void nameAsInitialOperation(Operation *op,
                              SmallPtrSet<Operation *, 32> &visited);
  void nameAsRegularOperation(Operation *op,
                              SmallPtrSet<Operation *, 32> &visited);

  /// Computes the output footprint for the given \p op.
  SetVector<int> getOutputFootprint(Operation *op,
                                    SmallPtrSet<Operation *, 32> &visited);
  /// Simplifies the name of the given \p op.
  void foldOperationName(Operation *op);

  Operation *getDominateOp(const SmallVectorImpl<Operation *> &ops);

  void appendCallAndOperandNames(Operation *op, SmallString<512> &name,
                                 SmallVectorImpl<StringRef> &operandNames);

  /// Collapses "$-"..."-$" nesting in \p name beyond \p depth, keeping only
  /// the outer \p depth levels of markers.
  std::string trimNameByDepth(StringRef name, int64_t depth);

  IRRewriter &rewriter;
  DominanceInfo &domInfo;
  NormalizePassOptions &options;

  /// Outputs collected by collectOutputs.
  SmallVector<Operation *> outputs;

  /// Caches, for output ops, their accumulated nested distance: the sum of
  /// each enclosing region's op-position, walked upward until the parent
  /// operation is a FunctionOpInterface.
  DenseMap<Operation *, int64_t> footprintCache;

  // Random constant for hashing, so the state isn't zero.
  const uint64_t magicHashConstant = 0x6acaa36bef8325c5ULL;
};

// Frozen mixer; basic-block names derived from these hashes appear in
// the normalized IR text and must be deterministic across processes
// for the normalizer's "compare normalized IR" workflow to work.
static constexpr uint64_t hash_16_bytes(uint64_t Low, uint64_t High) {
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t A = (Low ^ High) * kMul;
  A ^= (A >> 47);
  uint64_t B = (High ^ A) * kMul;
  B ^= (B >> 47);
  B *= kMul;
  return B;
}

/// Computes the 64-bit FNV-1a hash value of the given string \p data.
static constexpr uint64_t strHash(std::string_view data) noexcept {
  const uint64_t fnvOffset = 0xcbf29ce484222325ULL;
  const uint64_t fnvPrime = 0x100000001b3ULL;
  uint64_t hash = fnvOffset;
  for (const auto &c : data) {
    hash ^= static_cast<uint64_t>(c);
    hash *= fnvPrime;
  }
  return hash;
}

bool Normalize::isOutput(Operation *op) {
  if (!op)
    return false;
  return !isMemoryEffectFree(op) || op->hasTrait<OpTrait::IsTerminator>();
}

void Normalize::collectOutputs(Operation *root) {
  root->walk([&](Operation *op) {
    if (op == root)
      return WalkResult::advance();
    if (isOutput(op)) {
      LDBG() << "insert " << OpWithFlags(op, OpPrintingFlags().skipRegions())
             << " to outputs";
      outputs.push_back(op);
    }
    return WalkResult::advance();
  });
}

Operation *Normalize::getDominateOp(const SmallVectorImpl<Operation *> &ops) {
  if (ops.empty())
    return {};
  Operation *curDomOp = ops.front();
  for (size_t i = 1, e = ops.size(); i < e; ++i) {
    bool dominateA = domInfo.dominates(ops[i], curDomOp);
    if (dominateA) {
      LDBG() << OpWithFlags(ops[i], OpPrintingFlags().skipRegions())
             << "\ndominate\n"
             << OpWithFlags(curDomOp, OpPrintingFlags().skipRegions());
      curDomOp = ops[i];
      continue;
    }
    bool dominateB = domInfo.dominates(curDomOp, ops[i]);
    if (!dominateB) {
      LDBG() << OpWithFlags(ops[i], OpPrintingFlags().skipRegions())
             << "\nand\n"
             << OpWithFlags(curDomOp, OpPrintingFlags().skipRegions())
             << "\ndo not dominate each other";
      return {};
    }
  }
  return curDomOp;
}

void Normalize::reorderOutput(Operation *producer) {
  if (!isPure(producer))
    return;
  SmallVector<Operation *> users(producer->getUsers());
  if (Operation *domOp = getDominateOp(users)) {
    rewriter.moveOpBefore(producer, domOp);
    for (Value operand : producer->getOperands())
      if (Operation *defineOp = operand.getDefiningOp())
        reorderOutput(defineOp);
  }
}

void Normalize::reorderOutputs() {
  SmallPtrSet<Operation *, 16> visited;
  for (Operation *output : outputs) {
    for (Value operand : output->getOperands()) {
      if (Operation *defineOp = operand.getDefiningOp();
          defineOp && !visited.contains(defineOp)) {
        visited.insert(defineOp);
        reorderOutput(defineOp);
      }
    }
  }
}

bool Normalize::isInitialOperation(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (Operation *define = operand.getDefiningOp();
        !define || !define->hasTrait<OpTrait::ConstantLike>())
      return false;
  }
  return true;
}

/// Computes the output footprint for the given \p op.
///
/// Traverses downstream users recursively to find all reachable output
/// operations. For each output operation, it calculates its precise distance
/// (in terms of operation count) relative to the entry block of its enclosing
/// `FunctionOpInterface`.
SetVector<int>
Normalize::getOutputFootprint(Operation *op,
                              SmallPtrSet<Operation *, 32> &visited) {
  SetVector<int> outputs;
  if (visited.count(op))
    return outputs;
  visited.insert(op);

  // If the operation is an output, compute its nested absolute distance to the
  // Function entry.
  if (isOutput(op)) {
    if (footprintCache.contains(op)) {
      outputs.insert(footprintCache[op]);
      return outputs;
    }

    // Calculates the total distance of 'op' to its enclosing parent region's
    // start, accumulating nested offsets upward until the parent operation
    // matches `FunctionOpInterface`.
    int count = 0;
    Operation *curOp = op;
    do {
      Region *parentRegion = curOp->getParentRegion();
      int distance = 0;
      for (Operation &it : parentRegion->getOps()) {
        if (&it == curOp)
          break;
        ++distance;
      }
      count += distance;
      curOp = parentRegion->getParentOp();
    } while (!isa<FunctionOpInterface>(curOp));
    outputs.insert(count);
    footprintCache[op] = count;
    return outputs;
  }

  // Otherwise, recursively aggregate footprints from all downstream users.
  for (Operation *user : op->getUsers()) {
    SetVector<int> outputsUser = getOutputFootprint(user, visited);
    outputs.insert(outputsUser.begin(), outputsUser.end());
  }
  return outputs;
}

void Normalize::appendCallAndOperandNames(
    Operation *op, SmallString<512> &name,
    SmallVectorImpl<StringRef> &operandNames) {
  // In case of CallInst, consider callee in the operation name.
  if (auto callOp = dyn_cast<CallOpInterface>(op))
    if (auto funcOp = dyn_cast<FunctionOpInterface>(callOp.resolveCallable()))
      name.append(funcOp.getNameAttr());

  if (operandNames.size() > 0) {
    name.append("$-");
    for (size_t i = 0, e = operandNames.size(); i < e; ++i) {
      name.append(operandNames[i]);
      if (i < e - 1)
        name.append(".");
    }
    name.append("-$");
  }
  NameLoc loc = NameLoc::get(StringAttr::get(op->getContext(), name));
  LDBG() << "set NameLoc: " << loc
         << "\nfor: " << OpWithFlags(op, OpPrintingFlags().skipRegions());
  op->setLoc(loc);
}

/// Names operation following the scheme:
/// vl00000Callee(Operands)
///
/// Where 00000 is a hash calculated considering operation's opcode, output
/// footprint and block position. Callee's name is only included when
/// operation's type is `CallOpInterface`. The Operands are derived from the
/// names of the operation's operands.
void Normalize::nameAsInitialOperation(Operation *op,
                                       SmallPtrSet<Operation *, 32> &visited) {
  // Recursively name defining ops of operands and collect their names.
  SmallVector<StringRef, 4> operandNames;
  for (Value operand : op->getOperands()) {
    if (Operation *define = operand.getDefiningOp())
      nameOperation(define, visited);
    if (NameLoc loc = dyn_cast<NameLoc>(operand.getLoc()))
      operandNames.push_back(loc.getName());
  }

  // Early exit if the op don't have results.
  if (!op->getNumResults())
    return;

  // Initialize to a magic constant, so the state isn't zero.
  uint64_t hash = magicHashConstant;

  // Consider operation's opcode in the hash.
  hash = hash_16_bytes(hash, strHash(op->getName().getStringRef().str()));

  // Get output footprint for \p op.
  SmallPtrSet<Operation *, 32> visitedOutputFoot;
  SetVector<int> outputFootprint = getOutputFootprint(op, visitedOutputFoot);

  // Consider output footprint in the hash.
  for (const int &output : outputFootprint)
    hash = hash_16_bytes(hash, output);

  // Include the operation's relative position within its basic block.
  hash = hash_16_bytes(
      hash, std::distance(op->getBlock()->begin(), op->getIterator()));

  // Base operation name.
  SmallString<512> name;
  name.append("vl" + std::to_string(hash).substr(0, 5));

  // Append call and operand name.
  appendCallAndOperandNames(op, name, operandNames);
}

/// Names operation following the scheme:
/// op00000Callee(Operands)
///
/// Where 00000 is a hash calculated considering operation's opcode, its
/// operands' opcodes, and block position. Callee's name is only included
/// when operation's type is `CallOpInterface`, The Operands are derived from
/// the names of the operation's operands.
void Normalize::nameAsRegularOperation(Operation *op,
                                       SmallPtrSet<Operation *, 32> &visited) {
  // Recursively name defining ops of operands and collect their names.
  SmallVector<StringRef, 2> operandNames;
  for (Value operand : op->getOperands()) {
    if (Operation *define = operand.getDefiningOp())
      nameOperation(define, visited);
    if (NameLoc loc = dyn_cast<NameLoc>(operand.getLoc())) {
      operandNames.push_back(loc.getName());
    }
  }

  // Early exit if the op don't have results.
  if (!op->getNumResults())
    return;

  // Initialize to a magic constant, so the state isn't zero.
  uint64_t hash = magicHashConstant;

  // Consider operation opcode in the hash.
  uint64_t ophash = strHash(op->getName().getStringRef().str());
  hash = hash_16_bytes(hash, ophash);

  // Fuses the opcodes of upstream defining ops into the hash.
  for (Value operand : op->getOperands())
    if (Operation *define = operand.getDefiningOp())
      hash =
          hash_16_bytes(hash, strHash(define->getName().getStringRef().str()));

  // Include the operation's relative position within its basic block.
  hash = hash_16_bytes(
      hash, std::distance(op->getBlock()->begin(), op->getIterator()));

  // Base operation name.
  SmallString<512> name;
  name.append("op" + std::to_string(hash).substr(0, 5));

  // Append call and operand name.
  appendCallAndOperandNames(op, name, operandNames);
}

void Normalize::nameOperation(Operation *op,
                              SmallPtrSet<Operation *, 32> &visited) {
  if (visited.count(op))
    return;
  visited.insert(op);

  // Determine the type of operation to name.
  if (isInitialOperation(op)) {
    // This is an initial operation.
    nameAsInitialOperation(op, visited);
  } else {
    // This must be a regular operation.
    nameAsRegularOperation(op, visited);
  }
}

void Normalize::nameOperations() {
  SmallPtrSet<Operation *, 32> visited;
  for (Operation *op : outputs)
    nameOperation(op, visited);
}

void Normalize::nameBlockArguments(Operation *root) {
  size_t argumentCount = 0;
  MLIRContext *context = root->getContext();
  root->walk<WalkOrder::PreOrder>([&](Block *b) {
    for (auto argument : b->getArguments()) {
      NameLoc loc =
          NameLoc::get(StringAttr::get(context, "a" + Twine(argumentCount++)));
      argument.setLoc(loc);
    }
    return;
  });
}

/// Collapses nested "$-"..."-$" segments in `name` beyond the given
/// `depth`, keeping only the outer `depth` levels of markers.
///
/// "$-" / "-$" mark the start/end of a nested scope, e.g.
/// op80011$-a0.op11483-$ has one level of nesting.
/// depth == 0 strips it down to "op80011" (no markers left);
/// depth < 0, or fewer than depth+1 marker pairs, returns `name` unchanged.
std::string Normalize::trimNameByDepth(StringRef name, int64_t depth) {
  std::string str = name.str();
  if (depth < 0)
    return str;

  int targetCount = depth + 1;

  size_t startPos = 0;
  for (int i = 0, e = targetCount; i < e; ++i) {
    startPos = str.find("$-", startPos);
    if (startPos == std::string::npos)
      return str;
    startPos += 2;
  }

  size_t endPos = str.size() - 1;
  for (int i = 0, e = targetCount; i < e; ++i) {
    endPos = str.rfind("-$", endPos);
    if (endPos == std::string::npos)
      return str;
    endPos -= 2;
  }

  startPos -= 2;
  endPos += 2;

  if (startPos >= endPos)
    return str;

  return str.substr(0, startPos) + str.substr(endPos + 2);
}

/// Folds the name of \p op into a simplified form containing a truncated
/// prefix of its own name and its operands' names,
void Normalize::foldOperationName(Operation *op) {
  NameLoc loc = dyn_cast<NameLoc>(op->getLoc());

  // Only process operations prefixed with "op" since their names are complex
  // and need simplification.
  if (!loc || loc.getName().empty() || loc.getName().str().substr(0, 2) != "op")
    return;

  std::string name = trimNameByDepth(loc.getName().str(), options.foldDepth);
  loc = NameLoc::get(StringAttr::get(op->getContext(), name));
  LDBG() << "fold NameLoc: " << loc
         << "\nfor: " << OpWithFlags(op, OpPrintingFlags().skipRegions());
  op->setLoc(loc);
}

void Normalize::foldOperationsName(Operation *root) {
  if (options.foldDepth < 0)
    return;
  root->walk<WalkOrder::PreOrder>(
      [&](Operation *op) { foldOperationName(op); });
}

LogicalResult Normalize::sortCommutativeOperands(Operation *root) {
  MLIRContext *context = root->getContext();
  RewritePatternSet patterns(context);
  populateCommutativityUtilsPatterns(patterns);
  if (failed(applyPatternsGreedily(root, std::move(patterns))))
    return failure();
  return success();
}

struct NormalizePass : public impl::NormalizePassBase<NormalizePass> {
  using impl::NormalizePassBase<NormalizePass>::NormalizePassBase;
  void runOnOperation() override;
};
} // namespace

void NormalizePass::runOnOperation() {
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  IRRewriter rewriter(&getContext());
  NormalizePassOptions options = {foldDepth};
  Normalize normalize(rewriter, domInfo, options);

  // Sort commutative operands up front, so operand order doesn't need to be
  // re-sorted (by name) later when renaming ops.
  if (failed(normalize.sortCommutativeOperands(getOperation())))
    signalPassFailure();
  normalize.collectOutputs(getOperation());
  normalize.reorderOutputs();
  normalize.nameBlockArguments(getOperation());
  normalize.nameOperations();
  normalize.foldOperationsName(getOperation());

  // Since we only changed the positions of the operations, `DominanceInfo` and
  // `PostDominanceInfo` are marked as preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
}
