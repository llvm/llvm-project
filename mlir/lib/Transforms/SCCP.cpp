//===- SCCP.cpp - Sparse Conditional Constant Propagation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs a sparse conditional constant propagation
// in MLIR. It identifies values known to be constant, propagates that
// information throughout the IR, and replaces them. This is done with an
// optimistic dataflow analysis that assumes that all values are constant until
// proven otherwise.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// SCCP Analysis
//===----------------------------------------------------------------------===//

namespace {
struct SCCPLatticeValue {
  SCCPLatticeValue(Attribute constant = {}, Dialect *dialect = nullptr)
      : constant(constant), constantDialect(dialect) {}

  /// The pessimistic state of SCCP is non-constant.
  static SCCPLatticeValue getPessimisticValueState(MLIRContext *context) {
    return SCCPLatticeValue();
  }
  static SCCPLatticeValue getPessimisticValueState(Value value) {
    return SCCPLatticeValue();
  }

  /// Equivalence for SCCP only accounts for the constant, not the originating
  /// dialect.
  bool operator==(const SCCPLatticeValue &rhs) const {
    return constant == rhs.constant;
  }

  /// To join the state of two values, we simply check for equivalence.
  static SCCPLatticeValue join(const SCCPLatticeValue &lhs,
                               const SCCPLatticeValue &rhs) {
    return lhs == rhs ? lhs : SCCPLatticeValue();
  }

  /// The constant attribute value.
  Attribute constant;

  /// The dialect the constant originated from. This is not used as part of the
  /// key, and is only needed to materialize the held constant if necessary.
  Dialect *constantDialect;
};

struct SCCPAnalysis : public ForwardDataFlowAnalysis<SCCPLatticeValue> {
  using ForwardDataFlowAnalysis<SCCPLatticeValue>::ForwardDataFlowAnalysis;
  ~SCCPAnalysis() override = default;

  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<SCCPLatticeValue> *> operands) final {
    // Don't try to simulate the results of a region operation as we can't
    // guarantee that folding will be out-of-place. We don't allow in-place
    // folds as the desire here is for simulated execution, and not general
    // folding.
    if (op->getNumRegions())
      return markAllPessimisticFixpoint(op->getResults());

    SmallVector<Attribute> constantOperands(
        llvm::map_range(operands, [](LatticeElement<SCCPLatticeValue> *value) {
          return value->getValue().constant;
        }));

    // Save the original operands and attributes just in case the operation
    // folds in-place. The constant passed in may not correspond to the real
    // runtime value, so in-place updates are not allowed.
    SmallVector<Value, 8> originalOperands(op->getOperands());
    DictionaryAttr originalAttrs = op->getAttrDictionary();

    // Simulate the result of folding this operation to a constant. If folding
    // fails or was an in-place fold, mark the results as overdefined.
    SmallVector<OpFoldResult, 8> foldResults;
    foldResults.reserve(op->getNumResults());
    if (failed(op->fold(constantOperands, foldResults)))
      return markAllPessimisticFixpoint(op->getResults());

    // If the folding was in-place, mark the results as overdefined and reset
    // the operation. We don't allow in-place folds as the desire here is for
    // simulated execution, and not general folding.
    if (foldResults.empty()) {
      op->setOperands(originalOperands);
      op->setAttrs(originalAttrs);
      return markAllPessimisticFixpoint(op->getResults());
    }

    // Merge the fold results into the lattice for this operation.
    assert(foldResults.size() == op->getNumResults() && "invalid result size");
    Dialect *dialect = op->getDialect();
    ChangeResult result = ChangeResult::NoChange;
    for (unsigned i = 0, e = foldResults.size(); i != e; ++i) {
      LatticeElement<SCCPLatticeValue> &lattice =
          getLatticeElement(op->getResult(i));

      // Merge in the result of the fold, either a constant or a value.
      OpFoldResult foldResult = foldResults[i];
      if (Attribute attr = foldResult.dyn_cast<Attribute>())
        result |= lattice.join(SCCPLatticeValue(attr, dialect));
      else
        result |= lattice.join(getLatticeElement(foldResult.get<Value>()));
    }
    return result;
  }

  /// Implementation of `getSuccessorsForOperands` that uses constant operands
  /// to potentially remove dead successors.
  LogicalResult getSuccessorsForOperands(
      BranchOpInterface branch,
      ArrayRef<LatticeElement<SCCPLatticeValue> *> operands,
      SmallVectorImpl<Block *> &successors) final {
    SmallVector<Attribute> constantOperands(
        llvm::map_range(operands, [](LatticeElement<SCCPLatticeValue> *value) {
          return value->getValue().constant;
        }));
    if (Block *singleSucc = branch.getSuccessorForOperands(constantOperands)) {
      successors.push_back(singleSucc);
      return success();
    }
    return failure();
  }

  /// Implementation of `getSuccessorsForOperands` that uses constant operands
  /// to potentially remove dead region successors.
  void getSuccessorsForOperands(
      RegionBranchOpInterface branch, Optional<unsigned> sourceIndex,
      ArrayRef<LatticeElement<SCCPLatticeValue> *> operands,
      SmallVectorImpl<RegionSuccessor> &successors) final {
    SmallVector<Attribute> constantOperands(
        llvm::map_range(operands, [](LatticeElement<SCCPLatticeValue> *value) {
          return value->getValue().constant;
        }));
    branch.getSuccessorRegions(sourceIndex, constantOperands, successors);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// SCCP Rewrites
//===----------------------------------------------------------------------===//

/// Replace the given value with a constant if the corresponding lattice
/// represents a constant. Returns success if the value was replaced, failure
/// otherwise.
static LogicalResult replaceWithConstant(SCCPAnalysis &analysis,
                                         OpBuilder &builder,
                                         OperationFolder &folder, Value value) {
  LatticeElement<SCCPLatticeValue> *lattice =
      analysis.lookupLatticeElement(value);
  if (!lattice)
    return failure();
  SCCPLatticeValue &latticeValue = lattice->getValue();
  if (!latticeValue.constant)
    return failure();

  // Attempt to materialize a constant for the given value.
  Dialect *dialect = latticeValue.constantDialect;
  Value constant = folder.getOrCreateConstant(
      builder, dialect, latticeValue.constant, value.getType(), value.getLoc());
  if (!constant)
    return failure();

  value.replaceAllUsesWith(constant);
  return success();
}

/// Rewrite the given regions using the computing analysis. This replaces the
/// uses of all values that have been computed to be constant, and erases as
/// many newly dead operations.
static void rewrite(SCCPAnalysis &analysis, MLIRContext *context,
                    MutableArrayRef<Region> initialRegions) {
  SmallVector<Block *> worklist;
  auto addToWorklist = [&](MutableArrayRef<Region> regions) {
    for (Region &region : regions)
      for (Block &block : llvm::reverse(region))
        worklist.push_back(&block);
  };

  // An operation folder used to create and unique constants.
  OperationFolder folder(context);
  OpBuilder builder(context);

  addToWorklist(initialRegions);
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();

    for (Operation &op : llvm::make_early_inc_range(*block)) {
      builder.setInsertionPoint(&op);

      // Replace any result with constants.
      bool replacedAll = op.getNumResults() != 0;
      for (Value res : op.getResults())
        replacedAll &=
            succeeded(replaceWithConstant(analysis, builder, folder, res));

      // If all of the results of the operation were replaced, try to erase
      // the operation completely.
      if (replacedAll && wouldOpBeTriviallyDead(&op)) {
        assert(op.use_empty() && "expected all uses to be replaced");
        op.erase();
        continue;
      }

      // Add any the regions of this operation to the worklist.
      addToWorklist(op.getRegions());
    }

    // Replace any block arguments with constants.
    builder.setInsertionPointToStart(block);
    for (BlockArgument arg : block->getArguments())
      (void)replaceWithConstant(analysis, builder, folder, arg);
  }
}

//===----------------------------------------------------------------------===//
// SCCP Pass
//===----------------------------------------------------------------------===//

namespace {
struct SCCP : public SCCPBase<SCCP> {
  void runOnOperation() override;
};
} // end anonymous namespace

void SCCP::runOnOperation() {
  Operation *op = getOperation();

  SCCPAnalysis analysis(op->getContext());
  analysis.run(op);
  rewrite(analysis, op->getContext(), op->getRegions());
}

std::unique_ptr<Pass> mlir::createSCCPPass() {
  return std::make_unique<SCCP>();
}
