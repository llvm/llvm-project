#include "mlir/Analysis/SliceWalk.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

using namespace mlir;

WalkContinuation mlir::walkSlice(ValueRange rootValues,
                                 WalkCallback walkCallback) {
  // Search the backward slice starting from the root values.
  SmallVector<Value> workList = rootValues;
  llvm::SmallDenseSet<Value, 16> seenValues;
  while (!workList.empty()) {
    // Search the backward slice of the current value.
    Value current = workList.pop_back_val();

    // Skip the current value if it has already been seen.
    if (!seenValues.insert(current).second)
      continue;

    // Call the walk callback with the current value.
    WalkContinuation continuation = walkCallback(current);
    if (continuation.wasInterrupted())
      return continuation;
    if (continuation.wasSkipped())
      continue;

    assert(continuation.wasAdvancedTo());
    // Add the next values to the work list if the walk should continue.
    workList.append(continuation.getNextValues().begin(),
                    continuation.getNextValues().end());
  }

  return WalkContinuation::skip();
}

/// Returns the predecessor branch operands that match `blockArg`, or nullopt if
/// some of the predecessor terminators do not implement the BranchOpInterface.
static std::optional<SmallVector<Value>>
getBlockPredecessorOperands(BlockArgument blockArg) {
  Block *block = blockArg.getOwner();

  // Search the predecessor operands for all predecessor terminators.
  SmallVector<Value> predecessorOperands;
  for (auto it = block->pred_begin(); it != block->pred_end(); ++it) {
    Block *predecessor = *it;
    auto branchOp = dyn_cast<BranchOpInterface>(predecessor->getTerminator());
    if (!branchOp)
      return std::nullopt;
    SuccessorOperands successorOperands =
        branchOp.getSuccessorOperands(it.getSuccessorIndex());
    // Store the predecessor operand if the block argument matches an operand
    // and is not produced by the terminator.
    if (Value operand = successorOperands[blockArg.getArgNumber()])
      predecessorOperands.push_back(operand);
  }

  return predecessorOperands;
}

std::optional<SmallVector<Value>>
mlir::getControlFlowPredecessors(Value value) {
  if (OpResult opResult = dyn_cast<OpResult>(value)) {
    if (auto selectOp = opResult.getDefiningOp<SelectLikeOpInterface>())
      return SmallVector<Value>(
          {selectOp.getTrueValue(), selectOp.getFalseValue()});
    auto regionOp = opResult.getDefiningOp<RegionBranchOpInterface>();
    // If the interface is not implemented, there are no control flow
    // predecessors to work with.
    if (!regionOp)
      return std::nullopt;
    // Add the control flow predecessor operands to the work list.
    RegionSuccessor parentSuccessor = RegionSuccessor::parent();
    // Find the position of `opResult` in the successor inputs of the parent.
    // `getPredecessorValues` indexes into the successor inputs, not into the
    // op results directly, since some results may not be successor inputs.
    ValueRange successorInputs = regionOp.getSuccessorInputs(parentSuccessor);
    auto it = llvm::find(successorInputs, opResult);
    if (it == successorInputs.end())
      return std::nullopt;
    SmallVector<Value> predecessorOperands;
    regionOp.getPredecessorValues(parentSuccessor,
                                  std::distance(successorInputs.begin(), it),
                                  predecessorOperands);
    return predecessorOperands;
  }

  auto blockArg = cast<BlockArgument>(value);
  Block *block = blockArg.getOwner();
  // Search the region predecessor operands for structured control flow.
  if (block->isEntryBlock()) {
    if (auto regionBranchOp =
            dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
      RegionSuccessor regionSuccessor(blockArg.getParentRegion());
      // Find the position of `blockArg` in the successor inputs of the region.
      // `getPredecessorValues` indexes into the successor inputs, not into the
      // block arguments directly, since some block arguments may not be
      // successor inputs (e.g., block arguments produced by the terminator).
      ValueRange successorInputs =
          regionBranchOp.getSuccessorInputs(regionSuccessor);
      auto it = llvm::find(successorInputs, blockArg);
      if (it == successorInputs.end())
        return std::nullopt;
      SmallVector<Value> predecessorOperands;
      regionBranchOp.getPredecessorValues(
          regionSuccessor, std::distance(successorInputs.begin(), it),
          predecessorOperands);
      return predecessorOperands;
    }
    // If the interface is not implemented, there are no control flow
    // predecessors to work with.
    return std::nullopt;
  }

  // Search the block predecessor operands for unstructured control flow.
  return getBlockPredecessorOperands(blockArg);
}
