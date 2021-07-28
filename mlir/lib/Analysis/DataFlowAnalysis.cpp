//===- DataFlowAnalysis.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
/// This class contains various state used when computing the lattice elements
/// of a callable operation.
class CallableLatticeState {
public:
  /// Build a lattice state with a given callable region, and a specified number
  /// of results to be initialized to the default lattice element.
  CallableLatticeState(ForwardDataFlowAnalysisBase &analysis,
                       Region *callableRegion, unsigned numResults)
      : callableArguments(callableRegion->getArguments()),
        resultLatticeElements(numResults) {
    for (AbstractLatticeElement *&it : resultLatticeElements)
      it = analysis.createLatticeElement();
  }

  /// Returns the arguments to the callable region.
  Block::BlockArgListType getCallableArguments() const {
    return callableArguments;
  }

  /// Returns the lattice element for the results of the callable region.
  auto getResultLatticeElements() {
    return llvm::make_pointee_range(resultLatticeElements);
  }

  /// Add a call to this callable. This is only used if the callable defines a
  /// symbol.
  void addSymbolCall(Operation *op) { symbolCalls.push_back(op); }

  /// Return the calls that reference this callable. This is only used
  /// if the callable defines a symbol.
  ArrayRef<Operation *> getSymbolCalls() const { return symbolCalls; }

private:
  /// The arguments of the callable region.
  Block::BlockArgListType callableArguments;

  /// The lattice state for each of the results of this region. The return
  /// values of the callable aren't SSA values, so we need to track them
  /// separately.
  SmallVector<AbstractLatticeElement *, 4> resultLatticeElements;

  /// The calls referencing this callable if this callable defines a symbol.
  /// This removes the need to recompute symbol references during propagation.
  /// Value based references are trivial to resolve, so they can be done
  /// in-place.
  SmallVector<Operation *, 4> symbolCalls;
};

/// This class represents the solver for a forward dataflow analysis. This class
/// acts as the propagation engine for computing which lattice elements.
class ForwardDataFlowSolver {
public:
  /// Initialize the solver with the given top-level operation.
  ForwardDataFlowSolver(ForwardDataFlowAnalysisBase &analysis, Operation *op);

  /// Run the solver until it converges.
  void solve();

private:
  /// Initialize the set of symbol defining callables that can have their
  /// arguments and results tracked. 'op' is the top-level operation that the
  /// solver is operating on.
  void initializeSymbolCallables(Operation *op);

  /// Visit the users of the given IR that reside within executable blocks.
  template <typename T>
  void visitUsers(T &value) {
    for (Operation *user : value.getUsers())
      if (isBlockExecutable(user->getBlock()))
        visitOperation(user);
  }

  /// Visit the given operation and compute any necessary lattice state.
  void visitOperation(Operation *op);

  /// Visit the given call operation and compute any necessary lattice state.
  void visitCallOperation(CallOpInterface op);

  /// Visit the given callable operation and compute any necessary lattice
  /// state.
  void visitCallableOperation(Operation *op);

  /// Visit the given region branch operation, which defines regions, and
  /// compute any necessary lattice state. This also resolves the lattice state
  /// of both the operation results and any nested regions.
  void visitRegionBranchOperation(
      RegionBranchOpInterface branch,
      ArrayRef<AbstractLatticeElement *> operandLattices);

  /// Visit the given set of region successors, computing any necessary lattice
  /// state. The provided function returns the input operands to the region at
  /// the given index. If the index is 'None', the input operands correspond to
  /// the parent operation results.
  void visitRegionSuccessors(
      Operation *parentOp, ArrayRef<RegionSuccessor> regionSuccessors,
      function_ref<OperandRange(Optional<unsigned>)> getInputsForRegion);

  /// Visit the given terminator operation and compute any necessary lattice
  /// state.
  void
  visitTerminatorOperation(Operation *op,
                           ArrayRef<AbstractLatticeElement *> operandLattices);

  /// Visit the given terminator operation that exits a callable region. These
  /// are terminators with no CFG successors.
  void visitCallableTerminatorOperation(
      Operation *callable, Operation *terminator,
      ArrayRef<AbstractLatticeElement *> operandLattices);

  /// Visit the given block and compute any necessary lattice state.
  void visitBlock(Block *block);

  /// Visit argument #'i' of the given block and compute any necessary lattice
  /// state.
  void visitBlockArgument(Block *block, int i);

  /// Mark the entry block of the given region as executable. Returns NoChange
  /// if the block was already marked executable. If `markPessimisticFixpoint`
  /// is true, the arguments of the entry block are also marked as having
  /// reached the pessimistic fixpoint.
  ChangeResult markEntryBlockExecutable(Region *region,
                                        bool markPessimisticFixpoint);

  /// Mark the given block as executable. Returns NoChange if the block was
  /// already marked executable.
  ChangeResult markBlockExecutable(Block *block);

  /// Returns true if the given block is executable.
  bool isBlockExecutable(Block *block) const;

  /// Mark the edge between 'from' and 'to' as executable.
  void markEdgeExecutable(Block *from, Block *to);

  /// Return true if the edge between 'from' and 'to' is executable.
  bool isEdgeExecutable(Block *from, Block *to) const;

  /// Mark the given value as having reached the pessimistic fixpoint. This
  /// means that we cannot further refine the state of this value.
  void markPessimisticFixpoint(Value value);

  /// Mark all of the given values as having reaching the pessimistic fixpoint.
  template <typename ValuesT>
  void markAllPessimisticFixpoint(ValuesT values) {
    for (auto value : values)
      markPessimisticFixpoint(value);
  }
  template <typename ValuesT>
  void markAllPessimisticFixpoint(Operation *op, ValuesT values) {
    markAllPessimisticFixpoint(values);
    opWorklist.push_back(op);
  }
  template <typename ValuesT>
  void markAllPessimisticFixpointAndVisitUsers(ValuesT values) {
    for (auto value : values) {
      AbstractLatticeElement &lattice = analysis.getLatticeElement(value);
      if (lattice.markPessimisticFixpoint() == ChangeResult::Change)
        visitUsers(value);
    }
  }

  /// Returns true if the given value was marked as having reached the
  /// pessimistic fixpoint.
  bool isAtFixpoint(Value value) const;

  /// Merge in the given lattice 'from' into the lattice 'to'. 'owner'
  /// corresponds to the parent operation of the lattice for 'to'.
  void join(Operation *owner, AbstractLatticeElement &to,
            const AbstractLatticeElement &from);

  /// A reference to the dataflow analysis being computed.
  ForwardDataFlowAnalysisBase &analysis;

  /// The set of blocks that are known to execute, or are intrinsically live.
  SmallPtrSet<Block *, 16> executableBlocks;

  /// The set of control flow edges that are known to execute.
  DenseSet<std::pair<Block *, Block *>> executableEdges;

  /// A worklist containing blocks that need to be processed.
  SmallVector<Block *, 64> blockWorklist;

  /// A worklist of operations that need to be processed.
  SmallVector<Operation *, 64> opWorklist;

  /// The callable operations that have their argument/result state tracked.
  DenseMap<Operation *, CallableLatticeState> callableLatticeState;

  /// A map between a call operation and the resolved symbol callable. This
  /// avoids re-resolving symbol references during propagation. Value based
  /// callables are trivial to resolve, so they can be done in-place.
  DenseMap<Operation *, Operation *> callToSymbolCallable;

  /// A symbol table used for O(1) symbol lookups during simplification.
  SymbolTableCollection symbolTable;
};
} // end anonymous namespace

ForwardDataFlowSolver::ForwardDataFlowSolver(
    ForwardDataFlowAnalysisBase &analysis, Operation *op)
    : analysis(analysis) {
  /// Initialize the solver with the regions within this operation.
  for (Region &region : op->getRegions()) {
    // Mark the entry block as executable. The values passed to these regions
    // are also invisible, so mark any arguments as reaching the pessimistic
    // fixpoint.
    markEntryBlockExecutable(&region, /*markPessimisticFixpoint=*/true);
  }
  initializeSymbolCallables(op);
}

void ForwardDataFlowSolver::solve() {
  while (!blockWorklist.empty() || !opWorklist.empty()) {
    // Process any operations in the op worklist.
    while (!opWorklist.empty())
      visitUsers(*opWorklist.pop_back_val());

    // Process any blocks in the block worklist.
    while (!blockWorklist.empty())
      visitBlock(blockWorklist.pop_back_val());
  }
}

void ForwardDataFlowSolver::initializeSymbolCallables(Operation *op) {
  // Initialize the set of symbol callables that can have their state tracked.
  // This tracks which symbol callable operations we can propagate within and
  // out of.
  auto walkFn = [&](Operation *symTable, bool allUsesVisible) {
    Region &symbolTableRegion = symTable->getRegion(0);
    Block *symbolTableBlock = &symbolTableRegion.front();
    for (auto callable : symbolTableBlock->getOps<CallableOpInterface>()) {
      // We won't be able to track external callables.
      Region *callableRegion = callable.getCallableRegion();
      if (!callableRegion)
        continue;
      // We only care about symbol defining callables here.
      auto symbol = dyn_cast<SymbolOpInterface>(callable.getOperation());
      if (!symbol)
        continue;
      callableLatticeState.try_emplace(callable, analysis, callableRegion,
                                       callable.getCallableResults().size());

      // If not all of the uses of this symbol are visible, we can't track the
      // state of the arguments.
      if (symbol.isPublic() || (!allUsesVisible && symbol.isNested())) {
        for (Region &region : callable->getRegions())
          markEntryBlockExecutable(&region, /*markPessimisticFixpoint=*/true);
      }
    }
    if (callableLatticeState.empty())
      return;

    // After computing the valid callables, walk any symbol uses to check
    // for non-call references. We won't be able to track the lattice state
    // for arguments to these callables, as we can't guarantee that we can see
    // all of its calls.
    Optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(&symbolTableRegion);
    if (!uses) {
      // If we couldn't gather the symbol uses, conservatively assume that
      // we can't track information for any nested symbols.
      op->walk([&](CallableOpInterface op) { callableLatticeState.erase(op); });
      return;
    }

    for (const SymbolTable::SymbolUse &use : *uses) {
      // If the use is a call, track it to avoid the need to recompute the
      // reference later.
      if (auto callOp = dyn_cast<CallOpInterface>(use.getUser())) {
        Operation *symCallable = callOp.resolveCallable(&symbolTable);
        auto callableLatticeIt = callableLatticeState.find(symCallable);
        if (callableLatticeIt != callableLatticeState.end()) {
          callToSymbolCallable.try_emplace(callOp, symCallable);

          // We only need to record the call in the lattice if it produces any
          // values.
          if (callOp->getNumResults())
            callableLatticeIt->second.addSymbolCall(callOp);
        }
        continue;
      }
      // This use isn't a call, so don't we know all of the callers.
      auto *symbol = symbolTable.lookupSymbolIn(op, use.getSymbolRef());
      auto it = callableLatticeState.find(symbol);
      if (it != callableLatticeState.end()) {
        for (Region &region : it->first->getRegions())
          markEntryBlockExecutable(&region, /*markPessimisticFixpoint=*/true);
      }
    }
  };
  SymbolTable::walkSymbolTables(op, /*allSymUsesVisible=*/!op->getBlock(),
                                walkFn);
}

void ForwardDataFlowSolver::visitOperation(Operation *op) {
  // Collect all of the lattice elements feeding into this operation. If any are
  // not yet resolved, bail out and wait for them to resolve.
  SmallVector<AbstractLatticeElement *, 8> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    AbstractLatticeElement *operandLattice =
        analysis.lookupLatticeElement(operand);
    if (!operandLattice || operandLattice->isUninitialized())
      return;
    operandLattices.push_back(operandLattice);
  }

  // If this is a terminator operation, process any control flow lattice state.
  if (op->hasTrait<OpTrait::IsTerminator>())
    visitTerminatorOperation(op, operandLattices);

  // Process call operations. The call visitor processes result values, so we
  // can exit afterwards.
  if (CallOpInterface call = dyn_cast<CallOpInterface>(op))
    return visitCallOperation(call);

  // Process callable operations. These are specially handled region operations
  // that track dataflow via calls.
  if (isa<CallableOpInterface>(op)) {
    // If this callable has a tracked lattice state, it will be visited by calls
    // that reference it instead. This way, we don't assume that it is
    // executable unless there is a proper reference to it.
    if (callableLatticeState.count(op))
      return;
    return visitCallableOperation(op);
  }

  // Process region holding operations.
  if (op->getNumRegions()) {
    // Check to see if we can reason about the internal control flow of this
    // region operation.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(op))
      return visitRegionBranchOperation(branch, operandLattices);

    // If we can't, conservatively mark all regions as executable.
    // TODO: Let the `visitOperation` method decide how to propagate
    // information to the block arguments.
    for (Region &region : op->getRegions())
      markEntryBlockExecutable(&region, /*markPessimisticFixpoint=*/true);
  }

  // If this op produces no results, it can't produce any constants.
  if (op->getNumResults() == 0)
    return;

  // If all of the results of this operation are already resolved, bail out
  // early.
  auto isAtFixpointFn = [&](Value value) { return isAtFixpoint(value); };
  if (llvm::all_of(op->getResults(), isAtFixpointFn))
    return;

  // Visit the current operation.
  if (analysis.visitOperation(op, operandLattices) == ChangeResult::Change)
    opWorklist.push_back(op);

  // `visitOperation` is required to define all of the result lattices.
  assert(llvm::none_of(
             op->getResults(),
             [&](Value value) {
               return analysis.getLatticeElement(value).isUninitialized();
             }) &&
         "expected `visitOperation` to define all result lattices");
}

void ForwardDataFlowSolver::visitCallableOperation(Operation *op) {
  // Mark the regions as executable. If we aren't tracking lattice state for
  // this callable, mark all of the region arguments as having reached a
  // fixpoint.
  bool isTrackingLatticeState = callableLatticeState.count(op);
  for (Region &region : op->getRegions())
    markEntryBlockExecutable(&region, !isTrackingLatticeState);

  // TODO: Add support for non-symbol callables when necessary. If the callable
  // has non-call uses we would mark as having reached pessimistic fixpoint,
  // otherwise allow for propagating the return values out.
  markAllPessimisticFixpoint(op, op->getResults());
}

void ForwardDataFlowSolver::visitCallOperation(CallOpInterface op) {
  ResultRange callResults = op->getResults();

  // Resolve the callable operation for this call.
  Operation *callableOp = nullptr;
  if (Value callableValue = op.getCallableForCallee().dyn_cast<Value>())
    callableOp = callableValue.getDefiningOp();
  else
    callableOp = callToSymbolCallable.lookup(op);

  // The callable of this call can't be resolved, mark any results overdefined.
  if (!callableOp)
    return markAllPessimisticFixpoint(op, callResults);

  // If this callable is tracking state, merge the argument operands with the
  // arguments of the callable.
  auto callableLatticeIt = callableLatticeState.find(callableOp);
  if (callableLatticeIt == callableLatticeState.end())
    return markAllPessimisticFixpoint(op, callResults);

  OperandRange callOperands = op.getArgOperands();
  auto callableArgs = callableLatticeIt->second.getCallableArguments();
  for (auto it : llvm::zip(callOperands, callableArgs)) {
    BlockArgument callableArg = std::get<1>(it);
    AbstractLatticeElement &argValue = analysis.getLatticeElement(callableArg);
    AbstractLatticeElement &operandValue =
        analysis.getLatticeElement(std::get<0>(it));
    if (argValue.join(operandValue) == ChangeResult::Change)
      visitUsers(callableArg);
  }

  // Visit the callable.
  visitCallableOperation(callableOp);

  // Merge in the lattice state for the callable results as well.
  auto callableResults = callableLatticeIt->second.getResultLatticeElements();
  for (auto it : llvm::zip(callResults, callableResults))
    join(/*owner=*/op,
         /*to=*/analysis.getLatticeElement(std::get<0>(it)),
         /*from=*/std::get<1>(it));
}

void ForwardDataFlowSolver::visitRegionBranchOperation(
    RegionBranchOpInterface branch,
    ArrayRef<AbstractLatticeElement *> operandLattices) {
  // Check to see which regions are executable.
  SmallVector<RegionSuccessor, 1> successors;
  analysis.getSuccessorsForOperands(branch, /*sourceIndex=*/llvm::None,
                                    operandLattices, successors);

  // If the interface identified that no region will be executed. Mark
  // any results of this operation as overdefined, as we can't reason about
  // them.
  // TODO: If we had an interface to detect pass through operands, we could
  // resolve some results based on the lattice state of the operands. We could
  // also allow for the parent operation to have itself as a region successor.
  if (successors.empty())
    return markAllPessimisticFixpoint(branch, branch->getResults());
  return visitRegionSuccessors(
      branch, successors, [&](Optional<unsigned> index) {
        assert(index && "expected valid region index");
        return branch.getSuccessorEntryOperands(*index);
      });
}

void ForwardDataFlowSolver::visitRegionSuccessors(
    Operation *parentOp, ArrayRef<RegionSuccessor> regionSuccessors,
    function_ref<OperandRange(Optional<unsigned>)> getInputsForRegion) {
  for (const RegionSuccessor &it : regionSuccessors) {
    Region *region = it.getSuccessor();
    ValueRange succArgs = it.getSuccessorInputs();

    // Check to see if this is the parent operation.
    if (!region) {
      ResultRange results = parentOp->getResults();
      if (llvm::all_of(results, [&](Value res) { return isAtFixpoint(res); }))
        continue;

      // Mark the results outside of the input range as having reached the
      // pessimistic fixpoint.
      // TODO: This isn't exactly ideal. There may be situations in which a
      // region operation can provide information for certain results that
      // aren't part of the control flow.
      if (succArgs.size() != results.size()) {
        opWorklist.push_back(parentOp);
        if (succArgs.empty()) {
          markAllPessimisticFixpoint(results);
          continue;
        }

        unsigned firstResIdx = succArgs[0].cast<OpResult>().getResultNumber();
        markAllPessimisticFixpoint(results.take_front(firstResIdx));
        markAllPessimisticFixpoint(
            results.drop_front(firstResIdx + succArgs.size()));
      }

      // Update the lattice for any operation results.
      OperandRange operands = getInputsForRegion(/*index=*/llvm::None);
      for (auto it : llvm::zip(succArgs, operands))
        join(parentOp, analysis.getLatticeElement(std::get<0>(it)),
             analysis.getLatticeElement(std::get<1>(it)));
      continue;
    }
    assert(!region->empty() && "expected region to be non-empty");
    Block *entryBlock = &region->front();
    markBlockExecutable(entryBlock);

    // If all of the arguments have already reached a fixpoint, the arguments
    // have already been fully resolved.
    Block::BlockArgListType arguments = entryBlock->getArguments();
    if (llvm::all_of(arguments, [&](Value arg) { return isAtFixpoint(arg); }))
      continue;

    // Mark any arguments that do not receive inputs as having reached a
    // pessimistic fixpoint, we won't be able to discern if they are constant.
    // TODO: This isn't exactly ideal. There may be situations in which a
    // region operation can provide information for certain results that
    // aren't part of the control flow.
    if (succArgs.size() != arguments.size()) {
      if (succArgs.empty()) {
        markAllPessimisticFixpoint(arguments);
        continue;
      }

      unsigned firstArgIdx = succArgs[0].cast<BlockArgument>().getArgNumber();
      markAllPessimisticFixpointAndVisitUsers(
          arguments.take_front(firstArgIdx));
      markAllPessimisticFixpointAndVisitUsers(
          arguments.drop_front(firstArgIdx + succArgs.size()));
    }

    // Update the lattice of arguments that have inputs from the predecessor.
    OperandRange succOperands = getInputsForRegion(region->getRegionNumber());
    for (auto it : llvm::zip(succArgs, succOperands)) {
      AbstractLatticeElement &argValue =
          analysis.getLatticeElement(std::get<0>(it));
      AbstractLatticeElement &operandValue =
          analysis.getLatticeElement(std::get<1>(it));
      if (argValue.join(operandValue) == ChangeResult::Change)
        visitUsers(std::get<0>(it));
    }
  }
}

void ForwardDataFlowSolver::visitTerminatorOperation(
    Operation *op, ArrayRef<AbstractLatticeElement *> operandLattices) {
  // If this operation has no successors, we treat it as an exiting terminator.
  if (op->getNumSuccessors() == 0) {
    Region *parentRegion = op->getParentRegion();
    Operation *parentOp = parentRegion->getParentOp();

    // Check to see if this is a terminator for a callable region.
    if (isa<CallableOpInterface>(parentOp))
      return visitCallableTerminatorOperation(parentOp, op, operandLattices);

    // Otherwise, check to see if the parent tracks region control flow.
    auto regionInterface = dyn_cast<RegionBranchOpInterface>(parentOp);
    if (!regionInterface || !isBlockExecutable(parentOp->getBlock()))
      return;

    // Query the set of successors of the current region using the current
    // optimistic lattice state.
    SmallVector<RegionSuccessor, 1> regionSuccessors;
    analysis.getSuccessorsForOperands(regionInterface,
                                      parentRegion->getRegionNumber(),
                                      operandLattices, regionSuccessors);
    if (regionSuccessors.empty())
      return;

    // Try to get "region-like" successor operands if possible in order to
    // propagate the operand states to the successors.
    if (isRegionReturnLike(op)) {
      return visitRegionSuccessors(
          parentOp, regionSuccessors, [&](Optional<unsigned> regionIndex) {
            // Determine the individual region successor operands for the given
            // region index (if any).
            return *getRegionBranchSuccessorOperands(op, regionIndex);
          });
    }

    // If this terminator is not "region-like", conservatively mark all of the
    // successor values as having reached the pessimistic fixpoint.
    for (auto &it : regionSuccessors) {
      // If the successor is a region, mark the entry block as executable so
      // that we visit operations defined within. If the successor is the
      // parent operation, we simply mark the control flow results as having
      // reached the pessimistic state.
      if (Region *region = it.getSuccessor())
        markEntryBlockExecutable(region, /*markPessimisticFixpoint=*/true);
      else
        markAllPessimisticFixpointAndVisitUsers(it.getSuccessorInputs());
    }
  }

  // Try to resolve to a specific set of successors with the current optimistic
  // lattice state.
  Block *block = op->getBlock();
  if (auto branch = dyn_cast<BranchOpInterface>(op)) {
    SmallVector<Block *> successors;
    if (succeeded(analysis.getSuccessorsForOperands(branch, operandLattices,
                                                    successors))) {
      for (Block *succ : successors)
        markEdgeExecutable(block, succ);
      return;
    }
  }

  // Otherwise, conservatively treat all edges as executable.
  for (Block *succ : op->getSuccessors())
    markEdgeExecutable(block, succ);
}

void ForwardDataFlowSolver::visitCallableTerminatorOperation(
    Operation *callable, Operation *terminator,
    ArrayRef<AbstractLatticeElement *> operandLattices) {
  // If there are no exiting values, we have nothing to track.
  if (terminator->getNumOperands() == 0)
    return;

  // If this callable isn't tracking any lattice state there is nothing to do.
  auto latticeIt = callableLatticeState.find(callable);
  if (latticeIt == callableLatticeState.end())
    return;
  assert(callable->getNumResults() == 0 && "expected symbol callable");

  // If this terminator is not "return-like", conservatively mark all of the
  // call-site results as having reached the pessimistic fixpoint.
  auto callableResultLattices = latticeIt->second.getResultLatticeElements();
  if (!terminator->hasTrait<OpTrait::ReturnLike>()) {
    for (auto &it : callableResultLattices)
      it.markPessimisticFixpoint();
    for (Operation *call : latticeIt->second.getSymbolCalls())
      markAllPessimisticFixpoint(call, call->getResults());
    return;
  }

  // Merge the lattice state for terminator operands into the results.
  ChangeResult result = ChangeResult::NoChange;
  for (auto it : llvm::zip(operandLattices, callableResultLattices))
    result |= std::get<1>(it).join(*std::get<0>(it));
  if (result == ChangeResult::NoChange)
    return;

  // If any of the result lattices changed, update the callers.
  for (Operation *call : latticeIt->second.getSymbolCalls())
    for (auto it : llvm::zip(call->getResults(), callableResultLattices))
      join(call, analysis.getLatticeElement(std::get<0>(it)), std::get<1>(it));
}

void ForwardDataFlowSolver::visitBlock(Block *block) {
  // If the block is not the entry block we need to compute the lattice state
  // for the block arguments. Entry block argument lattices are computed
  // elsewhere, such as when visiting the parent operation.
  if (!block->isEntryBlock()) {
    for (int i : llvm::seq<int>(0, block->getNumArguments()))
      visitBlockArgument(block, i);
  }

  // Visit all of the operations within the block.
  for (Operation &op : *block)
    visitOperation(&op);
}

void ForwardDataFlowSolver::visitBlockArgument(Block *block, int i) {
  BlockArgument arg = block->getArgument(i);
  AbstractLatticeElement &argLattice = analysis.getLatticeElement(arg);
  if (argLattice.isAtFixpoint())
    return;

  ChangeResult updatedLattice = ChangeResult::NoChange;
  for (auto it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
    Block *pred = *it;

    // We only care about this predecessor if it is going to execute.
    if (!isEdgeExecutable(pred, block))
      continue;

    // Try to get the operand forwarded by the predecessor. If we can't reason
    // about the terminator of the predecessor, mark as having reached a
    // fixpoint.
    Optional<OperandRange> branchOperands;
    if (auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator()))
      branchOperands = branch.getSuccessorOperands(it.getSuccessorIndex());
    if (!branchOperands) {
      updatedLattice |= argLattice.markPessimisticFixpoint();
      break;
    }

    // If the operand hasn't been resolved, it is uninitialized and can merge
    // with anything.
    AbstractLatticeElement *operandLattice =
        analysis.lookupLatticeElement((*branchOperands)[i]);
    if (!operandLattice)
      continue;

    // Otherwise, join the operand lattice into the argument lattice.
    updatedLattice |= argLattice.join(*operandLattice);
    if (argLattice.isAtFixpoint())
      break;
  }

  // If the lattice changed, visit users of the argument.
  if (updatedLattice == ChangeResult::Change)
    visitUsers(arg);
}

ChangeResult
ForwardDataFlowSolver::markEntryBlockExecutable(Region *region,
                                                bool markPessimisticFixpoint) {
  if (!region->empty()) {
    if (markPessimisticFixpoint)
      markAllPessimisticFixpoint(region->front().getArguments());
    return markBlockExecutable(&region->front());
  }
  return ChangeResult::NoChange;
}

ChangeResult ForwardDataFlowSolver::markBlockExecutable(Block *block) {
  bool marked = executableBlocks.insert(block).second;
  if (marked)
    blockWorklist.push_back(block);
  return marked ? ChangeResult::Change : ChangeResult::NoChange;
}

bool ForwardDataFlowSolver::isBlockExecutable(Block *block) const {
  return executableBlocks.count(block);
}

void ForwardDataFlowSolver::markEdgeExecutable(Block *from, Block *to) {
  if (!executableEdges.insert(std::make_pair(from, to)).second)
    return;

  // Mark the destination as executable, and reprocess its arguments if it was
  // already executable.
  if (markBlockExecutable(to) == ChangeResult::NoChange) {
    for (int i : llvm::seq<int>(0, to->getNumArguments()))
      visitBlockArgument(to, i);
  }
}

bool ForwardDataFlowSolver::isEdgeExecutable(Block *from, Block *to) const {
  return executableEdges.count(std::make_pair(from, to));
}

void ForwardDataFlowSolver::markPessimisticFixpoint(Value value) {
  analysis.getLatticeElement(value).markPessimisticFixpoint();
}

bool ForwardDataFlowSolver::isAtFixpoint(Value value) const {
  if (auto *lattice = analysis.lookupLatticeElement(value))
    return lattice->isAtFixpoint();
  return false;
}

void ForwardDataFlowSolver::join(Operation *owner, AbstractLatticeElement &to,
                                 const AbstractLatticeElement &from) {
  if (to.join(from) == ChangeResult::Change)
    opWorklist.push_back(owner);
}

//===----------------------------------------------------------------------===//
// AbstractLatticeElement
//===----------------------------------------------------------------------===//

AbstractLatticeElement::~AbstractLatticeElement() {}

//===----------------------------------------------------------------------===//
// ForwardDataFlowAnalysisBase
//===----------------------------------------------------------------------===//

ForwardDataFlowAnalysisBase::~ForwardDataFlowAnalysisBase() {}

AbstractLatticeElement &
ForwardDataFlowAnalysisBase::getLatticeElement(Value value) {
  AbstractLatticeElement *&latticeValue = latticeValues[value];
  if (!latticeValue)
    latticeValue = createLatticeElement(value);
  return *latticeValue;
}

AbstractLatticeElement *
ForwardDataFlowAnalysisBase::lookupLatticeElement(Value value) {
  return latticeValues.lookup(value);
}

void ForwardDataFlowAnalysisBase::run(Operation *topLevelOp) {
  // Run the main dataflow solver.
  ForwardDataFlowSolver solver(*this, topLevelOp);
  solver.solve();

  // Any values that are still uninitialized now go to a pessimistic fixpoint,
  // otherwise we assume an optimistic fixpoint has been reached.
  for (auto &it : latticeValues)
    if (it.second->isUninitialized())
      it.second->markPessimisticFixpoint();
    else
      it.second->markOptimisticFixpoint();
}
