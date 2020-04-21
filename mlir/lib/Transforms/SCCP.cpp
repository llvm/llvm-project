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
// optimisitic dataflow analysis that assumes that all values are constant until
// proven otherwise.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
/// This class represents a single lattice value. A lattive value corresponds to
/// the various different states that a value in the SCCP dataflow anaylsis can
/// take. See 'Kind' below for more details on the different states a value can
/// take.
class LatticeValue {
  enum Kind {
    /// A value with a yet to be determined value. This state may be changed to
    /// anything.
    Unknown,

    /// A value that is known to be a constant. This state may be changed to
    /// overdefined.
    Constant,

    /// A value that cannot statically be determined to be a constant. This
    /// state cannot be changed.
    Overdefined
  };

public:
  /// Initialize a lattice value with "Unknown".
  LatticeValue()
      : constantAndTag(nullptr, Kind::Unknown), constantDialect(nullptr) {}
  /// Initialize a lattice value with a constant.
  LatticeValue(Attribute attr, Dialect *dialect)
      : constantAndTag(attr, Kind::Constant), constantDialect(dialect) {}

  /// Returns true if this lattice value is unknown.
  bool isUnknown() const { return constantAndTag.getInt() == Kind::Unknown; }

  /// Mark the lattice value as overdefined.
  void markOverdefined() {
    constantAndTag.setPointerAndInt(nullptr, Kind::Overdefined);
    constantDialect = nullptr;
  }

  /// Returns true if the lattice is overdefined.
  bool isOverdefined() const {
    return constantAndTag.getInt() == Kind::Overdefined;
  }

  /// Mark the lattice value as constant.
  void markConstant(Attribute value, Dialect *dialect) {
    constantAndTag.setPointerAndInt(value, Kind::Constant);
    constantDialect = dialect;
  }

  /// If this lattice is constant, return the constant. Returns nullptr
  /// otherwise.
  Attribute getConstant() const { return constantAndTag.getPointer(); }

  /// If this lattice is constant, return the dialect to use when materializing
  /// the constant.
  Dialect *getConstantDialect() const {
    assert(getConstant() && "expected valid constant");
    return constantDialect;
  }

  /// Merge in the value of the 'rhs' lattice into this one. Returns true if the
  /// lattice value changed.
  bool meet(const LatticeValue &rhs) {
    // If we are already overdefined, or rhs is unknown, there is nothing to do.
    if (isOverdefined() || rhs.isUnknown())
      return false;
    // If we are unknown, just take the value of rhs.
    if (isUnknown()) {
      constantAndTag = rhs.constantAndTag;
      constantDialect = rhs.constantDialect;
      return true;
    }

    // Otherwise, if this value doesn't match rhs go straight to overdefined.
    if (constantAndTag != rhs.constantAndTag) {
      markOverdefined();
      return true;
    }
    return false;
  }

private:
  /// The attribute value if this is a constant and the tag for the element
  /// kind.
  llvm::PointerIntPair<Attribute, 2, Kind> constantAndTag;

  /// The dialect the constant originated from. This is only valid if the
  /// lattice is a constant. This is not used as part of the key, and is only
  /// needed to materialize the held constant if necessary.
  Dialect *constantDialect;
};

/// This class represents the solver for the SCCP analysis. This class acts as
/// the propagation engine for computing which values form constants.
class SCCPSolver {
public:
  /// Initialize the solver with a given set of regions.
  SCCPSolver(MutableArrayRef<Region> regions);

  /// Run the solver until it converges.
  void solve();

  /// Rewrite the given regions using the computing analysis. This replaces the
  /// uses of all values that have been computed to be constant, and erases as
  /// many newly dead operations.
  void rewrite(MLIRContext *context, MutableArrayRef<Region> regions);

private:
  /// Replace the given value with a constant if the corresponding lattice
  /// represents a constant. Returns success if the value was replaced, failure
  /// otherwise.
  LogicalResult replaceWithConstant(OpBuilder &builder, OperationFolder &folder,
                                    Value value);

  /// Visit the users of the given IR that reside within executable blocks.
  template <typename T>
  void visitUsers(T &value) {
    for (Operation *user : value.getUsers())
      if (isBlockExecutable(user->getBlock()))
        visitOperation(user);
  }

  /// Visit the given operation and compute any necessary lattice state.
  void visitOperation(Operation *op);

  /// Visit the given operation, which defines regions, and compute any
  /// necessary lattice state. This also resolves the lattice state of both the
  /// operation results and any nested regions.
  void visitRegionOperation(Operation *op,
                            ArrayRef<Attribute> constantOperands);

  /// Visit the given set of region successors, computing any necessary lattice
  /// state. The provided function returns the input operands to the region at
  /// the given index. If the index is 'None', the input operands correspond to
  /// the parent operation results.
  void visitRegionSuccessors(
      Operation *parentOp, ArrayRef<RegionSuccessor> regionSuccessors,
      function_ref<OperandRange(Optional<unsigned>)> getInputsForRegion);

  /// Visit the given terminator operation and compute any necessary lattice
  /// state.
  void visitTerminatorOperation(Operation *op,
                                ArrayRef<Attribute> constantOperands);

  /// Visit the given block and compute any necessary lattice state.
  void visitBlock(Block *block);

  /// Visit argument #'i' of the given block and compute any necessary lattice
  /// state.
  void visitBlockArgument(Block *block, int i);

  /// Mark the given block as executable. Returns false if the block was already
  /// marked executable.
  bool markBlockExecutable(Block *block);

  /// Returns true if the given block is executable.
  bool isBlockExecutable(Block *block) const;

  /// Mark the edge between 'from' and 'to' as executable.
  void markEdgeExecutable(Block *from, Block *to);

  /// Return true if the edge between 'from' and 'to' is executable.
  bool isEdgeExecutable(Block *from, Block *to) const;

  /// Mark the given value as overdefined. This means that we cannot refine a
  /// specific constant for this value.
  void markOverdefined(Value value);

  /// Mark all of the given values as overdefined.
  template <typename ValuesT>
  void markAllOverdefined(ValuesT values) {
    for (auto value : values)
      markOverdefined(value);
  }
  template <typename ValuesT>
  void markAllOverdefined(Operation *op, ValuesT values) {
    markAllOverdefined(values);
    opWorklist.push_back(op);
  }
  template <typename ValuesT>
  void markAllOverdefinedAndVisitUsers(ValuesT values) {
    for (auto value : values) {
      auto &lattice = latticeValues[value];
      if (!lattice.isOverdefined()) {
        lattice.markOverdefined();
        visitUsers(value);
      }
    }
  }

  /// Returns true if the given value was marked as overdefined.
  bool isOverdefined(Value value) const;

  /// Merge in the given lattice 'from' into the lattice 'to'. 'owner'
  /// corresponds to the parent operation of 'to'.
  void meet(Operation *owner, LatticeValue &to, const LatticeValue &from);

  /// The lattice for each SSA value.
  DenseMap<Value, LatticeValue> latticeValues;

  /// The set of blocks that are known to execute, or are intrinsically live.
  SmallPtrSet<Block *, 16> executableBlocks;

  /// The set of control flow edges that are known to execute.
  DenseSet<std::pair<Block *, Block *>> executableEdges;

  /// A worklist containing blocks that need to be processed.
  SmallVector<Block *, 64> blockWorklist;

  /// A worklist of operations that need to be processed.
  SmallVector<Operation *, 64> opWorklist;
};
} // end anonymous namespace

SCCPSolver::SCCPSolver(MutableArrayRef<Region> regions) {
  for (Region &region : regions) {
    if (region.empty())
      continue;
    Block *entryBlock = &region.front();

    // Mark the entry block as executable.
    markBlockExecutable(entryBlock);

    // The values passed to these regions are invisible, so mark any arguments
    // as overdefined.
    markAllOverdefined(entryBlock->getArguments());
  }
}

void SCCPSolver::solve() {
  while (!blockWorklist.empty() || !opWorklist.empty()) {
    // Process any operations in the op worklist.
    while (!opWorklist.empty())
      visitUsers(*opWorklist.pop_back_val());

    // Process any blocks in the block worklist.
    while (!blockWorklist.empty())
      visitBlock(blockWorklist.pop_back_val());
  }
}

void SCCPSolver::rewrite(MLIRContext *context,
                         MutableArrayRef<Region> initialRegions) {
  SmallVector<Block *, 8> worklist;
  auto addToWorklist = [&](MutableArrayRef<Region> regions) {
    for (Region &region : regions)
      for (Block &block : region)
        if (isBlockExecutable(&block))
          worklist.push_back(&block);
  };

  // An operation folder used to create and unique constants.
  OperationFolder folder(context);
  OpBuilder builder(context);

  addToWorklist(initialRegions);
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();

    // Replace any block arguments with constants.
    builder.setInsertionPointToStart(block);
    for (BlockArgument arg : block->getArguments())
      replaceWithConstant(builder, folder, arg);

    for (Operation &op : llvm::make_early_inc_range(*block)) {
      builder.setInsertionPoint(&op);

      // Replace any result with constants.
      bool replacedAll = op.getNumResults() != 0;
      for (Value res : op.getResults())
        replacedAll &= succeeded(replaceWithConstant(builder, folder, res));

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
  }
}

LogicalResult SCCPSolver::replaceWithConstant(OpBuilder &builder,
                                              OperationFolder &folder,
                                              Value value) {
  auto it = latticeValues.find(value);
  auto attr = it == latticeValues.end() ? nullptr : it->second.getConstant();
  if (!attr)
    return failure();

  // Attempt to materialize a constant for the given value.
  Dialect *dialect = it->second.getConstantDialect();
  Value constant = folder.getOrCreateConstant(builder, dialect, attr,
                                              value.getType(), value.getLoc());
  if (!constant)
    return failure();

  value.replaceAllUsesWith(constant);
  latticeValues.erase(it);
  return success();
}

void SCCPSolver::visitOperation(Operation *op) {
  // Collect all of the constant operands feeding into this operation. If any
  // are not ready to be resolved, bail out and wait for them to resolve.
  SmallVector<Attribute, 8> operandConstants;
  operandConstants.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    // Make sure all of the operands are resolved first.
    auto &operandLattice = latticeValues[operand];
    if (operandLattice.isUnknown())
      return;
    operandConstants.push_back(operandLattice.getConstant());
  }

  // If this is a terminator operation, process any control flow lattice state.
  if (op->isKnownTerminator())
    visitTerminatorOperation(op, operandConstants);

  // Process region holding operations. The region visitor processes result
  // values, so we can exit afterwards.
  if (op->getNumRegions())
    return visitRegionOperation(op, operandConstants);

  // If this op produces no results, it can't produce any constants.
  if (op->getNumResults() == 0)
    return;

  // If all of the results of this operation are already overdefined, bail out
  // early.
  auto isOverdefinedFn = [&](Value value) { return isOverdefined(value); };
  if (llvm::all_of(op->getResults(), isOverdefinedFn))
    return;

  // Save the original operands and attributes just in case the operation folds
  // in-place. The constant passed in may not correspond to the real runtime
  // value, so in-place updates are not allowed.
  SmallVector<Value, 8> originalOperands(op->getOperands());
  NamedAttributeList originalAttrs = op->getAttrList();

  // Simulate the result of folding this operation to a constant. If folding
  // fails or was an in-place fold, mark the results as overdefined.
  SmallVector<OpFoldResult, 8> foldResults;
  foldResults.reserve(op->getNumResults());
  if (failed(op->fold(operandConstants, foldResults)))
    return markAllOverdefined(op, op->getResults());

  // If the folding was in-place, mark the results as overdefined and reset the
  // operation. We don't allow in-place folds as the desire here is for
  // simulated execution, and not general folding.
  if (foldResults.empty()) {
    op->setOperands(originalOperands);
    op->setAttrs(originalAttrs);
    return markAllOverdefined(op, op->getResults());
  }

  // Merge the fold results into the lattice for this operation.
  assert(foldResults.size() == op->getNumResults() && "invalid result size");
  Dialect *opDialect = op->getDialect();
  for (unsigned i = 0, e = foldResults.size(); i != e; ++i) {
    LatticeValue &resultLattice = latticeValues[op->getResult(i)];

    // Merge in the result of the fold, either a constant or a value.
    OpFoldResult foldResult = foldResults[i];
    if (Attribute foldAttr = foldResult.dyn_cast<Attribute>())
      meet(op, resultLattice, LatticeValue(foldAttr, opDialect));
    else
      meet(op, resultLattice, latticeValues[foldResult.get<Value>()]);
  }
}

void SCCPSolver::visitRegionOperation(Operation *op,
                                      ArrayRef<Attribute> constantOperands) {
  // Check to see if we can reason about the internal control flow of this
  // region operation.
  auto regionInterface = dyn_cast<RegionBranchOpInterface>(op);
  if (!regionInterface) {
    // If we can't, conservatively mark all regions as executable.
    for (Region &region : op->getRegions()) {
      if (region.empty())
        continue;
      Block *entryBlock = &region.front();
      markBlockExecutable(entryBlock);
      markAllOverdefined(entryBlock->getArguments());
    }

    // Don't try to simulate the results of a region operation as we can't
    // guarantee that folding will be out-of-place. We don't allow in-place
    // folds as the desire here is for simulated execution, and not general
    // folding.
    return markAllOverdefined(op, op->getResults());
  }

  // Check to see which regions are executable.
  SmallVector<RegionSuccessor, 1> successors;
  regionInterface.getSuccessorRegions(/*index=*/llvm::None, constantOperands,
                                      successors);

  // If the interface identified that no region will be executed. Mark
  // any results of this operation as overdefined, as we can't reason about
  // them.
  // TODO: If we had an interface to detect pass through operands, we could
  // resolve some results based on the lattice state of the operands. We could
  // also allow for the parent operation to have itself as a region successor.
  if (successors.empty())
    return markAllOverdefined(op, op->getResults());
  return visitRegionSuccessors(op, successors, [&](Optional<unsigned> index) {
    assert(index && "expected valid region index");
    return regionInterface.getSuccessorEntryOperands(*index);
  });
}

void SCCPSolver::visitRegionSuccessors(
    Operation *parentOp, ArrayRef<RegionSuccessor> regionSuccessors,
    function_ref<OperandRange(Optional<unsigned>)> getInputsForRegion) {
  for (const RegionSuccessor &it : regionSuccessors) {
    Region *region = it.getSuccessor();
    ValueRange succArgs = it.getSuccessorInputs();

    // Check to see if this is the parent operation.
    if (!region) {
      ResultRange results = parentOp->getResults();
      if (llvm::all_of(results, [&](Value res) { return isOverdefined(res); }))
        continue;

      // Mark the results outside of the input range as overdefined.
      if (succArgs.size() != results.size()) {
        opWorklist.push_back(parentOp);
        if (succArgs.empty())
          return markAllOverdefined(results);

        unsigned firstResIdx = succArgs[0].cast<OpResult>().getResultNumber();
        markAllOverdefined(results.take_front(firstResIdx));
        markAllOverdefined(results.drop_front(firstResIdx + succArgs.size()));
      }

      // Update the lattice for any operation results.
      OperandRange operands = getInputsForRegion(/*index=*/llvm::None);
      for (auto it : llvm::zip(succArgs, operands))
        meet(parentOp, latticeValues[std::get<0>(it)],
             latticeValues[std::get<1>(it)]);
      return;
    }
    assert(!region->empty() && "expected region to be non-empty");
    Block *entryBlock = &region->front();
    markBlockExecutable(entryBlock);

    // If all of the arguments are already overdefined, the arguments have
    // already been fully resolved.
    auto arguments = entryBlock->getArguments();
    if (llvm::all_of(arguments, [&](Value arg) { return isOverdefined(arg); }))
      continue;

    // Mark any arguments that do not receive inputs as overdefined, we won't be
    // able to discern if they are constant.
    if (succArgs.size() != arguments.size()) {
      if (succArgs.empty()) {
        markAllOverdefined(arguments);
        continue;
      }

      unsigned firstArgIdx = succArgs[0].cast<BlockArgument>().getArgNumber();
      markAllOverdefinedAndVisitUsers(arguments.take_front(firstArgIdx));
      markAllOverdefinedAndVisitUsers(
          arguments.drop_front(firstArgIdx + succArgs.size()));
    }

    // Update the lattice for arguments that have inputs from the predecessor.
    OperandRange succOperands = getInputsForRegion(region->getRegionNumber());
    for (auto it : llvm::zip(succArgs, succOperands)) {
      LatticeValue &argLattice = latticeValues[std::get<0>(it)];
      if (argLattice.meet(latticeValues[std::get<1>(it)]))
        visitUsers(std::get<0>(it));
    }
  }
}

void SCCPSolver::visitTerminatorOperation(
    Operation *op, ArrayRef<Attribute> constantOperands) {
  // If this operation has no successors, we treat it as an exiting terminator.
  if (op->getNumSuccessors() == 0) {
    // Check to see if the parent tracks region control flow.
    Region *parentRegion = op->getParentRegion();
    Operation *parentOp = parentRegion->getParentOp();
    auto regionInterface = dyn_cast<RegionBranchOpInterface>(parentOp);
    if (!regionInterface || !isBlockExecutable(parentOp->getBlock()))
      return;

    // Query the set of successors from the current region.
    SmallVector<RegionSuccessor, 1> regionSuccessors;
    regionInterface.getSuccessorRegions(parentRegion->getRegionNumber(),
                                        constantOperands, regionSuccessors);
    if (regionSuccessors.empty())
      return;

    // If this terminator is not "region-like", conservatively mark all of the
    // successor values as overdefined.
    if (!op->hasTrait<OpTrait::ReturnLike>()) {
      for (auto &it : regionSuccessors)
        markAllOverdefinedAndVisitUsers(it.getSuccessorInputs());
      return;
    }

    // Otherwise, propagate the operand lattice states to each of the
    // successors.
    OperandRange operands = op->getOperands();
    return visitRegionSuccessors(parentOp, regionSuccessors,
                                 [&](Optional<unsigned>) { return operands; });
  }

  // Try to resolve to a specific successor with the constant operands.
  if (auto branch = dyn_cast<BranchOpInterface>(op)) {
    if (Block *singleSucc = branch.getSuccessorForOperands(constantOperands)) {
      markEdgeExecutable(op->getBlock(), singleSucc);
      return;
    }
  }

  // Otherwise, conservatively treat all edges as executable.
  Block *block = op->getBlock();
  for (Block *succ : op->getSuccessors())
    markEdgeExecutable(block, succ);
}

void SCCPSolver::visitBlock(Block *block) {
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

void SCCPSolver::visitBlockArgument(Block *block, int i) {
  BlockArgument arg = block->getArgument(i);
  LatticeValue &argLattice = latticeValues[arg];
  if (argLattice.isOverdefined())
    return;

  bool updatedLattice = false;
  for (auto it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
    Block *pred = *it;

    // We only care about this predecessor if it is going to execute.
    if (!isEdgeExecutable(pred, block))
      continue;

    // Try to get the operand forwarded by the predecessor. If we can't reason
    // about the terminator of the predecessor, mark overdefined.
    Optional<OperandRange> branchOperands;
    if (auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator()))
      branchOperands = branch.getSuccessorOperands(it.getSuccessorIndex());
    if (!branchOperands) {
      updatedLattice = true;
      argLattice.markOverdefined();
      break;
    }

    // If the operand hasn't been resolved, it is unknown which can merge with
    // anything.
    auto operandLattice = latticeValues.find((*branchOperands)[i]);
    if (operandLattice == latticeValues.end())
      continue;

    // Otherwise, meet the two lattice values.
    updatedLattice |= argLattice.meet(operandLattice->second);
    if (argLattice.isOverdefined())
      break;
  }

  // If the lattice was updated, visit any executable users of the argument.
  if (updatedLattice)
    visitUsers(arg);
}

bool SCCPSolver::markBlockExecutable(Block *block) {
  bool marked = executableBlocks.insert(block).second;
  if (marked)
    blockWorklist.push_back(block);
  return marked;
}

bool SCCPSolver::isBlockExecutable(Block *block) const {
  return executableBlocks.count(block);
}

void SCCPSolver::markEdgeExecutable(Block *from, Block *to) {
  if (!executableEdges.insert(std::make_pair(from, to)).second)
    return;
  // Mark the destination as executable, and reprocess its arguments if it was
  // already executable.
  if (!markBlockExecutable(to)) {
    for (int i : llvm::seq<int>(0, to->getNumArguments()))
      visitBlockArgument(to, i);
  }
}

bool SCCPSolver::isEdgeExecutable(Block *from, Block *to) const {
  return executableEdges.count(std::make_pair(from, to));
}

void SCCPSolver::markOverdefined(Value value) {
  latticeValues[value].markOverdefined();
}

bool SCCPSolver::isOverdefined(Value value) const {
  auto it = latticeValues.find(value);
  return it != latticeValues.end() && it->second.isOverdefined();
}

void SCCPSolver::meet(Operation *owner, LatticeValue &to,
                      const LatticeValue &from) {
  if (to.meet(from))
    opWorklist.push_back(owner);
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

  // Solve for SCCP constraints within nested regions.
  SCCPSolver solver(op->getRegions());
  solver.solve();

  // Cleanup any operations using the solver analysis.
  solver.rewrite(&getContext(), op->getRegions());
}

std::unique_ptr<Pass> mlir::createSCCPPass() {
  return std::make_unique<SCCP>();
}
