//======- BufferViewFlowAnalysis.cpp - Buffer alias analysis -*- C++ -*-======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"

#include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetOperations.h"

using namespace mlir;
using namespace mlir::bufferization;

//===----------------------------------------------------------------------===//
// BufferViewFlowAnalysis
//===----------------------------------------------------------------------===//

/// Constructs a new alias analysis using the op provided.
BufferViewFlowAnalysis::BufferViewFlowAnalysis(Operation *op) { build(op); }

static BufferViewFlowAnalysis::ValueSetT
resolveValues(const BufferViewFlowAnalysis::ValueMapT &map, Value value) {
  BufferViewFlowAnalysis::ValueSetT result;
  SmallVector<Value, 8> queue;
  queue.push_back(value);
  while (!queue.empty()) {
    Value currentValue = queue.pop_back_val();
    if (result.insert(currentValue).second) {
      auto it = map.find(currentValue);
      if (it != map.end()) {
        for (Value aliasValue : it->second)
          queue.push_back(aliasValue);
      }
    }
  }
  return result;
}

/// Find all immediate and indirect dependent buffers this value could
/// potentially have. Note that the resulting set will also contain the value
/// provided as it is a dependent alias of itself.
BufferViewFlowAnalysis::ValueSetT
BufferViewFlowAnalysis::resolve(Value rootValue) const {
  return resolveValues(dependencies, rootValue);
}

BufferViewFlowAnalysis::ValueSetT
BufferViewFlowAnalysis::resolveReverse(Value rootValue) const {
  return resolveValues(reverseDependencies, rootValue);
}

/// Removes the given values from all alias sets.
void BufferViewFlowAnalysis::remove(const SetVector<Value> &aliasValues) {
  for (auto &entry : dependencies)
    llvm::set_subtract(entry.second, aliasValues);
}

void BufferViewFlowAnalysis::rename(Value from, Value to) {
  dependencies[to] = dependencies[from];
  dependencies.erase(from);

  for (auto &[_, value] : dependencies) {
    if (value.contains(from)) {
      value.insert(to);
      value.erase(from);
    }
  }
}

/// This function constructs a mapping from values to its immediate
/// dependencies. It iterates over all blocks, gets their predecessors,
/// determines the values that will be passed to the corresponding block
/// arguments and inserts them into the underlying map. Furthermore, it wires
/// successor regions and branch-like return operations from nested regions.
void BufferViewFlowAnalysis::build(Operation *op) {
  // Registers all dependencies of the given values.
  auto registerDependencies = [&](ValueRange values, ValueRange dependencies) {
    for (auto [value, dep] : llvm::zip_equal(values, dependencies)) {
      this->dependencies[value].insert(dep);
      this->reverseDependencies[dep].insert(value);
    }
  };

  // Mark all buffer results and buffer region entry block arguments of the
  // given op as terminals.
  auto populateTerminalValues = [&](Operation *op) {
    for (Value v : op->getResults())
      if (isa<BaseMemRefType>(v.getType()))
        this->terminals.insert(v);
    for (Region &r : op->getRegions())
      for (BlockArgument v : r.getArguments())
        if (isa<BaseMemRefType>(v.getType()))
          this->terminals.insert(v);
  };

  op->walk([&](Operation *op) {
    // Query BufferViewFlowOpInterface. If the op does not implement that
    // interface, try to infer the dependencies from other interfaces that the
    // op may implement.
    if (auto bufferViewFlowOp = dyn_cast<BufferViewFlowOpInterface>(op)) {
      bufferViewFlowOp.populateDependencies(registerDependencies);
      for (Value v : op->getResults())
        if (isa<BaseMemRefType>(v.getType()) &&
            bufferViewFlowOp.mayBeTerminalBuffer(v))
          this->terminals.insert(v);
      for (Region &r : op->getRegions())
        for (BlockArgument v : r.getArguments())
          if (isa<BaseMemRefType>(v.getType()) &&
              bufferViewFlowOp.mayBeTerminalBuffer(v))
            this->terminals.insert(v);
      return WalkResult::advance();
    }

    // Add additional dependencies created by view changes to the alias list.
    if (auto viewInterface = dyn_cast<ViewLikeOpInterface>(op)) {
      registerDependencies(viewInterface.getViewSource(),
                           viewInterface.getViewDest());
      return WalkResult::advance();
    }

    if (auto branchInterface = dyn_cast<BranchOpInterface>(op)) {
      // Query all branch interfaces to link block argument dependencies.
      Block *parentBlock = branchInterface->getBlock();
      for (auto it = parentBlock->succ_begin(), e = parentBlock->succ_end();
           it != e; ++it) {
        // Query the branch op interface to get the successor operands.
        auto successorOperands =
            branchInterface.getSuccessorOperands(it.getIndex());
        // Build the actual mapping of values to their immediate dependencies.
        registerDependencies(successorOperands.getForwardedOperands(),
                             (*it)->getArguments().drop_front(
                                 successorOperands.getProducedOperandCount()));
      }
      return WalkResult::advance();
    }

    if (auto regionInterface = dyn_cast<RegionBranchOpInterface>(op)) {
      // Query the RegionBranchOpInterface to find potential successor regions.
      // Extract all entry regions and wire all initial entry successor inputs.
      SmallVector<RegionSuccessor, 2> entrySuccessors;
      regionInterface.getSuccessorRegions(/*point=*/RegionBranchPoint::parent(),
                                          entrySuccessors);
      for (RegionSuccessor &entrySuccessor : entrySuccessors) {
        // Wire the entry region's successor arguments with the initial
        // successor inputs.
        registerDependencies(
            regionInterface.getEntrySuccessorOperands(entrySuccessor),
            entrySuccessor.getSuccessorInputs());
      }

      // Wire flow between regions and from region exits.
      for (Region &region : regionInterface->getRegions()) {
        // Iterate over all successor region entries that are reachable from the
        // current region.
        SmallVector<RegionSuccessor, 2> successorRegions;
        regionInterface.getSuccessorRegions(region, successorRegions);
        for (RegionSuccessor &successorRegion : successorRegions) {
          // Iterate over all immediate terminator operations and wire the
          // successor inputs with the successor operands of each terminator.
          for (Block &block : region)
            if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(
                    block.getTerminator()))
              registerDependencies(
                  terminator.getSuccessorOperands(successorRegion),
                  successorRegion.getSuccessorInputs());
        }
      }

      return WalkResult::advance();
    }

    // Region terminators are handled together with RegionBranchOpInterface.
    if (isa<RegionBranchTerminatorOpInterface>(op))
      return WalkResult::advance();

    if (isa<CallOpInterface>(op)) {
      // This is an intra-function analysis. We have no information about other
      // functions. Conservatively assume that each operand may alias with each
      // result. Also mark the results are terminals because the function could
      // return newly allocated buffers.
      populateTerminalValues(op);
      for (Value operand : op->getOperands())
        for (Value result : op->getResults())
          registerDependencies({operand}, {result});
      return WalkResult::advance();
    }

    // We have no information about unknown ops.
    populateTerminalValues(op);

    return WalkResult::advance();
  });
}

bool BufferViewFlowAnalysis::mayBeTerminalBuffer(Value value) const {
  assert(isa<BaseMemRefType>(value.getType()) && "expected memref");
  return terminals.contains(value);
}

//===----------------------------------------------------------------------===//
// BufferOriginAnalysis
//===----------------------------------------------------------------------===//

/// Return "true" if the given value is the result of a memory allocation.
static bool hasAllocateSideEffect(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return false;
  return hasEffect<MemoryEffects::Allocate>(op, v);
}

/// Return "true" if the given value is a function block argument.
static bool isFunctionArgument(Value v) {
  auto bbArg = dyn_cast<BlockArgument>(v);
  if (!bbArg)
    return false;
  Block *b = bbArg.getOwner();
  auto funcOp = dyn_cast<FunctionOpInterface>(b->getParentOp());
  if (!funcOp)
    return false;
  return bbArg.getOwner() == &funcOp.getFunctionBody().front();
}

/// Given a memref value, return the "base" value by skipping over all
/// ViewLikeOpInterface ops (if any) in the reverse use-def chain.
static Value getViewBase(Value value) {
  while (auto viewLikeOp = value.getDefiningOp<ViewLikeOpInterface>()) {
    if (value != viewLikeOp.getViewDest()) {
      break;
    }
    value = viewLikeOp.getViewSource();
  }
  return value;
}

BufferOriginAnalysis::BufferOriginAnalysis(Operation *op) : analysis(op) {}

std::optional<bool> BufferOriginAnalysis::isSameAllocation(Value v1, Value v2) {
  assert(isa<BaseMemRefType>(v1.getType()) && "expected buffer");
  assert(isa<BaseMemRefType>(v2.getType()) && "expected buffer");

  // Skip over all view-like ops.
  v1 = getViewBase(v1);
  v2 = getViewBase(v2);

  // Fast path: If both buffers are the same SSA value, we can be sure that
  // they originate from the same allocation.
  if (v1 == v2)
    return true;

  // Compute the SSA values from which the buffers `v1` and `v2` originate.
  SmallPtrSet<Value, 16> origin1 = analysis.resolveReverse(v1);
  SmallPtrSet<Value, 16> origin2 = analysis.resolveReverse(v2);

  // Originating buffers are "terminal" if they could not be traced back any
  // further by the `BufferViewFlowAnalysis`. Examples of terminal buffers:
  // - function block arguments
  // - values defined by allocation ops such as "memref.alloc"
  // - values defined by ops that are unknown to the buffer view flow analysis
  // - values that are marked as "terminal" in the `BufferViewFlowOpInterface`
  SmallPtrSet<Value, 16> terminal1, terminal2;

  // While gathering terminal buffers, keep track of whether all terminal
  // buffers are newly allocated buffer or function entry arguments.
  bool allAllocs1 = true, allAllocs2 = true;
  bool allAllocsOrFuncEntryArgs1 = true, allAllocsOrFuncEntryArgs2 = true;

  // Helper function that gathers terminal buffers among `origin`.
  auto gatherTerminalBuffers = [this](const SmallPtrSet<Value, 16> &origin,
                                      SmallPtrSet<Value, 16> &terminal,
                                      bool &allAllocs,
                                      bool &allAllocsOrFuncEntryArgs) {
    for (Value v : origin) {
      if (isa<BaseMemRefType>(v.getType()) && analysis.mayBeTerminalBuffer(v)) {
        terminal.insert(v);
        allAllocs &= hasAllocateSideEffect(v);
        allAllocsOrFuncEntryArgs &=
            isFunctionArgument(v) || hasAllocateSideEffect(v);
      }
    }
    assert(!terminal.empty() && "expected non-empty terminal set");
  };

  // Gather terminal buffers for `v1` and `v2`.
  gatherTerminalBuffers(origin1, terminal1, allAllocs1,
                        allAllocsOrFuncEntryArgs1);
  gatherTerminalBuffers(origin2, terminal2, allAllocs2,
                        allAllocsOrFuncEntryArgs2);

  // If both `v1` and `v2` have a single matching terminal buffer, they are
  // guaranteed to originate from the same buffer allocation.
  if (llvm::hasSingleElement(terminal1) && llvm::hasSingleElement(terminal2) &&
      *terminal1.begin() == *terminal2.begin())
    return true;

  // At least one of the two values has multiple terminals.

  // Check if there is overlap between the terminal buffers of `v1` and `v2`.
  bool distinctTerminalSets = true;
  for (Value v : terminal1)
    distinctTerminalSets &= !terminal2.contains(v);
  // If there is overlap between the terminal buffers of `v1` and `v2`, we
  // cannot make an accurate decision without further analysis.
  if (!distinctTerminalSets)
    return std::nullopt;

  // If `v1` originates from only allocs, and `v2` is guaranteed to originate
  // from different allocations (that is guaranteed if `v2` originates from
  // only distinct allocs or function entry arguments), we can be sure that
  // `v1` and `v2` originate from different allocations. The same argument can
  // be made when swapping `v1` and `v2`.
  bool isolatedAlloc1 = allAllocs1 && (allAllocs2 || allAllocsOrFuncEntryArgs2);
  bool isolatedAlloc2 = (allAllocs1 || allAllocsOrFuncEntryArgs1) && allAllocs2;
  if (isolatedAlloc1 || isolatedAlloc2)
    return false;

  // Otherwise: We do not know whether `v1` and `v2` originate from the same
  // allocation or not.
  // TODO: Function arguments are currently handled conservatively. We assume
  // that they could be the same allocation.
  // TODO: Terminals other than allocations and function arguments are
  // currently handled conservatively. We assume that they could be the same
  // allocation. E.g., we currently return "nullopt" for values that originate
  // from different "memref.get_global" ops (with different symbols).
  return std::nullopt;
}
