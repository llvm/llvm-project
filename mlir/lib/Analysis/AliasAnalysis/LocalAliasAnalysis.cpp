//===- LocalAliasAnalysis.cpp - Local stateless alias Analysis for MLIR ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include <cassert>
#include <optional>
#include <utility>

using namespace mlir;

#define DEBUG_TYPE "local-alias-analysis"

//===----------------------------------------------------------------------===//
// Underlying Address Computation
//===----------------------------------------------------------------------===//

/// The maximum depth that will be searched when trying to find an underlying
/// value.
static constexpr unsigned maxUnderlyingValueSearchDepth = 10;

/// Given a value, collect all of the underlying values being addressed.
static void collectUnderlyingAddressValues(Value value, unsigned maxDepth,
                                           DenseSet<Value> &visited,
                                           SmallVectorImpl<Value> &output);

/// Given a RegionBranchOpInterface operation  (`branch`), a Value`inputValue`
/// which is an input for the provided successor (`initialSuccessor`), try to
/// find the possible sources for the value along the control flow edges.
static void collectUnderlyingAddressValues2(
    RegionBranchOpInterface branch, RegionSuccessor initialSuccessor,
    Value inputValue, unsigned inputIndex, unsigned maxDepth,
    DenseSet<Value> &visited, SmallVectorImpl<Value> &output) {
  LDBG() << "collectUnderlyingAddressValues2: "
         << OpWithFlags(branch.getOperation(), OpPrintingFlags().skipRegions());
  LDBG() << " with initialSuccessor " << initialSuccessor;
  LDBG() << "  inputValue: " << inputValue;
  LDBG() << "  inputIndex: " << inputIndex;
  LDBG() << "  maxDepth: " << maxDepth;
  ValueRange inputs = branch.getSuccessorInputs(initialSuccessor);
  if (inputs.empty()) {
    LDBG() << "  input is empty, enqueue value";
    output.push_back(inputValue);
    return;
  }
  unsigned firstInputIndex, lastInputIndex;
  if (isa<BlockArgument>(inputs[0])) {
    firstInputIndex = cast<BlockArgument>(inputs[0]).getArgNumber();
    lastInputIndex = cast<BlockArgument>(inputs.back()).getArgNumber();
  } else {
    firstInputIndex = cast<OpResult>(inputs[0]).getResultNumber();
    lastInputIndex = cast<OpResult>(inputs.back()).getResultNumber();
  }
  if (firstInputIndex > inputIndex || lastInputIndex < inputIndex) {
    LDBG() << "  !! Input index " << inputIndex << " out of range "
           << firstInputIndex << " to " << lastInputIndex
           << ", adding input value to output";
    output.push_back(inputValue);
    return;
  }
  SmallVector<Value> predecessorValues;
  branch.getPredecessorValues(initialSuccessor, inputIndex - firstInputIndex,
                              predecessorValues);
  LDBG() << "  Found " << predecessorValues.size() << " predecessor values";
  for (Value predecessorValue : predecessorValues) {
    LDBG() << "    Processing predecessor value: " << predecessorValue;
    collectUnderlyingAddressValues(predecessorValue, maxDepth, visited, output);
  }
}

/// Given a result, collect all of the underlying values being addressed.
static void collectUnderlyingAddressValues(OpResult result, unsigned maxDepth,
                                           DenseSet<Value> &visited,
                                           SmallVectorImpl<Value> &output) {
  LDBG() << "collectUnderlyingAddressValues (OpResult): " << result;
  LDBG() << "  maxDepth: " << maxDepth;

  Operation *op = result.getOwner();

  // If this is a view, unwrap to the source.
  if (ViewLikeOpInterface view = dyn_cast<ViewLikeOpInterface>(op)) {
    if (result == view.getViewDest()) {
      LDBG() << "  Unwrapping view to source: " << view.getViewSource();
      return collectUnderlyingAddressValues(view.getViewSource(), maxDepth,
                                            visited, output);
    }
  }
  // Check to see if we can reason about the control flow of this op.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    LDBG() << "  Processing region branch operation";
    return collectUnderlyingAddressValues2(branch, RegionSuccessor::parent(),
                                           result, result.getResultNumber(),
                                           maxDepth, visited, output);
  }

  LDBG() << "  Adding result to output: " << result;
  output.push_back(result);
}

/// Given a block argument, collect all of the underlying values being
/// addressed.
static void collectUnderlyingAddressValues(BlockArgument arg, unsigned maxDepth,
                                           DenseSet<Value> &visited,
                                           SmallVectorImpl<Value> &output) {
  LDBG() << "collectUnderlyingAddressValues (BlockArgument): " << arg;
  LDBG() << "  maxDepth: " << maxDepth;
  LDBG() << "  argNumber: " << arg.getArgNumber();
  LDBG() << "  isEntryBlock: " << arg.getOwner()->isEntryBlock();

  Block *block = arg.getOwner();
  unsigned argNumber = arg.getArgNumber();

  // Handle the case of a non-entry block.
  if (!block->isEntryBlock()) {
    LDBG() << "  Processing non-entry block with "
           << std::distance(block->pred_begin(), block->pred_end())
           << " predecessors";
    for (auto it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
      auto branch = dyn_cast<BranchOpInterface>((*it)->getTerminator());
      if (!branch) {
        LDBG() << "    Cannot analyze control flow, adding argument to output";
        // We can't analyze the control flow, so bail out early.
        output.push_back(arg);
        return;
      }

      // Try to get the operand passed for this argument.
      unsigned index = it.getSuccessorIndex();
      Value operand = branch.getSuccessorOperands(index)[argNumber];
      if (!operand) {
        LDBG() << "    No operand found for argument, adding to output";
        // We can't analyze the control flow, so bail out early.
        output.push_back(arg);
        return;
      }
      LDBG() << "    Processing operand from predecessor: " << operand;
      collectUnderlyingAddressValues(operand, maxDepth, visited, output);
    }
    return;
  }

  // Otherwise, check to see if we can reason about the control flow of this op.
  Region *region = block->getParent();
  Operation *op = region->getParentOp();
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    LDBG() << "  Processing region branch operation for entry block";
    // We have to find the successor matching the region, so that the input
    // arguments are correctly set.
    // TODO: this isn't comprehensive: the successor may not be reachable from
    // the entry block.
    SmallVector<RegionSuccessor> successors;
    branch.getSuccessorRegions(RegionBranchPoint::parent(), successors);
    for (RegionSuccessor &successor : successors) {
      if (successor.getSuccessor() == region) {
        LDBG() << "  Found matching region successor: " << successor;
        return collectUnderlyingAddressValues2(
            branch, successor, arg, argNumber, maxDepth, visited, output);
      }
    }
    LDBG() << "  No matching region successor found, adding argument to output";
    output.push_back(arg);
    return;
  }

  LDBG()
      << "  Cannot reason about underlying address, adding argument to output";
  // We can't reason about the underlying address of this argument.
  output.push_back(arg);
}

/// Given a value, collect all of the underlying values being addressed.
static void collectUnderlyingAddressValues(Value value, unsigned maxDepth,
                                           DenseSet<Value> &visited,
                                           SmallVectorImpl<Value> &output) {
  LDBG() << "collectUnderlyingAddressValues: " << value;
  LDBG() << "  maxDepth: " << maxDepth;

  // Check that we don't infinitely recurse.
  if (!visited.insert(value).second) {
    LDBG() << "  Value already visited, skipping";
    return;
  }
  if (maxDepth == 0) {
    LDBG() << "  Max depth reached, adding value to output";
    output.push_back(value);
    return;
  }
  --maxDepth;

  if (BlockArgument arg = dyn_cast<BlockArgument>(value)) {
    LDBG() << "  Processing as BlockArgument";
    return collectUnderlyingAddressValues(arg, maxDepth, visited, output);
  }
  LDBG() << "  Processing as OpResult";
  collectUnderlyingAddressValues(cast<OpResult>(value), maxDepth, visited,
                                 output);
}

/// Given a value, collect all of the underlying values being addressed.
static void collectUnderlyingAddressValues(Value value,
                                           SmallVectorImpl<Value> &output) {
  LDBG() << "collectUnderlyingAddressValues: " << value;
  DenseSet<Value> visited;
  collectUnderlyingAddressValues(value, maxUnderlyingValueSearchDepth, visited,
                                 output);
  LDBG() << "  Collected " << output.size() << " underlying values";
}

//===----------------------------------------------------------------------===//
// LocalAliasAnalysis: alias
//===----------------------------------------------------------------------===//

/// Given a value, try to get an allocation effect attached to it. If
/// successful, `allocEffect` is populated with the effect. If an effect was
/// found, `allocScopeOp` is also specified if a parent operation of `value`
/// could be identified that bounds the scope of the allocated value; i.e. if
/// non-null it specifies the parent operation that the allocation does not
/// escape. If no scope is found, `allocScopeOp` is set to nullptr.
static LogicalResult
getAllocEffectFor(Value value,
                  std::optional<MemoryEffects::EffectInstance> &effect,
                  Operation *&allocScopeOp) {
  LDBG() << "getAllocEffectFor: " << value;

  // Try to get a memory effect interface for the parent operation.
  Operation *op;
  if (BlockArgument arg = dyn_cast<BlockArgument>(value)) {
    op = arg.getOwner()->getParentOp();
    LDBG() << "  BlockArgument, parent op: "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());
  } else {
    op = cast<OpResult>(value).getOwner();
    LDBG() << "  OpResult, owner op: "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());
  }

  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) {
    LDBG() << "  No memory effect interface found";
    return failure();
  }

  // Try to find an allocation effect on the resource.
  if (!(effect = interface.getEffectOnValue<MemoryEffects::Allocate>(value))) {
    LDBG() << "  No allocation effect found on value";
    return failure();
  }

  LDBG() << "  Found allocation effect";

  // If we found an allocation effect, try to find a scope for the allocation.
  // If the resource of this allocation is automatically scoped, find the parent
  // operation that bounds the allocation scope.
  if (llvm::isa<SideEffects::AutomaticAllocationScopeResource>(
          effect->getResource())) {
    allocScopeOp = op->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
    if (allocScopeOp) {
      LDBG() << "  Automatic allocation scope found: "
             << OpWithFlags(allocScopeOp, OpPrintingFlags().skipRegions());
    } else {
      LDBG() << "  Automatic allocation scope found: null";
    }
    return success();
  }

  // TODO: Here we could look at the users to see if the resource is either
  // freed on all paths within the region, or is just not captured by anything.
  // For now assume allocation scope to the function scope (we don't care if
  // pointer escape outside function).
  allocScopeOp = op->getParentOfType<FunctionOpInterface>();
  if (allocScopeOp) {
    LDBG() << "  Function scope found: "
           << OpWithFlags(allocScopeOp, OpPrintingFlags().skipRegions());
  } else {
    LDBG() << "  Function scope found: null";
  }
  return success();
}

static Operation *isDistinctObjectsOp(Operation *op) {
  if (op && op->hasTrait<OpTrait::DistinctObjectsTrait>())
    return op;

  return nullptr;
}

static Value getDistinctObjectsOperand(Operation *op, Value value) {
  unsigned argNumber = cast<OpResult>(value).getResultNumber();
  return op->getOperand(argNumber);
}

static std::optional<AliasResult> checkDistinctObjects(Value lhs, Value rhs) {
  // We should already checked that lhs and rhs are different.
  assert(lhs != rhs && "lhs and rhs must be different");

  // Result and corresponding operand must alias.
  auto lhsOp = isDistinctObjectsOp(lhs.getDefiningOp());
  if (lhsOp && getDistinctObjectsOperand(lhsOp, lhs) == rhs)
    return AliasResult::MustAlias;

  auto rhsOp = isDistinctObjectsOp(rhs.getDefiningOp());
  if (rhsOp && getDistinctObjectsOperand(rhsOp, rhs) == lhs)
    return AliasResult::MustAlias;

  // If two different values come from the same `DistinctObjects` operation,
  // they don't alias.
  if (lhsOp && lhsOp == rhsOp)
    return AliasResult::NoAlias;

  return std::nullopt;
}

/// Given the two values, return their aliasing behavior.
AliasResult LocalAliasAnalysis::aliasImpl(Value lhs, Value rhs) {
  LDBG() << "aliasImpl: " << lhs << " vs " << rhs;

  if (lhs == rhs) {
    LDBG() << "  Same value, must alias";
    return AliasResult::MustAlias;
  }

  Operation *lhsAllocScope = nullptr, *rhsAllocScope = nullptr;
  std::optional<MemoryEffects::EffectInstance> lhsAlloc, rhsAlloc;

  // Handle the case where lhs is a constant.
  Attribute lhsAttr, rhsAttr;
  if (matchPattern(lhs, m_Constant(&lhsAttr))) {
    LDBG() << "  lhs is constant";
    // TODO: This is overly conservative. Two matching constants don't
    // necessarily map to the same address. For example, if the two values
    // correspond to different symbols that both represent a definition.
    if (matchPattern(rhs, m_Constant(&rhsAttr))) {
      LDBG() << "  rhs is also constant, may alias";
      return AliasResult::MayAlias;
    }

    // Try to find an alloc effect on rhs. If an effect was found we can't
    // alias, otherwise we might.
    bool rhsHasAlloc =
        succeeded(getAllocEffectFor(rhs, rhsAlloc, rhsAllocScope));
    LDBG() << "  rhs has alloc effect: " << rhsHasAlloc;
    return rhsHasAlloc ? AliasResult::NoAlias : AliasResult::MayAlias;
  }
  // Handle the case where rhs is a constant.
  if (matchPattern(rhs, m_Constant(&rhsAttr))) {
    LDBG() << "  rhs is constant";
    // Try to find an alloc effect on lhs. If an effect was found we can't
    // alias, otherwise we might.
    bool lhsHasAlloc =
        succeeded(getAllocEffectFor(lhs, lhsAlloc, lhsAllocScope));
    LDBG() << "  lhs has alloc effect: " << lhsHasAlloc;
    return lhsHasAlloc ? AliasResult::NoAlias : AliasResult::MayAlias;
  }

  if (std::optional<AliasResult> result = checkDistinctObjects(lhs, rhs))
    return *result;

  // Otherwise, neither of the values are constant so check to see if either has
  // an allocation effect.
  bool lhsHasAlloc = succeeded(getAllocEffectFor(lhs, lhsAlloc, lhsAllocScope));
  bool rhsHasAlloc = succeeded(getAllocEffectFor(rhs, rhsAlloc, rhsAllocScope));
  LDBG() << "  lhs has alloc effect: " << lhsHasAlloc;
  LDBG() << "  rhs has alloc effect: " << rhsHasAlloc;

  if (lhsHasAlloc == rhsHasAlloc) {
    // If both values have an allocation effect we know they don't alias, and if
    // neither have an effect we can't make an assumptions.
    LDBG() << "  Both have same alloc status: "
           << (lhsHasAlloc ? "NoAlias" : "MayAlias");
    return lhsHasAlloc ? AliasResult::NoAlias : AliasResult::MayAlias;
  }

  // When we reach this point we have one value with a known allocation effect,
  // and one without. Move the one with the effect to the lhs to make the next
  // checks simpler.
  if (rhsHasAlloc) {
    LDBG() << "  Swapping lhs and rhs to put alloc effect on lhs";
    std::swap(lhs, rhs);
    lhsAlloc = rhsAlloc;
    lhsAllocScope = rhsAllocScope;
  }

  // If the effect has a scoped allocation region, check to see if the
  // non-effect value is defined above that scope.
  if (lhsAllocScope) {
    LDBG() << "  Checking allocation scope: "
           << OpWithFlags(lhsAllocScope, OpPrintingFlags().skipRegions());
    // If the parent operation of rhs is an ancestor of the allocation scope, or
    // if rhs is an entry block argument of the allocation scope we know the two
    // values can't alias.
    Operation *rhsParentOp = rhs.getParentRegion()->getParentOp();
    if (rhsParentOp->isProperAncestor(lhsAllocScope)) {
      LDBG() << "  rhs parent is ancestor of alloc scope, no alias";
      return AliasResult::NoAlias;
    }
    if (rhsParentOp == lhsAllocScope) {
      BlockArgument rhsArg = dyn_cast<BlockArgument>(rhs);
      if (rhsArg && rhs.getParentBlock()->isEntryBlock()) {
        LDBG() << "  rhs is entry block arg of alloc scope, no alias";
        return AliasResult::NoAlias;
      }
    }
  }

  // If we couldn't reason about the relationship between the two values,
  // conservatively assume they might alias.
  LDBG() << "  Cannot reason about relationship, may alias";
  return AliasResult::MayAlias;
}

/// Given the two values, return their aliasing behavior.
AliasResult LocalAliasAnalysis::alias(Value lhs, Value rhs) {
  LDBG() << "alias: " << lhs << " vs " << rhs;

  if (lhs == rhs) {
    LDBG() << "  Same value, must alias";
    return AliasResult::MustAlias;
  }

  // Get the underlying values being addressed.
  SmallVector<Value, 8> lhsValues, rhsValues;
  collectUnderlyingAddressValues(lhs, lhsValues);
  collectUnderlyingAddressValues(rhs, rhsValues);

  LDBG() << "  lhs underlying values: " << lhsValues.size();
  LDBG() << "  rhs underlying values: " << rhsValues.size();

  // If we failed to collect for either of the values somehow, conservatively
  // assume they may alias.
  if (lhsValues.empty() || rhsValues.empty()) {
    LDBG() << "  Failed to collect underlying values, may alias";
    return AliasResult::MayAlias;
  }

  // Check the alias results against each of the underlying values.
  std::optional<AliasResult> result;
  for (Value lhsVal : lhsValues) {
    for (Value rhsVal : rhsValues) {
      LDBG() << "  Checking underlying values: " << lhsVal << " vs " << rhsVal;
      AliasResult nextResult = aliasImpl(lhsVal, rhsVal);
      LDBG() << "  Result: "
             << (nextResult == AliasResult::MustAlias ? "MustAlias"
                 : nextResult == AliasResult::NoAlias ? "NoAlias"
                                                      : "MayAlias");
      result = result ? result->merge(nextResult) : nextResult;
    }
  }

  // We should always have a valid result here.
  LDBG() << "  Final result: "
         << (result->isMust() ? "MustAlias"
             : result->isNo() ? "NoAlias"
                              : "MayAlias");
  return *result;
}

//===----------------------------------------------------------------------===//
// LocalAliasAnalysis: getModRef
//===----------------------------------------------------------------------===//

ModRefResult LocalAliasAnalysis::getModRef(Operation *op, Value location) {
  LDBG() << "getModRef: " << OpWithFlags(op, OpPrintingFlags().skipRegions())
         << " on location " << location;

  // Check to see if this operation relies on nested side effects.
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    LDBG() << "  Operation has recursive memory effects, returning ModAndRef";
    // TODO: To check recursive operations we need to check all of the nested
    // operations, which can result in a quadratic number of queries. We should
    // introduce some caching of some kind to help alleviate this, especially as
    // this caching could be used in other areas of the codebase (e.g. when
    // checking `wouldOpBeTriviallyDead`).
    return ModRefResult::getModAndRef();
  }

  // Otherwise, check to see if this operation has a memory effect interface.
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) {
    LDBG() << "  No memory effect interface, returning ModAndRef";
    return ModRefResult::getModAndRef();
  }

  // Build a ModRefResult by merging the behavior of the effects of this
  // operation.
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  LDBG() << "  Found " << effects.size() << " memory effects";

  ModRefResult result = ModRefResult::getNoModRef();
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (isa<MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect())) {
      LDBG() << "    Skipping alloc/free effect";
      continue;
    }

    // Check for an alias between the effect and our memory location.
    // TODO: Add support for checking an alias with a symbol reference.
    AliasResult aliasResult = AliasResult::MayAlias;
    if (Value effectValue = effect.getValue()) {
      LDBG() << "    Checking alias between effect value " << effectValue
             << " and location " << location;
      aliasResult = alias(effectValue, location);
      LDBG() << "    Alias result: "
             << (aliasResult.isMust() ? "MustAlias"
                 : aliasResult.isNo() ? "NoAlias"
                                      : "MayAlias");
    } else {
      LDBG() << "    No effect value, assuming MayAlias";
    }

    // If we don't alias, ignore this effect.
    if (aliasResult.isNo()) {
      LDBG() << "    No alias, ignoring effect";
      continue;
    }

    // Merge in the corresponding mod or ref for this effect.
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      LDBG() << "    Adding Ref to result";
      result = result.merge(ModRefResult::getRef());
    } else {
      assert(isa<MemoryEffects::Write>(effect.getEffect()));
      LDBG() << "    Adding Mod to result";
      result = result.merge(ModRefResult::getMod());
    }
    if (result.isModAndRef()) {
      LDBG() << "    Result is now ModAndRef, breaking";
      break;
    }
  }

  LDBG() << "  Final ModRef result: "
         << (result.isModAndRef() ? "ModAndRef"
             : result.isMod()     ? "Mod"
             : result.isRef()     ? "Ref"
                                  : "NoModRef");
  return result;
}
