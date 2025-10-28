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
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <optional>
#include <utility>

using namespace mlir;

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

/// Given a successor (`region`) of a RegionBranchOpInterface, collect all of
/// the underlying values being addressed by one of the successor inputs. If the
/// provided `region` is null, as per `RegionBranchOpInterface` this represents
/// the parent operation.
static void collectUnderlyingAddressValues(RegionBranchOpInterface branch,
                                           Region *region, Value inputValue,
                                           unsigned inputIndex,
                                           unsigned maxDepth,
                                           DenseSet<Value> &visited,
                                           SmallVectorImpl<Value> &output) {
  // Given the index of a region of the branch (`predIndex`), or std::nullopt to
  // represent the parent operation, try to return the index into the outputs of
  // this region predecessor that correspond to the input values of `region`. If
  // an index could not be found, std::nullopt is returned instead.
  auto getOperandIndexIfPred =
      [&](RegionBranchPoint pred) -> std::optional<unsigned> {
    SmallVector<RegionSuccessor, 2> successors;
    branch.getSuccessorRegions(pred, successors);
    for (RegionSuccessor &successor : successors) {
      if (successor.getSuccessor() != region)
        continue;
      // Check that the successor inputs map to the given input value.
      ValueRange inputs = successor.getSuccessorInputs();
      if (inputs.empty()) {
        output.push_back(inputValue);
        break;
      }
      unsigned firstInputIndex, lastInputIndex;
      if (region) {
        firstInputIndex = cast<BlockArgument>(inputs[0]).getArgNumber();
        lastInputIndex = cast<BlockArgument>(inputs.back()).getArgNumber();
      } else {
        firstInputIndex = cast<OpResult>(inputs[0]).getResultNumber();
        lastInputIndex = cast<OpResult>(inputs.back()).getResultNumber();
      }
      if (firstInputIndex > inputIndex || lastInputIndex < inputIndex) {
        output.push_back(inputValue);
        break;
      }
      return inputIndex - firstInputIndex;
    }
    return std::nullopt;
  };

  // Check branches from the parent operation.
  auto branchPoint = RegionBranchPoint::parent();
  if (region)
    branchPoint = region;

  if (std::optional<unsigned> operandIndex =
          getOperandIndexIfPred(/*predIndex=*/RegionBranchPoint::parent())) {
    collectUnderlyingAddressValues(
        branch.getEntrySuccessorOperands(branchPoint)[*operandIndex], maxDepth,
        visited, output);
  }
  // Check branches from each child region.
  Operation *op = branch.getOperation();
  for (Region &region : op->getRegions()) {
    if (std::optional<unsigned> operandIndex = getOperandIndexIfPred(region)) {
      for (Block &block : region) {
        // Try to determine possible region-branch successor operands for the
        // current region.
        if (auto term = dyn_cast<RegionBranchTerminatorOpInterface>(
                block.getTerminator())) {
          collectUnderlyingAddressValues(
              term.getSuccessorOperands(branchPoint)[*operandIndex], maxDepth,
              visited, output);
        } else if (block.getNumSuccessors()) {
          // Otherwise, if this terminator may exit the region we can't make
          // any assumptions about which values get passed.
          output.push_back(inputValue);
          return;
        }
      }
    }
  }
}

/// Given a result, collect all of the underlying values being addressed.
static void collectUnderlyingAddressValues(OpResult result, unsigned maxDepth,
                                           DenseSet<Value> &visited,
                                           SmallVectorImpl<Value> &output) {
  Operation *op = result.getOwner();

  // If this is a view, unwrap to the source.
  if (ViewLikeOpInterface view = dyn_cast<ViewLikeOpInterface>(op)) {
    if (result == view.getViewDest()) {
      return collectUnderlyingAddressValues(view.getViewSource(), maxDepth,
                                            visited, output);
    }
  }
  // Check to see if we can reason about the control flow of this op.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    return collectUnderlyingAddressValues(branch, /*region=*/nullptr, result,
                                          result.getResultNumber(), maxDepth,
                                          visited, output);
  }

  output.push_back(result);
}

/// Given a block argument, collect all of the underlying values being
/// addressed.
static void collectUnderlyingAddressValues(BlockArgument arg, unsigned maxDepth,
                                           DenseSet<Value> &visited,
                                           SmallVectorImpl<Value> &output) {
  Block *block = arg.getOwner();
  unsigned argNumber = arg.getArgNumber();

  // Handle the case of a non-entry block.
  if (!block->isEntryBlock()) {
    for (auto it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
      auto branch = dyn_cast<BranchOpInterface>((*it)->getTerminator());
      if (!branch) {
        // We can't analyze the control flow, so bail out early.
        output.push_back(arg);
        return;
      }

      // Try to get the operand passed for this argument.
      unsigned index = it.getSuccessorIndex();
      Value operand = branch.getSuccessorOperands(index)[argNumber];
      if (!operand) {
        // We can't analyze the control flow, so bail out early.
        output.push_back(arg);
        return;
      }
      collectUnderlyingAddressValues(operand, maxDepth, visited, output);
    }
    return;
  }

  // Otherwise, check to see if we can reason about the control flow of this op.
  Region *region = block->getParent();
  Operation *op = region->getParentOp();
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    return collectUnderlyingAddressValues(branch, region, arg, argNumber,
                                          maxDepth, visited, output);
  }

  // We can't reason about the underlying address of this argument.
  output.push_back(arg);
}

/// Given a value, collect all of the underlying values being addressed.
static void collectUnderlyingAddressValues(Value value, unsigned maxDepth,
                                           DenseSet<Value> &visited,
                                           SmallVectorImpl<Value> &output) {
  // Check that we don't infinitely recurse.
  if (!visited.insert(value).second)
    return;
  if (maxDepth == 0) {
    output.push_back(value);
    return;
  }
  --maxDepth;

  if (BlockArgument arg = dyn_cast<BlockArgument>(value))
    return collectUnderlyingAddressValues(arg, maxDepth, visited, output);
  collectUnderlyingAddressValues(cast<OpResult>(value), maxDepth, visited,
                                 output);
}

/// Given a value, collect all of the underlying values being addressed.
static void collectUnderlyingAddressValues(Value value,
                                           SmallVectorImpl<Value> &output) {
  DenseSet<Value> visited;
  collectUnderlyingAddressValues(value, maxUnderlyingValueSearchDepth, visited,
                                 output);
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
  // Try to get a memory effect interface for the parent operation.
  Operation *op;
  if (BlockArgument arg = dyn_cast<BlockArgument>(value))
    op = arg.getOwner()->getParentOp();
  else
    op = cast<OpResult>(value).getOwner();
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return failure();

  // Try to find an allocation effect on the resource.
  if (!(effect = interface.getEffectOnValue<MemoryEffects::Allocate>(value)))
    return failure();

  // If we found an allocation effect, try to find a scope for the allocation.
  // If the resource of this allocation is automatically scoped, find the parent
  // operation that bounds the allocation scope.
  if (llvm::isa<SideEffects::AutomaticAllocationScopeResource>(
          effect->getResource())) {
    allocScopeOp = op->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
    return success();
  }

  // TODO: Here we could look at the users to see if the resource is either
  // freed on all paths within the region, or is just not captured by anything.
  // For now assume allocation scope to the function scope (we don't care if
  // pointer escape outside function).
  allocScopeOp = op->getParentOfType<FunctionOpInterface>();
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
  if (lhs == rhs)
    return AliasResult::MustAlias;
  Operation *lhsAllocScope = nullptr, *rhsAllocScope = nullptr;
  std::optional<MemoryEffects::EffectInstance> lhsAlloc, rhsAlloc;

  // Handle the case where lhs is a constant.
  Attribute lhsAttr, rhsAttr;
  if (matchPattern(lhs, m_Constant(&lhsAttr))) {
    // TODO: This is overly conservative. Two matching constants don't
    // necessarily map to the same address. For example, if the two values
    // correspond to different symbols that both represent a definition.
    if (matchPattern(rhs, m_Constant(&rhsAttr)))
      return AliasResult::MayAlias;

    // Try to find an alloc effect on rhs. If an effect was found we can't
    // alias, otherwise we might.
    return succeeded(getAllocEffectFor(rhs, rhsAlloc, rhsAllocScope))
               ? AliasResult::NoAlias
               : AliasResult::MayAlias;
  }
  // Handle the case where rhs is a constant.
  if (matchPattern(rhs, m_Constant(&rhsAttr))) {
    // Try to find an alloc effect on lhs. If an effect was found we can't
    // alias, otherwise we might.
    return succeeded(getAllocEffectFor(lhs, lhsAlloc, lhsAllocScope))
               ? AliasResult::NoAlias
               : AliasResult::MayAlias;
  }

  if (std::optional<AliasResult> result = checkDistinctObjects(lhs, rhs))
    return *result;

  // Otherwise, neither of the values are constant so check to see if either has
  // an allocation effect.
  bool lhsHasAlloc = succeeded(getAllocEffectFor(lhs, lhsAlloc, lhsAllocScope));
  bool rhsHasAlloc = succeeded(getAllocEffectFor(rhs, rhsAlloc, rhsAllocScope));
  if (lhsHasAlloc == rhsHasAlloc) {
    // If both values have an allocation effect we know they don't alias, and if
    // neither have an effect we can't make an assumptions.
    return lhsHasAlloc ? AliasResult::NoAlias : AliasResult::MayAlias;
  }

  // When we reach this point we have one value with a known allocation effect,
  // and one without. Move the one with the effect to the lhs to make the next
  // checks simpler.
  if (rhsHasAlloc) {
    std::swap(lhs, rhs);
    lhsAlloc = rhsAlloc;
    lhsAllocScope = rhsAllocScope;
  }

  // If the effect has a scoped allocation region, check to see if the
  // non-effect value is defined above that scope.
  if (lhsAllocScope) {
    // If the parent operation of rhs is an ancestor of the allocation scope, or
    // if rhs is an entry block argument of the allocation scope we know the two
    // values can't alias.
    Operation *rhsParentOp = rhs.getParentRegion()->getParentOp();
    if (rhsParentOp->isProperAncestor(lhsAllocScope))
      return AliasResult::NoAlias;
    if (rhsParentOp == lhsAllocScope) {
      BlockArgument rhsArg = dyn_cast<BlockArgument>(rhs);
      if (rhsArg && rhs.getParentBlock()->isEntryBlock())
        return AliasResult::NoAlias;
    }
  }

  // If we couldn't reason about the relationship between the two values,
  // conservatively assume they might alias.
  return AliasResult::MayAlias;
}

/// Given the two values, return their aliasing behavior.
AliasResult LocalAliasAnalysis::alias(Value lhs, Value rhs) {
  if (lhs == rhs)
    return AliasResult::MustAlias;

  // Get the underlying values being addressed.
  SmallVector<Value, 8> lhsValues, rhsValues;
  collectUnderlyingAddressValues(lhs, lhsValues);
  collectUnderlyingAddressValues(rhs, rhsValues);

  // If we failed to collect for either of the values somehow, conservatively
  // assume they may alias.
  if (lhsValues.empty() || rhsValues.empty())
    return AliasResult::MayAlias;

  // Check the alias results against each of the underlying values.
  std::optional<AliasResult> result;
  for (Value lhsVal : lhsValues) {
    for (Value rhsVal : rhsValues) {
      AliasResult nextResult = aliasImpl(lhsVal, rhsVal);
      result = result ? result->merge(nextResult) : nextResult;
    }
  }

  // We should always have a valid result here.
  return *result;
}

//===----------------------------------------------------------------------===//
// LocalAliasAnalysis: getModRef
//===----------------------------------------------------------------------===//

ModRefResult LocalAliasAnalysis::getModRef(Operation *op, Value location) {
  // Check to see if this operation relies on nested side effects.
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    // TODO: To check recursive operations we need to check all of the nested
    // operations, which can result in a quadratic number of queries. We should
    // introduce some caching of some kind to help alleviate this, especially as
    // this caching could be used in other areas of the codebase (e.g. when
    // checking `wouldOpBeTriviallyDead`).
    return ModRefResult::getModAndRef();
  }

  // Otherwise, check to see if this operation has a memory effect interface.
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return ModRefResult::getModAndRef();

  // Build a ModRefResult by merging the behavior of the effects of this
  // operation.
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);

  ModRefResult result = ModRefResult::getNoModRef();
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (isa<MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect()))
      continue;

    // Check for an alias between the effect and our memory location.
    // TODO: Add support for checking an alias with a symbol reference.
    AliasResult aliasResult = AliasResult::MayAlias;
    if (Value effectValue = effect.getValue())
      aliasResult = alias(effectValue, location);

    // If we don't alias, ignore this effect.
    if (aliasResult.isNo())
      continue;

    // Merge in the corresponding mod or ref for this effect.
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      result = result.merge(ModRefResult::getRef());
    } else {
      assert(isa<MemoryEffects::Write>(effect.getEffect()));
      result = result.merge(ModRefResult::getMod());
    }
    if (result.isModAndRef())
      break;
  }
  return result;
}
