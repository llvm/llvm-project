//===- LocalAliasAnalysis.cpp - Local stateless alias Analysis for MLIR ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

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
  // Given the index of a region of the branch (`predIndex`), or None to
  // represent the parent operation, try to return the index into the outputs of
  // this region predecessor that correspond to the input values of `region`. If
  // an index could not be found, None is returned instead.
  auto getOperandIndexIfPred =
      [&](Optional<unsigned> predIndex) -> Optional<unsigned> {
    SmallVector<RegionSuccessor, 2> successors;
    branch.getSuccessorRegions(predIndex, successors);
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
        firstInputIndex = inputs[0].cast<BlockArgument>().getArgNumber();
        lastInputIndex = inputs.back().cast<BlockArgument>().getArgNumber();
      } else {
        firstInputIndex = inputs[0].cast<OpResult>().getResultNumber();
        lastInputIndex = inputs.back().cast<OpResult>().getResultNumber();
      }
      if (firstInputIndex > inputIndex || lastInputIndex < inputIndex) {
        output.push_back(inputValue);
        break;
      }
      return inputIndex - firstInputIndex;
    }
    return llvm::None;
  };

  // Check branches from the parent operation.
  if (region) {
    if (Optional<unsigned> operandIndex =
            getOperandIndexIfPred(/*predIndex=*/llvm::None)) {
      collectUnderlyingAddressValues(
          branch.getSuccessorEntryOperands(
              region->getRegionNumber())[*operandIndex],
          maxDepth, visited, output);
    }
  }
  // Check branches from each child region.
  Operation *op = branch.getOperation();
  for (int i = 0, e = op->getNumRegions(); i != e; ++i) {
    if (Optional<unsigned> operandIndex = getOperandIndexIfPred(i)) {
      for (Block &block : op->getRegion(i)) {
        Operation *term = block.getTerminator();
        if (term->hasTrait<OpTrait::ReturnLike>()) {
          collectUnderlyingAddressValues(term->getOperand(*operandIndex),
                                         maxDepth, visited, output);
        } else if (term->getNumSuccessors()) {
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
  if (ViewLikeOpInterface view = dyn_cast<ViewLikeOpInterface>(op))
    return collectUnderlyingAddressValues(view.getViewSource(), maxDepth,
                                          visited, output);
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
      Optional<OperandRange> operands = branch.getSuccessorOperands(index);
      if (!operands) {
        // We can't analyze the control flow, so bail out early.
        output.push_back(arg);
        return;
      }
      collectUnderlyingAddressValues((*operands)[argNumber], maxDepth, visited,
                                     output);
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

  if (BlockArgument arg = value.dyn_cast<BlockArgument>())
    return collectUnderlyingAddressValues(arg, maxDepth, visited, output);
  collectUnderlyingAddressValues(value.cast<OpResult>(), maxDepth, visited,
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
// LocalAliasAnalysis
//===----------------------------------------------------------------------===//

/// Given a value, try to get an allocation effect attached to it. If
/// successful, `allocEffect` is populated with the effect. If an effect was
/// found, `allocScopeOp` is also specified if a parent operation of `value`
/// could be identified that bounds the scope of the allocated value; i.e. if
/// non-null it specifies the parent operation that the allocation does not
/// escape. If no scope is found, `allocScopeOp` is set to nullptr.
static LogicalResult
getAllocEffectFor(Value value, Optional<MemoryEffects::EffectInstance> &effect,
                  Operation *&allocScopeOp) {
  // Try to get a memory effect interface for the parent operation.
  Operation *op;
  if (BlockArgument arg = value.dyn_cast<BlockArgument>())
    op = arg.getOwner()->getParentOp();
  else
    op = value.cast<OpResult>().getOwner();
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
  allocScopeOp = nullptr;
  return success();
}

/// Given the two values, return their aliasing behavior.
static AliasResult aliasImpl(Value lhs, Value rhs) {
  if (lhs == rhs)
    return AliasResult::MustAlias;
  Operation *lhsAllocScope = nullptr, *rhsAllocScope = nullptr;
  Optional<MemoryEffects::EffectInstance> lhsAlloc, rhsAlloc;

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
      BlockArgument rhsArg = rhs.dyn_cast<BlockArgument>();
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
  Optional<AliasResult> result;
  for (Value lhsVal : lhsValues) {
    for (Value rhsVal : rhsValues) {
      AliasResult nextResult = aliasImpl(lhsVal, rhsVal);
      result = result ? result->merge(nextResult) : nextResult;
    }
  }

  // We should always have a valid result here.
  return *result;
}
