//===- ControlFlowInterfaces.cpp - ControlFlow Interfaces -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/DebugLog.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ControlFlowInterfaces
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ControlFlowInterfaces.cpp.inc"

SuccessorOperands::SuccessorOperands(MutableOperandRange forwardedOperands)
    : producedOperandCount(0), forwardedOperands(std::move(forwardedOperands)) {
}

SuccessorOperands::SuccessorOperands(unsigned int producedOperandCount,
                                     MutableOperandRange forwardedOperands)
    : producedOperandCount(producedOperandCount),
      forwardedOperands(std::move(forwardedOperands)) {}

//===----------------------------------------------------------------------===//
// BranchOpInterface
//===----------------------------------------------------------------------===//

/// Returns the `BlockArgument` corresponding to operand `operandIndex` in some
/// successor if 'operandIndex' is within the range of 'operands', or
/// std::nullopt if `operandIndex` isn't a successor operand index.
std::optional<BlockArgument>
detail::getBranchSuccessorArgument(const SuccessorOperands &operands,
                                   unsigned operandIndex, Block *successor) {
  LDBG() << "Getting branch successor argument for operand index "
         << operandIndex << " in successor block";

  OperandRange forwardedOperands = operands.getForwardedOperands();
  // Check that the operands are valid.
  if (forwardedOperands.empty()) {
    LDBG() << "No forwarded operands, returning nullopt";
    return std::nullopt;
  }

  // Check to ensure that this operand is within the range.
  unsigned operandsStart = forwardedOperands.getBeginOperandIndex();
  if (operandIndex < operandsStart ||
      operandIndex >= (operandsStart + forwardedOperands.size())) {
    LDBG() << "Operand index " << operandIndex << " out of range ["
           << operandsStart << ", "
           << (operandsStart + forwardedOperands.size())
           << "), returning nullopt";
    return std::nullopt;
  }

  // Index the successor.
  unsigned argIndex =
      operands.getProducedOperandCount() + operandIndex - operandsStart;
  LDBG() << "Computed argument index " << argIndex << " for successor block";
  return successor->getArgument(argIndex);
}

/// Verify that the given operands match those of the given successor block.
LogicalResult
detail::verifyBranchSuccessorOperands(Operation *op, unsigned succNo,
                                      const SuccessorOperands &operands) {
  LDBG() << "Verifying branch successor operands for successor #" << succNo
         << " in operation " << op->getName();

  // Check the count.
  unsigned operandCount = operands.size();
  Block *destBB = op->getSuccessor(succNo);
  LDBG() << "Branch has " << operandCount << " operands, target block has "
         << destBB->getNumArguments() << " arguments";

  if (operandCount != destBB->getNumArguments())
    return op->emitError() << "branch has " << operandCount
                           << " operands for successor #" << succNo
                           << ", but target block has "
                           << destBB->getNumArguments();

  // Check the types.
  LDBG() << "Checking type compatibility for "
         << (operandCount - operands.getProducedOperandCount())
         << " forwarded operands";
  for (unsigned i = operands.getProducedOperandCount(); i != operandCount;
       ++i) {
    Type operandType = operands[i].getType();
    Type argType = destBB->getArgument(i).getType();
    LDBG() << "Checking type compatibility: operand type " << operandType
           << " vs argument type " << argType;

    if (!cast<BranchOpInterface>(op).areTypesCompatible(operandType, argType))
      return op->emitError() << "type mismatch for bb argument #" << i
                             << " of successor #" << succNo;
  }

  LDBG() << "Branch successor operand verification successful";
  return success();
}

//===----------------------------------------------------------------------===//
// WeightedBranchOpInterface
//===----------------------------------------------------------------------===//

static LogicalResult verifyWeights(Operation *op,
                                   llvm::ArrayRef<int32_t> weights,
                                   std::size_t expectedWeightsNum,
                                   llvm::StringRef weightAnchorName,
                                   llvm::StringRef weightRefName) {
  if (weights.empty())
    return success();

  if (weights.size() != expectedWeightsNum)
    return op->emitError() << "expects number of " << weightAnchorName
                           << " weights to match number of " << weightRefName
                           << ": " << weights.size() << " vs "
                           << expectedWeightsNum;

  if (llvm::all_of(weights, [](int32_t value) { return value == 0; }))
    return op->emitError() << "branch weights cannot all be zero";

  return success();
}

LogicalResult detail::verifyBranchWeights(Operation *op) {
  llvm::ArrayRef<int32_t> weights =
      cast<WeightedBranchOpInterface>(op).getWeights();
  return verifyWeights(op, weights, op->getNumSuccessors(), "branch",
                       "successors");
}

//===----------------------------------------------------------------------===//
// WeightedRegionBranchOpInterface
//===----------------------------------------------------------------------===//

LogicalResult detail::verifyRegionBranchWeights(Operation *op) {
  llvm::ArrayRef<int32_t> weights =
      cast<WeightedRegionBranchOpInterface>(op).getWeights();
  return verifyWeights(op, weights, op->getNumRegions(), "region", "regions");
}

//===----------------------------------------------------------------------===//
// RegionBranchOpInterface
//===----------------------------------------------------------------------===//

/// Verify that types match along control flow edges described the given op.
LogicalResult detail::verifyRegionBranchOpInterface(Operation *op) {
  auto regionInterface = cast<RegionBranchOpInterface>(op);

  // Verify all control flow edges from region branch points to region
  // successors.
  SmallVector<RegionBranchPoint> regionBranchPoints =
      regionInterface.getAllRegionBranchPoints();
  for (const RegionBranchPoint &branchPoint : regionBranchPoints) {
    SmallVector<RegionSuccessor> successors;
    regionInterface.getSuccessorRegions(branchPoint, successors);
    for (const RegionSuccessor &successor : successors) {
      // Helper function that print the region branch point and the region
      // successor.
      auto emitRegionEdgeError = [&]() {
        InFlightDiagnostic diag =
            regionInterface->emitOpError("along control flow edge from ");
        if (branchPoint.isParent()) {
          diag << "parent";
          diag.attachNote(op->getLoc()) << "region branch point";
        } else {
          diag << "Operation "
               << branchPoint.getTerminatorPredecessorOrNull()->getName();
          diag.attachNote(
              branchPoint.getTerminatorPredecessorOrNull()->getLoc())
              << "region branch point";
        }
        diag << " to ";
        if (Region *region = successor.getSuccessor()) {
          diag << "Region #" << region->getRegionNumber();
        } else {
          diag << "parent";
        }
        return diag;
      };

      // Verify number of successor operands and successor inputs.
      OperandRange succOperands =
          regionInterface.getSuccessorOperands(branchPoint, successor);
      ValueRange succInputs = regionInterface.getSuccessorInputs(successor);
      if (succOperands.size() != succInputs.size()) {
        return emitRegionEdgeError()
               << ": region branch point has " << succOperands.size()
               << " operands, but region successor needs " << succInputs.size()
               << " inputs";
      }

      // Verify that the types are compatible.
      TypeRange succInputTypes = succInputs.getTypes();
      TypeRange succOperandTypes = succOperands.getTypes();
      for (const auto &typesIdx :
           llvm::enumerate(llvm::zip(succOperandTypes, succInputTypes))) {
        Type succOperandType = std::get<0>(typesIdx.value());
        Type succInputType = std::get<1>(typesIdx.value());
        if (!regionInterface.areTypesCompatible(succOperandType, succInputType))
          return emitRegionEdgeError()
                 << ": successor operand type #" << typesIdx.index() << " "
                 << succOperandType << " should match successor input type #"
                 << typesIdx.index() << " " << succInputType;
      }
    }
  }
  return success();
}

/// Stop condition for `traverseRegionGraph`. The traversal is interrupted if
/// this function returns "true" for a successor region. The first parameter is
/// the successor region. The second parameter indicates all already visited
/// regions.
using StopConditionFn = function_ref<bool(Region *, ArrayRef<bool> visited)>;

/// Traverse the region graph starting at `begin`. The traversal is interrupted
/// if `stopCondition` evaluates to "true" for a successor region. In that case,
/// this function returns "true". Otherwise, if the traversal was not
/// interrupted, this function returns "false".
static bool traverseRegionGraph(Region *begin,
                                StopConditionFn stopConditionFn) {
  auto op = cast<RegionBranchOpInterface>(begin->getParentOp());
  LDBG() << "Starting region graph traversal from region #"
         << begin->getRegionNumber() << " in operation " << op->getName();

  SmallVector<bool> visited(op->getNumRegions(), false);
  visited[begin->getRegionNumber()] = true;
  LDBG() << "Initialized visited array with " << op->getNumRegions()
         << " regions";

  // Retrieve all successors of the region and enqueue them in the worklist.
  SmallVector<Region *> worklist;
  auto enqueueAllSuccessors = [&](Region *region) {
    LDBG() << "Enqueuing successors for region #" << region->getRegionNumber();
    SmallVector<Attribute> operandAttributes(op->getNumOperands());
    for (Block &block : *region) {
      if (block.empty())
        continue;
      auto terminator =
          dyn_cast<RegionBranchTerminatorOpInterface>(block.back());
      if (!terminator)
        continue;
      SmallVector<RegionSuccessor> successors;
      operandAttributes.resize(terminator->getNumOperands());
      terminator.getSuccessorRegions(operandAttributes, successors);
      LDBG() << "Found " << successors.size()
             << " successors from terminator in block";
      for (RegionSuccessor successor : successors) {
        if (!successor.isParent()) {
          worklist.push_back(successor.getSuccessor());
          LDBG() << "Added region #"
                 << successor.getSuccessor()->getRegionNumber()
                 << " to worklist";
        } else {
          LDBG() << "Skipping parent successor";
        }
      }
    }
  };
  enqueueAllSuccessors(begin);
  LDBG() << "Initial worklist size: " << worklist.size();

  // Process all regions in the worklist via DFS.
  while (!worklist.empty()) {
    Region *nextRegion = worklist.pop_back_val();
    LDBG() << "Processing region #" << nextRegion->getRegionNumber()
           << " from worklist (remaining: " << worklist.size() << ")";

    if (stopConditionFn(nextRegion, visited)) {
      LDBG() << "Stop condition met for region #"
             << nextRegion->getRegionNumber() << ", returning true";
      return true;
    }
    if (!nextRegion->getParentOp()) {
      llvm::errs() << "Region " << *nextRegion << " has no parent op\n";
      return false;
    }
    if (visited[nextRegion->getRegionNumber()]) {
      LDBG() << "Region #" << nextRegion->getRegionNumber()
             << " already visited, skipping";
      continue;
    }
    visited[nextRegion->getRegionNumber()] = true;
    LDBG() << "Marking region #" << nextRegion->getRegionNumber()
           << " as visited";
    enqueueAllSuccessors(nextRegion);
  }

  LDBG() << "Traversal completed, returning false";
  return false;
}

/// Return `true` if region `r` is reachable from region `begin` according to
/// the RegionBranchOpInterface (by taking a branch).
static bool isRegionReachable(Region *begin, Region *r) {
  assert(begin->getParentOp() == r->getParentOp() &&
         "expected that both regions belong to the same op");
  return traverseRegionGraph(begin,
                             [&](Region *nextRegion, ArrayRef<bool> visited) {
                               // Interrupt traversal if `r` was reached.
                               return nextRegion == r;
                             });
}

/// Return `true` if `a` and `b` are in mutually exclusive regions.
///
/// 1. Find the first common of `a` and `b` (ancestor) that implements
///    RegionBranchOpInterface.
/// 2. Determine the regions `regionA` and `regionB` in which `a` and `b` are
///    contained.
/// 3. Check if `regionA` and `regionB` are mutually exclusive. They are
///    mutually exclusive if they are not reachable from each other as per
///    RegionBranchOpInterface::getSuccessorRegions.
bool mlir::insideMutuallyExclusiveRegions(Operation *a, Operation *b) {
  LDBG() << "Checking if operations are in mutually exclusive regions: "
         << a->getName() << " and " << b->getName();

  assert(a && "expected non-empty operation");
  assert(b && "expected non-empty operation");

  auto branchOp = a->getParentOfType<RegionBranchOpInterface>();
  while (branchOp) {
    LDBG() << "Checking branch operation " << branchOp->getName();

    // Check if b is inside branchOp. (We already know that a is.)
    if (!branchOp->isProperAncestor(b)) {
      LDBG() << "Operation b is not inside branchOp, checking next ancestor";
      // Check next enclosing RegionBranchOpInterface.
      branchOp = branchOp->getParentOfType<RegionBranchOpInterface>();
      continue;
    }

    LDBG() << "Both operations are inside branchOp, finding their regions";

    // b is contained in branchOp. Retrieve the regions in which `a` and `b`
    // are contained.
    Region *regionA = nullptr, *regionB = nullptr;
    for (Region &r : branchOp->getRegions()) {
      if (r.findAncestorOpInRegion(*a)) {
        assert(!regionA && "already found a region for a");
        regionA = &r;
        LDBG() << "Found region #" << r.getRegionNumber() << " for operation a";
      }
      if (r.findAncestorOpInRegion(*b)) {
        assert(!regionB && "already found a region for b");
        regionB = &r;
        LDBG() << "Found region #" << r.getRegionNumber() << " for operation b";
      }
    }
    assert(regionA && regionB && "could not find region of op");

    LDBG() << "Region A: #" << regionA->getRegionNumber() << ", Region B: #"
           << regionB->getRegionNumber();

    // `a` and `b` are in mutually exclusive regions if both regions are
    // distinct and neither region is reachable from the other region.
    bool regionsAreDistinct = (regionA != regionB);
    bool aNotReachableFromB = !isRegionReachable(regionA, regionB);
    bool bNotReachableFromA = !isRegionReachable(regionB, regionA);

    LDBG() << "Regions distinct: " << regionsAreDistinct
           << ", A not reachable from B: " << aNotReachableFromB
           << ", B not reachable from A: " << bNotReachableFromA;

    bool mutuallyExclusive =
        regionsAreDistinct && aNotReachableFromB && bNotReachableFromA;
    LDBG() << "Operations are mutually exclusive: " << mutuallyExclusive;

    return mutuallyExclusive;
  }

  // Could not find a common RegionBranchOpInterface among a's and b's
  // ancestors.
  LDBG() << "No common RegionBranchOpInterface found, operations are not "
            "mutually exclusive";
  return false;
}

bool RegionBranchOpInterface::isRepetitiveRegion(unsigned index) {
  LDBG() << "Checking if region #" << index << " is repetitive in operation "
         << getOperation()->getName();

  Region *region = &getOperation()->getRegion(index);
  bool isRepetitive = isRegionReachable(region, region);

  LDBG() << "Region #" << index << " is repetitive: " << isRepetitive;
  return isRepetitive;
}

bool RegionBranchOpInterface::hasLoop() {
  LDBG() << "Checking if operation " << getOperation()->getName()
         << " has loops";

  SmallVector<RegionSuccessor> entryRegions;
  getSuccessorRegions(RegionBranchPoint::parent(), entryRegions);
  LDBG() << "Found " << entryRegions.size() << " entry regions";

  for (RegionSuccessor successor : entryRegions) {
    if (!successor.isParent()) {
      LDBG() << "Checking entry region #"
             << successor.getSuccessor()->getRegionNumber() << " for loops";

      bool hasLoop =
          traverseRegionGraph(successor.getSuccessor(),
                              [](Region *nextRegion, ArrayRef<bool> visited) {
                                // Interrupt traversal if the region was already
                                // visited.
                                return visited[nextRegion->getRegionNumber()];
                              });

      if (hasLoop) {
        LDBG() << "Found loop in entry region #"
               << successor.getSuccessor()->getRegionNumber();
        return true;
      }
    } else {
      LDBG() << "Skipping parent successor";
    }
  }

  LDBG() << "No loops found in operation";
  return false;
}

OperandRange
RegionBranchOpInterface::getSuccessorOperands(RegionBranchPoint src,
                                              RegionSuccessor dest) {
  if (src.isParent())
    return getEntrySuccessorOperands(dest);
  return src.getTerminatorPredecessorOrNull().getSuccessorOperands(dest);
}

static MutableArrayRef<OpOperand> operandsToOpOperands(OperandRange &operands) {
  return MutableArrayRef<OpOperand>(operands.getBase(), operands.size());
}

static void
getSuccessorOperandInputMapping(RegionBranchOpInterface branchOp,
                                RegionBranchSuccessorMapping &mapping,
                                RegionBranchPoint src) {
  SmallVector<RegionSuccessor> successors;
  branchOp.getSuccessorRegions(src, successors);
  for (RegionSuccessor dst : successors) {
    OperandRange operands = branchOp.getSuccessorOperands(src, dst);
    assert(operands.size() == branchOp.getSuccessorInputs(dst).size() &&
           "expected the same number of operands and inputs");
    for (const auto &[operand, input] : llvm::zip_equal(
             operandsToOpOperands(operands), branchOp.getSuccessorInputs(dst)))
      mapping[&operand].push_back(input);
  }
}
void RegionBranchOpInterface::getSuccessorOperandInputMapping(
    RegionBranchSuccessorMapping &mapping,
    std::optional<RegionBranchPoint> src) {
  if (src.has_value()) {
    ::getSuccessorOperandInputMapping(*this, mapping, src.value());
  } else {
    // No region branch point specified: populate the mapping for all possible
    // region branch points.
    for (RegionBranchPoint branchPoint : getAllRegionBranchPoints())
      ::getSuccessorOperandInputMapping(*this, mapping, branchPoint);
  }
}

static RegionBranchInverseSuccessorMapping invertRegionBranchSuccessorMapping(
    const RegionBranchSuccessorMapping &operandToInputs) {
  RegionBranchInverseSuccessorMapping inputToOperands;
  for (const auto &[operand, inputs] : operandToInputs) {
    for (Value input : inputs)
      inputToOperands[input].push_back(operand);
  }
  return inputToOperands;
}

void RegionBranchOpInterface::getSuccessorInputOperandMapping(
    RegionBranchInverseSuccessorMapping &mapping) {
  RegionBranchSuccessorMapping operandToInputs;
  getSuccessorOperandInputMapping(operandToInputs);
  mapping = invertRegionBranchSuccessorMapping(operandToInputs);
}

SmallVector<RegionBranchPoint>
RegionBranchOpInterface::getAllRegionBranchPoints() {
  SmallVector<RegionBranchPoint> branchPoints;
  branchPoints.push_back(RegionBranchPoint::parent());
  for (Region &region : getOperation()->getRegions()) {
    for (Block &block : region) {
      if (block.empty())
        continue;
      if (auto terminator =
              dyn_cast<RegionBranchTerminatorOpInterface>(block.back()))
        branchPoints.push_back(RegionBranchPoint(terminator));
    }
  }
  return branchPoints;
}

Region *mlir::getEnclosingRepetitiveRegion(Operation *op) {
  LDBG() << "Finding enclosing repetitive region for operation "
         << op->getName();

  while (Region *region = op->getParentRegion()) {
    LDBG() << "Checking region #" << region->getRegionNumber()
           << " in operation " << region->getParentOp()->getName();

    op = region->getParentOp();
    if (auto branchOp = dyn_cast<RegionBranchOpInterface>(op)) {
      LDBG()
          << "Found RegionBranchOpInterface, checking if region is repetitive";
      if (branchOp.isRepetitiveRegion(region->getRegionNumber())) {
        LDBG() << "Found repetitive region #" << region->getRegionNumber();
        return region;
      }
    } else {
      LDBG() << "Parent operation does not implement RegionBranchOpInterface";
    }
  }

  LDBG() << "No enclosing repetitive region found";
  return nullptr;
}

Region *mlir::getEnclosingRepetitiveRegion(Value value) {
  LDBG() << "Finding enclosing repetitive region for value";

  Region *region = value.getParentRegion();
  while (region) {
    LDBG() << "Checking region #" << region->getRegionNumber()
           << " in operation " << region->getParentOp()->getName();

    Operation *op = region->getParentOp();
    if (auto branchOp = dyn_cast<RegionBranchOpInterface>(op)) {
      LDBG()
          << "Found RegionBranchOpInterface, checking if region is repetitive";
      if (branchOp.isRepetitiveRegion(region->getRegionNumber())) {
        LDBG() << "Found repetitive region #" << region->getRegionNumber();
        return region;
      }
    } else {
      LDBG() << "Parent operation does not implement RegionBranchOpInterface";
    }
    region = op->getParentRegion();
  }

  LDBG() << "No enclosing repetitive region found for value";
  return nullptr;
}

/// Return "true" if `a` can be used in lieu of `b`, where `b` is a region
/// successor input and `a` is a "reachable value" of `b`. Reachable values
/// are successor operand values that are (maybe transitively) forwarded to
/// `b`.
static bool isDefinedBefore(Operation *regionBranchOp, Value a, Value b) {
  assert((b.getDefiningOp() == regionBranchOp ||
          b.getParentRegion()->getParentOp() == regionBranchOp) &&
         "b must be a region successor input");

  // Case 1: `a` is defined inside of the region branch op. `a` must be
  // directly nested in the region branch op. Otherwise, it could not have
  // been among the reachable values for a region successor input.
  if (a.getParentRegion()->getParentOp() == regionBranchOp) {
    // Case 1.1: If `b` is a result of the region branch op, `a` is not in
    // scope for `b`.
    // Example:
    // %b = region_op({
    // ^bb0(%a1: ...):
    //   %a2 = ...
    // })
    if (isa<OpResult>(b))
      return false;

    // Case 1.2: `b` is an entry block argument of a region. `a` is in scope
    // for `b` only if it is also an entry block argument of the same region.
    // Example:
    // region_op({
    // ^bb0(%b: ..., %a: ...):
    //   ...
    // })
    assert(isa<BlockArgument>(b) && "b must be a block argument");
    return isa<BlockArgument>(a) && cast<BlockArgument>(a).getOwner() ==
                                        cast<BlockArgument>(b).getOwner();
  }

  // Case 2: `a` is defined outside of the region branch op. In that case, we
  // can safely assume that `a` was defined before `b`. Otherwise, it could not
  // be among the reachable values for a region successor input.
  // Example:
  // {   <- %a1 parent region begins here.
  // ^bb0(%a1: ...):
  //   %a2 = ...
  //   %b1 = reigon_op({
  //   ^bb1(%b2: ...):
  //     ...
  //   })
  // }
  return true;
}

/// Compute all non-successor-input values that a successor input could have
/// based on the given successor input to successor operand mapping.
///
/// Example 1:
/// %r = scf.if ... {
///   scf.yield %a : ...
/// } else {
///   scf.yield %b : ...
/// }
/// reachableValues(%r) = {%a, %b}
///
/// Example 2:
/// %r = scf.for ... iter_args(%arg0 = %0) -> ... {
///   scf.yield %arg0 : ...
/// }
/// reachableValues(%arg0) = {%0}
/// reachableValues(%r) = {%0}
///
/// Example 3:
/// %r = scf.for ... iter_args(%arg0 = %0) -> ... {
///   ...
///   scf.yield %1 : ...
/// }
/// reachableValues(%arg0) = {%0, %1}
/// reachableValues(%r) = {%0, %1}
static llvm::SmallDenseSet<Value> computeReachableValuesFromSuccessorInput(
    Value value, const RegionBranchInverseSuccessorMapping &inputToOperands) {
  assert(inputToOperands.contains(value) && "value must be a successor input");
  // Starting with the given value, trace back all predecessor values (i.e.,
  // preceding successor operands) and add them to the set of reachable values.
  // If the successor operand is again a successor input, do not add it to
  // result set, but instead continue the traversal.
  llvm::SmallDenseSet<Value> reachableValues;
  llvm::SmallDenseSet<Value> visited;
  SmallVector<Value> worklist;
  worklist.push_back(value);
  while (!worklist.empty()) {
    Value next = worklist.pop_back_val();
    auto it = inputToOperands.find(next);
    if (it == inputToOperands.end()) {
      reachableValues.insert(next);
      continue;
    }
    for (OpOperand *operand : it->second)
      if (visited.insert(operand->get()).second)
        worklist.push_back(operand->get());
  }
  // Note: The result does not contain any successor inputs. (Therefore,
  // `value` is also guaranteed to be excluded.)
  return reachableValues;
}

namespace {
/// Try to make successor inputs dead by replacing their uses with values that
/// are not successor inputs. This pattern enables additional canonicalization
/// opportunities for RemoveDeadRegionBranchOpSuccessorInputs.
///
/// Example:
///
/// %r0, %r1 = scf.for ... iter_args(%arg0 = %0, %arg1 = %1) -> ... {
///   scf.yield %arg1, %arg1 : ...
/// }
/// use(%r0, %r1)
///
/// reachableValues(%r0) = {%0, %1}
/// reachableValues(%r1) = {%1} ==> replace uses of %r1 with %1.
/// reachableValues(%arg0) = {%0, %1}
/// reachableValues(%arg1) = {%1} ==> replace uses of %arg1 with %1.
///
/// IR after pattern application:
///
/// %r0, %r1 = scf.for ... iter_args(%arg0 = %0, %arg1 = %1) -> ... {
///   scf.yield %1, %1 : ...
/// }
/// use(%r0, %1)
///
/// Note that %r1 and %arg1 are dead now. The IR can now be further
/// canonicalized by RemoveDeadRegionBranchOpSuccessorInputs.
struct MakeRegionBranchOpSuccessorInputsDead : public RewritePattern {
  MakeRegionBranchOpSuccessorInputsDead(MLIRContext *context, StringRef name,
                                        PatternBenefit benefit = 1)
      : RewritePattern(name, benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    assert(!op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
           "isolated-from-above ops are not supported");

    // Compute the mapping of successor inputs to successor operands.
    auto regionBranchOp = cast<RegionBranchOpInterface>(op);
    RegionBranchInverseSuccessorMapping inputToOperands;
    regionBranchOp.getSuccessorInputOperandMapping(inputToOperands);

    // Try to replace the uses of each successor input one-by-one.
    bool changed = false;
    for (Value value : inputToOperands.keys()) {
      // Nothing to do for successor inputs that are already dead.
      if (value.use_empty())
        continue;
      // Nothing to do for successor inputs that may have multiple reachable
      // values.
      llvm::SmallDenseSet<Value> reachableValues =
          computeReachableValuesFromSuccessorInput(value, inputToOperands);
      if (reachableValues.size() != 1)
        continue;
      assert(*reachableValues.begin() != value &&
             "successor inputs are supposed to be excluded");
      // Do not replace `value` with the found reachable value if doing so
      // would violate dominance. Example:
      // %r = scf.execute_region ... {
      //   %a = ...
      //   scf.yield %a : ...
      // }
      // use(%r)
      // In the above example, reachableValues(%r) = {%a}, but %a cannot be
      // used as a replacement for %r due to dominance / scope.
      if (!isDefinedBefore(regionBranchOp, *reachableValues.begin(), value))
        continue;
      rewriter.replaceAllUsesWith(value, *reachableValues.begin());
      changed = true;
    }
    return success(changed);
  }
};

/// Lookup a bit vector in the given mapping (DenseMap). If the key was not
/// found, create a new bit vector with the given size and initialize it with
/// false.
template <typename MappingTy, typename KeyTy>
static BitVector &lookupOrCreateBitVector(MappingTy &mapping, KeyTy key,
                                          unsigned size) {
  return mapping.try_emplace(key, size, false).first->second;
}

/// Compute tied successor inputs. Tied successor inputs are successor inputs
/// that come as a set. If you erase one value from a set, you must erase all
/// values from the set. Otherwise, the op would become structurally invalid.
/// Each successor input appears in exactly one set.
///
/// Example:
/// %r0, %r1 = scf.for ... iter_args(%arg0 = %0, %arg1 = %1) -> ... {
///   ...
/// }
/// There are two sets: {{%r0, %arg0}, {%r1, %arg1}}.
static llvm::EquivalenceClasses<Value> computeTiedSuccessorInputs(
    const RegionBranchSuccessorMapping &operandToInputs) {
  llvm::EquivalenceClasses<Value> tiedSuccessorInputs;
  for (const auto &[operand, inputs] : operandToInputs) {
    assert(!inputs.empty() && "expected non-empty inputs");
    Value firstInput = inputs.front();
    tiedSuccessorInputs.insert(firstInput);
    for (Value nextInput : llvm::drop_begin(inputs)) {
      // As we explore more successor operand to successor input mappings,
      // existing sets may get merged.
      tiedSuccessorInputs.unionSets(firstInput, nextInput);
    }
  }
  return tiedSuccessorInputs;
}

/// Remove dead successor inputs from region branch ops. A successor input is
/// dead if it has no uses. Successor inputs come in sets of tied values: if
/// you remove one value from a set, you must remove all values from the set.
/// Furthermore, successor operands must also be removed. (Op operands are not
/// part of the set, but the set is built based on the successor operand to
/// successor input mapping.)
///
/// Example 1:
/// %r0, %r1 = scf.for ... iter_args(%arg0 = %0, %arg1 = %1) -> ... {
///   scf.yield %0, %arg1 : ...
/// }
/// use(%0, %1)
///
/// There are two sets: {{%r0, %arg0}, {%r1, %arg1}}. All values in the first
/// set are dead, so %arg0 and %r0 can be removed, but not %r1 and %arg1. The
/// resulting IR is as follows:
///
/// %r1 = scf.for ... iter_args(%arg1 = %1) -> ... {
///   scf.yield %arg1 : ...
/// }
/// use(%0, %1)
///
/// Example 2:
/// %r0, %r1 = scf.while (%arg0 = %0) {
///   scf.condition(...) %arg0, %arg0 : ...
/// } do {
/// ^bb0(%arg1: ..., %arg2: ...):
///   scf.yield %arg1 : ...
/// }
/// There are three sets: {{%r0, %arg1}, {%r1, %arg2}, {%r0}}.
///
/// Example 3:
/// %r1, %r2 = scf.if ... {
///   scf.yield %0, %1 : ...
/// } else {
///   scf.yield %2, %3 : ...
/// }
/// There are two sets: {{%r1}, {%r2}}. Each set has one value, so there each
/// value can be removed independently of the other values.
struct RemoveDeadRegionBranchOpSuccessorInputs : public RewritePattern {
  RemoveDeadRegionBranchOpSuccessorInputs(MLIRContext *context, StringRef name,
                                          PatternBenefit benefit = 1)
      : RewritePattern(name, benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    assert(!op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
           "isolated-from-above ops are not supported");

    // Compute tied values: values that must come as a set. If you remove one,
    // you must remove all. If a successor op operand is forwarded to two
    // successor inputs %a and %b, both %a and %b are in the same set.
    auto regionBranchOp = cast<RegionBranchOpInterface>(op);
    RegionBranchSuccessorMapping operandToInputs;
    regionBranchOp.getSuccessorOperandInputMapping(operandToInputs);
    llvm::EquivalenceClasses<Value> tiedSuccessorInputs =
        computeTiedSuccessorInputs(operandToInputs);

    // Determine which values to remove and group them by block and operation.
    SmallVector<Value> valuesToRemove;
    DenseMap<Block *, BitVector> blockArgsToRemove;
    BitVector resultsToRemove(regionBranchOp->getNumResults(), false);
    // Iterate over all sets of tied successor inputs.
    for (auto it = tiedSuccessorInputs.begin(), e = tiedSuccessorInputs.end();
         it != e; ++it) {
      if (!(*it)->isLeader())
        continue;

      // Value can be removed if it is dead and all other tied values are also
      // dead.
      bool allDead = true;
      for (auto memberIt = tiedSuccessorInputs.member_begin(**it);
           memberIt != tiedSuccessorInputs.member_end(); ++memberIt) {
        // Iterate over all values in the set and check their liveness.
        if (!memberIt->use_empty()) {
          allDead = false;
          break;
        }
      }
      if (!allDead)
        continue;

      // The entire set is dead. Group values by block and operation to
      // simplify removal.
      for (auto memberIt = tiedSuccessorInputs.member_begin(**it);
           memberIt != tiedSuccessorInputs.member_end(); ++memberIt) {
        if (auto arg = dyn_cast<BlockArgument>(*memberIt)) {
          // Set blockArgsToRemove[block][arg_number] = true.
          BitVector &vector =
              lookupOrCreateBitVector(blockArgsToRemove, arg.getOwner(),
                                      arg.getOwner()->getNumArguments());
          vector.set(arg.getArgNumber());
        } else {
          // Set resultsToRemove[result_number] = true.
          OpResult result = cast<OpResult>(*memberIt);
          assert(result.getDefiningOp() == regionBranchOp &&
                 "result must be a region branch op result");
          resultsToRemove.set(result.getResultNumber());
        }
        valuesToRemove.push_back(*memberIt);
      }
    }

    if (valuesToRemove.empty())
      return rewriter.notifyMatchFailure(op, "no values to remove");

    // Find operands that must be removed together with the values.
    RegionBranchInverseSuccessorMapping inputsToOperands =
        invertRegionBranchSuccessorMapping(operandToInputs);
    DenseMap<Operation *, llvm::BitVector> operandsToRemove;
    for (Value value : valuesToRemove) {
      for (OpOperand *operand : inputsToOperands[value]) {
        // Set operandsToRemove[op][operand_number] = true.
        BitVector &vector =
            lookupOrCreateBitVector(operandsToRemove, operand->getOwner(),
                                    operand->getOwner()->getNumOperands());
        vector.set(operand->getOperandNumber());
      }
    }

    // Erase operands.
    for (auto &pair : operandsToRemove) {
      Operation *op = pair.first;
      BitVector &operands = pair.second;
      rewriter.modifyOpInPlace(op, [&]() { op->eraseOperands(operands); });
    }

    // Erase block arguments.
    for (auto &pair : blockArgsToRemove) {
      Block *block = pair.first;
      BitVector &blockArg = pair.second;
      rewriter.modifyOpInPlace(block->getParentOp(),
                               [&]() { block->eraseArguments(blockArg); });
    }

    // Erase op results.
    if (resultsToRemove.any())
      rewriter.eraseOpResults(regionBranchOp, resultsToRemove);

    return success();
  }
};

/// Return "true" if the two values are owned by the same operation or block.
static bool haveSameOwner(Value a, Value b) {
  void *aOwner, *bOwner;
  if (auto arg = dyn_cast<BlockArgument>(a))
    aOwner = arg.getOwner();
  else
    aOwner = a.getDefiningOp();
  if (auto arg = dyn_cast<BlockArgument>(b))
    bOwner = arg.getOwner();
  else
    bOwner = b.getDefiningOp();
  return aOwner == bOwner;
}

/// Get the block argument or op result number of the given value.
static unsigned getArgOrResultNumber(Value value) {
  if (auto opResult = llvm::dyn_cast<OpResult>(value))
    return opResult.getResultNumber();
  return llvm::cast<BlockArgument>(value).getArgNumber();
}

/// Find duplicate successor inputs and make all dead except for one. Two
/// successor inputs are "duplicate" if their corresponding successor operands
/// have the same values. This pattern enables additional canonicalization
/// opportunities for RemoveDeadRegionBranchOpSuccessorInputs.
///
/// Example:
/// %r0, %r1 = scf.for ... iter_args(%arg0 = %0, %arg1 = %0) -> ... {
///   use(%arg0, %arg1)
///   ...
///   scf.yield %x, %x : ...
/// }
/// use(%r0, %r1)
///
/// Operands of successor input %r0: [%0, %x]
/// Operands of successor input %r1: [%0, %x] ==> DUPLICATE!
/// Replace %r1 with %r0.
///
/// Operands of successor input %arg0: [%0, %x]
/// Operands of successor input %arg1: [%0, %x] ==> DUPLICATE!
/// Replace %arg1 with %arg0. (We have to make sure that we make same decision
/// as for the other tied successor inputs above. Otherwise, a set of tied
/// successor inputs may not become entirely dead.)
///
/// The resulting IR is as follows:
/// %r0, %r1 = scf.for ... iter_args(%arg0 = %0, %arg1 = %0) -> ... {
///   use(%arg0, %arg0)
///   ...
///   scf.yield %x, %x : ...
/// }
/// use(%r0, %r0)  // Note: We don't want use(%r1, %r1), which is also correct,
///                // but does not help with further canonicalizations.
struct RemoveDuplicateSuccessorInputUses : public RewritePattern {
  RemoveDuplicateSuccessorInputUses(MLIRContext *context, StringRef name,
                                    PatternBenefit benefit = 1)
      : RewritePattern(name, benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    assert(!op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
           "isolated-from-above ops are not supported");

    // Collect all successor inputs and sort them. When dropping the uses of a
    // successor input, we'd like to also drop the uses of the same tied
    // successor inputs. Otherwise, a set of tied successor inputs may not
    // become entirely dead, which is required for
    // RemoveDeadRegionBranchOpSuccessorInputs to be able to erase them.
    // (Sorting is not required for correctness.)
    auto regionBranchOp = cast<RegionBranchOpInterface>(op);
    RegionBranchInverseSuccessorMapping inputsToOperands;
    regionBranchOp.getSuccessorInputOperandMapping(inputsToOperands);
    SmallVector<Value> inputs = llvm::to_vector(inputsToOperands.keys());
    llvm::sort(inputs, [](Value a, Value b) {
      return getArgOrResultNumber(a) < getArgOrResultNumber(b);
    });

    // Check every distinct pair of successor inputs for duplicates. Replace
    // `input2` with `input1` if they are duplicates.
    bool changed = false;
    unsigned numInputs = inputs.size();
    for (auto i : llvm::seq<unsigned>(0, numInputs)) {
      Value input1 = inputs[i];
      for (auto j : llvm::seq<unsigned>(i + 1, numInputs)) {
        Value input2 = inputs[j];
        // Nothing to do if input2 is already dead.
        if (input2.use_empty())
          continue;
        // Replace only values that belong to the same block / operation.
        // This implies that the two values are either both block arguments or
        // both op results.
        if (!haveSameOwner(input1, input2))
          continue;

        // Gather the predecessor value for each predecessor (region branch
        // point). The two inputs are duplicates if each predecessor forwards
        // the same value.
        llvm::SmallDenseMap<Operation *, Value> operands1, operands2;
        for (OpOperand *operand : inputsToOperands[input1]) {
          assert(!operands1.contains(operand->getOwner()));
          operands1[operand->getOwner()] = operand->get();
        }
        for (OpOperand *operand : inputsToOperands[input2]) {
          assert(!operands2.contains(operand->getOwner()));
          operands2[operand->getOwner()] = operand->get();
        }
        if (operands1 == operands2) {
          rewriter.replaceAllUsesWith(input2, input1);
          changed = true;
        }
      }
    }
    return success(changed);
  }
};
} // namespace

void mlir::populateRegionBranchOpInterfaceCanonicalizationPatterns(
    RewritePatternSet &patterns, StringRef opName, PatternBenefit benefit) {
  patterns.add<MakeRegionBranchOpSuccessorInputsDead,
               RemoveDuplicateSuccessorInputUses,
               RemoveDeadRegionBranchOpSuccessorInputs>(patterns.getContext(),
                                                        opName, benefit);
}
