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
#include "mlir/Interfaces/ControlFlowInterfaces.h"
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

static InFlightDiagnostic &printRegionEdgeName(InFlightDiagnostic &diag,
                                               RegionBranchPoint sourceNo,
                                               RegionSuccessor succRegionNo) {
  diag << "from ";
  if (Operation *op = sourceNo.getTerminatorPredecessorOrNull())
    diag << "Operation " << op->getName();
  else
    diag << "parent operands";

  diag << " to ";
  if (Region *region = succRegionNo.getSuccessor())
    diag << "Region #" << region->getRegionNumber();
  else
    diag << "parent results";
  return diag;
}

/// Verify that types match along all region control flow edges originating from
/// `sourcePoint`. `getInputsTypesForRegion` is a function that returns the
/// types of the inputs that flow to a successor region.
static LogicalResult
verifyTypesAlongAllEdges(RegionBranchOpInterface branchOp,
                         RegionBranchPoint sourcePoint,
                         function_ref<FailureOr<TypeRange>(RegionSuccessor)>
                             getInputsTypesForRegion) {
  SmallVector<RegionSuccessor, 2> successors;
  branchOp.getSuccessorRegions(sourcePoint, successors);

  for (RegionSuccessor &succ : successors) {
    FailureOr<TypeRange> sourceTypes = getInputsTypesForRegion(succ);
    if (failed(sourceTypes))
      return failure();

    TypeRange succInputsTypes = succ.getSuccessorInputs().getTypes();
    if (sourceTypes->size() != succInputsTypes.size()) {
      InFlightDiagnostic diag =
          branchOp->emitOpError("region control flow edge ");
      std::string succStr;
      llvm::raw_string_ostream os(succStr);
      os << succ;
      return printRegionEdgeName(diag, sourcePoint, succ)
             << ": source has " << sourceTypes->size()
             << " operands, but target successor " << os.str() << " needs "
             << succInputsTypes.size();
    }

    for (const auto &typesIdx :
         llvm::enumerate(llvm::zip(*sourceTypes, succInputsTypes))) {
      Type sourceType = std::get<0>(typesIdx.value());
      Type inputType = std::get<1>(typesIdx.value());

      if (!branchOp.areTypesCompatible(sourceType, inputType)) {
        InFlightDiagnostic diag =
            branchOp->emitOpError("along control flow edge ");
        return printRegionEdgeName(diag, sourcePoint, succ)
               << ": source type #" << typesIdx.index() << " " << sourceType
               << " should match input type #" << typesIdx.index() << " "
               << inputType;
      }
    }
  }

  return success();
}

/// Verify that types match along control flow edges described the given op.
LogicalResult detail::verifyTypesAlongControlFlowEdges(Operation *op) {
  auto regionInterface = cast<RegionBranchOpInterface>(op);

  auto inputTypesFromParent = [&](RegionSuccessor successor) -> TypeRange {
    return regionInterface.getEntrySuccessorOperands(successor).getTypes();
  };

  // Verify types along control flow edges originating from the parent.
  if (failed(verifyTypesAlongAllEdges(
          regionInterface, RegionBranchPoint::parent(), inputTypesFromParent)))
    return failure();

  // Verify types along control flow edges originating from each region.
  for (Region &region : op->getRegions()) {
    // Collect all return-like terminators in the region.
    SmallVector<RegionBranchTerminatorOpInterface> regionReturnOps;
    for (Block &block : region)
      if (!block.empty())
        if (auto terminator =
                dyn_cast<RegionBranchTerminatorOpInterface>(block.back()))
          regionReturnOps.push_back(terminator);

    // If there is no return-like terminator, the op itself should verify
    // type consistency.
    if (regionReturnOps.empty())
      continue;

    // Verify types along control flow edges originating from each return-like
    // terminator.
    for (RegionBranchTerminatorOpInterface regionReturnOp : regionReturnOps) {

      auto inputTypesForRegion =
          [&](RegionSuccessor successor) -> FailureOr<TypeRange> {
        OperandRange terminatorOperands =
            regionReturnOp.getSuccessorOperands(successor);
        return TypeRange(terminatorOperands.getTypes());
      };
      if (failed(verifyTypesAlongAllEdges(regionInterface, regionReturnOp,
                                          inputTypesForRegion)))
        return failure();
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
