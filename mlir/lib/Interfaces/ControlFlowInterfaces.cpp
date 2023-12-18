//===- ControlFlowInterfaces.cpp - ControlFlow Interfaces -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

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
  OperandRange forwardedOperands = operands.getForwardedOperands();
  // Check that the operands are valid.
  if (forwardedOperands.empty())
    return std::nullopt;

  // Check to ensure that this operand is within the range.
  unsigned operandsStart = forwardedOperands.getBeginOperandIndex();
  if (operandIndex < operandsStart ||
      operandIndex >= (operandsStart + forwardedOperands.size()))
    return std::nullopt;

  // Index the successor.
  unsigned argIndex =
      operands.getProducedOperandCount() + operandIndex - operandsStart;
  return successor->getArgument(argIndex);
}

/// Verify that the given operands match those of the given successor block.
LogicalResult
detail::verifyBranchSuccessorOperands(Operation *op, unsigned succNo,
                                      const SuccessorOperands &operands) {
  // Check the count.
  unsigned operandCount = operands.size();
  Block *destBB = op->getSuccessor(succNo);
  if (operandCount != destBB->getNumArguments())
    return op->emitError() << "branch has " << operandCount
                           << " operands for successor #" << succNo
                           << ", but target block has "
                           << destBB->getNumArguments();

  // Check the types.
  for (unsigned i = operands.getProducedOperandCount(); i != operandCount;
       ++i) {
    if (!cast<BranchOpInterface>(op).areTypesCompatible(
            operands[i].getType(), destBB->getArgument(i).getType()))
      return op->emitError() << "type mismatch for bb argument #" << i
                             << " of successor #" << succNo;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RegionBranchOpInterface
//===----------------------------------------------------------------------===//

static InFlightDiagnostic &printRegionEdgeName(InFlightDiagnostic &diag,
                                               RegionBranchPoint sourceNo,
                                               RegionBranchPoint succRegionNo) {
  diag << "from ";
  if (Region *region = sourceNo.getRegionOrNull())
    diag << "Region #" << region->getRegionNumber();
  else
    diag << "parent operands";

  diag << " to ";
  if (Region *region = succRegionNo.getRegionOrNull())
    diag << "Region #" << region->getRegionNumber();
  else
    diag << "parent results";
  return diag;
}

/// Verify that types match along all region control flow edges originating from
/// `sourcePoint`. `getInputsTypesForRegion` is a function that returns the
/// types of the inputs that flow to a successor region.
static LogicalResult
verifyTypesAlongAllEdges(Operation *op, RegionBranchPoint sourcePoint,
                         function_ref<FailureOr<TypeRange>(RegionBranchPoint)>
                             getInputsTypesForRegion) {
  auto regionInterface = cast<RegionBranchOpInterface>(op);

  SmallVector<RegionSuccessor, 2> successors;
  regionInterface.getSuccessorRegions(sourcePoint, successors);

  for (RegionSuccessor &succ : successors) {
    FailureOr<TypeRange> sourceTypes = getInputsTypesForRegion(succ);
    if (failed(sourceTypes))
      return failure();

    TypeRange succInputsTypes = succ.getSuccessorInputs().getTypes();
    if (sourceTypes->size() != succInputsTypes.size()) {
      InFlightDiagnostic diag = op->emitOpError(" region control flow edge ");
      return printRegionEdgeName(diag, sourcePoint, succ)
             << ": source has " << sourceTypes->size()
             << " operands, but target successor needs "
             << succInputsTypes.size();
    }

    for (const auto &typesIdx :
         llvm::enumerate(llvm::zip(*sourceTypes, succInputsTypes))) {
      Type sourceType = std::get<0>(typesIdx.value());
      Type inputType = std::get<1>(typesIdx.value());
      if (!regionInterface.areTypesCompatible(sourceType, inputType)) {
        InFlightDiagnostic diag = op->emitOpError(" along control flow edge ");
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

  auto inputTypesFromParent = [&](RegionBranchPoint point) -> TypeRange {
    return regionInterface.getEntrySuccessorOperands(point).getTypes();
  };

  // Verify types along control flow edges originating from the parent.
  if (failed(verifyTypesAlongAllEdges(op, RegionBranchPoint::parent(),
                                      inputTypesFromParent)))
    return failure();

  auto areTypesCompatible = [&](TypeRange lhs, TypeRange rhs) {
    if (lhs.size() != rhs.size())
      return false;
    for (auto types : llvm::zip(lhs, rhs)) {
      if (!regionInterface.areTypesCompatible(std::get<0>(types),
                                              std::get<1>(types))) {
        return false;
      }
    }
    return true;
  };

  // Verify types along control flow edges originating from each region.
  for (Region &region : op->getRegions()) {

    // Since there can be multiple terminators implementing the
    // `RegionBranchTerminatorOpInterface`, all should have the same operand
    // types when passing them to the same region.

    SmallVector<RegionBranchTerminatorOpInterface> regionReturnOps;
    for (Block &block : region)
      if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(
              block.getTerminator()))
        regionReturnOps.push_back(terminator);

    // If there is no return-like terminator, the op itself should verify
    // type consistency.
    if (regionReturnOps.empty())
      continue;

    auto inputTypesForRegion =
        [&](RegionBranchPoint point) -> FailureOr<TypeRange> {
      std::optional<OperandRange> regionReturnOperands;
      for (RegionBranchTerminatorOpInterface regionReturnOp : regionReturnOps) {
        auto terminatorOperands = regionReturnOp.getSuccessorOperands(point);

        if (!regionReturnOperands) {
          regionReturnOperands = terminatorOperands;
          continue;
        }

        // Found more than one ReturnLike terminator. Make sure the operand
        // types match with the first one.
        if (!areTypesCompatible(regionReturnOperands->getTypes(),
                                terminatorOperands.getTypes())) {
          InFlightDiagnostic diag = op->emitOpError(" along control flow edge");
          return printRegionEdgeName(diag, region, point)
                 << " operands mismatch between return-like terminators";
        }
      }

      // All successors get the same set of operand types.
      return TypeRange(regionReturnOperands->getTypes());
    };

    if (failed(verifyTypesAlongAllEdges(op, region, inputTypesForRegion)))
      return failure();
  }

  return success();
}

/// Return `true` if region `r` is reachable from region `begin` according to
/// the RegionBranchOpInterface (by taking a branch).
static bool isRegionReachable(Region *begin, Region *r) {
  assert(begin->getParentOp() == r->getParentOp() &&
         "expected that both regions belong to the same op");
  auto op = cast<RegionBranchOpInterface>(begin->getParentOp());
  SmallVector<bool> visited(op->getNumRegions(), false);
  visited[begin->getRegionNumber()] = true;

  // Retrieve all successors of the region and enqueue them in the worklist.
  SmallVector<Region *> worklist;
  auto enqueueAllSuccessors = [&](Region *region) {
    SmallVector<RegionSuccessor> successors;
    op.getSuccessorRegions(region, successors);
    for (RegionSuccessor successor : successors)
      if (!successor.isParent())
        worklist.push_back(successor.getSuccessor());
  };
  enqueueAllSuccessors(begin);

  // Process all regions in the worklist via DFS.
  while (!worklist.empty()) {
    Region *nextRegion = worklist.pop_back_val();
    if (nextRegion == r)
      return true;
    if (visited[nextRegion->getRegionNumber()])
      continue;
    visited[nextRegion->getRegionNumber()] = true;
    enqueueAllSuccessors(nextRegion);
  }

  return false;
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
  assert(a && "expected non-empty operation");
  assert(b && "expected non-empty operation");

  auto branchOp = a->getParentOfType<RegionBranchOpInterface>();
  while (branchOp) {
    // Check if b is inside branchOp. (We already know that a is.)
    if (!branchOp->isProperAncestor(b)) {
      // Check next enclosing RegionBranchOpInterface.
      branchOp = branchOp->getParentOfType<RegionBranchOpInterface>();
      continue;
    }

    // b is contained in branchOp. Retrieve the regions in which `a` and `b`
    // are contained.
    Region *regionA = nullptr, *regionB = nullptr;
    for (Region &r : branchOp->getRegions()) {
      if (r.findAncestorOpInRegion(*a)) {
        assert(!regionA && "already found a region for a");
        regionA = &r;
      }
      if (r.findAncestorOpInRegion(*b)) {
        assert(!regionB && "already found a region for b");
        regionB = &r;
      }
    }
    assert(regionA && regionB && "could not find region of op");

    // `a` and `b` are in mutually exclusive regions if both regions are
    // distinct and neither region is reachable from the other region.
    return regionA != regionB && !isRegionReachable(regionA, regionB) &&
           !isRegionReachable(regionB, regionA);
  }

  // Could not find a common RegionBranchOpInterface among a's and b's
  // ancestors.
  return false;
}

bool RegionBranchOpInterface::isRepetitiveRegion(unsigned index) {
  Region *region = &getOperation()->getRegion(index);
  return isRegionReachable(region, region);
}

Region *mlir::getEnclosingRepetitiveRegion(Operation *op) {
  while (Region *region = op->getParentRegion()) {
    op = region->getParentOp();
    if (auto branchOp = dyn_cast<RegionBranchOpInterface>(op))
      if (branchOp.isRepetitiveRegion(region->getRegionNumber()))
        return region;
  }
  return nullptr;
}

Region *mlir::getEnclosingRepetitiveRegion(Value value) {
  Region *region = value.getParentRegion();
  while (region) {
    Operation *op = region->getParentOp();
    if (auto branchOp = dyn_cast<RegionBranchOpInterface>(op))
      if (branchOp.isRepetitiveRegion(region->getRegionNumber()))
        return region;
    region = op->getParentRegion();
  }
  return nullptr;
}
