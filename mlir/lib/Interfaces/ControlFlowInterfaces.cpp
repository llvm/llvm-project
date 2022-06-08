//===- ControlFlowInterfaces.cpp - ControlFlow Interfaces -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

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
/// successor if 'operandIndex' is within the range of 'operands', or None if
/// `operandIndex` isn't a successor operand index.
Optional<BlockArgument>
detail::getBranchSuccessorArgument(const SuccessorOperands &operands,
                                   unsigned operandIndex, Block *successor) {
  OperandRange forwardedOperands = operands.getForwardedOperands();
  // Check that the operands are valid.
  if (forwardedOperands.empty())
    return llvm::None;

  // Check to ensure that this operand is within the range.
  unsigned operandsStart = forwardedOperands.getBeginOperandIndex();
  if (operandIndex < operandsStart ||
      operandIndex >= (operandsStart + forwardedOperands.size()))
    return llvm::None;

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

/// Verify that types match along all region control flow edges originating from
/// `sourceNo` (region # if source is a region, llvm::None if source is parent
/// op). `getInputsTypesForRegion` is a function that returns the types of the
/// inputs that flow from `sourceIndex' to the given region, or llvm::None if
/// the exact type match verification is not necessary (e.g., if the Op verifies
/// the match itself).
static LogicalResult
verifyTypesAlongAllEdges(Operation *op, Optional<unsigned> sourceNo,
                         function_ref<Optional<TypeRange>(Optional<unsigned>)>
                             getInputsTypesForRegion) {
  auto regionInterface = cast<RegionBranchOpInterface>(op);

  SmallVector<RegionSuccessor, 2> successors;
  regionInterface.getSuccessorRegions(sourceNo, successors);

  for (RegionSuccessor &succ : successors) {
    Optional<unsigned> succRegionNo;
    if (!succ.isParent())
      succRegionNo = succ.getSuccessor()->getRegionNumber();

    auto printEdgeName = [&](InFlightDiagnostic &diag) -> InFlightDiagnostic & {
      diag << "from ";
      if (sourceNo)
        diag << "Region #" << sourceNo.getValue();
      else
        diag << "parent operands";

      diag << " to ";
      if (succRegionNo)
        diag << "Region #" << succRegionNo.getValue();
      else
        diag << "parent results";
      return diag;
    };

    Optional<TypeRange> sourceTypes = getInputsTypesForRegion(succRegionNo);
    if (!sourceTypes.hasValue())
      continue;

    TypeRange succInputsTypes = succ.getSuccessorInputs().getTypes();
    if (sourceTypes->size() != succInputsTypes.size()) {
      InFlightDiagnostic diag = op->emitOpError(" region control flow edge ");
      return printEdgeName(diag) << ": source has " << sourceTypes->size()
                                 << " operands, but target successor needs "
                                 << succInputsTypes.size();
    }

    for (const auto &typesIdx :
         llvm::enumerate(llvm::zip(*sourceTypes, succInputsTypes))) {
      Type sourceType = std::get<0>(typesIdx.value());
      Type inputType = std::get<1>(typesIdx.value());
      if (!regionInterface.areTypesCompatible(sourceType, inputType)) {
        InFlightDiagnostic diag = op->emitOpError(" along control flow edge ");
        return printEdgeName(diag)
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

  auto inputTypesFromParent = [&](Optional<unsigned> regionNo) -> TypeRange {
    if (regionNo.hasValue()) {
      return regionInterface.getSuccessorEntryOperands(regionNo.getValue())
          .getTypes();
    }

    // If the successor of a parent op is the parent itself
    // RegionBranchOpInterface does not have an API to query what the entry
    // operands will be in that case. Vend out the result types of the op in
    // that case so that type checking succeeds for this case.
    return op->getResultTypes();
  };

  // Verify types along control flow edges originating from the parent.
  if (failed(verifyTypesAlongAllEdges(op, llvm::None, inputTypesFromParent)))
    return failure();

  // RegionBranchOpInterface should not be implemented by Ops that do not have
  // attached regions.
  assert(op->getNumRegions() != 0);

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
  for (unsigned regionNo : llvm::seq(0U, op->getNumRegions())) {
    Region &region = op->getRegion(regionNo);

    // Since there can be multiple `ReturnLike` terminators or others
    // implementing the `RegionBranchTerminatorOpInterface`, all should have the
    // same operand types when passing them to the same region.

    Optional<OperandRange> regionReturnOperands;
    for (Block &block : region) {
      Operation *terminator = block.getTerminator();
      auto terminatorOperands =
          getRegionBranchSuccessorOperands(terminator, regionNo);
      if (!terminatorOperands)
        continue;

      if (!regionReturnOperands) {
        regionReturnOperands = terminatorOperands;
        continue;
      }

      // Found more than one ReturnLike terminator. Make sure the operand types
      // match with the first one.
      if (!areTypesCompatible(regionReturnOperands->getTypes(),
                              terminatorOperands->getTypes()))
        return op->emitOpError("Region #")
               << regionNo
               << " operands mismatch between return-like terminators";
    }

    auto inputTypesFromRegion =
        [&](Optional<unsigned> regionNo) -> Optional<TypeRange> {
      // If there is no return-like terminator, the op itself should verify
      // type consistency.
      if (!regionReturnOperands)
        return llvm::None;

      // All successors get the same set of operand types.
      return TypeRange(regionReturnOperands->getTypes());
    };

    if (failed(verifyTypesAlongAllEdges(op, regionNo, inputTypesFromRegion)))
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
  SmallVector<unsigned> worklist;
  auto enqueueAllSuccessors = [&](unsigned index) {
    SmallVector<RegionSuccessor> successors;
    op.getSuccessorRegions(index, successors);
    for (RegionSuccessor successor : successors)
      if (!successor.isParent())
        worklist.push_back(successor.getSuccessor()->getRegionNumber());
  };
  enqueueAllSuccessors(begin->getRegionNumber());

  // Process all regions in the worklist via DFS.
  while (!worklist.empty()) {
    unsigned nextRegion = worklist.pop_back_val();
    if (nextRegion == r->getRegionNumber())
      return true;
    if (visited[nextRegion])
      continue;
    visited[nextRegion] = true;
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

void RegionBranchOpInterface::getSuccessorRegions(
    Optional<unsigned> index, SmallVectorImpl<RegionSuccessor> &regions) {
  unsigned numInputs = 0;
  if (index) {
    // If the predecessor is a region, get the number of operands from an
    // exiting terminator in the region.
    for (Block &block : getOperation()->getRegion(*index)) {
      Operation *terminator = block.getTerminator();
      if (getRegionBranchSuccessorOperands(terminator, *index)) {
        numInputs = terminator->getNumOperands();
        break;
      }
    }
  } else {
    // Otherwise, use the number of parent operation operands.
    numInputs = getOperation()->getNumOperands();
  }
  SmallVector<Attribute, 2> operands(numInputs, nullptr);
  getSuccessorRegions(index, operands, regions);
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

//===----------------------------------------------------------------------===//
// RegionBranchTerminatorOpInterface
//===----------------------------------------------------------------------===//

/// Returns true if the given operation is either annotated with the
/// `ReturnLike` trait or implements the `RegionBranchTerminatorOpInterface`.
bool mlir::isRegionReturnLike(Operation *operation) {
  return dyn_cast<RegionBranchTerminatorOpInterface>(operation) ||
         operation->hasTrait<OpTrait::ReturnLike>();
}

/// Returns the mutable operands that are passed to the region with the given
/// `regionIndex`. If the operation does not implement the
/// `RegionBranchTerminatorOpInterface` and is not marked as `ReturnLike`, the
/// result will be `llvm::None`. In all other cases, the resulting
/// `OperandRange` represents all operands that are passed to the specified
/// successor region. If `regionIndex` is `llvm::None`, all operands that are
/// passed to the parent operation will be returned.
Optional<MutableOperandRange>
mlir::getMutableRegionBranchSuccessorOperands(Operation *operation,
                                              Optional<unsigned> regionIndex) {
  // Try to query a RegionBranchTerminatorOpInterface to determine
  // all successor operands that will be passed to the successor
  // input arguments.
  if (auto regionTerminatorInterface =
          dyn_cast<RegionBranchTerminatorOpInterface>(operation))
    return regionTerminatorInterface.getMutableSuccessorOperands(regionIndex);

  // TODO: The ReturnLike trait should imply a default implementation of the
  // RegionBranchTerminatorOpInterface. This would make this code significantly
  // easier. Furthermore, this may even make this function obsolete.
  if (operation->hasTrait<OpTrait::ReturnLike>())
    return MutableOperandRange(operation);
  return llvm::None;
}

/// Returns the read only operands that are passed to the region with the given
/// `regionIndex`. See `getMutableRegionBranchSuccessorOperands` for more
/// information.
Optional<OperandRange>
mlir::getRegionBranchSuccessorOperands(Operation *operation,
                                       Optional<unsigned> regionIndex) {
  auto range = getMutableRegionBranchSuccessorOperands(operation, regionIndex);
  return range ? Optional<OperandRange>(*range) : llvm::None;
}
