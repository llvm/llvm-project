//===- ControlFlowInterfaces.cpp - ControlFlow Interfaces -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ControlFlowInterfaces
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ControlFlowInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// BranchOpInterface
//===----------------------------------------------------------------------===//

/// Returns the `BlockArgument` corresponding to operand `operandIndex` in some
/// successor if 'operandIndex' is within the range of 'operands', or None if
/// `operandIndex` isn't a successor operand index.
Optional<BlockArgument>
detail::getBranchSuccessorArgument(Optional<OperandRange> operands,
                                   unsigned operandIndex, Block *successor) {
  // Check that the operands are valid.
  if (!operands || operands->empty())
    return llvm::None;

  // Check to ensure that this operand is within the range.
  unsigned operandsStart = operands->getBeginOperandIndex();
  if (operandIndex < operandsStart ||
      operandIndex >= (operandsStart + operands->size()))
    return llvm::None;

  // Index the successor.
  unsigned argIndex = operandIndex - operandsStart;
  return successor->getArgument(argIndex);
}

/// Verify that the given operands match those of the given successor block.
LogicalResult
detail::verifyBranchSuccessorOperands(Operation *op, unsigned succNo,
                                      Optional<OperandRange> operands) {
  if (!operands)
    return success();

  // Check the count.
  unsigned operandCount = operands->size();
  Block *destBB = op->getSuccessor(succNo);
  if (operandCount != destBB->getNumArguments())
    return op->emitError() << "branch has " << operandCount
                           << " operands for successor #" << succNo
                           << ", but target block has "
                           << destBB->getNumArguments();

  // Check the types.
  auto operandIt = operands->begin();
  for (unsigned i = 0; i != operandCount; ++i, ++operandIt) {
    if ((*operandIt).getType() != destBB->getArgument(i).getType())
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
/// inputs that flow from `sourceIndex' to the given region.
static LogicalResult verifyTypesAlongAllEdges(
    Operation *op, Optional<unsigned> sourceNo,
    function_ref<TypeRange(Optional<unsigned>)> getInputsTypesForRegion) {
  auto regionInterface = cast<RegionBranchOpInterface>(op);

  SmallVector<RegionSuccessor, 2> successors;
  unsigned numInputs;
  if (sourceNo) {
    Region &srcRegion = op->getRegion(sourceNo.getValue());
    numInputs = srcRegion.getNumArguments();
  } else {
    numInputs = op->getNumOperands();
  }
  SmallVector<Attribute, 2> operands(numInputs, nullptr);
  regionInterface.getSuccessorRegions(sourceNo, operands, successors);

  for (RegionSuccessor &succ : successors) {
    Optional<unsigned> succRegionNo;
    if (!succ.isParent())
      succRegionNo = succ.getSuccessor()->getRegionNumber();

    auto printEdgeName = [&](InFlightDiagnostic &diag) -> InFlightDiagnostic & {
      diag << "from ";
      if (sourceNo)
        diag << "Region #" << sourceNo.getValue();
      else
        diag << op->getName();

      diag << " to ";
      if (succRegionNo)
        diag << "Region #" << succRegionNo.getValue();
      else
        diag << op->getName();
      return diag;
    };

    TypeRange sourceTypes = getInputsTypesForRegion(succRegionNo);
    TypeRange succInputsTypes = succ.getSuccessorInputs().getTypes();
    if (sourceTypes.size() != succInputsTypes.size()) {
      InFlightDiagnostic diag = op->emitOpError(" region control flow edge ");
      return printEdgeName(diag)
             << " has " << sourceTypes.size()
             << " source operands, but target successor needs "
             << succInputsTypes.size();
    }

    for (auto typesIdx :
         llvm::enumerate(llvm::zip(sourceTypes, succInputsTypes))) {
      Type sourceType = std::get<0>(typesIdx.value());
      Type inputType = std::get<1>(typesIdx.value());
      if (sourceType != inputType) {
        InFlightDiagnostic diag = op->emitOpError(" along control flow edge ");
        return printEdgeName(diag)
               << " source #" << typesIdx.index() << " type " << sourceType
               << " should match input #" << typesIdx.index() << " type "
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

  // Verify types along control flow edges originating from each region.
  for (unsigned regionNo : llvm::seq(0U, op->getNumRegions())) {
    Region &region = op->getRegion(regionNo);

    // Since the interface cannnot distinguish between different ReturnLike
    // ops within the region branching to different successors, all ReturnLike
    // ops in this region should have the same operand types. We will then use
    // one of them as the representative for type matching.

    Operation *regionReturn = nullptr;
    for (Block &block : region) {
      Operation *terminator = block.getTerminator();
      if (!terminator->hasTrait<OpTrait::ReturnLike>())
        continue;

      if (!regionReturn) {
        regionReturn = terminator;
        continue;
      }

      // Found more than one ReturnLike terminator. Make sure the operand types
      // match with the first one.
      if (regionReturn->getOperandTypes() != terminator->getOperandTypes())
        return op->emitOpError("Region #")
               << regionNo
               << " operands mismatch between return-like terminators";
    }

    auto inputTypesFromRegion = [&](Optional<unsigned> regionNo) -> TypeRange {
      // All successors get the same set of operands.
      return regionReturn ? TypeRange(regionReturn->getOperands().getTypes())
                          : TypeRange();
    };

    if (failed(verifyTypesAlongAllEdges(op, regionNo, inputTypesFromRegion)))
      return failure();
  }

  return success();
}
