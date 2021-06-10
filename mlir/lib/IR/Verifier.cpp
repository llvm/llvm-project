//===- Verifier.cpp - MLIR Verifier Implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the verify() methods on the various IR types, performing
// (potentially expensive) checks on the holistic structure of the code.  This
// can be used for detecting bugs in compiler transformations and hand written
// .mlir files.
//
// The checks in this file are only for things that can occur as part of IR
// transformations: e.g. violation of dominance information, malformed operation
// attributes, etc.  MLIR supports transformations moving IR through locally
// invalid states (e.g. unlinking an operation from a block before re-inserting
// it in a new place), but each transformation must complete with the IR in a
// valid form.
//
// This should not check for things that are always wrong by construction (e.g.
// attributes or other immutable structures that are incorrect), because those
// are not mutable and can be checked at time of construction.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/PrettyStackTrace.h"

#include <atomic>

using namespace mlir;

namespace {
/// This class encapsulates all the state used to verify an operation region.
class OperationVerifier {
public:
  explicit OperationVerifier(MLIRContext *context)
      : parallelismEnabled(context->isMultithreadingEnabled()) {}

  /// Verify the given operation.
  LogicalResult verifyOpAndDominance(Operation &op);

private:
  /// Verify the given potentially nested region or block.
  LogicalResult verifyRegion(Region &region);
  LogicalResult verifyBlock(Block &block);
  LogicalResult verifyOperation(Operation &op);

  /// Verify the dominance property of regions contained within the given
  /// Operation.
  LogicalResult verifyDominanceOfContainedRegions(Operation &op,
                                                  DominanceInfo &domInfo);

  /// Emit an error for the given block.
  InFlightDiagnostic emitError(Block &bb, const Twine &message) {
    // Take the location information for the first operation in the block.
    if (!bb.empty())
      return bb.front().emitError(message);

    // Worst case, fall back to using the parent's location.
    return mlir::emitError(bb.getParent()->getLoc(), message);
  }

  /// This is true if parallelism is enabled on the MLIRContext.
  const bool parallelismEnabled;
};
} // end anonymous namespace

/// Verify the given operation.
LogicalResult OperationVerifier::verifyOpAndDominance(Operation &op) {
  // Verify the operation first.
  if (failed(verifyOperation(op)))
    return failure();

  // Since everything looks structurally ok to this point, we do a dominance
  // check for any nested regions. We do this as a second pass since malformed
  // CFG's can cause dominator analysis constructure to crash and we want the
  // verifier to be resilient to malformed code.
  if (op.getNumRegions() != 0) {
    DominanceInfo domInfo;
    if (failed(verifyDominanceOfContainedRegions(op, /*domInfo*/ domInfo)))
      return failure();
  }
  return success();
}

LogicalResult OperationVerifier::verifyRegion(Region &region) {
  if (region.empty())
    return success();

  // Verify the first block has no predecessors.
  auto *firstBB = &region.front();
  if (!firstBB->hasNoPredecessors())
    return mlir::emitError(region.getLoc(),
                           "entry block of region may not have predecessors");

  // Verify each of the blocks within the region.
  for (Block &block : region)
    if (failed(verifyBlock(block)))
      return failure();
  return success();
}

/// Returns true if this block may be valid without terminator. That is if:
/// - it does not have a parent region.
/// - Or the parent region have a single block and:
///    - This region does not have a parent op.
///    - Or the parent op is unregistered.
///    - Or the parent op has the NoTerminator trait.
static bool mayBeValidWithoutTerminator(Block *block) {
  if (!block->getParent())
    return true;
  if (!llvm::hasSingleElement(*block->getParent()))
    return false;
  Operation *op = block->getParentOp();
  return !op || op->mightHaveTrait<OpTrait::NoTerminator>();
}

LogicalResult OperationVerifier::verifyBlock(Block &block) {
  for (auto arg : block.getArguments())
    if (arg.getOwner() != &block)
      return emitError(block, "block argument not owned by block");

  // Verify that this block has a terminator.
  if (block.empty()) {
    if (mayBeValidWithoutTerminator(&block))
      return success();
    return emitError(block, "empty block: expect at least a terminator");
  }

  // Check each operation, and make sure there are no branches out of the
  // middle of this block.
  for (auto &op : llvm::make_range(block.begin(), block.end())) {
    // Only the last instructions is allowed to have successors.
    if (op.getNumSuccessors() != 0 && &op != &block.back())
      return op.emitError(
          "operation with block successors must terminate its parent block");

    if (failed(verifyOperation(op)))
      return failure();
  }

  // Verify that this block is not branching to a block of a different
  // region.
  for (Block *successor : block.getSuccessors())
    if (successor->getParent() != block.getParent())
      return block.back().emitOpError(
          "branching to block of a different region");

  // If this block doesn't have to have a terminator, don't require it.
  if (mayBeValidWithoutTerminator(&block))
    return success();

  Operation &terminator = block.back();
  if (!terminator.mightHaveTrait<OpTrait::IsTerminator>())
    return block.back().emitError("block with no terminator, has ")
           << terminator;

  return success();
}

LogicalResult OperationVerifier::verifyOperation(Operation &op) {
  // Check that operands are non-nil and structurally ok.
  for (auto operand : op.getOperands())
    if (!operand)
      return op.emitError("null operand found");

  /// Verify that all of the attributes are okay.
  for (auto attr : op.getAttrs()) {
    // Check for any optional dialect specific attributes.
    if (auto *dialect = attr.first.getDialect())
      if (failed(dialect->verifyOperationAttribute(&op, attr)))
        return failure();
  }

  // If we can get operation info for this, check the custom hook.
  OperationName opName = op.getName();
  auto *opInfo = opName.getAbstractOperation();
  if (opInfo && failed(opInfo->verifyInvariants(&op)))
    return failure();

  if (unsigned numRegions = op.getNumRegions()) {
    auto kindInterface = dyn_cast<mlir::RegionKindInterface>(op);

    // Verify that all child regions are ok.
    for (unsigned i = 0; i < numRegions; ++i) {
      Region &region = op.getRegion(i);
      RegionKind kind =
          kindInterface ? kindInterface.getRegionKind(i) : RegionKind::SSACFG;
      // Check that Graph Regions only have a single basic block. This is
      // similar to the code in SingleBlockImplicitTerminator, but doesn't
      // require the trait to be specified. This arbitrary limitation is
      // designed to limit the number of cases that have to be handled by
      // transforms and conversions.
      if (op.isRegistered() && kind == RegionKind::Graph) {
        // Empty regions are fine.
        if (region.empty())
          continue;

        // Non-empty regions must contain a single basic block.
        if (std::next(region.begin()) != region.end())
          return op.emitOpError("expects graph region #")
                 << i << " to have 0 or 1 blocks";
      }
      if (failed(verifyRegion(region)))
        return failure();
    }
  }

  // If this is a registered operation, there is nothing left to do.
  if (opInfo)
    return success();

  // Otherwise, verify that the parent dialect allows un-registered operations.
  Dialect *dialect = opName.getDialect();
  if (!dialect) {
    if (!op.getContext()->allowsUnregisteredDialects()) {
      return op.emitOpError()
             << "created with unregistered dialect. If this is "
                "intended, please call allowUnregisteredDialects() on the "
                "MLIRContext, or use -allow-unregistered-dialect with "
                "mlir-opt";
    }
    return success();
  }

  if (!dialect->allowsUnknownOperations()) {
    return op.emitError("unregistered operation '")
           << op.getName() << "' found in dialect ('" << dialect->getNamespace()
           << "') that does not allow unknown operations";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Dominance Checking
//===----------------------------------------------------------------------===//

/// Emit an error when the specified operand of the specified operation is an
/// invalid use because of dominance properties.
static void diagnoseInvalidOperandDominance(Operation &op, unsigned operandNo) {
  InFlightDiagnostic diag = op.emitError("operand #")
                            << operandNo << " does not dominate this use";

  Value operand = op.getOperand(operandNo);

  /// Attach a note to an in-flight diagnostic that provide more information
  /// about where an op operand is defined.
  if (auto *useOp = operand.getDefiningOp()) {
    Diagnostic &note = diag.attachNote(useOp->getLoc());
    note << "operand defined here";
    Block *block1 = op.getBlock();
    Block *block2 = useOp->getBlock();
    Region *region1 = block1->getParent();
    Region *region2 = block2->getParent();
    if (block1 == block2)
      note << " (op in the same block)";
    else if (region1 == region2)
      note << " (op in the same region)";
    else if (region2->isProperAncestor(region1))
      note << " (op in a parent region)";
    else if (region1->isProperAncestor(region2))
      note << " (op in a child region)";
    else
      note << " (op is neither in a parent nor in a child region)";
    return;
  }
  // Block argument case.
  Block *block1 = op.getBlock();
  Block *block2 = operand.cast<BlockArgument>().getOwner();
  Region *region1 = block1->getParent();
  Region *region2 = block2->getParent();
  Location loc = UnknownLoc::get(op.getContext());
  if (block2->getParentOp())
    loc = block2->getParentOp()->getLoc();
  Diagnostic &note = diag.attachNote(loc);
  if (!region2) {
    note << " (block without parent)";
    return;
  }
  if (block1 == block2)
    llvm::report_fatal_error("Internal error in dominance verification");
  int index = std::distance(region2->begin(), block2->getIterator());
  note << "operand defined as a block argument (block #" << index;
  if (region1 == region2)
    note << " in the same region)";
  else if (region2->isProperAncestor(region1))
    note << " in a parent region)";
  else if (region1->isProperAncestor(region2))
    note << " in a child region)";
  else
    note << " neither in a parent nor in a child region)";
}

/// Verify the dominance of each of the nested blocks within the given
/// operation.  domInfo may be present or absent (null), depending on whether
/// the caller computed it for a higher level.
LogicalResult
OperationVerifier::verifyDominanceOfContainedRegions(Operation &opWithRegions,
                                                     DominanceInfo &domInfo) {
  // This vector keeps track of ops that have regions which should be checked
  // in parallel.
  SmallVector<Operation *> opsWithRegionsToCheckInParallel;

  // Get information about the requirements on the regions in this op.
  for (Region &region : opWithRegions.getRegions()) {
    for (Block &block : region) {
      // Dominance is only meaningful inside reachable blocks.
      bool isReachable = domInfo.isReachableFromEntry(&block);

      // Check each operation in this block, and any operations in regions
      // that these operations contain.
      opsWithRegionsToCheckInParallel.clear();

      for (Operation &op : block) {
        if (isReachable) {
          // Check that operands properly dominate this use.
          for (auto &operand : op.getOpOperands()) {
            // If the operand doesn't dominate the user, then emit an error.
            if (!domInfo.properlyDominates(operand.get(), &op)) {
              diagnoseInvalidOperandDominance(op, operand.getOperandNumber());
              return failure();
            }
          }
        }

        // If this operation has any regions, we need to recursively verify
        // dominance of the ops within it.
        if (op.getNumRegions() == 0)
          continue;

        // If this is a non-isolated region (e.g. an affine for loop), pass down
        // the current dominator information.
        if (!op.hasTrait<OpTrait::IsIsolatedFromAbove>()) {
          if (failed(verifyDominanceOfContainedRegions(op, domInfo)))
            return failure();
        } else if (parallelismEnabled) {
          // If this is an IsolatedFromAbove op and parallelism is enabled, then
          // enqueue this for processing later.
          opsWithRegionsToCheckInParallel.push_back(&op);
        } else {
          // If not, just verify inline with a local dom scope.
          DominanceInfo localDomInfo;
          if (failed(verifyDominanceOfContainedRegions(op, localDomInfo)))
            return failure();
        }
      }

      // If we have multiple parallelizable subregions, check them in parallel.
      if (opsWithRegionsToCheckInParallel.size() == 1) {
        // Each isolated operation gets its own dom info.
        Operation *op = opsWithRegionsToCheckInParallel[0];
        DominanceInfo localDomInfo;
        if (failed(verifyDominanceOfContainedRegions(*op, localDomInfo)))
          return failure();
      } else if (!opsWithRegionsToCheckInParallel.empty()) {
        ParallelDiagnosticHandler handler(opWithRegions.getContext());
        std::atomic<bool> passFailed(false);
        llvm::parallelForEachN(
            0, opsWithRegionsToCheckInParallel.size(), [&](size_t opIdx) {
              handler.setOrderIDForThread(opIdx);
              Operation *op = opsWithRegionsToCheckInParallel[opIdx];

              // Each isolated operation gets its own dom info.
              DominanceInfo localDomInfo;
              if (failed(verifyDominanceOfContainedRegions(*op, localDomInfo)))
                passFailed = true;
              handler.eraseOrderIDForThread();
            });
        if (passFailed)
          return failure();
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Entrypoint
//===----------------------------------------------------------------------===//

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  On error, this reports the error through the MLIRContext
/// and returns failure.
LogicalResult mlir::verify(Operation *op) {
  return OperationVerifier(op->getContext()).verifyOpAndDominance(*op);
}
