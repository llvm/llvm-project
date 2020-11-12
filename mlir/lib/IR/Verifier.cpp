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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"

using namespace mlir;

namespace {
/// This class encapsulates all the state used to verify an operation region.
class OperationVerifier {
public:
  explicit OperationVerifier(MLIRContext *ctx) : ctx(ctx) {}

  /// Verify the given operation.
  LogicalResult verify(Operation &op);

  /// Returns the registered dialect for a dialect-specific attribute.
  Dialect *getDialectForAttribute(const NamedAttribute &attr) {
    assert(attr.first.strref().contains('.') && "expected dialect attribute");
    auto dialectNamePair = attr.first.strref().split('.');
    return ctx->getLoadedDialect(dialectNamePair.first);
  }

private:
  /// Verify the given potentially nested region or block.
  LogicalResult verifyRegion(Region &region);
  LogicalResult verifyBlock(Block &block);
  LogicalResult verifyOperation(Operation &op);

  /// Verify the dominance property of operations within the given Region.
  LogicalResult verifyDominance(Region &region);

  /// Verify the dominance property of regions contained within the given
  /// Operation.
  LogicalResult verifyDominanceOfContainedRegions(Operation &op);

  /// Emit an error for the given block.
  InFlightDiagnostic emitError(Block &bb, const Twine &message) {
    // Take the location information for the first operation in the block.
    if (!bb.empty())
      return bb.front().emitError(message);

    // Worst case, fall back to using the parent's location.
    return mlir::emitError(bb.getParent()->getLoc(), message);
  }

  /// The current context for the verifier.
  MLIRContext *ctx;

  /// Dominance information for this operation, when checking dominance.
  DominanceInfo *domInfo = nullptr;

  /// Mapping between dialect namespace and if that dialect supports
  /// unregistered operations.
  llvm::StringMap<bool> dialectAllowsUnknownOps;
};
} // end anonymous namespace

/// Verify the given operation.
LogicalResult OperationVerifier::verify(Operation &op) {
  // Verify the operation first.
  if (failed(verifyOperation(op)))
    return failure();

  // Since everything looks structurally ok to this point, we do a dominance
  // check for any nested regions. We do this as a second pass since malformed
  // CFG's can cause dominator analysis constructure to crash and we want the
  // verifier to be resilient to malformed code.
  DominanceInfo theDomInfo(&op);
  domInfo = &theDomInfo;
  if (failed(verifyDominanceOfContainedRegions(op)))
    return failure();

  domInfo = nullptr;
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

LogicalResult OperationVerifier::verifyBlock(Block &block) {
  for (auto arg : block.getArguments())
    if (arg.getOwner() != &block)
      return emitError(block, "block argument not owned by block");

  // Verify that this block has a terminator.
  if (block.empty())
    return emitError(block, "block with no terminator");

  // Verify the non-terminator operations separately so that we can verify
  // they has no successors.
  for (auto &op : llvm::make_range(block.begin(), std::prev(block.end()))) {
    if (op.getNumSuccessors() != 0)
      return op.emitError(
          "operation with block successors must terminate its parent block");

    if (failed(verifyOperation(op)))
      return failure();
  }

  // Verify the terminator.
  if (failed(verifyOperation(block.back())))
    return failure();
  if (block.back().isKnownNonTerminator())
    return block.back().emitError("block with no terminator");

  // Verify that this block is not branching to a block of a different
  // region.
  for (Block *successor : block.getSuccessors())
    if (successor->getParent() != block.getParent())
      return block.back().emitOpError(
          "branching to block of a different region");

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
    if (!attr.first.strref().contains('.'))
      continue;
    if (auto *dialect = getDialectForAttribute(attr))
      if (failed(dialect->verifyOperationAttribute(&op, attr)))
        return failure();
  }

  // If we can get operation info for this, check the custom hook.
  auto *opInfo = op.getAbstractOperation();
  if (opInfo && failed(opInfo->verifyInvariants(&op)))
    return failure();

  auto kindInterface = dyn_cast<mlir::RegionKindInterface>(op);

  // Verify that all child regions are ok.
  unsigned numRegions = op.getNumRegions();
  for (unsigned i = 0; i < numRegions; i++) {
    Region &region = op.getRegion(i);
    // Check that Graph Regions only have a single basic block. This is
    // similar to the code in SingleBlockImplicitTerminator, but doesn't
    // require the trait to be specified. This arbitrary limitation is
    // designed to limit the number of cases that have to be handled by
    // transforms and conversions until the concept stabilizes.
    if (op.isRegistered() && kindInterface &&
        kindInterface.getRegionKind(i) == RegionKind::Graph) {
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

  // If this is a registered operation, there is nothing left to do.
  if (opInfo)
    return success();

  // Otherwise, verify that the parent dialect allows un-registered operations.
  auto dialectPrefix = op.getName().getDialect();

  // Check for an existing answer for the operation dialect.
  auto it = dialectAllowsUnknownOps.find(dialectPrefix);
  if (it == dialectAllowsUnknownOps.end()) {
    // If the operation dialect is registered, query it directly.
    if (auto *dialect = ctx->getLoadedDialect(dialectPrefix))
      it = dialectAllowsUnknownOps
               .try_emplace(dialectPrefix, dialect->allowsUnknownOperations())
               .first;
    // Otherwise, unregistered dialects (when allowed by the context)
    // conservatively allow unknown operations.
    else {
      if (!op.getContext()->allowsUnregisteredDialects() && !op.getDialect())
        return op.emitOpError()
               << "created with unregistered dialect. If this is "
                  "intended, please call allowUnregisteredDialects() on the "
                  "MLIRContext, or use -allow-unregistered-dialect with "
                  "mlir-opt";

      it = dialectAllowsUnknownOps.try_emplace(dialectPrefix, true).first;
    }
  }

  if (!it->second) {
    return op.emitError("unregistered operation '")
           << op.getName() << "' found in dialect ('" << dialectPrefix
           << "') that does not allow unknown operations";
  }

  return success();
}

LogicalResult OperationVerifier::verifyDominance(Region &region) {
  // Verify the dominance of each of the held operations.
  for (Block &block : region) {
    // Dominance is only meaningful inside reachable blocks.
    if (domInfo->isReachableFromEntry(&block))
      for (Operation &op : block)
        // Check that operands properly dominate this use.
        for (unsigned operandNo = 0, e = op.getNumOperands(); operandNo != e;
             ++operandNo) {
          auto operand = op.getOperand(operandNo);
          if (domInfo->properlyDominates(operand, &op))
            continue;

          auto diag = op.emitError("operand #")
                      << operandNo << " does not dominate this use";
          if (auto *useOp = operand.getDefiningOp())
            diag.attachNote(useOp->getLoc()) << "operand defined here";
          return failure();
        }
    // Recursively verify dominance within each operation in the
    // block, even if the block itself is not reachable, or we are in
    // a region which doesn't respect dominance.
    for (Operation &op : block)
      if (failed(verifyDominanceOfContainedRegions(op)))
        return failure();
  }
  return success();
}

/// Verify the dominance of each of the nested blocks within the given operation
LogicalResult
OperationVerifier::verifyDominanceOfContainedRegions(Operation &op) {
  for (Region &region : op.getRegions()) {
    if (failed(verifyDominance(region)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Entrypoint
//===----------------------------------------------------------------------===//

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  On error, this reports the error through the MLIRContext and
/// returns failure.
LogicalResult mlir::verify(Operation *op) {
  return OperationVerifier(op->getContext()).verify(*op);
}
