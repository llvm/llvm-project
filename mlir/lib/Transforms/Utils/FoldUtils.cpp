//===- FoldUtils.cpp ---- Fold Utilities ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various operation fold utilities. These utilities are
// intended to be used by passes to unify and simply their logic.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/FoldUtils.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

/// Given an operation, find the parent region that folded constants should be
/// inserted into.
static Region *
getInsertionRegion(DialectInterfaceCollection<DialectFoldInterface> &interfaces,
                   Block *insertionBlock) {
  while (Region *region = insertionBlock->getParent()) {
    // Insert in this region for any of the following scenarios:
    //  * The parent is unregistered, or is known to be isolated from above.
    //  * The parent is a top-level operation.
    auto *parentOp = region->getParentOp();
    if (!parentOp->isRegistered() || parentOp->isKnownIsolatedFromAbove() ||
        !parentOp->getBlock())
      return region;

    // Otherwise, check if this region is a desired insertion region.
    auto *interface = interfaces.getInterfaceFor(parentOp);
    if (LLVM_UNLIKELY(interface && interface->shouldMaterializeInto(region)))
      return region;

    // Traverse up the parent looking for an insertion region.
    insertionBlock = parentOp->getBlock();
  }
  llvm_unreachable("expected valid insertion region");
}

/// A utility function used to materialize a constant for a given attribute and
/// type. On success, a valid constant value is returned. Otherwise, null is
/// returned
static Operation *materializeConstant(Dialect *dialect, OpBuilder &builder,
                                      Attribute value, Type type,
                                      Location loc) {
  auto insertPt = builder.getInsertionPoint();
  (void)insertPt;

  // Ask the dialect to materialize a constant operation for this value.
  if (auto *constOp = dialect->materializeConstant(builder, value, type, loc)) {
    assert(insertPt == builder.getInsertionPoint());
    assert(matchPattern(constOp, m_Constant()));
    return constOp;
  }

  // TODO: To facilitate splitting the std dialect (PR48490), have a special
  // case for falling back to std.constant. Eventually, we will have separate
  // ops tensor.constant, int.constant, float.constant, etc. that live in their
  // respective dialects, which will allow each dialect to implement the
  // materializeConstant hook above.
  //
  // The special case is needed because in the interim state while we are
  // splitting out those dialects from std, the std dialect depends on the
  // tensor dialect, which makes it impossible for the tensor dialect to use
  // std.constant (it would be a cyclic dependency) as part of its
  // materializeConstant hook.
  //
  // If the dialect is unable to materialize a constant, check to see if the
  // standard constant can be used.
  if (ConstantOp::isBuildableWith(value, type))
    return builder.create<ConstantOp>(loc, type, value);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// OperationFolder
//===----------------------------------------------------------------------===//

LogicalResult OperationFolder::tryToFold(
    Operation *op, function_ref<void(Operation *)> processGeneratedConstants,
    function_ref<void(Operation *)> preReplaceAction, bool *inPlaceUpdate) {
  if (inPlaceUpdate)
    *inPlaceUpdate = false;

  // If this is a unique'd constant, return failure as we know that it has
  // already been folded.
  if (referencedDialects.count(op))
    return failure();

  // Try to fold the operation.
  SmallVector<Value, 8> results;
  OpBuilder builder(op);
  if (failed(tryToFold(builder, op, results, processGeneratedConstants)))
    return failure();

  // Check to see if the operation was just updated in place.
  if (results.empty()) {
    if (inPlaceUpdate)
      *inPlaceUpdate = true;
    return success();
  }

  // Constant folding succeeded. We will start replacing this op's uses and
  // erase this op. Invoke the callback provided by the caller to perform any
  // pre-replacement action.
  if (preReplaceAction)
    preReplaceAction(op);

  // Replace all of the result values and erase the operation.
  for (unsigned i = 0, e = results.size(); i != e; ++i)
    op->getResult(i).replaceAllUsesWith(results[i]);
  op->erase();
  return success();
}

/// Notifies that the given constant `op` should be remove from this
/// OperationFolder's internal bookkeeping.
void OperationFolder::notifyRemoval(Operation *op) {
  // Check to see if this operation is uniqued within the folder.
  auto it = referencedDialects.find(op);
  if (it == referencedDialects.end())
    return;

  // Get the constant value for this operation, this is the value that was used
  // to unique the operation internally.
  Attribute constValue;
  matchPattern(op, m_Constant(&constValue));
  assert(constValue);

  // Get the constant map that this operation was uniqued in.
  auto &uniquedConstants =
      foldScopes[getInsertionRegion(interfaces, op->getBlock())];

  // Erase all of the references to this operation.
  auto type = op->getResult(0).getType();
  for (auto *dialect : it->second)
    uniquedConstants.erase(std::make_tuple(dialect, constValue, type));
  referencedDialects.erase(it);
}

/// Clear out any constants cached inside of the folder.
void OperationFolder::clear() {
  foldScopes.clear();
  referencedDialects.clear();
}

/// Get or create a constant using the given builder. On success this returns
/// the constant operation, nullptr otherwise.
Value OperationFolder::getOrCreateConstant(OpBuilder &builder, Dialect *dialect,
                                           Attribute value, Type type,
                                           Location loc) {
  OpBuilder::InsertionGuard foldGuard(builder);

  // Use the builder insertion block to find an insertion point for the
  // constant.
  auto *insertRegion =
      getInsertionRegion(interfaces, builder.getInsertionBlock());
  auto &entry = insertRegion->front();
  builder.setInsertionPoint(&entry, entry.begin());

  // Get the constant map for the insertion region of this operation.
  auto &uniquedConstants = foldScopes[insertRegion];
  Operation *constOp = tryGetOrCreateConstant(uniquedConstants, dialect,
                                              builder, value, type, loc);
  return constOp ? constOp->getResult(0) : Value();
}

/// Tries to perform folding on the given `op`. If successful, populates
/// `results` with the results of the folding.
LogicalResult OperationFolder::tryToFold(
    OpBuilder &builder, Operation *op, SmallVectorImpl<Value> &results,
    function_ref<void(Operation *)> processGeneratedConstants) {
  SmallVector<Attribute, 8> operandConstants;
  SmallVector<OpFoldResult, 8> foldResults;

  // If this is a commutative operation, move constants to be trailing operands.
  if (op->getNumOperands() >= 2 && op->isCommutative()) {
    std::stable_partition(
        op->getOpOperands().begin(), op->getOpOperands().end(),
        [&](OpOperand &O) { return !matchPattern(O.get(), m_Constant()); });
  }

  // Check to see if any operands to the operation is constant and whether
  // the operation knows how to constant fold itself.
  operandConstants.assign(op->getNumOperands(), Attribute());
  for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
    matchPattern(op->getOperand(i), m_Constant(&operandConstants[i]));

  // Attempt to constant fold the operation.
  if (failed(op->fold(operandConstants, foldResults)))
    return failure();

  // Check to see if the operation was just updated in place.
  if (foldResults.empty())
    return success();
  assert(foldResults.size() == op->getNumResults());

  // Create a builder to insert new operations into the entry block of the
  // insertion region.
  auto *insertRegion =
      getInsertionRegion(interfaces, builder.getInsertionBlock());
  auto &entry = insertRegion->front();
  OpBuilder::InsertionGuard foldGuard(builder);
  builder.setInsertionPoint(&entry, entry.begin());

  // Get the constant map for the insertion region of this operation.
  auto &uniquedConstants = foldScopes[insertRegion];

  // Create the result constants and replace the results.
  auto *dialect = op->getDialect();
  for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
    assert(!foldResults[i].isNull() && "expected valid OpFoldResult");

    // Check if the result was an SSA value.
    if (auto repl = foldResults[i].dyn_cast<Value>()) {
      results.emplace_back(repl);
      continue;
    }

    // Check to see if there is a canonicalized version of this constant.
    auto res = op->getResult(i);
    Attribute attrRepl = foldResults[i].get<Attribute>();
    if (auto *constOp =
            tryGetOrCreateConstant(uniquedConstants, dialect, builder, attrRepl,
                                   res.getType(), op->getLoc())) {
      results.push_back(constOp->getResult(0));
      continue;
    }
    // If materialization fails, cleanup any operations generated for the
    // previous results and return failure.
    for (Operation &op : llvm::make_early_inc_range(
             llvm::make_range(entry.begin(), builder.getInsertionPoint()))) {
      notifyRemoval(&op);
      op.erase();
    }
    return failure();
  }

  // Process any newly generated operations.
  if (processGeneratedConstants) {
    for (auto i = entry.begin(), e = builder.getInsertionPoint(); i != e; ++i)
      processGeneratedConstants(&*i);
  }

  return success();
}

/// Try to get or create a new constant entry. On success this returns the
/// constant operation value, nullptr otherwise.
Operation *OperationFolder::tryGetOrCreateConstant(
    ConstantMap &uniquedConstants, Dialect *dialect, OpBuilder &builder,
    Attribute value, Type type, Location loc) {
  // Check if an existing mapping already exists.
  auto constKey = std::make_tuple(dialect, value, type);
  auto *&constInst = uniquedConstants[constKey];
  if (constInst)
    return constInst;

  // If one doesn't exist, try to materialize one.
  if (!(constInst = materializeConstant(dialect, builder, value, type, loc)))
    return nullptr;

  // Check to see if the generated constant is in the expected dialect.
  auto *newDialect = constInst->getDialect();
  if (newDialect == dialect) {
    referencedDialects[constInst].push_back(dialect);
    return constInst;
  }

  // If it isn't, then we also need to make sure that the mapping for the new
  // dialect is valid.
  auto newKey = std::make_tuple(newDialect, value, type);

  // If an existing operation in the new dialect already exists, delete the
  // materialized operation in favor of the existing one.
  if (auto *existingOp = uniquedConstants.lookup(newKey)) {
    constInst->erase();
    referencedDialects[existingOp].push_back(dialect);
    return constInst = existingOp;
  }

  // Otherwise, update the new dialect to the materialized operation.
  referencedDialects[constInst].assign({dialect, newDialect});
  auto newIt = uniquedConstants.insert({newKey, constInst});
  return newIt.first->second;
}
