//===- PatternMatch.cpp - Base classes for pattern match ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// PatternBenefit
//===----------------------------------------------------------------------===//

PatternBenefit::PatternBenefit(unsigned benefit) : representation(benefit) {
  assert(representation == benefit && benefit != ImpossibleToMatchSentinel &&
         "This pattern match benefit is too large to represent");
}

unsigned short PatternBenefit::getBenefit() const {
  assert(!isImpossibleToMatch() && "Pattern doesn't match");
  return representation;
}

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// OperationName Root Constructors

Pattern::Pattern(StringRef rootName, PatternBenefit benefit,
                 MLIRContext *context, ArrayRef<StringRef> generatedNames)
    : Pattern(OperationName(rootName, context).getAsOpaquePointer(),
              RootKind::OperationName, generatedNames, benefit, context) {}

//===----------------------------------------------------------------------===//
// MatchAnyOpTypeTag Root Constructors

Pattern::Pattern(MatchAnyOpTypeTag tag, PatternBenefit benefit,
                 MLIRContext *context, ArrayRef<StringRef> generatedNames)
    : Pattern(nullptr, RootKind::Any, generatedNames, benefit, context) {}

//===----------------------------------------------------------------------===//
// MatchInterfaceOpTypeTag Root Constructors

Pattern::Pattern(MatchInterfaceOpTypeTag tag, TypeID interfaceID,
                 PatternBenefit benefit, MLIRContext *context,
                 ArrayRef<StringRef> generatedNames)
    : Pattern(interfaceID.getAsOpaquePointer(), RootKind::InterfaceID,
              generatedNames, benefit, context) {}

//===----------------------------------------------------------------------===//
// MatchTraitOpTypeTag Root Constructors

Pattern::Pattern(MatchTraitOpTypeTag tag, TypeID traitID,
                 PatternBenefit benefit, MLIRContext *context,
                 ArrayRef<StringRef> generatedNames)
    : Pattern(traitID.getAsOpaquePointer(), RootKind::TraitID, generatedNames,
              benefit, context) {}

//===----------------------------------------------------------------------===//
// General Constructors

Pattern::Pattern(const void *rootValue, RootKind rootKind,
                 ArrayRef<StringRef> generatedNames, PatternBenefit benefit,
                 MLIRContext *context)
    : rootValue(rootValue), rootKind(rootKind), benefit(benefit),
      contextAndHasBoundedRecursion(context, false) {
  if (generatedNames.empty())
    return;
  generatedOps.reserve(generatedNames.size());
  std::transform(generatedNames.begin(), generatedNames.end(),
                 std::back_inserter(generatedOps), [context](StringRef name) {
                   return OperationName(name, context);
                 });
}

//===----------------------------------------------------------------------===//
// RewritePattern
//===----------------------------------------------------------------------===//

void RewritePattern::rewrite(Operation *op, PatternRewriter &rewriter) const {
  llvm_unreachable("need to implement either matchAndRewrite or one of the "
                   "rewrite functions!");
}

LogicalResult RewritePattern::match(Operation *op) const {
  llvm_unreachable("need to implement either match or matchAndRewrite!");
}

/// Out-of-line vtable anchor.
void RewritePattern::anchor() {}

//===----------------------------------------------------------------------===//
// PDLValue
//===----------------------------------------------------------------------===//

void PDLValue::print(raw_ostream &os) const {
  if (!value) {
    os << "<NULL-PDLValue>";
    return;
  }
  switch (kind) {
  case Kind::Attribute:
    os << cast<Attribute>();
    break;
  case Kind::Operation:
    os << *cast<Operation *>();
    break;
  case Kind::Type:
    os << cast<Type>();
    break;
  case Kind::TypeRange:
    llvm::interleaveComma(cast<TypeRange>(), os);
    break;
  case Kind::Value:
    os << cast<Value>();
    break;
  case Kind::ValueRange:
    llvm::interleaveComma(cast<ValueRange>(), os);
    break;
  }
}

void PDLValue::print(raw_ostream &os, Kind kind) {
  switch (kind) {
  case Kind::Attribute:
    os << "Attribute";
    break;
  case Kind::Operation:
    os << "Operation";
    break;
  case Kind::Type:
    os << "Type";
    break;
  case Kind::TypeRange:
    os << "TypeRange";
    break;
  case Kind::Value:
    os << "Value";
    break;
  case Kind::ValueRange:
    os << "ValueRange";
    break;
  }
}

//===----------------------------------------------------------------------===//
// PDLPatternModule
//===----------------------------------------------------------------------===//

void PDLPatternModule::mergeIn(PDLPatternModule &&other) {
  // Ignore the other module if it has no patterns.
  if (!other.pdlModule)
    return;

  // Steal the functions and config of the other module.
  for (auto &it : other.constraintFunctions)
    registerConstraintFunction(it.first(), std::move(it.second));
  for (auto &it : other.rewriteFunctions)
    registerRewriteFunction(it.first(), std::move(it.second));
  for (auto &it : other.configs)
    configs.emplace_back(std::move(it));
  for (auto &it : other.configMap)
    configMap.insert(it);

  // Steal the other state if we have no patterns.
  if (!pdlModule) {
    pdlModule = std::move(other.pdlModule);
    return;
  }

  // Merge the pattern operations from the other module into this one.
  Block *block = pdlModule->getBody();
  block->getOperations().splice(block->end(),
                                other.pdlModule->getBody()->getOperations());
}

void PDLPatternModule::attachConfigToPatterns(ModuleOp module,
                                              PDLPatternConfigSet &configSet) {
  // Attach the configuration to the symbols within the module. We only add
  // to symbols to avoid hardcoding any specific operation names here (given
  // that we don't depend on any PDL dialect). We can't use
  // cast<SymbolOpInterface> here because patterns may be optional symbols.
  module->walk([&](Operation *op) {
    if (op->hasTrait<SymbolOpInterface::Trait>())
      configMap[op] = &configSet;
  });
}

//===----------------------------------------------------------------------===//
// Function Registry

void PDLPatternModule::registerConstraintFunction(
    StringRef name, PDLConstraintFunction constraintFn) {
  // TODO: Is it possible to diagnose when `name` is already registered to
  // a function that is not equivalent to `constraintFn`?
  // Allow existing mappings in the case multiple patterns depend on the same
  // constraint.
  constraintFunctions.try_emplace(name, std::move(constraintFn));
}

void PDLPatternModule::registerRewriteFunction(StringRef name,
                                               PDLRewriteFunction rewriteFn) {
  // TODO: Is it possible to diagnose when `name` is already registered to
  // a function that is not equivalent to `rewriteFn`?
  // Allow existing mappings in the case multiple patterns depend on the same
  // rewrite.
  rewriteFunctions.try_emplace(name, std::move(rewriteFn));
}

//===----------------------------------------------------------------------===//
// RewriterBase
//===----------------------------------------------------------------------===//

bool RewriterBase::Listener::classof(const OpBuilder::Listener *base) {
  return base->getKind() == OpBuilder::ListenerBase::Kind::RewriterBaseListener;
}

RewriterBase::~RewriterBase() {
  // Out of line to provide a vtable anchor for the class.
}

/// This method replaces the uses of the results of `op` with the values in
/// `newValues` when the provided `functor` returns true for a specific use.
/// The number of values in `newValues` is required to match the number of
/// results of `op`.
void RewriterBase::replaceOpWithIf(
    Operation *op, ValueRange newValues, bool *allUsesReplaced,
    llvm::unique_function<bool(OpOperand &) const> functor) {
  assert(op->getNumResults() == newValues.size() &&
         "incorrect number of values to replace operation");

  // Notify the rewriter subclass that we're about to replace this root.
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyOperationReplaced(op, newValues);

  // Replace each use of the results when the functor is true.
  bool replacedAllUses = true;
  for (auto it : llvm::zip(op->getResults(), newValues)) {
    std::get<0>(it).replaceUsesWithIf(std::get<1>(it), functor);
    replacedAllUses &= std::get<0>(it).use_empty();
  }
  if (allUsesReplaced)
    *allUsesReplaced = replacedAllUses;
}

/// This method replaces the uses of the results of `op` with the values in
/// `newValues` when a use is nested within the given `block`. The number of
/// values in `newValues` is required to match the number of results of `op`.
/// If all uses of this operation are replaced, the operation is erased.
void RewriterBase::replaceOpWithinBlock(Operation *op, ValueRange newValues,
                                        Block *block, bool *allUsesReplaced) {
  replaceOpWithIf(op, newValues, allUsesReplaced, [block](OpOperand &use) {
    return block->getParentOp()->isProperAncestor(use.getOwner());
  });
}

/// This method replaces the results of the operation with the specified list of
/// values. The number of provided values must match the number of results of
/// the operation.
void RewriterBase::replaceOp(Operation *op, ValueRange newValues) {
  // Notify the rewriter subclass that we're about to replace this root.
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyOperationReplaced(op, newValues);

  assert(op->getNumResults() == newValues.size() &&
         "incorrect # of replacement values");
  op->replaceAllUsesWith(newValues);

  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyOperationRemoved(op);
  op->erase();
}

/// This method erases an operation that is known to have no uses. The uses of
/// the given operation *must* be known to be dead.
void RewriterBase::eraseOp(Operation *op) {
  assert(op->use_empty() && "expected 'op' to have no uses");
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyOperationRemoved(op);
  op->erase();
}

void RewriterBase::eraseBlock(Block *block) {
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
    assert(op.use_empty() && "expected 'op' to have no uses");
    eraseOp(&op);
  }
  block->erase();
}

/// Merge the operations of block 'source' into the end of block 'dest'.
/// 'source's predecessors must be empty or only contain 'dest`.
/// 'argValues' is used to replace the block arguments of 'source' after
/// merging.
void RewriterBase::mergeBlocks(Block *source, Block *dest,
                               ValueRange argValues) {
  assert(llvm::all_of(source->getPredecessors(),
                      [dest](Block *succ) { return succ == dest; }) &&
         "expected 'source' to have no predecessors or only 'dest'");
  assert(argValues.size() == source->getNumArguments() &&
         "incorrect # of argument replacement values");

  // Replace all of the successor arguments with the provided values.
  for (auto it : llvm::zip(source->getArguments(), argValues))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

  // Splice the operations of the 'source' block into the 'dest' block and erase
  // it.
  dest->getOperations().splice(dest->end(), source->getOperations());
  source->dropAllUses();
  source->erase();
}

/// Find uses of `from` and replace them with `to` if the `functor` returns
/// true. It also marks every modified uses and notifies the rewriter that an
/// in-place operation modification is about to happen.
void RewriterBase::replaceUseIf(
    Value from, Value to,
    llvm::unique_function<bool(OpOperand &) const> functor) {
  for (OpOperand &operand : llvm::make_early_inc_range(from.getUses())) {
    if (functor(operand))
      updateRootInPlace(operand.getOwner(), [&]() { operand.set(to); });
  }
}

// Merge the operations of block 'source' before the operation 'op'. Source
// block should not have existing predecessors or successors.
void RewriterBase::mergeBlockBefore(Block *source, Operation *op,
                                    ValueRange argValues) {
  assert(source->hasNoPredecessors() &&
         "expected 'source' to have no predecessors");
  assert(source->hasNoSuccessors() &&
         "expected 'source' to have no successors");

  // Split the block containing 'op' into two, one containing all operations
  // before 'op' (prologue) and another (epilogue) containing 'op' and all
  // operations after it.
  Block *prologue = op->getBlock();
  Block *epilogue = splitBlock(prologue, op->getIterator());

  // Merge the source block at the end of the prologue.
  mergeBlocks(source, prologue, argValues);

  // Merge the epilogue at the end the prologue.
  mergeBlocks(epilogue, prologue);
}

/// Split the operations starting at "before" (inclusive) out of the given
/// block into a new block, and return it.
Block *RewriterBase::splitBlock(Block *block, Block::iterator before) {
  return block->splitBlock(before);
}

/// 'op' and 'newOp' are known to have the same number of results, replace the
/// uses of op with uses of newOp
void RewriterBase::replaceOpWithResultsOfAnotherOp(Operation *op,
                                                   Operation *newOp) {
  assert(op->getNumResults() == newOp->getNumResults() &&
         "replacement op doesn't match results of original op");
  if (op->getNumResults() == 1)
    return replaceOp(op, newOp->getResult(0));
  return replaceOp(op, newOp->getResults());
}

/// Move the blocks that belong to "region" before the given position in
/// another region.  The two regions must be different.  The caller is in
/// charge to update create the operation transferring the control flow to the
/// region and pass it the correct block arguments.
void RewriterBase::inlineRegionBefore(Region &region, Region &parent,
                                      Region::iterator before) {
  parent.getBlocks().splice(before, region.getBlocks());
}
void RewriterBase::inlineRegionBefore(Region &region, Block *before) {
  inlineRegionBefore(region, *before->getParent(), before->getIterator());
}

/// Clone the blocks that belong to "region" before the given position in
/// another region "parent". The two regions must be different. The caller is
/// responsible for creating or updating the operation transferring flow of
/// control to the region and passing it the correct block arguments.
void RewriterBase::cloneRegionBefore(Region &region, Region &parent,
                                     Region::iterator before,
                                     IRMapping &mapping) {
  region.cloneInto(&parent, before, mapping);
}
void RewriterBase::cloneRegionBefore(Region &region, Region &parent,
                                     Region::iterator before) {
  IRMapping mapping;
  cloneRegionBefore(region, parent, before, mapping);
}
void RewriterBase::cloneRegionBefore(Region &region, Block *before) {
  cloneRegionBefore(region, *before->getParent(), before->getIterator());
}
