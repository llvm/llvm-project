//===- LLVMInlining.cpp - LLVM inlining interface and logic -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Logic for inlining LLVM functions and the definition of the
// LLVMInliningInterface.
//
//===----------------------------------------------------------------------===//

#include "LLVMInlining.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llvm-inliner"

using namespace mlir;

/// Check whether the given alloca is an input to a lifetime intrinsic,
/// optionally passing through one or more casts on the way. This is not
/// transitive through block arguments.
static bool hasLifetimeMarkers(LLVM::AllocaOp allocaOp) {
  SmallVector<Operation *> stack(allocaOp->getUsers().begin(),
                                 allocaOp->getUsers().end());
  while (!stack.empty()) {
    Operation *op = stack.pop_back_val();
    if (isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(op))
      return true;
    if (isa<LLVM::BitcastOp>(op))
      stack.append(op->getUsers().begin(), op->getUsers().end());
  }
  return false;
}

/// Handles alloca operations in the inlined blocks:
/// - Moves all alloca operations with a constant size in the former entry block
///   of the callee into the entry block of the caller, so they become part of
///   the function prologue/epilogue during code generation.
/// - Inserts lifetime intrinsics that limit the scope of inlined static allocas
///   to the inlined blocks.
/// - Inserts StackSave and StackRestore operations if dynamic allocas were
///   inlined.
static void
handleInlinedAllocas(Operation *call,
                     iterator_range<Region::iterator> inlinedBlocks) {
  Block *calleeEntryBlock = &(*inlinedBlocks.begin());
  Block *callerEntryBlock = &(*calleeEntryBlock->getParent()->begin());
  if (calleeEntryBlock == callerEntryBlock)
    // Nothing to do.
    return;
  SmallVector<std::tuple<LLVM::AllocaOp, IntegerAttr, bool>> allocasToMove;
  bool shouldInsertLifetimes = false;
  bool hasDynamicAlloca = false;
  // Conservatively only move static alloca operations that are part of the
  // entry block and do not inspect nested regions, since they may execute
  // conditionally or have other unknown semantics.
  for (auto allocaOp : calleeEntryBlock->getOps<LLVM::AllocaOp>()) {
    IntegerAttr arraySize;
    if (!matchPattern(allocaOp.getArraySize(), m_Constant(&arraySize))) {
      hasDynamicAlloca = true;
      continue;
    }
    bool shouldInsertLifetime =
        arraySize.getValue() != 0 && !hasLifetimeMarkers(allocaOp);
    shouldInsertLifetimes |= shouldInsertLifetime;
    allocasToMove.emplace_back(allocaOp, arraySize, shouldInsertLifetime);
  }
  // Check the remaining inlined blocks for dynamic allocas as well.
  for (Block &block : llvm::drop_begin(inlinedBlocks)) {
    if (hasDynamicAlloca)
      break;
    hasDynamicAlloca =
        llvm::any_of(block.getOps<LLVM::AllocaOp>(), [](auto allocaOp) {
          return !matchPattern(allocaOp.getArraySize(), m_Constant());
        });
  }
  if (allocasToMove.empty() && !hasDynamicAlloca)
    return;
  OpBuilder builder(calleeEntryBlock, calleeEntryBlock->begin());
  Value stackPtr;
  if (hasDynamicAlloca) {
    // This may result in multiple stacksave/stackrestore intrinsics in the same
    // scope if some are already present in the body of the caller. This is not
    // invalid IR, but LLVM cleans these up in InstCombineCalls.cpp, along with
    // other cases where the stacksave/stackrestore is redundant.
    stackPtr = builder.create<LLVM::StackSaveOp>(
        call->getLoc(), LLVM::LLVMPointerType::get(call->getContext()));
  }
  builder.setInsertionPoint(callerEntryBlock, callerEntryBlock->begin());
  for (auto &[allocaOp, arraySize, shouldInsertLifetime] : allocasToMove) {
    auto newConstant = builder.create<LLVM::ConstantOp>(
        allocaOp->getLoc(), allocaOp.getArraySize().getType(), arraySize);
    // Insert a lifetime start intrinsic where the alloca was before moving it.
    if (shouldInsertLifetime) {
      OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPoint(allocaOp);
      builder.create<LLVM::LifetimeStartOp>(
          allocaOp.getLoc(), arraySize.getValue().getLimitedValue(),
          allocaOp.getResult());
    }
    allocaOp->moveAfter(newConstant);
    allocaOp.getArraySizeMutable().assign(newConstant.getResult());
  }
  if (!shouldInsertLifetimes && !hasDynamicAlloca)
    return;
  // Insert a lifetime end intrinsic before each return in the callee function.
  for (Block &block : inlinedBlocks) {
    if (!block.getTerminator()->hasTrait<OpTrait::ReturnLike>())
      continue;
    builder.setInsertionPoint(block.getTerminator());
    if (hasDynamicAlloca)
      builder.create<LLVM::StackRestoreOp>(call->getLoc(), stackPtr);
    for (auto &[allocaOp, arraySize, shouldInsertLifetime] : allocasToMove) {
      if (shouldInsertLifetime)
        builder.create<LLVM::LifetimeEndOp>(
            allocaOp.getLoc(), arraySize.getValue().getLimitedValue(),
            allocaOp.getResult());
    }
  }
}

/// Maps all alias scopes in the inlined operations to deep clones of the scopes
/// and domain. This is required for code such as `foo(a, b); foo(a2, b2);` to
/// not incorrectly return `noalias` for e.g. operations on `a` and `a2`.
static void
deepCloneAliasScopes(iterator_range<Region::iterator> inlinedBlocks) {
  DenseMap<Attribute, Attribute> mapping;

  // Register handles in the walker to create the deep clones.
  // The walker ensures that an attribute is only ever walked once and does a
  // post-order walk, ensuring the domain is visited prior to the scope.
  AttrTypeWalker walker;

  // Perform the deep clones while visiting. Builders create a distinct
  // attribute to make sure that new instances are always created by the
  // uniquer.
  walker.addWalk([&](LLVM::AliasScopeDomainAttr domainAttr) {
    mapping[domainAttr] = LLVM::AliasScopeDomainAttr::get(
        domainAttr.getContext(), domainAttr.getDescription());
  });

  walker.addWalk([&](LLVM::AliasScopeAttr scopeAttr) {
    mapping[scopeAttr] = LLVM::AliasScopeAttr::get(
        cast<LLVM::AliasScopeDomainAttr>(mapping.lookup(scopeAttr.getDomain())),
        scopeAttr.getDescription());
  });

  // Map an array of scopes to an array of deep clones.
  auto convertScopeList = [&](ArrayAttr arrayAttr) -> ArrayAttr {
    if (!arrayAttr)
      return nullptr;

    // Create the deep clones if necessary.
    walker.walk(arrayAttr);

    return ArrayAttr::get(arrayAttr.getContext(),
                          llvm::map_to_vector(arrayAttr, [&](Attribute attr) {
                            return mapping.lookup(attr);
                          }));
  };

  for (Block &block : inlinedBlocks) {
    for (Operation &op : block) {
      if (auto aliasInterface = dyn_cast<LLVM::AliasAnalysisOpInterface>(op)) {
        aliasInterface.setAliasScopes(
            convertScopeList(aliasInterface.getAliasScopesOrNull()));
        aliasInterface.setNoAliasScopes(
            convertScopeList(aliasInterface.getNoAliasScopesOrNull()));
      }

      if (auto noAliasScope = dyn_cast<LLVM::NoAliasScopeDeclOp>(op)) {
        // Create the deep clones if necessary.
        walker.walk(noAliasScope.getScopeAttr());

        noAliasScope.setScopeAttr(cast<LLVM::AliasScopeAttr>(
            mapping.lookup(noAliasScope.getScopeAttr())));
      }
    }
  }
}

/// Creates a new ArrayAttr by concatenating `lhs` with `rhs`.
/// Returns null if both parameters are null. If only one attribute is null,
/// return the other.
static ArrayAttr concatArrayAttr(ArrayAttr lhs, ArrayAttr rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;

  SmallVector<Attribute> result;
  llvm::append_range(result, lhs);
  llvm::append_range(result, rhs);
  return ArrayAttr::get(lhs.getContext(), result);
}

/// Attempts to return the underlying pointer value that `pointerValue` is based
/// on. This traverses down the chain of operations to the last operation
/// producing the base pointer and returns it. If it encounters an operation it
/// cannot further traverse through, returns the operation's result.
static Value getUnderlyingObject(Value pointerValue) {
  while (true) {
    if (auto gepOp = pointerValue.getDefiningOp<LLVM::GEPOp>()) {
      pointerValue = gepOp.getBase();
      continue;
    }

    if (auto addrCast = pointerValue.getDefiningOp<LLVM::AddrSpaceCastOp>()) {
      pointerValue = addrCast.getOperand();
      continue;
    }

    break;
  }

  return pointerValue;
}

/// Attempts to return the set of all underlying pointer values that
/// `pointerValue` is based on. This function traverses through select
/// operations and block arguments unlike getUnderlyingObject.
static SmallVector<Value> getUnderlyingObjectSet(Value pointerValue) {
  SmallVector<Value> result;

  SmallVector<Value> workList{pointerValue};
  // Avoid dataflow loops.
  SmallPtrSet<Value, 4> seen;
  do {
    Value current = workList.pop_back_val();
    current = getUnderlyingObject(current);

    if (!seen.insert(current).second)
      continue;

    if (auto selectOp = current.getDefiningOp<LLVM::SelectOp>()) {
      workList.push_back(selectOp.getTrueValue());
      workList.push_back(selectOp.getFalseValue());
      continue;
    }

    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      Block *parentBlock = blockArg.getParentBlock();

      // Attempt to find all block argument operands for every predecessor.
      // If any operand to the block argument wasn't found in a predecessor,
      // conservatively add the block argument to the result set.
      SmallVector<Value> operands;
      bool anyUnknown = false;
      for (auto iter = parentBlock->pred_begin();
           iter != parentBlock->pred_end(); iter++) {
        auto branch = dyn_cast<BranchOpInterface>((*iter)->getTerminator());
        if (!branch) {
          result.push_back(blockArg);
          anyUnknown = true;
          break;
        }

        Value operand = branch.getSuccessorOperands(
            iter.getSuccessorIndex())[blockArg.getArgNumber()];
        if (!operand) {
          result.push_back(blockArg);
          anyUnknown = true;
          break;
        }

        operands.push_back(operand);
      }

      if (!anyUnknown)
        llvm::append_range(workList, operands);

      continue;
    }

    result.push_back(current);
  } while (!workList.empty());

  return result;
}

/// Creates a new AliasScopeAttr for every noalias parameter and attaches it to
/// the appropriate inlined memory operations in an attempt to preserve the
/// original semantics of the parameter attribute.
static void createNewAliasScopesFromNoAliasParameter(
    Operation *call, iterator_range<Region::iterator> inlinedBlocks) {

  // First collect all noalias parameters. These have been specially marked by
  // the `handleArgument` implementation by using the `ssa.copy` intrinsic and
  // attaching a `noalias` attribute to it.
  // These are only meant to be temporary and should therefore be deleted after
  // we're done using them here.
  SetVector<LLVM::SSACopyOp> noAliasParams;
  for (Value argument : cast<LLVM::CallOp>(call).getArgOperands()) {
    for (Operation *user : argument.getUsers()) {
      auto ssaCopy = llvm::dyn_cast<LLVM::SSACopyOp>(user);
      if (!ssaCopy)
        continue;
      if (!ssaCopy->hasAttr(LLVM::LLVMDialect::getNoAliasAttrName()))
        continue;

      noAliasParams.insert(ssaCopy);
    }
  }

  // If there were none, we have nothing to do here.
  if (noAliasParams.empty())
    return;

  // Scope exit block to make it impossible to forget to get rid of the
  // intrinsics.
  auto exit = llvm::make_scope_exit([&] {
    for (LLVM::SSACopyOp ssaCopyOp : noAliasParams) {
      ssaCopyOp.replaceAllUsesWith(ssaCopyOp.getOperand());
      ssaCopyOp->erase();
    }
  });

  // Create a new domain for this specific inlining and a new scope for every
  // noalias parameter.
  auto functionDomain = LLVM::AliasScopeDomainAttr::get(
      call->getContext(), cast<LLVM::CallOp>(call).getCalleeAttr().getAttr());
  DenseMap<Value, LLVM::AliasScopeAttr> pointerScopes;
  for (LLVM::SSACopyOp copyOp : noAliasParams) {
    auto scope = LLVM::AliasScopeAttr::get(functionDomain);
    pointerScopes[copyOp] = scope;

    OpBuilder(call).create<LLVM::NoAliasScopeDeclOp>(call->getLoc(), scope);
  }

  // Go through every instruction and attempt to find which noalias parameters
  // it is definitely based on and definitely not based on.
  for (Block &inlinedBlock : inlinedBlocks) {
    for (auto aliasInterface :
         inlinedBlock.getOps<LLVM::AliasAnalysisOpInterface>()) {

      // Collect the pointer arguments affected by the alias scopes.
      SmallVector<Value> pointerArgs = aliasInterface.getAccessedOperands();

      // Find the set of underlying pointers that this pointer is based on.
      SmallPtrSet<Value, 4> basedOnPointers;
      for (Value pointer : pointerArgs)
        llvm::copy(getUnderlyingObjectSet(pointer),
                   std::inserter(basedOnPointers, basedOnPointers.begin()));

      bool aliasesOtherKnownObject = false;
      // Go through the based on pointers and check that they are either:
      // * Constants that can be ignored (undef, poison, null pointer).
      // * Based on a noalias parameter.
      // * Other pointers that we know can't alias with our noalias parameter.
      //
      // Any other value might be a pointer based on any noalias parameter that
      // hasn't been identified. In that case conservatively don't add any
      // scopes to this operation indicating either aliasing or not aliasing
      // with any parameter.
      if (llvm::any_of(basedOnPointers, [&](Value object) {
            if (matchPattern(object, m_Constant()))
              return false;

            if (noAliasParams.contains(object.getDefiningOp<LLVM::SSACopyOp>()))
              return false;

            // TODO: This should include other arguments from the inlined
            //       callable.
            if (isa_and_nonnull<LLVM::AllocaOp, LLVM::AddressOfOp>(
                    object.getDefiningOp())) {
              aliasesOtherKnownObject = true;
              return false;
            }
            return true;
          }))
        continue;

      // Add all noalias parameter scopes to the noalias scope list that we are
      // not based on.
      SmallVector<Attribute> noAliasScopes;
      for (LLVM::SSACopyOp noAlias : noAliasParams) {
        if (basedOnPointers.contains(noAlias))
          continue;

        noAliasScopes.push_back(pointerScopes[noAlias]);
      }

      if (!noAliasScopes.empty())
        aliasInterface.setNoAliasScopes(
            concatArrayAttr(aliasInterface.getNoAliasScopesOrNull(),
                            ArrayAttr::get(call->getContext(), noAliasScopes)));

      // Don't add alias scopes to call operations or operations that might
      // operate on pointers not based on any noalias parameter.
      // Since we add all scopes to an operation's noalias list that it
      // definitely doesn't alias, we mustn't do the same for the alias.scope
      // list if other objects are involved.
      //
      // Consider the following case:
      // %0 = llvm.alloca
      // %1 = select %magic, %0, %noalias_param
      // store 5, %1  (1) noalias=[scope(...)]
      // ...
      // store 3, %0  (2) noalias=[scope(noalias_param), scope(...)]
      //
      // We can add the scopes of any noalias parameters that aren't
      // noalias_param's scope to (1) and add all of them to (2). We mustn't add
      // the scope of noalias_param to the alias.scope list of (1) since
      // that would mean (2) cannot alias with (1) which is wrong since both may
      // store to %0.
      //
      // In conclusion, only add scopes to the alias.scope list if all pointers
      // have a corresponding scope.
      // Call operations are included in this list since we do not know whether
      // the callee accesses any memory besides the ones passed as its
      // arguments.
      if (aliasesOtherKnownObject ||
          isa<LLVM::CallOp>(aliasInterface.getOperation()))
        continue;

      SmallVector<Attribute> aliasScopes;
      for (LLVM::SSACopyOp noAlias : noAliasParams)
        if (basedOnPointers.contains(noAlias))
          aliasScopes.push_back(pointerScopes[noAlias]);

      if (!aliasScopes.empty())
        aliasInterface.setAliasScopes(
            concatArrayAttr(aliasInterface.getAliasScopesOrNull(),
                            ArrayAttr::get(call->getContext(), aliasScopes)));
    }
  }
}

/// Appends any alias scopes of the call operation to any inlined memory
/// operation.
static void
appendCallOpAliasScopes(Operation *call,
                        iterator_range<Region::iterator> inlinedBlocks) {
  auto callAliasInterface = dyn_cast<LLVM::AliasAnalysisOpInterface>(call);
  if (!callAliasInterface)
    return;

  ArrayAttr aliasScopes = callAliasInterface.getAliasScopesOrNull();
  ArrayAttr noAliasScopes = callAliasInterface.getNoAliasScopesOrNull();
  // If the call has neither alias scopes or noalias scopes we have nothing to
  // do here.
  if (!aliasScopes && !noAliasScopes)
    return;

  // Simply append the call op's alias and noalias scopes to any operation
  // implementing AliasAnalysisOpInterface.
  for (Block &block : inlinedBlocks) {
    for (auto aliasInterface : block.getOps<LLVM::AliasAnalysisOpInterface>()) {
      if (aliasScopes)
        aliasInterface.setAliasScopes(concatArrayAttr(
            aliasInterface.getAliasScopesOrNull(), aliasScopes));

      if (noAliasScopes)
        aliasInterface.setNoAliasScopes(concatArrayAttr(
            aliasInterface.getNoAliasScopesOrNull(), noAliasScopes));
    }
  }
}

/// Handles all interactions with alias scopes during inlining.
static void handleAliasScopes(Operation *call,
                              iterator_range<Region::iterator> inlinedBlocks) {
  deepCloneAliasScopes(inlinedBlocks);
  createNewAliasScopesFromNoAliasParameter(call, inlinedBlocks);
  appendCallOpAliasScopes(call, inlinedBlocks);
}

/// Appends any access groups of the call operation to any inlined memory
/// operation.
static void handleAccessGroups(Operation *call,
                               iterator_range<Region::iterator> inlinedBlocks) {
  auto callAccessGroupInterface = dyn_cast<LLVM::AccessGroupOpInterface>(call);
  if (!callAccessGroupInterface)
    return;

  auto accessGroups = callAccessGroupInterface.getAccessGroupsOrNull();
  if (!accessGroups)
    return;

  // Simply append the call op's access groups to any operation implementing
  // AccessGroupOpInterface.
  for (Block &block : inlinedBlocks)
    for (auto accessGroupOpInterface :
         block.getOps<LLVM::AccessGroupOpInterface>())
      accessGroupOpInterface.setAccessGroups(concatArrayAttr(
          accessGroupOpInterface.getAccessGroupsOrNull(), accessGroups));
}

/// If `requestedAlignment` is higher than the alignment specified on `alloca`,
/// realigns `alloca` if this does not exceed the natural stack alignment.
/// Returns the post-alignment of `alloca`, whether it was realigned or not.
static unsigned tryToEnforceAllocaAlignment(LLVM::AllocaOp alloca,
                                            unsigned requestedAlignment,
                                            DataLayout const &dataLayout) {
  unsigned allocaAlignment = alloca.getAlignment().value_or(1);
  if (requestedAlignment <= allocaAlignment)
    // No realignment necessary.
    return allocaAlignment;
  unsigned naturalStackAlignmentBits = dataLayout.getStackAlignment();
  // If the natural stack alignment is not specified, the data layout returns
  // zero. Optimistically allow realignment in this case.
  if (naturalStackAlignmentBits == 0 ||
      // If the requested alignment exceeds the natural stack alignment, this
      // will trigger a dynamic stack realignment, so we prefer to copy...
      8 * requestedAlignment <= naturalStackAlignmentBits ||
      // ...unless the alloca already triggers dynamic stack realignment. Then
      // we might as well further increase the alignment to avoid a copy.
      8 * allocaAlignment > naturalStackAlignmentBits) {
    alloca.setAlignment(requestedAlignment);
    allocaAlignment = requestedAlignment;
  }
  return allocaAlignment;
}

/// Tries to find and return the alignment of the pointer `value` by looking for
/// an alignment attribute on the defining allocation op or function argument.
/// If the found alignment is lower than `requestedAlignment`, tries to realign
/// the pointer, then returns the resulting post-alignment, regardless of
/// whether it was realigned or not. If no existing alignment attribute is
/// found, returns 1 (i.e., assume that no alignment is guaranteed).
static unsigned tryToEnforceAlignment(Value value, unsigned requestedAlignment,
                                      DataLayout const &dataLayout) {
  if (Operation *definingOp = value.getDefiningOp()) {
    if (auto alloca = dyn_cast<LLVM::AllocaOp>(definingOp))
      return tryToEnforceAllocaAlignment(alloca, requestedAlignment,
                                         dataLayout);
    if (auto addressOf = dyn_cast<LLVM::AddressOfOp>(definingOp))
      if (auto global = SymbolTable::lookupNearestSymbolFrom<LLVM::GlobalOp>(
              definingOp, addressOf.getGlobalNameAttr()))
        return global.getAlignment().value_or(1);
    // We don't currently handle this operation; assume no alignment.
    return 1;
  }
  // Since there is no defining op, this is a block argument. Probably this
  // comes directly from a function argument, so check that this is the case.
  Operation *parentOp = value.getParentBlock()->getParentOp();
  if (auto func = dyn_cast<LLVM::LLVMFuncOp>(parentOp)) {
    // Use the alignment attribute set for this argument in the parent function
    // if it has been set.
    auto blockArg = llvm::cast<BlockArgument>(value);
    if (Attribute alignAttr = func.getArgAttr(
            blockArg.getArgNumber(), LLVM::LLVMDialect::getAlignAttrName()))
      return cast<IntegerAttr>(alignAttr).getValue().getLimitedValue();
  }
  // We didn't find anything useful; assume no alignment.
  return 1;
}

/// Introduces a new alloca and copies the memory pointed to by `argument` to
/// the address of the new alloca, then returns the value of the new alloca.
static Value handleByValArgumentInit(OpBuilder &builder, Location loc,
                                     Value argument, Type elementType,
                                     unsigned elementTypeSize,
                                     unsigned targetAlignment) {
  // Allocate the new value on the stack.
  Value allocaOp;
  {
    // Since this is a static alloca, we can put it directly in the entry block,
    // so they can be absorbed into the prologue/epilogue at code generation.
    OpBuilder::InsertionGuard insertionGuard(builder);
    Block *entryBlock = &(*argument.getParentRegion()->begin());
    builder.setInsertionPointToStart(entryBlock);
    Value one = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                 builder.getI64IntegerAttr(1));
    allocaOp = builder.create<LLVM::AllocaOp>(
        loc, argument.getType(), elementType, one, targetAlignment);
  }
  // Copy the pointee to the newly allocated value.
  Value copySize = builder.create<LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(elementTypeSize));
  builder.create<LLVM::MemcpyOp>(loc, allocaOp, argument, copySize,
                                 /*isVolatile=*/false);
  return allocaOp;
}

/// Handles a function argument marked with the byval attribute by introducing a
/// memcpy or realigning the defining operation, if required either due to the
/// pointee being writeable in the callee, and/or due to an alignment mismatch.
/// `requestedAlignment` specifies the alignment set in the "align" argument
/// attribute (or 1 if no align attribute was set).
static Value handleByValArgument(OpBuilder &builder, Operation *callable,
                                 Value argument, Type elementType,
                                 unsigned requestedAlignment) {
  auto func = cast<LLVM::LLVMFuncOp>(callable);
  LLVM::MemoryEffectsAttr memoryEffects = func.getMemoryAttr();
  // If there is no memory effects attribute, assume that the function is
  // not read-only.
  bool isReadOnly = memoryEffects &&
                    memoryEffects.getArgMem() != LLVM::ModRefInfo::ModRef &&
                    memoryEffects.getArgMem() != LLVM::ModRefInfo::Mod;
  // Check if there's an alignment mismatch requiring us to copy.
  DataLayout dataLayout = DataLayout::closest(callable);
  unsigned minimumAlignment = dataLayout.getTypeABIAlignment(elementType);
  if (isReadOnly) {
    if (requestedAlignment <= minimumAlignment)
      return argument;
    unsigned currentAlignment =
        tryToEnforceAlignment(argument, requestedAlignment, dataLayout);
    if (currentAlignment >= requestedAlignment)
      return argument;
  }
  unsigned targetAlignment = std::max(requestedAlignment, minimumAlignment);
  return handleByValArgumentInit(builder, func.getLoc(), argument, elementType,
                                 dataLayout.getTypeSize(elementType),
                                 targetAlignment);
}

namespace {
struct LLVMInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  LLVMInlinerInterface(Dialect *dialect)
      : DialectInlinerInterface(dialect),
        // Cache set of StringAttrs for fast lookup in `isLegalToInline`.
        disallowedFunctionAttrs({
            StringAttr::get(dialect->getContext(), "noduplicate"),
            StringAttr::get(dialect->getContext(), "noinline"),
            StringAttr::get(dialect->getContext(), "optnone"),
            StringAttr::get(dialect->getContext(), "presplitcoroutine"),
            StringAttr::get(dialect->getContext(), "returns_twice"),
            StringAttr::get(dialect->getContext(), "strictfp"),
        }) {}

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    if (!wouldBeCloned)
      return false;
    auto callOp = dyn_cast<LLVM::CallOp>(call);
    if (!callOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot inline: call is not an LLVM::CallOp\n");
      return false;
    }
    auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(callable);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot inline: callable is not an LLVM::LLVMFuncOp\n");
      return false;
    }
    // TODO: Generate aliasing metadata from noalias argument/result attributes.
    if (auto attrs = funcOp.getArgAttrs()) {
      for (DictionaryAttr attrDict : attrs->getAsRange<DictionaryAttr>()) {
        if (attrDict.contains(LLVM::LLVMDialect::getInAllocaAttrName())) {
          LLVM_DEBUG(llvm::dbgs() << "Cannot inline " << funcOp.getSymName()
                                  << ": inalloca arguments not supported\n");
          return false;
        }
      }
    }
    // TODO: Handle exceptions.
    if (funcOp.getPersonality()) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot inline " << funcOp.getSymName()
                              << ": unhandled function personality\n");
      return false;
    }
    if (funcOp.getPassthrough()) {
      // TODO: Used attributes should not be passthrough.
      if (llvm::any_of(*funcOp.getPassthrough(), [&](Attribute attr) {
            auto stringAttr = dyn_cast<StringAttr>(attr);
            if (!stringAttr)
              return false;
            if (disallowedFunctionAttrs.contains(stringAttr)) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Cannot inline " << funcOp.getSymName()
                         << ": found disallowed function attribute "
                         << stringAttr << "\n");
              return true;
            }
            return false;
          }))
        return false;
    }
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  /// Conservative allowlist of operations supported so far.
  bool isLegalToInline(Operation *op, Region *, bool, IRMapping &) const final {
    if (isPure(op))
      return true;
    // clang-format off
    if (isa<LLVM::AllocaOp,
            LLVM::AssumeOp,
            LLVM::AtomicRMWOp,
            LLVM::AtomicCmpXchgOp,
            LLVM::CallOp,
            LLVM::DbgDeclareOp,
            LLVM::DbgLabelOp,
            LLVM::DbgValueOp,
            LLVM::FenceOp,
            LLVM::InlineAsmOp,
            LLVM::LifetimeEndOp,
            LLVM::LifetimeStartOp,
            LLVM::LoadOp,
            LLVM::MemcpyOp,
            LLVM::MemcpyInlineOp,
            LLVM::MemmoveOp,
            LLVM::MemsetOp,
            LLVM::NoAliasScopeDeclOp,
            LLVM::StackRestoreOp,
            LLVM::StackSaveOp,
            LLVM::StoreOp,
            LLVM::UnreachableOp>(op))
      return true;
    // clang-format on
    LLVM_DEBUG(llvm::dbgs()
               << "Cannot inline: unhandled side effecting operation \""
               << op->getName() << "\"\n");
    return false;
  }

  /// Handle the given inlined return by replacing it with a branch. This
  /// overload is called when the inlined region has more than one block.
  void handleTerminator(Operation *op, Block *newDest) const final {
    // Only return needs to be handled here.
    auto returnOp = dyn_cast<LLVM::ReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the return with a branch to the dest.
    OpBuilder builder(op);
    builder.create<LLVM::BrOp>(op->getLoc(), returnOp.getOperands(), newDest);
    op->erase();
  }

  /// Handle the given inlined return by replacing the uses of the call with the
  /// operands of the return. This overload is called when the inlined region
  /// only contains one block.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Return will be the only terminator present.
    auto returnOp = cast<LLVM::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &[dst, src] :
         llvm::zip(valuesToRepl, returnOp.getOperands()))
      dst.replaceAllUsesWith(src);
  }

  Value handleArgument(OpBuilder &builder, Operation *call, Operation *callable,
                       Value argument,
                       DictionaryAttr argumentAttrs) const final {
    if (std::optional<NamedAttribute> attr =
            argumentAttrs.getNamed(LLVM::LLVMDialect::getByValAttrName())) {
      Type elementType = cast<TypeAttr>(attr->getValue()).getValue();
      unsigned requestedAlignment = 1;
      if (std::optional<NamedAttribute> alignAttr =
              argumentAttrs.getNamed(LLVM::LLVMDialect::getAlignAttrName())) {
        requestedAlignment = cast<IntegerAttr>(alignAttr->getValue())
                                 .getValue()
                                 .getLimitedValue();
      }
      return handleByValArgument(builder, callable, argument, elementType,
                                 requestedAlignment);
    }
    if ([[maybe_unused]] std::optional<NamedAttribute> attr =
            argumentAttrs.getNamed(LLVM::LLVMDialect::getNoAliasAttrName())) {
      if (argument.use_empty())
        return argument;

      // This code is essentially a workaround for deficiencies in the
      // inliner interface: We need to transform operations *after* inlined
      // based on the argument attributes of the parameters *before* inlining.
      // This method runs prior to actual inlining and thus cannot transform the
      // post-inlining code, while `processInlinedCallBlocks` does not have
      // access to pre-inlining function arguments. Additionally, it is required
      // to distinguish which parameter an SSA value originally came from.
      // As a workaround until this is changed: Create an ssa.copy intrinsic
      // with the noalias attribute that can easily be found, and is extremely
      // unlikely to exist in the code prior to inlining, using this to
      // communicate between this method and `processInlinedCallBlocks`.
      // TODO: Fix this by refactoring the inliner interface.
      auto copyOp = builder.create<LLVM::SSACopyOp>(call->getLoc(), argument);
      copyOp->setDiscardableAttr(
          builder.getStringAttr(LLVM::LLVMDialect::getNoAliasAttrName()),
          builder.getUnitAttr());
      return copyOp;
    }
    return argument;
  }

  void processInlinedCallBlocks(
      Operation *call,
      iterator_range<Region::iterator> inlinedBlocks) const override {
    handleInlinedAllocas(call, inlinedBlocks);
    handleAliasScopes(call, inlinedBlocks);
    handleAccessGroups(call, inlinedBlocks);
  }

  // Keeping this (immutable) state on the interface allows us to look up
  // StringAttrs instead of looking up strings, since StringAttrs are bound to
  // the current context and thus cannot be initialized as static fields.
  const DenseSet<StringAttr> disallowedFunctionAttrs;
};

} // end anonymous namespace

void LLVM::detail::addLLVMInlinerInterface(LLVM::LLVMDialect *dialect) {
  dialect->addInterfaces<LLVMInlinerInterface>();
}
