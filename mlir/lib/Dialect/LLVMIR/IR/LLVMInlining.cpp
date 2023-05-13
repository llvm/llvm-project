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
  Value isVolatile = builder.create<LLVM::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(false));
  builder.create<LLVM::MemcpyOp>(loc, allocaOp, argument, copySize, isVolatile);
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

/// Returns true if the given argument or result attribute is supported by the
/// inliner, false otherwise.
static bool isArgOrResAttrSupported(NamedAttribute attr) {
  if (attr.getName() == LLVM::LLVMDialect::getInAllocaAttrName())
    return false;
  if (attr.getName() == LLVM::LLVMDialect::getNoAliasAttrName())
    return false;
  return true;
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
    if (auto attrs = funcOp.getArgAttrs()) {
      for (DictionaryAttr attrDict : attrs->getAsRange<DictionaryAttr>()) {
        for (NamedAttribute attr : attrDict) {
          if (!isArgOrResAttrSupported(attr)) {
            LLVM_DEBUG(llvm::dbgs() << "Cannot inline " << funcOp.getSymName()
                                    << ": unhandled argument attribute "
                                    << attr.getName() << "\n");
            return false;
          }
        }
      }
    }
    if (auto attrs = funcOp.getResAttrs()) {
      for (DictionaryAttr attrDict : attrs->getAsRange<DictionaryAttr>()) {
        for (NamedAttribute attr : attrDict) {
          if (!isArgOrResAttrSupported(attr)) {
            LLVM_DEBUG(llvm::dbgs() << "Cannot inline " << funcOp.getSymName()
                                    << ": unhandled return attribute "
                                    << attr.getName() << "\n");
            return false;
          }
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
    // Some attributes on memory operations require handling during
    // inlining. Since this is not yet implemented, refuse to inline memory
    // operations that have any of these attributes.
    if (auto iface = dyn_cast<LLVM::AliasAnalysisOpInterface>(op)) {
      if (iface.getAliasScopesOrNull() || iface.getNoAliasScopesOrNull()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Cannot inline: unhandled alias analysis metadata\n");
        return false;
      }
    }
    if (auto iface = dyn_cast<LLVM::AccessGroupOpInterface>(op)) {
      if (iface.getAccessGroupsOrNull()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Cannot inline: unhandled access group metadata\n");
        return false;
      }
    }
    // clang-format off
    if (isa<LLVM::AllocaOp,
            LLVM::AtomicRMWOp,
            LLVM::AtomicCmpXchgOp,
            LLVM::CallOp,
            LLVM::DbgDeclareOp,
            LLVM::DbgValueOp,
            LLVM::FenceOp,
            LLVM::InlineAsmOp,
            LLVM::LifetimeEndOp,
            LLVM::LifetimeStartOp,
            LLVM::LoadOp,
            LLVM::MemcpyOp,
            LLVM::MemmoveOp,
            LLVM::MemsetOp,
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
    return argument;
  }

  void processInlinedCallBlocks(
      Operation *call,
      iterator_range<Region::iterator> inlinedBlocks) const override {
    handleInlinedAllocas(call, inlinedBlocks);
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
