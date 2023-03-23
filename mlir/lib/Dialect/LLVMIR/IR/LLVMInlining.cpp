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
#include "mlir/Transforms/InliningUtils.h"

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

/// Move all alloca operations with a constant size in the former entry block of
/// the newly inlined callee into the entry block of the caller, and insert
/// lifetime intrinsics that limit their scope to the inlined blocks.
static void moveConstantAllocasToEntryBlock(
    iterator_range<Region::iterator> inlinedBlocks) {
  Block *calleeEntryBlock = &(*inlinedBlocks.begin());
  Block *callerEntryBlock = &(*calleeEntryBlock->getParent()->begin());
  if (calleeEntryBlock == callerEntryBlock)
    // Nothing to do.
    return;
  SmallVector<std::tuple<LLVM::AllocaOp, IntegerAttr, bool>> allocasToMove;
  bool shouldInsertLifetimes = false;
  // Conservatively only move alloca operations that are part of the entry block
  // and do not inspect nested regions, since they may execute conditionally or
  // have other unknown semantics.
  for (auto allocaOp : calleeEntryBlock->getOps<LLVM::AllocaOp>()) {
    IntegerAttr arraySize;
    if (!matchPattern(allocaOp.getArraySize(), m_Constant(&arraySize)))
      continue;
    bool shouldInsertLifetime =
        arraySize.getValue() != 0 && !hasLifetimeMarkers(allocaOp);
    shouldInsertLifetimes |= shouldInsertLifetime;
    allocasToMove.emplace_back(allocaOp, arraySize, shouldInsertLifetime);
  }
  if (allocasToMove.empty())
    return;
  OpBuilder builder(callerEntryBlock, callerEntryBlock->begin());
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
  if (!shouldInsertLifetimes)
    return;
  // Insert a lifetime end intrinsic before each return in the callee function.
  for (Block &block : inlinedBlocks) {
    if (!block.getTerminator()->hasTrait<OpTrait::ReturnLike>())
      continue;
    builder.setInsertionPoint(block.getTerminator());
    for (auto &[allocaOp, arraySize, shouldInsertLifetime] : allocasToMove) {
      if (!shouldInsertLifetime)
        continue;
      builder.create<LLVM::LifetimeEndOp>(
          allocaOp.getLoc(), arraySize.getValue().getLimitedValue(),
          allocaOp.getResult());
    }
  }
}

static Value handleByValArgument(OpBuilder &builder, Operation *callable,
                                 Value argument,
                                 NamedAttribute byValAttribute) {
  auto func = cast<LLVM::LLVMFuncOp>(callable);
  LLVM::MemoryEffectsAttr memoryEffects = func.getMemoryAttr();
  // If there is no memory effects attribute, assume that the function is
  // not read-only.
  bool isReadOnly = memoryEffects &&
                    memoryEffects.getArgMem() != LLVM::ModRefInfo::ModRef &&
                    memoryEffects.getArgMem() != LLVM::ModRefInfo::Mod;
  if (isReadOnly)
    return argument;
  // Resolve the pointee type and its size.
  auto ptrType = cast<LLVM::LLVMPointerType>(argument.getType());
  Type elementType = cast<TypeAttr>(byValAttribute.getValue()).getValue();
  unsigned int typeSize =
      DataLayout(callable->getParentOfType<DataLayoutOpInterface>())
          .getTypeSize(elementType);
  // Allocate the new value on the stack.
  Value one = builder.create<LLVM::ConstantOp>(
      func.getLoc(), builder.getI64Type(), builder.getI64IntegerAttr(1));
  Value allocaOp =
      builder.create<LLVM::AllocaOp>(func.getLoc(), ptrType, elementType, one);
  // Copy the pointee to the newly allocated value.
  Value copySize = builder.create<LLVM::ConstantOp>(
      func.getLoc(), builder.getI64Type(), builder.getI64IntegerAttr(typeSize));
  Value isVolatile = builder.create<LLVM::ConstantOp>(
      func.getLoc(), builder.getI1Type(), builder.getBoolAttr(false));
  builder.create<LLVM::MemcpyOp>(func.getLoc(), allocaOp, argument, copySize,
                                 isVolatile);
  return allocaOp;
}

namespace {
struct LLVMInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    if (!wouldBeCloned)
      return false;
    auto callOp = dyn_cast<LLVM::CallOp>(call);
    auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(callable);
    if (!callOp || !funcOp)
      return false;
    if (auto attrs = funcOp.getArgAttrs()) {
      for (Attribute attr : *attrs) {
        auto attrDict = cast<DictionaryAttr>(attr);
        for (NamedAttribute attr : attrDict) {
          if (attr.getName() == LLVM::LLVMDialect::getByValAttrName())
            continue;
          // TODO: Handle all argument attributes;
          return false;
        }
      }
    }
    // TODO: Handle result attributes;
    if (funcOp.getResAttrs())
      return false;
    // TODO: Handle exceptions.
    if (funcOp.getPersonality())
      return false;
    if (funcOp.getPassthrough()) {
      // TODO: Used attributes should not be passthrough.
      DenseSet<StringAttr> disallowed(
          {StringAttr::get(funcOp->getContext(), "noduplicate"),
           StringAttr::get(funcOp->getContext(), "noinline"),
           StringAttr::get(funcOp->getContext(), "optnone"),
           StringAttr::get(funcOp->getContext(), "presplitcoroutine"),
           StringAttr::get(funcOp->getContext(), "returns_twice"),
           StringAttr::get(funcOp->getContext(), "strictfp")});
      if (llvm::any_of(*funcOp.getPassthrough(), [&](Attribute attr) {
            auto stringAttr = dyn_cast<StringAttr>(attr);
            if (!stringAttr)
              return false;
            return disallowed.contains(stringAttr);
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
    if (auto iface = dyn_cast<LLVM::AliasAnalysisOpInterface>(op))
      if (iface.getAliasScopesOrNull() || iface.getNoAliasScopesOrNull())
        return false;
    if (auto iface = dyn_cast<LLVM::AccessGroupOpInterface>(op))
      if (iface.getAccessGroupsOrNull())
        return false;
    return isa<LLVM::CallOp, LLVM::AllocaOp, LLVM::LifetimeStartOp,
               LLVM::LifetimeEndOp, LLVM::LoadOp, LLVM::StoreOp>(op);
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
                       Value argument, Type targetType,
                       DictionaryAttr argumentAttrs) const final {
    if (auto attr =
            argumentAttrs.getNamed(LLVM::LLVMDialect::getByValAttrName()))
      return handleByValArgument(builder, callable, argument, *attr);
    return argument;
  }

  void processInlinedCallBlocks(
      Operation *call,
      iterator_range<Region::iterator> inlinedBlocks) const override {
    // Alloca operations with a constant size that were in the entry block of
    // the callee should be moved to the entry block of the caller, as this will
    // fold into prologue/epilogue code during code generation.
    // This is not implemented as a standalone pattern because we need to know
    // which newly inlined block was previously the entry block of the callee.
    moveConstantAllocasToEntryBlock(inlinedBlocks);
  }
};

} // end anonymous namespace

void LLVM::detail::addLLVMInlinerInterface(LLVM::LLVMDialect *dialect) {
  dialect->addInterfaces<LLVMInlinerInterface>();
}
