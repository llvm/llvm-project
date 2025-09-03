//===- PtrToLLVMIRTranslation.cpp - Translate `ptr` to LLVM IR ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR `ptr` dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/Ptr/PtrToLLVMIRTranslation.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

using namespace mlir;
using namespace mlir::ptr;

namespace {

/// Converts ptr::AtomicOrdering to llvm::AtomicOrdering
static llvm::AtomicOrdering
convertAtomicOrdering(ptr::AtomicOrdering ordering) {
  switch (ordering) {
  case ptr::AtomicOrdering::not_atomic:
    return llvm::AtomicOrdering::NotAtomic;
  case ptr::AtomicOrdering::unordered:
    return llvm::AtomicOrdering::Unordered;
  case ptr::AtomicOrdering::monotonic:
    return llvm::AtomicOrdering::Monotonic;
  case ptr::AtomicOrdering::acquire:
    return llvm::AtomicOrdering::Acquire;
  case ptr::AtomicOrdering::release:
    return llvm::AtomicOrdering::Release;
  case ptr::AtomicOrdering::acq_rel:
    return llvm::AtomicOrdering::AcquireRelease;
  case ptr::AtomicOrdering::seq_cst:
    return llvm::AtomicOrdering::SequentiallyConsistent;
  }
  llvm_unreachable("Unknown atomic ordering");
}

/// Convert ptr.ptr_add operation
static LogicalResult
convertPtrAddOp(PtrAddOp ptrAddOp, llvm::IRBuilderBase &builder,
                LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *basePtr = moduleTranslation.lookupValue(ptrAddOp.getBase());
  llvm::Value *offset = moduleTranslation.lookupValue(ptrAddOp.getOffset());

  if (!basePtr || !offset)
    return ptrAddOp.emitError("Failed to lookup operands");

  // Create the GEP flags
  llvm::GEPNoWrapFlags gepFlags;
  switch (ptrAddOp.getFlags()) {
  case ptr::PtrAddFlags::none:
    break;
  case ptr::PtrAddFlags::nusw:
    gepFlags = llvm::GEPNoWrapFlags::noUnsignedSignedWrap();
    break;
  case ptr::PtrAddFlags::nuw:
    gepFlags = llvm::GEPNoWrapFlags::noUnsignedWrap();
    break;
  case ptr::PtrAddFlags::inbounds:
    gepFlags = llvm::GEPNoWrapFlags::inBounds();
    break;
  }

  // Create GEP instruction for pointer arithmetic
  llvm::Value *gep =
      builder.CreateGEP(builder.getInt8Ty(), basePtr, {offset}, "", gepFlags);

  moduleTranslation.mapValue(ptrAddOp.getResult(), gep);
  return success();
}

/// Convert ptr.load operation
static LogicalResult convertLoadOp(LoadOp loadOp, llvm::IRBuilderBase &builder,
                                   LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *ptr = moduleTranslation.lookupValue(loadOp.getPtr());
  if (!ptr)
    return loadOp.emitError("Failed to lookup pointer operand");

  // Convert result type to LLVM type
  llvm::Type *resultType =
      moduleTranslation.convertType(loadOp.getValue().getType());
  if (!resultType)
    return loadOp.emitError("Failed to convert result type");

  // Create the load instruction.
  llvm::MaybeAlign alignment(loadOp.getAlignment().value_or(0));
  llvm::LoadInst *loadInst = builder.CreateAlignedLoad(
      resultType, ptr, alignment, loadOp.getVolatile_());

  // Set op flags and metadata.
  loadInst->setAtomic(convertAtomicOrdering(loadOp.getOrdering()));
  // Set sync scope if specified
  if (loadOp.getSyncscope().has_value()) {
    llvm::LLVMContext &ctx = builder.getContext();
    llvm::SyncScope::ID syncScope =
        ctx.getOrInsertSyncScopeID(loadOp.getSyncscope().value());
    loadInst->setSyncScopeID(syncScope);
  }

  // Set metadata for nontemporal, invariant, and invariant_group
  if (loadOp.getNontemporal()) {
    llvm::MDNode *nontemporalMD =
        llvm::MDNode::get(builder.getContext(),
                          llvm::ConstantAsMetadata::get(builder.getInt32(1)));
    loadInst->setMetadata(llvm::LLVMContext::MD_nontemporal, nontemporalMD);
  }

  if (loadOp.getInvariant()) {
    llvm::MDNode *invariantMD = llvm::MDNode::get(builder.getContext(), {});
    loadInst->setMetadata(llvm::LLVMContext::MD_invariant_load, invariantMD);
  }

  if (loadOp.getInvariantGroup()) {
    llvm::MDNode *invariantGroupMD =
        llvm::MDNode::get(builder.getContext(), {});
    loadInst->setMetadata(llvm::LLVMContext::MD_invariant_group,
                          invariantGroupMD);
  }

  moduleTranslation.mapValue(loadOp.getResult(), loadInst);
  return success();
}

/// Convert ptr.store operation
static LogicalResult
convertStoreOp(StoreOp storeOp, llvm::IRBuilderBase &builder,
               LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *value = moduleTranslation.lookupValue(storeOp.getValue());
  llvm::Value *ptr = moduleTranslation.lookupValue(storeOp.getPtr());

  if (!value || !ptr)
    return storeOp.emitError("Failed to lookup operands");

  // Create the store instruction.
  llvm::MaybeAlign alignment(storeOp.getAlignment().value_or(0));
  llvm::StoreInst *storeInst =
      builder.CreateAlignedStore(value, ptr, alignment, storeOp.getVolatile_());

  // Set op flags and metadata.
  storeInst->setAtomic(convertAtomicOrdering(storeOp.getOrdering()));
  // Set sync scope if specified
  if (storeOp.getSyncscope().has_value()) {
    llvm::LLVMContext &ctx = builder.getContext();
    llvm::SyncScope::ID syncScope =
        ctx.getOrInsertSyncScopeID(storeOp.getSyncscope().value());
    storeInst->setSyncScopeID(syncScope);
  }

  // Set metadata for nontemporal and invariant_group
  if (storeOp.getNontemporal()) {
    llvm::MDNode *nontemporalMD =
        llvm::MDNode::get(builder.getContext(),
                          llvm::ConstantAsMetadata::get(builder.getInt32(1)));
    storeInst->setMetadata(llvm::LLVMContext::MD_nontemporal, nontemporalMD);
  }

  if (storeOp.getInvariantGroup()) {
    llvm::MDNode *invariantGroupMD =
        llvm::MDNode::get(builder.getContext(), {});
    storeInst->setMetadata(llvm::LLVMContext::MD_invariant_group,
                           invariantGroupMD);
  }

  return success();
}

/// Convert ptr.type_offset operation
static LogicalResult
convertTypeOffsetOp(TypeOffsetOp typeOffsetOp, llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation) {
  // Convert the element type to LLVM type
  llvm::Type *elementType =
      moduleTranslation.convertType(typeOffsetOp.getElementType());
  if (!elementType)
    return typeOffsetOp.emitError("Failed to convert the element type");

  // Convert result type
  llvm::Type *resultType =
      moduleTranslation.convertType(typeOffsetOp.getResult().getType());
  if (!resultType)
    return typeOffsetOp.emitError("Failed to convert the result type");

  // Use GEP with null pointer to compute type size/offset.
  llvm::Value *nullPtr = llvm::Constant::getNullValue(builder.getPtrTy(0));
  llvm::Value *offsetPtr =
      builder.CreateGEP(elementType, nullPtr, {builder.getInt32(1)});
  llvm::Value *offset = builder.CreatePtrToInt(offsetPtr, resultType);

  moduleTranslation.mapValue(typeOffsetOp.getResult(), offset);
  return success();
}

/// Implementation of the dialect interface that converts operations belonging
/// to the `ptr` dialect to LLVM IR.
class PtrDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {

    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case([&](PtrAddOp ptrAddOp) {
          return convertPtrAddOp(ptrAddOp, builder, moduleTranslation);
        })
        .Case([&](LoadOp loadOp) {
          return convertLoadOp(loadOp, builder, moduleTranslation);
        })
        .Case([&](StoreOp storeOp) {
          return convertStoreOp(storeOp, builder, moduleTranslation);
        })
        .Case([&](TypeOffsetOp typeOffsetOp) {
          return convertTypeOffsetOp(typeOffsetOp, builder, moduleTranslation);
        })
        .Default([&](Operation *op) {
          return op->emitError("Translation for operation '")
                 << op->getName() << "' is not implemented.";
        });
  }

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    // No special amendments needed for ptr dialect operations
    return success();
  }
};
} // namespace

void mlir::registerPtrDialectTranslation(DialectRegistry &registry) {
  registry.insert<ptr::PtrDialect>();
  registry.addExtension(+[](MLIRContext *ctx, ptr::PtrDialect *dialect) {
    dialect->addInterfaces<PtrDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerPtrDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerPtrDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
