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
translateAtomicOrdering(ptr::AtomicOrdering ordering) {
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

/// Translate ptr.ptr_add operation to LLVM IR.
static LogicalResult
translatePtrAddOp(PtrAddOp ptrAddOp, llvm::IRBuilderBase &builder,
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

/// Translate ptr.load operation to LLVM IR.
static LogicalResult
translateLoadOp(LoadOp loadOp, llvm::IRBuilderBase &builder,
                LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *ptr = moduleTranslation.lookupValue(loadOp.getPtr());
  if (!ptr)
    return loadOp.emitError("Failed to lookup pointer operand");

  // Translate result type to LLVM type
  llvm::Type *resultType =
      moduleTranslation.convertType(loadOp.getValue().getType());
  if (!resultType)
    return loadOp.emitError("Failed to translate result type");

  // Create the load instruction.
  llvm::MaybeAlign alignment(loadOp.getAlignment().value_or(0));
  llvm::LoadInst *loadInst = builder.CreateAlignedLoad(
      resultType, ptr, alignment, loadOp.getVolatile_());

  // Set op flags and metadata.
  loadInst->setAtomic(translateAtomicOrdering(loadOp.getOrdering()));
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

/// Translate ptr.store operation to LLVM IR.
static LogicalResult
translateStoreOp(StoreOp storeOp, llvm::IRBuilderBase &builder,
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
  storeInst->setAtomic(translateAtomicOrdering(storeOp.getOrdering()));
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

/// Translate ptr.type_offset operation to LLVM IR.
static LogicalResult
translateTypeOffsetOp(TypeOffsetOp typeOffsetOp, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  // Translate the element type to LLVM type
  llvm::Type *elementType =
      moduleTranslation.convertType(typeOffsetOp.getElementType());
  if (!elementType)
    return typeOffsetOp.emitError("Failed to translate the element type");

  // Translate result type
  llvm::Type *resultType =
      moduleTranslation.convertType(typeOffsetOp.getResult().getType());
  if (!resultType)
    return typeOffsetOp.emitError("Failed to translate the result type");

  // Use GEP with null pointer to compute type size/offset.
  llvm::Value *nullPtr = llvm::Constant::getNullValue(builder.getPtrTy(0));
  llvm::Value *offsetPtr =
      builder.CreateGEP(elementType, nullPtr, {builder.getInt32(1)});
  llvm::Value *offset = builder.CreatePtrToInt(offsetPtr, resultType);

  moduleTranslation.mapValue(typeOffsetOp.getResult(), offset);
  return success();
}

/// Translate ptr.gather operation to LLVM IR.
static LogicalResult
translateGatherOp(GatherOp gatherOp, llvm::IRBuilderBase &builder,
                  LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *ptrs = moduleTranslation.lookupValue(gatherOp.getPtrs());
  llvm::Value *mask = moduleTranslation.lookupValue(gatherOp.getMask());
  llvm::Value *passthrough =
      moduleTranslation.lookupValue(gatherOp.getPassthrough());

  if (!ptrs || !mask || !passthrough)
    return gatherOp.emitError("Failed to lookup operands");

  // Translate result type to LLVM type.
  llvm::Type *resultType =
      moduleTranslation.convertType(gatherOp.getResult().getType());
  if (!resultType)
    return gatherOp.emitError("Failed to translate result type");

  // Get the alignment.
  llvm::MaybeAlign alignment(gatherOp.getAlignment().value_or(0));

  // Create the masked gather intrinsic call.
  llvm::Value *result = builder.CreateMaskedGather(
      resultType, ptrs, alignment.valueOrOne(), mask, passthrough);

  moduleTranslation.mapValue(gatherOp.getResult(), result);
  return success();
}

/// Translate ptr.masked_load operation to LLVM IR.
static LogicalResult
translateMaskedLoadOp(MaskedLoadOp maskedLoadOp, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *ptr = moduleTranslation.lookupValue(maskedLoadOp.getPtr());
  llvm::Value *mask = moduleTranslation.lookupValue(maskedLoadOp.getMask());
  llvm::Value *passthrough =
      moduleTranslation.lookupValue(maskedLoadOp.getPassthrough());

  if (!ptr || !mask || !passthrough)
    return maskedLoadOp.emitError("Failed to lookup operands");

  // Translate result type to LLVM type.
  llvm::Type *resultType =
      moduleTranslation.convertType(maskedLoadOp.getResult().getType());
  if (!resultType)
    return maskedLoadOp.emitError("Failed to translate result type");

  // Get the alignment.
  llvm::MaybeAlign alignment(maskedLoadOp.getAlignment().value_or(0));

  // Create the masked load intrinsic call.
  llvm::Value *result = builder.CreateMaskedLoad(
      resultType, ptr, alignment.valueOrOne(), mask, passthrough);

  moduleTranslation.mapValue(maskedLoadOp.getResult(), result);
  return success();
}

/// Translate ptr.masked_store operation to LLVM IR.
static LogicalResult
translateMaskedStoreOp(MaskedStoreOp maskedStoreOp,
                       llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *value = moduleTranslation.lookupValue(maskedStoreOp.getValue());
  llvm::Value *ptr = moduleTranslation.lookupValue(maskedStoreOp.getPtr());
  llvm::Value *mask = moduleTranslation.lookupValue(maskedStoreOp.getMask());

  if (!value || !ptr || !mask)
    return maskedStoreOp.emitError("Failed to lookup operands");

  // Get the alignment.
  llvm::MaybeAlign alignment(maskedStoreOp.getAlignment().value_or(0));

  // Create the masked store intrinsic call.
  builder.CreateMaskedStore(value, ptr, alignment.valueOrOne(), mask);
  return success();
}

/// Translate ptr.scatter operation to LLVM IR.
static LogicalResult
translateScatterOp(ScatterOp scatterOp, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *value = moduleTranslation.lookupValue(scatterOp.getValue());
  llvm::Value *ptrs = moduleTranslation.lookupValue(scatterOp.getPtrs());
  llvm::Value *mask = moduleTranslation.lookupValue(scatterOp.getMask());

  if (!value || !ptrs || !mask)
    return scatterOp.emitError("Failed to lookup operands");

  // Get the alignment.
  llvm::MaybeAlign alignment(scatterOp.getAlignment().value_or(0));

  // Create the masked scatter intrinsic call.
  builder.CreateMaskedScatter(value, ptrs, alignment.valueOrOne(), mask);
  return success();
}

/// Translate ptr.constant operation to LLVM IR.
static LogicalResult
translateConstantOp(ConstantOp constantOp, llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation) {
  // Translate result type to LLVM type
  llvm::PointerType *resultType = dyn_cast_or_null<llvm::PointerType>(
      moduleTranslation.convertType(constantOp.getResult().getType()));
  if (!resultType)
    return constantOp.emitError("Expected a valid pointer type");

  llvm::Value *result = nullptr;

  TypedAttr value = constantOp.getValue();
  if (auto nullAttr = dyn_cast<ptr::NullAttr>(value)) {
    // Create a null pointer constant
    result = llvm::ConstantPointerNull::get(resultType);
  } else if (auto addressAttr = dyn_cast<ptr::AddressAttr>(value)) {
    // Create an integer constant and translate it to pointer
    llvm::APInt addressValue = addressAttr.getValue();

    // Determine the integer type width based on the target's pointer size
    llvm::DataLayout dataLayout =
        moduleTranslation.getLLVMModule()->getDataLayout();
    unsigned pointerSizeInBits =
        dataLayout.getPointerSizeInBits(resultType->getAddressSpace());

    // Extend or truncate the address value to match pointer size if needed
    if (addressValue.getBitWidth() != pointerSizeInBits) {
      if (addressValue.getBitWidth() > pointerSizeInBits) {
        constantOp.emitWarning()
            << "Truncating address value to fit pointer size";
      }
      addressValue = addressValue.getBitWidth() < pointerSizeInBits
                         ? addressValue.zext(pointerSizeInBits)
                         : addressValue.trunc(pointerSizeInBits);
    }

    // Create integer constant and translate to pointer
    llvm::Type *intType = builder.getIntNTy(pointerSizeInBits);
    llvm::Value *intValue = llvm::ConstantInt::get(intType, addressValue);
    result = builder.CreateIntToPtr(intValue, resultType);
  } else {
    return constantOp.emitError("Unsupported constant attribute type");
  }

  moduleTranslation.mapValue(constantOp.getResult(), result);
  return success();
}

/// Implementation of the dialect interface that translates operations belonging
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
        .Case([&](ConstantOp constantOp) {
          return translateConstantOp(constantOp, builder, moduleTranslation);
        })
        .Case([&](PtrAddOp ptrAddOp) {
          return translatePtrAddOp(ptrAddOp, builder, moduleTranslation);
        })
        .Case([&](LoadOp loadOp) {
          return translateLoadOp(loadOp, builder, moduleTranslation);
        })
        .Case([&](StoreOp storeOp) {
          return translateStoreOp(storeOp, builder, moduleTranslation);
        })
        .Case([&](TypeOffsetOp typeOffsetOp) {
          return translateTypeOffsetOp(typeOffsetOp, builder,
                                       moduleTranslation);
        })
        .Case<GatherOp>([&](GatherOp gatherOp) {
          return translateGatherOp(gatherOp, builder, moduleTranslation);
        })
        .Case<MaskedLoadOp>([&](MaskedLoadOp maskedLoadOp) {
          return translateMaskedLoadOp(maskedLoadOp, builder,
                                       moduleTranslation);
        })
        .Case<MaskedStoreOp>([&](MaskedStoreOp maskedStoreOp) {
          return translateMaskedStoreOp(maskedStoreOp, builder,
                                        moduleTranslation);
        })
        .Case<ScatterOp>([&](ScatterOp scatterOp) {
          return translateScatterOp(scatterOp, builder, moduleTranslation);
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
