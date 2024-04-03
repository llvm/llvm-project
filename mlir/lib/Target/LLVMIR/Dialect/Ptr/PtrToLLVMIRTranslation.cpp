//===- PtrToLLVMIRTranslation.cpp - Translate Ptr dialect to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR Ptr dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/Ptr/PtrToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"

using namespace mlir;
using namespace mlir::ptr;

namespace {
::llvm::AtomicRMWInst::BinOp convertAtomicBinOpToLLVM(AtomicBinOp value) {
  switch (value) {
  case AtomicBinOp::xchg:
    return ::llvm::AtomicRMWInst::BinOp::Xchg;
  case AtomicBinOp::add:
    return ::llvm::AtomicRMWInst::BinOp::Add;
  case AtomicBinOp::sub:
    return ::llvm::AtomicRMWInst::BinOp::Sub;
  case AtomicBinOp::_and:
    return ::llvm::AtomicRMWInst::BinOp::And;
  case AtomicBinOp::nand:
    return ::llvm::AtomicRMWInst::BinOp::Nand;
  case AtomicBinOp::_or:
    return ::llvm::AtomicRMWInst::BinOp::Or;
  case AtomicBinOp::_xor:
    return ::llvm::AtomicRMWInst::BinOp::Xor;
  case AtomicBinOp::max:
    return ::llvm::AtomicRMWInst::BinOp::Max;
  case AtomicBinOp::min:
    return ::llvm::AtomicRMWInst::BinOp::Min;
  case AtomicBinOp::umax:
    return ::llvm::AtomicRMWInst::BinOp::UMax;
  case AtomicBinOp::umin:
    return ::llvm::AtomicRMWInst::BinOp::UMin;
  case AtomicBinOp::fadd:
    return ::llvm::AtomicRMWInst::BinOp::FAdd;
  case AtomicBinOp::fsub:
    return ::llvm::AtomicRMWInst::BinOp::FSub;
  case AtomicBinOp::fmax:
    return ::llvm::AtomicRMWInst::BinOp::FMax;
  case AtomicBinOp::fmin:
    return ::llvm::AtomicRMWInst::BinOp::FMin;
  case AtomicBinOp::uinc_wrap:
    return ::llvm::AtomicRMWInst::BinOp::UIncWrap;
  case AtomicBinOp::udec_wrap:
    return ::llvm::AtomicRMWInst::BinOp::UDecWrap;
  }
  llvm_unreachable("unknown AtomicBinOp type");
}

::llvm::AtomicOrdering convertAtomicOrderingToLLVM(AtomicOrdering value) {
  switch (value) {
  case AtomicOrdering::not_atomic:
    return ::llvm::AtomicOrdering::NotAtomic;
  case AtomicOrdering::unordered:
    return ::llvm::AtomicOrdering::Unordered;
  case AtomicOrdering::monotonic:
    return ::llvm::AtomicOrdering::Monotonic;
  case AtomicOrdering::acquire:
    return ::llvm::AtomicOrdering::Acquire;
  case AtomicOrdering::release:
    return ::llvm::AtomicOrdering::Release;
  case AtomicOrdering::acq_rel:
    return ::llvm::AtomicOrdering::AcquireRelease;
  case AtomicOrdering::seq_cst:
    return ::llvm::AtomicOrdering::SequentiallyConsistent;
  }
  llvm_unreachable("unknown AtomicOrdering type");
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

LogicalResult convertAtomicRMWOp(AtomicRMWOp op, llvm::IRBuilderBase &builder,
                                 LLVM::ModuleTranslation &moduleTranslation) {
  auto *inst = builder.CreateAtomicRMW(
      convertAtomicBinOpToLLVM(op.getBinOp()),
      moduleTranslation.lookupValue(op.getPtr()),
      moduleTranslation.lookupValue(op.getVal()), llvm::MaybeAlign(),
      convertAtomicOrderingToLLVM(op.getOrdering()));
  moduleTranslation.mapValue(op.getRes()) = inst;
  // Set volatile flag.
  inst->setVolatile(op.getVolatile_());
  // Set sync scope.
  if (op.getSyncscope().has_value()) {
    llvm::LLVMContext &llvmContext = builder.getContext();
    inst->setSyncScopeID(
        llvmContext.getOrInsertSyncScopeID(*op.getSyncscope()));
  }
  // Set alignment.
  if (op.getAlignment().has_value()) {
    auto align = *op.getAlignment();
    if (align != 0)
      inst->setAlignment(llvm::Align(align));
  }
  // Set access group.
  if (auto accessGroup =
          dyn_cast<LLVM::AccessGroupOpInterface>(op.getOperation()))
    moduleTranslation.setAccessGroupsMetadata(accessGroup, inst);
  // Set alias analysis.
  if (auto aa = dyn_cast<LLVM::AliasAnalysisOpInterface>(op.getOperation())) {
    moduleTranslation.setAliasScopeMetadata(aa, inst);
    moduleTranslation.setTBAAMetadata(aa, inst);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AtomicCmpXchgOp
//===----------------------------------------------------------------------===//

LogicalResult
convertAtomicCmpXchgOp(AtomicCmpXchgOp op, llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  auto *inst = builder.CreateAtomicCmpXchg(
      moduleTranslation.lookupValue(op.getPtr()),
      moduleTranslation.lookupValue(op.getCmp()),
      moduleTranslation.lookupValue(op.getVal()), llvm::MaybeAlign(),
      convertAtomicOrderingToLLVM(op.getSuccessOrdering()),
      convertAtomicOrderingToLLVM(op.getFailureOrdering()));
  moduleTranslation.mapValue(op.getRes()) =
      builder.CreateExtractValue(inst, {0});
  moduleTranslation.mapValue(op.getStatus()) =
      builder.CreateExtractValue(inst, {1});
  inst->setWeak(op.getWeak());
  // Set volatile flag.
  inst->setVolatile(op.getVolatile_());
  // Set sync scope.
  if (op.getSyncscope().has_value()) {
    llvm::LLVMContext &llvmContext = builder.getContext();
    inst->setSyncScopeID(
        llvmContext.getOrInsertSyncScopeID(*op.getSyncscope()));
  }
  // Set alignment.
  if (op.getAlignment().has_value()) {
    auto align = *op.getAlignment();
    if (align != 0)
      inst->setAlignment(llvm::Align(align));
  }
  // Set access group.
  if (auto accessGroup =
          dyn_cast<LLVM::AccessGroupOpInterface>(op.getOperation()))
    moduleTranslation.setAccessGroupsMetadata(accessGroup, inst);
  // Set alias analysis.
  if (auto aa = dyn_cast<LLVM::AliasAnalysisOpInterface>(op.getOperation())) {
    moduleTranslation.setAliasScopeMetadata(aa, inst);
    moduleTranslation.setTBAAMetadata(aa, inst);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult convertLoadOp(LoadOp op, llvm::IRBuilderBase &builder,
                            LLVM::ModuleTranslation &moduleTranslation) {
  auto *inst = builder.CreateLoad(
      moduleTranslation.convertType(op.getResult().getType()),
      moduleTranslation.lookupValue(op.getAddr()), op.getVolatile_());
  moduleTranslation.mapValue(op.getRes()) = inst;
  if (op.getInvariant()) {
    llvm::MDNode *metadata =
        llvm::MDNode::get(inst->getContext(), std::nullopt);
    inst->setMetadata(llvm::LLVMContext::MD_invariant_load, metadata);
  }
  // Set atomic ordering.
  inst->setAtomic(convertAtomicOrderingToLLVM(op.getOrdering()));
  // Set sync scope.
  if (op.getSyncscope().has_value()) {
    llvm::LLVMContext &llvmContext = builder.getContext();
    inst->setSyncScopeID(
        llvmContext.getOrInsertSyncScopeID(*op.getSyncscope()));
  }
  // Set alignment.
  if (op.getAlignment().has_value()) {
    auto align = *op.getAlignment();
    if (align != 0)
      inst->setAlignment(llvm::Align(align));
  }
  // Set non-temporal.
  if (op.getNontemporal()) {
    llvm::MDNode *metadata = llvm::MDNode::get(
        inst->getContext(), llvm::ConstantAsMetadata::get(builder.getInt32(1)));
    inst->setMetadata(llvm::LLVMContext::MD_nontemporal, metadata);
  }
  // Set access group.
  if (auto accessGroup =
          dyn_cast<LLVM::AccessGroupOpInterface>(op.getOperation()))
    moduleTranslation.setAccessGroupsMetadata(accessGroup, inst);
  // Set alias analysis.
  if (auto aa = dyn_cast<LLVM::AliasAnalysisOpInterface>(op.getOperation())) {
    moduleTranslation.setAliasScopeMetadata(aa, inst);
    moduleTranslation.setTBAAMetadata(aa, inst);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult convertStoreOp(StoreOp op, llvm::IRBuilderBase &builder,
                             LLVM::ModuleTranslation &moduleTranslation) {
  auto *inst = builder.CreateStore(moduleTranslation.lookupValue(op.getValue()),
                                   moduleTranslation.lookupValue(op.getAddr()),
                                   op.getVolatile_());

  // Set atomic ordering.
  inst->setAtomic(convertAtomicOrderingToLLVM(op.getOrdering()));
  // Set sync scope.
  if (op.getSyncscope().has_value()) {
    llvm::LLVMContext &llvmContext = builder.getContext();
    inst->setSyncScopeID(
        llvmContext.getOrInsertSyncScopeID(*op.getSyncscope()));
  }
  // Set alignment.
  if (op.getAlignment().has_value()) {
    auto align = *op.getAlignment();
    if (align != 0)
      inst->setAlignment(llvm::Align(align));
  }
  // Set non-temporal.
  if (op.getNontemporal()) {
    llvm::MDNode *metadata = llvm::MDNode::get(
        inst->getContext(), llvm::ConstantAsMetadata::get(builder.getInt32(1)));
    inst->setMetadata(llvm::LLVMContext::MD_nontemporal, metadata);
  }
  // Set access group.
  if (auto accessGroup =
          dyn_cast<LLVM::AccessGroupOpInterface>(op.getOperation()))
    moduleTranslation.setAccessGroupsMetadata(accessGroup, inst);
  // Set alias analysis.
  if (auto aa = dyn_cast<LLVM::AliasAnalysisOpInterface>(op.getOperation())) {
    moduleTranslation.setAliasScopeMetadata(aa, inst);
    moduleTranslation.setTBAAMetadata(aa, inst);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AddrSpaceCastOp
//===----------------------------------------------------------------------===//

LogicalResult
convertAddrSpaceCastOp(AddrSpaceCastOp op, llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  moduleTranslation.mapValue(op.getRes()) = builder.CreateAddrSpaceCast(
      moduleTranslation.lookupValue(op.getArg()),
      moduleTranslation.convertType(op.getResult().getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// AddrSpaceCastOp
//===----------------------------------------------------------------------===//

LogicalResult convertIntToPtrOp(IntToPtrOp op, llvm::IRBuilderBase &builder,
                                LLVM::ModuleTranslation &moduleTranslation) {
  moduleTranslation.mapValue(op.getRes()) = builder.CreateIntToPtr(
      moduleTranslation.lookupValue(op.getArg()),
      moduleTranslation.convertType(op.getResult().getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// AddrSpaceCastOp
//===----------------------------------------------------------------------===//

LogicalResult convertPtrToIntOp(PtrToIntOp op, llvm::IRBuilderBase &builder,
                                LLVM::ModuleTranslation &moduleTranslation) {
  moduleTranslation.mapValue(op.getRes()) = builder.CreatePtrToInt(
      moduleTranslation.lookupValue(op.getArg()),
      moduleTranslation.convertType(op.getResult().getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult convertConstantOp(ConstantOp op, llvm::IRBuilderBase &builder,
                                LLVM::ModuleTranslation &moduleTranslation) {
  return success();
}

//===----------------------------------------------------------------------===//
// TypeOffsetOp
//===----------------------------------------------------------------------===//

LogicalResult convertTypeOffsetOp(TypeOffsetOp op, llvm::IRBuilderBase &builder,
                                  LLVM::ModuleTranslation &moduleTranslation) {
  return success();
}

//===----------------------------------------------------------------------===//
// PtrAddOp
//===----------------------------------------------------------------------===//

LogicalResult convertPtrAddOp(PtrAddOp op, llvm::IRBuilderBase &builder,
                              LLVM::ModuleTranslation &moduleTranslation) {
  return success();
}

class PtrDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *operation, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    return llvm::TypeSwitch<Operation *, LogicalResult>(operation)
        .Case([&](ptr::AtomicRMWOp op) {
          return convertAtomicRMWOp(op, builder, moduleTranslation);
        })
        .Case([&](ptr::AtomicCmpXchgOp op) {
          return convertAtomicCmpXchgOp(op, builder, moduleTranslation);
        })
        .Case([&](ptr::LoadOp op) {
          return convertLoadOp(op, builder, moduleTranslation);
        })
        .Case([&](ptr::StoreOp op) {
          return convertStoreOp(op, builder, moduleTranslation);
        })
        .Case([&](ptr::AddrSpaceCastOp op) {
          return convertAddrSpaceCastOp(op, builder, moduleTranslation);
        })
        .Case([&](ptr::IntToPtrOp op) {
          return convertIntToPtrOp(op, builder, moduleTranslation);
        })
        .Case([&](ptr::PtrToIntOp op) {
          return convertPtrToIntOp(op, builder, moduleTranslation);
        })
        .Case([&](ptr::ConstantOp op) {
          return convertConstantOp(op, builder, moduleTranslation);
        })
        .Case([&](ptr::TypeOffsetOp op) {
          return convertTypeOffsetOp(op, builder, moduleTranslation);
        })
        .Case([&](ptr::PtrAddOp op) {
          return convertPtrAddOp(op, builder, moduleTranslation);
        })
        .Default([&](Operation *op) {
          return op->emitError("unsupported Ptr operation: ") << op->getName();
        });
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
