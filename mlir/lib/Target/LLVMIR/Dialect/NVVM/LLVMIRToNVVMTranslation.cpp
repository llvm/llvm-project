//===- LLVMIRToNVVMTranslation.cpp - Translate LLVM IR to NVVM dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between LLVM IR and the MLIR NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/NVVM/LLVMIRToNVVMTranslation.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"
#include "llvm/IR/ConstantRange.h"

using namespace mlir;
using namespace mlir::NVVM;

/// LLVM intrinsic IDs of the four `bar.sync` variants imported into
/// `nvvm.barrier`.
static constexpr llvm::Intrinsic::ID kBarrierSyncIntrinsics[] = {
    llvm::Intrinsic::nvvm_barrier_cta_sync_all,
    llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all,
    llvm::Intrinsic::nvvm_barrier_cta_sync_count,
    llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_count,
};

/// Returns true if `id` is one of the `bar.sync` intrinsic IDs imported into
/// `nvvm.barrier`.
static bool isBarrierSyncIntrinsic(llvm::Intrinsic::ID id) {
  return llvm::is_contained(kBarrierSyncIntrinsics, id);
}

/// Imports one of the four `bar.sync` LLVM intrinsic variants into a single
/// `nvvm.barrier` op, deriving the `aligned` attribute and the optional
/// `numberOfThreads` operand from the specific intrinsic ID.
static LogicalResult
convertBarrierSyncIntrinsic(OpBuilder &odsBuilder, llvm::CallInst *inst,
                            LLVM::ModuleImport &moduleImport,
                            ArrayRef<llvm::Value *> llvmOperands,
                            ArrayRef<llvm::OperandBundleUse> llvmOpBundles) {
  llvm::Intrinsic::ID id = inst->getIntrinsicID();
  bool aligned = (id == llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all ||
                  id == llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_count);
  bool hasCount = (id == llvm::Intrinsic::nvvm_barrier_cta_sync_count ||
                   id == llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_count);

  SmallVector<Value> mlirOperands;
  SmallVector<NamedAttribute> mlirAttrs;
  if (failed(moduleImport.convertIntrinsicArguments(
          llvmOperands, llvmOpBundles, /*requiresOpBundles=*/false, {}, {},
          mlirOperands, mlirAttrs)))
    return failure();

  auto op = NVVM::BarrierOp::create(
      odsBuilder, moduleImport.translateLoc(inst->getDebugLoc()),
      mlirOperands.front(), hasCount ? mlirOperands.back() : Value{},
      odsBuilder.getBoolAttr(aligned));
  moduleImport.mapNoResultOp(inst, op);
  return success();
}

/// Returns true if the LLVM intrinsic `id` has a corresponding MLIR NVVM op.
static bool isConvertibleIntrinsic(llvm::Intrinsic::ID id) {
  static const DenseSet<unsigned> convertibleIntrinsics = {
#include "mlir/Dialect/LLVMIR/NVVMConvertibleLLVMIRIntrinsics.inc"
  };
  return convertibleIntrinsics.contains(id) || isBarrierSyncIntrinsic(id);
}

/// Returns the LLVM intrinsic IDs that have a corresponding MLIR NVVM op.
static ArrayRef<unsigned> getSupportedIntrinsicsImpl() {
  static const SmallVector<unsigned> convertibleIntrinsics = []() {
    SmallVector<unsigned> ids = {
#include "mlir/Dialect/LLVMIR/NVVMConvertibleLLVMIRIntrinsics.inc"
    };
    ids.append(std::begin(kBarrierSyncIntrinsics),
               std::end(kBarrierSyncIntrinsics));
    return ids;
  }();
  return convertibleIntrinsics;
}

/// Converts the LLVM intrinsic to an MLIR NVVM dialect operation if a
/// conversion exits. Returns failure otherwise.
static LogicalResult convertIntrinsicImpl(OpBuilder &odsBuilder,
                                          llvm::CallInst *inst,
                                          LLVM::ModuleImport &moduleImport) {
  llvm::Intrinsic::ID intrinsicID = inst->getIntrinsicID();

  // Check if the intrinsic is convertible to an MLIR dialect counterpart and
  // copy the arguments to an an LLVM operands array reference for conversion.
  if (isConvertibleIntrinsic(intrinsicID)) {
    SmallVector<llvm::Value *> args(inst->args());
    ArrayRef<llvm::Value *> llvmOperands(args);

    SmallVector<llvm::OperandBundleUse> llvmOpBundles;
    llvmOpBundles.reserve(inst->getNumOperandBundles());
    for (unsigned i = 0; i < inst->getNumOperandBundles(); ++i)
      llvmOpBundles.push_back(inst->getOperandBundleAt(i));

    // Route `bar.sync` intrinsics to `nvvm.barrier`.
    if (isBarrierSyncIntrinsic(intrinsicID))
      return convertBarrierSyncIntrinsic(odsBuilder, inst, moduleImport,
                                         llvmOperands, llvmOpBundles);

#include "mlir/Dialect/LLVMIR/NVVMFromLLVMIRConversions.inc"
  }

  return failure();
}

namespace {

/// Implementation of the dialect interface that converts operations belonging
/// to the NVVM dialect.
class NVVMDialectLLVMIRImportInterface : public LLVMImportDialectInterface {
public:
  using LLVMImportDialectInterface::LLVMImportDialectInterface;

  /// Converts the LLVM intrinsic to an MLIR NVVM dialect operation if a
  /// conversion exits. Returns failure otherwise.
  LogicalResult convertIntrinsic(OpBuilder &builder, llvm::CallInst *inst,
                                 LLVM::ModuleImport &moduleImport) const final {
    return convertIntrinsicImpl(builder, inst, moduleImport);
  }

  /// Returns the list of LLVM IR intrinsic identifiers that are convertible to
  /// MLIR NVVM dialect intrinsics.
  ArrayRef<unsigned> getSupportedIntrinsics() const final {
    return getSupportedIntrinsicsImpl();
  }
};

} // namespace

void mlir::registerNVVMDialectImport(DialectRegistry &registry) {
  registry.insert<NVVM::NVVMDialect>();
  registry.addExtension(+[](MLIRContext *ctx, NVVM::NVVMDialect *dialect) {
    dialect->addInterfaces<NVVMDialectLLVMIRImportInterface>();
  });
}

void mlir::registerNVVMDialectImport(MLIRContext &context) {
  DialectRegistry registry;
  registerNVVMDialectImport(registry);
  context.appendDialectRegistry(registry);
}
