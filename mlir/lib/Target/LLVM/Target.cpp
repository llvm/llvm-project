//===- Target.cpp - MLIR LLVM target interface impls ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/Target.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "llvm-target"

using namespace mlir;

namespace {
// Implementation of the `LLVM::TargetAttrInterface` model.
class LLVMTargetAttrImpl
    : public LLVM::TargetAttrInterface::FallbackModel<LLVMTargetAttrImpl> {
public:
  FailureOr<std::unique_ptr<llvm::TargetMachine>>
  getTargetMachine(Attribute attribute) const {
    auto attr = llvm::cast<LLVM::TargetAttr>(attribute);

    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();

    std::string error;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(attr.getTriple(), error);
    if (!target || !error.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "Looking up target '" << attr.getTriple()
                     << "' failed: " << error << "\n";
      });
      return failure();
    }

    return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
        llvm::Triple(attr.getTriple().getValue()), attr.getChip().getValue(),
        attr.getFeatures() ? attr.getFeatures().getValue() : "", {}, {}));
  }

  FailureOr<llvm::DataLayout> getDataLayout(Attribute attribute) const {
    auto attr = llvm::cast<LLVM::TargetAttrInterface>(attribute);

    FailureOr<std::unique_ptr<llvm::TargetMachine>> targetMachine =
        attr.getTargetMachine();
    if (failed(targetMachine)) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "Failed to retrieve the target machine for data layout.\n";
      });
      return failure();
    }
    return (targetMachine.value())->createDataLayout();
  }

  StringAttr getTripleAttr(Attribute attribute) const {
    return llvm::cast<LLVM::TargetAttr>(attribute).getTriple();
  }

  StringAttr getChipAttr(Attribute attribute) const {
    return llvm::cast<LLVM::TargetAttr>(attribute).getChip();
  }

  StringAttr getFeaturesAttr(Attribute attribute) const {
    return llvm::cast<LLVM::TargetAttr>(attribute).getFeatures();
  }
};
} // namespace

// Register the LLVM target interface.
void LLVM::registerLLVMTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    LLVM::TargetAttr::attachInterface<LLVMTargetAttrImpl>(*ctx);
  });
}

void LLVM::registerLLVMTargetInterfaceExternalModels(MLIRContext &context) {
  DialectRegistry registry;
  registerLLVMTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}
