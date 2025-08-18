//===- TargetToDataLayout.cpp - extract data layout from TargetMachine ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/LLVMIR/Transforms/Passes.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/DataLayoutImporter.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "mlir-llvm-target-to-data-layout"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMTARGETTODATALAYOUT
#include "mlir/Target/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

static FailureOr<std::unique_ptr<llvm::TargetMachine>>
getTargetMachine(LLVM::TargetAttrInterface attr) {
  StringRef triple = attr.getTripleAttr();
  StringRef chipAKAcpu = attr.getChipAttr();
  StringRef features =
      attr.getFeaturesAttr() ? attr.getFeaturesAttr().getValue() : "";

  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target || !error.empty()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Looking up target '" << triple << "' failed: " << error
                   << "\n";
    });
    return failure();
  }

  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      llvm::Triple(triple), chipAKAcpu, features, {}, {}));
}

static FailureOr<llvm::DataLayout>
getDataLayout(LLVM::TargetAttrInterface attr) {
  FailureOr<std::unique_ptr<llvm::TargetMachine>> targetMachine =
      getTargetMachine(attr);
  if (failed(targetMachine)) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "Failed to retrieve the target machine for data layout.\n";
    });
    return failure();
  }
  return (targetMachine.value())->createDataLayout();
}

struct TargetToDataLayoutPass
    : public LLVM::impl::LLVMTargetToDataLayoutBase<TargetToDataLayoutPass> {
  using LLVM::impl::LLVMTargetToDataLayoutBase<
      TargetToDataLayoutPass>::LLVMTargetToDataLayoutBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    if (initializeLLVMTargets) {
      // Ensure that the targets, that LLVM has been configured to support,
      // are loaded into the TargetRegistry.
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
    }

    auto targetAttr = op->getAttrOfType<LLVM::TargetAttrInterface>(
        LLVM::LLVMDialect::getTargetAttrName());
    if (!targetAttr)
      return;

    FailureOr<llvm::DataLayout> dataLayout = getDataLayout(targetAttr);
    if (failed(dataLayout)) {
      op->emitError() << "failed to obtain llvm::DataLayout for " << targetAttr;
      return signalPassFailure();
    }

    StringRef dataLayoutStr = dataLayout->getStringRepresentation();

    auto importer =
        LLVM::detail::DataLayoutImporter(&getContext(), dataLayoutStr);
    DataLayoutSpecInterface dataLayoutSpec = importer.getDataLayoutSpec();

    if (auto existingDlSpec = op->getAttrOfType<DataLayoutSpecInterface>(
            DLTIDialect::kDataLayoutAttrName)) {
      dataLayoutSpec = existingDlSpec.combineWith({dataLayoutSpec});
    }

    op->setAttr(DLTIDialect::kDataLayoutAttrName, dataLayoutSpec);
  }
};
