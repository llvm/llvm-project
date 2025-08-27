//===- TargetToTargetFeatures.cpp - extract features from TargetMachine ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Transforms/TargetUtils.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/MC/MCSubtargetInfo.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMTARGETTOTARGETFEATURES
#include "mlir/Target/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

struct TargetToTargetFeaturesPass
    : public LLVM::impl::LLVMTargetToTargetFeaturesBase<
          TargetToTargetFeaturesPass> {
  using LLVM::impl::LLVMTargetToTargetFeaturesBase<
      TargetToTargetFeaturesPass>::LLVMTargetToTargetFeaturesBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    if (initializeLLVMTargets)
      LLVM::detail::initializeBackendsOnce();

    auto targetAttr = op->getAttrOfType<LLVM::TargetAttr>(
        LLVM::LLVMDialect::getTargetAttrName());
    if (!targetAttr) {
      op->emitError() << "no LLVM::TargetAttr attribute at key \""
                      << LLVM::LLVMDialect::getTargetAttrName() << "\"";
      return signalPassFailure();
    }

    FailureOr<std::unique_ptr<llvm::TargetMachine>> targetMachine =
        LLVM::detail::getTargetMachine(targetAttr);
    if (failed(targetMachine)) {
      op->emitError() << "failed to obtain llvm::TargetMachine for "
                      << targetAttr;
      return signalPassFailure();
    }

    llvm::MCSubtargetInfo const *subTargetInfo =
        (*targetMachine)->getMCSubtargetInfo();

    const std::vector<llvm::SubtargetFeatureKV> enabledFeatures =
        subTargetInfo->getEnabledProcessorFeatures();

    auto plussedFeatures = llvm::to_vector(
        llvm::map_range(enabledFeatures, [](llvm::SubtargetFeatureKV feature) {
          return std::string("+") + feature.Key;
        }));

    auto plussedFeaturesRefs = llvm::to_vector(llvm::map_range(
        plussedFeatures, [](auto &it) { return StringRef(it.c_str()); }));

    auto fullTargetFeaturesAttr =
        LLVM::TargetFeaturesAttr::get(&getContext(), plussedFeaturesRefs);

    auto updatedTargetAttr =
        LLVM::TargetAttr::get(&getContext(), targetAttr.getTriple(),
                              targetAttr.getChip(), fullTargetFeaturesAttr);

    op->setAttr(LLVM::LLVMDialect::getTargetAttrName(), updatedTargetAttr);
  }
};
