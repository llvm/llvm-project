//===- TargetToDataLayout.cpp - extract data layout from TargetMachine ----===//
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

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMTARGETTODATALAYOUT
#include "mlir/Target/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

struct TargetToDataLayoutPass
    : public LLVM::impl::LLVMTargetToDataLayoutBase<TargetToDataLayoutPass> {
  using LLVM::impl::LLVMTargetToDataLayoutBase<
      TargetToDataLayoutPass>::LLVMTargetToDataLayoutBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    if (initializeLLVMTargets)
      LLVM::detail::initializeBackendsOnce();

    auto targetAttr = op->getAttrOfType<LLVM::TargetAttrInterface>(
        LLVM::LLVMDialect::getTargetAttrName());
    if (!targetAttr) {
      op->emitError()
          << "no TargetAttrInterface-implementing attribute at key \""
          << LLVM::LLVMDialect::getTargetAttrName() << "\"";
      return signalPassFailure();
    }

    FailureOr<llvm::DataLayout> dataLayout =
        LLVM::detail::getDataLayout(targetAttr);
    if (failed(dataLayout)) {
      op->emitError() << "failed to obtain llvm::DataLayout for " << targetAttr;
      return signalPassFailure();
    }

    DataLayoutSpecInterface dataLayoutSpec =
        mlir::translateDataLayout(dataLayout.value(), &getContext());

    if (auto existingDlSpec = op->getAttrOfType<DataLayoutSpecInterface>(
            DLTIDialect::kDataLayoutAttrName)) {
      dataLayoutSpec = existingDlSpec.combineWith({dataLayoutSpec});
    }

    op->setAttr(DLTIDialect::kDataLayoutAttrName, dataLayoutSpec);
  }
};
