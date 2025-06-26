//===- DataLayoutFromTarget.cpp - extract data layout from TargetMachine --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/DataLayoutImporter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMDATALAYOUTFROMTARGET
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;
using namespace mlir::LLVM;

struct DataLayoutFromTargetPass
    : public LLVM::impl::LLVMDataLayoutFromTargetBase<
          DataLayoutFromTargetPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    bool passFailed = false;

    mod->walk([&](ModuleOp mod) {
      auto targetAttr =
          mod->getAttrOfType<LLVM::TargetAttrInterface>("llvm.target");
      if (!targetAttr)
        return;

      FailureOr<llvm::DataLayout> dataLayout = targetAttr.getDataLayout();
      if (failed(dataLayout)) {
        mod->emitError() << "failed to obtain llvm::DataLayout from "
                         << targetAttr;
        passFailed = true;
        return;
      }
      auto dataLayoutAttr = DataLayoutAttr::get(
          &getContext(), dataLayout->getStringRepresentation());

      StringRef dlSpecIdentifier = "dlti.dl_spec";
      auto existingDlSpec =
          mod->getAttrOfType<DataLayoutSpecInterface>(dlSpecIdentifier);
      if (existingDlSpec) {
        DataLayoutSpecInterface dataLayoutSpec =
            existingDlSpec.combineWith({dataLayoutAttr.getDataLayoutSpec()});
        mod->setAttr(dlSpecIdentifier, dataLayoutSpec);
      } else {
        mod->setAttr(dlSpecIdentifier, dataLayoutAttr);
      }
    });

    if (passFailed) {
      return signalPassFailure();
    }
  }
};
