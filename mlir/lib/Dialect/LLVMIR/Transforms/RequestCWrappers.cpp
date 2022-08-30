//===- RequestCWrappers.cpp - Annotate funcs with wrap attributes ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMREQUESTCWRAPPERSPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

namespace {
struct LLVMRequestCWrappersPass
    : public LLVM::impl::LLVMRequestCWrappersPassBase<
          LLVMRequestCWrappersPass> {
  using LLVMRequestCWrappersPassBase::LLVMRequestCWrappersPassBase;

  void runOnOperation() override {
    getOperation()->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                            UnitAttr::get(&getContext()));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::LLVM::createRequestCWrappersPass() {
  return std::make_unique<LLVMRequestCWrappersPass>();
}
