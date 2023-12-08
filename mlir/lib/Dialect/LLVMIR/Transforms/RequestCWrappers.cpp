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
#define GEN_PASS_DEF_LLVMREQUESTCWRAPPERS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

namespace {
class RequestCWrappersPass
    : public LLVM::impl::LLVMRequestCWrappersBase<RequestCWrappersPass> {
public:
  void runOnOperation() override {
    getOperation()->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                            UnitAttr::get(&getContext()));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::LLVM::createRequestCWrappersPass() {
  return std::make_unique<RequestCWrappersPass>();
}
