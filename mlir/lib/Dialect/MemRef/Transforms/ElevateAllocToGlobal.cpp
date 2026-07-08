//===- ElevateAllocToGlobal.cpp - Elevate memref.alloc to memref.global
//----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to elevate local memref.alloc operations inside
// functions to be memref.global operations outside the functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_ELEVATEALLOCTOGLOBALPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;
using namespace mlir::memref;

namespace {
struct ElevateAllocToGlobal
    : public memref::impl::ElevateAllocToGlobalPassBase<ElevateAllocToGlobal> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    SmallVector<memref::AllocOp> allocOps;
    moduleOp.walk([&](memref::AllocOp allocOp) {
      if (allocOp->getParentOfType<func::FuncOp>()) {
        allocOps.push_back(allocOp);
      }
    });

    for (memref::AllocOp allocOp : allocOps) {
      auto memrefType = cast<MemRefType>(allocOp.getType());
      // memref.global requires statically shaped memrefs.
      if (!memrefType.hasStaticShape() || !allocOp.getDynamicSizes().empty())
        continue;

      // Create a global memref variable in the module's scope.
      OpBuilder globalBuilder(moduleOp.getContext());
      std::string globalName = "global_alloc";
      if (auto funcOp = allocOp->getParentOfType<func::FuncOp>()) {
        globalName = (funcOp.getName() + "_alloc").str();
      }

      auto globalOp = memref::GlobalOp::create(
          globalBuilder, allocOp.getLoc(), globalName,
          /*sym_visibility=*/globalBuilder.getStringAttr("private"),
          /*type=*/memrefType,
          /*initial_value=*/globalBuilder.getUnitAttr(),
          /*constant=*/false,
          /*alignment=*/allocOp.getAlignmentAttr());

      symbolTable.insert(globalOp);
      globalOp->moveBefore(&moduleOp.front());

      // Replace allocOp users and erase deallocations.
      OpBuilder rewriter(allocOp);
      auto getGlobalOp = memref::GetGlobalOp::create(
          rewriter, allocOp.getLoc(), memrefType, globalOp.getName());

      SmallVector<Operation *> deallocsToDelete;
      for (OpOperand &use : allocOp.getResult().getUses()) {
        Operation *user = use.getOwner();
        if (isa<memref::DeallocOp>(user)) {
          deallocsToDelete.push_back(user);
        }
      }
      for (Operation *dealloc : deallocsToDelete) {
        dealloc->erase();
      }

      allocOp.getResult().replaceAllUsesWith(getGlobalOp.getResult());
      allocOp->erase();
    }
  }
};
} // namespace
