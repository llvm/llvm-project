//===- AddComdats.cpp - Add comdats to linkonce functions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/AddComdats.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMADDCOMDATS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

static void addComdat(LLVM::LLVMFuncOp &op, OpBuilder &builder,
                      SymbolTable &symbolTable, ModuleOp &module) {
  const char *comdatName = "__llvm_comdat";
  mlir::LLVM::ComdatOp comdatOp =
      symbolTable.lookup<mlir::LLVM::ComdatOp>(comdatName);
  if (!comdatOp) {
    PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    comdatOp =
        builder.create<mlir::LLVM::ComdatOp>(module.getLoc(), comdatName);
    symbolTable.insert(comdatOp);
  }

  PatternRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&comdatOp.getBody().back());
  auto selectorOp = builder.create<mlir::LLVM::ComdatSelectorOp>(
      comdatOp.getLoc(), op.getSymName(), mlir::LLVM::comdat::Comdat::Any);
  op.setComdatAttr(mlir::SymbolRefAttr::get(
      builder.getContext(), comdatName,
      mlir::FlatSymbolRefAttr::get(selectorOp.getSymNameAttr())));
}

namespace {
struct AddComdatsPass : public LLVM::impl::LLVMAddComdatsBase<AddComdatsPass> {
  void runOnOperation() override {
    OpBuilder builder{&getContext()};
    ModuleOp mod = getOperation();

    std::unique_ptr<SymbolTable> symbolTable;
    auto getSymTab = [&]() -> SymbolTable & {
      if (!symbolTable)
        symbolTable = std::make_unique<SymbolTable>(mod);
      return *symbolTable;
    };
    for (auto op : mod.getBody()->getOps<LLVM::LLVMFuncOp>()) {
      if (op.getLinkage() == LLVM::Linkage::Linkonce ||
          op.getLinkage() == LLVM::Linkage::LinkonceODR) {
        addComdat(op, builder, getSymTab(), mod);
      }
    }
  }
};
} // namespace
