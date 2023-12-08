//===- DILineTableFromLocations.cpp - -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_DISCOPEFORLLVMFUNCOP
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

/// Attempt to extract a filename for the given loc.
static FileLineColLoc extractFileLoc(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return extractFileLoc(nameLoc.getChildLoc());
  if (auto opaqueLoc = dyn_cast<OpaqueLoc>(loc))
    return extractFileLoc(opaqueLoc.getFallbackLocation());
  return FileLineColLoc();
}

namespace {
/// Add a debug info scope to LLVMFuncOp that are missing it.
struct DIScopeForLLVMFuncOp
    : public LLVM::impl::DIScopeForLLVMFuncOpBase<DIScopeForLLVMFuncOp> {
  void runOnOperation() override {
    LLVM::LLVMFuncOp llvmFunc = getOperation();
    Location loc = llvmFunc.getLoc();
    if (loc->findInstanceOf<mlir::FusedLocWith<LLVM::DISubprogramAttr>>())
      return;

    MLIRContext *context = &getContext();

    // To find a DICompileUnitAttr attached to a parent (the module for
    // example), otherwise create a default one.
    LLVM::DICompileUnitAttr compileUnitAttr;
    if (ModuleOp module = llvmFunc->getParentOfType<ModuleOp>()) {
      auto fusedCompileUnitAttr =
          module->getLoc()
              ->findInstanceOf<mlir::FusedLocWith<LLVM::DICompileUnitAttr>>();
      if (fusedCompileUnitAttr)
        compileUnitAttr = fusedCompileUnitAttr.getMetadata();
    }

    // Filename, line and colmun to associate to the function.
    LLVM::DIFileAttr fileAttr;
    int64_t line = 1, col = 1;
    FileLineColLoc fileLoc = extractFileLoc(loc);
    if (!fileLoc && compileUnitAttr) {
      fileAttr = compileUnitAttr.getFile();
    } else if (!fileLoc) {
      fileAttr = LLVM::DIFileAttr::get(context, "<unknown>", "");
    } else {
      line = fileLoc.getLine();
      col = fileLoc.getColumn();
      StringRef inputFilePath = fileLoc.getFilename().getValue();
      fileAttr = LLVM::DIFileAttr::get(
          context, llvm::sys::path::filename(inputFilePath),
          llvm::sys::path::parent_path(inputFilePath));
    }
    if (!compileUnitAttr) {
      compileUnitAttr = LLVM::DICompileUnitAttr::get(
          context, llvm::dwarf::DW_LANG_C, fileAttr,
          StringAttr::get(context, "MLIR"), /*isOptimized=*/true,
          LLVM::DIEmissionKind::LineTablesOnly);
    }
    auto subroutineTypeAttr =
        LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, {});

    StringAttr funcNameAttr = llvmFunc.getNameAttr();
    auto subprogramAttr =
        LLVM::DISubprogramAttr::get(context, compileUnitAttr, fileAttr,
                                    funcNameAttr, funcNameAttr, fileAttr,
                                    /*line=*/line,
                                    /*scopeline=*/col,
                                    LLVM::DISubprogramFlags::Definition |
                                        LLVM::DISubprogramFlags::Optimized,
                                    subroutineTypeAttr);
    llvmFunc->setLoc(FusedLoc::get(context, {loc}, subprogramAttr));
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> mlir::LLVM::createDIScopeForLLVMFuncOpPass() {
  return std::make_unique<DIScopeForLLVMFuncOp>();
}