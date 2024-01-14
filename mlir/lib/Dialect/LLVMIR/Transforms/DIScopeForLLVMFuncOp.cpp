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

/// Creates a DISubprogramAttr with the provided compile unit and attaches it
/// to the function. Does nothing when the function already has an attached
/// subprogram.
static void addScopeToFunction(LLVM::LLVMFuncOp llvmFunc,
                               LLVM::DICompileUnitAttr compileUnitAttr) {

  Location loc = llvmFunc.getLoc();
  if (loc->findInstanceOf<mlir::FusedLocWith<LLVM::DISubprogramAttr>>())
    return;

  MLIRContext *context = llvmFunc->getContext();

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
    fileAttr =
        LLVM::DIFileAttr::get(context, llvm::sys::path::filename(inputFilePath),
                              llvm::sys::path::parent_path(inputFilePath));
  }
  auto subroutineTypeAttr =
      LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, {});

  StringAttr funcNameAttr = llvmFunc.getNameAttr();
  // Only definitions need a distinct identifier and a compilation unit.
  mlir::DistinctAttr id;
  if (!llvmFunc.isExternal())
    id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
  else
    compileUnitAttr = {};
  mlir::LLVM::DISubprogramAttr subprogramAttr = LLVM::DISubprogramAttr::get(
      context, id, compileUnitAttr, fileAttr, funcNameAttr, funcNameAttr,
      fileAttr,
      /*line=*/line,
      /*scopeline=*/col,
      LLVM::DISubprogramFlags::Definition | LLVM::DISubprogramFlags::Optimized,
      subroutineTypeAttr);
  llvmFunc->setLoc(FusedLoc::get(context, {loc}, subprogramAttr));
}

namespace {
/// Add a debug info scope to LLVMFuncOp that are missing it.
struct DIScopeForLLVMFuncOp
    : public LLVM::impl::DIScopeForLLVMFuncOpBase<DIScopeForLLVMFuncOp> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Location loc = module.getLoc();

    MLIRContext *context = &getContext();

    // To find a DICompileUnitAttr attached to a parent (the module for
    // example), otherwise create a default one.
    // Find a DICompileUnitAttr attached to the module, otherwise create a
    // default one.
    LLVM::DICompileUnitAttr compileUnitAttr;
    auto fusedCompileUnitAttr =
        module->getLoc()
            ->findInstanceOf<mlir::FusedLocWith<LLVM::DICompileUnitAttr>>();
    if (fusedCompileUnitAttr) {
      compileUnitAttr = fusedCompileUnitAttr.getMetadata();
    } else {
      LLVM::DIFileAttr fileAttr;
      if (FileLineColLoc fileLoc = extractFileLoc(loc)) {
        StringRef inputFilePath = fileLoc.getFilename().getValue();
        fileAttr = LLVM::DIFileAttr::get(
            context, llvm::sys::path::filename(inputFilePath),
            llvm::sys::path::parent_path(inputFilePath));
      } else {
        fileAttr = LLVM::DIFileAttr::get(context, "<unknown>", "");
      }

      compileUnitAttr = LLVM::DICompileUnitAttr::get(
          context, DistinctAttr::create(UnitAttr::get(context)),
          llvm::dwarf::DW_LANG_C, fileAttr, StringAttr::get(context, "MLIR"),
          /*isOptimized=*/true, LLVM::DIEmissionKind::LineTablesOnly);
    }

    // Create subprograms for each function with the same distinct compile unit.
    module.walk([&](LLVM::LLVMFuncOp func) {
      addScopeToFunction(func, compileUnitAttr);
    });
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> mlir::LLVM::createDIScopeForLLVMFuncOpPass() {
  return std::make_unique<DIScopeForLLVMFuncOp>();
}
