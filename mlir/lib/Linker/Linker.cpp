//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/Linker.h"

#include "mlir/IR/Builders.h"
#include "mlir/Linker/LinkerInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#define DEBUG_TYPE "mlir-linker"

using namespace mlir;
using namespace mlir::link;

LogicalResult Linker::initializeLinker(ModuleOp src) {
  ModuleLinkerInterface *iface = getModuleLinkerInterface(src);
  if (!iface)
    return emitError("Module does not have a linker interface");

  if (failed(iface->initialize(src)))
    return emitError("Failed to initialize module linker interface");

  composite = iface->createCompositeModule(src);
  if (!composite)
    return emitError("Failed to create composite module");

  return success();
}

ModuleLinkerInterface *Linker::getModuleLinkerInterface(ModuleOp op) {
  return dyn_cast_or_null<ModuleLinkerInterface>(op->getDialect());
}

LogicalResult Linker::addModule(OwningOpRef<ModuleOp> src) {
  unsigned flags = getFlags();

  ModuleOp mod = [&] {
    if (options.shouldKeepModulesAlive()) {
      modules.push_back(std::move(src));
      return modules.back().get();
    }
    return src.get();
  }();

  // If this is the first module, setup the linker based on it
  if (!composite) {
    if (failed(initializeLinker(mod)))
      return failure();

    // We always override from source for the first module.
    flags &= LinkerFlags::OverrideFromSrc;
  }

  return process(mod, flags);
}

LogicalResult Linker::process(ModuleOp src, unsigned flags) {
  ModuleLinkerInterface *iface = getModuleLinkerInterface(src);
  if (!iface)
    return emitError("Module does not have a linker interface");
  return iface->process(src, flags);
}

OwningOpRef<ModuleOp> Linker::link(bool sortSymbols) {
  ModuleOp mod = composite.get();
  if (failed(getModuleLinkerInterface(mod)->link(mod)))
    return nullptr;

  if (sortSymbols) {
    std::vector<Operation *> symbols;

    mod->walk([&](Operation *op) {
      if (auto iface = dyn_cast<SymbolLinkerInterface>(op->getDialect())) {
        if (iface->canBeLinked(op)) {
          symbols.push_back(op);
        }
      }
    });

    llvm::stable_sort(symbols, [](Operation *lhs, Operation *rhs) {
      auto lhsSym = cast<SymbolLinkerInterface>(lhs->getDialect());
      auto rhsSym = cast<SymbolLinkerInterface>(rhs->getDialect());
      return lhsSym->getSymbol(lhs) < rhsSym->getSymbol(rhs);
    });

    for (Operation *symbol : llvm::reverse(symbols)) {
      symbol->moveBefore(&mod.front());
    }
  }

  return std::move(composite);
}

unsigned Linker::getFlags() const {
  unsigned flags = None;

  if (options.shouldLinkOnlyNeeded())
    flags |= LinkerFlags::LinkOnlyNeeded;

  return flags;
}

LogicalResult Linker::emitFileError(const Twine &file, const Twine &msg) {
  return emitError("Error processing file '" + file + "': " + msg);
}

LogicalResult Linker::emitError(const Twine &msg) {
  return mlir::emitError(UnknownLoc::get(context), msg);
}
