//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to link builtin dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinLinkerInterface.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Linker/LinkerInterface.h"

using namespace mlir;
using namespace mlir::link;

//===----------------------------------------------------------------------===//
// BuiltinLinkerInterface
//===----------------------------------------------------------------------===//

class BuiltinLinkerInterface : public ModuleLinkerInterface {
public:
  using ModuleLinkerInterface::ModuleLinkerInterface;

  LogicalResult initialize(ModuleOp src) override {
    symbolLinkers = SymbolLinkerInterfaces(src.getContext());
    return symbolLinkers.initialize(src);
  }

  LogicalResult summarize(ModuleOp src, unsigned flags) override {
    WalkResult result = src.walk([&](Operation *op) {
      if (op == src)
        return WalkResult::advance();

      auto linker = dyn_cast<SymbolLinkerInterface>(op->getDialect());
      if (!linker)
        return WalkResult::advance();

      // TODO do this in init
      linker->setFlags(flags);

      if (!linker->canBeLinked(op))
        return WalkResult::advance();

      ConflictPair conflict = linker->findConflict(op);
      if (!linker->isLinkNeeded(conflict))
        return WalkResult::advance();

      if (conflict.hasConflict())
        return failed(linker->resolveConflict(conflict))
                   ? WalkResult::interrupt()
                   : WalkResult::advance();

      // TODO rename: registerForLink
      linker->registerForLink(op);
      return WalkResult::advance();
    });

    // TODO deal with references

    return failure(result.wasInterrupted());
  }

  LogicalResult link(ModuleOp dst) const override {
    return symbolLinkers.link(dst);
  }

  OwningOpRef<ModuleOp> createCompositeModule(ModuleOp src) override {
    return ModuleOp::create(
        FileLineColLoc::get(src.getContext(), "composite", 0, 0));
  }

private:
  SymbolLinkerInterfaces symbolLinkers;
};

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::builtin::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    dialect->addInterface<BuiltinLinkerInterface>();
  });
}
