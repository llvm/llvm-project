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
// BuiltinLinkerState
//===----------------------------------------------------------------------===//

struct BuiltinLinkerState : LinkerState::Base<BuiltinLinkerState> {
  static std::unique_ptr<LinkerState> create(ModuleOp src) {
    auto state = std::make_unique<BuiltinLinkerState>();

    // Walk the module and setup all operations with symbol linker interface
    WalkResult result = src.walk([&](Operation *op) {
      if (op == src)
        return WalkResult::advance();

      if (SymbolLinkerInterface *linker = getSymbolLinker(op))
        if (failed(linker->initialize(src)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return nullptr;

    return state;
  }

  Operation *lookup(Operation *op) const override {
    if (SymbolLinkerInterface *linker = getSymbolLinker(op))
      return linker->lookup(op);
    return nullptr;
  }

  void insertSymbolLinker(SymbolLinkerInterface *linker) {
    symbolLinkers.insert(linker);
  }

  ArrayRef<SymbolLinkerInterface *> getSymbolLinkers() const {
    return symbolLinkers.getArrayRef();
  }

private:
  static SymbolLinkerInterface *getSymbolLinker(Operation *op) {
    Dialect *dialect = op->getDialect();
    return dyn_cast<SymbolLinkerInterface>(dialect);
  }

  SetVector<SymbolLinkerInterface *> symbolLinkers;
};

//===----------------------------------------------------------------------===//
// BuiltinLinkerInterface
//===----------------------------------------------------------------------===//

struct BuiltinLinkerInterface
    : ModuleLinkerInterface::Base<BuiltinLinkerInterface, BuiltinLinkerState> {

  using Base =
      ModuleLinkerInterface::Base<BuiltinLinkerInterface, BuiltinLinkerState>;

  BuiltinLinkerInterface(Dialect *dialect) : Base(dialect) {}

  std::unique_ptr<LinkerState> init(ModuleOp src) const override {
    return BuiltinLinkerState::create(src);
  }

  LogicalResult process(ModuleOp src, unsigned flags) override {
    BuiltinLinkerState &state = getLinkerState();
    WalkResult result = src.walk([&](Operation *op) {
      if (op == src)
        return WalkResult::advance();

      auto linker = dyn_cast<SymbolLinkerInterface>(op->getDialect());
      if (!linker)
        return WalkResult::advance();

      state.insertSymbolLinker(linker);
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

      linker->registerOperation(op);
      return WalkResult::advance();
    });

    // TODO deal with references

    return failure(result.wasInterrupted());
  }

  LogicalResult link(ModuleOp dst) const override {
    const BuiltinLinkerState &state = getLinkerState();

    for (const auto *linker : state.getSymbolLinkers()) {
      if (failed(linker->link(dst)))
        return failure();
    }

    return success();
  }

  OwningOpRef<ModuleOp> createCompositeModule(ModuleOp src) override {
    return ModuleOp::create(
        FileLineColLoc::get(src.getContext(), "composite", 0, 0));
  }
};

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::builtin::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    dialect->addInterface<BuiltinLinkerInterface>();
  });
}
