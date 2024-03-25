//===- TestSymbolUses.cpp - Pass to test symbol uselists ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This is a symbol test pass that tests the symbol uselist functionality
/// provided by the symbol table along with erasing from the symbol table.
struct SymbolUsesPass
    : public PassWrapper<SymbolUsesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SymbolUsesPass)

  StringRef getArgument() const final { return "test-symbol-uses"; }
  StringRef getDescription() const final {
    return "Test detection of symbol uses";
  }
  WalkResult operateOnSymbol(Operation *symbol, ModuleOp module,
                             SmallVectorImpl<func::FuncOp> &deadFunctions) {
    // Test computing uses on a non symboltable op.
    std::optional<SymbolTable::UseRange> symbolUses =
        SymbolTable::getSymbolUses(symbol);

    // Test the conservative failure case.
    if (!symbolUses) {
      symbol->emitRemark()
          << "symbol contains an unknown nested operation that "
             "'may' define a new symbol table";
      return WalkResult::interrupt();
    }
    if (unsigned numUses = llvm::size(*symbolUses))
      symbol->emitRemark() << "symbol contains " << numUses
                           << " nested references";

    // Test the functionality of symbolKnownUseEmpty.
    if (SymbolTable::symbolKnownUseEmpty(symbol, &module.getBodyRegion())) {
      func::FuncOp funcSymbol = dyn_cast<func::FuncOp>(symbol);
      if (funcSymbol && funcSymbol.isExternal())
        deadFunctions.push_back(funcSymbol);

      symbol->emitRemark() << "symbol has no uses";
      return WalkResult::advance();
    }

    // Test the functionality of getSymbolUses.
    symbolUses = SymbolTable::getSymbolUses(symbol, &module.getBodyRegion());
    assert(symbolUses && "expected no unknown operations");
    for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
      // Check that we can resolve back to our symbol.
      if (SymbolTable::lookupNearestSymbolFrom(
              symbolUse.getUser()->getParentOp(), symbolUse.getSymbolRef())) {
        symbolUse.getUser()->emitRemark()
            << "found use of symbol : " << symbolUse.getSymbolRef() << " : "
            << *symbol->getInherentAttr(SymbolTable::getSymbolAttrName());
      }
    }
    symbol->emitRemark() << "symbol has " << llvm::size(*symbolUses) << " uses";
    return WalkResult::advance();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Walk nested symbols.
    SmallVector<func::FuncOp, 4> deadFunctions;
    module.getBodyRegion().walk([&](Operation *nestedOp) {
      if (isa<SymbolOpInterface>(nestedOp))
        return operateOnSymbol(nestedOp, module, deadFunctions);
      return WalkResult::advance();
    });

    SymbolTable table(module);
    for (Operation *op : deadFunctions) {
      // In order to test the SymbolTable::erase method, also erase completely
      // useless functions.
      auto name = SymbolTable::getSymbolName(op);
      assert(table.lookup(name) && "expected no unknown operations");
      table.erase(op);
      assert(!table.lookup(name) &&
             "expected erased operation to be unknown now");
      module.emitRemark() << name.getValue() << " function successfully erased";
    }
  }
};

/// This is a symbol test pass that tests the symbol use replacement
/// functionality provided by the symbol table.
struct SymbolReplacementPass
    : public PassWrapper<SymbolReplacementPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SymbolReplacementPass)

  StringRef getArgument() const final { return "test-symbol-rauw"; }
  StringRef getDescription() const final {
    return "Test replacement of symbol uses";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Don't try to replace if we can't collect symbol uses.
    if (!SymbolTable::getSymbolUses(&module.getBodyRegion()))
      return;

    SymbolTableCollection symbolTable;
    SymbolUserMap symbolUsers(symbolTable, module);
    module.getBodyRegion().walk([&](Operation *nestedOp) {
      StringAttr newName = nestedOp->getAttrOfType<StringAttr>("sym.new_name");
      if (!newName)
        return;
      symbolUsers.replaceAllUsesWith(nestedOp, newName);
      SymbolTable::setSymbolName(nestedOp, newName);
    });
  }
};
} // namespace

namespace mlir {
void registerSymbolTestPasses() {
  PassRegistration<SymbolUsesPass>();

  PassRegistration<SymbolReplacementPass>();
}
} // namespace mlir
