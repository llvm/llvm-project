//===- TestFunc.cpp - Pass to test helpers on function utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This is a test pass for verifying FunctionOpInterface's insertArgument
/// method.
struct TestFuncInsertArg
    : public PassWrapper<TestFuncInsertArg, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFuncInsertArg)

  StringRef getArgument() const final { return "test-func-insert-arg"; }
  StringRef getDescription() const final { return "Test inserting func args."; }
  void runOnOperation() override {
    auto module = getOperation();

    UnknownLoc unknownLoc = UnknownLoc::get(module.getContext());
    for (auto func : module.getOps<FunctionOpInterface>()) {
      auto inserts = func->getAttrOfType<ArrayAttr>("test.insert_args");
      if (!inserts || inserts.empty())
        continue;
      SmallVector<unsigned, 4> indicesToInsert;
      SmallVector<Type, 4> typesToInsert;
      SmallVector<DictionaryAttr, 4> attrsToInsert;
      SmallVector<Location, 4> locsToInsert;
      for (auto insert : inserts.getAsRange<ArrayAttr>()) {
        indicesToInsert.push_back(
            cast<IntegerAttr>(insert[0]).getValue().getZExtValue());
        typesToInsert.push_back(cast<TypeAttr>(insert[1]).getValue());
        attrsToInsert.push_back(insert.size() > 2
                                    ? cast<DictionaryAttr>(insert[2])
                                    : DictionaryAttr::get(&getContext()));
        locsToInsert.push_back(insert.size() > 3
                                   ? Location(cast<LocationAttr>(insert[3]))
                                   : unknownLoc);
      }
      func->removeAttr("test.insert_args");
      func.insertArguments(indicesToInsert, typesToInsert, attrsToInsert,
                           locsToInsert);
    }
  }
};

/// This is a test pass for verifying FunctionOpInterface's insertResult method.
struct TestFuncInsertResult
    : public PassWrapper<TestFuncInsertResult, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFuncInsertResult)

  StringRef getArgument() const final { return "test-func-insert-result"; }
  StringRef getDescription() const final {
    return "Test inserting func results.";
  }
  void runOnOperation() override {
    auto module = getOperation();

    for (auto func : module.getOps<FunctionOpInterface>()) {
      auto inserts = func->getAttrOfType<ArrayAttr>("test.insert_results");
      if (!inserts || inserts.empty())
        continue;
      SmallVector<unsigned, 4> indicesToInsert;
      SmallVector<Type, 4> typesToInsert;
      SmallVector<DictionaryAttr, 4> attrsToInsert;
      for (auto insert : inserts.getAsRange<ArrayAttr>()) {
        indicesToInsert.push_back(
            cast<IntegerAttr>(insert[0]).getValue().getZExtValue());
        typesToInsert.push_back(cast<TypeAttr>(insert[1]).getValue());
        attrsToInsert.push_back(insert.size() > 2
                                    ? cast<DictionaryAttr>(insert[2])
                                    : DictionaryAttr::get(&getContext()));
      }
      func->removeAttr("test.insert_results");
      func.insertResults(indicesToInsert, typesToInsert, attrsToInsert);
    }
  }
};

/// This is a test pass for verifying FunctionOpInterface's eraseArgument
/// method.
struct TestFuncEraseArg
    : public PassWrapper<TestFuncEraseArg, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFuncEraseArg)

  StringRef getArgument() const final { return "test-func-erase-arg"; }
  StringRef getDescription() const final { return "Test erasing func args."; }
  void runOnOperation() override {
    auto module = getOperation();

    for (auto func : module.getOps<FunctionOpInterface>()) {
      BitVector indicesToErase(func.getNumArguments());
      for (auto argIndex : llvm::seq<int>(0, func.getNumArguments()))
        if (func.getArgAttr(argIndex, "test.erase_this_arg"))
          indicesToErase.set(argIndex);
      func.eraseArguments(indicesToErase);
    }
  }
};

/// This is a test pass for verifying FunctionOpInterface's eraseResult method.
struct TestFuncEraseResult
    : public PassWrapper<TestFuncEraseResult, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFuncEraseResult)

  StringRef getArgument() const final { return "test-func-erase-result"; }
  StringRef getDescription() const final {
    return "Test erasing func results.";
  }
  void runOnOperation() override {
    auto module = getOperation();

    for (auto func : module.getOps<FunctionOpInterface>()) {
      BitVector indicesToErase(func.getNumResults());
      for (auto resultIndex : llvm::seq<int>(0, func.getNumResults()))
        if (func.getResultAttr(resultIndex, "test.erase_this_result"))
          indicesToErase.set(resultIndex);
      func.eraseResults(indicesToErase);
    }
  }
};

/// This is a test pass for verifying FunctionOpInterface's setType method.
struct TestFuncSetType
    : public PassWrapper<TestFuncSetType, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFuncSetType)

  StringRef getArgument() const final { return "test-func-set-type"; }
  StringRef getDescription() const final {
    return "Test FunctionOpInterface::setType.";
  }
  void runOnOperation() override {
    auto module = getOperation();
    SymbolTable symbolTable(module);

    for (auto func : module.getOps<FunctionOpInterface>()) {
      auto sym = func->getAttrOfType<FlatSymbolRefAttr>("test.set_type_from");
      if (!sym)
        continue;
      func.setType(symbolTable.lookup<FunctionOpInterface>(sym.getValue())
                       .getFunctionType());
    }
  }
};
} // namespace

namespace mlir {
void registerTestFunc() {
  PassRegistration<TestFuncInsertArg>();

  PassRegistration<TestFuncInsertResult>();

  PassRegistration<TestFuncEraseArg>();

  PassRegistration<TestFuncEraseResult>();

  PassRegistration<TestFuncSetType>();
}
} // namespace mlir
