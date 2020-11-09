//===- ModuleCombiner.cpp - MLIR SPIR-V Module Combiner ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the the SPIR-V module combiner library.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/ModuleCombiner.h"

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;

static constexpr unsigned maxFreeID = 1 << 20;

static SmallString<64> renameSymbol(StringRef oldSymName, unsigned &lastUsedID,
                                    spirv::ModuleOp combinedModule) {
  SmallString<64> newSymName(oldSymName);
  newSymName.push_back('_');

  while (lastUsedID < maxFreeID) {
    std::string possible = (newSymName + llvm::utostr(++lastUsedID)).str();

    if (!SymbolTable::lookupSymbolIn(combinedModule, possible)) {
      newSymName += llvm::utostr(lastUsedID);
      break;
    }
  }

  return newSymName;
}

/// Check if a symbol with the same name as op already exists in source. If so,
/// rename op and update all its references in target.
static LogicalResult updateSymbolAndAllUses(SymbolOpInterface op,
                                            spirv::ModuleOp target,
                                            spirv::ModuleOp source,
                                            unsigned &lastUsedID) {
  if (!SymbolTable::lookupSymbolIn(source, op.getName()))
    return success();

  StringRef oldSymName = op.getName();
  SmallString<64> newSymName = renameSymbol(oldSymName, lastUsedID, target);

  if (failed(SymbolTable::replaceAllSymbolUses(op, newSymName, target)))
    return op.emitError("unable to update all symbol uses for ")
           << oldSymName << " to " << newSymName;

  SymbolTable::setSymbolName(op, newSymName);
  return success();
}

namespace mlir {
namespace spirv {

// TODO Properly test symbol rename listener mechanism.

OwningSPIRVModuleRef
combine(llvm::MutableArrayRef<spirv::ModuleOp> modules,
        OpBuilder &combinedModuleBuilder,
        llvm::function_ref<void(ModuleOp, StringRef, StringRef)>
            symRenameListener) {
  unsigned lastUsedID = 0;

  if (modules.empty())
    return nullptr;

  auto addressingModel = modules[0].addressing_model();
  auto memoryModel = modules[0].memory_model();

  auto combinedModule = combinedModuleBuilder.create<spirv::ModuleOp>(
      modules[0].getLoc(), addressingModel, memoryModel);
  combinedModuleBuilder.setInsertionPointToStart(&*combinedModule.getBody());

  // In some cases, a symbol in the (current state of the) combined module is
  // renamed in order to maintain the conflicting symbol in the input module
  // being merged. For example, if the conflict is between a global variable in
  // the current combined module and a function in the input module, the global
  // varaible is renamed. In order to notify listeners of the symbol updates in
  // such cases, we need to keep track of the module from which the renamed
  // symbol in the combined module originated. This map keeps such information.
  DenseMap<StringRef, spirv::ModuleOp> symNameToModuleMap;

  for (auto module : modules) {
    if (module.addressing_model() != addressingModel ||
        module.memory_model() != memoryModel) {
      module.emitError(
          "input modules differ in addressing model and/or memory model");
      return nullptr;
    }

    spirv::ModuleOp moduleClone = module.clone();

    // In the combined module, rename all symbols that conflict with symbols
    // from the current input module. This renmaing applies to all ops except
    // for spv.funcs. This way, if the conflicting op in the input module is
    // non-spv.func, we rename that symbol instead and maintain the spv.func in
    // the combined module name as it is.
    for (auto &op : combinedModule.getBlock().without_terminator()) {
      if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
        StringRef oldSymName = symbolOp.getName();

        if (!isa<FuncOp>(op) &&
            failed(updateSymbolAndAllUses(symbolOp, combinedModule, moduleClone,
                                          lastUsedID)))
          return nullptr;

        StringRef newSymName = symbolOp.getName();

        if (symRenameListener && oldSymName != newSymName) {
          spirv::ModuleOp originalModule =
              symNameToModuleMap.lookup(oldSymName);

          if (!originalModule) {
            module.emitError("unable to find original ModuleOp for symbol ")
                << oldSymName;
            return nullptr;
          }

          symRenameListener(originalModule, oldSymName, newSymName);

          // Since the symbol name is updated, there is no need to maintain the
          // entry that assocaites the old symbol name with the original module.
          symNameToModuleMap.erase(oldSymName);
          // Instead, add a new entry to map the new symbol name to the original
          // module in case it gets renamed again later.
          symNameToModuleMap[newSymName] = originalModule;
        }
      }
    }

    // In the current input module, rename all symbols that conflict with
    // symbols from the combined module. This includes renaming spv.funcs.
    for (auto &op : moduleClone.getBlock().without_terminator()) {
      if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
        StringRef oldSymName = symbolOp.getName();

        if (failed(updateSymbolAndAllUses(symbolOp, moduleClone, combinedModule,
                                          lastUsedID)))
          return nullptr;

        StringRef newSymName = symbolOp.getName();

        if (symRenameListener && oldSymName != newSymName) {
          symRenameListener(module, oldSymName, newSymName);

          // Insert the module associated with the symbol name.
          auto emplaceResult =
              symNameToModuleMap.try_emplace(symbolOp.getName(), module);

          // If an entry with the same symbol name is already present, this must
          // be a problem with the implementation, specially clean-up of the map
          // while iterating over the combined module above.
          if (!emplaceResult.second) {
            module.emitError("did not expect to find an entry for symbol ")
                << symbolOp.getName();
            return nullptr;
          }
        }
      }
    }

    // Clone all the module's ops to the combined module.
    for (auto &op : moduleClone.getBlock().without_terminator())
      combinedModuleBuilder.insert(op.clone());
  }

  return combinedModule;
}

} // namespace spirv
} // namespace mlir
