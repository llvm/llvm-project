//===- ModuleCombiner.cpp - MLIR SPIR-V Module Combiner ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPIR-V module combiner library.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Linking/ModuleCombiner.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

static constexpr unsigned maxFreeID = 1 << 20;

/// Returns an unused symbol in `module` for `oldSymbolName` by trying numeric
/// suffix in `lastUsedID`.
static StringAttr renameSymbol(StringRef oldSymName, unsigned &lastUsedID,
                               spirv::ModuleOp module) {
  SmallString<64> newSymName(oldSymName);
  newSymName.push_back('_');

  MLIRContext *ctx = module->getContext();

  while (lastUsedID < maxFreeID) {
    auto possible = StringAttr::get(ctx, newSymName + Twine(++lastUsedID));
    if (!SymbolTable::lookupSymbolIn(module, possible))
      return possible;
  }

  return StringAttr::get(ctx, newSymName);
}

/// Checks if a symbol with the same name as `op` already exists in `source`.
/// If so, renames `op` and updates all its references in `target`.
static LogicalResult updateSymbolAndAllUses(SymbolOpInterface op,
                                            spirv::ModuleOp target,
                                            spirv::ModuleOp source,
                                            unsigned &lastUsedID) {
  if (!SymbolTable::lookupSymbolIn(source, op.getName()))
    return success();

  StringRef oldSymName = op.getName();
  StringAttr newSymName = renameSymbol(oldSymName, lastUsedID, target);

  if (failed(SymbolTable::replaceAllSymbolUses(op, newSymName, target)))
    return op.emitError("unable to update all symbol uses for ")
           << oldSymName << " to " << newSymName;

  SymbolTable::setSymbolName(op, newSymName);
  return success();
}

/// Computes a hash code to represent `symbolOp` based on all its attributes
/// except for the symbol name.
///
/// Note: We use the operation's name (not the symbol name) as part of the hash
/// computation. This prevents, for example, mistakenly considering a global
/// variable and a spec constant as duplicates because their descriptor set +
/// binding and spec_id, respectively, happen to hash to the same value.
static llvm::hash_code computeHash(SymbolOpInterface symbolOp) {
  auto range =
      llvm::make_filter_range(symbolOp->getAttrs(), [](NamedAttribute attr) {
        return attr.getName() != SymbolTable::getSymbolAttrName();
      });

  return llvm::hash_combine(
      symbolOp->getName(),
      llvm::hash_combine_range(range.begin(), range.end()));
}

namespace mlir {
namespace spirv {

OwningOpRef<spirv::ModuleOp> combine(ArrayRef<spirv::ModuleOp> inputModules,
                                     OpBuilder &combinedModuleBuilder,
                                     SymbolRenameListener symRenameListener) {
  if (inputModules.empty())
    return nullptr;

  spirv::ModuleOp firstModule = inputModules.front();
  auto addressingModel = firstModule.getAddressingModel();
  auto memoryModel = firstModule.getMemoryModel();
  auto vceTriple = firstModule.getVceTriple();

  // First check whether there are conflicts between addressing/memory model.
  // Return early if so.
  for (auto module : inputModules) {
    if (module.getAddressingModel() != addressingModel ||
        module.getMemoryModel() != memoryModel ||
        module.getVceTriple() != vceTriple) {
      module.emitError("input modules differ in addressing model, memory "
                       "model, and/or VCE triple");
      return nullptr;
    }
  }

  auto combinedModule = combinedModuleBuilder.create<spirv::ModuleOp>(
      firstModule.getLoc(), addressingModel, memoryModel, vceTriple);
  combinedModuleBuilder.setInsertionPointToStart(combinedModule.getBody());

  // In some cases, a symbol in the (current state of the) combined module is
  // renamed in order to enable the conflicting symbol in the input module
  // being merged. For example, if the conflict is between a global variable in
  // the current combined module and a function in the input module, the global
  // variable is renamed. In order to notify listeners of the symbol updates in
  // such cases, we need to keep track of the module from which the renamed
  // symbol in the combined module originated. This map keeps such information.
  llvm::StringMap<spirv::ModuleOp> symNameToModuleMap;

  unsigned lastUsedID = 0;

  for (auto inputModule : inputModules) {
    OwningOpRef<spirv::ModuleOp> moduleClone = inputModule.clone();

    // In the combined module, rename all symbols that conflict with symbols
    // from the current input module. This renaming applies to all ops except
    // for spirv.funcs. This way, if the conflicting op in the input module is
    // non-spirv.func, we rename that symbol instead and maintain the spirv.func
    // in the combined module name as it is.
    for (auto &op : *combinedModule.getBody()) {
      auto symbolOp = dyn_cast<SymbolOpInterface>(op);
      if (!symbolOp)
        continue;

      StringRef oldSymName = symbolOp.getName();

      if (!isa<FuncOp>(op) &&
          failed(updateSymbolAndAllUses(symbolOp, combinedModule, *moduleClone,
                                        lastUsedID)))
        return nullptr;

      StringRef newSymName = symbolOp.getName();

      if (symRenameListener && oldSymName != newSymName) {
        spirv::ModuleOp originalModule = symNameToModuleMap.lookup(oldSymName);

        if (!originalModule) {
          inputModule.emitError(
              "unable to find original spirv::ModuleOp for symbol ")
              << oldSymName;
          return nullptr;
        }

        symRenameListener(originalModule, oldSymName, newSymName);

        // Since the symbol name is updated, there is no need to maintain the
        // entry that associates the old symbol name with the original module.
        symNameToModuleMap.erase(oldSymName);
        // Instead, add a new entry to map the new symbol name to the original
        // module in case it gets renamed again later.
        symNameToModuleMap[newSymName] = originalModule;
      }
    }

    // In the current input module, rename all symbols that conflict with
    // symbols from the combined module. This includes renaming spirv.funcs.
    for (auto &op : *moduleClone->getBody()) {
      auto symbolOp = dyn_cast<SymbolOpInterface>(op);
      if (!symbolOp)
        continue;

      StringRef oldSymName = symbolOp.getName();

      if (failed(updateSymbolAndAllUses(symbolOp, *moduleClone, combinedModule,
                                        lastUsedID)))
        return nullptr;

      StringRef newSymName = symbolOp.getName();

      if (symRenameListener) {
        if (oldSymName != newSymName)
          symRenameListener(inputModule, oldSymName, newSymName);

        // Insert the module associated with the symbol name.
        auto emplaceResult =
            symNameToModuleMap.try_emplace(newSymName, inputModule);

        // If an entry with the same symbol name is already present, this must
        // be a problem with the implementation, specially clean-up of the map
        // while iterating over the combined module above.
        if (!emplaceResult.second) {
          inputModule.emitError("did not expect to find an entry for symbol ")
              << symbolOp.getName();
          return nullptr;
        }
      }
    }

    // Clone all the module's ops to the combined module.
    for (auto &op : *moduleClone->getBody())
      combinedModuleBuilder.insert(op.clone());
  }

  // Deduplicate identical global variables, spec constants, and functions.
  DenseMap<llvm::hash_code, SymbolOpInterface> hashToSymbolOp;
  SmallVector<SymbolOpInterface, 0> eraseList;

  for (auto &op : *combinedModule.getBody()) {
    SymbolOpInterface symbolOp = dyn_cast<SymbolOpInterface>(op);
    if (!symbolOp)
      continue;

    // Do not support ops with operands or results.
    // Global variables, spec constants, and functions won't have
    // operands/results, but just for safety here.
    if (op.getNumOperands() != 0 || op.getNumResults() != 0)
      continue;

    // Deduplicating functions are not supported yet.
    if (isa<FuncOp>(op))
      continue;

    auto result = hashToSymbolOp.try_emplace(computeHash(symbolOp), symbolOp);
    if (result.second)
      continue;

    SymbolOpInterface replacementSymOp = result.first->second;

    if (failed(SymbolTable::replaceAllSymbolUses(
            symbolOp, replacementSymOp.getNameAttr(), combinedModule))) {
      symbolOp.emitError("unable to update all symbol uses for ")
          << symbolOp.getName() << " to " << replacementSymOp.getName();
      return nullptr;
    }

    eraseList.push_back(symbolOp);
  }

  for (auto symbolOp : eraseList)
    symbolOp.erase();

  return combinedModule;
}

} // namespace spirv
} // namespace mlir
