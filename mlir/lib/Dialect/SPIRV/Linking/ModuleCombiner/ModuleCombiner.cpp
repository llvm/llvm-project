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

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
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

template <typename KeyTy, typename SymbolOpTy>
static SymbolOpTy
emplaceOrGetReplacementSymbol(KeyTy key, SymbolOpTy symbolOp,
                              DenseMap<KeyTy, SymbolOpTy> &deduplicationMap) {
  auto result = deduplicationMap.try_emplace(key, symbolOp);

  if (result.second)
    return SymbolOpTy();

  return result.first->second;
}

/// Computes a hash code to represent the argument SymbolOpInterface based on
/// all the Op's attributes except for the symbol name.
///
/// \return the hash code computed from the Op's attributes as described above.
///
/// Note: We use the operation's name (not the symbol name) as part of the hash
/// computation. This prevents, for example, mistakenly considering a global
/// variable and a spec constant as duplicates because their descriptor set +
/// binding and spec_id, respectively, happen to hash to the same value.
static llvm::hash_code computeHash(SymbolOpInterface symbolOp) {
  llvm::hash_code hashCode(0);
  hashCode = llvm::hash_combine(symbolOp->getName());

  for (auto attr : symbolOp->getAttrs()) {
    if (attr.first == SymbolTable::getSymbolAttrName())
      continue;
    hashCode = llvm::hash_combine(hashCode, attr);
  }

  return hashCode;
}

/// Computes a hash code from the argument Block.
llvm::hash_code computeHash(Block *block) {
  // TODO: Consider extracting BlockEquivalenceData into a common header and
  // re-using it here.
  llvm::hash_code hash(0);

  for (Operation &op : *block) {
    // TODO: Properly handle operations with regions.
    if (op.getNumRegions() > 0)
      return 0;

    hash = llvm::hash_combine(
        hash, OperationEquivalence::computeHash(
                  &op, OperationEquivalence::Flags::IgnoreOperands));
  }

  return hash;
}

namespace mlir {
namespace spirv {

// TODO Properly test symbol rename listener mechanism.

OwningOpRef<spirv::ModuleOp>
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
  // variable is renamed. In order to notify listeners of the symbol updates in
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
    // from the current input module. This renaming applies to all ops except
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
          // entry that associates the old symbol name with the original module.
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

  // Deduplicate identical global variables, spec constants, and functions.
  DenseMap<llvm::hash_code, SymbolOpInterface> hashToSymbolOp;
  SmallVector<SymbolOpInterface, 0> eraseList;

  for (auto &op : combinedModule.getBlock().without_terminator()) {
    llvm::hash_code hashCode(0);
    SymbolOpInterface symbolOp = dyn_cast<SymbolOpInterface>(op);

    if (!symbolOp)
      continue;

    hashCode = computeHash(symbolOp);

    // A 0 hash code means the op is not suitable for deduplication and should
    // be skipped. An example of this is when a function has ops with regions
    // which are not properly supported yet.
    if (!hashCode)
      continue;

    if (auto funcOp = dyn_cast<FuncOp>(op))
      for (auto &blk : funcOp)
        hashCode = llvm::hash_combine(hashCode, computeHash(&blk));

    SymbolOpInterface replacementSymOp =
        emplaceOrGetReplacementSymbol(hashCode, symbolOp, hashToSymbolOp);

    if (!replacementSymOp)
      continue;

    if (failed(SymbolTable::replaceAllSymbolUses(
            symbolOp, replacementSymOp.getName(), combinedModule))) {
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
