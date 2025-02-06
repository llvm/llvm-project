//===- LinkageInterfaces.cpp - Interfaces for Linkage -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/LinkageInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Linker/Comdat.h"

using namespace mlir;
using namespace mlir::link;

/// Include the definitions of the interface.
#include "mlir/Interfaces/LinkageInterfaces.cpp.inc"

ComdatPair *
LinkableModuleOpInterface::getOrInsertComdat(ComdatSymbolTable &table,
                                             StringRef name) {
  auto &entry = *table.try_emplace(name, ComdatPair()).first;
  entry.second.name = &entry;
  return &(entry.second);
}

ComdatSymbolTable LinkableModuleOpInterface::getComdatSymbolTable() {
  ComdatSymbolTable table;

  this->walk([&](GlobalValueLinkageOpInterface op) {
    if (auto pair = op.getComdatPair()) {
      auto [name, kind] = *pair;
      ComdatPair *comdat = getOrInsertComdat(table, name);
      comdat->setSelectionKind(kind);
    }

    return WalkResult::skip();
  });

  return table;
}
