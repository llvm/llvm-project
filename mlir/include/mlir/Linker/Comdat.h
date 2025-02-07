//===- Comdat.h - MLIR Comdat -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_COMDAT_H
#define MLIR_LINKER_COMDAT_H

#include "llvm/ADT/StringMap.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

namespace mlir {

class LinkableModuleOpInterface;

namespace link {

using ComdatSelectionKind = LLVM::comdat::Comdat;

class ComdatEntry {
public:
  ComdatEntry(const ComdatEntry &) = delete;
  ComdatEntry(ComdatEntry &&) = default;

  StringRef getName() const { return name->getKey(); }

  ComdatSelectionKind getSelectionKind() const { return kind; }
  void setSelectionKind(ComdatSelectionKind selectionKind) {
    kind = selectionKind;
  }

  const llvm::SmallPtrSetImpl<Operation *> &getUsers() const { return users; }

private:
  friend class ::mlir::LinkableModuleOpInterface;

  ComdatEntry() = default;

  void addUser(Operation *user) { users.insert(user); }
  void removeUser(Operation *user) { users.erase(user); }

  llvm::StringMapEntry<ComdatEntry> *name = nullptr;
  ComdatSelectionKind kind = ComdatSelectionKind::Any;

  // Globals using this comdat.
  llvm::SmallPtrSet<Operation *, 2> users;
};

using ComdatSymbolTable = llvm::StringMap<ComdatEntry>;

using ComdatPair = std::pair<StringRef, ComdatSelectionKind>;

} // namespace link
} // namespace mlir

#endif // MLIR_LINKER_COMDAT_H
