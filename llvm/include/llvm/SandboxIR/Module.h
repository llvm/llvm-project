//===- Module.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_MODULE_H
#define LLVM_SANDBOXIR_MODULE_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Module.h"
#include <string>

namespace llvm {

class DataLayout;

namespace sandboxir {

class Context;
class Function;
class GlobalVariable;
class Type;
class Constant;
class GlobalAlias;
class GlobalIFunc;

/// In SandboxIR the Module is mainly used to access the list of global objects.
class Module {
  llvm::Module &LLVMM;
  Context &Ctx;

  Module(llvm::Module &LLVMM, Context &Ctx) : LLVMM(LLVMM), Ctx(Ctx) {}
  friend class Context; // For constructor.

public:
  Context &getContext() const { return Ctx; }

  Function *getFunction(StringRef Name) const;

  const DataLayout &getDataLayout() const { return LLVMM.getDataLayout(); }

  const std::string &getSourceFileName() const {
    return LLVMM.getSourceFileName();
  }

  /// Look up the specified global variable in the module symbol table. If it
  /// does not exist, return null. If AllowInternal is set to true, this
  /// function will return types that have InternalLinkage. By default, these
  /// types are not returned.
  GlobalVariable *getGlobalVariable(StringRef Name, bool AllowInternal) const;
  GlobalVariable *getGlobalVariable(StringRef Name) const {
    return getGlobalVariable(Name, /*AllowInternal=*/false);
  }
  /// Return the global variable in the module with the specified name, of
  /// arbitrary type. This method returns null if a global with the specified
  /// name is not found.
  GlobalVariable *getNamedGlobal(StringRef Name) const {
    return getGlobalVariable(Name, true);
  }

  // TODO: missing getOrInsertGlobal().

  /// Return the global alias in the module with the specified name, of
  /// arbitrary type. This method returns null if a global with the specified
  /// name is not found.
  GlobalAlias *getNamedAlias(StringRef Name) const;

  /// Return the global ifunc in the module with the specified name, of
  /// arbitrary type. This method returns null if a global with the specified
  /// name is not found.
  GlobalIFunc *getNamedIFunc(StringRef Name) const;

  // TODO: Missing removeGlobalVariable() eraseGlobalVariable(),
  // insertGlobalVariable()

  // TODO: Missing global_begin(), global_end(), globals().

  // TODO: Missing many other functions.

#ifndef NDEBUG
  void dumpOS(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif // NDEBUG
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_SANDBOXIR_MODULE_H
