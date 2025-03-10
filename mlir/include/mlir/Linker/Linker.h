//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_LINKER_H
#define MLIR_LINKER_LINKER_H

#include "mlir/IR/BuiltinOps.h"

#include "mlir/Linker/LinkerInterface.h"
namespace mlir::link {

/// These are gathered alphabetically sorted linker options
class LinkerOptions {
public:
  LinkerOptions &allowUnregisteredDialects(bool allow) {
    allowUnregisteredDialectsFlag = allow;
    return *this;
  }
  bool shouldAllowUnregisteredDialects() const {
    return allowUnregisteredDialectsFlag;
  }

  LinkerOptions &linkOnlyNeeded(bool allow) {
    linkOnlyNeededFlag = allow;
    return *this;
  }
  bool shouldLinkOnlyNeeded() const { return linkOnlyNeededFlag; }

  LinkerOptions &keepModulesAlive(bool keep) {
    keepModulesAliveFlag = keep;
    return *this;
  }
  bool shouldKeepModulesAlive() const { return keepModulesAliveFlag; }

protected:
  /// Whether to allow operation with no registered dialects
  bool allowUnregisteredDialectsFlag = false;

  /// Whether to link only needed symbols
  bool linkOnlyNeededFlag = false;

  /// Keep modules alive until the end of the linking
  /// TODO: Add on-the-fly linking
  bool keepModulesAliveFlag = true;
};

/// This class provides the core functionality of linking in MLIR, it mirrors
/// functionality from `llvm/Linker/Linker.h` for MLIR. It keeps a pointer to
/// the merged module so far. It doesn't take ownership of the module since it
/// is assumed that the user of this class will want to do something with it
/// after the linking.
class Linker {
public:
  Linker(MLIRContext *context, const LinkerOptions &options = {})
      : context(context), options(options) {}

  /// Add a module to be linked
  LogicalResult addModule(OwningOpRef<ModuleOp> src);

  /// Perform linking and materialization in the destination module.
  /// Returns the linked module.
  OwningOpRef<ModuleOp> link(bool sortSymbols = false);

  MLIRContext *getContext() { return context; }

  LogicalResult emitFileError(const Twine &fileName, const Twine &message);
  LogicalResult emitError(const Twine &message);

private:
  /// Setup the linker based on the first module
  LogicalResult initializeLinker(ModuleOp src);

  /// Obtain the linker interface for the given module
  ModuleLinkerInterface *getModuleLinkerInterface(ModuleOp op);

  /// Return the flags controlling the linker behavior for the current module
  unsigned getFlags() const;

  /// Preprocess the given module before linking with the given flags
  LogicalResult process(ModuleOp src, unsigned flags);

  /// The context used for the linker
  MLIRContext *context;

  /// The options controlling the linker behavior
  LinkerOptions options;

  /// The composti module that will contain the linked result
  OwningOpRef<ModuleOp> composite;

  /// Modules registry used if `keepModulesAlive` is true
  std::vector<OwningOpRef<ModuleOp>> modules;
};

} // namespace mlir::link

#endif
