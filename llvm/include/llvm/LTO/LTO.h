//===-LTO.h - LLVM Link Time Optimizer ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares functions and classes used to support LTO. It is intended
// to be used both by LTO classes as well as by clients (gold-plugin) that
// don't utilize the LTO code generator interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_LTO_H
#define LLVM_LTO_LTO_H

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/ModuleSummaryIndex.h"

namespace llvm {

class LLVMContext;
class MemoryBufferRef;
class Module;

/// Helper to load a module from bitcode.
std::unique_ptr<Module> loadModuleFromBuffer(const MemoryBufferRef &Buffer,
                                             LLVMContext &Context, bool Lazy);

/// Provide a "loader" for the FunctionImporter to access function from other
/// modules.
class ModuleLoader {
  /// The context that will be used for importing.
  LLVMContext &Context;

  /// Map from Module identifier to MemoryBuffer. Used by clients like the
  /// FunctionImported to request loading a Module.
  StringMap<MemoryBufferRef> &ModuleMap;

public:
  ModuleLoader(LLVMContext &Context, StringMap<MemoryBufferRef> &ModuleMap)
      : Context(Context), ModuleMap(ModuleMap) {}

  /// Load a module on demand.
  std::unique_ptr<Module> operator()(StringRef Identifier) {
    return loadModuleFromBuffer(ModuleMap[Identifier], Context, /*Lazy*/ true);
  }
};


/// Resolve Weak and LinkOnce values in the \p Index. Linkage changes recorded
/// in the index and the ThinLTO backends must apply the changes to the Module
/// via thinLTOResolveWeakForLinkerModule.
///
/// This is done for correctness (if value exported, ensure we always
/// emit a copy), and compile-time optimization (allow drop of duplicates).
void thinLTOResolveWeakForLinkerInIndex(
    ModuleSummaryIndex &Index,
    function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
        isPrevailing,
    function_ref<bool(StringRef, GlobalValue::GUID)> isExported,
    function_ref<void(StringRef, GlobalValue::GUID, GlobalValue::LinkageTypes)>
        recordNewLinkage);

/// Update the linkages in the given \p Index to mark exported values
/// as external and non-exported values as internal. The ThinLTO backends
/// must apply the changes to the Module via thinLTOInternalizeModule.
void thinLTOInternalizeAndPromoteInIndex(
    ModuleSummaryIndex &Index,
    function_ref<bool(StringRef, GlobalValue::GUID)> isExported);
}

#endif
