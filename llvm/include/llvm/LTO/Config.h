//===-Config.h - LLVM Link Time Optimizer Configuration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the lto::Config data structure, which allows clients to
// configure LTO.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_CONFIG_H
#define LLVM_LTO_CONFIG_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/LibraryOptionsContext.h"
#include "llvm/Target/TargetOptions.h"

#include <functional>
#include <optional>

namespace llvm {

class Error;
class Module;
class ModuleSummaryIndex;
class raw_pwrite_stream;
class PassPlugin;

namespace lto {

/// LTO configuration. A linker can configure LTO by setting fields in this data
/// structure and passing it to the lto::LTO constructor.
struct Config {
  enum VisScheme {
    FromPrevailing,
    ELF,
  };
  // When adding fields, consider whether they affect computeLTOCacheKey in
  // LTO.cpp. Put serializable fields in ConfigFields.def.
#define LLVM_LTO_CONFIG_FIELD(Type, Name, Initializer) Type Name Initializer;
#define LLVM_LTO_CONFIG_PTO_FIELD(Type, Name) Type Name;
#include "llvm/LTO/ConfigFields.def"
#undef LLVM_LTO_CONFIG_PTO_FIELD
#undef LLVM_LTO_CONFIG_FIELD

  // Process-local state.
  // The library options of every LLVMContext that LTO creates.
  // TODO: Make it non-null once every client sets it. Null means the defaults.
  const cl::LibraryOptionsContext *LibraryOptions = nullptr;

  // Pass-plugins already loaded by the client. LTO does not take ownership, so
  // they must outlive the lto::LTO object.
  std::vector<llvm::PassPlugin *> LoadedPassPlugins;
  std::function<void(PassBuilder &)> PassBuilderCallback;
  DiagnosticHandlerFunction DiagHandler;

  /// If this field is set, LTO will write input file paths and symbol
  /// resolutions here in llvm-lto2 command line flag format. This can be
  /// used for testing and for running the LTO pipeline outside of the linker
  /// with llvm-lto2.
  std::unique_ptr<raw_ostream> ResolutionFile;

  /// The following callbacks deal with tasks, which normally represent the
  /// entire optimization and code generation pipeline for what will become a
  /// single native object file. Each task has a unique identifier between 0 and
  /// getMaxTasks()-1, which is supplied to the callback via the Task parameter.
  /// A task represents the entire pipeline for ThinLTO and regular
  /// (non-parallel) LTO, but a parallel code generation task will be split into
  /// N tasks before code generation, where N is the parallelism level.
  ///
  /// LTO may decide to stop processing a task at any time, for example if the
  /// module is empty or if a module hook (see below) returns false. For this
  /// reason, the client should not expect to receive exactly getMaxTasks()
  /// native object files.

  /// A module hook may be used by a linker to perform actions during the LTO
  /// pipeline. For example, a linker may use this function to implement
  /// -save-temps. If this function returns false, any further processing for
  /// that task is aborted.
  ///
  /// Module hooks must be thread safe with respect to the linker's internal
  /// data structures. A module hook will never be called concurrently from
  /// multiple threads with the same task ID, or the same module.
  ///
  /// Note that in out-of-process backend scenarios, none of the hooks will be
  /// called for ThinLTO tasks.
  using ModuleHookFn = std::function<bool(unsigned Task, const Module &)>;

  /// This module hook is called after linking (regular LTO) or loading
  /// (ThinLTO) the module, before modifying it.
  ModuleHookFn PreOptModuleHook;

  /// This hook is called after promoting any internal functions
  /// (ThinLTO-specific).
  ModuleHookFn PostPromoteModuleHook;

  /// This hook is called after internalizing the module.
  ModuleHookFn PostInternalizeModuleHook;

  /// This hook is called after importing from other modules (ThinLTO-specific).
  ModuleHookFn PostImportModuleHook;

  /// This module hook is called after optimization is complete.
  ModuleHookFn PostOptModuleHook;

  /// This module hook is called before code generation. It is similar to the
  /// PostOptModuleHook, but for parallel code generation it is called after
  /// splitting the module.
  ModuleHookFn PreCodeGenModuleHook;

  /// A combined index hook is called after all per-module indexes have been
  /// combined (ThinLTO-specific). It can be used to implement -save-temps for
  /// the combined index.
  ///
  /// If this function returns false, any further processing for ThinLTO tasks
  /// is aborted.
  ///
  /// It is called regardless of whether the backend is in-process, although it
  /// is not called from individual backend processes.
  using CombinedIndexHookFn = std::function<bool(
      const ModuleSummaryIndex &Index,
      const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols)>;
  CombinedIndexHookFn CombinedIndexHook;

  /// This hook is called after regular LTO modules with summaries (e.g., split
  /// LTO units) have been linked into the combined module, before optimizing
  /// the combined module or running ThinLTO.
  using PostRegularLTOLinkHookFn = std::function<void()>;
  PostRegularLTOLinkHookFn PostRegularLTOLinkHook;

  /// This is a convenience function that configures this Config object to write
  /// temporary files named after the given OutputFileName for each of the LTO
  /// phases to disk. A client can use this function to implement -save-temps.
  ///
  /// FIXME: Temporary files derived from ThinLTO backends are currently named
  /// after the input file name, rather than the output file name, when
  /// UseInputModulePath is set to true.
  ///
  /// Specifically, it (1) sets each of the above module hooks and the combined
  /// index hook to a function that calls the hook function (if any) that was
  /// present in the appropriate field when the addSaveTemps function was
  /// called, and writes the module to a bitcode file with a name prefixed by
  /// the given output file name, and (2) creates a resolution file whose name
  /// is prefixed by the given output file name and sets ResolutionFile to its
  /// file handle.
  ///
  /// SaveTempsArgs can be specified to select which temps to save.
  /// If SaveTempsArgs is not provided, all temps are saved.
  LLVM_ABI Error addSaveTemps(std::string OutputFileName,
                              bool UseInputModulePath = false,
                              const DenseSet<StringRef> &SaveTempsArgs = {});

  /// Called by WriteIndexesThinBackend when it needs to write a bitcode
  /// module's summary index. The callback should return a stream to write
  /// the index into. If not set, the backend falls back
  /// to writing the summary index to a file.
  std::function<std::unique_ptr<raw_pwrite_stream>(size_t Task)>
      GetSummaryIndexOutputStream;
  /// Called by WriteIndexesThinBackend when it needs to store a bitcode
  /// module's imports list. The callback should return a vector that the
  /// backend will populate with the imported module paths. If not set, the
  /// backend writes the imports list to a file instead.
  std::function<std::vector<std::string> &(size_t Task)>
      GetImportsListOutputArray;
  /// Called by WriteIndexesThinBackend when it needs to store a bitcode
  /// module's cache key. The callback should return a string that the backend
  /// will fill with the computed cache key. If not set, the cache key is
  /// discarded.
  std::function<std::string &(size_t Task)> GetCacheKeyOutputString;
};

struct LTOLLVMDiagnosticHandler : public DiagnosticHandler {
  DiagnosticHandlerFunction *Fn;
  LTOLLVMDiagnosticHandler(DiagnosticHandlerFunction *DiagHandlerFn)
      : Fn(DiagHandlerFn) {}
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    (*Fn)(DI);
    return true;
  }
};
/// A derived class of LLVMContext that initializes itself according to a given
/// Config object. The purpose of this class is to tie ownership of the
/// diagnostic handler to the context, as opposed to the Config object (which
/// may be ephemeral).
// FIXME: This should not be required as diagnostic handler is not callback.
struct LTOLLVMContext : LLVMContext {

  LTOLLVMContext(const Config &C) : DiagHandler(C.DiagHandler) {
    setDiscardValueNames(C.ShouldDiscardValueNames);
    enableDebugTypeODRUniquing();
    setDiagnosticHandler(
        std::make_unique<LTOLLVMDiagnosticHandler>(&DiagHandler), true);
    if (C.LibraryOptions)
      getLibraryOptions().assign(*C.LibraryOptions);
  }
  DiagnosticHandlerFunction DiagHandler;
};

}
}

#endif
