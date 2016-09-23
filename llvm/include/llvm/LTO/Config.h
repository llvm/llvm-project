//===-Config.h - LLVM Link Time Optimizer Configuration -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the lto::Config data structure, which allows clients to
// configure LTO.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_CONFIG_H
#define LLVM_LTO_CONFIG_H

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"

#include <functional>

namespace llvm {

class Error;
class Module;
class ModuleSummaryIndex;
class raw_pwrite_stream;

namespace lto {

/// LTO configuration. A linker can configure LTO by setting fields in this data
/// structure and passing it to the lto::LTO constructor.
struct Config {
  std::string CPU;
  std::string Features;
  TargetOptions Options;
  std::vector<std::string> MAttrs;
  Reloc::Model RelocModel = Reloc::PIC_;
  CodeModel::Model CodeModel = CodeModel::Default;
  CodeGenOpt::Level CGOptLevel = CodeGenOpt::Default;
  unsigned OptLevel = 2;
  bool DisableVerify = false;

  /// Disable entirely the optimizer, including importing for ThinLTO
  bool CodeGenOnly = false;

  /// If this field is set, the set of passes run in the middle-end optimizer
  /// will be the one specified by the string. Only works with the new pass
  /// manager as the old one doesn't have this ability.
  std::string OptPipeline;

  // If this field is set, it has the same effect of specifying an AA pipeline
  // identified by the string. Only works with the new pass manager, in
  // conjunction OptPipeline.
  std::string AAPipeline;

  /// Setting this field will replace target triples in input files with this
  /// triple.
  std::string OverrideTriple;

  /// Setting this field will replace unspecified target triples in input files
  /// with this triple.
  std::string DefaultTriple;

  bool ShouldDiscardValueNames = true;
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
  typedef std::function<bool(unsigned Task, const Module &)> ModuleHookFn;

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
  typedef std::function<bool(const ModuleSummaryIndex &Index)>
      CombinedIndexHookFn;
  CombinedIndexHookFn CombinedIndexHook;

  Config() {}
  // FIXME: Remove once MSVC can synthesize move ops.
  Config(Config &&X)
      : CPU(std::move(X.CPU)), Features(std::move(X.Features)),
        Options(std::move(X.Options)), MAttrs(std::move(X.MAttrs)),
        RelocModel(std::move(X.RelocModel)), CodeModel(std::move(X.CodeModel)),
        CGOptLevel(std::move(X.CGOptLevel)), OptLevel(std::move(X.OptLevel)),
        DisableVerify(std::move(X.DisableVerify)),
        OptPipeline(std::move(X.OptPipeline)),
        AAPipeline(std::move(X.AAPipeline)),
        OverrideTriple(std::move(X.OverrideTriple)),
        DefaultTriple(std::move(X.DefaultTriple)),
        ShouldDiscardValueNames(std::move(X.ShouldDiscardValueNames)),
        DiagHandler(std::move(X.DiagHandler)),
        ResolutionFile(std::move(X.ResolutionFile)),
        PreOptModuleHook(std::move(X.PreOptModuleHook)),
        PostPromoteModuleHook(std::move(X.PostPromoteModuleHook)),
        PostInternalizeModuleHook(std::move(X.PostInternalizeModuleHook)),
        PostImportModuleHook(std::move(X.PostImportModuleHook)),
        PostOptModuleHook(std::move(X.PostOptModuleHook)),
        PreCodeGenModuleHook(std::move(X.PreCodeGenModuleHook)),
        CombinedIndexHook(std::move(X.CombinedIndexHook)) {}
  // FIXME: Remove once MSVC can synthesize move ops.
  Config &operator=(Config &&X) {
    CPU = std::move(X.CPU);
    Features = std::move(X.Features);
    Options = std::move(X.Options);
    MAttrs = std::move(X.MAttrs);
    RelocModel = std::move(X.RelocModel);
    CodeModel = std::move(X.CodeModel);
    CGOptLevel = std::move(X.CGOptLevel);
    OptLevel = std::move(X.OptLevel);
    DisableVerify = std::move(X.DisableVerify);
    OptPipeline = std::move(X.OptPipeline);
    AAPipeline = std::move(X.AAPipeline);
    OverrideTriple = std::move(X.OverrideTriple);
    DefaultTriple = std::move(X.DefaultTriple);
    ShouldDiscardValueNames = std::move(X.ShouldDiscardValueNames);
    DiagHandler = std::move(X.DiagHandler);
    ResolutionFile = std::move(X.ResolutionFile);
    PreOptModuleHook = std::move(X.PreOptModuleHook);
    PostPromoteModuleHook = std::move(X.PostPromoteModuleHook);
    PostInternalizeModuleHook = std::move(X.PostInternalizeModuleHook);
    PostImportModuleHook = std::move(X.PostImportModuleHook);
    PostOptModuleHook = std::move(X.PostOptModuleHook);
    PreCodeGenModuleHook = std::move(X.PreCodeGenModuleHook);
    CombinedIndexHook = std::move(X.CombinedIndexHook);
    return *this;
  }

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
  Error addSaveTemps(std::string OutputFileName,
                     bool UseInputModulePath = false);
};

/// A derived class of LLVMContext that initializes itself according to a given
/// Config object. The purpose of this class is to tie ownership of the
/// diagnostic handler to the context, as opposed to the Config object (which
/// may be ephemeral).
struct LTOLLVMContext : LLVMContext {
  static void funcDiagHandler(const DiagnosticInfo &DI, void *Context) {
    auto *Fn = static_cast<DiagnosticHandlerFunction *>(Context);
    (*Fn)(DI);
  }

  LTOLLVMContext(const Config &C) : DiagHandler(C.DiagHandler) {
    setDiscardValueNames(C.ShouldDiscardValueNames);
    enableDebugTypeODRUniquing();
    setDiagnosticHandler(funcDiagHandler, &DiagHandler, true);
  }
  DiagnosticHandlerFunction DiagHandler;
};

}
}

#endif
