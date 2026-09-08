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
  using ModuleHookFn = std::function<bool(unsigned Task, const Module &)>;
  using CombinedIndexHookFn = std::function<bool(
      const ModuleSummaryIndex &Index,
      const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols)>;

#define LTO_CONFIG_OPTION(Type, Name, Default, Kind) Type Name = Default;
#define LTO_CONFIG_MUTABLE_OPTION(Type, Name, Default, Kind)                   \
  mutable Type Name = Default;
#include "llvm/LTO/Config.def"

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
  }
  DiagnosticHandlerFunction DiagHandler;
};

} // namespace lto
} // namespace llvm

#endif
