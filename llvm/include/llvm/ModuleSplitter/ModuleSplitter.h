//===- ModuleSplitter.h - Module Splitter Functions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MODULESPLITTER_MODULESPLITTER_H
#define LLVM_MODULESPLITTER_MODULESPLITTER_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
namespace llvm {

//===----------------------------------------------------------------------===//
// LLVMModuleAndContext
//===----------------------------------------------------------------------===//

/// A pair of an LLVM module and the LLVM context that holds ownership of the
/// objects. This is a useful class for parallelizing LLVM and managing
/// ownership of LLVM instances.
class LLVMModuleAndContext {
public:
  /// Expose the underlying LLVM context to create the module. This is the only
  /// way to access the LLVM context to prevent accidental sharing.
  Expected<bool> create(
      function_ref<Expected<std::unique_ptr<llvm::Module>>(llvm::LLVMContext &)>
          CreateModule);

  llvm::Module &operator*() { return *Module; }
  llvm::Module *operator->() { return Module.get(); }

  void reset();

private:
  /// LLVM context stored in a unique pointer so that we can move this type.
  std::unique_ptr<llvm::LLVMContext> Ctx =
      std::make_unique<llvm::LLVMContext>();
  /// The paired LLVM module.
  std::unique_ptr<llvm::Module> Module;
};

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

using LLVMSplitProcessFn =
    function_ref<void(llvm::unique_function<LLVMModuleAndContext()>,
                      std::optional<int64_t>, unsigned)>;

/// Helper to create a lambda that just forwards a preexisting Module.
inline llvm::unique_function<LLVMModuleAndContext()>
forwardModule(LLVMModuleAndContext &&Module) {
  return [Module = std::move(Module)]() mutable { return std::move(Module); };
}

/// Support for splitting an LLVM module into multiple parts using anchored
/// functions (e.g. exported functions), and pull in all dependency on the
// call stack into one module.
void splitPerAnchored(LLVMModuleAndContext Module, LLVMSplitProcessFn ProcessFn,
                      llvm::SmallVectorImpl<llvm::Function> &Anchors);

/// Support for splitting an LLVM module into multiple parts with each part
/// contains only one function.
void splitPerFunction(
    LLVMModuleAndContext Module, LLVMSplitProcessFn ProcessFn,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &SymbolLinkageTypes,
    unsigned NumFunctionBase);

} // namespace llvm

#endif
