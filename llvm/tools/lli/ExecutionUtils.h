//===- ExecutionUtils.h - Utilities for executing code in lli ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains utilities for executing code in lli.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLI_EXECUTIONUTILS_H
#define LLVM_TOOLS_LLI_EXECUTIONUTILS_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ToolOutputFile.h"

#include <memory>
#include <utility>

namespace llvm {

enum class BuiltinFunctionKind {
  DumpDebugDescriptor,
  DumpDebugObjects,
};

// Utility class to expose symbols for special-purpose functions to the JIT.
class LLIBuiltinFunctionGenerator : public orc::DefinitionGenerator {
public:
  LLIBuiltinFunctionGenerator(std::vector<BuiltinFunctionKind> Enabled,
                              orc::MangleAndInterner &Mangle);

  Error tryToGenerate(orc::LookupState &LS, orc::LookupKind K,
                      orc::JITDylib &JD, orc::JITDylibLookupFlags JDLookupFlags,
                      const orc::SymbolLookupSet &Symbols) override;

  void appendDebugObject(const char *Addr, size_t Size) {
    TestOut->os().write(Addr, Size);
  }

private:
  orc::SymbolMap BuiltinFunctions;
  std::unique_ptr<ToolOutputFile> TestOut;

  template <typename T> void expose(orc::SymbolStringPtr Name, T *Handler) {
    BuiltinFunctions[Name] = {orc::ExecutorAddr::fromPtr(Handler),
                              JITSymbolFlags::Exported};
  }

  static std::unique_ptr<ToolOutputFile> createToolOutput();
};

} // end namespace llvm

#endif // LLVM_TOOLS_LLI_EXECUTIONUTILS_H
