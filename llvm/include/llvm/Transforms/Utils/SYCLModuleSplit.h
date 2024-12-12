//===-------- SYCLModuleSplit.h - module split ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Functionality to split a module into callgraphs. A callgraph here is a set
// of entry points with all functions reachable from them via a call. The result
// of the split is new modules containing corresponding callgraph.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_MODULE_SPLIT_H
#define LLVM_SYCL_MODULE_SPLIT_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>

namespace llvm {

class Module;

enum class IRSplitMode {
  IRSM_PER_TU,     // one module per translation unit
  IRSM_PER_KERNEL, // one module per kernel
  IRSM_NONE        // no splitting
};

/// \returns IRSplitMode value if \p S is recognized. Otherwise, std::nullopt is
/// returned.
std::optional<IRSplitMode> convertStringToSplitMode(StringRef S);

/// The structure represents a split LLVM Module accompanied by additional
/// information. Split Modules are being stored at disk due to the high RAM
/// consumption during the whole splitting process.
struct SYCLSplitModule {
  std::string ModuleFilePath;
  std::string Symbols;

  SYCLSplitModule() = default;
  SYCLSplitModule(const SYCLSplitModule &) = default;
  SYCLSplitModule &operator=(const SYCLSplitModule &) = default;
  SYCLSplitModule(SYCLSplitModule &&) = default;
  SYCLSplitModule &operator=(SYCLSplitModule &&) = default;

  SYCLSplitModule(std::string_view File, std::string Symbols)
      : ModuleFilePath(File), Symbols(std::move(Symbols)) {}
};

struct ModuleSplitterSettings {
  IRSplitMode Mode;
  bool OutputAssembly = false; // Bitcode or LLVM IR.
  StringRef OutputPrefix;
};

/// Parses the string table.
Expected<SmallVector<SYCLSplitModule, 0>>
parseSYCLSplitModulesFromFile(StringRef File);

/// Splits the given module \p M according to the given \p Settings.
Expected<SmallVector<SYCLSplitModule, 0>>
splitSYCLModule(std::unique_ptr<Module> M, ModuleSplitterSettings Settings);

} // namespace llvm

#endif // LLVM_SYCL_MODULE_SPLIT_H
