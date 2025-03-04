//===-------- SYCLSplitModule.h - module split ------------------*- C++ -*-===//
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

#ifndef LLVM_TRANSFORMS_UTILS_SYCLSPLITMODULE_H
#define LLVM_TRANSFORMS_UTILS_SYCLSPLITMODULE_H

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
struct ModuleAndSYCLMetadata {
  std::string ModuleFilePath;
  std::string Symbols;

  ModuleAndSYCLMetadata() = default;
  ModuleAndSYCLMetadata(const ModuleAndSYCLMetadata &) = default;
  ModuleAndSYCLMetadata &operator=(const ModuleAndSYCLMetadata &) = default;
  ModuleAndSYCLMetadata(ModuleAndSYCLMetadata &&) = default;
  ModuleAndSYCLMetadata &operator=(ModuleAndSYCLMetadata &&) = default;

  ModuleAndSYCLMetadata(std::string_view File, std::string Symbols)
      : ModuleFilePath(File), Symbols(std::move(Symbols)) {}
};

struct ModuleSplitterSettings {
  IRSplitMode Mode;
  bool OutputAssembly = false; // Bitcode or LLVM IR.
  StringRef OutputPrefix;
};

/// Parses the string table.
Expected<SmallVector<ModuleAndSYCLMetadata, 0>>
parseModuleAndSYCLMetadataFromFile(StringRef File);

/// Splits the given module \p M according to the given \p Settings.
Expected<SmallVector<ModuleAndSYCLMetadata, 0>>
SYCLSplitModule(std::unique_ptr<Module> M, ModuleSplitterSettings Settings);

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SYCLSPLITMODULE_H
