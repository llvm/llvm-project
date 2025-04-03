//===------------ Utils.h - SYCL utility functions ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for SYCL.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_UTILS_SYCLUTILS_H
#define LLVM_TRANSFORMS_UTILS_SYCLUTILS_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace llvm {

class Module;
class Function;
class raw_ostream;

namespace sycl {

enum class IRSplitMode {
  IRSM_PER_TU,     // one module per translation unit
  IRSM_PER_KERNEL, // one module per kernel
  IRSM_NONE        // no splitting
};

/// \returns IRSplitMode value if \p S is recognized. Otherwise, std::nullopt is
/// returned.
std::optional<IRSplitMode> convertStringToSplitMode(StringRef S);

/// The structure represents a LLVM Module accompanied by additional
/// information. Split Modules are being stored at disk due to the high RAM
/// consumption during the whole splitting process.
struct ModuleAndSYCLMetadata {
  std::string ModuleFilePath;
  std::string Symbols;

  ModuleAndSYCLMetadata() = delete;
  ModuleAndSYCLMetadata(const ModuleAndSYCLMetadata &) = default;
  ModuleAndSYCLMetadata &operator=(const ModuleAndSYCLMetadata &) = default;
  ModuleAndSYCLMetadata(ModuleAndSYCLMetadata &&) = default;
  ModuleAndSYCLMetadata &operator=(ModuleAndSYCLMetadata &&) = default;

  ModuleAndSYCLMetadata(const Twine &File, std::string Symbols)
      : ModuleFilePath(File.str()), Symbols(std::move(Symbols)) {}
};

/// Checks whether the function is a SYCL entry point.
bool isEntryPoint(const Function &F);

std::string makeSymbolTable(const Module &M);

using StringTable = SmallVector<SmallVector<SmallString<64>>>;

void writeStringTable(const StringTable &Table, raw_ostream &OS);

} // namespace sycl
} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SYCLUTILS_H
