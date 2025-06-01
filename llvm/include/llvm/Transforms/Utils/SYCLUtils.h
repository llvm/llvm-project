//===------------ SYCLUtils.h - SYCL utility functions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for SYCL.
//===----------------------------------------------------------------------===//
#ifndef LLVM_FRONTEND_SYCL_UTILS_H
#define LLVM_FRONTEND_SYCL_UTILS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <optional>
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

/// FunctionCategorizer used for splitting in SYCL compilation flow.
class FunctionCategorizer {
public:
  FunctionCategorizer(IRSplitMode SM);

  FunctionCategorizer() = delete;
  FunctionCategorizer(FunctionCategorizer &) = delete;
  FunctionCategorizer &operator=(const FunctionCategorizer &) = delete;
  FunctionCategorizer(FunctionCategorizer &&) = default;
  FunctionCategorizer &operator=(FunctionCategorizer &&) = default;

  /// Returns integer specifying the category for the entry point.
  /// If the given function isn't an entry point then returns std::nullopt.
  std::optional<int> operator()(const Function &F);

private:
  struct KeyInfo {
    static SmallString<0> getEmptyKey() { return SmallString<0>(""); }

    static SmallString<0> getTombstoneKey() { return SmallString<0>("-"); }

    static bool isEqual(const SmallString<0> &LHS, const SmallString<0> &RHS) {
      return LHS == RHS;
    }

    static unsigned getHashValue(const SmallString<0> &S) {
      return llvm::hash_value(StringRef(S));
    }
  };

  IRSplitMode SM;
  DenseMap<SmallString<0>, int, KeyInfo> StrKeyToID;
};

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

std::string makeSymbolTable(const Module &M);

using StringTable = SmallVector<SmallVector<SmallString<64>>>;

void writeStringTable(const StringTable &Table, raw_ostream &OS);

} // namespace sycl
} // namespace llvm

#endif // LLVM_FRONTEND_SYCL_UTILS_H
