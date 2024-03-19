//=- tools/dsymutil/SymbolMap.h -----------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_DSYMUTIL_SYMBOLMAP_H
#define LLVM_TOOLS_DSYMUTIL_SYMBOLMAP_H

#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace llvm {
namespace dsymutil {
class DebugMap;

/// Callable class to unobfuscate strings based on a BCSymbolMap.
class SymbolMapTranslator {
public:
  SymbolMapTranslator() : MangleNames(false) {}

  SymbolMapTranslator(std::vector<std::string> UnobfuscatedStrings,
                      bool MangleNames)
      : UnobfuscatedStrings(std::move(UnobfuscatedStrings)),
        MangleNames(MangleNames) {}

  StringRef operator()(StringRef Input);

  operator bool() const { return !UnobfuscatedStrings.empty(); }

private:
  std::vector<std::string> UnobfuscatedStrings;
  bool MangleNames;
};

/// Class to initialize SymbolMapTranslators from a BCSymbolMap.
class SymbolMapLoader {
public:
  SymbolMapLoader(std::string SymbolMap) : SymbolMap(std::move(SymbolMap)) {}

  SymbolMapTranslator Load(StringRef InputFile, const DebugMap &Map) const;

private:
  const std::string SymbolMap;
};
} // namespace dsymutil
} // namespace llvm

#endif // LLVM_TOOLS_DSYMUTIL_SYMBOLMAP_H
