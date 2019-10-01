//===- tools/dsymutil/LinkUtils.h - Dwarf linker utilities ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_DSYMUTIL_LINKOPTIONS_H
#define LLVM_TOOLS_DSYMUTIL_LINKOPTIONS_H

#include "SymbolMap.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/WithColor.h"

#include <string>

namespace llvm {
namespace dsymutil {

enum class OutputFileType {
  Object,
  Assembly,
};

/// The kind of accelerator tables we should emit.
enum class AccelTableKind {
  Apple,   ///< .apple_names, .apple_namespaces, .apple_types, .apple_objc.
  Dwarf,   ///< DWARF v5 .debug_names.
  Default, ///< Dwarf for DWARF5 or later, Apple otherwise.
};

struct LinkOptions {
  /// Verbosity
  bool Verbose = false;

  /// Skip emitting output
  bool NoOutput = false;

  /// Do not unique types according to ODR
  bool NoODR = false;

  /// Update
  bool Update = false;

  /// Minimize
  bool Minimize = false;

  /// Do not check swiftmodule timestamp
  bool NoTimestamp = false;

  /// Number of threads.
  unsigned Threads = 1;

  // Output file type.
  OutputFileType FileType = OutputFileType::Object;

  /// The accelerator table kind
  AccelTableKind TheAccelTableKind;

  /// -oso-prepend-path
  std::string PrependPath;

  /// The Resources directory in the .dSYM bundle.
  Optional<std::string> ResourceDir;

  /// Symbol map translator.
  SymbolMapTranslator Translator;

  LinkOptions() = default;
};

inline void warn(Twine Warning, Twine Context = {}) {
  WithColor::warning() << Warning + "\n";
  if (!Context.isTriviallyEmpty())
    WithColor::note() << Twine("while processing ") + Context + "\n";
}

inline bool error(Twine Error, Twine Context = {}) {
  WithColor::error() << Error + "\n";
  if (!Context.isTriviallyEmpty())
    WithColor::note() << Twine("while processing ") + Context + "\n";
  return false;
}

} // end namespace dsymutil
} // end namespace llvm

#endif // LLVM_TOOLS_DSYMUTIL_LINKOPTIONS_H
