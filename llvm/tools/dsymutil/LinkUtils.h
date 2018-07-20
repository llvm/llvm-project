//===- tools/dsymutil/LinkUtils.h - Dwarf linker utilities ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_DSYMUTIL_LINKOPTIONS_H
#define LLVM_TOOLS_DSYMUTIL_LINKOPTIONS_H

#include "llvm/ADT/Twine.h"
#include "llvm/Support/WithColor.h"
#include <string>

namespace llvm {
namespace dsymutil {

enum class OutputFileType {
  Object,
  Assembly,
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

  /// -oso-prepend-path
  std::string PrependPath;

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
