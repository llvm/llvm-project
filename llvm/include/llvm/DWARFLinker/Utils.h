//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKER_UTILS_H
#define LLVM_DWARFLINKER_UTILS_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace dwarf_linker {

/// This function calls \p Iteration() until it returns false.
/// If number of iterations exceeds \p MaxCounter then an Error is returned.
/// This function should be used for loops which assumed to have number of
/// iterations significantly smaller than \p MaxCounter to avoid infinite
/// looping in error cases.
inline Error finiteLoop(function_ref<Expected<bool>()> Iteration,
                        size_t MaxCounter = 100000) {
  size_t iterationsCounter = 0;
  while (iterationsCounter++ < MaxCounter) {
    Expected<bool> IterationResultOrError = Iteration();
    if (!IterationResultOrError)
      return IterationResultOrError.takeError();
    if (!IterationResultOrError.get())
      return Error::success();
  }
  return createStringError(std::errc::invalid_argument, "Infinite recursion");
}

/// Make a best effort to guess the
/// Xcode.app/Contents/Developer/Toolchains/ path from an SDK path.
inline SmallString<128> guessToolchainBaseDir(StringRef SysRoot) {
  SmallString<128> Result;
  // Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
  StringRef Base = sys::path::parent_path(SysRoot);
  if (sys::path::filename(Base) != "SDKs")
    return Result;
  Base = sys::path::parent_path(Base);
  Result = Base;
  Result += "/Toolchains";
  return Result;
}

inline bool isPathAbsoluteOnWindowsOrPosix(const Twine &Path) {
  // Debug info can contain paths from any OS, not necessarily
  // an OS we're currently running on. Moreover different compilation units can
  // be compiled on different operating systems and linked together later.
  return sys::path::is_absolute(Path, sys::path::Style::posix) ||
         sys::path::is_absolute(Path, sys::path::Style::windows);
}

} // end of namespace dwarf_linker
} // end of namespace llvm

#endif // LLVM_DWARFLINKER_UTILS_H
