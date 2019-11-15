//===----- DebugUtils.h - Utilities for debugging ORC JITs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for debugging ORC-based JITs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_DEBUGUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_DEBUGUTILS_H

#include "llvm/Support/Error.h"
#include <memory>
#include <string>

namespace llvm {

class MemoryBuffer;

namespace orc {

/// A function object that can be used as an ObjectTransformLayer transform
/// to dump object files to disk at a specified path.
class DumpObjects {
public:
  /// Construct a DumpObjects transform that will dump objects to disk.
  ///
  /// @param DumpDir specifies the path to write dumped objects to. DumpDir may
  /// be empty, in which case files will be dumped to the working directory. If
  /// DumpDir is non-empty then any trailing separators will be discarded.
  ///
  /// @param IdentifierOverride specifies a file name stem to use when dumping
  /// objects. If empty, each MemoryBuffer's identifier will be used (with a .o
  /// suffix added if not already present). If an identifier override is
  /// supplied it will be used instead (since all buffers will use the same
  /// identifier, the resulting files will be named <ident>.o, <ident>.2.o,
  /// <ident>.3.o, and so on). IdentifierOverride should not contain an
  /// extension, as a .o suffix will be added by DumpObjects.
  DumpObjects(std::string DumpDir = "", std::string IdentifierOverride = "");

  /// Dumps the given buffer to disk.
  Expected<std::unique_ptr<MemoryBuffer>>
  operator()(std::unique_ptr<MemoryBuffer> Obj);

private:
  StringRef getBufferIdentifier(MemoryBuffer &B);
  std::string DumpDir;
  std::string IdentifierOverride;
};

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_DEBUGUTILS_H
