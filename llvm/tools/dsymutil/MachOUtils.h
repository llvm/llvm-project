//===-- MachOUtils.h - Mach-o specific helpers for dsymutil  --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_DSYMUTIL_MACHOUTILS_H
#define LLVM_TOOLS_DSYMUTIL_MACHOUTILS_H

#include "SymbolMap.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"

#include <string>

namespace llvm {
class MCStreamer;
class raw_fd_ostream;
namespace dsymutil {
class DebugMap;
struct LinkOptions;
namespace MachOUtils {

struct ArchAndFile {
  std::string Arch;
  // Optional because TempFile has no default constructor.
  Optional<llvm::sys::fs::TempFile> File;

  llvm::Error createTempFile();
  llvm::StringRef path() const;

  ArchAndFile(StringRef Arch) : Arch(Arch) {}
  ArchAndFile(ArchAndFile &&A) = default;
  ~ArchAndFile();
};

bool generateUniversalBinary(SmallVectorImpl<ArchAndFile> &ArchFiles,
                             StringRef OutputFileName, const LinkOptions &,
                             StringRef SDKPath);

bool generateDsymCompanion(const DebugMap &DM, SymbolMapTranslator &Translator,
                           MCStreamer &MS, raw_fd_ostream &OutFile);

std::string getArchName(StringRef Arch);
} // namespace MachOUtils
} // namespace dsymutil
} // namespace llvm
#endif // LLVM_TOOLS_DSYMUTIL_MACHOUTILS_H
