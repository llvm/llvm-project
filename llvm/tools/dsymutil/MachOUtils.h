//===-- MachOUtils.h - Mach-o specific helpers for dsymutil  --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
