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
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/WithColor.h"

#include "llvm/DWARFLinker/DWARFLinker.h"
#include "llvm/DWARFLinker/DWARFStreamer.h"
#include <string>

namespace llvm {
namespace dsymutil {

struct LinkOptions {
  /// Verbosity
  bool Verbose = false;

  /// Statistics
  bool Statistics = false;

  /// Skip emitting output
  bool NoOutput = false;

  /// Do not unique types according to ODR
  bool NoODR = false;

  /// Update
  bool Update = false;

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

  /// The -object-prefix-map.
  std::map<std::string, std::string> ObjectPrefixMap;

  /// The Resources directory in the .dSYM bundle.
  Optional<std::string> ResourceDir;

  /// Symbol map translator.
  SymbolMapTranslator Translator;

  /// Virtual File System.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS =
      vfs::getRealFileSystem();

  /// Fields used for linking and placing remarks into the .dSYM bundle.
  /// @{

  /// Number of debug maps processed in total.
  unsigned NumDebugMaps = 0;

  /// -remarks-prepend-path: prepend a path to all the external remark file
  /// paths found in remark metadata.
  std::string RemarksPrependPath;

  /// The output format of the remarks.
  remarks::Format RemarksFormat = remarks::Format::Bitstream;

  /// @}

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
