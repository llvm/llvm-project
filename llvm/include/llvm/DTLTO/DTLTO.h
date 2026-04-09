//===- DTLTO.h - Distributed ThinLTO functions and classes ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_DTLTO_DTLTO_H
#define LLVM_DTLTO_DTLTO_H

#include "llvm/LTO/LTO.h"

namespace llvm {
namespace lto {

// The purpose of this class is to prepare inputs so that distributed ThinLTO
// backend compilations can succeed.
//
// For distributed compilation, each input must exist as an individual bitcode
// file on disk and be loadable via its ModuleID. This requirement is not met
// for archive members, as an archive is a collection of files rather than a
// standalone file. Similarly, for FatLTO objects, the bitcode is stored in a
// section of the containing ELF object file. To address this, the class
// updates the ModuleID of such bitcode inputs to a unique temporary path to
// which the extracted input bitcode can be written. On Windows it also
// normalizes the paths to avoid machine-local 8.3 short names that remote
// workers cannot reliably load.
//
// The bitcode is not immediately written. Instead, the class records the
// original input buffer and lets the ThinLTO backend write the file only if
// backend compilation is actually required. This avoids writing the file when
// the ThinLTO object cache already satisfies the backend job.
//
// The class ensures that lto::InputFile objects are preserved until enough of
// the LTO pipeline has executed to determine the required per-module
// information, such as whether a module will participate in ThinLTO and to
// allow the ThinLTO backend to write out the associated bitcode buffer.
class DTLTO : public LTO {
  using Base = LTO;

public:
  LLVM_ABI DTLTO(Config Conf, ThinBackend Backend,
                 unsigned ParallelCodeGenParallelismLevel, LTOKind LTOMode,
                 StringRef LinkerOutputFile)
      : Base(std::move(Conf), Backend, ParallelCodeGenParallelismLevel,
             LTOMode),
        LinkerOutputFile(LinkerOutputFile) {
    assert(!LinkerOutputFile.empty() && "expected a valid linker output file");
  }

  // Add an input file and prepare it for distribution.
  LLVM_ABI Expected<std::shared_ptr<InputFile>>
  addInput(std::unique_ptr<InputFile> InputPtr) override;

private:
  // Bump allocator for saving updated module IDs.
  BumpPtrAllocator PtrAlloc;
  StringSaver Saver{PtrAlloc};

  /// The output file to which this LTO invocation will contribute.
  StringRef LinkerOutputFile;

  /// The normalized output directory, derived from LinkerOutputFile.
  StringRef LinkerOutputDir;

  // Array of input bitcode files for LTO. We use shared_ptr here to
  // keep InputFile objects alive whilst the ThinLTO backend is invoked.
  std::vector<std::shared_ptr<lto::InputFile>> InputFiles;

  // Cache of whether a path refers to a thin archive.
  StringMap<bool> ArchiveIsThinCache;

  // Determines if the file at the given path is a thin archive.
  Expected<bool> isThinArchive(const StringRef ArchivePath);
};

} // namespace lto
} // namespace llvm

#endif // LLVM_DTLTO_DTLTO_H
