//===- DTLTO.h - Distributed ThinLTO functions and classes ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_DTLTO_H
#define LLVM_DTLTO_H

#include "llvm/LTO/LTO.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace lto {

class DTLTO : public LTO {
  using Base = LTO;

public:
  LLVM_ABI DTLTO(Config Conf, ThinBackend Backend,
                 unsigned ParallelCodeGenParallelismLevel, LTOKind LTOMode,
                 StringRef LinkerOutputFile, bool SaveTemps)
      : Base(std::move(Conf), Backend, ParallelCodeGenParallelismLevel,
             LTOMode),
        LinkerOutputFile(LinkerOutputFile), SaveTemps(SaveTemps) {
    assert(!LinkerOutputFile.empty() && "expected a valid linker output file");
  }

  // Add an input file and prepare it for distribution.
  LLVM_ABI Expected<std::shared_ptr<InputFile>>
  addInput(std::unique_ptr<InputFile> InputPtr) override;

protected:
  LLVM_ABI llvm::Error handleArchiveInputs() override;

  LLVM_ABI void cleanup() override;

private:
  // Bump allocator for a purpose of saving updated module IDs.
  BumpPtrAllocator PtrAlloc;
  StringSaver Saver{PtrAlloc};

  /// The output file to which this LTO invocation will contribute.
  StringRef LinkerOutputFile;

  /// Controls preservation of any created temporary files.
  bool SaveTemps;

  // Determines if a file at the given path is a thin archive file.
  Expected<bool> isThinArchive(const StringRef ArchivePath);

  // Write the archive member content to a file named after the module ID.
  Error saveInputArchiveMember(lto::InputFile *Input);

  // Iterates through all input files and saves their content
  // to files if they are regular archive members.
  Error saveInputArchiveMembers();

  // Array of input bitcode files for LTO.
  std::vector<std::shared_ptr<lto::InputFile>> InputFiles;

  // A cache to avoid repeatedly reading the same archive file.
  StringMap<bool> ArchiveFiles;
};

} // namespace lto
} // namespace llvm

#endif // LLVM_DTLTO_H
