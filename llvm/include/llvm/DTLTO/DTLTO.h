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
public:
  // Inherit contructors from LTO base class.
  using LTO::LTO;
  ~DTLTO() { removeTempFiles(); }

private:
  // Bump allocator for a purpose of saving updated module IDs.
  BumpPtrAllocator PtrAlloc;
  StringSaver Saver{PtrAlloc};

  // Removes temporary files.
  LLVM_ABI void removeTempFiles();

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

public:
  // Adds the input file to the LTO object's list of input files.
  // For archive members, generates a new module ID which is a path to a real
  // file on a filesystem.
  LLVM_ABI virtual Expected<std::shared_ptr<lto::InputFile>>
  addInput(std::unique_ptr<lto::InputFile> InputPtr) override;

  // Entry point for DTLTO archives support.
  LLVM_ABI virtual llvm::Error handleArchiveInputs() override;
};
} // namespace lto
} // namespace llvm

#endif // LLVM_DTLTO_H
