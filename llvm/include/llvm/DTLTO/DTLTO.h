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
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace lto {

// The primary purpose of this class is to enable distributed ThinLTO backend
// compilations to support archive members as inputs. For distributed
// compilation, each input must exist as an individual bitcode file on disk and
// be identified by its ModuleID.
//
// This requirement is not met for archive members, as an archive is a
// collection of files rather than a standalone file. To address this, the class
// ensures that an individual bitcode file exists for each input file (by
// writing it out if necessary) and updates the ModuleID to point to it.
//
// The class also ensures that bitcode input files are preserved until enough of
// the LTO pipeline has executed to determine the required per-module
// information, such as whether a module will participate in ThinLTO.
class DTLTO : public LTO {
  using Base = LTO;

public:
  // Inherit constructors.
  using Base::Base;
  ~DTLTO() override = default;

  // Add an input file and prepare it for distribution.
  LLVM_ABI Expected<std::shared_ptr<InputFile>>
  addInput(std::unique_ptr<InputFile> InputPtr) override;

protected:
  // Save the content of ThinLTO-enabled archive members to individual bitcode
  // files named after the module ID.
  LLVM_ABI llvm::Error handleArchiveInputs() override;

  // Remove temporary archive member files created to enable distribution.
  LLVM_ABI void cleanup() override;

private:
  // Bump allocator for a purpose of saving updated module IDs.
  BumpPtrAllocator PtrAlloc;
  StringSaver Saver{PtrAlloc};

  // Return true if the file at the given path is a thin archive.
  Expected<bool> isThinArchive(StringRef ArchivePath);

  // Array of input bitcode files for LTO.
  std::vector<std::shared_ptr<InputFile>> InputFiles;

  // Cache of whether an archive path refers to a thin archive.
  StringMap<bool> ArchiveIsThinCache;
};

} // namespace lto
} // namespace llvm

#endif // LLVM_DTLTO_DTLTO_H
