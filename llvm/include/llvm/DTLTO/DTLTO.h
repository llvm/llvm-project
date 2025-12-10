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

  BumpPtrAllocator PtrAlloc;
  StringSaver Saver{PtrAlloc};

  // Remove temporary files.
  LLVM_ABI void removeTempFiles();

  // Array of input bitcode files for LTO.
  std::vector<std::shared_ptr<lto::InputFile>> InputFiles;

  LLVM_ABI virtual Expected<std::shared_ptr<lto::InputFile>>
  addInput(std::unique_ptr<lto::InputFile> InputPtr) override;

  LLVM_ABI virtual llvm::Error handleArchiveInputs() override;

  StringMap<bool> ArchiveFiles;
};
} // namespace lto
} // namespace llvm

#endif // LLVM_DTLTO_H
