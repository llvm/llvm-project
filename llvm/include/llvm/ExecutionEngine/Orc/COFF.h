//===-------------- COFF.h - COFF format utilities --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains utilities for load COFF relocatable object files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_COFF_H
#define LLVM_EXECUTIONENGINE_ORC_COFF_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <set>
#include <string>

namespace llvm {

namespace object {
class Archive;
} // namespace object

namespace orc {

class ObjectLinkingLayer;

class COFFImportFileScanner {
public:
  COFFImportFileScanner(std::set<std::string> &ImportedDynamicLibraries)
      : ImportedDynamicLibraries(ImportedDynamicLibraries) {}
  LLVM_ABI Expected<bool>
  operator()(object::Archive &A, MemoryBufferRef MemberBuf, size_t Index) const;

private:
  std::set<std::string> &ImportedDynamicLibraries;
};

/// A COFF-aware static-library definition generator.
///
/// Ordinary object members are handled by StaticLibraryDefinitionGenerator.
/// COFF short-import members are interpreted separately to synthesize their
/// IAT slots and call thunks and to report their referenced dynamic libraries.
/// This generator currently supports x86-64 COFF targets only.
class LLVM_ABI COFFStaticLibraryDefinitionGenerator
    : public DefinitionGenerator {
public:
  static Expected<std::unique_ptr<COFFStaticLibraryDefinitionGenerator>>
  Load(ObjectLinkingLayer &L, const char *FileName,
       std::set<std::string> &ImportedDynamicLibraries);

  static Expected<std::unique_ptr<COFFStaticLibraryDefinitionGenerator>>
  Create(ObjectLinkingLayer &L, std::unique_ptr<MemoryBuffer> ArchiveBuffer,
         std::unique_ptr<object::Archive> Archive,
         std::set<std::string> &ImportedDynamicLibraries);

  ~COFFStaticLibraryDefinitionGenerator() override;

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Symbols) override;

private:
  struct Impl;

  COFFStaticLibraryDefinitionGenerator(std::unique_ptr<Impl> P);

  std::unique_ptr<Impl> P;
};

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_COFF_H
