//===- SymbolicFile.cpp - Interface that only provides symbols ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a file format independent SymbolicFile class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SymbolicFile.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

using namespace llvm;
using namespace object;

namespace llvm {
class LLVMContext;
}

SymbolicFile::SymbolicFile(unsigned int Type, MemoryBufferRef Source)
    : Binary(Type, Source) {}

SymbolicFile::~SymbolicFile() = default;

bool SymbolicFile::isSymbolicFile(file_magic Type, const LLVMContext *Context) {
  switch (Type) {
  case file_magic::bitcode:
    return Context != nullptr;
  case file_magic::elf:
  case file_magic::elf_executable:
  case file_magic::elf_shared_object:
  case file_magic::elf_core:
  case file_magic::goff_object:
  case file_magic::macho_executable:
  case file_magic::macho_fixed_virtual_memory_shared_lib:
  case file_magic::macho_core:
  case file_magic::macho_preload_executable:
  case file_magic::macho_dynamically_linked_shared_lib:
  case file_magic::macho_dynamic_linker:
  case file_magic::macho_bundle:
  case file_magic::macho_dynamically_linked_shared_lib_stub:
  case file_magic::macho_dsym_companion:
  case file_magic::macho_kext_bundle:
  case file_magic::macho_file_set:
  case file_magic::pecoff_executable:
  case file_magic::xcoff_object_32:
  case file_magic::xcoff_object_64:
  case file_magic::wasm_object:
  case file_magic::coff_import_library:
  case file_magic::elf_relocatable:
  case file_magic::macho_object:
  case file_magic::coff_object:
  case file_magic::dxcontainer_object:
    return true;
  default:
    return false;
  }
}
