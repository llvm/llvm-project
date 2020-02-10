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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <memory>

using namespace llvm;
using namespace object;

SymbolicFile::SymbolicFile(unsigned int Type, MemoryBufferRef Source)
    : Binary(Type, Source) {}

SymbolicFile::~SymbolicFile() = default;

Expected<std::unique_ptr<SymbolicFile>>
SymbolicFile::createSymbolicFile(MemoryBufferRef Object, file_magic Type,
                                 LLVMContext *Context) {
  StringRef Data = Object.getBuffer();
  if (Type == file_magic::unknown)
    Type = identify_magic(Data);

  switch (Type) {
  case file_magic::bitcode:
    if (Context)
      return IRObjectFile::create(Object, *Context);
    LLVM_FALLTHROUGH;
  case file_magic::unknown:
  case file_magic::archive:
  case file_magic::coff_cl_gl_object:
  case file_magic::macho_universal_binary:
  case file_magic::windows_resource:
  case file_magic::pdb:
  case file_magic::minidump:
  case file_magic::tapi_file:
    return errorCodeToError(object_error::invalid_file_type);
  case file_magic::elf:
  case file_magic::elf_executable:
  case file_magic::elf_shared_object:
  case file_magic::elf_core:
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
  case file_magic::pecoff_executable:
  case file_magic::xcoff_object_32:
  case file_magic::xcoff_object_64:
  case file_magic::wasm_object:
    return ObjectFile::createObjectFile(Object, Type);
  case file_magic::coff_import_library:
    return std::unique_ptr<SymbolicFile>(new COFFImportFile(Object));
  case file_magic::elf_relocatable:
  case file_magic::macho_object:
  case file_magic::coff_object: {
    Expected<std::unique_ptr<ObjectFile>> Obj =
        ObjectFile::createObjectFile(Object, Type);
    if (!Obj || !Context)
      return std::move(Obj);

    Expected<MemoryBufferRef> BCData =
        IRObjectFile::findBitcodeInObject(*Obj->get());
    if (!BCData) {
      consumeError(BCData.takeError());
      return std::move(Obj);
    }

    return IRObjectFile::create(
        MemoryBufferRef(BCData->getBuffer(), Object.getBufferIdentifier()),
        *Context);
  }
  }
  llvm_unreachable("Unexpected Binary File Type");
}
