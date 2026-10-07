//===- ObjectFactories.cpp - Create a Binary from a file of any format ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These factories reference every object file format. They live apart from
// the Binary, SymbolicFile and ObjectFile base classes so that a tool which
// only uses one format does not link all of them.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/DXContainer.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/Minidump.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Object/TapiUniversal.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Object/WindowsResource.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include <system_error>

using namespace llvm;
using namespace object;

Expected<std::unique_ptr<Binary>> object::createBinary(MemoryBufferRef Buffer,
                                                       LLVMContext *Context,
                                                       bool InitContent) {
  file_magic Type = identify_magic(Buffer.getBuffer());

  switch (Type) {
  case file_magic::archive:
    return Archive::create(Buffer);
  case file_magic::elf:
  case file_magic::elf_relocatable:
  case file_magic::elf_executable:
  case file_magic::elf_shared_object:
  case file_magic::elf_core:
  case file_magic::goff_object:
  case file_magic::macho_object:
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
  case file_magic::coff_object:
  case file_magic::coff_import_library:
  case file_magic::pecoff_executable:
  case file_magic::bitcode:
  case file_magic::xcoff_object_32:
  case file_magic::xcoff_object_64:
  case file_magic::wasm_object:
  case file_magic::dxcontainer_object:
    return ObjectFile::createSymbolicFile(Buffer, Type, Context, InitContent);
  case file_magic::macho_universal_binary:
    return MachOUniversalBinary::create(Buffer);
  case file_magic::windows_resource:
    return WindowsResource::createWindowsResource(Buffer);
  case file_magic::pdb:
    // PDB does not support the Binary interface.
    return errorCodeToError(object_error::invalid_file_type);
  case file_magic::unknown:
  case file_magic::clang_ast:
  case file_magic::cuda_fatbinary:
  case file_magic::coff_cl_gl_object:
  case file_magic::offload_bundle:
  case file_magic::offload_bundle_compressed:
  case file_magic::spirv_object:
    // Unrecognized object file format.
    return errorCodeToError(object_error::invalid_file_type);
  case file_magic::offload_binary: {
    auto OffloadBinaryOrErr = OffloadBinary::create(Buffer);
    if (!OffloadBinaryOrErr)
      return OffloadBinaryOrErr.takeError();
    return std::move((*OffloadBinaryOrErr)[0]);
  }
  case file_magic::minidump:
    return MinidumpFile::create(Buffer);
  case file_magic::tapi_file:
    return TapiUniversal::create(Buffer);
  }
  llvm_unreachable("Unexpected Binary File Type");
}

Expected<OwningBinary<Binary>>
object::createBinary(StringRef Path, LLVMContext *Context, bool InitContent) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(Path, /*IsText=*/false,
                                   /*RequiresNullTerminator=*/false);
  if (std::error_code EC = FileOrErr.getError())
    return errorCodeToError(EC);
  std::unique_ptr<MemoryBuffer> &Buffer = FileOrErr.get();

  Expected<std::unique_ptr<Binary>> BinOrErr =
      createBinary(Buffer->getMemBufferRef(), Context, InitContent);
  if (!BinOrErr)
    return BinOrErr.takeError();
  std::unique_ptr<Binary> &Bin = BinOrErr.get();

  return OwningBinary<Binary>(std::move(Bin), std::move(Buffer));
}

Expected<std::unique_ptr<ObjectFile>>
ObjectFile::createObjectFile(MemoryBufferRef Object, file_magic Type,
                             bool InitContent) {
  StringRef Data = Object.getBuffer();
  if (Type == file_magic::unknown)
    Type = identify_magic(Data);

  switch (Type) {
  case file_magic::unknown:
  case file_magic::bitcode:
  case file_magic::clang_ast:
  case file_magic::coff_cl_gl_object:
  case file_magic::archive:
  case file_magic::macho_universal_binary:
  case file_magic::windows_resource:
  case file_magic::pdb:
  case file_magic::minidump:
  case file_magic::cuda_fatbinary:
  case file_magic::offload_binary:
  case file_magic::offload_bundle:
  case file_magic::offload_bundle_compressed:
  case file_magic::spirv_object:
    return errorCodeToError(object_error::invalid_file_type);
  case file_magic::tapi_file:
    return errorCodeToError(object_error::invalid_file_type);
  case file_magic::elf:
  case file_magic::elf_relocatable:
  case file_magic::elf_executable:
  case file_magic::elf_shared_object:
  case file_magic::elf_core:
    return createELFObjectFile(Object, InitContent);
  case file_magic::macho_object:
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
    return createMachOObjectFile(Object);
  case file_magic::coff_object:
  case file_magic::coff_import_library:
  case file_magic::pecoff_executable:
    return createCOFFObjectFile(Object);
  case file_magic::xcoff_object_32:
    return createXCOFFObjectFile(Object, Binary::ID_XCOFF32);
  case file_magic::xcoff_object_64:
    return createXCOFFObjectFile(Object, Binary::ID_XCOFF64);
  case file_magic::wasm_object:
    return createWasmObjectFile(Object);
  case file_magic::dxcontainer_object:
    return createDXContainerObjectFile(Object);
  case file_magic::goff_object:
    return createGOFFObjectFile(Object);
  }
  llvm_unreachable("Unexpected Object File Type");
}

Expected<OwningBinary<ObjectFile>>
ObjectFile::createObjectFile(StringRef ObjectPath) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFile(ObjectPath);
  if (std::error_code EC = FileOrErr.getError())
    return errorCodeToError(EC);
  std::unique_ptr<MemoryBuffer> Buffer = std::move(FileOrErr.get());

  Expected<std::unique_ptr<ObjectFile>> ObjOrErr =
      createObjectFile(Buffer->getMemBufferRef());
  if (Error Err = ObjOrErr.takeError())
    return std::move(Err);
  std::unique_ptr<ObjectFile> Obj = std::move(ObjOrErr.get());

  return OwningBinary<ObjectFile>(std::move(Obj), std::move(Buffer));
}

Expected<std::unique_ptr<SymbolicFile>>
SymbolicFile::createSymbolicFile(MemoryBufferRef Object, file_magic Type,
                                 LLVMContext *Context, bool InitContent) {
  StringRef Data = Object.getBuffer();
  if (Type == file_magic::unknown)
    Type = identify_magic(Data);

  if (!isSymbolicFile(Type, Context))
    return errorCodeToError(object_error::invalid_file_type);

  switch (Type) {
  case file_magic::bitcode:
    // Context is guaranteed to be non-null here, because bitcode magic only
    // indicates a symbolic file when Context is non-null.
    return IRObjectFile::create(Object, *Context);
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
  case file_magic::dxcontainer_object:
    return ObjectFile::createObjectFile(Object, Type, InitContent);
  case file_magic::coff_import_library:
    return std::unique_ptr<SymbolicFile>(new COFFImportFile(Object));
  case file_magic::elf_relocatable:
  case file_magic::macho_object:
  case file_magic::coff_object: {
    Expected<std::unique_ptr<ObjectFile>> Obj =
        ObjectFile::createObjectFile(Object, Type, InitContent);
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
  default:
    llvm_unreachable("Unexpected Binary File Type");
  }
}
