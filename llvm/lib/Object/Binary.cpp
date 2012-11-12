//===- Binary.cpp - A generic binary file -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Binary class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Binary.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

// Include headers for createBinary.
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace object;

Binary::~Binary() {
  delete Data;
}

Binary::Binary(unsigned int Type, MemoryBuffer *Source)
  : TypeID(Type)
  , Data(Source) {}

StringRef Binary::getData() const {
  return Data->getBuffer();
}

StringRef Binary::getFileName() const {
  return Data->getBufferIdentifier();
}

error_code object::createBinary(MemoryBuffer *Source,
                                OwningPtr<Binary> &Result) {
  OwningPtr<MemoryBuffer> scopedSource(Source);
  if (!Source)
    return make_error_code(errc::invalid_argument);
  if (Source->getBufferSize() < 64)
    return object_error::invalid_file_type;
  sys::LLVMFileType type = sys::IdentifyFileType(Source->getBufferStart(),
                                static_cast<unsigned>(Source->getBufferSize()));
  error_code ec;
  switch (type) {
    case sys::Archive_FileType: {
      OwningPtr<Binary> ret(new Archive(scopedSource.take(), ec));
      if (ec) return ec;
      Result.swap(ret);
      return object_error::success;
    }
    case sys::ELF_Relocatable_FileType:
    case sys::ELF_Executable_FileType:
    case sys::ELF_SharedObject_FileType:
    case sys::ELF_Core_FileType: {
      OwningPtr<Binary> ret(
        ObjectFile::createELFObjectFile(scopedSource.take()));
      if (!ret)
        return object_error::invalid_file_type;
      Result.swap(ret);
      return object_error::success;
    }
    case sys::Mach_O_Object_FileType:
    case sys::Mach_O_Executable_FileType:
    case sys::Mach_O_FixedVirtualMemorySharedLib_FileType:
    case sys::Mach_O_Core_FileType:
    case sys::Mach_O_PreloadExecutable_FileType:
    case sys::Mach_O_DynamicallyLinkedSharedLib_FileType:
    case sys::Mach_O_DynamicLinker_FileType:
    case sys::Mach_O_Bundle_FileType:
    case sys::Mach_O_DynamicallyLinkedSharedLibStub_FileType: {
      OwningPtr<Binary> ret(
        ObjectFile::createMachOObjectFile(scopedSource.take()));
      if (!ret)
        return object_error::invalid_file_type;
      Result.swap(ret);
      return object_error::success;
    }
    case sys::COFF_FileType: {
      OwningPtr<Binary> ret(new COFFObjectFile(scopedSource.take(), ec));
      if (ec) return ec;
      Result.swap(ret);
      return object_error::success;
    }
    default: // Unrecognized object file format.
      return object_error::invalid_file_type;
  }
}

error_code object::createBinary(StringRef Path, OwningPtr<Binary> &Result) {
  OwningPtr<MemoryBuffer> File;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(Path, File))
    return ec;
  return createBinary(File.take(), Result);
}
