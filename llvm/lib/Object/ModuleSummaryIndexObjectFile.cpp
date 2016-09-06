//===- ModuleSummaryIndexObjectFile.cpp - Summary index file implementation ==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Part of the ModuleSummaryIndexObjectFile class implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ModuleSummaryIndexObjectFile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace object;

ModuleSummaryIndexObjectFile::ModuleSummaryIndexObjectFile(
    MemoryBufferRef Object, std::unique_ptr<ModuleSummaryIndex> I)
    : SymbolicFile(Binary::ID_ModuleSummaryIndex, Object), Index(std::move(I)) {
}

ModuleSummaryIndexObjectFile::~ModuleSummaryIndexObjectFile() {}

std::unique_ptr<ModuleSummaryIndex> ModuleSummaryIndexObjectFile::takeIndex() {
  return std::move(Index);
}

ErrorOr<MemoryBufferRef>
ModuleSummaryIndexObjectFile::findBitcodeInObject(const ObjectFile &Obj) {
  for (const SectionRef &Sec : Obj.sections()) {
    if (Sec.isBitcode()) {
      StringRef SecContents;
      if (std::error_code EC = Sec.getContents(SecContents))
        return EC;
      return MemoryBufferRef(SecContents, Obj.getFileName());
    }
  }

  return object_error::bitcode_section_not_found;
}

ErrorOr<MemoryBufferRef>
ModuleSummaryIndexObjectFile::findBitcodeInMemBuffer(MemoryBufferRef Object) {
  sys::fs::file_magic Type = sys::fs::identify_magic(Object.getBuffer());
  switch (Type) {
  case sys::fs::file_magic::bitcode:
    return Object;
  case sys::fs::file_magic::elf_relocatable:
  case sys::fs::file_magic::macho_object:
  case sys::fs::file_magic::coff_object: {
    Expected<std::unique_ptr<ObjectFile>> ObjFile =
        ObjectFile::createObjectFile(Object, Type);
    if (!ObjFile)
      return errorToErrorCode(ObjFile.takeError());
    return findBitcodeInObject(*ObjFile->get());
  }
  default:
    return object_error::invalid_file_type;
  }
}

// Looks for module summary index in the given memory buffer.
// returns true if found, else false.
bool ModuleSummaryIndexObjectFile::hasGlobalValueSummaryInMemBuffer(
    MemoryBufferRef Object,
    const DiagnosticHandlerFunction &DiagnosticHandler) {
  ErrorOr<MemoryBufferRef> BCOrErr = findBitcodeInMemBuffer(Object);
  if (!BCOrErr)
    return false;

  return hasGlobalValueSummary(BCOrErr.get(), DiagnosticHandler);
}

// Parse module summary index in the given memory buffer.
// Return new ModuleSummaryIndexObjectFile instance containing parsed
// module summary/index.
ErrorOr<std::unique_ptr<ModuleSummaryIndexObjectFile>>
ModuleSummaryIndexObjectFile::create(
    MemoryBufferRef Object,
    const DiagnosticHandlerFunction &DiagnosticHandler) {
  std::unique_ptr<ModuleSummaryIndex> Index;

  ErrorOr<MemoryBufferRef> BCOrErr = findBitcodeInMemBuffer(Object);
  if (!BCOrErr)
    return BCOrErr.getError();

  ErrorOr<std::unique_ptr<ModuleSummaryIndex>> IOrErr =
      getModuleSummaryIndex(BCOrErr.get(), DiagnosticHandler);

  if (std::error_code EC = IOrErr.getError())
    return EC;

  Index = std::move(IOrErr.get());

  return llvm::make_unique<ModuleSummaryIndexObjectFile>(Object,
                                                         std::move(Index));
}

// Parse the module summary index out of an IR file and return the summary
// index object if found, or nullptr if not.
ErrorOr<std::unique_ptr<ModuleSummaryIndex>> llvm::getModuleSummaryIndexForFile(
    StringRef Path, const DiagnosticHandlerFunction &DiagnosticHandler) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(Path);
  std::error_code EC = FileOrErr.getError();
  if (EC)
    return EC;
  MemoryBufferRef BufferRef = (FileOrErr.get())->getMemBufferRef();
  ErrorOr<std::unique_ptr<object::ModuleSummaryIndexObjectFile>> ObjOrErr =
      object::ModuleSummaryIndexObjectFile::create(BufferRef,
                                                   DiagnosticHandler);
  EC = ObjOrErr.getError();
  if (EC)
    return EC;

  object::ModuleSummaryIndexObjectFile &Obj = **ObjOrErr;
  return Obj.takeIndex();
}
