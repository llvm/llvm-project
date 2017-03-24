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
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace object;

static llvm::cl::opt<bool> IgnoreEmptyThinLTOIndexFile(
    "ignore-empty-index-file", llvm::cl::ZeroOrMore,
    llvm::cl::desc(
        "Ignore an empty index file and perform non-ThinLTO compilation"),
    llvm::cl::init(false));

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

// Parse module summary index in the given memory buffer.
// Return new ModuleSummaryIndexObjectFile instance containing parsed
// module summary/index.
Expected<std::unique_ptr<ModuleSummaryIndexObjectFile>>
ModuleSummaryIndexObjectFile::create(MemoryBufferRef Object) {
  ErrorOr<MemoryBufferRef> BCOrErr = findBitcodeInMemBuffer(Object);
  if (!BCOrErr)
    return errorCodeToError(BCOrErr.getError());

  Expected<std::unique_ptr<ModuleSummaryIndex>> IOrErr =
      getModuleSummaryIndex(BCOrErr.get());

  if (!IOrErr)
    return IOrErr.takeError();

  std::unique_ptr<ModuleSummaryIndex> Index = std::move(IOrErr.get());
  return llvm::make_unique<ModuleSummaryIndexObjectFile>(Object,
                                                         std::move(Index));
}

// Parse the module summary index out of an IR file and return the summary
// index object if found, or nullptr if not.
Expected<std::unique_ptr<ModuleSummaryIndex>>
llvm::getModuleSummaryIndexForFile(StringRef Path, StringRef Identifier) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(Path);
  std::error_code EC = FileOrErr.getError();
  if (EC)
    return errorCodeToError(EC);
  std::unique_ptr<MemoryBuffer> MemBuffer = std::move(FileOrErr.get());
  // If Identifier is non-empty, use it as the buffer identifier, which
  // will become the module path in the index.
  if (Identifier.empty())
    Identifier = MemBuffer->getBufferIdentifier();
  MemoryBufferRef BufferRef(MemBuffer->getBuffer(), Identifier);
  if (IgnoreEmptyThinLTOIndexFile && !BufferRef.getBufferSize())
    return nullptr;
  Expected<std::unique_ptr<object::ModuleSummaryIndexObjectFile>> ObjOrErr =
      object::ModuleSummaryIndexObjectFile::create(BufferRef);
  if (!ObjOrErr)
    return ObjOrErr.takeError();

  object::ModuleSummaryIndexObjectFile &Obj = **ObjOrErr;
  return Obj.takeIndex();
}
