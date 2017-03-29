//===- NativeSession.cpp - Native implementation of IPDBSession -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeSession.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBSourceFile.h"
#include "llvm/DebugInfo/PDB/Native/NativeExeSymbol.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <memory>

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;

NativeSession::NativeSession(std::unique_ptr<PDBFile> PdbFile,
                             std::unique_ptr<BumpPtrAllocator> Allocator)
    : Pdb(std::move(PdbFile)), Allocator(std::move(Allocator)) {}

NativeSession::~NativeSession() = default;

Error NativeSession::createFromPdb(StringRef Path,
                                   std::unique_ptr<IPDBSession> &Session) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> ErrorOrBuffer =
      MemoryBuffer::getFileOrSTDIN(Path, /*FileSize=*/-1,
                                   /*RequiresNullTerminator=*/false);
  if (!ErrorOrBuffer)
    return make_error<GenericError>(generic_error_code::invalid_path);

  std::unique_ptr<MemoryBuffer> Buffer = std::move(*ErrorOrBuffer);
  auto Stream = llvm::make_unique<MemoryBufferByteStream>(
      std::move(Buffer), llvm::support::little);

  auto Allocator = llvm::make_unique<BumpPtrAllocator>();
  auto File = llvm::make_unique<PDBFile>(Path, std::move(Stream), *Allocator);
  if (auto EC = File->parseFileHeaders())
    return EC;
  if (auto EC = File->parseStreamData())
    return EC;

  Session =
      llvm::make_unique<NativeSession>(std::move(File), std::move(Allocator));

  return Error::success();
}

Error NativeSession::createFromExe(StringRef Path,
                                   std::unique_ptr<IPDBSession> &Session) {
  return make_error<RawError>(raw_error_code::feature_unsupported);
}

uint64_t NativeSession::getLoadAddress() const { return 0; }

void NativeSession::setLoadAddress(uint64_t Address) {}

std::unique_ptr<PDBSymbolExe> NativeSession::getGlobalScope() {
  auto RawSymbol = llvm::make_unique<NativeExeSymbol>(*this);
  auto PdbSymbol(PDBSymbol::create(*this, std::move(RawSymbol)));
  std::unique_ptr<PDBSymbolExe> ExeSymbol(
    static_cast<PDBSymbolExe *>(PdbSymbol.release()));
  return ExeSymbol;
}

std::unique_ptr<PDBSymbol>
NativeSession::getSymbolById(uint32_t SymbolId) const {
  return nullptr;
}

std::unique_ptr<PDBSymbol>
NativeSession::findSymbolByAddress(uint64_t Address, PDB_SymType Type) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumLineNumbers>
NativeSession::findLineNumbers(const PDBSymbolCompiland &Compiland,
                               const IPDBSourceFile &File) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumLineNumbers>
NativeSession::findLineNumbersByAddress(uint64_t Address,
                                        uint32_t Length) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumSourceFiles>
NativeSession::findSourceFiles(const PDBSymbolCompiland *Compiland,
                               StringRef Pattern,
                               PDB_NameSearchFlags Flags) const {
  return nullptr;
}

std::unique_ptr<IPDBSourceFile>
NativeSession::findOneSourceFile(const PDBSymbolCompiland *Compiland,
                                 StringRef Pattern,
                                 PDB_NameSearchFlags Flags) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumChildren<PDBSymbolCompiland>>
NativeSession::findCompilandsForSourceFile(StringRef Pattern,
                                           PDB_NameSearchFlags Flags) const {
  return nullptr;
}

std::unique_ptr<PDBSymbolCompiland>
NativeSession::findOneCompilandForSourceFile(StringRef Pattern,
                                             PDB_NameSearchFlags Flags) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumSourceFiles> NativeSession::getAllSourceFiles() const {
  return nullptr;
}

std::unique_ptr<IPDBEnumSourceFiles> NativeSession::getSourceFilesForCompiland(
    const PDBSymbolCompiland &Compiland) const {
  return nullptr;
}

std::unique_ptr<IPDBSourceFile>
NativeSession::getSourceFileById(uint32_t FileId) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumDataStreams> NativeSession::getDebugStreams() const {
  return nullptr;
}
