//===- RawSession.cpp - Raw implementation of IPDBSession -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/RawSession.h"

#include "llvm/DebugInfo/MSF/ByteStream.h"
#include "llvm/DebugInfo/MSF/StreamInterface.h"
#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBSourceFile.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;

RawSession::RawSession(std::unique_ptr<PDBFile> PdbFile,
                       std::unique_ptr<BumpPtrAllocator> Allocator)
    : Pdb(std::move(PdbFile)), Allocator(std::move(Allocator)) {}

RawSession::~RawSession() {}

Error RawSession::createFromPdb(StringRef Path,
                                std::unique_ptr<IPDBSession> &Session) {

  ErrorOr<std::unique_ptr<MemoryBuffer>> ErrorOrBuffer =
      MemoryBuffer::getFileOrSTDIN(Path, /*FileSize=*/-1,
                                   /*RequiresNullTerminator=*/false);
  if (!ErrorOrBuffer)
    return llvm::make_error<GenericError>(generic_error_code::invalid_path);

  std::unique_ptr<MemoryBuffer> Buffer = std::move(*ErrorOrBuffer);
  auto Stream = llvm::make_unique<MemoryBufferByteStream>(std::move(Buffer));

  auto Allocator = llvm::make_unique<BumpPtrAllocator>();
  auto File = llvm::make_unique<PDBFile>(std::move(Stream), *Allocator);
  if (auto EC = File->parseFileHeaders())
    return EC;
  if (auto EC = File->parseStreamData())
    return EC;

  Session =
      llvm::make_unique<RawSession>(std::move(File), std::move(Allocator));

  return Error::success();
}

Error RawSession::createFromExe(StringRef Path,
                                std::unique_ptr<IPDBSession> &Session) {
  return llvm::make_error<RawError>(raw_error_code::feature_unsupported);
}

uint64_t RawSession::getLoadAddress() const { return 0; }

void RawSession::setLoadAddress(uint64_t Address) {}

std::unique_ptr<PDBSymbolExe> RawSession::getGlobalScope() const {
  return nullptr;
}

std::unique_ptr<PDBSymbol> RawSession::getSymbolById(uint32_t SymbolId) const {
  return nullptr;
}

std::unique_ptr<PDBSymbol>
RawSession::findSymbolByAddress(uint64_t Address, PDB_SymType Type) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumLineNumbers>
RawSession::findLineNumbers(const PDBSymbolCompiland &Compiland,
                            const IPDBSourceFile &File) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumLineNumbers>
RawSession::findLineNumbersByAddress(uint64_t Address, uint32_t Length) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumSourceFiles>
RawSession::findSourceFiles(const PDBSymbolCompiland *Compiland,
                            llvm::StringRef Pattern,
                            PDB_NameSearchFlags Flags) const {
  return nullptr;
}

std::unique_ptr<IPDBSourceFile>
RawSession::findOneSourceFile(const PDBSymbolCompiland *Compiland,
                              llvm::StringRef Pattern,
                              PDB_NameSearchFlags Flags) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumChildren<PDBSymbolCompiland>>
RawSession::findCompilandsForSourceFile(llvm::StringRef Pattern,
                                        PDB_NameSearchFlags Flags) const {
  return nullptr;
}

std::unique_ptr<PDBSymbolCompiland>
RawSession::findOneCompilandForSourceFile(llvm::StringRef Pattern,
                                          PDB_NameSearchFlags Flags) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumSourceFiles> RawSession::getAllSourceFiles() const {
  return nullptr;
}

std::unique_ptr<IPDBEnumSourceFiles> RawSession::getSourceFilesForCompiland(
    const PDBSymbolCompiland &Compiland) const {
  return nullptr;
}

std::unique_ptr<IPDBSourceFile>
RawSession::getSourceFileById(uint32_t FileId) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumDataStreams> RawSession::getDebugStreams() const {
  return nullptr;
}
