//===- ModuleDebugStream.cpp - PDB Module Info Stream Access --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cstdint>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

ModuleDebugStream::ModuleDebugStream(const DbiModuleDescriptor &Module,
                                     std::unique_ptr<MappedBlockStream> Stream)
    : Mod(Module), Stream(std::move(Stream)) {}

ModuleDebugStream::~ModuleDebugStream() = default;

Error ModuleDebugStream::reload() {
  BinaryStreamReader Reader(*Stream);

  uint32_t SymbolSize = Mod.getSymbolDebugInfoByteSize();
  uint32_t C11Size = Mod.getLineInfoByteSize();
  uint32_t C13Size = Mod.getC13LineInfoByteSize();

  if (C11Size > 0 && C13Size > 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Module has both C11 and C13 line info");

  BinaryStreamRef S;

  if (auto EC = Reader.readInteger(Signature))
    return EC;
  if (auto EC = Reader.readArray(SymbolsSubstream, SymbolSize - 4))
    return EC;

  if (auto EC = Reader.readStreamRef(LinesSubstream, C11Size))
    return EC;
  if (auto EC = Reader.readStreamRef(C13LinesSubstream, C13Size))
    return EC;

  BinaryStreamReader LineReader(C13LinesSubstream);
  if (auto EC =
          LineReader.readArray(LinesAndChecksums, LineReader.bytesRemaining()))
    return EC;

  uint32_t GlobalRefsSize;
  if (auto EC = Reader.readInteger(GlobalRefsSize))
    return EC;
  if (auto EC = Reader.readStreamRef(GlobalRefsSubstream, GlobalRefsSize))
    return EC;
  if (Reader.bytesRemaining() > 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Unexpected bytes in module stream.");

  return Error::success();
}

iterator_range<codeview::CVSymbolArray::Iterator>
ModuleDebugStream::symbols(bool *HadError) const {
  // It's OK if the stream is empty.
  if (SymbolsSubstream.getUnderlyingStream().getLength() == 0)
    return make_range(SymbolsSubstream.end(), SymbolsSubstream.end());
  return make_range(SymbolsSubstream.begin(HadError), SymbolsSubstream.end());
}

llvm::iterator_range<ModuleDebugStream::LinesAndChecksumsIterator>
ModuleDebugStream::linesAndChecksums() const {
  return make_range(LinesAndChecksums.begin(), LinesAndChecksums.end());
}

bool ModuleDebugStream::hasLineInfo() const {
  return C13LinesSubstream.getLength() > 0;
}

Error ModuleDebugStream::commit() { return Error::success(); }
