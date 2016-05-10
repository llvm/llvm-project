//===- ModStream.cpp - PDB Module Info Stream Access ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/ModStream.h"
#include "llvm/DebugInfo/PDB/Raw/ModInfo.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"

using namespace llvm;
using namespace llvm::pdb;

ModStream::ModStream(PDBFile &File, const ModInfo &Module)
    : Mod(Module), Stream(Module.getModuleStreamIndex(), File) {}

ModStream::~ModStream() {}

Error ModStream::reload() {
  StreamReader Reader(Stream);

  uint32_t SymbolSize = Mod.getSymbolDebugInfoByteSize();
  uint32_t C11Size = Mod.getLineInfoByteSize();
  uint32_t C13Size = Mod.getC13LineInfoByteSize();

  if (C11Size > 0 && C13Size > 0)
    return llvm::make_error<RawError>(raw_error_code::corrupt_file,
                                      "Module has both C11 and C13 line info");

  if (auto EC = SymbolsSubstream.initialize(Reader, SymbolSize))
    return EC;
  if (auto EC = LinesSubstream.initialize(Reader, C11Size))
    return EC;
  if (auto EC = C13LinesSubstream.initialize(Reader, C13Size))
    return EC;

  uint32_t GlobalRefsSize;
  if (auto EC = Reader.readInteger(GlobalRefsSize))
    return EC;
  if (auto EC = GlobalRefsSubstream.initialize(Reader, GlobalRefsSize))
    return EC;
  if (Reader.bytesRemaining() > 0)
    return llvm::make_error<RawError>(raw_error_code::corrupt_file,
                                      "Unexpected bytes in module stream.");

  return Error::success();
}

iterator_range<codeview::SymbolIterator> ModStream::symbols() const {
  return codeview::makeSymbolRange(SymbolsSubstream.data().slice(4));
}
