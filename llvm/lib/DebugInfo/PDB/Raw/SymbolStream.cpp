//===- SymbolStream.cpp - PDB Symbol Stream Access ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/SymbolStream.h"

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/StreamReader.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support;
using namespace llvm::pdb;

SymbolStream::SymbolStream(PDBFile &File, uint32_t StreamNum)
    : MappedStream(StreamNum, File) {}

SymbolStream::~SymbolStream() {}

Error SymbolStream::reload() {
  codeview::StreamReader Reader(MappedStream);

  if (Stream.initialize(Reader, MappedStream.getLength()))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Could not load symbol stream.");

  return Error::success();
}

iterator_range<codeview::SymbolIterator> SymbolStream::getSymbols() const {
  using codeview::SymbolIterator;
  ArrayRef<uint8_t> Data;
  if (auto Error = Stream.getArrayRef(0, Data, Stream.getLength())) {
    consumeError(std::move(Error));
    return iterator_range<SymbolIterator>(SymbolIterator(), SymbolIterator());
  }

  return codeview::makeSymbolRange(Data, nullptr);
}
