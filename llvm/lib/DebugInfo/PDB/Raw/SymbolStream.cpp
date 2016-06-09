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
#include "llvm/DebugInfo/PDB/Raw/IndexedStreamData.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support;
using namespace llvm::pdb;

SymbolStream::SymbolStream(std::unique_ptr<MappedBlockStream> Stream)
    : Stream(std::move(Stream)) {}

SymbolStream::~SymbolStream() {}

Error SymbolStream::reload() {
  codeview::StreamReader Reader(*Stream);

  if (auto EC = Reader.readArray(SymbolRecords, Stream->getLength()))
    return EC;

  return Error::success();
}

iterator_range<codeview::CVSymbolArray::Iterator>
SymbolStream::getSymbols(bool *HadError) const {
  return llvm::make_range(SymbolRecords.begin(HadError), SymbolRecords.end());
}
