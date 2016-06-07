//===- IndexedStreamData.cpp - Standard PDB Stream Data ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/IndexedStreamData.h"
#include "llvm/DebugInfo/PDB/Raw/IPDBFile.h"

using namespace llvm;
using namespace llvm::pdb;

IndexedStreamData::IndexedStreamData(uint32_t StreamIdx, const IPDBFile &File)
    : StreamIdx(StreamIdx), File(File) {}

uint32_t IndexedStreamData::getLength() {
  return File.getStreamByteSize(StreamIdx);
}

ArrayRef<support::ulittle32_t> IndexedStreamData::getStreamBlocks() {
  return File.getStreamBlockList(StreamIdx);
}
