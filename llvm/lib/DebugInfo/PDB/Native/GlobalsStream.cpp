//===- GlobalsStream.cpp - PDB Index of Symbols by Name ---- ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "GSI.h"
#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"
#include "llvm/Support/Error.h"
#include <algorithm>

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;

GlobalsStream::GlobalsStream(std::unique_ptr<MappedBlockStream> Stream)
    : Stream(std::move(Stream)) {}

GlobalsStream::~GlobalsStream() = default;

Error GlobalsStream::reload() {
  StreamReader Reader(*Stream);

  const GSIHashHeader *HashHdr;
  if (auto EC = readGSIHashHeader(HashHdr, Reader))
    return EC;

  if (auto EC = readGSIHashRecords(HashRecords, HashHdr, Reader))
    return EC;

  if (auto EC = readGSIHashBuckets(HashBuckets, HashHdr, Reader))
    return EC;
  NumBuckets = HashBuckets.size();

  return Error::success();
}

Error GlobalsStream::commit() { return Error::success(); }
