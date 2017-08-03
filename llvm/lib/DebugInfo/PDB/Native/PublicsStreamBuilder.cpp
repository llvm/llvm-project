//===- DbiStreamBuilder.cpp - PDB Dbi Stream Creation -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/PublicsStreamBuilder.h"

#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"

#include "GSI.h"

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;

PublicsStreamBuilder::PublicsStreamBuilder(msf::MSFBuilder &Msf) : Msf(Msf) {}

PublicsStreamBuilder::~PublicsStreamBuilder() {}

uint32_t PublicsStreamBuilder::calculateSerializedLength() const {
  uint32_t Size = 0;
  Size += sizeof(PublicsStreamHeader);
  Size += sizeof(GSIHashHeader);
  Size += HashRecords.size() * sizeof(PSHashRecord);
  size_t BitmapSizeInBits = alignTo(IPHR_HASH + 1, 32);
  uint32_t NumBitmapEntries = BitmapSizeInBits / 8;
  Size += NumBitmapEntries;

  // FIXME: Account for hash buckets.  For now since we we write a zero-bitmap
  // indicating that no hash buckets are valid, we also write zero byets of hash
  // bucket data.
  Size += 0;
  return Size;
}

Error PublicsStreamBuilder::finalizeMsfLayout() {
  Expected<uint32_t> Idx = Msf.addStream(calculateSerializedLength());
  if (!Idx)
    return Idx.takeError();
  StreamIdx = *Idx;

  Expected<uint32_t> RecordIdx = Msf.addStream(0);
  if (!RecordIdx)
    return RecordIdx.takeError();
  RecordStreamIdx = *RecordIdx;
  return Error::success();
}

Error PublicsStreamBuilder::commit(BinaryStreamWriter &PublicsWriter) {
  PublicsStreamHeader PSH;
  GSIHashHeader GSH;

  // FIXME: Figure out what to put for these values.
  PSH.AddrMap = 0;
  PSH.ISectThunkTable = 0;
  PSH.NumSections = 0;
  PSH.NumThunks = 0;
  PSH.OffThunkTable = 0;
  PSH.SizeOfThunk = 0;
  PSH.SymHash = 0;

  GSH.VerSignature = GSIHashHeader::HdrSignature;
  GSH.VerHdr = GSIHashHeader::HdrVersion;
  GSH.HrSize = 0;
  GSH.NumBuckets = 0;

  if (auto EC = PublicsWriter.writeObject(PSH))
    return EC;
  if (auto EC = PublicsWriter.writeObject(GSH))
    return EC;
  if (auto EC = PublicsWriter.writeArray(makeArrayRef(HashRecords)))
    return EC;

  size_t BitmapSizeInBits = alignTo(IPHR_HASH + 1, 32);
  uint32_t NumBitmapEntries = BitmapSizeInBits / 8;
  std::vector<uint8_t> BitmapData(NumBitmapEntries);
  // FIXME: Build an actual bitmap
  if (auto EC = PublicsWriter.writeBytes(makeArrayRef(BitmapData)))
    return EC;

  // FIXME: Write actual hash buckets.
  return Error::success();
}
