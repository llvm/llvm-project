//===- GlobalsStreamBuilder.cpp - PDB Globals Stream Creation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/GlobalsStreamBuilder.h"

#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;

GlobalsStreamBuilder::GlobalsStreamBuilder(msf::MSFBuilder &Msf) : Msf(Msf) {}

GlobalsStreamBuilder::~GlobalsStreamBuilder() {}

uint32_t GlobalsStreamBuilder::calculateSerializedLength() const {
  uint32_t Size = 0;
  // First is the header
  Size += sizeof(GSIHashHeader);

  // Next is the records.  For now we don't write any records, just an empty
  // stream.
  // FIXME: Write records and account for their size here.
  Size += 0;

  // Next is a bitmap indicating which hash buckets are valid.  The bitmap
  // is alway present, but we only write buckets for bitmap entries which are
  // non-zero, which now is noting.
  size_t BitmapSizeInBits = alignTo(IPHR_HASH + 1, 32);
  uint32_t NumBitmapEntries = BitmapSizeInBits / 8;
  Size += NumBitmapEntries;

  // FIXME: Account for hash buckets.  For now since we we write a zero-bitmap
  // indicating that no hash buckets are valid, we also write zero byets of hash
  // bucket data.
  Size += 0;
  return Size;
}

Error GlobalsStreamBuilder::finalizeMsfLayout() {
  Expected<uint32_t> Idx = Msf.addStream(calculateSerializedLength());
  if (!Idx)
    return Idx.takeError();
  StreamIdx = *Idx;
  return Error::success();
}

Error GlobalsStreamBuilder::commit(BinaryStreamWriter &PublicsWriter) {
  GSIHashHeader GSH;

  GSH.VerSignature = GSIHashHeader::HdrSignature;
  GSH.VerHdr = GSIHashHeader::HdrVersion;
  GSH.HrSize = 0;
  GSH.NumBuckets = 0;

  if (auto EC = PublicsWriter.writeObject(GSH))
    return EC;

  // FIXME: Once we start writing a value other than 0 for GSH.HrSize, we need
  // to write the hash records here.
  size_t BitmapSizeInBits = alignTo(IPHR_HASH + 1, 32);
  uint32_t NumBitmapEntries = BitmapSizeInBits / 8;
  std::vector<uint8_t> BitmapData(NumBitmapEntries);
  // FIXME: Build an actual bitmap
  if (auto EC = PublicsWriter.writeBytes(makeArrayRef(BitmapData)))
    return EC;

  // FIXME: Write actual hash buckets.
  return Error::success();
}
