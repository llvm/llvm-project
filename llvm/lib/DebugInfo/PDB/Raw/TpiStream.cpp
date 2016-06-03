//===- TpiStream.cpp - PDB Type Info (TPI) Stream 2 Access ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/TpiStream.h"

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/StreamReader.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support;
using namespace llvm::pdb;

namespace {
const uint32_t MinHashBuckets = 0x1000;
const uint32_t MaxHashBuckets = 0x40000;
}

static uint32_t HashBufferV8(uint8_t *buffer, uint32_t NumBuckets) {
  // Not yet implemented, this is probably some variation of CRC32 but we need
  // to be sure of the precise implementation otherwise we won't be able to work
  // with persisted hash values.
  return 0;
}

// This corresponds to `HDR` in PDB/dbi/tpi.h.
struct TpiStream::HeaderInfo {
  struct EmbeddedBuf {
    little32_t Off;
    ulittle32_t Length;
  };

  ulittle32_t Version;
  ulittle32_t HeaderSize;
  ulittle32_t TypeIndexBegin;
  ulittle32_t TypeIndexEnd;
  ulittle32_t TypeRecordBytes;

  // The following members correspond to `TpiHash` in PDB/dbi/tpi.h.
  ulittle16_t HashStreamIndex;
  ulittle16_t HashAuxStreamIndex;
  ulittle32_t HashKeySize;
  ulittle32_t NumHashBuckets;

  EmbeddedBuf HashValueBuffer;
  EmbeddedBuf IndexOffsetBuffer;
  EmbeddedBuf HashAdjBuffer;
};

TpiStream::TpiStream(PDBFile &File, uint32_t StreamIdx)
    : Pdb(File), Stream(StreamIdx, File), HashFunction(nullptr) {}

TpiStream::~TpiStream() {}

Error TpiStream::reload() {
  codeview::StreamReader Reader(Stream);

  if (Reader.bytesRemaining() < sizeof(HeaderInfo))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "TPI Stream does not contain a header.");

  if (Reader.readObject(Header))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "TPI Stream does not contain a header.");

  if (Header->Version != PdbTpiV80)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Unsupported TPI Version.");

  if (Header->HeaderSize != sizeof(HeaderInfo))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Corrupt TPI Header size.");

  if (Header->HashKeySize != sizeof(ulittle32_t))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "TPI Stream expected 4 byte hash key size.");

  if (Header->NumHashBuckets < MinHashBuckets ||
      Header->NumHashBuckets > MaxHashBuckets)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "TPI Stream Invalid number of hash buckets.");

  HashFunction = HashBufferV8;

  // The actual type records themselves come from this stream
  if (auto EC = Reader.readArray(TypeRecords, Header->TypeRecordBytes))
    return EC;

  // Hash indices, hash values, etc come from the hash stream.
  MappedBlockStream HS(Header->HashStreamIndex, Pdb);
  codeview::StreamReader HSR(HS);
  HSR.setOffset(Header->HashValueBuffer.Off);
  if (auto EC =
          HSR.readStreamRef(HashValuesBuffer, Header->HashValueBuffer.Length))
    return EC;

  HSR.setOffset(Header->HashAdjBuffer.Off);
  if (auto EC = HSR.readStreamRef(HashAdjBuffer, Header->HashAdjBuffer.Length))
    return EC;

  HSR.setOffset(Header->IndexOffsetBuffer.Off);
  if (auto EC = HSR.readStreamRef(TypeIndexOffsetBuffer,
                                  Header->IndexOffsetBuffer.Length))
    return EC;

  return Error::success();
}

PdbRaw_TpiVer TpiStream::getTpiVersion() const {
  uint32_t Value = Header->Version;
  return static_cast<PdbRaw_TpiVer>(Value);
}

uint32_t TpiStream::TypeIndexBegin() const { return Header->TypeIndexBegin; }

uint32_t TpiStream::TypeIndexEnd() const { return Header->TypeIndexEnd; }

uint32_t TpiStream::NumTypeRecords() const {
  return TypeIndexEnd() - TypeIndexBegin();
}

uint16_t TpiStream::getTypeHashStreamIndex() const {
  return Header->HashStreamIndex;
}

uint16_t TpiStream::getTypeHashStreamAuxIndex() const {
  return Header->HashAuxStreamIndex;
}

iterator_range<codeview::CVTypeArray::Iterator>
TpiStream::types(bool *HadError) const {
  return llvm::make_range(TypeRecords.begin(HadError), TypeRecords.end());
}
