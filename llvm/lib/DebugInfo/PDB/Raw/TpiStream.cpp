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
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"

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

  ulittle16_t HashStreamIndex;
  ulittle16_t HashAuxStreamIndex;
  ulittle32_t HashKeySize;
  ulittle32_t NumHashBuckets;

  EmbeddedBuf HashValueBuffer;
  EmbeddedBuf IndexOffsetBuffer;
  EmbeddedBuf HashAdjBuffer;
};

TpiStream::TpiStream(PDBFile &File)
    : Pdb(File), Stream(StreamTPI, File), HashFunction(nullptr) {}

TpiStream::~TpiStream() {}

std::error_code TpiStream::reload() {
  StreamReader Reader(Stream);

  if (Reader.bytesRemaining() < sizeof(HeaderInfo))
    return std::make_error_code(std::errc::illegal_byte_sequence);

  Header.reset(new HeaderInfo());
  Reader.readObject(Header.get());

  if (Header->Version != PdbTpiV80)
    return std::make_error_code(std::errc::not_supported);

  if (Header->HeaderSize != sizeof(HeaderInfo))
    return std::make_error_code(std::errc::illegal_byte_sequence);

  if (Header->HashKeySize != sizeof(ulittle32_t))
    return std::make_error_code(std::errc::illegal_byte_sequence);

  if (Header->NumHashBuckets < MinHashBuckets ||
      Header->NumHashBuckets > MaxHashBuckets)
    return std::make_error_code(std::errc::illegal_byte_sequence);

  HashFunction = HashBufferV8;

  // The actual type records themselves come from this stream
  RecordsBuffer.initialize(Reader, Header->TypeRecordBytes);

  // Hash indices, hash values, etc come from the hash stream.
  MappedBlockStream HS(Header->HashStreamIndex, Pdb);
  StreamReader HSR(HS);
  HSR.setOffset(Header->HashValueBuffer.Off);
  HashValuesBuffer.initialize(HSR, Header->HashValueBuffer.Length);

  HSR.setOffset(Header->HashAdjBuffer.Off);
  HashAdjBuffer.initialize(HSR, Header->HashAdjBuffer.Length);

  HSR.setOffset(Header->IndexOffsetBuffer.Off);
  TypeIndexOffsetBuffer.initialize(HSR, Header->IndexOffsetBuffer.Length);

  return std::error_code();
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

iterator_range<codeview::TypeIterator> TpiStream::types() const {
  return codeview::makeTypeRange(RecordsBuffer.str());
}
