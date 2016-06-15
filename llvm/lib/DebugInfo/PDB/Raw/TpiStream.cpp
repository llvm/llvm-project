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
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/Raw/Hash.h"
#include "llvm/DebugInfo/PDB/Raw/IndexedStreamData.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/RawTypes.h"

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

TpiStream::TpiStream(const PDBFile &File,
                     std::unique_ptr<MappedBlockStream> Stream)
    : Pdb(File), Stream(std::move(Stream)), HashFunction(nullptr) {}

TpiStream::~TpiStream() {}

// Verifies that a given type record matches with a given hash value.
// Currently we only verify SRC_LINE records.
static Error verifyTIHash(const codeview::CVType &Rec, uint32_t Expected,
                          uint32_t NumHashBuckets) {
  using namespace codeview;

  ArrayRef<uint8_t> D = Rec.Data;
  uint32_t Hash;

  switch (Rec.Type) {
  case LF_UDT_SRC_LINE:
  case LF_UDT_MOD_SRC_LINE:
    Hash = hashStringV1(StringRef((const char *)D.data(), 4));
    break;
  case LF_ENUM: {
    ErrorOr<EnumRecord> Enum = EnumRecord::deserialize(TypeRecordKind::Enum, D);
    if (Enum.getError())
      return make_error<RawError>(raw_error_code::corrupt_file,
                                  "Corrupt TPI hash table.");
    Hash = hashStringV1(Enum->getName());
    break;
  }
  default:
    return Error::success();
  }

  if ((Hash % NumHashBuckets) != Expected)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Corrupt TPI hash table.");
  return Error::success();
}

Error TpiStream::reload() {
  codeview::StreamReader Reader(*Stream);

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
  if (Header->HashStreamIndex >= Pdb.getNumStreams())
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid TPI hash stream index.");

  auto HS =
      MappedBlockStream::createIndexedStream(Header->HashStreamIndex, Pdb);
  if (!HS)
    return HS.takeError();
  codeview::StreamReader HSR(**HS);

  uint32_t NumHashValues = Header->HashValueBuffer.Length / sizeof(ulittle32_t);
  if (NumHashValues != NumTypeRecords())
    return make_error<RawError>(
        raw_error_code::corrupt_file,
        "TPI hash count does not match with the number of type records.");
  HSR.setOffset(Header->HashValueBuffer.Off);
  if (auto EC = HSR.readArray(HashValues, NumHashValues))
    return EC;

  HSR.setOffset(Header->IndexOffsetBuffer.Off);
  uint32_t NumTypeIndexOffsets =
      Header->IndexOffsetBuffer.Length / sizeof(TypeIndexOffset);
  if (auto EC = HSR.readArray(TypeIndexOffsets, NumTypeIndexOffsets))
    return EC;

  HSR.setOffset(Header->HashAdjBuffer.Off);
  uint32_t NumHashAdjustments =
      Header->HashAdjBuffer.Length / sizeof(TypeIndexOffset);
  if (auto EC = HSR.readArray(HashAdjustments, NumHashAdjustments))
    return EC;

  HashStream = std::move(*HS);

  // TPI hash table is a parallel array for the type records.
  // Verify that the hash values match with type records.
  size_t I = 0;
  bool HasError;
  for (const codeview::CVType &Rec : types(&HasError)) {
    if (auto EC = verifyTIHash(Rec, HashValues[I], Header->NumHashBuckets))
      return EC;
    ++I;
  }

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

uint32_t TpiStream::NumHashBuckets() const { return Header->NumHashBuckets; }
uint32_t TpiStream::getHashKeySize() const { return Header->HashKeySize; }

codeview::FixedStreamArray<support::ulittle32_t>
TpiStream::getHashValues() const {
  return HashValues;
}

codeview::FixedStreamArray<TypeIndexOffset>
TpiStream::getTypeIndexOffsets() const {
  return TypeIndexOffsets;
}

codeview::FixedStreamArray<TypeIndexOffset>
TpiStream::getHashAdjustments() const {
  return HashAdjustments;
}

iterator_range<codeview::CVTypeArray::Iterator>
TpiStream::types(bool *HadError) const {
  return llvm::make_range(TypeRecords.begin(HadError), TypeRecords.end());
}
