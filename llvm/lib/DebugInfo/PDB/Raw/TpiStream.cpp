//===- TpiStream.cpp - PDB Type Info (TPI) Stream 2 Access ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/TpiStream.h"

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/PDB/Raw/Hash.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/RawTypes.h"

#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::support;
using namespace llvm::msf;
using namespace llvm::pdb;

namespace {
const uint32_t MinHashBuckets = 0x1000;
const uint32_t MaxHashBuckets = 0x40000;
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
    : Pdb(File), Stream(std::move(Stream)) {}

TpiStream::~TpiStream() {}

// Corresponds to `fUDTAnon`.
template <typename T> static bool isAnonymous(T &Rec) {
  StringRef Name = Rec.getName();
  return Name == "<unnamed-tag>" || Name == "__unnamed" ||
      Name.endswith("::<unnamed-tag>") || Name.endswith("::__unnamed");
}

// Computes a hash for a given TPI record.
template <typename T>
static uint32_t getTpiHash(T &Rec, const CVRecord<TypeLeafKind> &RawRec) {
  auto Opts = static_cast<uint16_t>(Rec.getOptions());

  bool ForwardRef =
      Opts & static_cast<uint16_t>(ClassOptions::ForwardReference);
  bool Scoped = Opts & static_cast<uint16_t>(ClassOptions::Scoped);
  bool UniqueName = Opts & static_cast<uint16_t>(ClassOptions::HasUniqueName);
  bool IsAnon = UniqueName && isAnonymous(Rec);

  if (!ForwardRef && !Scoped && !IsAnon)
    return hashStringV1(Rec.getName());
  if (!ForwardRef && UniqueName && !IsAnon)
    return hashStringV1(Rec.getUniqueName());
  return hashBufferV8(RawRec.RawData);
}

namespace {
class TpiHashVerifier : public TypeVisitorCallbacks {
public:
  TpiHashVerifier(FixedStreamArray<support::ulittle32_t> &HashValues,
                  uint32_t NumHashBuckets)
      : HashValues(HashValues), NumHashBuckets(NumHashBuckets) {}

  Error visitKnownRecord(const CVRecord<TypeLeafKind> &CVR,
                         UdtSourceLineRecord &Rec) override {
    return verifySourceLine(Rec);
  }

  Error visitKnownRecord(const CVRecord<TypeLeafKind> &CVR,
                         UdtModSourceLineRecord &Rec) override {
    return verifySourceLine(Rec);
  }

  Error visitKnownRecord(const CVRecord<TypeLeafKind> &CVR,
                         ClassRecord &Rec) override {
    return verify(Rec);
  }
  Error visitKnownRecord(const CVRecord<TypeLeafKind> &CVR,
                         EnumRecord &Rec) override {
    return verify(Rec);
  }
  Error visitKnownRecord(const CVRecord<TypeLeafKind> &CVR,
                         UnionRecord &Rec) override {
    return verify(Rec);
  }

  Error visitTypeBegin(const CVRecord<TypeLeafKind> &Rec) override {
    ++Index;
    RawRecord = &Rec;
    return Error::success();
  }

private:
  template <typename T> Error verify(T &Rec) {
    uint32_t Hash = getTpiHash(Rec, *RawRecord);
    if (Hash % NumHashBuckets != HashValues[Index])
      return errorInvalidHash();
    return Error::success();
  }

  template <typename T> Error verifySourceLine(T &Rec) {
    char Buf[4];
    support::endian::write32le(Buf, Rec.getUDT().getIndex());
    uint32_t Hash = hashStringV1(StringRef(Buf, 4));
    if (Hash % NumHashBuckets != HashValues[Index])
      return errorInvalidHash();
    return Error::success();
  }

  Error errorInvalidHash() {
    return make_error<RawError>(
        raw_error_code::invalid_tpi_hash,
        "Type index is 0x" + utohexstr(TypeIndex::FirstNonSimpleIndex + Index));
  }

  FixedStreamArray<support::ulittle32_t> HashValues;
  const CVRecord<TypeLeafKind> *RawRecord;
  uint32_t NumHashBuckets;
  uint32_t Index = -1;
};
}

// Verifies that a given type record matches with a given hash value.
// Currently we only verify SRC_LINE records.
Error TpiStream::verifyHashValues() {
  TpiHashVerifier Verifier(HashValues, Header->NumHashBuckets);
  TypeDeserializer Deserializer(Verifier);
  CVTypeVisitor Visitor(Deserializer);
  return Visitor.visitTypeStream(TypeRecords);
}

Error TpiStream::reload() {
  StreamReader Reader(*Stream);

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

  // The actual type records themselves come from this stream
  if (auto EC = Reader.readArray(TypeRecords, Header->TypeRecordBytes))
    return EC;

  // Hash indices, hash values, etc come from the hash stream.
  if (Header->HashStreamIndex >= Pdb.getNumStreams())
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid TPI hash stream index.");
  auto HS = MappedBlockStream::createIndexedStream(
      Pdb.getMsfLayout(), Pdb.getMsfBuffer(), Header->HashStreamIndex);
  StreamReader HSR(*HS);

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

  HashStream = std::move(HS);

  // TPI hash table is a parallel array for the type records.
  // Verify that the hash values match with type records.
  if (auto EC = verifyHashValues())
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

uint32_t TpiStream::NumHashBuckets() const { return Header->NumHashBuckets; }
uint32_t TpiStream::getHashKeySize() const { return Header->HashKeySize; }

FixedStreamArray<support::ulittle32_t>
TpiStream::getHashValues() const {
  return HashValues;
}

FixedStreamArray<TypeIndexOffset>
TpiStream::getTypeIndexOffsets() const {
  return TypeIndexOffsets;
}

FixedStreamArray<TypeIndexOffset>
TpiStream::getHashAdjustments() const {
  return HashAdjustments;
}

iterator_range<CVTypeArray::Iterator>
TpiStream::types(bool *HadError) const {
  return llvm::make_range(TypeRecords.begin(HadError), TypeRecords.end());
}

Error TpiStream::commit() { return Error::success(); }
