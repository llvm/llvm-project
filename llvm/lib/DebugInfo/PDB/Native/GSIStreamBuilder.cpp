//===- DbiStreamBuilder.cpp - PDB Dbi Stream Creation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The data structures defined in this file are based on the reference
// implementation which is available at
// https://github.com/Microsoft/microsoft-pdb/blob/master/PDB/dbi/gsi.cpp
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/GSIStreamBuilder.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/DebugInfo/CodeView/RecordName.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolSerializer.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/DebugInfo/PDB/Native/Hash.h"
#include "llvm/Support/BinaryItemStream.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/xxhash.h"
#include <algorithm>
#include <vector>

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;
using namespace llvm::codeview;

struct llvm::pdb::GSIHashStreamBuilder {
  struct SymbolDenseMapInfo {
    static inline CVSymbol getEmptyKey() {
      static CVSymbol Empty;
      return Empty;
    }
    static inline CVSymbol getTombstoneKey() {
      static CVSymbol Tombstone(
          DenseMapInfo<ArrayRef<uint8_t>>::getTombstoneKey());
      return Tombstone;
    }
    static unsigned getHashValue(const CVSymbol &Val) {
      return xxHash64(Val.RecordData);
    }
    static bool isEqual(const CVSymbol &LHS, const CVSymbol &RHS) {
      return LHS.RecordData == RHS.RecordData;
    }
  };

  std::vector<CVSymbol> Records;
  llvm::DenseSet<CVSymbol, SymbolDenseMapInfo> SymbolHashes;
  std::vector<PSHashRecord> HashRecords;
  std::array<support::ulittle32_t, (IPHR_HASH + 32) / 32> HashBitmap;
  std::vector<support::ulittle32_t> HashBuckets;

  uint32_t calculateSerializedLength() const;
  uint32_t calculateRecordByteSize() const;
  Error commit(BinaryStreamWriter &Writer);

  void finalizeBuckets(uint32_t RecordZeroOffset);

  // Finalize public symbol buckets.
  void finalizeBuckets(uint32_t RecordZeroOffset,
                       std::vector<BulkPublic> &&Publics);

  template <typename T> void addSymbol(const T &Symbol, MSFBuilder &Msf) {
    T Copy(Symbol);
    addSymbol(SymbolSerializer::writeOneSymbol(Copy, Msf.getAllocator(),
                                               CodeViewContainer::Pdb));
  }
  void addSymbol(const CVSymbol &Symbol);
};

void GSIHashStreamBuilder::addSymbol(const codeview::CVSymbol &Symbol) {
  // Ignore duplicate typedefs and constants.
  if (Symbol.kind() == S_UDT || Symbol.kind() == S_CONSTANT) {
    auto Iter = SymbolHashes.insert(Symbol);
    if (!Iter.second)
      return;
  }
  Records.push_back(Symbol);
}

namespace {
LLVM_PACKED_START
struct PublicSym32Layout {
  RecordPrefix Prefix;
  PublicSym32Header Pub;
  // char Name[];
};
LLVM_PACKED_END
} // namespace

// Calculate how much memory this public needs when serialized.
static uint32_t sizeOfPublic(const BulkPublic &Pub) {
  uint32_t NameLen = Pub.NameLen;
  NameLen = std::min(NameLen,
                     uint32_t(MaxRecordLength - sizeof(PublicSym32Layout) - 1));
  return alignTo(sizeof(PublicSym32Layout) + NameLen + 1, 4);
}

static CVSymbol serializePublic(uint8_t *Mem, const BulkPublic &Pub) {
  // Assume the caller has allocated sizeOfPublic bytes.
  uint32_t NameLen = std::min(
      Pub.NameLen, uint32_t(MaxRecordLength - sizeof(PublicSym32Layout) - 1));
  size_t Size = alignTo(sizeof(PublicSym32Layout) + NameLen + 1, 4);
  assert(Size == sizeOfPublic(Pub));
  auto *FixedMem = reinterpret_cast<PublicSym32Layout *>(Mem);
  FixedMem->Prefix.RecordKind = static_cast<uint16_t>(codeview::S_PUB32);
  FixedMem->Prefix.RecordLen = static_cast<uint16_t>(Size - 2);
  FixedMem->Pub.Flags = Pub.Flags;
  FixedMem->Pub.Offset = Pub.Offset;
  FixedMem->Pub.Segment = Pub.U.Segment;
  char *NameMem = reinterpret_cast<char *>(FixedMem + 1);
  memcpy(NameMem, Pub.Name, NameLen);
  // Zero the null terminator and remaining bytes.
  memset(&NameMem[NameLen], 0, Size - sizeof(PublicSym32Layout) - NameLen);
  return CVSymbol(makeArrayRef(reinterpret_cast<uint8_t *>(Mem), Size));
}

uint32_t GSIHashStreamBuilder::calculateSerializedLength() const {
  uint32_t Size = sizeof(GSIHashHeader);
  Size += HashRecords.size() * sizeof(PSHashRecord);
  Size += HashBitmap.size() * sizeof(uint32_t);
  Size += HashBuckets.size() * sizeof(uint32_t);
  return Size;
}

uint32_t GSIHashStreamBuilder::calculateRecordByteSize() const {
  uint32_t Size = 0;
  for (const auto &Sym : Records)
    Size += Sym.length();
  return Size;
}

Error GSIHashStreamBuilder::commit(BinaryStreamWriter &Writer) {
  GSIHashHeader Header;
  Header.VerSignature = GSIHashHeader::HdrSignature;
  Header.VerHdr = GSIHashHeader::HdrVersion;
  Header.HrSize = HashRecords.size() * sizeof(PSHashRecord);
  Header.NumBuckets = HashBitmap.size() * 4 + HashBuckets.size() * 4;

  if (auto EC = Writer.writeObject(Header))
    return EC;

  if (auto EC = Writer.writeArray(makeArrayRef(HashRecords)))
    return EC;
  if (auto EC = Writer.writeArray(makeArrayRef(HashBitmap)))
    return EC;
  if (auto EC = Writer.writeArray(makeArrayRef(HashBuckets)))
    return EC;
  return Error::success();
}

static bool isAsciiString(StringRef S) {
  return llvm::all_of(S, [](char C) { return unsigned(C) < 0x80; });
}

// See `caseInsensitiveComparePchPchCchCch` in gsi.cpp
static int gsiRecordCmp(StringRef S1, StringRef S2) {
  size_t LS = S1.size();
  size_t RS = S2.size();
  // Shorter strings always compare less than longer strings.
  if (LS != RS)
    return LS - RS;

  // If either string contains non ascii characters, memcmp them.
  if (LLVM_UNLIKELY(!isAsciiString(S1) || !isAsciiString(S2)))
    return memcmp(S1.data(), S2.data(), LS);

  // Both strings are ascii, perform a case-insenstive comparison.
  return S1.compare_lower(S2.data());
}

void GSIHashStreamBuilder::finalizeBuckets(uint32_t RecordZeroOffset) {
  // Build up a list of globals to be bucketed. This repurposes the BulkPublic
  // struct with different meanings for the fields to avoid reallocating a new
  // vector during public symbol table hash construction.
  std::vector<BulkPublic> Globals;
  Globals.resize(Records.size());
  uint32_t SymOffset = RecordZeroOffset;
  for (size_t I = 0, E = Records.size(); I < E; ++I) {
    StringRef Name = getSymbolName(Records[I]);
    Globals[I].Name = Name.data();
    Globals[I].NameLen = Name.size();
    Globals[I].SymOffset = SymOffset;
    SymOffset += Records[I].length();
  }

  finalizeBuckets(RecordZeroOffset, std::move(Globals));
}

void GSIHashStreamBuilder::finalizeBuckets(uint32_t RecordZeroOffset,
                                           std::vector<BulkPublic> &&Globals) {
  // Hash every name in parallel. The Segment field is no longer needed, so
  // store the BucketIdx in a union.
  parallelForEachN(0, Globals.size(), [&](size_t I) {
    Globals[I].U.BucketIdx = hashStringV1(Globals[I].Name) % IPHR_HASH;
  });

  // Parallel sort by bucket index, then name within the buckets. Within the
  // buckets, sort each bucket by memcmp of the symbol's name.  It's important
  // that we use the same sorting algorithm as is used by the reference
  // implementation to ensure that the search for a record within a bucket can
  // properly early-out when it detects the record won't be found.  The
  // algorithm used here corredsponds to the function
  // caseInsensitiveComparePchPchCchCch in the reference implementation.
  auto BucketCmp = [](const BulkPublic &L, const BulkPublic &R) {
    if (L.U.BucketIdx != R.U.BucketIdx)
      return L.U.BucketIdx < R.U.BucketIdx;
    int Cmp = gsiRecordCmp(L.getName(), R.getName());
    if (Cmp != 0)
      return Cmp < 0;
    // This comparison is necessary to make the sorting stable in the presence
    // of two static globals with the same name. The easiest way to observe
    // this is with S_LDATA32 records.
    return L.SymOffset < R.SymOffset;
  };
  parallelSort(Globals, BucketCmp);

  // Zero out the bucket index bitmap.
  for (ulittle32_t &Word : HashBitmap)
    Word = 0;

  // Compute the three tables: the hash records in bucket and chain order, the
  // bucket presence bitmap, and the bucket chain start offsets.
  HashRecords.reserve(Globals.size());
  uint32_t LastBucketIdx = ~0U;
  for (const BulkPublic &Global : Globals) {
    // If this is a new bucket, add it to the bitmap and the start offset map.
    uint32_t BucketIdx = Global.U.BucketIdx;
    if (LastBucketIdx != BucketIdx) {
      HashBitmap[BucketIdx / 32] |= 1U << (BucketIdx % 32);

      // Calculate what the offset of the first hash record in the chain would
      // be if it were inflated to contain 32-bit pointers. On a 32-bit system,
      // each record would be 12 bytes. See HROffsetCalc in gsi.h.
      const int SizeOfHROffsetCalc = 12;
      ulittle32_t ChainStartOff =
          ulittle32_t(HashRecords.size() * SizeOfHROffsetCalc);
      HashBuckets.push_back(ChainStartOff);
      LastBucketIdx = BucketIdx;
    }

    // Create the hash record. Add one when writing symbol offsets to disk.
    // See GSI1::fixSymRecs. Always use a refcount of 1 for now.
    PSHashRecord HRec;
    HRec.Off = Global.SymOffset + 1;
    HRec.CRef = 1;
    HashRecords.push_back(HRec);
  }
}

GSIStreamBuilder::GSIStreamBuilder(msf::MSFBuilder &Msf)
    : Msf(Msf), PSH(std::make_unique<GSIHashStreamBuilder>()),
      GSH(std::make_unique<GSIHashStreamBuilder>()) {}

GSIStreamBuilder::~GSIStreamBuilder() {}

uint32_t GSIStreamBuilder::calculatePublicsHashStreamSize() const {
  uint32_t Size = 0;
  Size += sizeof(PublicsStreamHeader);
  Size += PSH->calculateSerializedLength();
  Size += PubAddrMap.size() * sizeof(uint32_t); // AddrMap
  // FIXME: Add thunk map and section offsets for incremental linking.

  return Size;
}

uint32_t GSIStreamBuilder::calculateGlobalsHashStreamSize() const {
  return GSH->calculateSerializedLength();
}

Error GSIStreamBuilder::finalizeMsfLayout() {
  // First we write public symbol records, then we write global symbol records.
  uint32_t PublicsSize = PSH->calculateRecordByteSize();
  uint32_t GlobalsSize = GSH->calculateRecordByteSize();
  GSH->finalizeBuckets(PublicsSize);

  Expected<uint32_t> Idx = Msf.addStream(calculateGlobalsHashStreamSize());
  if (!Idx)
    return Idx.takeError();
  GlobalsStreamIndex = *Idx;

  Idx = Msf.addStream(calculatePublicsHashStreamSize());
  if (!Idx)
    return Idx.takeError();
  PublicsStreamIndex = *Idx;

  uint32_t RecordBytes = PublicsSize + GlobalsSize;

  Idx = Msf.addStream(RecordBytes);
  if (!Idx)
    return Idx.takeError();
  RecordStreamIndex = *Idx;
  return Error::success();
}

void GSIStreamBuilder::addPublicSymbols(std::vector<BulkPublic> &&Publics) {
  // Sort the symbols by name. PDBs contain lots of symbols, so use parallelism.
  parallelSort(Publics, [](const BulkPublic &L, const BulkPublic &R) {
    return L.getName() < R.getName();
  });

  // Assign offsets and allocate one contiguous block of memory for all public
  // symbols.
  uint32_t SymOffset = 0;
  for (BulkPublic &Pub : Publics) {
    Pub.SymOffset = SymOffset;
    SymOffset += sizeOfPublic(Pub);
  }
  uint8_t *Mem =
      reinterpret_cast<uint8_t *>(Msf.getAllocator().Allocate(SymOffset, 4));

  // Instead of storing individual CVSymbol records, store them as one giant
  // buffer.
  // FIXME: This is kind of a hack. This makes Records.size() wrong, and we have
  // to account for that elsewhere.
  PSH->Records.push_back(CVSymbol(makeArrayRef(Mem, SymOffset)));

  // Serialize them in parallel.
  parallelForEachN(0, Publics.size(), [&](size_t I) {
    const BulkPublic &Pub = Publics[I];
    serializePublic(Mem + Pub.SymOffset, Pub);
  });

  // Re-sort the publics by address so we can build the address map. We no
  // longer need the original ordering.
  auto AddrCmp = [](const BulkPublic &L, const BulkPublic &R) {
    if (L.U.Segment != R.U.Segment)
      return L.U.Segment < R.U.Segment;
    if (L.Offset != R.Offset)
      return L.Offset < R.Offset;
    // parallelSort is unstable, so we have to do name comparison to ensure
    // that two names for the same location come out in a determinstic order.
    return L.getName() < R.getName();
  };
  parallelSort(Publics, AddrCmp);

  // Fill in the symbol offsets in the appropriate order.
  PubAddrMap.reserve(Publics.size());
  for (const BulkPublic &Pub : Publics)
    PubAddrMap.push_back(ulittle32_t(Pub.SymOffset));

  // Finalize public symbol buckets immediately after they have been added.
  // They should all be warm in the cache at this point, so go ahead and do it
  // now.
  PSH->finalizeBuckets(0, std::move(Publics));
}

void GSIStreamBuilder::addGlobalSymbol(const ProcRefSym &Sym) {
  GSH->addSymbol(Sym, Msf);
}

void GSIStreamBuilder::addGlobalSymbol(const DataSym &Sym) {
  GSH->addSymbol(Sym, Msf);
}

void GSIStreamBuilder::addGlobalSymbol(const ConstantSym &Sym) {
  GSH->addSymbol(Sym, Msf);
}

void GSIStreamBuilder::addGlobalSymbol(const codeview::CVSymbol &Sym) {
  GSH->addSymbol(Sym);
}

static Error writeRecords(BinaryStreamWriter &Writer,
                          ArrayRef<CVSymbol> Records) {
  BinaryItemStream<CVSymbol> ItemStream(support::endianness::little);
  ItemStream.setItems(Records);
  BinaryStreamRef RecordsRef(ItemStream);
  return Writer.writeStreamRef(RecordsRef);
}

Error GSIStreamBuilder::commitSymbolRecordStream(
    WritableBinaryStreamRef Stream) {
  BinaryStreamWriter Writer(Stream);

  // Write public symbol records first, followed by global symbol records.  This
  // must match the order that we assume in finalizeMsfLayout when computing
  // PSHZero and GSHZero.
  if (auto EC = writeRecords(Writer, PSH->Records))
    return EC;
  if (auto EC = writeRecords(Writer, GSH->Records))
    return EC;

  return Error::success();
}

Error GSIStreamBuilder::commitPublicsHashStream(
    WritableBinaryStreamRef Stream) {
  BinaryStreamWriter Writer(Stream);
  PublicsStreamHeader Header;

  // FIXME: Fill these in. They are for incremental linking.
  Header.SymHash = PSH->calculateSerializedLength();
  Header.AddrMap = PubAddrMap.size() * 4;
  Header.NumThunks = 0;
  Header.SizeOfThunk = 0;
  Header.ISectThunkTable = 0;
  memset(Header.Padding, 0, sizeof(Header.Padding));
  Header.OffThunkTable = 0;
  Header.NumSections = 0;
  if (auto EC = Writer.writeObject(Header))
    return EC;

  if (auto EC = PSH->commit(Writer))
    return EC;

  if (auto EC = Writer.writeArray(makeArrayRef(PubAddrMap)))
    return EC;

  return Error::success();
}

Error GSIStreamBuilder::commitGlobalsHashStream(
    WritableBinaryStreamRef Stream) {
  BinaryStreamWriter Writer(Stream);
  return GSH->commit(Writer);
}

Error GSIStreamBuilder::commit(const msf::MSFLayout &Layout,
                               WritableBinaryStreamRef Buffer) {
  auto GS = WritableMappedBlockStream::createIndexedStream(
      Layout, Buffer, getGlobalsStreamIndex(), Msf.getAllocator());
  auto PS = WritableMappedBlockStream::createIndexedStream(
      Layout, Buffer, getPublicsStreamIndex(), Msf.getAllocator());
  auto PRS = WritableMappedBlockStream::createIndexedStream(
      Layout, Buffer, getRecordStreamIndex(), Msf.getAllocator());

  if (auto EC = commitSymbolRecordStream(*PRS))
    return EC;
  if (auto EC = commitGlobalsHashStream(*GS))
    return EC;
  if (auto EC = commitPublicsHashStream(*PS))
    return EC;
  return Error::success();
}
