//===- DbiStreamBuilder.cpp - PDB Dbi Stream Creation -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/GSIStreamBuilder.h"

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
#include <algorithm>
#include <vector>

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;
using namespace llvm::codeview;

struct llvm::pdb::GSIHashStreamBuilder {
  std::vector<CVSymbol> Records;
  uint32_t StreamIndex;
  std::vector<PSHashRecord> HashRecords;
  std::array<support::ulittle32_t, (IPHR_HASH + 32) / 32> HashBitmap;
  std::vector<support::ulittle32_t> HashBuckets;

  uint32_t calculateSerializedLength() const;
  uint32_t calculateRecordByteSize() const;
  Error commit(BinaryStreamWriter &Writer);
  void finalizeBuckets(uint32_t RecordZeroOffset);

  template <typename T> void addSymbol(const T &Symbol, MSFBuilder &Msf) {
    T Copy(Symbol);
    Records.push_back(SymbolSerializer::writeOneSymbol(Copy, Msf.getAllocator(),
                                                       CodeViewContainer::Pdb));
  }
  void addSymbol(const CVSymbol &Symbol) { Records.push_back(Symbol); }
};

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

void GSIHashStreamBuilder::finalizeBuckets(uint32_t RecordZeroOffset) {
  std::array<std::vector<PSHashRecord>, IPHR_HASH + 1> TmpBuckets;
  uint32_t SymOffset = RecordZeroOffset;
  for (const CVSymbol &Sym : Records) {
    PSHashRecord HR;
    // Add one when writing symbol offsets to disk. See GSI1::fixSymRecs.
    HR.Off = SymOffset + 1;
    HR.CRef = 1; // Always use a refcount of 1.

    // Hash the name to figure out which bucket this goes into.
    StringRef Name = getSymbolName(Sym);
    size_t BucketIdx = hashStringV1(Name) % IPHR_HASH;
    TmpBuckets[BucketIdx].push_back(HR); // FIXME: Does order matter?

    SymOffset += Sym.length();
  }

  // Compute the three tables: the hash records in bucket and chain order, the
  // bucket presence bitmap, and the bucket chain start offsets.
  HashRecords.reserve(Records.size());
  for (ulittle32_t &Word : HashBitmap)
    Word = 0;
  for (size_t BucketIdx = 0; BucketIdx < IPHR_HASH + 1; ++BucketIdx) {
    auto &Bucket = TmpBuckets[BucketIdx];
    if (Bucket.empty())
      continue;
    HashBitmap[BucketIdx / 32] |= 1U << (BucketIdx % 32);

    // Calculate what the offset of the first hash record in the chain would
    // be if it were inflated to contain 32-bit pointers. On a 32-bit system,
    // each record would be 12 bytes. See HROffsetCalc in gsi.h.
    const int SizeOfHROffsetCalc = 12;
    ulittle32_t ChainStartOff =
        ulittle32_t(HashRecords.size() * SizeOfHROffsetCalc);
    HashBuckets.push_back(ChainStartOff);
    for (const auto &HR : Bucket)
      HashRecords.push_back(HR);
  }
}

GSIStreamBuilder::GSIStreamBuilder(msf::MSFBuilder &Msf)
    : Msf(Msf), PSH(llvm::make_unique<GSIHashStreamBuilder>()),
      GSH(llvm::make_unique<GSIHashStreamBuilder>()) {}

GSIStreamBuilder::~GSIStreamBuilder() {}

uint32_t GSIStreamBuilder::calculatePublicsHashStreamSize() const {
  uint32_t Size = 0;
  Size += sizeof(PublicsStreamHeader);
  Size += PSH->calculateSerializedLength();
  Size += PSH->Records.size() * sizeof(uint32_t); // AddrMap
  // FIXME: Add thunk map and section offsets for incremental linking.

  return Size;
}

uint32_t GSIStreamBuilder::calculateGlobalsHashStreamSize() const {
  return GSH->calculateSerializedLength();
}

Error GSIStreamBuilder::finalizeMsfLayout() {
  // First we write public symbol records, then we write global symbol records.
  uint32_t PSHZero = 0;
  uint32_t GSHZero = PSH->calculateRecordByteSize();

  PSH->finalizeBuckets(PSHZero);
  GSH->finalizeBuckets(GSHZero);

  Expected<uint32_t> Idx = Msf.addStream(calculatePublicsHashStreamSize());
  if (!Idx)
    return Idx.takeError();
  PSH->StreamIndex = *Idx;
  Idx = Msf.addStream(calculateGlobalsHashStreamSize());
  if (!Idx)
    return Idx.takeError();
  GSH->StreamIndex = *Idx;

  uint32_t RecordBytes =
      GSH->calculateRecordByteSize() + PSH->calculateRecordByteSize();

  Idx = Msf.addStream(RecordBytes);
  if (!Idx)
    return Idx.takeError();
  RecordStreamIdx = *Idx;
  return Error::success();
}

static bool comparePubSymByAddrAndName(const CVSymbol *LS, const CVSymbol *RS) {
  assert(LS->kind() == SymbolKind::S_PUB32);
  assert(RS->kind() == SymbolKind::S_PUB32);

  PublicSym32 PSL =
      cantFail(SymbolDeserializer::deserializeAs<PublicSym32>(*LS));
  PublicSym32 PSR =
      cantFail(SymbolDeserializer::deserializeAs<PublicSym32>(*RS));

  if (PSL.Segment != PSR.Segment)
    return PSL.Segment < PSR.Segment;
  if (PSL.Offset != PSR.Offset)
    return PSL.Offset < PSR.Offset;

  return PSL.Name < PSR.Name;
}

/// Compute the address map. The address map is an array of symbol offsets
/// sorted so that it can be binary searched by address.
static std::vector<ulittle32_t> computeAddrMap(ArrayRef<CVSymbol> Records) {
  // Make a vector of pointers to the symbols so we can sort it by address.
  // Also gather the symbol offsets while we're at it.
  std::vector<const CVSymbol *> PublicsByAddr;
  std::vector<uint32_t> SymOffsets;
  PublicsByAddr.reserve(Records.size());
  uint32_t SymOffset = 0;
  for (const CVSymbol &Sym : Records) {
    PublicsByAddr.push_back(&Sym);
    SymOffsets.push_back(SymOffset);
    SymOffset += Sym.length();
  }
  std::stable_sort(PublicsByAddr.begin(), PublicsByAddr.end(),
                   comparePubSymByAddrAndName);

  // Fill in the symbol offsets in the appropriate order.
  std::vector<ulittle32_t> AddrMap;
  AddrMap.reserve(Records.size());
  for (const CVSymbol *Sym : PublicsByAddr) {
    ptrdiff_t Idx = std::distance(Records.data(), Sym);
    assert(Idx >= 0 && size_t(Idx) < Records.size());
    AddrMap.push_back(ulittle32_t(SymOffsets[Idx]));
  }
  return AddrMap;
}

uint32_t GSIStreamBuilder::getPublicsStreamIndex() const {
  return PSH->StreamIndex;
}

uint32_t GSIStreamBuilder::getGlobalsStreamIndex() const {
  return GSH->StreamIndex;
}

void GSIStreamBuilder::addPublicSymbol(const PublicSym32 &Pub) {
  PSH->addSymbol(Pub, Msf);
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

void GSIStreamBuilder::addGlobalSymbol(const UDTSym &Sym) {
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
  Header.NumThunks = 0;
  Header.SizeOfThunk = 0;
  Header.ISectThunkTable = 0;
  Header.OffThunkTable = 0;
  Header.NumSections = 0;
  Header.SymHash = PSH->calculateSerializedLength();
  Header.AddrMap = PSH->Records.size() * 4;
  if (auto EC = Writer.writeObject(Header))
    return EC;

  if (auto EC = PSH->commit(Writer))
    return EC;

  std::vector<ulittle32_t> AddrMap = computeAddrMap(PSH->Records);
  if (auto EC = Writer.writeArray(makeArrayRef(AddrMap)))
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
      Layout, Buffer, getRecordStreamIdx(), Msf.getAllocator());

  if (auto EC = commitSymbolRecordStream(*PRS))
    return EC;
  if (auto EC = commitGlobalsHashStream(*GS))
    return EC;
  if (auto EC = commitPublicsHashStream(*PS))
    return EC;
  return Error::success();
}
