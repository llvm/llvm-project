//===- DbiStreamBuilder.cpp - PDB Dbi Stream Creation -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/PublicsStreamBuilder.h"
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

PublicsStreamBuilder::PublicsStreamBuilder(msf::MSFBuilder &Msf)
    : Table(new GSIHashTableBuilder), Msf(Msf) {}

PublicsStreamBuilder::~PublicsStreamBuilder() {}

uint32_t PublicsStreamBuilder::calculateSerializedLength() const {
  uint32_t Size = 0;
  Size += sizeof(PublicsStreamHeader);
  Size += sizeof(GSIHashHeader);
  Size += Table->HashRecords.size() * sizeof(PSHashRecord);
  Size += Table->HashBitmap.size() * sizeof(uint32_t);
  Size += Table->HashBuckets.size() * sizeof(uint32_t);

  Size += Publics.size() * sizeof(uint32_t); // AddrMap

  // FIXME: Add thunk map and section offsets for incremental linking.

  return Size;
}

Error PublicsStreamBuilder::finalizeMsfLayout() {
  Table->addSymbols(Publics);

  Expected<uint32_t> Idx = Msf.addStream(calculateSerializedLength());
  if (!Idx)
    return Idx.takeError();
  StreamIdx = *Idx;

  uint32_t PublicRecordBytes = 0;
  for (auto &Pub : Publics)
    PublicRecordBytes += Pub.length();

  Expected<uint32_t> RecordIdx = Msf.addStream(PublicRecordBytes);
  if (!RecordIdx)
    return RecordIdx.takeError();
  RecordStreamIdx = *RecordIdx;
  return Error::success();
}

void PublicsStreamBuilder::addPublicSymbol(const PublicSym32 &Pub) {
  Publics.push_back(SymbolSerializer::writeOneSymbol(
      const_cast<PublicSym32 &>(Pub), Msf.getAllocator(),
      CodeViewContainer::Pdb));
}

// FIXME: Put this back in the header.
struct PubSymLayout {
  ulittle16_t reclen;
  ulittle16_t reckind;
  ulittle32_t flags;
  ulittle32_t off;
  ulittle16_t seg;
  char name[1];
};

bool comparePubSymByAddrAndName(const CVSymbol *LS, const CVSymbol *RS) {
  assert(LS->length() > sizeof(PubSymLayout) &&
         RS->length() > sizeof(PubSymLayout));
  auto *L = reinterpret_cast<const PubSymLayout *>(LS->data().data());
  auto *R = reinterpret_cast<const PubSymLayout *>(RS->data().data());
  if (L->seg < R->seg)
    return true;
  if (L->seg > R->seg)
    return false;
  if (L->off < R->off)
    return true;
  if (L->off > R->off)
    return false;
  return strcmp(L->name, R->name) < 0;
}

static StringRef getSymbolName(const CVSymbol &Sym) {
  assert(Sym.kind() == S_PUB32 && "handle other kinds");
  ArrayRef<uint8_t> NameBytes =
      Sym.data().drop_front(offsetof(PubSymLayout, name));
  return StringRef(reinterpret_cast<const char *>(NameBytes.data()),
                   NameBytes.size())
      .trim('\0');
}

/// Compute the address map. The address map is an array of symbol offsets
/// sorted so that it can be binary searched by address.
static std::vector<ulittle32_t> computeAddrMap(ArrayRef<CVSymbol> Publics) {
  // Make a vector of pointers to the symbols so we can sort it by address.
  // Also gather the symbol offsets while we're at it.
  std::vector<const CVSymbol *> PublicsByAddr;
  std::vector<uint32_t> SymOffsets;
  PublicsByAddr.reserve(Publics.size());
  uint32_t SymOffset = 0;
  for (const CVSymbol &Sym : Publics) {
    PublicsByAddr.push_back(&Sym);
    SymOffsets.push_back(SymOffset);
    SymOffset += Sym.length();
  }
  std::stable_sort(PublicsByAddr.begin(), PublicsByAddr.end(),
                   comparePubSymByAddrAndName);

  // Fill in the symbol offsets in the appropriate order.
  std::vector<ulittle32_t> AddrMap;
  AddrMap.reserve(Publics.size());
  for (const CVSymbol *Sym : PublicsByAddr) {
    ptrdiff_t Idx = std::distance(Publics.data(), Sym);
    assert(Idx >= 0 && size_t(Idx) < Publics.size());
    AddrMap.push_back(ulittle32_t(SymOffsets[Idx]));
  }
  return AddrMap;
}

Error PublicsStreamBuilder::commit(BinaryStreamWriter &PublicsWriter,
                                   BinaryStreamWriter &RecWriter) {
  assert(Table->HashRecords.size() == Publics.size());

  PublicsStreamHeader PSH;
  GSIHashHeader GSH;

  PSH.AddrMap = Publics.size() * 4;

  // FIXME: Fill these in. They are for incremental linking.
  PSH.NumThunks = 0;
  PSH.SizeOfThunk = 0;
  PSH.ISectThunkTable = 0;
  PSH.OffThunkTable = 0;
  PSH.NumSections = 0;

  GSH.VerSignature = GSIHashHeader::HdrSignature;
  GSH.VerHdr = GSIHashHeader::HdrVersion;
  GSH.HrSize = Table->HashRecords.size() * sizeof(PSHashRecord);
  GSH.NumBuckets = Table->HashBitmap.size() * 4 + Table->HashBuckets.size() * 4;

  PSH.SymHash = sizeof(GSH) + GSH.HrSize + GSH.NumBuckets;

  if (auto EC = PublicsWriter.writeObject(PSH))
    return EC;
  if (auto EC = PublicsWriter.writeObject(GSH))
    return EC;

  if (auto EC = PublicsWriter.writeArray(makeArrayRef(Table->HashRecords)))
    return EC;
  if (auto EC = PublicsWriter.writeArray(makeArrayRef(Table->HashBitmap)))
    return EC;
  if (auto EC = PublicsWriter.writeArray(makeArrayRef(Table->HashBuckets)))
    return EC;

  std::vector<ulittle32_t> AddrMap = computeAddrMap(Publics);
  if (auto EC = PublicsWriter.writeArray(makeArrayRef(AddrMap)))
    return EC;

  BinaryItemStream<CVSymbol> Records(support::endianness::little);
  Records.setItems(Publics);
  BinaryStreamRef RecordsRef(Records);
  if (auto EC = RecWriter.writeStreamRef(RecordsRef))
    return EC;

  return Error::success();
}

void GSIHashTableBuilder::addSymbols(ArrayRef<CVSymbol> Symbols) {
  std::array<std::vector<PSHashRecord>, IPHR_HASH + 1> TmpBuckets;
  uint32_t SymOffset = 0;
  for (const CVSymbol &Sym : Symbols) {
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
  HashRecords.reserve(Symbols.size());
  for (ulittle32_t &Word : HashBitmap)
    Word = 0;
  for (size_t BucketIdx = 0; BucketIdx < IPHR_HASH + 1; ++BucketIdx) {
    auto &Bucket = TmpBuckets[BucketIdx];
    if (Bucket.empty())
      continue;
    HashBitmap[BucketIdx / 32] |= 1U << (BucketIdx % 32);

    // Calculate what the offset of the first hash record in the chain would be
    // if it were inflated to contain 32-bit pointers. On a 32-bit system, each
    // record would be 12 bytes. See HROffsetCalc in gsi.h.
    const int SizeOfHROffsetCalc = 12;
    ulittle32_t ChainStartOff =
        ulittle32_t(HashRecords.size() * SizeOfHROffsetCalc);
    HashBuckets.push_back(ChainStartOff);
    for (const auto &HR : Bucket)
      HashRecords.push_back(HR);
  }
}
