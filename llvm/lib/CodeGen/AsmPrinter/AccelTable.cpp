//===- llvm/CodeGen/AsmPrinter/AccelTable.cpp - Accelerator Tables --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing accelerator tables.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/AccelTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace llvm;

void AppleAccelTableHeader::emit(AsmPrinter *Asm) {
  // Emit Header.
  Asm->OutStreamer->AddComment("Header Magic");
  Asm->EmitInt32(Header.Magic);
  Asm->OutStreamer->AddComment("Header Version");
  Asm->EmitInt16(Header.Version);
  Asm->OutStreamer->AddComment("Header Hash Function");
  Asm->EmitInt16(Header.HashFunction);
  Asm->OutStreamer->AddComment("Header Bucket Count");
  Asm->EmitInt32(Header.BucketCount);
  Asm->OutStreamer->AddComment("Header Hash Count");
  Asm->EmitInt32(Header.HashCount);
  Asm->OutStreamer->AddComment("Header Data Length");
  Asm->EmitInt32(Header.HeaderDataLength);

  //  Emit Header Data
  Asm->OutStreamer->AddComment("HeaderData Die Offset Base");
  Asm->EmitInt32(HeaderData.DieOffsetBase);
  Asm->OutStreamer->AddComment("HeaderData Atom Count");
  Asm->EmitInt32(HeaderData.Atoms.size());

  for (size_t i = 0; i < HeaderData.Atoms.size(); i++) {
    Atom A = HeaderData.Atoms[i];
    Asm->OutStreamer->AddComment(dwarf::AtomTypeString(A.Type));
    Asm->EmitInt16(A.Type);
    Asm->OutStreamer->AddComment(dwarf::FormEncodingString(A.Form));
    Asm->EmitInt16(A.Form);
  }
}

void AppleAccelTableHeader::setBucketAndHashCount(uint32_t HashCount) {
  if (HashCount > 1024)
    Header.BucketCount = HashCount / 4;
  else if (HashCount > 16)
    Header.BucketCount = HashCount / 2;
  else
    Header.BucketCount = HashCount > 0 ? HashCount : 1;

  Header.HashCount = HashCount;
}

constexpr const AppleAccelTableHeader::Atom AppleAccelTableTypeData::Atoms[];
constexpr const AppleAccelTableHeader::Atom AppleAccelTableOffsetData::Atoms[];
constexpr const AppleAccelTableHeader::Atom AppleAccelTableStaticOffsetData::Atoms[];
constexpr const AppleAccelTableHeader::Atom AppleAccelTableStaticTypeData::Atoms[];

void AppleAccelTableBase::emitHeader(AsmPrinter *Asm) { Header.emit(Asm); }

void AppleAccelTableBase::emitBuckets(AsmPrinter *Asm) {
  unsigned index = 0;
  for (size_t i = 0, e = Buckets.size(); i < e; ++i) {
    Asm->OutStreamer->AddComment("Bucket " + Twine(i));
    if (!Buckets[i].empty())
      Asm->EmitInt32(index);
    else
      Asm->EmitInt32(std::numeric_limits<uint32_t>::max());
    // Buckets point in the list of hashes, not to the data. Do not increment
    // the index multiple times in case of hash collisions.
    uint64_t PrevHash = std::numeric_limits<uint64_t>::max();
    for (auto *HD : Buckets[i]) {
      uint32_t HashValue = HD->HashValue;
      if (PrevHash != HashValue)
        ++index;
      PrevHash = HashValue;
    }
  }
}

void AppleAccelTableBase::emitHashes(AsmPrinter *Asm) {
  uint64_t PrevHash = std::numeric_limits<uint64_t>::max();
  unsigned BucketIdx = 0;
  for (auto &Bucket : Buckets) {
    for (auto &Hash : Bucket) {
      uint32_t HashValue = Hash->HashValue;
      if (PrevHash == HashValue)
        continue;
      Asm->OutStreamer->AddComment("Hash in Bucket " + Twine(BucketIdx));
      Asm->EmitInt32(HashValue);
      PrevHash = HashValue;
    }
    BucketIdx++;
  }
}

void AppleAccelTableBase::emitOffsets(AsmPrinter *Asm,
                                      const MCSymbol *SecBegin) {
  uint64_t PrevHash = std::numeric_limits<uint64_t>::max();
  for (size_t i = 0, e = Buckets.size(); i < e; ++i) {
    for (auto HI = Buckets[i].begin(), HE = Buckets[i].end(); HI != HE; ++HI) {
      uint32_t HashValue = (*HI)->HashValue;
      if (PrevHash == HashValue)
        continue;
      PrevHash = HashValue;
      Asm->OutStreamer->AddComment("Offset in Bucket " + Twine(i));
      MCContext &Context = Asm->OutStreamer->getContext();
      const MCExpr *Sub = MCBinaryExpr::createSub(
          MCSymbolRefExpr::create((*HI)->Sym, Context),
          MCSymbolRefExpr::create(SecBegin, Context), Context);
      Asm->OutStreamer->EmitValue(Sub, sizeof(uint32_t));
    }
  }
}

void AppleAccelTableBase::emitData(AsmPrinter *Asm) {
  for (size_t i = 0, e = Buckets.size(); i < e; ++i) {
    uint64_t PrevHash = std::numeric_limits<uint64_t>::max();
    for (auto &Hash : Buckets[i]) {
      // Terminate the previous entry if there is no hash collision with the
      // current one.
      if (PrevHash != std::numeric_limits<uint64_t>::max() &&
          PrevHash != Hash->HashValue)
        Asm->EmitInt32(0);
      // Remember to emit the label for our offset.
      Asm->OutStreamer->EmitLabel(Hash->Sym);
      Asm->OutStreamer->AddComment(Hash->Str);
      Asm->emitDwarfStringOffset(Hash->Data.Name);
      Asm->OutStreamer->AddComment("Num DIEs");
      Asm->EmitInt32(Hash->Data.Values.size());
      for (const auto *V : Hash->Data.Values) {
        V->emit(Asm);
      }
      PrevHash = Hash->HashValue;
    }
    // Emit the final end marker for the bucket.
    if (!Buckets[i].empty())
      Asm->EmitInt32(0);
  }
}

void AppleAccelTableBase::computeBucketCount() {
  // First get the number of unique hashes.
  std::vector<uint32_t> uniques(Data.size());
  for (size_t i = 0, e = Data.size(); i < e; ++i)
    uniques[i] = Data[i]->HashValue;
  array_pod_sort(uniques.begin(), uniques.end());
  std::vector<uint32_t>::iterator p =
      std::unique(uniques.begin(), uniques.end());

  // Compute the hashes count and use it to set that together with the bucket
  // count in the header.
  Header.setBucketAndHashCount(std::distance(uniques.begin(), p));
}

void AppleAccelTableBase::finalizeTable(AsmPrinter *Asm, StringRef Prefix) {
  // Create the individual hash data outputs.
  Data.reserve(Entries.size());
  for (auto &E : Entries) {
    // Unique the entries.
    std::stable_sort(E.second.Values.begin(), E.second.Values.end(),
                     [](const AppleAccelTableData *A,
                        const AppleAccelTableData *B) { return *A < *B; });
    E.second.Values.erase(
        std::unique(E.second.Values.begin(), E.second.Values.end()),
        E.second.Values.end());

    HashData *Entry = new (Allocator) HashData(E.first(), E.second);
    Data.push_back(Entry);
  }

  // Figure out how many buckets we need, then compute the bucket contents and
  // the final ordering. We'll emit the hashes and offsets by doing a walk
  // during the emission phase. We add temporary symbols to the data so that we
  // can reference them during the offset later, we'll emit them when we emit
  // the data.
  computeBucketCount();

  // Compute bucket contents and final ordering.
  Buckets.resize(Header.getBucketCount());
  for (auto &D : Data) {
    uint32_t bucket = D->HashValue % Header.getBucketCount();
    Buckets[bucket].push_back(D);
    D->Sym = Asm->createTempSymbol(Prefix);
  }

  // Sort the contents of the buckets by hash value so that hash collisions end
  // up together. Stable sort makes testing easier and doesn't cost much more.
  for (auto &Bucket : Buckets)
    std::stable_sort(Bucket.begin(), Bucket.end(),
                     [](HashData *LHS, HashData *RHS) {
                       return LHS->HashValue < RHS->HashValue;
                     });
}

void AppleAccelTableOffsetData::emit(AsmPrinter *Asm) const {
  Asm->EmitInt32(Die->getDebugSectionOffset());
}

void AppleAccelTableTypeData::emit(AsmPrinter *Asm) const {
  Asm->EmitInt32(Die->getDebugSectionOffset());
  Asm->EmitInt16(Die->getTag());
  Asm->EmitInt8(0);
}

void AppleAccelTableStaticOffsetData::emit(AsmPrinter *Asm) const {
  Asm->EmitInt32(Offset);
}

void AppleAccelTableStaticTypeData::emit(AsmPrinter *Asm) const {
  Asm->EmitInt32(Offset);
  Asm->EmitInt16(Tag);
  Asm->EmitInt8(ObjCClassIsImplementation ? dwarf::DW_FLAG_type_implementation
                                          : 0);
  Asm->EmitInt32(QualifiedNameHash);
}
