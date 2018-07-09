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
#include "DwarfCompileUnit.h"
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

void AccelTableBase::computeBucketCount() {
  // First get the number of unique hashes.
  std::vector<uint32_t> Uniques;
  Uniques.reserve(Entries.size());
  for (const auto &E : Entries)
    Uniques.push_back(E.second.HashValue);
  array_pod_sort(Uniques.begin(), Uniques.end());
  std::vector<uint32_t>::iterator P =
      std::unique(Uniques.begin(), Uniques.end());

  UniqueHashCount = std::distance(Uniques.begin(), P);

  if (UniqueHashCount > 1024)
    BucketCount = UniqueHashCount / 4;
  else if (UniqueHashCount > 16)
    BucketCount = UniqueHashCount / 2;
  else
    BucketCount = std::max<uint32_t>(UniqueHashCount, 1);
}

void AccelTableBase::finalize(AsmPrinter *Asm, StringRef Prefix) {
  // Create the individual hash data outputs.
  for (auto &E : Entries) {
    // Unique the entries.
    std::stable_sort(E.second.Values.begin(), E.second.Values.end(),
                     [](const AccelTableData *A, const AccelTableData *B) {
                       return *A < *B;
                     });
    E.second.Values.erase(
        std::unique(E.second.Values.begin(), E.second.Values.end()),
        E.second.Values.end());
  }

  // Figure out how many buckets we need, then compute the bucket contents and
  // the final ordering. The hashes and offsets can be emitted by walking these
  // data structures. We add temporary symbols to the data so they can be
  // referenced when emitting the offsets.
  computeBucketCount();

  // Compute bucket contents and final ordering.
  Buckets.resize(BucketCount);
  for (auto &E : Entries) {
    uint32_t Bucket = E.second.HashValue % BucketCount;
    Buckets[Bucket].push_back(&E.second);
    E.second.Sym = Asm->createTempSymbol(Prefix);
  }

  // Sort the contents of the buckets by hash value so that hash collisions end
  // up together. Stable sort makes testing easier and doesn't cost much more.
  for (auto &Bucket : Buckets)
    std::stable_sort(Bucket.begin(), Bucket.end(),
                     [](HashData *LHS, HashData *RHS) {
                       return LHS->HashValue < RHS->HashValue;
                     });
}

namespace {
/// Base class for writing out Accelerator tables. It holds the common
/// functionality for the two Accelerator table types.
class AccelTableWriter {
protected:
  AsmPrinter *const Asm;          ///< Destination.
  const AccelTableBase &Contents; ///< Data to emit.

  /// Controls whether to emit duplicate hash and offset table entries for names
  /// with identical hashes. Apple tables don't emit duplicate entries, DWARF v5
  /// tables do.
  const bool SkipIdenticalHashes;

  void emitHashes() const;

  /// Emit offsets to lists of entries with identical names. The offsets are
  /// relative to the Base argument.
  void emitOffsets(const MCSymbol *Base) const;

public:
  AccelTableWriter(AsmPrinter *Asm, const AccelTableBase &Contents,
                   bool SkipIdenticalHashes)
      : Asm(Asm), Contents(Contents), SkipIdenticalHashes(SkipIdenticalHashes) {
  }
};

class AppleAccelTableWriter : public AccelTableWriter {
  using Atom = AppleAccelTableData::Atom;

  /// The fixed header of an Apple Accelerator Table.
  struct Header {
    uint32_t Magic = MagicHash;
    uint16_t Version = 1;
    uint16_t HashFunction = dwarf::DW_hash_function_djb;
    uint32_t BucketCount;
    uint32_t HashCount;
    uint32_t HeaderDataLength;

    /// 'HASH' magic value to detect endianness.
    static const uint32_t MagicHash = 0x48415348;

    Header(uint32_t BucketCount, uint32_t UniqueHashCount, uint32_t DataLength)
        : BucketCount(BucketCount), HashCount(UniqueHashCount),
          HeaderDataLength(DataLength) {}

    void emit(AsmPrinter *Asm) const;
#ifndef NDEBUG
    void print(raw_ostream &OS) const;
    void dump() const { print(dbgs()); }
#endif
  };

  /// The HeaderData describes the structure of an Apple accelerator table
  /// through a list of Atoms.
  struct HeaderData {
    /// In the case of data that is referenced via DW_FORM_ref_* the offset
    /// base is used to describe the offset for all forms in the list of atoms.
    uint32_t DieOffsetBase;

    const SmallVector<Atom, 4> Atoms;

    HeaderData(ArrayRef<Atom> AtomList, uint32_t Offset = 0)
        : DieOffsetBase(Offset), Atoms(AtomList.begin(), AtomList.end()) {}

    void emit(AsmPrinter *Asm) const;
#ifndef NDEBUG
    void print(raw_ostream &OS) const;
    void dump() const { print(dbgs()); }
#endif
  };

  Header Header;
  HeaderData HeaderData;
  const MCSymbol *SecBegin;

  void emitBuckets() const;
  void emitData() const;

public:
  AppleAccelTableWriter(AsmPrinter *Asm, const AccelTableBase &Contents,
                        ArrayRef<Atom> Atoms, const MCSymbol *SecBegin)
      : AccelTableWriter(Asm, Contents, true),
        Header(Contents.getBucketCount(), Contents.getUniqueHashCount(),
               8 + (Atoms.size() * 4)),
        HeaderData(Atoms), SecBegin(SecBegin) {}

  void emit() const;

#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  void dump() const { print(dbgs()); }
#endif
};

/// Class responsible for emitting a DWARF v5 Accelerator Table. The only public
/// function is emit(), which performs the actual emission.
class Dwarf5AccelTableWriter : public AccelTableWriter {
  struct Header {
    uint32_t UnitLength = 0;
    uint16_t Version = 5;
    uint16_t Padding = 0;
    uint32_t CompUnitCount;
    uint32_t LocalTypeUnitCount = 0;
    uint32_t ForeignTypeUnitCount = 0;
    uint32_t BucketCount;
    uint32_t NameCount;
    uint32_t AbbrevTableSize = 0;
    uint32_t AugmentationStringSize = sizeof(AugmentationString);
    char AugmentationString[8] = {'L', 'L', 'V', 'M', '0', '7', '0', '0'};

    Header(uint32_t CompUnitCount, uint32_t BucketCount, uint32_t NameCount)
        : CompUnitCount(CompUnitCount), BucketCount(BucketCount),
          NameCount(NameCount) {}

    void emit(const Dwarf5AccelTableWriter &Ctx) const;
  };
  struct AttributeEncoding {
    dwarf::Index Index;
    dwarf::Form Form;
  };

  Header Header;
  DenseMap<uint32_t, SmallVector<AttributeEncoding, 2>> Abbreviations;
  std::unique_ptr<AccelTableWriterInfo> WriterInfo;
  MCSymbol *ContributionStart = Asm->createTempSymbol("names_start");
  MCSymbol *ContributionEnd = Asm->createTempSymbol("names_end");
  MCSymbol *AbbrevStart = Asm->createTempSymbol("names_abbrev_start");
  MCSymbol *AbbrevEnd = Asm->createTempSymbol("names_abbrev_end");
  MCSymbol *EntryPool = Asm->createTempSymbol("names_entries");

  DenseSet<uint32_t> getUniqueTags() const;

  // Right now, we emit uniform attributes for all tags.
  SmallVector<AttributeEncoding, 2> getUniformAttributes() const;

  void emitCUList() const;
  void emitBuckets() const;
  void emitStringOffsets() const;
  void emitAbbrevs() const;
  void emitEntry(const DWARF5AccelTableData &Data) const;
  void emitData() const;

public:
  Dwarf5AccelTableWriter(AsmPrinter *Asm, const AccelTableBase &Contents,
                          std::unique_ptr<AccelTableWriterInfo> WriterInfo);

  void emit() const;
};

/// Default emitter info used by DwarfDebug.
class DefaultAccelTableWriterInfo final : public AccelTableWriterInfo {
private:
  const DwarfDebug &DD;
  ArrayRef<std::unique_ptr<DwarfCompileUnit>> CompUnits;

public:
  DefaultAccelTableWriterInfo(const DwarfDebug &DD, ArrayRef<std::unique_ptr<DwarfCompileUnit>> CompUnits);
  MCSymbol *getLabelForCU(unsigned Idx) const override;
  unsigned getNumberOfCUs() const override;
  unsigned getUnqiueIDForUnitDie(const DIE *UnitDie) const override;
};
} // namespace

DefaultAccelTableWriterInfo::DefaultAccelTableWriterInfo(
    const DwarfDebug &DD, ArrayRef<std::unique_ptr<DwarfCompileUnit>> CompUnits)
    : DD(DD), CompUnits(CompUnits) {}

MCSymbol *DefaultAccelTableWriterInfo::getLabelForCU(unsigned Idx) const {
  assert(Idx < CompUnits.size());
  const auto &CU = CompUnits[Idx];
  assert(Idx == CU->getUniqueID());
  const DwarfCompileUnit *MainCU =
      DD.useSplitDwarf() ? CU->getSkeleton() : CU.get();
  return MainCU->getLabelBegin();
}

unsigned DefaultAccelTableWriterInfo::getNumberOfCUs() const {
  return CompUnits.size();
}

unsigned
DefaultAccelTableWriterInfo::getUnqiueIDForUnitDie(const DIE *UnitDie) const {
  return DD.lookupCU(UnitDie)->getUniqueID();
}

void AccelTableWriter::emitHashes() const {
  uint64_t PrevHash = std::numeric_limits<uint64_t>::max();
  unsigned BucketIdx = 0;
  for (auto &Bucket : Contents.getBuckets()) {
    for (auto &Hash : Bucket) {
      uint32_t HashValue = Hash->HashValue;
      if (SkipIdenticalHashes && PrevHash == HashValue)
        continue;
      Asm->OutStreamer->AddComment("Hash in Bucket " + Twine(BucketIdx));
      Asm->emitInt32(HashValue);
      PrevHash = HashValue;
    }
    BucketIdx++;
  }
}

void AccelTableWriter::emitOffsets(const MCSymbol *Base) const {
  const auto &Buckets = Contents.getBuckets();
  uint64_t PrevHash = std::numeric_limits<uint64_t>::max();
  for (size_t i = 0, e = Buckets.size(); i < e; ++i) {
    for (auto *Hash : Buckets[i]) {
      uint32_t HashValue = Hash->HashValue;
      if (SkipIdenticalHashes && PrevHash == HashValue)
        continue;
      PrevHash = HashValue;
      Asm->OutStreamer->AddComment("Offset in Bucket " + Twine(i));
      Asm->EmitLabelDifference(Hash->Sym, Base, sizeof(uint32_t));
    }
  }
}

void AppleAccelTableWriter::Header::emit(AsmPrinter *Asm) const {
  Asm->OutStreamer->AddComment("Header Magic");
  Asm->emitInt32(Magic);
  Asm->OutStreamer->AddComment("Header Version");
  Asm->emitInt16(Version);
  Asm->OutStreamer->AddComment("Header Hash Function");
  Asm->emitInt16(HashFunction);
  Asm->OutStreamer->AddComment("Header Bucket Count");
  Asm->emitInt32(BucketCount);
  Asm->OutStreamer->AddComment("Header Hash Count");
  Asm->emitInt32(HashCount);
  Asm->OutStreamer->AddComment("Header Data Length");
  Asm->emitInt32(HeaderDataLength);
}

void AppleAccelTableWriter::HeaderData::emit(AsmPrinter *Asm) const {
  Asm->OutStreamer->AddComment("HeaderData Die Offset Base");
  Asm->emitInt32(DieOffsetBase);
  Asm->OutStreamer->AddComment("HeaderData Atom Count");
  Asm->emitInt32(Atoms.size());

  for (const Atom &A : Atoms) {
    Asm->OutStreamer->AddComment(dwarf::AtomTypeString(A.Type));
    Asm->emitInt16(A.Type);
    Asm->OutStreamer->AddComment(dwarf::FormEncodingString(A.Form));
    Asm->emitInt16(A.Form);
  }
}

void AppleAccelTableWriter::emitBuckets() const {
  const auto &Buckets = Contents.getBuckets();
  unsigned index = 0;
  for (size_t i = 0, e = Buckets.size(); i < e; ++i) {
    Asm->OutStreamer->AddComment("Bucket " + Twine(i));
    if (!Buckets[i].empty())
      Asm->emitInt32(index);
    else
      Asm->emitInt32(std::numeric_limits<uint32_t>::max());
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

void AppleAccelTableWriter::emitData() const {
  const auto &Buckets = Contents.getBuckets();
  for (size_t i = 0, e = Buckets.size(); i < e; ++i) {
    uint64_t PrevHash = std::numeric_limits<uint64_t>::max();
    for (auto &Hash : Buckets[i]) {
      // Terminate the previous entry if there is no hash collision with the
      // current one.
      if (PrevHash != std::numeric_limits<uint64_t>::max() &&
          PrevHash != Hash->HashValue)
        Asm->emitInt32(0);
      // Remember to emit the label for our offset.
      Asm->OutStreamer->EmitLabel(Hash->Sym);
      Asm->OutStreamer->AddComment(Hash->Name.getString());
      Asm->emitDwarfStringOffset(Hash->Name);
      Asm->OutStreamer->AddComment("Num DIEs");
      Asm->emitInt32(Hash->Values.size());
      for (const auto *V : Hash->Values)
        static_cast<const AppleAccelTableData *>(V)->emit(Asm);
      PrevHash = Hash->HashValue;
    }
    // Emit the final end marker for the bucket.
    if (!Buckets[i].empty())
      Asm->emitInt32(0);
  }
}

void AppleAccelTableWriter::emit() const {
  Header.emit(Asm);
  HeaderData.emit(Asm);
  emitBuckets();
  emitHashes();
  emitOffsets(SecBegin);
  emitData();
}

void Dwarf5AccelTableWriter::Header::emit(
    const Dwarf5AccelTableWriter &Ctx) const {
  assert(CompUnitCount > 0 && "Index must have at least one CU.");

  AsmPrinter *Asm = Ctx.Asm;
  Asm->OutStreamer->AddComment("Header: unit length");
  Asm->EmitLabelDifference(Ctx.ContributionEnd, Ctx.ContributionStart,
                           sizeof(uint32_t));
  Asm->OutStreamer->EmitLabel(Ctx.ContributionStart);
  Asm->OutStreamer->AddComment("Header: version");
  Asm->emitInt16(Version);
  Asm->OutStreamer->AddComment("Header: padding");
  Asm->emitInt16(Padding);
  Asm->OutStreamer->AddComment("Header: compilation unit count");
  Asm->emitInt32(CompUnitCount);
  Asm->OutStreamer->AddComment("Header: local type unit count");
  Asm->emitInt32(LocalTypeUnitCount);
  Asm->OutStreamer->AddComment("Header: foreign type unit count");
  Asm->emitInt32(ForeignTypeUnitCount);
  Asm->OutStreamer->AddComment("Header: bucket count");
  Asm->emitInt32(BucketCount);
  Asm->OutStreamer->AddComment("Header: name count");
  Asm->emitInt32(NameCount);
  Asm->OutStreamer->AddComment("Header: abbreviation table size");
  Asm->EmitLabelDifference(Ctx.AbbrevEnd, Ctx.AbbrevStart, sizeof(uint32_t));
  Asm->OutStreamer->AddComment("Header: augmentation string size");
  assert(AugmentationStringSize % 4 == 0);
  Asm->emitInt32(AugmentationStringSize);
  Asm->OutStreamer->AddComment("Header: augmentation string");
  Asm->OutStreamer->EmitBytes({AugmentationString, AugmentationStringSize});
}

DenseSet<uint32_t> Dwarf5AccelTableWriter::getUniqueTags() const {
  DenseSet<uint32_t> UniqueTags;
  for (auto &Bucket : Contents.getBuckets()) {
    for (auto *Hash : Bucket) {
      for (auto *Value : Hash->Values) {
        const DIE &Die =
            static_cast<const DWARF5AccelTableData *>(Value)->getDie();
        UniqueTags.insert(Die.getTag());
      }
    }
  }
  return UniqueTags;
}

SmallVector<Dwarf5AccelTableWriter::AttributeEncoding, 2>
Dwarf5AccelTableWriter::getUniformAttributes() const {
  SmallVector<AttributeEncoding, 2> UA;
  if (WriterInfo->getNumberOfCUs() > 1) {
    size_t LargestCUIndex = WriterInfo->getNumberOfCUs() - 1;
    dwarf::Form Form = DIEInteger::BestForm(/*IsSigned*/ false, LargestCUIndex);
    UA.push_back({dwarf::DW_IDX_compile_unit, Form});
  }
  UA.push_back({dwarf::DW_IDX_die_offset, dwarf::DW_FORM_ref4});
  return UA;
}

void Dwarf5AccelTableWriter::emitCUList() const {
  for (unsigned Idx = 0, CUs = WriterInfo->getNumberOfCUs(); Idx < CUs;
       ++Idx) {
    Asm->OutStreamer->AddComment("Compilation unit " + Twine(Idx));
    Asm->emitDwarfSymbolReference(WriterInfo->getLabelForCU(Idx));
  }
}

void Dwarf5AccelTableWriter::emitBuckets() const {
  uint32_t Index = 1;
  for (const auto &Bucket : enumerate(Contents.getBuckets())) {
    Asm->OutStreamer->AddComment("Bucket " + Twine(Bucket.index()));
    Asm->emitInt32(Bucket.value().empty() ? 0 : Index);
    Index += Bucket.value().size();
  }
}

void Dwarf5AccelTableWriter::emitStringOffsets() const {
  for (const auto &Bucket : enumerate(Contents.getBuckets())) {
    for (auto *Hash : Bucket.value()) {
      DwarfStringPoolEntryRef String = Hash->Name;
      Asm->OutStreamer->AddComment("String in Bucket " + Twine(Bucket.index()) +
                                   ": " + String.getString());
      Asm->emitDwarfStringOffset(String);
    }
  }
}

void Dwarf5AccelTableWriter::emitAbbrevs() const {
  Asm->OutStreamer->EmitLabel(AbbrevStart);
  for (const auto &Abbrev : Abbreviations) {
    Asm->OutStreamer->AddComment("Abbrev code");
    assert(Abbrev.first != 0);
    Asm->EmitULEB128(Abbrev.first);
    Asm->OutStreamer->AddComment(dwarf::TagString(Abbrev.first));
    Asm->EmitULEB128(Abbrev.first);
    for (const auto &AttrEnc : Abbrev.second) {
      Asm->EmitULEB128(AttrEnc.Index, dwarf::IndexString(AttrEnc.Index).data());
      Asm->EmitULEB128(AttrEnc.Form,
                       dwarf::FormEncodingString(AttrEnc.Form).data());
    }
    Asm->EmitULEB128(0, "End of abbrev");
    Asm->EmitULEB128(0, "End of abbrev");
  }
  Asm->EmitULEB128(0, "End of abbrev list");
  Asm->OutStreamer->EmitLabel(AbbrevEnd);
}

void Dwarf5AccelTableWriter::emitEntry(
    const DWARF5AccelTableData &Entry) const {
  auto AbbrevIt = Abbreviations.find(Entry.getDie().getTag());
  assert(AbbrevIt != Abbreviations.end() &&
         "Why wasn't this abbrev generated?");

  Asm->EmitULEB128(AbbrevIt->first, "Abbreviation code");
  for (const auto &AttrEnc : AbbrevIt->second) {
    Asm->OutStreamer->AddComment(dwarf::IndexString(AttrEnc.Index));
    switch (AttrEnc.Index) {
    case dwarf::DW_IDX_compile_unit: {
      const DIE *CUDie = Entry.getDie().getUnitDie();
      DIEInteger ID(WriterInfo->getUnqiueIDForUnitDie(CUDie));
      ID.EmitValue(Asm, AttrEnc.Form);
      break;
    }
    case dwarf::DW_IDX_die_offset:
      assert(AttrEnc.Form == dwarf::DW_FORM_ref4);
      Asm->emitInt32(Entry.getDie().getOffset());
      break;
    default:
      llvm_unreachable("Unexpected index attribute!");
    }
  }
}

void Dwarf5AccelTableWriter::emitData() const {
  Asm->OutStreamer->EmitLabel(EntryPool);
  for (auto &Bucket : Contents.getBuckets()) {
    for (auto *Hash : Bucket) {
      // Remember to emit the label for our offset.
      Asm->OutStreamer->EmitLabel(Hash->Sym);
      for (const auto *Value : Hash->Values)
        emitEntry(*static_cast<const DWARF5AccelTableData *>(Value));
      Asm->OutStreamer->AddComment("End of list: " + Hash->Name.getString());
      Asm->emitInt32(0);
    }
  }
}

Dwarf5AccelTableWriter::Dwarf5AccelTableWriter(
    AsmPrinter *Asm, const AccelTableBase &Contents,
    std::unique_ptr<AccelTableWriterInfo> WriterInfo)
    : AccelTableWriter(Asm, Contents, false),
      Header(WriterInfo->getNumberOfCUs(), Contents.getBucketCount(),
             Contents.getUniqueNameCount()),
      WriterInfo(std::move(WriterInfo)) {
  DenseSet<uint32_t> UniqueTags = getUniqueTags();
  SmallVector<AttributeEncoding, 2> UniformAttributes = getUniformAttributes();

  Abbreviations.reserve(UniqueTags.size());
  for (uint32_t Tag : UniqueTags)
    Abbreviations.try_emplace(Tag, UniformAttributes);
}

void Dwarf5AccelTableWriter::emit() const {
  Header.emit(*this);
  emitCUList();
  emitBuckets();
  emitHashes();
  emitStringOffsets();
  emitOffsets(EntryPool);
  emitAbbrevs();
  emitData();
  Asm->OutStreamer->EmitValueToAlignment(4, 0);
  Asm->OutStreamer->EmitLabel(ContributionEnd);
}

void llvm::emitAppleAccelTableImpl(AsmPrinter *Asm, AccelTableBase &Contents,
                                   StringRef Prefix, const MCSymbol *SecBegin,
                                   ArrayRef<AppleAccelTableData::Atom> Atoms) {
  Contents.finalize(Asm, Prefix);
  AppleAccelTableWriter(Asm, Contents, Atoms, SecBegin).emit();
}

void llvm::emitDWARF5AccelTable(
    AsmPrinter *Asm, AccelTable<DWARF5AccelTableData> &Contents,
    const DwarfDebug &DD, ArrayRef<std::unique_ptr<DwarfCompileUnit>> CUs) {
  auto WriterInfo = llvm::make_unique<DefaultAccelTableWriterInfo>(DD, CUs);
  emitDWARF5AccelTable(Asm, Contents, std::move(WriterInfo));
}

void llvm::emitDWARF5AccelTable(
    AsmPrinter *Asm, AccelTable<DWARF5AccelTableData> &Contents,
    std::unique_ptr<AccelTableWriterInfo> WriterInfo) {
  Contents.finalize(Asm, "names");
  Dwarf5AccelTableWriter(Asm, Contents, std::move(WriterInfo)).emit();
}

void AppleAccelTableOffsetData::emit(AsmPrinter *Asm) const {
  Asm->emitInt32(Die->getDebugSectionOffset());
}

void AppleAccelTableTypeData::emit(AsmPrinter *Asm) const {
  Asm->emitInt32(Die->getDebugSectionOffset());
  Asm->emitInt16(Die->getTag());
  Asm->emitInt8(0);
}

void AppleAccelTableStaticOffsetData::emit(AsmPrinter *Asm) const {
  Asm->emitInt32(Offset);
}

void AppleAccelTableStaticTypeData::emit(AsmPrinter *Asm) const {
  Asm->emitInt32(Offset);
  Asm->emitInt16(Tag);
  Asm->emitInt8(ObjCClassIsImplementation ? dwarf::DW_FLAG_type_implementation
                                          : 0);
  Asm->emitInt32(QualifiedNameHash);
}

#ifndef _MSC_VER
// The lines below are rejected by older versions (TBD) of MSVC.
constexpr AppleAccelTableData::Atom AppleAccelTableTypeData::Atoms[];
constexpr AppleAccelTableData::Atom AppleAccelTableOffsetData::Atoms[];
constexpr AppleAccelTableData::Atom AppleAccelTableStaticOffsetData::Atoms[];
constexpr AppleAccelTableData::Atom AppleAccelTableStaticTypeData::Atoms[];
#else
// FIXME: Erase this path once the minimum MSCV version has been bumped.
const SmallVector<AppleAccelTableData::Atom, 4>
    AppleAccelTableOffsetData::Atoms = {
        Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4)};
const SmallVector<AppleAccelTableData::Atom, 4> AppleAccelTableTypeData::Atoms =
    {Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4),
     Atom(dwarf::DW_ATOM_die_tag, dwarf::DW_FORM_data2),
     Atom(dwarf::DW_ATOM_type_flags, dwarf::DW_FORM_data1)};
const SmallVector<AppleAccelTableData::Atom, 4>
    AppleAccelTableStaticOffsetData::Atoms = {
        Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4)};
const SmallVector<AppleAccelTableData::Atom, 4>
    AppleAccelTableStaticTypeData::Atoms = {
        Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4),
        Atom(dwarf::DW_ATOM_die_tag, dwarf::DW_FORM_data2),
        Atom(5, dwarf::DW_FORM_data1), Atom(6, dwarf::DW_FORM_data4)};
#endif

#ifndef NDEBUG
void AppleAccelTableWriter::Header::print(raw_ostream &OS) const {
  OS << "Magic: " << format("0x%x", Magic) << "\n"
     << "Version: " << Version << "\n"
     << "Hash Function: " << HashFunction << "\n"
     << "Bucket Count: " << BucketCount << "\n"
     << "Header Data Length: " << HeaderDataLength << "\n";
}

void AppleAccelTableData::Atom::print(raw_ostream &OS) const {
  OS << "Type: " << dwarf::AtomTypeString(Type) << "\n"
     << "Form: " << dwarf::FormEncodingString(Form) << "\n";
}

void AppleAccelTableWriter::HeaderData::print(raw_ostream &OS) const {
  OS << "DIE Offset Base: " << DieOffsetBase << "\n";
  for (auto Atom : Atoms)
    Atom.print(OS);
}

void AppleAccelTableWriter::print(raw_ostream &OS) const {
  Header.print(OS);
  HeaderData.print(OS);
  Contents.print(OS);
  SecBegin->print(OS, nullptr);
}

void AccelTableBase::HashData::print(raw_ostream &OS) const {
  OS << "Name: " << Name.getString() << "\n";
  OS << "  Hash Value: " << format("0x%x", HashValue) << "\n";
  OS << "  Symbol: ";
  if (Sym)
    OS << *Sym;
  else
    OS << "<none>";
  OS << "\n";
  for (auto *Value : Values)
    Value->print(OS);
}

void AccelTableBase::print(raw_ostream &OS) const {
  // Print Content.
  OS << "Entries: \n";
  for (const auto &Entry : Entries) {
    OS << "Name: " << Entry.first() << "\n";
    for (auto *V : Entry.second.Values)
      V->print(OS);
  }

  OS << "Buckets and Hashes: \n";
  for (auto &Bucket : Buckets)
    for (auto &Hash : Bucket)
      Hash->print(OS);

  OS << "Data: \n";
  for (auto &E : Entries)
    E.second.print(OS);
}

void DWARF5AccelTableData::print(raw_ostream &OS) const {
  OS << "  Offset: " << Die.getOffset() << "\n";
  OS << "  Tag: " << dwarf::TagString(Die.getTag()) << "\n";
}

void AppleAccelTableOffsetData::print(raw_ostream &OS) const {
  OS << "  Offset: " << Die->getOffset() << "\n";
}

void AppleAccelTableTypeData::print(raw_ostream &OS) const {
  OS << "  Offset: " << Die->getOffset() << "\n";
  OS << "  Tag: " << dwarf::TagString(Die->getTag()) << "\n";
}

void AppleAccelTableStaticOffsetData::print(raw_ostream &OS) const {
  OS << "  Static Offset: " << Offset << "\n";
}

void AppleAccelTableStaticTypeData::print(raw_ostream &OS) const {
  OS << "  Static Offset: " << Offset << "\n";
  OS << "  QualifiedNameHash: " << format("%x\n", QualifiedNameHash) << "\n";
  OS << "  Tag: " << dwarf::TagString(Tag) << "\n";
  OS << "  ObjCClassIsImplementation: "
     << (ObjCClassIsImplementation ? "true" : "false");
  OS << "\n";
}
#endif
