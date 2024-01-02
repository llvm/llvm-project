//==- include/llvm/CodeGen/AccelTable.h - Accelerator Tables -----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains support for writing accelerator tables.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ACCELTABLE_H
#define LLVM_CODEGEN_ACCELTABLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/CodeGen/DwarfStringPoolEntry.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DJB.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <variant>
#include <vector>

/// \file
/// The DWARF and Apple accelerator tables are an indirect hash table optimized
/// for null lookup rather than access to known data. The Apple accelerator
/// tables are a precursor of the newer DWARF v5 accelerator tables. Both
/// formats share common design ideas.
///
/// The Apple accelerator table are output into an on-disk format that looks
/// like this:
///
/// .------------------.
/// |  HEADER          |
/// |------------------|
/// |  BUCKETS         |
/// |------------------|
/// |  HASHES          |
/// |------------------|
/// |  OFFSETS         |
/// |------------------|
/// |  DATA            |
/// `------------------'
///
/// The header contains a magic number, version, type of hash function,
/// the number of buckets, total number of hashes, and room for a special struct
/// of data and the length of that struct.
///
/// The buckets contain an index (e.g. 6) into the hashes array. The hashes
/// section contains all of the 32-bit hash values in contiguous memory, and the
/// offsets contain the offset into the data area for the particular hash.
///
/// For a lookup example, we could hash a function name and take it modulo the
/// number of buckets giving us our bucket. From there we take the bucket value
/// as an index into the hashes table and look at each successive hash as long
/// as the hash value is still the same modulo result (bucket value) as earlier.
/// If we have a match we look at that same entry in the offsets table and grab
/// the offset in the data for our final match.
///
/// The DWARF v5 accelerator table consists of zero or more name indices that
/// are output into an on-disk format that looks like this:
///
/// .------------------.
/// |  HEADER          |
/// |------------------|
/// |  CU LIST         |
/// |------------------|
/// |  LOCAL TU LIST   |
/// |------------------|
/// |  FOREIGN TU LIST |
/// |------------------|
/// |  HASH TABLE      |
/// |------------------|
/// |  NAME TABLE      |
/// |------------------|
/// |  ABBREV TABLE    |
/// |------------------|
/// |  ENTRY POOL      |
/// `------------------'
///
/// For the full documentation please refer to the DWARF 5 standard.
///
///
/// This file defines the class template AccelTable, which is represents an
/// abstract view of an Accelerator table, without any notion of an on-disk
/// layout. This class is parameterized by an entry type, which should derive
/// from AccelTableData. This is the type of individual entries in the table,
/// and it should store the data necessary to emit them. AppleAccelTableData is
/// the base class for Apple Accelerator Table entries, which have a uniform
/// structure based on a sequence of Atoms. There are different sub-classes
/// derived from AppleAccelTable, which differ in the set of Atoms and how they
/// obtain their values.
///
/// An Apple Accelerator Table can be serialized by calling emitAppleAccelTable
/// function.

namespace llvm {

class AsmPrinter;
class DwarfDebug;
class DwarfTypeUnit;
class MCSymbol;
class raw_ostream;

/// Interface which the different types of accelerator table data have to
/// conform. It serves as a base class for different values of the template
/// argument of the AccelTable class template.
class AccelTableData {
public:
  virtual ~AccelTableData() = default;

  bool operator<(const AccelTableData &Other) const {
    return order() < Other.order();
  }

    // Subclasses should implement:
    // static uint32_t hash(StringRef Name);

#ifndef NDEBUG
  virtual void print(raw_ostream &OS) const = 0;
#endif
protected:
  virtual uint64_t order() const = 0;
};

/// A base class holding non-template-dependant functionality of the AccelTable
/// class. Clients should not use this class directly but rather instantiate
/// AccelTable with a type derived from AccelTableData.
class AccelTableBase {
public:
  using HashFn = uint32_t(StringRef);

  /// Represents a group of entries with identical name (and hence, hash value).
  struct HashData {
    DwarfStringPoolEntryRef Name;
    uint32_t HashValue;
    std::vector<AccelTableData *> Values;
    MCSymbol *Sym;

#ifndef NDEBUG
    void print(raw_ostream &OS) const;
    void dump() const { print(dbgs()); }
#endif
  };
  using HashList = std::vector<HashData *>;
  using BucketList = std::vector<HashList>;

protected:
  /// Allocator for HashData and Values.
  BumpPtrAllocator Allocator;

  using StringEntries = MapVector<StringRef, HashData>;
  StringEntries Entries;

  HashFn *Hash;
  uint32_t BucketCount = 0;
  uint32_t UniqueHashCount = 0;

  HashList Hashes;
  BucketList Buckets;

  void computeBucketCount();

  AccelTableBase(HashFn *Hash) : Hash(Hash) {}

public:
  void finalize(AsmPrinter *Asm, StringRef Prefix);
  ArrayRef<HashList> getBuckets() const { return Buckets; }
  uint32_t getBucketCount() const { return BucketCount; }
  uint32_t getUniqueHashCount() const { return UniqueHashCount; }
  uint32_t getUniqueNameCount() const { return Entries.size(); }

#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  void dump() const { print(dbgs()); }
#endif

  AccelTableBase(const AccelTableBase &) = delete;
  void operator=(const AccelTableBase &) = delete;
};

/// This class holds an abstract representation of an Accelerator Table,
/// consisting of a sequence of buckets, each bucket containint a sequence of
/// HashData entries. The class is parameterized by the type of entries it
/// holds. The type template parameter also defines the hash function to use for
/// hashing names.
template <typename DataT> class AccelTable : public AccelTableBase {
public:
  AccelTable() : AccelTableBase(DataT::hash) {}

  template <typename... Types>
  void addName(DwarfStringPoolEntryRef Name, Types &&... Args);
  void clear() { Entries.clear(); }
  void addEntries(AccelTable<DataT> &Table);
  const StringEntries getEntries() const { return Entries; }
};

template <typename AccelTableDataT>
template <typename... Types>
void AccelTable<AccelTableDataT>::addName(DwarfStringPoolEntryRef Name,
                                          Types &&... Args) {
  assert(Buckets.empty() && "Already finalized!");
  // If the string is in the list already then add this die to the list
  // otherwise add a new one.
  auto &It = Entries[Name.getString()];
  if (It.Values.empty()) {
    It.Name = Name;
    It.HashValue = Hash(Name.getString());
  }
  It.Values.push_back(new (Allocator)
                          AccelTableDataT(std::forward<Types>(Args)...));
}

/// A base class for different implementations of Data classes for Apple
/// Accelerator Tables. The columns in the table are defined by the static Atoms
/// variable defined on the subclasses.
class AppleAccelTableData : public AccelTableData {
public:
  /// An Atom defines the form of the data in an Apple accelerator table.
  /// Conceptually it is a column in the accelerator consisting of a type and a
  /// specification of the form of its data.
  struct Atom {
    /// Atom Type.
    const uint16_t Type;
    /// DWARF Form.
    const uint16_t Form;

    constexpr Atom(uint16_t Type, uint16_t Form) : Type(Type), Form(Form) {}

#ifndef NDEBUG
    void print(raw_ostream &OS) const;
    void dump() const { print(dbgs()); }
#endif
  };
  // Subclasses should define:
  // static constexpr Atom Atoms[];

  virtual void emit(AsmPrinter *Asm) const = 0;

  static uint32_t hash(StringRef Buffer) { return djbHash(Buffer); }
};

/// The Data class implementation for DWARF v5 accelerator table. Unlike the
/// Apple Data classes, this class is just a DIE wrapper, and does not know to
/// serialize itself. The complete serialization logic is in the
/// emitDWARF5AccelTable function.
class DWARF5AccelTableData : public AccelTableData {
public:
  struct AttributeEncoding {
    dwarf::Index Index;
    dwarf::Form Form;
  };

  static uint32_t hash(StringRef Name) { return caseFoldingDjbHash(Name); }

  DWARF5AccelTableData(const DIE &Die, const uint32_t UnitID,
                       const bool IsTU = false);
  DWARF5AccelTableData(const uint64_t DieOffset, const unsigned DieTag,
                       const unsigned UnitID, const bool IsTU = false)
      : OffsetVal(DieOffset), DieTag(DieTag), UnitID(UnitID), IsTU(IsTU) {}

#ifndef NDEBUG
  void print(raw_ostream &OS) const override;
#endif

  uint64_t getDieOffset() const {
    assert(isNormalized() && "Accessing DIE Offset before normalizing.");
    return std::get<uint64_t>(OffsetVal);
  }
  unsigned getDieTag() const { return DieTag; }
  unsigned getUnitID() const { return UnitID; }
  bool isTU() const { return IsTU; }
  void normalizeDIEToOffset() {
    assert(!isNormalized() && "Accessing offset after normalizing.");
    OffsetVal = std::get<const DIE *>(OffsetVal)->getOffset();
  }
  bool isNormalized() const {
    return std::holds_alternative<uint64_t>(OffsetVal);
  }

protected:
  std::variant<const DIE *, uint64_t> OffsetVal;
  uint32_t DieTag : 16;
  uint32_t UnitID : 15;
  uint32_t IsTU : 1;

  uint64_t order() const override { return getDieOffset(); }
};

struct TypeUnitMetaInfo {
  // Symbol for start of the TU section or signature if this is SplitDwarf.
  std::variant<MCSymbol *, uint64_t> LabelOrSignature;
  // Unique ID of Type Unit.
  unsigned UniqueID;
};
using TUVectorTy = SmallVector<TypeUnitMetaInfo, 1>;
class DWARF5AccelTable : public AccelTable<DWARF5AccelTableData> {
  // Symbols to start of all the TU sections that were generated.
  TUVectorTy TUSymbolsOrHashes;

public:
  struct UnitIndexAndEncoding {
    unsigned Index;
    DWARF5AccelTableData::AttributeEncoding Encoding;
  };
  /// Returns type units that were constructed.
  const TUVectorTy &getTypeUnitsSymbols() { return TUSymbolsOrHashes; }
  /// Add a type unit start symbol.
  void addTypeUnitSymbol(DwarfTypeUnit &U);
  /// Add a type unit Signature.
  void addTypeUnitSignature(DwarfTypeUnit &U);
  /// Convert DIE entries to explicit offset.
  /// Needs to be called after DIE offsets are computed.
  void convertDieToOffset() {
    for (auto &Entry : Entries) {
      for (AccelTableData *Value : Entry.second.Values) {
        DWARF5AccelTableData *Data = static_cast<DWARF5AccelTableData *>(Value);
        // For TU we normalize as each Unit is emitted.
        // So when this is invoked after CU construction we will be in mixed
        // state.
        if (!Data->isNormalized())
          Data->normalizeDIEToOffset();
      }
    }
  }

  void addTypeEntries(DWARF5AccelTable &Table) {
    for (auto &Entry : Table.getEntries()) {
      for (AccelTableData *Value : Entry.second.Values) {
        DWARF5AccelTableData *Data = static_cast<DWARF5AccelTableData *>(Value);
        addName(Entry.second.Name, Data->getDieOffset(), Data->getDieTag(),
                Data->getUnitID(), true);
      }
    }
  }
};

void emitAppleAccelTableImpl(AsmPrinter *Asm, AccelTableBase &Contents,
                             StringRef Prefix, const MCSymbol *SecBegin,
                             ArrayRef<AppleAccelTableData::Atom> Atoms);

/// Emit an Apple Accelerator Table consisting of entries in the specified
/// AccelTable. The DataT template parameter should be derived from
/// AppleAccelTableData.
template <typename DataT>
void emitAppleAccelTable(AsmPrinter *Asm, AccelTable<DataT> &Contents,
                         StringRef Prefix, const MCSymbol *SecBegin) {
  static_assert(std::is_convertible<DataT *, AppleAccelTableData *>::value);
  emitAppleAccelTableImpl(Asm, Contents, Prefix, SecBegin, DataT::Atoms);
}

void emitDWARF5AccelTable(AsmPrinter *Asm, DWARF5AccelTable &Contents,
                          const DwarfDebug &DD,
                          ArrayRef<std::unique_ptr<DwarfCompileUnit>> CUs);

/// Emit a DWARFv5 Accelerator Table consisting of entries in the specified
/// AccelTable. The \p CUs contains either symbols keeping offsets to the
/// start of compilation unit, either offsets to the start of compilation
/// unit themselves.
void emitDWARF5AccelTable(
    AsmPrinter *Asm, DWARF5AccelTable &Contents,
    ArrayRef<std::variant<MCSymbol *, uint64_t>> CUs,
    llvm::function_ref<std::optional<DWARF5AccelTable::UnitIndexAndEncoding>(
        const DWARF5AccelTableData &)>
        getIndexForEntry);

/// Accelerator table data implementation for simple Apple accelerator tables
/// with just a DIE reference.
class AppleAccelTableOffsetData : public AppleAccelTableData {
public:
  AppleAccelTableOffsetData(const DIE &D) : Die(D) {}

  void emit(AsmPrinter *Asm) const override;

  static constexpr Atom Atoms[] = {
      Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4)};

#ifndef NDEBUG
  void print(raw_ostream &OS) const override;
#endif
protected:
  uint64_t order() const override { return Die.getOffset(); }

  const DIE &Die;
};

/// Accelerator table data implementation for Apple type accelerator tables.
class AppleAccelTableTypeData : public AppleAccelTableOffsetData {
public:
  AppleAccelTableTypeData(const DIE &D) : AppleAccelTableOffsetData(D) {}

  void emit(AsmPrinter *Asm) const override;

  static constexpr Atom Atoms[] = {
      Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4),
      Atom(dwarf::DW_ATOM_die_tag, dwarf::DW_FORM_data2),
      Atom(dwarf::DW_ATOM_type_flags, dwarf::DW_FORM_data1)};

#ifndef NDEBUG
  void print(raw_ostream &OS) const override;
#endif
};

/// Accelerator table data implementation for simple Apple accelerator tables
/// with a DIE offset but no actual DIE pointer.
class AppleAccelTableStaticOffsetData : public AppleAccelTableData {
public:
  AppleAccelTableStaticOffsetData(uint32_t Offset) : Offset(Offset) {}

  void emit(AsmPrinter *Asm) const override;

  static constexpr Atom Atoms[] = {
      Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4)};

#ifndef NDEBUG
  void print(raw_ostream &OS) const override;
#endif
protected:
  uint64_t order() const override { return Offset; }

  uint32_t Offset;
};

/// Accelerator table data implementation for type accelerator tables with
/// a DIE offset but no actual DIE pointer.
class AppleAccelTableStaticTypeData : public AppleAccelTableStaticOffsetData {
public:
  AppleAccelTableStaticTypeData(uint32_t Offset, uint16_t Tag,
                                bool ObjCClassIsImplementation,
                                uint32_t QualifiedNameHash)
      : AppleAccelTableStaticOffsetData(Offset),
        QualifiedNameHash(QualifiedNameHash), Tag(Tag),
        ObjCClassIsImplementation(ObjCClassIsImplementation) {}

  void emit(AsmPrinter *Asm) const override;

  static constexpr Atom Atoms[] = {
      Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4),
      Atom(dwarf::DW_ATOM_die_tag, dwarf::DW_FORM_data2),
      Atom(5, dwarf::DW_FORM_data1), Atom(6, dwarf::DW_FORM_data4)};

#ifndef NDEBUG
  void print(raw_ostream &OS) const override;
#endif
protected:
  uint64_t order() const override { return Offset; }

  uint32_t QualifiedNameHash;
  uint16_t Tag;
  bool ObjCClassIsImplementation;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_ACCELTABLE_H
