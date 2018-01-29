//==- include/llvm/CodeGen/AccelTable.h - Accelerator Tables -----*- C++ -*-==//
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

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFACCELTABLE_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFACCELTABLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/CodeGen/DwarfStringPoolEntry.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DJB.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <vector>

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
/// The DWARFv5 accelerator table consists of zero or more name indices that
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

namespace llvm {

class AsmPrinter;

/// Representation of the header of an Apple accelerator table. This consists
/// of the fixed header and the header data. The latter contains the atoms
/// which define the columns of the table.
class AppleAccelTableHeader {
  struct Header {
    uint32_t Magic = MagicHash;
    uint16_t Version = 1;
    uint16_t HashFunction = dwarf::DW_hash_function_djb;
    uint32_t BucketCount = 0;
    uint32_t HashCount = 0;
    uint32_t HeaderDataLength;

    /// 'HASH' magic value to detect endianness.
    static const uint32_t MagicHash = 0x48415348;

    Header(uint32_t DataLength) : HeaderDataLength(DataLength) {}

#ifndef NDEBUG
    void print(raw_ostream &OS) const {
      OS << "Magic: " << format("0x%x", Magic) << "\n"
         << "Version: " << Version << "\n"
         << "Hash Function: " << HashFunction << "\n"
         << "Bucket Count: " << BucketCount << "\n"
         << "Header Data Length: " << HeaderDataLength << "\n";
    }

    void dump() const { print(dbgs()); }
#endif
  };

public:
  /// An Atom defines the form of the data in the accelerator table.
  /// Conceptually it is a column in the accelerator consisting of a type and a
  /// specification of the form of its data.
  struct Atom {
    /// Atom Type.
    const uint16_t Type;
    /// DWARF Form.
    const uint16_t Form;

    constexpr Atom(uint16_t Type, uint16_t Form) : Type(Type), Form(Form) {}

#ifndef NDEBUG
    void print(raw_ostream &OS) const {
      OS << "Type: " << dwarf::AtomTypeString(Type) << "\n"
         << "Form: " << dwarf::FormEncodingString(Form) << "\n";
    }

    void dump() const { print(dbgs()); }
#endif
  };

private:
  /// The HeaderData describes the structure of the accelerator table through a
  /// list of Atoms.
  struct HeaderData {
    /// In the case of data that is referenced via DW_FORM_ref_* the offset
    /// base is used to describe the offset for all forms in the list of atoms.
    uint32_t DieOffsetBase;

    const SmallVector<Atom, 4> Atoms;

#ifndef _MSC_VER
    // See the `static constexpr` below why we need an alternative
    // implementation for MSVC.
    HeaderData(ArrayRef<Atom> AtomList, uint32_t Offset = 0)
        : DieOffsetBase(Offset), Atoms(AtomList.begin(), AtomList.end()) {}
#else
    // FIXME: Erase this path once the minimum MSCV version has been bumped.
    HeaderData(const SmallVectorImpl<Atom> &Atoms, uint32_t Offset = 0)
        : DieOffsetBase(Offset), Atoms(Atoms.begin(), Atoms.end()) {}
#endif

#ifndef NDEBUG
    void print(raw_ostream &OS) const {
      OS << "DIE Offset Base: " << DieOffsetBase << "\n";
      for (auto Atom : Atoms)
        Atom.print(OS);
    }

    void dump() const { print(dbgs()); }
#endif
  };

  Header Header;
  HeaderData HeaderData;

public:
  /// The length of the header data is always going to be 4 + 4 + 4*NumAtoms.
#ifndef _MSC_VER
  // See the `static constexpr` below why we need an alternative implementation
  // for MSVC.
  AppleAccelTableHeader(ArrayRef<AppleAccelTableHeader::Atom> Atoms)
      : Header(8 + (Atoms.size() * 4)), HeaderData(Atoms) {}
#else
  // FIXME: Erase this path once the minimum MSCV version has been bumped.
  AppleAccelTableHeader(const SmallVectorImpl<Atom> &Atoms)
      : Header(8 + (Atoms.size() * 4)), HeaderData(Atoms) {}
#endif

  /// Update header with hash and bucket count.
  void setBucketAndHashCount(uint32_t HashCount);

  uint32_t getHashCount() const { return Header.HashCount; }
  uint32_t getBucketCount() const { return Header.BucketCount; }

  /// Emits the header via the AsmPrinter.
  void emit(AsmPrinter *);

#ifndef NDEBUG
  void print(raw_ostream &OS) const {
    Header.print(OS);
    HeaderData.print(OS);
  }

  void dump() const { print(dbgs()); }
#endif
};

/// Interface which the different types of accelerator table data have to
/// conform.
class AppleAccelTableData {
public:
  virtual ~AppleAccelTableData() = default;

  virtual void emit(AsmPrinter *Asm) const = 0;

  bool operator<(const AppleAccelTableData &Other) const {
    return order() < Other.order();
  }

#ifndef NDEBUG
  virtual void print(raw_ostream &OS) const = 0;
#endif
protected:
  virtual uint64_t order() const = 0;
};

/// Apple-style accelerator table base class.
class AppleAccelTableBase {
protected:
  struct DataArray {
    DwarfStringPoolEntryRef Name;
    std::vector<AppleAccelTableData *> Values;
  };

  friend struct HashData;

  struct HashData {
    StringRef Str;
    uint32_t HashValue;
    MCSymbol *Sym;
    DataArray &Data;

    HashData(StringRef S, DataArray &Data) : Str(S), Data(Data) {
      HashValue = djbHash(S);
    }

#ifndef NDEBUG
    void print(raw_ostream &OS) {
      OS << "Name: " << Str << "\n";
      OS << "  Hash Value: " << format("0x%x", HashValue) << "\n";
      OS << "  Symbol: ";
      if (Sym)
        OS << *Sym;
      else
        OS << "<none>";
      OS << "\n";
      for (auto *Value : Data.Values)
        Value->print(OS);
    }

    void dump() { print(dbgs()); }
#endif
  };

  /// Allocator for HashData and Values.
  BumpPtrAllocator Allocator;

  /// Header containing both the header and header data.
  AppleAccelTableHeader Header;

  std::vector<HashData *> Data;

  using StringEntries = StringMap<DataArray, BumpPtrAllocator &>;
  StringEntries Entries;

  using HashList = std::vector<HashData *>;
  HashList Hashes;

  using BucketList = std::vector<HashList>;
  BucketList Buckets;

#ifndef _MSC_VER
  // See the `static constexpr` below why we need an alternative implementation
  // for MSVC.
  AppleAccelTableBase(ArrayRef<AppleAccelTableHeader::Atom> Atoms)
      : Header(Atoms), Entries(Allocator) {}
#else
  // FIXME: Erase this path once the minimum MSCV version has been bumped.
  AppleAccelTableBase(const SmallVectorImpl<AppleAccelTableHeader::Atom> &Atoms)
      : Header(Atoms), Entries(Allocator) {}
#endif

private:
  /// Emits the header for the table via the AsmPrinter.
  void emitHeader(AsmPrinter *Asm);

  /// Helper function to compute the number of buckets needed based on the
  /// number of unique hashes.
  void computeBucketCount();

  /// Walk through and emit the buckets for the table. Each index is an offset
  /// into the list of hashes.
  void emitBuckets(AsmPrinter *);

  /// Walk through the buckets and emit the individual hashes for each bucket.
  void emitHashes(AsmPrinter *);

  /// Walk through the buckets and emit the individual offsets for each element
  /// in each bucket. This is done via a symbol subtraction from the beginning
  /// of the section. The non-section symbol will be output later when we emit
  /// the actual data.
  void emitOffsets(AsmPrinter *, const MCSymbol *);

  /// Walk through the buckets and emit the full data for each element in the
  /// bucket. For the string case emit the dies and the various offsets.
  /// Terminate each HashData bucket with 0.
  void emitData(AsmPrinter *);

public:
  void finalizeTable(AsmPrinter *, StringRef);

  void emit(AsmPrinter *Asm, const MCSymbol *SecBegin) {
    emitHeader(Asm);
    emitBuckets(Asm);
    emitHashes(Asm);
    emitOffsets(Asm, SecBegin);
    emitData(Asm);
  }

#ifndef NDEBUG
  void print(raw_ostream &OS) const {
    // Print Header.
    Header.print(OS);

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
    for (auto &D : Data)
      D->print(OS);
  }
  void dump() const { print(dbgs()); }
#endif
};

template <typename AppleAccelTableDataT>
class AppleAccelTable : public AppleAccelTableBase {
public:
  AppleAccelTable() : AppleAccelTableBase(AppleAccelTableDataT::Atoms) {}
  AppleAccelTable(const AppleAccelTable &) = delete;
  AppleAccelTable &operator=(const AppleAccelTable &) = delete;

  template <class... Types>
  void addName(DwarfStringPoolEntryRef Name, Types... Args);
};

template <typename AppleAccelTableDataT>
template <class... Types>
void AppleAccelTable<AppleAccelTableDataT>::addName(
    DwarfStringPoolEntryRef Name, Types... Args) {
  assert(Data.empty() && "Already finalized!");
  // If the string is in the list already then add this die to the list
  // otherwise add a new one.
  DataArray &DA = Entries[Name.getString()];
  assert(!DA.Name || DA.Name == Name);
  DA.Name = Name;
  DA.Values.push_back(new (Allocator) AppleAccelTableDataT(Args...));
}

/// Accelerator table data implementation for simple accelerator tables with
/// just a DIE reference.
class AppleAccelTableOffsetData : public AppleAccelTableData {
public:
  AppleAccelTableOffsetData(const DIE *D) : Die(D) {}

  void emit(AsmPrinter *Asm) const override;

#ifndef _MSC_VER
  // The line below is rejected by older versions (TBD) of MSVC.
  static constexpr AppleAccelTableHeader::Atom Atoms[] = {
      AppleAccelTableHeader::Atom(dwarf::DW_ATOM_die_offset,
                                  dwarf::DW_FORM_data4)};
#else
  // FIXME: Erase this path once the minimum MSCV version has been bumped.
  static const SmallVector<AppleAccelTableHeader::Atom, 4> Atoms;
#endif

#ifndef NDEBUG
  void print(raw_ostream &OS) const override {
    OS << "  Offset: " << Die->getOffset() << "\n";
  }

#endif
protected:
  uint64_t order() const override { return Die->getOffset(); }

  const DIE *Die;
};

/// Accelerator table data implementation for type accelerator tables.
class AppleAccelTableTypeData : public AppleAccelTableOffsetData {
public:
  AppleAccelTableTypeData(const DIE *D) : AppleAccelTableOffsetData(D) {}

  void emit(AsmPrinter *Asm) const override;

#ifndef _MSC_VER
  // The line below is rejected by older versions (TBD) of MSVC.
  static constexpr AppleAccelTableHeader::Atom Atoms[] = {
      AppleAccelTableHeader::Atom(dwarf::DW_ATOM_die_offset,
                                  dwarf::DW_FORM_data4),
      AppleAccelTableHeader::Atom(dwarf::DW_ATOM_die_tag, dwarf::DW_FORM_data2),
      AppleAccelTableHeader::Atom(dwarf::DW_ATOM_type_flags,
                                  dwarf::DW_FORM_data1)};
#else
  // FIXME: Erase this path once the minimum MSCV version has been bumped.
  static const SmallVector<AppleAccelTableHeader::Atom, 4> Atoms;
#endif

#ifndef NDEBUG
  void print(raw_ostream &OS) const override {
    OS << "  Offset: " << Die->getOffset() << "\n";
    OS << "  Tag: " << dwarf::TagString(Die->getTag()) << "\n";
  }
#endif
};

/// Accelerator table data implementation for simple accelerator tables with
/// a DIE offset but no actual DIE pointer.
class AppleAccelTableStaticOffsetData : public AppleAccelTableData {
public:
  AppleAccelTableStaticOffsetData(uint32_t Offset) : Offset(Offset) {}

  void emit(AsmPrinter *Asm) const override;

#ifndef _MSC_VER
  // The line below is rejected by older versions (TBD) of MSVC.
  static constexpr AppleAccelTableHeader::Atom Atoms[] = {
      AppleAccelTableHeader::Atom(dwarf::DW_ATOM_die_offset,
                                  dwarf::DW_FORM_data4)};
#else
  // FIXME: Erase this path once the minimum MSCV version has been bumped.
  static const SmallVector<AppleAccelTableHeader::Atom, 4> Atoms;
#endif

#ifndef NDEBUG
  void print(raw_ostream &OS) const override {
    OS << "  Static Offset: " << Offset << "\n";
  }

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

#ifndef _MSC_VER
  // The line below is rejected by older versions (TBD) of MSVC.
  static constexpr AppleAccelTableHeader::Atom Atoms[] = {
      AppleAccelTableHeader::Atom(dwarf::DW_ATOM_die_offset,
                                  dwarf::DW_FORM_data4),
      AppleAccelTableHeader::Atom(dwarf::DW_ATOM_die_tag, dwarf::DW_FORM_data2),
      AppleAccelTableHeader::Atom(5, dwarf::DW_FORM_data1),
      AppleAccelTableHeader::Atom(6, dwarf::DW_FORM_data4)};
#else
  // FIXME: Erase this path once the minimum MSCV version has been bumped.
  static const SmallVector<AppleAccelTableHeader::Atom, 4> Atoms;
#endif

#ifndef NDEBUG
  void print(raw_ostream &OS) const override {
    OS << "  Static Offset: " << Offset << "\n";
    OS << "  QualifiedNameHash: " << format("%x\n", QualifiedNameHash) << "\n";
    OS << "  Tag: " << dwarf::TagString(Tag) << "\n";
    OS << "  ObjCClassIsImplementation: "
       << (ObjCClassIsImplementation ? "true" : "false");
    OS << "\n";
  }
#endif
protected:
  uint64_t order() const override { return Offset; }

  uint32_t QualifiedNameHash;
  uint16_t Tag;
  bool ObjCClassIsImplementation;
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_DWARFACCELTABLE_H
