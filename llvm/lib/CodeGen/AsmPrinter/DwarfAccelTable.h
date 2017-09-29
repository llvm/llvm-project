//==- llvm/CodeGen/DwarfAccelTable.h - Dwarf Accelerator Tables --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf accelerator tables.
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <vector>

// The dwarf accelerator tables are an indirect hash table optimized
// for null lookup rather than access to known data. They are output into
// an on-disk format that looks like this:
//
// .-------------.
// |  HEADER     |
// |-------------|
// |  BUCKETS    |
// |-------------|
// |  HASHES     |
// |-------------|
// |  OFFSETS    |
// |-------------|
// |  DATA       |
// `-------------'
//
// where the header contains a magic number, version, type of hash function,
// the number of buckets, total number of hashes, and room for a special
// struct of data and the length of that struct.
//
// The buckets contain an index (e.g. 6) into the hashes array. The hashes
// section contains all of the 32-bit hash values in contiguous memory, and
// the offsets contain the offset into the data area for the particular
// hash.
//
// For a lookup example, we could hash a function name and take it modulo the
// number of buckets giving us our bucket. From there we take the bucket value
// as an index into the hashes table and look at each successive hash as long
// as the hash value is still the same modulo result (bucket value) as earlier.
// If we have a match we look at that same entry in the offsets table and
// grab the offset in the data for our final match.

namespace llvm {

class AsmPrinter;
class DwarfDebug;

class DwarfAccelTable {
  // Helper function to compute the number of buckets needed based on
  // the number of unique hashes.
  void ComputeBucketCount();

  struct TableHeader {
    uint32_t magic = MagicHash; // 'HASH' magic value to allow endian detection
    uint16_t version = 1;       // Version number.
    uint16_t hash_function = dwarf::DW_hash_function_djb;
    // The hash function enumeration that was used.
    uint32_t bucket_count = 0;  // The number of buckets in this hash table.
    uint32_t hashes_count = 0;  // The total number of unique hash values
                                // and hash data offsets in this table.
    uint32_t header_data_len;   // The bytes to skip to get to the hash
                                // indexes (buckets) for correct alignment.
    // Also written to disk is the implementation specific header data.

    static const uint32_t MagicHash = 0x48415348;

    TableHeader(uint32_t data_len) : header_data_len(data_len) {}

#ifndef NDEBUG
    void print(raw_ostream &OS) {
      OS << "Magic: " << format("0x%x", magic) << "\n"
         << "Version: " << version << "\n"
         << "Hash Function: " << hash_function << "\n"
         << "Bucket Count: " << bucket_count << "\n"
         << "Header Data Length: " << header_data_len << "\n";
    }

    void dump() { print(dbgs()); }
#endif
  };

public:
  // The HeaderData describes the form of each set of data. In general this
  // is as a list of atoms (atom_count) where each atom contains a type
  // (AtomType type) of data, and an encoding form (form). In the case of
  // data that is referenced via DW_FORM_ref_* the die_offset_base is
  // used to describe the offset for all forms in the list of atoms.
  // This also serves as a public interface of sorts.
  // When written to disk this will have the form:
  //
  // uint32_t die_offset_base
  // uint32_t atom_count
  // atom_count Atoms

  // Make these public so that they can be used as a general interface to
  // the class.
  struct Atom {
    uint16_t type; // enum AtomType
    uint16_t form; // DWARF DW_FORM_ defines

    constexpr Atom(uint16_t type, uint16_t form) : type(type), form(form) {}

#ifndef NDEBUG
    void print(raw_ostream &OS) {
      OS << "Type: " << dwarf::AtomTypeString(type) << "\n"
         << "Form: " << dwarf::FormEncodingString(form) << "\n";
    }

    void dump() { print(dbgs()); }
#endif
  };

private:
  struct TableHeaderData {
    uint32_t die_offset_base;
    SmallVector<Atom, 3> Atoms;

    TableHeaderData(ArrayRef<Atom> AtomList, uint32_t offset = 0)
        : die_offset_base(offset), Atoms(AtomList.begin(), AtomList.end()) {}

#ifndef NDEBUG
    void print(raw_ostream &OS) {
      OS << "die_offset_base: " << die_offset_base << "\n";
      for (size_t i = 0; i < Atoms.size(); i++)
        Atoms[i].print(OS);
    }

    void dump() { print(dbgs()); }
#endif
  };

  // The data itself consists of a str_offset, a count of the DIEs in the
  // hash and the offsets to the DIEs themselves.
  // On disk each data section is ended with a 0 KeyType as the end of the
  // hash chain.
  // On output this looks like:
  // uint32_t str_offset
  // uint32_t hash_data_count
  // HashData[hash_data_count]
public:
  struct HashDataContents {
    const DIE *Die;   // Offsets
    char Flags; // Specific flags to output

    HashDataContents(const DIE *D, char Flags) : Die(D), Flags(Flags) {}

#ifndef NDEBUG
    void print(raw_ostream &OS) const {
      OS << "  Offset: " << Die->getOffset() << "\n"
         << "  Tag: " << dwarf::TagString(Die->getTag()) << "\n"
         << "  Flags: " << Flags << "\n";
    }
#endif
  };

private:
  // String Data
  struct DataArray {
    DwarfStringPoolEntryRef Name;
    std::vector<HashDataContents *> Values;
  };

  friend struct HashData;

  struct HashData {
    StringRef Str;
    uint32_t HashValue;
    MCSymbol *Sym;
    DwarfAccelTable::DataArray &Data; // offsets

    HashData(StringRef S, DwarfAccelTable::DataArray &Data)
        : Str(S), Data(Data) {
      HashValue = dwarf::djbHash(S);
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
      for (HashDataContents *C : Data.Values) {
        OS << "  Offset: " << C->Die->getOffset() << "\n";
        OS << "  Tag: " << dwarf::TagString(C->Die->getTag()) << "\n";
        OS << "  Flags: " << C->Flags << "\n";
      }
    }

    void dump() { print(dbgs()); }
#endif
  };

  // Internal Functions
  void EmitHeader(AsmPrinter *);
  void EmitBuckets(AsmPrinter *);
  void EmitHashes(AsmPrinter *);
  void emitOffsets(AsmPrinter *, const MCSymbol *);
  void EmitData(AsmPrinter *, DwarfDebug *D);

  // Allocator for HashData and HashDataContents.
  BumpPtrAllocator Allocator;

  // Output Variables
  TableHeader Header;
  TableHeaderData HeaderData;
  std::vector<HashData *> Data;

  using StringEntries = StringMap<DataArray, BumpPtrAllocator &>;

  StringEntries Entries;

  // Buckets/Hashes/Offsets
  using HashList = std::vector<HashData *>;
  using BucketList = std::vector<HashList>;
  BucketList Buckets;
  HashList Hashes;

  // Public Implementation
public:
  DwarfAccelTable(ArrayRef<DwarfAccelTable::Atom>);
  DwarfAccelTable(const DwarfAccelTable &) = delete;
  DwarfAccelTable &operator=(const DwarfAccelTable &) = delete;

  void AddName(DwarfStringPoolEntryRef Name, const DIE *Die, char Flags = 0);
  void FinalizeTable(AsmPrinter *, StringRef);
  void emit(AsmPrinter *, const MCSymbol *, DwarfDebug *);
#ifndef NDEBUG
  void print(raw_ostream &OS);
  void dump() { print(dbgs()); }
#endif
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_DWARFACCELTABLE_H
