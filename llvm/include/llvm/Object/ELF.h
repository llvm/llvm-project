//===- ELF.h - ELF object file implementation -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ELFObjectFile template class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ELF_H
#define LLVM_OBJECT_ELF_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace llvm {
namespace object {

// Subclasses of ELFObjectFile may need this for template instantiation
inline std::pair<unsigned char, unsigned char>
getElfArchType(MemoryBuffer *Object) {
  if (Object->getBufferSize() < ELF::EI_NIDENT)
    return std::make_pair((uint8_t)ELF::ELFCLASSNONE,(uint8_t)ELF::ELFDATANONE);
  return std::make_pair( (uint8_t)Object->getBufferStart()[ELF::EI_CLASS]
                       , (uint8_t)Object->getBufferStart()[ELF::EI_DATA]);
}

// Templates to choose Elf_Addr and Elf_Off depending on is64Bits.
template<support::endianness target_endianness>
struct ELFDataTypeTypedefHelperCommon {
  typedef support::detail::packed_endian_specific_integral
    <uint16_t, target_endianness, support::aligned> Elf_Half;
  typedef support::detail::packed_endian_specific_integral
    <uint32_t, target_endianness, support::aligned> Elf_Word;
  typedef support::detail::packed_endian_specific_integral
    <int32_t, target_endianness, support::aligned> Elf_Sword;
  typedef support::detail::packed_endian_specific_integral
    <uint64_t, target_endianness, support::aligned> Elf_Xword;
  typedef support::detail::packed_endian_specific_integral
    <int64_t, target_endianness, support::aligned> Elf_Sxword;
};

template<support::endianness target_endianness, bool is64Bits>
struct ELFDataTypeTypedefHelper;

/// ELF 32bit types.
template<support::endianness target_endianness>
struct ELFDataTypeTypedefHelper<target_endianness, false>
  : ELFDataTypeTypedefHelperCommon<target_endianness> {
  typedef uint32_t value_type;
  typedef support::detail::packed_endian_specific_integral
    <value_type, target_endianness, support::aligned> Elf_Addr;
  typedef support::detail::packed_endian_specific_integral
    <value_type, target_endianness, support::aligned> Elf_Off;
};

/// ELF 64bit types.
template<support::endianness target_endianness>
struct ELFDataTypeTypedefHelper<target_endianness, true>
  : ELFDataTypeTypedefHelperCommon<target_endianness>{
  typedef uint64_t value_type;
  typedef support::detail::packed_endian_specific_integral
    <value_type, target_endianness, support::aligned> Elf_Addr;
  typedef support::detail::packed_endian_specific_integral
    <value_type, target_endianness, support::aligned> Elf_Off;
};

// I really don't like doing this, but the alternative is copypasta.
#define LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits) \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Addr Elf_Addr; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Off Elf_Off; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Half Elf_Half; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Word Elf_Word; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Sword Elf_Sword; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Xword Elf_Xword; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Sxword Elf_Sxword;

  // Section header.
template<support::endianness target_endianness, bool is64Bits>
struct Elf_Shdr_Base;

template<support::endianness target_endianness>
struct Elf_Shdr_Base<target_endianness, false> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, false)
  Elf_Word sh_name;     // Section name (index into string table)
  Elf_Word sh_type;     // Section type (SHT_*)
  Elf_Word sh_flags;    // Section flags (SHF_*)
  Elf_Addr sh_addr;     // Address where section is to be loaded
  Elf_Off  sh_offset;   // File offset of section data, in bytes
  Elf_Word sh_size;     // Size of section, in bytes
  Elf_Word sh_link;     // Section type-specific header table index link
  Elf_Word sh_info;     // Section type-specific extra information
  Elf_Word sh_addralign;// Section address alignment
  Elf_Word sh_entsize;  // Size of records contained within the section
};

template<support::endianness target_endianness>
struct Elf_Shdr_Base<target_endianness, true> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, true)
  Elf_Word  sh_name;     // Section name (index into string table)
  Elf_Word  sh_type;     // Section type (SHT_*)
  Elf_Xword sh_flags;    // Section flags (SHF_*)
  Elf_Addr  sh_addr;     // Address where section is to be loaded
  Elf_Off   sh_offset;   // File offset of section data, in bytes
  Elf_Xword sh_size;     // Size of section, in bytes
  Elf_Word  sh_link;     // Section type-specific header table index link
  Elf_Word  sh_info;     // Section type-specific extra information
  Elf_Xword sh_addralign;// Section address alignment
  Elf_Xword sh_entsize;  // Size of records contained within the section
};

template<support::endianness target_endianness, bool is64Bits>
struct Elf_Shdr_Impl : Elf_Shdr_Base<target_endianness, is64Bits> {
  using Elf_Shdr_Base<target_endianness, is64Bits>::sh_entsize;
  using Elf_Shdr_Base<target_endianness, is64Bits>::sh_size;

  /// @brief Get the number of entities this section contains if it has any.
  unsigned getEntityCount() const {
    if (sh_entsize == 0)
      return 0;
    return sh_size / sh_entsize;
  }
};

template<support::endianness target_endianness, bool is64Bits>
struct Elf_Sym_Base;

template<support::endianness target_endianness>
struct Elf_Sym_Base<target_endianness, false> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, false)
  Elf_Word      st_name;  // Symbol name (index into string table)
  Elf_Addr      st_value; // Value or address associated with the symbol
  Elf_Word      st_size;  // Size of the symbol
  unsigned char st_info;  // Symbol's type and binding attributes
  unsigned char st_other; // Must be zero; reserved
  Elf_Half      st_shndx; // Which section (header table index) it's defined in
};

template<support::endianness target_endianness>
struct Elf_Sym_Base<target_endianness, true> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, true)
  Elf_Word      st_name;  // Symbol name (index into string table)
  unsigned char st_info;  // Symbol's type and binding attributes
  unsigned char st_other; // Must be zero; reserved
  Elf_Half      st_shndx; // Which section (header table index) it's defined in
  Elf_Addr      st_value; // Value or address associated with the symbol
  Elf_Xword     st_size;  // Size of the symbol
};

template<support::endianness target_endianness, bool is64Bits>
struct Elf_Sym_Impl : Elf_Sym_Base<target_endianness, is64Bits> {
  using Elf_Sym_Base<target_endianness, is64Bits>::st_info;

  // These accessors and mutators correspond to the ELF32_ST_BIND,
  // ELF32_ST_TYPE, and ELF32_ST_INFO macros defined in the ELF specification:
  unsigned char getBinding() const { return st_info >> 4; }
  unsigned char getType() const { return st_info & 0x0f; }
  void setBinding(unsigned char b) { setBindingAndType(b, getType()); }
  void setType(unsigned char t) { setBindingAndType(getBinding(), t); }
  void setBindingAndType(unsigned char b, unsigned char t) {
    st_info = (b << 4) + (t & 0x0f);
  }
};

/// Elf_Versym: This is the structure of entries in the SHT_GNU_versym section
/// (.gnu.version). This structure is identical for ELF32 and ELF64.
template<support::endianness target_endianness, bool is64Bits>
struct Elf_Versym_Impl {
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  Elf_Half vs_index;   // Version index with flags (e.g. VERSYM_HIDDEN)
};

template<support::endianness target_endianness, bool is64Bits>
struct Elf_Verdaux_Impl;

/// Elf_Verdef: This is the structure of entries in the SHT_GNU_verdef section
/// (.gnu.version_d). This structure is identical for ELF32 and ELF64.
template<support::endianness target_endianness, bool is64Bits>
struct Elf_Verdef_Impl {
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  typedef Elf_Verdaux_Impl<target_endianness, is64Bits> Elf_Verdaux;
  Elf_Half vd_version; // Version of this structure (e.g. VER_DEF_CURRENT)
  Elf_Half vd_flags;   // Bitwise flags (VER_DEF_*)
  Elf_Half vd_ndx;     // Version index, used in .gnu.version entries
  Elf_Half vd_cnt;     // Number of Verdaux entries
  Elf_Word vd_hash;    // Hash of name
  Elf_Word vd_aux;     // Offset to the first Verdaux entry (in bytes)
  Elf_Word vd_next;    // Offset to the next Verdef entry (in bytes)

  /// Get the first Verdaux entry for this Verdef.
  const Elf_Verdaux *getAux() const {
    return reinterpret_cast<const Elf_Verdaux*>((const char*)this + vd_aux);
  }
};

/// Elf_Verdaux: This is the structure of auxiliary data in the SHT_GNU_verdef
/// section (.gnu.version_d). This structure is identical for ELF32 and ELF64.
template<support::endianness target_endianness, bool is64Bits>
struct Elf_Verdaux_Impl {
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  Elf_Word vda_name; // Version name (offset in string table)
  Elf_Word vda_next; // Offset to next Verdaux entry (in bytes)
};

/// Elf_Verneed: This is the structure of entries in the SHT_GNU_verneed
/// section (.gnu.version_r). This structure is identical for ELF32 and ELF64.
template<support::endianness target_endianness, bool is64Bits>
struct Elf_Verneed_Impl {
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  Elf_Half vn_version; // Version of this structure (e.g. VER_NEED_CURRENT)
  Elf_Half vn_cnt;     // Number of associated Vernaux entries
  Elf_Word vn_file;    // Library name (string table offset)
  Elf_Word vn_aux;     // Offset to first Vernaux entry (in bytes)
  Elf_Word vn_next;    // Offset to next Verneed entry (in bytes)
};

/// Elf_Vernaux: This is the structure of auxiliary data in SHT_GNU_verneed
/// section (.gnu.version_r). This structure is identical for ELF32 and ELF64.
template<support::endianness target_endianness, bool is64Bits>
struct Elf_Vernaux_Impl {
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  Elf_Word vna_hash;  // Hash of dependency name
  Elf_Half vna_flags; // Bitwise Flags (VER_FLAG_*)
  Elf_Half vna_other; // Version index, used in .gnu.version entries
  Elf_Word vna_name;  // Dependency name
  Elf_Word vna_next;  // Offset to next Vernaux entry (in bytes)
};

/// Elf_Dyn_Base: This structure matches the form of entries in the dynamic
///               table section (.dynamic) look like.
template<support::endianness target_endianness, bool is64Bits>
struct Elf_Dyn_Base;

template<support::endianness target_endianness>
struct Elf_Dyn_Base<target_endianness, false> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, false)
  Elf_Sword d_tag;
  union {
    Elf_Word d_val;
    Elf_Addr d_ptr;
  } d_un;
};

template<support::endianness target_endianness>
struct Elf_Dyn_Base<target_endianness, true> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, true)
  Elf_Sxword d_tag;
  union {
    Elf_Xword d_val;
    Elf_Addr d_ptr;
  } d_un;
};

/// Elf_Dyn_Impl: This inherits from Elf_Dyn_Base, adding getters and setters.
template<support::endianness target_endianness, bool is64Bits>
struct Elf_Dyn_Impl : Elf_Dyn_Base<target_endianness, is64Bits> {
  using Elf_Dyn_Base<target_endianness, is64Bits>::d_tag;
  using Elf_Dyn_Base<target_endianness, is64Bits>::d_un;
  int64_t getTag() const { return d_tag; }
  uint64_t getVal() const { return d_un.d_val; }
  uint64_t getPtr() const { return d_un.ptr; }
};

template<support::endianness target_endianness, bool is64Bits>
class ELFObjectFile;

// DynRefImpl: Reference to an entry in the dynamic table
// This is an ELF-specific interface.
template<support::endianness target_endianness, bool is64Bits>
class DynRefImpl {
  typedef Elf_Dyn_Impl<target_endianness, is64Bits> Elf_Dyn;
  typedef ELFObjectFile<target_endianness, is64Bits> OwningType;

  DataRefImpl DynPimpl;
  const OwningType *OwningObject;

public:
  DynRefImpl() : OwningObject(NULL) { }

  DynRefImpl(DataRefImpl DynP, const OwningType *Owner);

  bool operator==(const DynRefImpl &Other) const;
  bool operator <(const DynRefImpl &Other) const;

  error_code getNext(DynRefImpl &Result) const;
  int64_t getTag() const;
  uint64_t getVal() const;
  uint64_t getPtr() const;

  DataRefImpl getRawDataRefImpl() const;
};

// Elf_Rel: Elf Relocation
template<support::endianness target_endianness, bool is64Bits, bool isRela>
struct Elf_Rel_Base;

template<support::endianness target_endianness>
struct Elf_Rel_Base<target_endianness, false, false> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, false)
  Elf_Addr      r_offset; // Location (file byte offset, or program virtual addr)
  Elf_Word      r_info;  // Symbol table index and type of relocation to apply
};

template<support::endianness target_endianness>
struct Elf_Rel_Base<target_endianness, true, false> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, true)
  Elf_Addr      r_offset; // Location (file byte offset, or program virtual addr)
  Elf_Xword     r_info;   // Symbol table index and type of relocation to apply
};

template<support::endianness target_endianness>
struct Elf_Rel_Base<target_endianness, false, true> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, false)
  Elf_Addr      r_offset; // Location (file byte offset, or program virtual addr)
  Elf_Word      r_info;   // Symbol table index and type of relocation to apply
  Elf_Sword     r_addend; // Compute value for relocatable field by adding this
};

template<support::endianness target_endianness>
struct Elf_Rel_Base<target_endianness, true, true> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, true)
  Elf_Addr      r_offset; // Location (file byte offset, or program virtual addr)
  Elf_Xword     r_info;   // Symbol table index and type of relocation to apply
  Elf_Sxword    r_addend; // Compute value for relocatable field by adding this.
};

template<support::endianness target_endianness, bool is64Bits, bool isRela>
struct Elf_Rel_Impl;

template<support::endianness target_endianness, bool isRela>
struct Elf_Rel_Impl<target_endianness, true, isRela>
       : Elf_Rel_Base<target_endianness, true, isRela> {
  using Elf_Rel_Base<target_endianness, true, isRela>::r_info;
  LLVM_ELF_IMPORT_TYPES(target_endianness, true)

  // These accessors and mutators correspond to the ELF64_R_SYM, ELF64_R_TYPE,
  // and ELF64_R_INFO macros defined in the ELF specification:
  uint64_t getSymbol() const { return (r_info >> 32); }
  unsigned char getType() const {
    return (unsigned char) (r_info & 0xffffffffL);
  }
  void setSymbol(uint64_t s) { setSymbolAndType(s, getType()); }
  void setType(unsigned char t) { setSymbolAndType(getSymbol(), t); }
  void setSymbolAndType(uint64_t s, unsigned char t) {
    r_info = (s << 32) + (t&0xffffffffL);
  }
};

template<support::endianness target_endianness, bool isRela>
struct Elf_Rel_Impl<target_endianness, false, isRela>
       : Elf_Rel_Base<target_endianness, false, isRela> {
  using Elf_Rel_Base<target_endianness, false, isRela>::r_info;
  LLVM_ELF_IMPORT_TYPES(target_endianness, false)

  // These accessors and mutators correspond to the ELF32_R_SYM, ELF32_R_TYPE,
  // and ELF32_R_INFO macros defined in the ELF specification:
  uint32_t getSymbol() const { return (r_info >> 8); }
  unsigned char getType() const { return (unsigned char) (r_info & 0x0ff); }
  void setSymbol(uint32_t s) { setSymbolAndType(s, getType()); }
  void setType(unsigned char t) { setSymbolAndType(getSymbol(), t); }
  void setSymbolAndType(uint32_t s, unsigned char t) {
    r_info = (s << 8) + t;
  }
};

template<support::endianness target_endianness, bool is64Bits>
struct Elf_Ehdr_Impl {
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  unsigned char e_ident[ELF::EI_NIDENT]; // ELF Identification bytes
  Elf_Half e_type;     // Type of file (see ET_*)
  Elf_Half e_machine;  // Required architecture for this file (see EM_*)
  Elf_Word e_version;  // Must be equal to 1
  Elf_Addr e_entry;    // Address to jump to in order to start program
  Elf_Off  e_phoff;    // Program header table's file offset, in bytes
  Elf_Off  e_shoff;    // Section header table's file offset, in bytes
  Elf_Word e_flags;    // Processor-specific flags
  Elf_Half e_ehsize;   // Size of ELF header, in bytes
  Elf_Half e_phentsize;// Size of an entry in the program header table
  Elf_Half e_phnum;    // Number of entries in the program header table
  Elf_Half e_shentsize;// Size of an entry in the section header table
  Elf_Half e_shnum;    // Number of entries in the section header table
  Elf_Half e_shstrndx; // Section header table index of section name
                                 // string table
  bool checkMagic() const {
    return (memcmp(e_ident, ELF::ElfMagic, strlen(ELF::ElfMagic))) == 0;
  }
   unsigned char getFileClass() const { return e_ident[ELF::EI_CLASS]; }
   unsigned char getDataEncoding() const { return e_ident[ELF::EI_DATA]; }
};

template<support::endianness target_endianness, bool is64Bits>
struct Elf_Phdr;

template<support::endianness target_endianness>
struct Elf_Phdr<target_endianness, false> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, false)
  Elf_Word p_type;   // Type of segment
  Elf_Off  p_offset; // FileOffset where segment is located, in bytes
  Elf_Addr p_vaddr;  // Virtual Address of beginning of segment 
  Elf_Addr p_paddr;  // Physical address of beginning of segment (OS-specific)
  Elf_Word p_filesz; // Num. of bytes in file image of segment (may be zero)
  Elf_Word p_memsz;  // Num. of bytes in mem image of segment (may be zero)
  Elf_Word p_flags;  // Segment flags
  Elf_Word p_align;  // Segment alignment constraint
};

template<support::endianness target_endianness>
struct Elf_Phdr<target_endianness, true> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, true)
  Elf_Word p_type;   // Type of segment
  Elf_Word p_flags;  // Segment flags
  Elf_Off  p_offset; // FileOffset where segment is located, in bytes
  Elf_Addr p_vaddr;  // Virtual Address of beginning of segment 
  Elf_Addr p_paddr;  // Physical address of beginning of segment (OS-specific)
  Elf_Word p_filesz; // Num. of bytes in file image of segment (may be zero)
  Elf_Word p_memsz;  // Num. of bytes in mem image of segment (may be zero)
  Elf_Word p_align;  // Segment alignment constraint
};

template<support::endianness target_endianness, bool is64Bits>
class ELFObjectFile : public ObjectFile {
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)

  typedef Elf_Ehdr_Impl<target_endianness, is64Bits> Elf_Ehdr;
  typedef Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;
  typedef Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;
  typedef Elf_Dyn_Impl<target_endianness, is64Bits> Elf_Dyn;
  typedef Elf_Rel_Impl<target_endianness, is64Bits, false> Elf_Rel;
  typedef Elf_Rel_Impl<target_endianness, is64Bits, true> Elf_Rela;
  typedef Elf_Verdef_Impl<target_endianness, is64Bits> Elf_Verdef;
  typedef Elf_Verdaux_Impl<target_endianness, is64Bits> Elf_Verdaux;
  typedef Elf_Verneed_Impl<target_endianness, is64Bits> Elf_Verneed;
  typedef Elf_Vernaux_Impl<target_endianness, is64Bits> Elf_Vernaux;
  typedef Elf_Versym_Impl<target_endianness, is64Bits> Elf_Versym;
  typedef DynRefImpl<target_endianness, is64Bits> DynRef;
  typedef content_iterator<DynRef> dyn_iterator;

protected:
  // This flag is used for classof, to distinguish ELFObjectFile from
  // its subclass. If more subclasses will be created, this flag will
  // have to become an enum.
  bool isDyldELFObject;

private:
  typedef SmallVector<const Elf_Shdr*, 1> Sections_t;
  typedef DenseMap<unsigned, unsigned> IndexMap_t;
  typedef DenseMap<const Elf_Shdr*, SmallVector<uint32_t, 1> > RelocMap_t;

  const Elf_Ehdr *Header;
  const Elf_Shdr *SectionHeaderTable;
  const Elf_Shdr *dot_shstrtab_sec; // Section header string table.
  const Elf_Shdr *dot_strtab_sec;   // Symbol header string table.
  const Elf_Shdr *dot_dynstr_sec;   // Dynamic symbol string table.

  // SymbolTableSections[0] always points to the dynamic string table section
  // header, or NULL if there is no dynamic string table.
  Sections_t SymbolTableSections;
  IndexMap_t SymbolTableSectionsIndexMap;
  DenseMap<const Elf_Sym*, ELF::Elf64_Word> ExtendedSymbolTable;

  const Elf_Shdr *dot_dynamic_sec;       // .dynamic
  const Elf_Shdr *dot_gnu_version_sec;   // .gnu.version
  const Elf_Shdr *dot_gnu_version_r_sec; // .gnu.version_r
  const Elf_Shdr *dot_gnu_version_d_sec; // .gnu.version_d

  // Pointer to SONAME entry in dynamic string table
  // This is set the first time getLoadName is called.
  mutable const char *dt_soname;

public:
  /// \brief Iterate over relocations in a .rel or .rela section.
  template<class RelocT>
  class ELFRelocationIterator {
  public:
    typedef void difference_type;
    typedef const RelocT value_type;
    typedef std::forward_iterator_tag iterator_category;
    typedef value_type &reference;
    typedef value_type *pointer;

    /// \brief Default construct iterator.
    ELFRelocationIterator() : Section(0), Current(0) {}
    ELFRelocationIterator(const Elf_Shdr *Sec, const char *Start)
      : Section(Sec)
      , Current(Start) {}

    reference operator *() {
      assert(Current && "Attempted to dereference an invalid iterator!");
      return *reinterpret_cast<const RelocT*>(Current);
    }

    pointer operator ->() {
      assert(Current && "Attempted to dereference an invalid iterator!");
      return reinterpret_cast<const RelocT*>(Current);
    }

    bool operator ==(const ELFRelocationIterator &Other) {
      return Section == Other.Section && Current == Other.Current;
    }

    bool operator !=(const ELFRelocationIterator &Other) {
      return !(*this == Other);
    }

    ELFRelocationIterator &operator ++(int) {
      assert(Current && "Attempted to increment an invalid iterator!");
      Current += Section->sh_entsize;
      return *this;
    }

    ELFRelocationIterator operator ++() {
      ELFRelocationIterator Tmp = *this;
      ++*this;
      return Tmp;
    }

  private:
    const Elf_Shdr *Section;
    const char *Current;
  };

private:
  // Records for each version index the corresponding Verdef or Vernaux entry.
  // This is filled the first time LoadVersionMap() is called.
  class VersionMapEntry : public PointerIntPair<const void*, 1> {
    public:
    // If the integer is 0, this is an Elf_Verdef*.
    // If the integer is 1, this is an Elf_Vernaux*.
    VersionMapEntry() : PointerIntPair<const void*, 1>(NULL, 0) { }
    VersionMapEntry(const Elf_Verdef *verdef)
        : PointerIntPair<const void*, 1>(verdef, 0) { }
    VersionMapEntry(const Elf_Vernaux *vernaux)
        : PointerIntPair<const void*, 1>(vernaux, 1) { }
    bool isNull() const { return getPointer() == NULL; }
    bool isVerdef() const { return !isNull() && getInt() == 0; }
    bool isVernaux() const { return !isNull() && getInt() == 1; }
    const Elf_Verdef *getVerdef() const {
      return isVerdef() ? (const Elf_Verdef*)getPointer() : NULL;
    }
    const Elf_Vernaux *getVernaux() const {
      return isVernaux() ? (const Elf_Vernaux*)getPointer() : NULL;
    }
  };
  mutable SmallVector<VersionMapEntry, 16> VersionMap;
  void LoadVersionDefs(const Elf_Shdr *sec) const;
  void LoadVersionNeeds(const Elf_Shdr *ec) const;
  void LoadVersionMap() const;

  /// @brief Map sections to an array of relocation sections that reference
  ///        them sorted by section index.
  RelocMap_t SectionRelocMap;

  /// @brief Get the relocation section that contains \a Rel.
  const Elf_Shdr *getRelSection(DataRefImpl Rel) const {
    return getSection(Rel.w.b);
  }

  bool            isRelocationHasAddend(DataRefImpl Rel) const;
  template<typename T>
  const T        *getEntry(uint16_t Section, uint32_t Entry) const;
  template<typename T>
  const T        *getEntry(const Elf_Shdr *Section, uint32_t Entry) const;
  const Elf_Shdr *getSection(DataRefImpl index) const;
  const Elf_Shdr *getSection(uint32_t index) const;
  const Elf_Rel  *getRel(DataRefImpl Rel) const;
  const Elf_Rela *getRela(DataRefImpl Rela) const;
  const char     *getString(uint32_t section, uint32_t offset) const;
  const char     *getString(const Elf_Shdr *section, uint32_t offset) const;
  error_code      getSymbolVersion(const Elf_Shdr *section,
                                   const Elf_Sym *Symb,
                                   StringRef &Version,
                                   bool &IsDefault) const;
  void VerifyStrTab(const Elf_Shdr *sh) const;

protected:
  const Elf_Sym  *getSymbol(DataRefImpl Symb) const; // FIXME: Should be private?
  void            validateSymbol(DataRefImpl Symb) const;

public:
  error_code      getSymbolName(const Elf_Shdr *section,
                                const Elf_Sym *Symb,
                                StringRef &Res) const;
  error_code      getSectionName(const Elf_Shdr *section,
                                 StringRef &Res) const;
  const Elf_Dyn  *getDyn(DataRefImpl DynData) const;
  error_code getSymbolVersion(SymbolRef Symb, StringRef &Version,
                              bool &IsDefault) const;
protected:
  virtual error_code getSymbolNext(DataRefImpl Symb, SymbolRef &Res) const;
  virtual error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const;
  virtual error_code getSymbolFileOffset(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolNMTypeChar(DataRefImpl Symb, char &Res) const;
  virtual error_code getSymbolFlags(DataRefImpl Symb, uint32_t &Res) const;
  virtual error_code getSymbolType(DataRefImpl Symb, SymbolRef::Type &Res) const;
  virtual error_code getSymbolSection(DataRefImpl Symb,
                                      section_iterator &Res) const;
  virtual error_code getSymbolValue(DataRefImpl Symb, uint64_t &Val) const;

  friend class DynRefImpl<target_endianness, is64Bits>;
  virtual error_code getDynNext(DataRefImpl DynData, DynRef &Result) const;

  virtual error_code getLibraryNext(DataRefImpl Data, LibraryRef &Result) const;
  virtual error_code getLibraryPath(DataRefImpl Data, StringRef &Res) const;

  virtual error_code getSectionNext(DataRefImpl Sec, SectionRef &Res) const;
  virtual error_code getSectionName(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionContents(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSectionAlignment(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code isSectionText(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionData(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionBSS(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionRequiredForExecution(DataRefImpl Sec,
                                                   bool &Res) const;
  virtual error_code isSectionVirtual(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionZeroInit(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionReadOnlyData(DataRefImpl Sec, bool &Res) const;
  virtual error_code sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                           bool &Result) const;
  virtual relocation_iterator getSectionRelBegin(DataRefImpl Sec) const;
  virtual relocation_iterator getSectionRelEnd(DataRefImpl Sec) const;

  virtual error_code getRelocationNext(DataRefImpl Rel,
                                       RelocationRef &Res) const;
  virtual error_code getRelocationAddress(DataRefImpl Rel,
                                          uint64_t &Res) const;
  virtual error_code getRelocationOffset(DataRefImpl Rel,
                                         uint64_t &Res) const;
  virtual error_code getRelocationSymbol(DataRefImpl Rel,
                                         SymbolRef &Res) const;
  virtual error_code getRelocationType(DataRefImpl Rel,
                                       uint64_t &Res) const;
  virtual error_code getRelocationTypeName(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;
  virtual error_code getRelocationAdditionalInfo(DataRefImpl Rel,
                                                 int64_t &Res) const;
  virtual error_code getRelocationValueString(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;

public:
  ELFObjectFile(MemoryBuffer *Object, error_code &ec);
  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;

  virtual symbol_iterator begin_dynamic_symbols() const;
  virtual symbol_iterator end_dynamic_symbols() const;

  virtual section_iterator begin_sections() const;
  virtual section_iterator end_sections() const;

  virtual library_iterator begin_libraries_needed() const;
  virtual library_iterator end_libraries_needed() const;

  virtual dyn_iterator begin_dynamic_table() const;
  virtual dyn_iterator end_dynamic_table() const;

  typedef ELFRelocationIterator<Elf_Rela> Elf_Rela_Iter;
  typedef ELFRelocationIterator<Elf_Rel> Elf_Rel_Iter;

  virtual Elf_Rela_Iter beginELFRela(const Elf_Shdr *sec) const {
    return Elf_Rela_Iter(sec, (const char *)(base() + sec->sh_offset));
  }

  virtual Elf_Rela_Iter endELFRela(const Elf_Shdr *sec) const {
    return Elf_Rela_Iter(sec, (const char *)
                         (base() + sec->sh_offset + sec->sh_size));
  }

  virtual Elf_Rel_Iter beginELFRel(const Elf_Shdr *sec) const {
    return Elf_Rel_Iter(sec, (const char *)(base() + sec->sh_offset));
  }

  virtual Elf_Rel_Iter endELFRel(const Elf_Shdr *sec) const {
    return Elf_Rel_Iter(sec, (const char *)
                        (base() + sec->sh_offset + sec->sh_size));
  }

  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getFileFormatName() const;
  virtual StringRef getObjectType() const { return "ELF"; }
  virtual unsigned getArch() const;
  virtual StringRef getLoadName() const;
  virtual error_code getSectionContents(const Elf_Shdr *sec,
                                        StringRef &Res) const;

  uint64_t getNumSections() const;
  uint64_t getStringTableIndex() const;
  ELF::Elf64_Word getSymbolTableIndex(const Elf_Sym *symb) const;
  const Elf_Shdr *getSection(const Elf_Sym *symb) const;
  const Elf_Shdr *getElfSection(section_iterator &It) const;
  const Elf_Sym *getElfSymbol(symbol_iterator &It) const;
  const Elf_Sym *getElfSymbol(uint32_t index) const;

  // Methods for type inquiry through isa, cast, and dyn_cast
  bool isDyldType() const { return isDyldELFObject; }
  static inline bool classof(const Binary *v) {
    return v->getType() == getELFType(target_endianness == support::little,
                                      is64Bits);
  }
};

// Iterate through the version definitions, and place each Elf_Verdef
// in the VersionMap according to its index.
template<support::endianness target_endianness, bool is64Bits>
void ELFObjectFile<target_endianness, is64Bits>::
                  LoadVersionDefs(const Elf_Shdr *sec) const {
  unsigned vd_size = sec->sh_size; // Size of section in bytes
  unsigned vd_count = sec->sh_info; // Number of Verdef entries
  const char *sec_start = (const char*)base() + sec->sh_offset;
  const char *sec_end = sec_start + vd_size;
  // The first Verdef entry is at the start of the section.
  const char *p = sec_start;
  for (unsigned i = 0; i < vd_count; i++) {
    if (p + sizeof(Elf_Verdef) > sec_end)
      report_fatal_error("Section ended unexpectedly while scanning "
                         "version definitions.");
    const Elf_Verdef *vd = reinterpret_cast<const Elf_Verdef *>(p);
    if (vd->vd_version != ELF::VER_DEF_CURRENT)
      report_fatal_error("Unexpected verdef version");
    size_t index = vd->vd_ndx & ELF::VERSYM_VERSION;
    if (index >= VersionMap.size())
      VersionMap.resize(index+1);
    VersionMap[index] = VersionMapEntry(vd);
    p += vd->vd_next;
  }
}

// Iterate through the versions needed section, and place each Elf_Vernaux
// in the VersionMap according to its index.
template<support::endianness target_endianness, bool is64Bits>
void ELFObjectFile<target_endianness, is64Bits>::
                  LoadVersionNeeds(const Elf_Shdr *sec) const {
  unsigned vn_size = sec->sh_size; // Size of section in bytes
  unsigned vn_count = sec->sh_info; // Number of Verneed entries
  const char *sec_start = (const char*)base() + sec->sh_offset;
  const char *sec_end = sec_start + vn_size;
  // The first Verneed entry is at the start of the section.
  const char *p = sec_start;
  for (unsigned i = 0; i < vn_count; i++) {
    if (p + sizeof(Elf_Verneed) > sec_end)
      report_fatal_error("Section ended unexpectedly while scanning "
                         "version needed records.");
    const Elf_Verneed *vn = reinterpret_cast<const Elf_Verneed *>(p);
    if (vn->vn_version != ELF::VER_NEED_CURRENT)
      report_fatal_error("Unexpected verneed version");
    // Iterate through the Vernaux entries
    const char *paux = p + vn->vn_aux;
    for (unsigned j = 0; j < vn->vn_cnt; j++) {
      if (paux + sizeof(Elf_Vernaux) > sec_end)
        report_fatal_error("Section ended unexpected while scanning auxiliary "
                           "version needed records.");
      const Elf_Vernaux *vna = reinterpret_cast<const Elf_Vernaux *>(paux);
      size_t index = vna->vna_other & ELF::VERSYM_VERSION;
      if (index >= VersionMap.size())
        VersionMap.resize(index+1);
      VersionMap[index] = VersionMapEntry(vna);
      paux += vna->vna_next;
    }
    p += vn->vn_next;
  }
}

template<support::endianness target_endianness, bool is64Bits>
void ELFObjectFile<target_endianness, is64Bits>::LoadVersionMap() const {
  // If there is no dynamic symtab or version table, there is nothing to do.
  if (SymbolTableSections[0] == NULL || dot_gnu_version_sec == NULL)
    return;

  // Has the VersionMap already been loaded?
  if (VersionMap.size() > 0)
    return;

  // The first two version indexes are reserved.
  // Index 0 is LOCAL, index 1 is GLOBAL.
  VersionMap.push_back(VersionMapEntry());
  VersionMap.push_back(VersionMapEntry());

  if (dot_gnu_version_d_sec)
    LoadVersionDefs(dot_gnu_version_d_sec);

  if (dot_gnu_version_r_sec)
    LoadVersionNeeds(dot_gnu_version_r_sec);
}

template<support::endianness target_endianness, bool is64Bits>
void ELFObjectFile<target_endianness, is64Bits>
                  ::validateSymbol(DataRefImpl Symb) const {
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *SymbolTableSection = SymbolTableSections[Symb.d.b];
  // FIXME: We really need to do proper error handling in the case of an invalid
  //        input file. Because we don't use exceptions, I think we'll just pass
  //        an error object around.
  if (!(  symb
        && SymbolTableSection
        && symb >= (const Elf_Sym*)(base()
                   + SymbolTableSection->sh_offset)
        && symb <  (const Elf_Sym*)(base()
                   + SymbolTableSection->sh_offset
                   + SymbolTableSection->sh_size)))
    // FIXME: Proper error handling.
    report_fatal_error("Symb must point to a valid symbol!");
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolNext(DataRefImpl Symb,
                                        SymbolRef &Result) const {
  validateSymbol(Symb);
  const Elf_Shdr *SymbolTableSection = SymbolTableSections[Symb.d.b];

  ++Symb.d.a;
  // Check to see if we are at the end of this symbol table.
  if (Symb.d.a >= SymbolTableSection->getEntityCount()) {
    // We are at the end. If there are other symbol tables, jump to them.
    // If the symbol table is .dynsym, we are iterating dynamic symbols,
    // and there is only one table of these.
    if (Symb.d.b != 0) {
      ++Symb.d.b;
      Symb.d.a = 1; // The 0th symbol in ELF is fake.
    }
    // Otherwise return the terminator.
    if (Symb.d.b == 0 || Symb.d.b >= SymbolTableSections.size()) {
      Symb.d.a = std::numeric_limits<uint32_t>::max();
      Symb.d.b = std::numeric_limits<uint32_t>::max();
    }
  }

  Result = SymbolRef(Symb, this);
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolName(DataRefImpl Symb,
                                        StringRef &Result) const {
  validateSymbol(Symb);
  const Elf_Sym *symb = getSymbol(Symb);
  return getSymbolName(SymbolTableSections[Symb.d.b], symb, Result);
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolVersion(SymbolRef SymRef,
                                           StringRef &Version,
                                           bool &IsDefault) const {
  DataRefImpl Symb = SymRef.getRawDataRefImpl();
  validateSymbol(Symb);
  const Elf_Sym *symb = getSymbol(Symb);
  return getSymbolVersion(SymbolTableSections[Symb.d.b], symb,
                          Version, IsDefault);
}

template<support::endianness target_endianness, bool is64Bits>
ELF::Elf64_Word ELFObjectFile<target_endianness, is64Bits>
                      ::getSymbolTableIndex(const Elf_Sym *symb) const {
  if (symb->st_shndx == ELF::SHN_XINDEX)
    return ExtendedSymbolTable.lookup(symb);
  return symb->st_shndx;
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Shdr *
ELFObjectFile<target_endianness, is64Bits>
                             ::getSection(const Elf_Sym *symb) const {
  if (symb->st_shndx == ELF::SHN_XINDEX)
    return getSection(ExtendedSymbolTable.lookup(symb));
  if (symb->st_shndx >= ELF::SHN_LORESERVE)
    return 0;
  return getSection(symb->st_shndx);
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Shdr *
ELFObjectFile<target_endianness, is64Bits>
                             ::getElfSection(section_iterator &It) const {
  llvm::object::DataRefImpl ShdrRef = It->getRawDataRefImpl();
  return reinterpret_cast<const Elf_Shdr *>(ShdrRef.p);
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Sym *
ELFObjectFile<target_endianness, is64Bits>
                             ::getElfSymbol(symbol_iterator &It) const {
  return getSymbol(It->getRawDataRefImpl());
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Sym *
ELFObjectFile<target_endianness, is64Bits>
                             ::getElfSymbol(uint32_t index) const {
  DataRefImpl SymbolData;
  SymbolData.d.a = index;
  SymbolData.d.b = 1;
  return getSymbol(SymbolData);
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolFileOffset(DataRefImpl Symb,
                                          uint64_t &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *Section;
  switch (getSymbolTableIndex(symb)) {
  case ELF::SHN_COMMON:
   // Unintialized symbols have no offset in the object file
  case ELF::SHN_UNDEF:
    Result = UnknownAddressOrSize;
    return object_error::success;
  case ELF::SHN_ABS:
    Result = symb->st_value;
    return object_error::success;
  default: Section = getSection(symb);
  }

  switch (symb->getType()) {
  case ELF::STT_SECTION:
    Result = Section ? Section->sh_addr : UnknownAddressOrSize;
    return object_error::success;
  case ELF::STT_FUNC:
  case ELF::STT_OBJECT:
  case ELF::STT_NOTYPE:
    Result = symb->st_value +
             (Section ? Section->sh_offset : 0);
    return object_error::success;
  default:
    Result = UnknownAddressOrSize;
    return object_error::success;
  }
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolAddress(DataRefImpl Symb,
                                           uint64_t &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *Section;
  switch (getSymbolTableIndex(symb)) {
  case ELF::SHN_COMMON:
  case ELF::SHN_UNDEF:
    Result = UnknownAddressOrSize;
    return object_error::success;
  case ELF::SHN_ABS:
    Result = symb->st_value;
    return object_error::success;
  default: Section = getSection(symb);
  }

  switch (symb->getType()) {
  case ELF::STT_SECTION:
    Result = Section ? Section->sh_addr : UnknownAddressOrSize;
    return object_error::success;
  case ELF::STT_FUNC:
  case ELF::STT_OBJECT:
  case ELF::STT_NOTYPE:
    bool IsRelocatable;
    switch(Header->e_type) {
    case ELF::ET_EXEC:
    case ELF::ET_DYN:
      IsRelocatable = false;
      break;
    default:
      IsRelocatable = true;
    }
    Result = symb->st_value;
    if (IsRelocatable && Section != 0)
      Result += Section->sh_addr;
    return object_error::success;
  default:
    Result = UnknownAddressOrSize;
    return object_error::success;
  }
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolSize(DataRefImpl Symb,
                                        uint64_t &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  if (symb->st_size == 0)
    Result = UnknownAddressOrSize;
  Result = symb->st_size;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolNMTypeChar(DataRefImpl Symb,
                                              char &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *Section = getSection(symb);

  char ret = '?';

  if (Section) {
    switch (Section->sh_type) {
    case ELF::SHT_PROGBITS:
    case ELF::SHT_DYNAMIC:
      switch (Section->sh_flags) {
      case (ELF::SHF_ALLOC | ELF::SHF_EXECINSTR):
        ret = 't'; break;
      case (ELF::SHF_ALLOC | ELF::SHF_WRITE):
        ret = 'd'; break;
      case ELF::SHF_ALLOC:
      case (ELF::SHF_ALLOC | ELF::SHF_MERGE):
      case (ELF::SHF_ALLOC | ELF::SHF_MERGE | ELF::SHF_STRINGS):
        ret = 'r'; break;
      }
      break;
    case ELF::SHT_NOBITS: ret = 'b';
    }
  }

  switch (getSymbolTableIndex(symb)) {
  case ELF::SHN_UNDEF:
    if (ret == '?')
      ret = 'U';
    break;
  case ELF::SHN_ABS: ret = 'a'; break;
  case ELF::SHN_COMMON: ret = 'c'; break;
  }

  switch (symb->getBinding()) {
  case ELF::STB_GLOBAL: ret = ::toupper(ret); break;
  case ELF::STB_WEAK:
    if (getSymbolTableIndex(symb) == ELF::SHN_UNDEF)
      ret = 'w';
    else
      if (symb->getType() == ELF::STT_OBJECT)
        ret = 'V';
      else
        ret = 'W';
  }

  if (ret == '?' && symb->getType() == ELF::STT_SECTION) {
    StringRef name;
    if (error_code ec = getSymbolName(Symb, name))
      return ec;
    Result = StringSwitch<char>(name)
      .StartsWith(".debug", 'N')
      .StartsWith(".note", 'n')
      .Default('?');
    return object_error::success;
  }

  Result = ret;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolType(DataRefImpl Symb,
                                        SymbolRef::Type &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);

  switch (symb->getType()) {
  case ELF::STT_NOTYPE:
    Result = SymbolRef::ST_Unknown;
    break;
  case ELF::STT_SECTION:
    Result = SymbolRef::ST_Debug;
    break;
  case ELF::STT_FILE:
    Result = SymbolRef::ST_File;
    break;
  case ELF::STT_FUNC:
    Result = SymbolRef::ST_Function;
    break;
  case ELF::STT_OBJECT:
  case ELF::STT_COMMON:
  case ELF::STT_TLS:
    Result = SymbolRef::ST_Data;
    break;
  default:
    Result = SymbolRef::ST_Other;
    break;
  }
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolFlags(DataRefImpl Symb,
                                         uint32_t &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);

  Result = SymbolRef::SF_None;

  if (symb->getBinding() != ELF::STB_LOCAL)
    Result |= SymbolRef::SF_Global;

  if (symb->getBinding() == ELF::STB_WEAK)
    Result |= SymbolRef::SF_Weak;

  if (symb->st_shndx == ELF::SHN_ABS)
    Result |= SymbolRef::SF_Absolute;

  if (symb->getType() == ELF::STT_FILE ||
      symb->getType() == ELF::STT_SECTION)
    Result |= SymbolRef::SF_FormatSpecific;

  if (getSymbolTableIndex(symb) == ELF::SHN_UNDEF)
    Result |= SymbolRef::SF_Undefined;

  if (symb->getType() == ELF::STT_COMMON ||
      getSymbolTableIndex(symb) == ELF::SHN_COMMON)
    Result |= SymbolRef::SF_Common;

  if (symb->getType() == ELF::STT_TLS)
    Result |= SymbolRef::SF_ThreadLocal;

  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolSection(DataRefImpl Symb,
                                           section_iterator &Res) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *sec = getSection(symb);
  if (!sec)
    Res = end_sections();
  else {
    DataRefImpl Sec;
    Sec.p = reinterpret_cast<intptr_t>(sec);
    Res = section_iterator(SectionRef(Sec, this));
  }
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolValue(DataRefImpl Symb,
                                         uint64_t &Val) const {
  validateSymbol(Symb);
  const Elf_Sym *symb = getSymbol(Symb);
  Val = symb->st_value;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSectionNext(DataRefImpl Sec, SectionRef &Result) const {
  const uint8_t *sec = reinterpret_cast<const uint8_t *>(Sec.p);
  sec += Header->e_shentsize;
  Sec.p = reinterpret_cast<intptr_t>(sec);
  Result = SectionRef(Sec, this);
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSectionName(DataRefImpl Sec,
                                         StringRef &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  Result = StringRef(getString(dot_shstrtab_sec, sec->sh_name));
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSectionAddress(DataRefImpl Sec,
                                            uint64_t &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  Result = sec->sh_addr;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSectionSize(DataRefImpl Sec,
                                         uint64_t &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  Result = sec->sh_size;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSectionContents(DataRefImpl Sec,
                                             StringRef &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  const char *start = (const char*)base() + sec->sh_offset;
  Result = StringRef(start, sec->sh_size);
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSectionContents(const Elf_Shdr *Sec,
                                             StringRef &Result) const {
  const char *start = (const char*)base() + Sec->sh_offset;
  Result = StringRef(start, Sec->sh_size);
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSectionAlignment(DataRefImpl Sec,
                                              uint64_t &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  Result = sec->sh_addralign;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::isSectionText(DataRefImpl Sec,
                                        bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_flags & ELF::SHF_EXECINSTR)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::isSectionData(DataRefImpl Sec,
                                        bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_flags & (ELF::SHF_ALLOC | ELF::SHF_WRITE)
      && sec->sh_type == ELF::SHT_PROGBITS)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::isSectionBSS(DataRefImpl Sec,
                                       bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_flags & (ELF::SHF_ALLOC | ELF::SHF_WRITE)
      && sec->sh_type == ELF::SHT_NOBITS)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::isSectionRequiredForExecution(DataRefImpl Sec,
                                                        bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_flags & ELF::SHF_ALLOC)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::isSectionVirtual(DataRefImpl Sec,
                                           bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_type == ELF::SHT_NOBITS)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::isSectionZeroInit(DataRefImpl Sec,
                                            bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  // For ELF, all zero-init sections are virtual (that is, they occupy no space
  //   in the object image) and vice versa.
  if (sec->sh_flags & ELF::SHT_NOBITS)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                       ::isSectionReadOnlyData(DataRefImpl Sec,
                                               bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_flags & ELF::SHF_WRITE || sec->sh_flags & ELF::SHF_EXECINSTR)
    Result = false;
  else
    Result = true;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                          ::sectionContainsSymbol(DataRefImpl Sec,
                                                  DataRefImpl Symb,
                                                  bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
relocation_iterator ELFObjectFile<target_endianness, is64Bits>
                                 ::getSectionRelBegin(DataRefImpl Sec) const {
  DataRefImpl RelData;
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  typename RelocMap_t::const_iterator ittr = SectionRelocMap.find(sec);
  if (sec != 0 && ittr != SectionRelocMap.end()) {
    RelData.w.a = getSection(ittr->second[0])->sh_info;
    RelData.w.b = ittr->second[0];
    RelData.w.c = 0;
  }
  return relocation_iterator(RelocationRef(RelData, this));
}

template<support::endianness target_endianness, bool is64Bits>
relocation_iterator ELFObjectFile<target_endianness, is64Bits>
                                 ::getSectionRelEnd(DataRefImpl Sec) const {
  DataRefImpl RelData;
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  typename RelocMap_t::const_iterator ittr = SectionRelocMap.find(sec);
  if (sec != 0 && ittr != SectionRelocMap.end()) {
    // Get the index of the last relocation section for this section.
    std::size_t relocsecindex = ittr->second[ittr->second.size() - 1];
    const Elf_Shdr *relocsec = getSection(relocsecindex);
    RelData.w.a = relocsec->sh_info;
    RelData.w.b = relocsecindex;
    RelData.w.c = relocsec->sh_size / relocsec->sh_entsize;
  }
  return relocation_iterator(RelocationRef(RelData, this));
}

// Relocations
template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getRelocationNext(DataRefImpl Rel,
                                            RelocationRef &Result) const {
  ++Rel.w.c;
  const Elf_Shdr *relocsec = getSection(Rel.w.b);
  if (Rel.w.c >= (relocsec->sh_size / relocsec->sh_entsize)) {
    // We have reached the end of the relocations for this section. See if there
    // is another relocation section.
    typename RelocMap_t::mapped_type relocseclist =
      SectionRelocMap.lookup(getSection(Rel.w.a));

    // Do a binary search for the current reloc section index (which must be
    // present). Then get the next one.
    typename RelocMap_t::mapped_type::const_iterator loc =
      std::lower_bound(relocseclist.begin(), relocseclist.end(), Rel.w.b);
    ++loc;

    // If there is no next one, don't do anything. The ++Rel.w.c above sets Rel
    // to the end iterator.
    if (loc != relocseclist.end()) {
      Rel.w.b = *loc;
      Rel.w.a = 0;
    }
  }
  Result = RelocationRef(Rel, this);
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getRelocationSymbol(DataRefImpl Rel,
                                              SymbolRef &Result) const {
  uint32_t symbolIdx;
  const Elf_Shdr *sec = getSection(Rel.w.b);
  switch (sec->sh_type) {
    default :
      report_fatal_error("Invalid section type in Rel!");
    case ELF::SHT_REL : {
      symbolIdx = getRel(Rel)->getSymbol();
      break;
    }
    case ELF::SHT_RELA : {
      symbolIdx = getRela(Rel)->getSymbol();
      break;
    }
  }
  DataRefImpl SymbolData;
  IndexMap_t::const_iterator it = SymbolTableSectionsIndexMap.find(sec->sh_link);
  if (it == SymbolTableSectionsIndexMap.end())
    report_fatal_error("Relocation symbol table not found!");
  SymbolData.d.a = symbolIdx;
  SymbolData.d.b = it->second;
  Result = SymbolRef(SymbolData, this);
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getRelocationAddress(DataRefImpl Rel,
                                               uint64_t &Result) const {
  uint64_t offset;
  const Elf_Shdr *sec = getSection(Rel.w.b);
  switch (sec->sh_type) {
    default :
      report_fatal_error("Invalid section type in Rel!");
    case ELF::SHT_REL : {
      offset = getRel(Rel)->r_offset;
      break;
    }
    case ELF::SHT_RELA : {
      offset = getRela(Rel)->r_offset;
      break;
    }
  }

  Result = offset;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getRelocationOffset(DataRefImpl Rel,
                                              uint64_t &Result) const {
  uint64_t offset;
  const Elf_Shdr *sec = getSection(Rel.w.b);
  switch (sec->sh_type) {
    default :
      report_fatal_error("Invalid section type in Rel!");
    case ELF::SHT_REL : {
      offset = getRel(Rel)->r_offset;
      break;
    }
    case ELF::SHT_RELA : {
      offset = getRela(Rel)->r_offset;
      break;
    }
  }

  Result = offset - sec->sh_addr;
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getRelocationType(DataRefImpl Rel,
                                            uint64_t &Result) const {
  const Elf_Shdr *sec = getSection(Rel.w.b);
  switch (sec->sh_type) {
    default :
      report_fatal_error("Invalid section type in Rel!");
    case ELF::SHT_REL : {
      Result = getRel(Rel)->getType();
      break;
    }
    case ELF::SHT_RELA : {
      Result = getRela(Rel)->getType();
      break;
    }
  }
  return object_error::success;
}

#define LLVM_ELF_SWITCH_RELOC_TYPE_NAME(enum) \
  case ELF::enum: res = #enum; break;

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getRelocationTypeName(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  const Elf_Shdr *sec = getSection(Rel.w.b);
  uint8_t type;
  StringRef res;
  switch (sec->sh_type) {
    default :
      return object_error::parse_failed;
    case ELF::SHT_REL : {
      type = getRel(Rel)->getType();
      break;
    }
    case ELF::SHT_RELA : {
      type = getRela(Rel)->getType();
      break;
    }
  }
  switch (Header->e_machine) {
  case ELF::EM_X86_64:
    switch (type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_PC32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_PLT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_COPY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GLOB_DAT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_JUMP_SLOT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_RELATIVE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTPCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_32S);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_PC16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_PC8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_DTPMOD64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_DTPOFF64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TPOFF64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TLSGD);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TLSLD);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_DTPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTTPOFF);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_PC64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTOFF64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTPC32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_SIZE32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_SIZE64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTPC32_TLSDESC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TLSDESC_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TLSDESC);
    default:
      res = "Unknown";
    }
    break;
  case ELF::EM_386:
    switch (type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_PC32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_GOT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_PLT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_COPY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_GLOB_DAT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_JUMP_SLOT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_RELATIVE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_GOTOFF);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_GOTPC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_32PLT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_TPOFF);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_IE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GOTIE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GD);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDM);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_PC16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_PC8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GD_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GD_PUSH);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GD_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GD_POP);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDM_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDM_PUSH);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDM_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDM_POP);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDO_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_IE_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LE_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_DTPMOD32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_DTPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_TPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GOTDESC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_DESC_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_DESC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_IRELATIVE);
    default:
      res = "Unknown";
    }
    break;
  case ELF::EM_ARM:
    switch (type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PC24);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ABS32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_REL32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_PC_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ABS16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ABS12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_ABS5);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ABS8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_SBREL32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_PC8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_BREL_ADJ);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_DESC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_SWI8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_XPC25);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_XPC22);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_DTPMOD32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_DTPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_TPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_COPY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GLOB_DAT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_JUMP_SLOT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_RELATIVE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOTOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_BASE_PREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOT_BREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PLT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_JUMP24);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_JUMP24);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_BASE_ABS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PCREL_7_0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PCREL_15_8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PCREL_23_15);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_SBREL_11_0_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SBREL_19_12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SBREL_27_20_CK);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TARGET1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_SBREL31);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_V4BX);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TARGET2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PREL31);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVW_ABS_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVT_ABS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVW_PREL_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVT_PREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVW_ABS_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVT_ABS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVW_PREL_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVT_PREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_JUMP19);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_JUMP6);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_ALU_PREL_11_0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_PC12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ABS32_NOI);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_REL32_NOI);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PC_G0_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PC_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PC_G1_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PC_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PC_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_PC_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_PC_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_PC_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_PC_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_PC_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_PC_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_PC_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_PC_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SB_G0_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SB_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SB_G1_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SB_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SB_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_SB_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_SB_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_SB_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_SB_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_SB_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_SB_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_SB_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_SB_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_SB_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVW_BREL_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVT_BREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVW_BREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVW_BREL_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVT_BREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVW_BREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_GOTDESC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_DESCSEQ);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_TLS_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PLT32_ABS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOT_ABS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOT_PREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOT_BREL12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOTOFF12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOTRELAX);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GNU_VTENTRY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GNU_VTINHERIT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_JUMP11);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_JUMP8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_GD32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_LDM32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_LDO32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_IE32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_LE32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_LDO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_LE12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_IE12GP);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_3);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_4);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_5);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_6);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_7);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_9);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_10);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_11);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_13);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_14);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_15);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ME_TOO);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_TLS_DESCSEQ16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_TLS_DESCSEQ32);
    default:
      res = "Unknown";
    }
    break;
  case ELF::EM_HEXAGON:
    switch (type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B22_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B15_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B7_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GPREL16_0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GPREL16_1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GPREL16_2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GPREL16_3);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_HL16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B13_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B9_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B32_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B22_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B15_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B13_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B9_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B7_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_12_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_10_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_9_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_8_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_7_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_32_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_COPY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GLOB_DAT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_JMP_SLOT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_RELATIVE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_PLT_B22_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPMOD_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_PLT_B22_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_6_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_11_X);
    default:
      res = "Unknown";
    }
    break;
  default:
    res = "Unknown";
  }
  Result.append(res.begin(), res.end());
  return object_error::success;
}

#undef LLVM_ELF_SWITCH_RELOC_TYPE_NAME

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getRelocationAdditionalInfo(DataRefImpl Rel,
                                                      int64_t &Result) const {
  const Elf_Shdr *sec = getSection(Rel.w.b);
  switch (sec->sh_type) {
    default :
      report_fatal_error("Invalid section type in Rel!");
    case ELF::SHT_REL : {
      Result = 0;
      return object_error::success;
    }
    case ELF::SHT_RELA : {
      Result = getRela(Rel)->r_addend;
      return object_error::success;
    }
  }
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getRelocationValueString(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  const Elf_Shdr *sec = getSection(Rel.w.b);
  uint8_t type;
  StringRef res;
  int64_t addend = 0;
  uint16_t symbol_index = 0;
  switch (sec->sh_type) {
    default:
      return object_error::parse_failed;
    case ELF::SHT_REL: {
      type = getRel(Rel)->getType();
      symbol_index = getRel(Rel)->getSymbol();
      // TODO: Read implicit addend from section data.
      break;
    }
    case ELF::SHT_RELA: {
      type = getRela(Rel)->getType();
      symbol_index = getRela(Rel)->getSymbol();
      addend = getRela(Rel)->r_addend;
      break;
    }
  }
  const Elf_Sym *symb = getEntry<Elf_Sym>(sec->sh_link, symbol_index);
  StringRef symname;
  if (error_code ec = getSymbolName(getSection(sec->sh_link), symb, symname))
    return ec;
  switch (Header->e_machine) {
  case ELF::EM_X86_64:
    switch (type) {
    case ELF::R_X86_64_PC8:
    case ELF::R_X86_64_PC16:
    case ELF::R_X86_64_PC32: {
        std::string fmtbuf;
        raw_string_ostream fmt(fmtbuf);
        fmt << symname << (addend < 0 ? "" : "+") << addend << "-P";
        fmt.flush();
        Result.append(fmtbuf.begin(), fmtbuf.end());
      }
      break;
    case ELF::R_X86_64_8:
    case ELF::R_X86_64_16:
    case ELF::R_X86_64_32:
    case ELF::R_X86_64_32S:
    case ELF::R_X86_64_64: {
        std::string fmtbuf;
        raw_string_ostream fmt(fmtbuf);
        fmt << symname << (addend < 0 ? "" : "+") << addend;
        fmt.flush();
        Result.append(fmtbuf.begin(), fmtbuf.end());
      }
      break;
    default:
      res = "Unknown";
    }
    break;
  case ELF::EM_ARM:
  case ELF::EM_HEXAGON:
    res = symname;
    break;
  default:
    res = "Unknown";
  }
  if (Result.empty())
    Result.append(res.begin(), res.end());
  return object_error::success;
}

// Verify that the last byte in the string table in a null.
template<support::endianness target_endianness, bool is64Bits>
void ELFObjectFile<target_endianness, is64Bits>
                  ::VerifyStrTab(const Elf_Shdr *sh) const {
  const char *strtab = (const char*)base() + sh->sh_offset;
  if (strtab[sh->sh_size - 1] != 0)
    // FIXME: Proper error handling.
    report_fatal_error("String table must end with a null terminator!");
}

template<support::endianness target_endianness, bool is64Bits>
ELFObjectFile<target_endianness, is64Bits>::ELFObjectFile(MemoryBuffer *Object
                                                          , error_code &ec)
  : ObjectFile(getELFType(target_endianness == support::little, is64Bits),
               Object, ec)
  , isDyldELFObject(false)
  , SectionHeaderTable(0)
  , dot_shstrtab_sec(0)
  , dot_strtab_sec(0)
  , dot_dynstr_sec(0)
  , dot_dynamic_sec(0)
  , dot_gnu_version_sec(0)
  , dot_gnu_version_r_sec(0)
  , dot_gnu_version_d_sec(0)
  , dt_soname(0)
 {

  const uint64_t FileSize = Data->getBufferSize();

  if (sizeof(Elf_Ehdr) > FileSize)
    // FIXME: Proper error handling.
    report_fatal_error("File too short!");

  Header = reinterpret_cast<const Elf_Ehdr *>(base());

  if (Header->e_shoff == 0)
    return;

  const uint64_t SectionTableOffset = Header->e_shoff;

  if (SectionTableOffset + sizeof(Elf_Shdr) > FileSize)
    // FIXME: Proper error handling.
    report_fatal_error("Section header table goes past end of file!");

  // The getNumSections() call below depends on SectionHeaderTable being set.
  SectionHeaderTable =
    reinterpret_cast<const Elf_Shdr *>(base() + SectionTableOffset);
  const uint64_t SectionTableSize = getNumSections() * Header->e_shentsize;

  if (SectionTableOffset + SectionTableSize > FileSize)
    // FIXME: Proper error handling.
    report_fatal_error("Section table goes past end of file!");

  // To find the symbol tables we walk the section table to find SHT_SYMTAB.
  const Elf_Shdr* SymbolTableSectionHeaderIndex = 0;
  const Elf_Shdr* sh = SectionHeaderTable;

  // Reserve SymbolTableSections[0] for .dynsym
  SymbolTableSections.push_back(NULL);

  for (uint64_t i = 0, e = getNumSections(); i != e; ++i) {
    switch (sh->sh_type) {
    case ELF::SHT_SYMTAB_SHNDX: {
      if (SymbolTableSectionHeaderIndex)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .symtab_shndx!");
      SymbolTableSectionHeaderIndex = sh;
      break;
    }
    case ELF::SHT_SYMTAB: {
      SymbolTableSectionsIndexMap[i] = SymbolTableSections.size();
      SymbolTableSections.push_back(sh);
      break;
    }
    case ELF::SHT_DYNSYM: {
      if (SymbolTableSections[0] != NULL)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .dynsym!");
      SymbolTableSectionsIndexMap[i] = 0;
      SymbolTableSections[0] = sh;
      break;
    }
    case ELF::SHT_REL:
    case ELF::SHT_RELA: {
      SectionRelocMap[getSection(sh->sh_info)].push_back(i);
      break;
    }
    case ELF::SHT_DYNAMIC: {
      if (dot_dynamic_sec != NULL)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .dynamic!");
      dot_dynamic_sec = sh;
      break;
    }
    case ELF::SHT_GNU_versym: {
      if (dot_gnu_version_sec != NULL)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .gnu.version section!");
      dot_gnu_version_sec = sh;
      break;
    }
    case ELF::SHT_GNU_verdef: {
      if (dot_gnu_version_d_sec != NULL)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .gnu.version_d section!");
      dot_gnu_version_d_sec = sh;
      break;
    }
    case ELF::SHT_GNU_verneed: {
      if (dot_gnu_version_r_sec != NULL)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .gnu.version_r section!");
      dot_gnu_version_r_sec = sh;
      break;
    }
    }
    ++sh;
  }

  // Sort section relocation lists by index.
  for (typename RelocMap_t::iterator i = SectionRelocMap.begin(),
                                     e = SectionRelocMap.end(); i != e; ++i) {
    std::sort(i->second.begin(), i->second.end());
  }

  // Get string table sections.
  dot_shstrtab_sec = getSection(getStringTableIndex());
  if (dot_shstrtab_sec) {
    // Verify that the last byte in the string table in a null.
    VerifyStrTab(dot_shstrtab_sec);
  }

  // Merge this into the above loop.
  for (const char *i = reinterpret_cast<const char *>(SectionHeaderTable),
                  *e = i + getNumSections() * Header->e_shentsize;
                   i != e; i += Header->e_shentsize) {
    const Elf_Shdr *sh = reinterpret_cast<const Elf_Shdr*>(i);
    if (sh->sh_type == ELF::SHT_STRTAB) {
      StringRef SectionName(getString(dot_shstrtab_sec, sh->sh_name));
      if (SectionName == ".strtab") {
        if (dot_strtab_sec != 0)
          // FIXME: Proper error handling.
          report_fatal_error("Already found section named .strtab!");
        dot_strtab_sec = sh;
        VerifyStrTab(dot_strtab_sec);
      } else if (SectionName == ".dynstr") {
        if (dot_dynstr_sec != 0)
          // FIXME: Proper error handling.
          report_fatal_error("Already found section named .dynstr!");
        dot_dynstr_sec = sh;
        VerifyStrTab(dot_dynstr_sec);
      }
    }
  }

  // Build symbol name side-mapping if there is one.
  if (SymbolTableSectionHeaderIndex) {
    const Elf_Word *ShndxTable = reinterpret_cast<const Elf_Word*>(base() +
                                      SymbolTableSectionHeaderIndex->sh_offset);
    error_code ec;
    for (symbol_iterator si = begin_symbols(),
                         se = end_symbols(); si != se; si.increment(ec)) {
      if (ec)
        report_fatal_error("Fewer extended symbol table entries than symbols!");
      if (*ShndxTable != ELF::SHN_UNDEF)
        ExtendedSymbolTable[getSymbol(si->getRawDataRefImpl())] = *ShndxTable;
      ++ShndxTable;
    }
  }
}

template<support::endianness target_endianness, bool is64Bits>
symbol_iterator ELFObjectFile<target_endianness, is64Bits>
                             ::begin_symbols() const {
  DataRefImpl SymbolData;
  if (SymbolTableSections.size() <= 1) {
    SymbolData.d.a = std::numeric_limits<uint32_t>::max();
    SymbolData.d.b = std::numeric_limits<uint32_t>::max();
  } else {
    SymbolData.d.a = 1; // The 0th symbol in ELF is fake.
    SymbolData.d.b = 1; // The 0th table is .dynsym
  }
  return symbol_iterator(SymbolRef(SymbolData, this));
}

template<support::endianness target_endianness, bool is64Bits>
symbol_iterator ELFObjectFile<target_endianness, is64Bits>
                             ::end_symbols() const {
  DataRefImpl SymbolData;
  SymbolData.d.a = std::numeric_limits<uint32_t>::max();
  SymbolData.d.b = std::numeric_limits<uint32_t>::max();
  return symbol_iterator(SymbolRef(SymbolData, this));
}

template<support::endianness target_endianness, bool is64Bits>
symbol_iterator ELFObjectFile<target_endianness, is64Bits>
                             ::begin_dynamic_symbols() const {
  DataRefImpl SymbolData;
  if (SymbolTableSections[0] == NULL) {
    SymbolData.d.a = std::numeric_limits<uint32_t>::max();
    SymbolData.d.b = std::numeric_limits<uint32_t>::max();
  } else {
    SymbolData.d.a = 1; // The 0th symbol in ELF is fake.
    SymbolData.d.b = 0; // The 0th table is .dynsym
  }
  return symbol_iterator(SymbolRef(SymbolData, this));
}

template<support::endianness target_endianness, bool is64Bits>
symbol_iterator ELFObjectFile<target_endianness, is64Bits>
                             ::end_dynamic_symbols() const {
  DataRefImpl SymbolData;
  SymbolData.d.a = std::numeric_limits<uint32_t>::max();
  SymbolData.d.b = std::numeric_limits<uint32_t>::max();
  return symbol_iterator(SymbolRef(SymbolData, this));
}

template<support::endianness target_endianness, bool is64Bits>
section_iterator ELFObjectFile<target_endianness, is64Bits>
                              ::begin_sections() const {
  DataRefImpl ret;
  ret.p = reinterpret_cast<intptr_t>(base() + Header->e_shoff);
  return section_iterator(SectionRef(ret, this));
}

template<support::endianness target_endianness, bool is64Bits>
section_iterator ELFObjectFile<target_endianness, is64Bits>
                              ::end_sections() const {
  DataRefImpl ret;
  ret.p = reinterpret_cast<intptr_t>(base()
                                     + Header->e_shoff
                                     + (Header->e_shentsize*getNumSections()));
  return section_iterator(SectionRef(ret, this));
}

template<support::endianness target_endianness, bool is64Bits>
typename ELFObjectFile<target_endianness, is64Bits>::dyn_iterator
ELFObjectFile<target_endianness, is64Bits>::begin_dynamic_table() const {
  DataRefImpl DynData;
  if (dot_dynamic_sec == NULL || dot_dynamic_sec->sh_size == 0) {
    DynData.d.a = std::numeric_limits<uint32_t>::max();
  } else {
    DynData.d.a = 0;
  }
  return dyn_iterator(DynRef(DynData, this));
}

template<support::endianness target_endianness, bool is64Bits>
typename ELFObjectFile<target_endianness, is64Bits>::dyn_iterator
ELFObjectFile<target_endianness, is64Bits>
                          ::end_dynamic_table() const {
  DataRefImpl DynData;
  DynData.d.a = std::numeric_limits<uint32_t>::max();
  return dyn_iterator(DynRef(DynData, this));
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getDynNext(DataRefImpl DynData,
                                     DynRef &Result) const {
  ++DynData.d.a;

  // Check to see if we are at the end of .dynamic
  if (DynData.d.a >= dot_dynamic_sec->getEntityCount()) {
    // We are at the end. Return the terminator.
    DynData.d.a = std::numeric_limits<uint32_t>::max();
  }

  Result = DynRef(DynData, this);
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
StringRef
ELFObjectFile<target_endianness, is64Bits>::getLoadName() const {
  if (!dt_soname) {
    // Find the DT_SONAME entry
    dyn_iterator it = begin_dynamic_table();
    dyn_iterator ie = end_dynamic_table();
    error_code ec;
    while (it != ie) {
      if (it->getTag() == ELF::DT_SONAME)
        break;
      it.increment(ec);
      if (ec)
        report_fatal_error("dynamic table iteration failed");
    }
    if (it != ie) {
      if (dot_dynstr_sec == NULL)
        report_fatal_error("Dynamic string table is missing");
      dt_soname = getString(dot_dynstr_sec, it->getVal());
    } else {
      dt_soname = "";
    }
  }
  return dt_soname;
}

template<support::endianness target_endianness, bool is64Bits>
library_iterator ELFObjectFile<target_endianness, is64Bits>
                             ::begin_libraries_needed() const {
  // Find the first DT_NEEDED entry
  dyn_iterator i = begin_dynamic_table();
  dyn_iterator e = end_dynamic_table();
  error_code ec;
  while (i != e) {
    if (i->getTag() == ELF::DT_NEEDED)
      break;
    i.increment(ec);
    if (ec)
      report_fatal_error("dynamic table iteration failed");
  }
  // Use the same DataRefImpl format as DynRef.
  return library_iterator(LibraryRef(i->getRawDataRefImpl(), this));
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getLibraryNext(DataRefImpl Data,
                                         LibraryRef &Result) const {
  // Use the same DataRefImpl format as DynRef.
  dyn_iterator i = dyn_iterator(DynRef(Data, this));
  dyn_iterator e = end_dynamic_table();

  // Skip the current dynamic table entry.
  error_code ec;
  if (i != e) {
    i.increment(ec);
    // TODO: proper error handling
    if (ec)
      report_fatal_error("dynamic table iteration failed");
  }

  // Find the next DT_NEEDED entry.
  while (i != e) {
    if (i->getTag() == ELF::DT_NEEDED)
      break;
    i.increment(ec);
    if (ec)
      report_fatal_error("dynamic table iteration failed");
  }
  Result = LibraryRef(i->getRawDataRefImpl(), this);
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
         ::getLibraryPath(DataRefImpl Data, StringRef &Res) const {
  dyn_iterator i = dyn_iterator(DynRef(Data, this));
  if (i == end_dynamic_table())
    report_fatal_error("getLibraryPath() called on iterator end");

  if (i->getTag() != ELF::DT_NEEDED)
    report_fatal_error("Invalid library_iterator");

  // This uses .dynstr to lookup the name of the DT_NEEDED entry.
  // THis works as long as DT_STRTAB == .dynstr. This is true most of
  // the time, but the specification allows exceptions.
  // TODO: This should really use DT_STRTAB instead. Doing this requires
  // reading the program headers.
  if (dot_dynstr_sec == NULL)
    report_fatal_error("Dynamic string table is missing");
  Res = getString(dot_dynstr_sec, i->getVal());
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
library_iterator ELFObjectFile<target_endianness, is64Bits>
                             ::end_libraries_needed() const {
  dyn_iterator e = end_dynamic_table();
  // Use the same DataRefImpl format as DynRef.
  return library_iterator(LibraryRef(e->getRawDataRefImpl(), this));
}

template<support::endianness target_endianness, bool is64Bits>
uint8_t ELFObjectFile<target_endianness, is64Bits>::getBytesInAddress() const {
  return is64Bits ? 8 : 4;
}

template<support::endianness target_endianness, bool is64Bits>
StringRef ELFObjectFile<target_endianness, is64Bits>
                       ::getFileFormatName() const {
  switch(Header->e_ident[ELF::EI_CLASS]) {
  case ELF::ELFCLASS32:
    switch(Header->e_machine) {
    case ELF::EM_386:
      return "ELF32-i386";
    case ELF::EM_X86_64:
      return "ELF32-x86-64";
    case ELF::EM_ARM:
      return "ELF32-arm";
    case ELF::EM_HEXAGON:
      return "ELF32-hexagon";
    default:
      return "ELF32-unknown";
    }
  case ELF::ELFCLASS64:
    switch(Header->e_machine) {
    case ELF::EM_386:
      return "ELF64-i386";
    case ELF::EM_X86_64:
      return "ELF64-x86-64";
    case ELF::EM_PPC64:
      return "ELF64-ppc64";
    default:
      return "ELF64-unknown";
    }
  default:
    // FIXME: Proper error handling.
    report_fatal_error("Invalid ELFCLASS!");
  }
}

template<support::endianness target_endianness, bool is64Bits>
unsigned ELFObjectFile<target_endianness, is64Bits>::getArch() const {
  switch(Header->e_machine) {
  case ELF::EM_386:
    return Triple::x86;
  case ELF::EM_X86_64:
    return Triple::x86_64;
  case ELF::EM_ARM:
    return Triple::arm;
  case ELF::EM_HEXAGON:
    return Triple::hexagon;
  case ELF::EM_MIPS:
    return (target_endianness == support::little) ?
           Triple::mipsel : Triple::mips;
  case ELF::EM_PPC64:
    return Triple::ppc64;
  default:
    return Triple::UnknownArch;
  }
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t ELFObjectFile<target_endianness, is64Bits>::getNumSections() const {
  assert(Header && "Header not initialized!");
  if (Header->e_shnum == ELF::SHN_UNDEF) {
    assert(SectionHeaderTable && "SectionHeaderTable not initialized!");
    return SectionHeaderTable->sh_size;
  }
  return Header->e_shnum;
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t
ELFObjectFile<target_endianness, is64Bits>::getStringTableIndex() const {
  if (Header->e_shnum == ELF::SHN_UNDEF) {
    if (Header->e_shstrndx == ELF::SHN_HIRESERVE)
      return SectionHeaderTable->sh_link;
    if (Header->e_shstrndx >= getNumSections())
      return 0;
  }
  return Header->e_shstrndx;
}


template<support::endianness target_endianness, bool is64Bits>
template<typename T>
inline const T *
ELFObjectFile<target_endianness, is64Bits>::getEntry(uint16_t Section,
                                                     uint32_t Entry) const {
  return getEntry<T>(getSection(Section), Entry);
}

template<support::endianness target_endianness, bool is64Bits>
template<typename T>
inline const T *
ELFObjectFile<target_endianness, is64Bits>::getEntry(const Elf_Shdr * Section,
                                                     uint32_t Entry) const {
  return reinterpret_cast<const T *>(
           base()
           + Section->sh_offset
           + (Entry * Section->sh_entsize));
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Sym *
ELFObjectFile<target_endianness, is64Bits>::getSymbol(DataRefImpl Symb) const {
  return getEntry<Elf_Sym>(SymbolTableSections[Symb.d.b], Symb.d.a);
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Dyn *
ELFObjectFile<target_endianness, is64Bits>::getDyn(DataRefImpl DynData) const {
  return getEntry<Elf_Dyn>(dot_dynamic_sec, DynData.d.a);
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Rel *
ELFObjectFile<target_endianness, is64Bits>::getRel(DataRefImpl Rel) const {
  return getEntry<Elf_Rel>(Rel.w.b, Rel.w.c);
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Rela *
ELFObjectFile<target_endianness, is64Bits>::getRela(DataRefImpl Rela) const {
  return getEntry<Elf_Rela>(Rela.w.b, Rela.w.c);
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Shdr *
ELFObjectFile<target_endianness, is64Bits>::getSection(DataRefImpl Symb) const {
  const Elf_Shdr *sec = getSection(Symb.d.b);
  if (sec->sh_type != ELF::SHT_SYMTAB || sec->sh_type != ELF::SHT_DYNSYM)
    // FIXME: Proper error handling.
    report_fatal_error("Invalid symbol table section!");
  return sec;
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Shdr *
ELFObjectFile<target_endianness, is64Bits>::getSection(uint32_t index) const {
  if (index == 0)
    return 0;
  if (!SectionHeaderTable || index >= getNumSections())
    // FIXME: Proper error handling.
    report_fatal_error("Invalid section index!");

  return reinterpret_cast<const Elf_Shdr *>(
         reinterpret_cast<const char *>(SectionHeaderTable)
         + (index * Header->e_shentsize));
}

template<support::endianness target_endianness, bool is64Bits>
const char *ELFObjectFile<target_endianness, is64Bits>
                         ::getString(uint32_t section,
                                     ELF::Elf32_Word offset) const {
  return getString(getSection(section), offset);
}

template<support::endianness target_endianness, bool is64Bits>
const char *ELFObjectFile<target_endianness, is64Bits>
                         ::getString(const Elf_Shdr *section,
                                     ELF::Elf32_Word offset) const {
  assert(section && section->sh_type == ELF::SHT_STRTAB && "Invalid section!");
  if (offset >= section->sh_size)
    // FIXME: Proper error handling.
    report_fatal_error("Symbol name offset outside of string table!");
  return (const char *)base() + section->sh_offset + offset;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolName(const Elf_Shdr *section,
                                        const Elf_Sym *symb,
                                        StringRef &Result) const {
  if (symb->st_name == 0) {
    const Elf_Shdr *section = getSection(symb);
    if (!section)
      Result = "";
    else
      Result = getString(dot_shstrtab_sec, section->sh_name);
    return object_error::success;
  }

  if (section == SymbolTableSections[0]) {
    // Symbol is in .dynsym, use .dynstr string table
    Result = getString(dot_dynstr_sec, symb->st_name);
  } else {
    // Use the default symbol table name section.
    Result = getString(dot_strtab_sec, symb->st_name);
  }
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSectionName(const Elf_Shdr *section,
                                        StringRef &Result) const {
  Result = StringRef(getString(dot_shstrtab_sec, section->sh_name));
  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFObjectFile<target_endianness, is64Bits>
                        ::getSymbolVersion(const Elf_Shdr *section,
                                           const Elf_Sym *symb,
                                           StringRef &Version,
                                           bool &IsDefault) const {
  // Handle non-dynamic symbols.
  if (section != SymbolTableSections[0]) {
    // Non-dynamic symbols can have versions in their names
    // A name of the form 'foo@V1' indicates version 'V1', non-default.
    // A name of the form 'foo@@V2' indicates version 'V2', default version.
    StringRef Name;
    error_code ec = getSymbolName(section, symb, Name);
    if (ec != object_error::success)
      return ec;
    size_t atpos = Name.find('@');
    if (atpos == StringRef::npos) {
      Version = "";
      IsDefault = false;
      return object_error::success;
    }
    ++atpos;
    if (atpos < Name.size() && Name[atpos] == '@') {
      IsDefault = true;
      ++atpos;
    } else {
      IsDefault = false;
    }
    Version = Name.substr(atpos);
    return object_error::success;
  }

  // This is a dynamic symbol. Look in the GNU symbol version table.
  if (dot_gnu_version_sec == NULL) {
    // No version table.
    Version = "";
    IsDefault = false;
    return object_error::success;
  }

  // Determine the position in the symbol table of this entry.
  const char *sec_start = (const char*)base() + section->sh_offset;
  size_t entry_index = ((const char*)symb - sec_start)/section->sh_entsize;

  // Get the corresponding version index entry
  const Elf_Versym *vs = getEntry<Elf_Versym>(dot_gnu_version_sec, entry_index);
  size_t version_index = vs->vs_index & ELF::VERSYM_VERSION;

  // Special markers for unversioned symbols.
  if (version_index == ELF::VER_NDX_LOCAL ||
      version_index == ELF::VER_NDX_GLOBAL) {
    Version = "";
    IsDefault = false;
    return object_error::success;
  }

  // Lookup this symbol in the version table
  LoadVersionMap();
  if (version_index >= VersionMap.size() || VersionMap[version_index].isNull())
    report_fatal_error("Symbol has version index without corresponding "
                       "define or reference entry");
  const VersionMapEntry &entry = VersionMap[version_index];

  // Get the version name string
  size_t name_offset;
  if (entry.isVerdef()) {
    // The first Verdaux entry holds the name.
    name_offset = entry.getVerdef()->getAux()->vda_name;
  } else {
    name_offset = entry.getVernaux()->vna_name;
  }
  Version = getString(dot_dynstr_sec, name_offset);

  // Set IsDefault
  if (entry.isVerdef()) {
    IsDefault = !(vs->vs_index & ELF::VERSYM_HIDDEN);
  } else {
    IsDefault = false;
  }

  return object_error::success;
}

template<support::endianness target_endianness, bool is64Bits>
inline DynRefImpl<target_endianness, is64Bits>
                 ::DynRefImpl(DataRefImpl DynP, const OwningType *Owner)
  : DynPimpl(DynP)
  , OwningObject(Owner) {}

template<support::endianness target_endianness, bool is64Bits>
inline bool DynRefImpl<target_endianness, is64Bits>
                      ::operator==(const DynRefImpl &Other) const {
  return DynPimpl == Other.DynPimpl;
}

template<support::endianness target_endianness, bool is64Bits>
inline bool DynRefImpl<target_endianness, is64Bits>
                      ::operator <(const DynRefImpl &Other) const {
  return DynPimpl < Other.DynPimpl;
}

template<support::endianness target_endianness, bool is64Bits>
inline error_code DynRefImpl<target_endianness, is64Bits>
                            ::getNext(DynRefImpl &Result) const {
  return OwningObject->getDynNext(DynPimpl, Result);
}

template<support::endianness target_endianness, bool is64Bits>
inline int64_t DynRefImpl<target_endianness, is64Bits>
                            ::getTag() const {
  return OwningObject->getDyn(DynPimpl)->d_tag;
}

template<support::endianness target_endianness, bool is64Bits>
inline uint64_t DynRefImpl<target_endianness, is64Bits>
                            ::getVal() const {
  return OwningObject->getDyn(DynPimpl)->d_un.d_val;
}

template<support::endianness target_endianness, bool is64Bits>
inline uint64_t DynRefImpl<target_endianness, is64Bits>
                            ::getPtr() const {
  return OwningObject->getDyn(DynPimpl)->d_un.d_ptr;
}

template<support::endianness target_endianness, bool is64Bits>
inline DataRefImpl DynRefImpl<target_endianness, is64Bits>
                             ::getRawDataRefImpl() const {
  return DynPimpl;
}

/// This is a generic interface for retrieving GNU symbol version
/// information from an ELFObjectFile.
static inline error_code GetELFSymbolVersion(const ObjectFile *Obj,
                                             const SymbolRef &Sym,
                                             StringRef &Version,
                                             bool &IsDefault) {
  // Little-endian 32-bit
  if (const ELFObjectFile<support::little, false> *ELFObj =
          dyn_cast<ELFObjectFile<support::little, false> >(Obj))
    return ELFObj->getSymbolVersion(Sym, Version, IsDefault);

  // Big-endian 32-bit
  if (const ELFObjectFile<support::big, false> *ELFObj =
          dyn_cast<ELFObjectFile<support::big, false> >(Obj))
    return ELFObj->getSymbolVersion(Sym, Version, IsDefault);

  // Little-endian 64-bit
  if (const ELFObjectFile<support::little, true> *ELFObj =
          dyn_cast<ELFObjectFile<support::little, true> >(Obj))
    return ELFObj->getSymbolVersion(Sym, Version, IsDefault);

  // Big-endian 64-bit
  if (const ELFObjectFile<support::big, true> *ELFObj =
          dyn_cast<ELFObjectFile<support::big, true> >(Obj))
    return ELFObj->getSymbolVersion(Sym, Version, IsDefault);

  llvm_unreachable("Object passed to GetELFSymbolVersion() is not ELF");
}

}
}

#endif
