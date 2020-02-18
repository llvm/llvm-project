//===-- HashedNameToDIE.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_HASHEDNAMETODIE_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_HASHEDNAMETODIE_H

#include <vector>

#include "lldb/Core/MappedHash.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/lldb-defines.h"

#include "DWARFDefines.h"
#include "DWARFFormValue.h"
#include "NameToDIE.h"

class DWARFMappedHash {
public:
  enum AtomType : uint16_t {
    eAtomTypeNULL = 0u,
    /// DIE offset, check form for encoding.
    eAtomTypeDIEOffset = 1u,
    /// DIE offset of the compiler unit header that contains the item in
    /// question.
    eAtomTypeCUOffset = 2u,
    /// DW_TAG_xxx value, should be encoded as DW_FORM_data1 (if no tags exceed
    /// 255) or DW_FORM_data2.
    eAtomTypeTag = 3u,
    // Flags from enum NameFlags.
    eAtomTypeNameFlags = 4u,
    // Flags from enum TypeFlags.
    eAtomTypeTypeFlags = 5u,
    /// A 32 bit hash of the full qualified name (since all hash entries are
    /// basename only) For example a type like "std::vector<int>::iterator"
    /// would have a name of "iterator" and a 32 bit hash for
    /// "std::vector<int>::iterator" to allow us to not have to pull in debug
    /// info for a type when we know the fully qualified name.
    eAtomTypeQualNameHash = 6u
  };

  /// Bit definitions for the eAtomTypeTypeFlags flags.
  enum TypeFlags {
    /// Always set for C++, only set for ObjC if this is the
    /// @implementation for class.
    eTypeFlagClassIsImplementation = (1u << 1)
  };

  struct DIEInfo {
    dw_offset_t die_offset = DW_INVALID_OFFSET;
    dw_tag_t tag = llvm::dwarf::DW_TAG_null;

    /// Any flags for this DIEInfo.
    uint32_t type_flags = 0;

    /// A 32 bit hash of the fully qualified name.
    uint32_t qualified_name_hash = 0;

    DIEInfo() = default;
    DIEInfo(dw_offset_t o, dw_tag_t t, uint32_t f, uint32_t h);

    explicit operator DIERef() const {
      return DIERef(llvm::None, DIERef::Section::DebugInfo, die_offset);
    }
  };

  struct Atom {
    AtomType type;
    dw_form_t form;
  };

  typedef std::vector<DIEInfo> DIEInfoArray;
  typedef std::vector<Atom> AtomArray;

  class Prologue {
  public:
    Prologue(dw_offset_t _die_base_offset = 0);

    void ClearAtoms();

    bool ContainsAtom(AtomType atom_type) const;

    void Clear();

    void AppendAtom(AtomType type, dw_form_t form);

    lldb::offset_t Read(const lldb_private::DataExtractor &data,
                        lldb::offset_t offset);

    size_t GetByteSize() const;

    size_t GetMinimumHashDataByteSize() const;

    bool HashDataHasFixedByteSize() const;

    /// DIE offset base so die offsets in hash_data can be CU relative.
    dw_offset_t die_base_offset;
    AtomArray atoms;
    uint32_t atom_mask;
    size_t min_hash_data_byte_size;
    bool hash_data_has_fixed_byte_size;
  };

  class Header : public MappedHash::Header<Prologue> {
  public:
    size_t GetByteSize(const HeaderData &header_data) override;

    lldb::offset_t Read(lldb_private::DataExtractor &data,
                        lldb::offset_t offset) override;

    bool Read(const lldb_private::DWARFDataExtractor &data,
              lldb::offset_t *offset_ptr, DIEInfo &hash_data) const;
  };

  /// A class for reading and using a saved hash table from a block of data in
  /// memory.
  class MemoryTable
      : public MappedHash::MemoryTable<uint32_t, DWARFMappedHash::Header,
                                       DIEInfoArray> {
  public:
    MemoryTable(lldb_private::DWARFDataExtractor &table_data,
                const lldb_private::DWARFDataExtractor &string_table,
                const char *name);

    const char *GetStringForKeyType(KeyType key) const override;

    bool ReadHashData(uint32_t hash_data_offset,
                      HashData &hash_data) const override;

    size_t
    AppendAllDIEsThatMatchingRegex(const lldb_private::RegularExpression &regex,
                                   DIEInfoArray &die_info_array) const;

    size_t AppendAllDIEsInRange(const uint32_t die_offset_start,
                                const uint32_t die_offset_end,
                                DIEInfoArray &die_info_array) const;

    size_t FindByName(llvm::StringRef name, DIEArray &die_offsets);

    size_t FindByNameAndTag(llvm::StringRef name, const dw_tag_t tag,
                            DIEArray &die_offsets);

    size_t FindByNameAndTagAndQualifiedNameHash(
        llvm::StringRef name, const dw_tag_t tag,
        const uint32_t qualified_name_hash, DIEArray &die_offsets);

    size_t FindCompleteObjCClassByName(llvm::StringRef name,
                                       DIEArray &die_offsets,
                                       bool must_be_implementation);

  protected:
    Result AppendHashDataForRegularExpression(
        const lldb_private::RegularExpression &regex,
        lldb::offset_t *hash_data_offset_ptr, Pair &pair) const;

    size_t FindByName(llvm::StringRef name, DIEInfoArray &die_info_array);

    Result GetHashDataForName(llvm::StringRef name,
                              lldb::offset_t *hash_data_offset_ptr,
                              Pair &pair) const override;

    lldb_private::DWARFDataExtractor m_data;
    lldb_private::DWARFDataExtractor m_string_table;
    std::string m_name;
  };

  static void ExtractDIEArray(const DIEInfoArray &die_info_array,
                              DIEArray &die_offsets);

protected:
  static void ExtractDIEArray(const DIEInfoArray &die_info_array,
                              const dw_tag_t tag, DIEArray &die_offsets);

  static void ExtractDIEArray(const DIEInfoArray &die_info_array,
                              const dw_tag_t tag,
                              const uint32_t qualified_name_hash,
                              DIEArray &die_offsets);

  static void
  ExtractClassOrStructDIEArray(const DIEInfoArray &die_info_array,
                               bool return_implementation_only_if_available,
                               DIEArray &die_offsets);

  static void ExtractTypesFromDIEArray(const DIEInfoArray &die_info_array,
                                       uint32_t type_flag_mask,
                                       uint32_t type_flag_value,
                                       DIEArray &die_offsets);

  static const char *GetAtomTypeName(uint16_t atom);
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_HASHEDNAMETODIE_H
