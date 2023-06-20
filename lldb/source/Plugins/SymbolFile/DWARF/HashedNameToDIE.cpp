//===-- HashedNameToDIE.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HashedNameToDIE.h"
#include "llvm/ADT/StringRef.h"

#include "lldb/Core/Mangled.h"

using namespace lldb_private::dwarf;

bool DWARFMappedHash::ExtractDIEArray(
    const DIEInfoArray &die_info_array,
    llvm::function_ref<bool(DIERef ref)> callback) {
  const size_t count = die_info_array.size();
  for (size_t i = 0; i < count; ++i)
    if (!callback(DIERef(die_info_array[i])))
      return false;
  return true;
}

const char *DWARFMappedHash::GetAtomTypeName(uint16_t atom) {
  switch (atom) {
  case eAtomTypeNULL:
    return "NULL";
  case eAtomTypeDIEOffset:
    return "die-offset";
  case eAtomTypeCUOffset:
    return "cu-offset";
  case eAtomTypeTag:
    return "die-tag";
  case eAtomTypeNameFlags:
    return "name-flags";
  case eAtomTypeTypeFlags:
    return "type-flags";
  case eAtomTypeQualNameHash:
    return "qualified-name-hash";
  }
  return "<invalid>";
}

DWARFMappedHash::DIEInfo::DIEInfo(dw_offset_t o, dw_tag_t t, uint32_t f,
                                  uint32_t h)
    : die_offset(o), tag(t), type_flags(f), qualified_name_hash(h) {}

DWARFMappedHash::Prologue::Prologue(dw_offset_t _die_base_offset)
    : die_base_offset(_die_base_offset), atoms() {
  // Define an array of DIE offsets by first defining an array, and then define
  // the atom type for the array, in this case we have an array of DIE offsets.
  AppendAtom(eAtomTypeDIEOffset, DW_FORM_data4);
}

void DWARFMappedHash::Prologue::ClearAtoms() {
  hash_data_has_fixed_byte_size = true;
  min_hash_data_byte_size = 0;
  atom_mask = 0;
  atoms.clear();
}

bool DWARFMappedHash::Prologue::ContainsAtom(AtomType atom_type) const {
  return (atom_mask & (1u << atom_type)) != 0;
}

void DWARFMappedHash::Prologue::Clear() {
  die_base_offset = 0;
  ClearAtoms();
}

void DWARFMappedHash::Prologue::AppendAtom(AtomType type, dw_form_t form) {
  atoms.push_back({type, form});
  atom_mask |= 1u << type;
  switch (form) {
  default:
  case DW_FORM_indirect:
  case DW_FORM_exprloc:
  case DW_FORM_flag_present:
  case DW_FORM_ref_sig8:
    llvm_unreachable("Unhandled atom form");

  case DW_FORM_addrx:
  case DW_FORM_string:
  case DW_FORM_block:
  case DW_FORM_block1:
  case DW_FORM_sdata:
  case DW_FORM_udata:
  case DW_FORM_ref_udata:
  case DW_FORM_GNU_addr_index:
  case DW_FORM_GNU_str_index:
    hash_data_has_fixed_byte_size = false;
    [[fallthrough]];
  case DW_FORM_flag:
  case DW_FORM_data1:
  case DW_FORM_ref1:
  case DW_FORM_sec_offset:
    min_hash_data_byte_size += 1;
    break;

  case DW_FORM_block2:
    hash_data_has_fixed_byte_size = false;
    [[fallthrough]];
  case DW_FORM_data2:
  case DW_FORM_ref2:
    min_hash_data_byte_size += 2;
    break;

  case DW_FORM_block4:
    hash_data_has_fixed_byte_size = false;
    [[fallthrough]];
  case DW_FORM_data4:
  case DW_FORM_ref4:
  case DW_FORM_addr:
  case DW_FORM_ref_addr:
  case DW_FORM_strp:
    min_hash_data_byte_size += 4;
    break;

  case DW_FORM_data8:
  case DW_FORM_ref8:
    min_hash_data_byte_size += 8;
    break;
  }
}

lldb::offset_t
DWARFMappedHash::Prologue::Read(const lldb_private::DataExtractor &data,
                                lldb::offset_t offset) {
  ClearAtoms();

  die_base_offset = data.GetU32(&offset);

  const uint32_t atom_count = data.GetU32(&offset);
  if (atom_count == 0x00060003u) {
    // Old format, deal with contents of old pre-release format.
    while (data.GetU32(&offset)) {
      /* do nothing */;
    }

    // Hardcode to the only known value for now.
    AppendAtom(eAtomTypeDIEOffset, DW_FORM_data4);
  } else {
    for (uint32_t i = 0; i < atom_count; ++i) {
      AtomType type = (AtomType)data.GetU16(&offset);
      auto form = static_cast<dw_form_t>(data.GetU16(&offset));
      AppendAtom(type, form);
    }
  }
  return offset;
}

size_t DWARFMappedHash::Prologue::GetByteSize() const {
  // Add an extra count to the atoms size for the zero termination Atom that
  // gets written to disk.
  return sizeof(die_base_offset) + sizeof(uint32_t) +
         atoms.size() * sizeof(Atom);
}

size_t DWARFMappedHash::Prologue::GetMinimumHashDataByteSize() const {
  return min_hash_data_byte_size;
}

bool DWARFMappedHash::Prologue::HashDataHasFixedByteSize() const {
  return hash_data_has_fixed_byte_size;
}

size_t DWARFMappedHash::Header::GetByteSize(const HeaderData &header_data) {
  return header_data.GetByteSize();
}

lldb::offset_t DWARFMappedHash::Header::Read(lldb_private::DataExtractor &data,
                                             lldb::offset_t offset) {
  offset = MappedHash::Header<Prologue>::Read(data, offset);
  if (offset != UINT32_MAX) {
    offset = header_data.Read(data, offset);
  }
  return offset;
}

bool DWARFMappedHash::Header::Read(const lldb_private::DWARFDataExtractor &data,
                                   lldb::offset_t *offset_ptr,
                                   DIEInfo &hash_data) const {
  const size_t num_atoms = header_data.atoms.size();
  if (num_atoms == 0)
    return false;

  for (size_t i = 0; i < num_atoms; ++i) {
    DWARFFormValue form_value(nullptr, header_data.atoms[i].form);

    if (!form_value.ExtractValue(data, offset_ptr))
      return false;

    switch (header_data.atoms[i].type) {
    case eAtomTypeDIEOffset: // DIE offset, check form for encoding
      hash_data.die_offset =
          DWARFFormValue::IsDataForm(form_value.Form())
              ? form_value.Unsigned()
              : form_value.Reference(header_data.die_base_offset);
      break;

    case eAtomTypeTag: // DW_TAG value for the DIE
      hash_data.tag = (dw_tag_t)form_value.Unsigned();
      break;

    case eAtomTypeTypeFlags: // Flags from enum TypeFlags
      hash_data.type_flags = (uint32_t)form_value.Unsigned();
      break;

    case eAtomTypeQualNameHash: // Flags from enum TypeFlags
      hash_data.qualified_name_hash = form_value.Unsigned();
      break;

    default:
      // We can always skip atoms we don't know about.
      break;
    }
  }
  return hash_data.die_offset != DW_INVALID_OFFSET;
}

DWARFMappedHash::MemoryTable::MemoryTable(
    lldb_private::DWARFDataExtractor &table_data,
    const lldb_private::DWARFDataExtractor &string_table, const char *name)
    : MappedHash::MemoryTable<uint32_t, Header, DIEInfoArray>(table_data),
      m_data(table_data), m_string_table(string_table), m_name(name) {}

const char *
DWARFMappedHash::MemoryTable::GetStringForKeyType(KeyType key) const {
  // The key in the DWARF table is the .debug_str offset for the string
  return m_string_table.PeekCStr(key);
}

bool DWARFMappedHash::MemoryTable::ReadHashData(uint32_t hash_data_offset,
                                                HashData &hash_data) const {
  lldb::offset_t offset = hash_data_offset;
  // Skip string table offset that contains offset of hash name in .debug_str.
  offset += 4;
  const uint32_t count = m_data.GetU32(&offset);
  if (count > 0) {
    hash_data.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
      if (!m_header.Read(m_data, &offset, hash_data[i]))
        return false;
    }
  } else
    hash_data.clear();
  return true;
}

DWARFMappedHash::MemoryTable::Result
DWARFMappedHash::MemoryTable::GetHashDataForName(
    llvm::StringRef name, lldb::offset_t *hash_data_offset_ptr,
    Pair &pair) const {
  pair.key = m_data.GetU32(hash_data_offset_ptr);
  pair.value.clear();

  // If the key is zero, this terminates our chain of HashData objects for this
  // hash value.
  if (pair.key == 0)
    return eResultEndOfHashData;

  // There definitely should be a string for this string offset, if there
  // isn't, there is something wrong, return and error.
  const char *strp_cstr = m_string_table.PeekCStr(pair.key);
  if (strp_cstr == nullptr) {
    *hash_data_offset_ptr = UINT32_MAX;
    return eResultError;
  }

  const uint32_t count = m_data.GetU32(hash_data_offset_ptr);
  const size_t min_total_hash_data_size =
      count * m_header.header_data.GetMinimumHashDataByteSize();
  if (count > 0 && m_data.ValidOffsetForDataOfSize(*hash_data_offset_ptr,
                                                   min_total_hash_data_size)) {
    // We have at least one HashData entry, and we have enough data to parse at
    // least "count" HashData entries.

    // First make sure the entire C string matches...
    const bool match = name == strp_cstr;

    if (!match && m_header.header_data.HashDataHasFixedByteSize()) {
      // If the string doesn't match and we have fixed size data, we can just
      // add the total byte size of all HashData objects to the hash data
      // offset and be done...
      *hash_data_offset_ptr += min_total_hash_data_size;
    } else {
      // If the string does match, or we don't have fixed size data then we
      // need to read the hash data as a stream. If the string matches we also
      // append all HashData objects to the value array.
      for (uint32_t i = 0; i < count; ++i) {
        DIEInfo die_info;
        if (m_header.Read(m_data, hash_data_offset_ptr, die_info)) {
          // Only happened if the HashData of the string matched...
          if (match)
            pair.value.push_back(die_info);
        } else {
          // Something went wrong while reading the data.
          *hash_data_offset_ptr = UINT32_MAX;
          return eResultError;
        }
      }
    }
    // Return the correct response depending on if the string matched or not...
    if (match) {
      // The key (cstring) matches and we have lookup results!
      return eResultKeyMatch;
    } else {
      // The key doesn't match, this function will get called again for the
      // next key/value or the key terminator which in our case is a zero
      // .debug_str offset.
      return eResultKeyMismatch;
    }
  } else {
    *hash_data_offset_ptr = UINT32_MAX;
    return eResultError;
  }
}

bool DWARFMappedHash::MemoryTable::FindByName(
    llvm::StringRef name, llvm::function_ref<bool(DIERef ref)> callback) {
  if (name.empty())
    return true;

  DIEInfoArray die_info_array;
  FindByName(name, die_info_array);
  return DWARFMappedHash::ExtractDIEArray(die_info_array, callback);
}

void DWARFMappedHash::MemoryTable::FindByName(llvm::StringRef name,
                                              DIEInfoArray &die_info_array) {
  if (name.empty())
    return;

  Pair kv_pair;
  if (Find(name, kv_pair))
    die_info_array.swap(kv_pair.value);
}
