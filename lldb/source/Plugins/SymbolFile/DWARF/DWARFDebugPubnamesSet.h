//===-- DWARFDebugPubnamesSet.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFDebugPubnamesSet_h_
#define SymbolFileDWARF_DWARFDebugPubnamesSet_h_

#include "SymbolFileDWARF.h"
#include <string>
#include <vector>
#if __cplusplus >= 201103L || defined(_MSC_VER)
#include <unordered_map>
#else
#include <ext/hash_map>
#endif

class DWARFDebugPubnamesSet {
public:
  struct Header {
    uint32_t length;  // length of the set of entries for this compilation unit,
                      // not including the length field itself
    uint16_t version; // The DWARF version number
    uint32_t die_offset; // compile unit .debug_info offset
    uint32_t die_length; // compile unit .debug_info length
    Header()
        : length(10), version(2), die_offset(DW_INVALID_OFFSET), die_length(0) {
    }
  };

  struct Descriptor {
    Descriptor() : offset(), name() {}

    Descriptor(dw_offset_t the_offset, const char *the_name)
        : offset(the_offset), name(the_name ? the_name : "") {}

    dw_offset_t offset;
    std::string name;
  };

  DWARFDebugPubnamesSet();
  DWARFDebugPubnamesSet(dw_offset_t debug_aranges_offset,
                        dw_offset_t cu_die_offset, dw_offset_t die_length);
  dw_offset_t GetOffset() const { return m_offset; }
  void SetOffset(dw_offset_t offset) { m_offset = offset; }
  DWARFDebugPubnamesSet::Header &GetHeader() { return m_header; }
  const DWARFDebugPubnamesSet::Header &GetHeader() const { return m_header; }
  const DWARFDebugPubnamesSet::Descriptor *GetDescriptor(uint32_t i) const {
    if (i < m_descriptors.size())
      return &m_descriptors[i];
    return NULL;
  }
  uint32_t NumDescriptors() const { return m_descriptors.size(); }
  void AddDescriptor(dw_offset_t cu_rel_offset, const char *name);
  void Clear();
  bool Extract(const lldb_private::DWARFDataExtractor &debug_pubnames_data,
               lldb::offset_t *offset_ptr);
  void Dump(lldb_private::Log *s) const;
  void InitNameIndexes() const;
  void Find(const char *name, bool ignore_case,
            std::vector<dw_offset_t> &die_offset_coll) const;
  void Find(const lldb_private::RegularExpression &regex,
            std::vector<dw_offset_t> &die_offsets) const;
  dw_offset_t GetOffsetOfNextEntry() const;

protected:
  typedef std::vector<Descriptor> DescriptorColl;
  typedef DescriptorColl::iterator DescriptorIter;
  typedef DescriptorColl::const_iterator DescriptorConstIter;

  dw_offset_t m_offset;
  Header m_header;
#if __cplusplus >= 201103L || defined(_MSC_VER)
  typedef std::unordered_multimap<const char *, uint32_t,
                                  std::hash<const char *>,
                                  CStringEqualBinaryPredicate>
      cstr_to_index_mmap;
#else
  typedef __gnu_cxx::hash_multimap<const char *, uint32_t,
                                   __gnu_cxx::hash<const char *>,
                                   CStringEqualBinaryPredicate>
      cstr_to_index_mmap;
#endif
  DescriptorColl m_descriptors;
  mutable cstr_to_index_mmap m_name_to_descriptor_index;
};

#endif // SymbolFileDWARF_DWARFDebugPubnamesSet_h_
