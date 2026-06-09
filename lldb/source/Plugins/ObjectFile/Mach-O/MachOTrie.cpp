//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MachOTrie.h"

#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Flags.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"

#include <cstdio>

using namespace lldb;
using namespace lldb_private;
using namespace llvm::MachO;

void TrieEntry::Dump() const {
  printf("0x%16.16llx 0x%16.16llx 0x%16.16llx \"%s\"",
         static_cast<unsigned long long>(address),
         static_cast<unsigned long long>(flags),
         static_cast<unsigned long long>(other), name.GetCString());
  if (import_name)
    printf(" -> \"%s\"\n", import_name.GetCString());
  else
    printf("\n");
}

void TrieEntryWithOffset::Dump(uint32_t idx) const {
  printf("[%3u] 0x%16.16llx: ", idx,
         static_cast<unsigned long long>(nodeOffset));
  entry.Dump();
}

namespace {

bool ParseTrieEntriesImpl(DataExtractor &data, lldb::offset_t offset,
                          const bool is_arm, addr_t text_seg_base_addr,
                          std::string &prefix,
                          std::set<lldb::addr_t> &resolver_addresses,
                          std::vector<TrieEntryWithOffset> &reexports,
                          std::vector<TrieEntryWithOffset> &ext_symbols,
                          std::set<lldb::offset_t> &visited_nodes) {
  if (!data.ValidOffset(offset))
    return true;

  // Every node in a well-formed trie is reached by exactly one path, so a node
  // offset seen twice means the trie is corrupt.
  if (!visited_nodes.insert(offset).second)
    return false;

  // Terminal node -- end of a branch, possibly add this to
  // the symbol table or resolver table.
  const uint64_t terminalSize = data.GetULEB128(&offset);
  lldb::offset_t children_offset = offset + terminalSize;
  if (terminalSize != 0) {
    TrieEntryWithOffset e(offset);
    e.entry.flags = data.GetULEB128(&offset);
    const char *import_name = nullptr;
    if (e.entry.flags & EXPORT_SYMBOL_FLAGS_REEXPORT) {
      e.entry.address = 0;
      e.entry.other = data.GetULEB128(&offset); // dylib ordinal
      import_name = data.GetCStr(&offset);
    } else {
      e.entry.address = data.GetULEB128(&offset);
      if (text_seg_base_addr != LLDB_INVALID_ADDRESS)
        e.entry.address += text_seg_base_addr;
      if (e.entry.flags & EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER) {
        e.entry.other = data.GetULEB128(&offset);
        uint64_t resolver_addr = e.entry.other;
        if (text_seg_base_addr != LLDB_INVALID_ADDRESS)
          resolver_addr += text_seg_base_addr;
        if (is_arm)
          resolver_addr &= THUMB_ADDRESS_BIT_MASK;
        resolver_addresses.insert(resolver_addr);
      } else
        e.entry.other = 0;
    }
    bool add_this_entry = false;
    if (Flags(e.entry.flags).Test(EXPORT_SYMBOL_FLAGS_REEXPORT) &&
        import_name && import_name[0]) {
      // add symbols that are reexport symbols with a valid import name.
      add_this_entry = true;
    } else if (e.entry.flags == 0 &&
               (import_name == nullptr || import_name[0] == '\0')) {
      // add externally visible symbols, in case the nlist record has
      // been stripped/omitted.
      add_this_entry = true;
    }
    if (add_this_entry) {
      if (prefix.size() > 1) {
        // Skip the leading '_'
        e.entry.name.SetString(llvm::StringRef(prefix).drop_front());
      }
      if (import_name) {
        // Skip the leading '_'
        e.entry.import_name.SetCString(import_name + 1);
      }
      if (Flags(e.entry.flags).Test(EXPORT_SYMBOL_FLAGS_REEXPORT)) {
        reexports.push_back(e);
      } else {
        if (is_arm && (e.entry.address & 1)) {
          e.entry.flags |= TRIE_SYMBOL_IS_THUMB;
          e.entry.address &= THUMB_ADDRESS_BIT_MASK;
        }
        ext_symbols.push_back(e);
      }
    }
  }

  const uint8_t childrenCount = data.GetU8(&children_offset);
  for (uint8_t i = 0; i < childrenCount; ++i) {
    const char *cstr = data.GetCStr(&children_offset);
    if (!cstr)
      return false; // Corrupt data
    const size_t prevSize = prefix.size();
    prefix.append(cstr);
    lldb::offset_t childNodeOffset = data.GetULEB128(&children_offset);
    if (childNodeOffset) {
      if (!ParseTrieEntriesImpl(data, childNodeOffset, is_arm,
                                text_seg_base_addr, prefix, resolver_addresses,
                                reexports, ext_symbols, visited_nodes)) {
        return false;
      }
    }
    prefix.resize(prevSize);
  }
  return true;
}

} // namespace

bool lldb_private::ParseTrieEntries(
    DataExtractor &data, lldb::offset_t offset, const bool is_arm,
    lldb::addr_t text_seg_base_addr, std::set<lldb::addr_t> &resolver_addresses,
    std::vector<TrieEntryWithOffset> &reexports,
    std::vector<TrieEntryWithOffset> &ext_symbols) {
  std::set<lldb::offset_t> visited_nodes;
  std::string prefix;
  return ParseTrieEntriesImpl(data, offset, is_arm, text_seg_base_addr, prefix,
                              resolver_addresses, reexports, ext_symbols,
                              visited_nodes);
}
