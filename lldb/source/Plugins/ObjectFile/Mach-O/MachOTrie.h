//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_OBJECTFILE_MACH_O_MACHOTRIE_H
#define LLDB_SOURCE_PLUGINS_OBJECTFILE_MACH_O_MACHOTRIE_H

#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace lldb_private {

class DataExtractor;

/// Set on TrieEntry::flags for an ARM symbol whose address has the low Thumb
/// bit set; the bit is stripped from the address and recorded here instead.
inline constexpr uint64_t TRIE_SYMBOL_IS_THUMB = 1ULL << 63;

/// Mask that clears the low Thumb bit from an ARM function address.
inline constexpr uint64_t THUMB_ADDRESS_BIT_MASK = 0xfffffffffffffffeull;

/// A single symbol recovered from the Mach-O export trie.
struct TrieEntry {
  void Dump() const;

  ConstString name;
  uint64_t address = LLDB_INVALID_ADDRESS;
  uint64_t flags =
      0; // EXPORT_SYMBOL_FLAGS_REEXPORT, EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER,
         // TRIE_SYMBOL_IS_THUMB
  uint64_t other = 0;
  ConstString import_name;
};

/// A TrieEntry paired with the offset of the trie node it was parsed from.
struct TrieEntryWithOffset {
  lldb::offset_t nodeOffset;
  TrieEntry entry;

  TrieEntryWithOffset(lldb::offset_t offset) : nodeOffset(offset), entry() {}

  void Dump(uint32_t idx) const;

  bool operator<(const TrieEntryWithOffset &other) const {
    return (nodeOffset < other.nodeOffset);
  }
};

/// Parse the Mach-O export trie (the dyld symbol trie from LC_DYLD_INFO or
/// LC_DYLD_EXPORTS_TRIE) starting at \a offset in \a data, collecting exported
/// and re-exported symbols and any stub resolver addresses.
///
/// \param[in] data The buffer holding the raw export trie.
/// \param[in] offset The node offset to start parsing from (0 for the root).
/// \param[in] is_arm Whether the image is ARM, which governs Thumb-bit
/// handling.
/// \param[in] text_seg_base_addr The __TEXT segment file address added to each
///     symbol address, or LLDB_INVALID_ADDRESS to leave addresses unbiased.
/// \param[out] resolver_addresses Stub-and-resolver addresses encountered.
/// \param[out] reexports Re-export entries with a valid import name.
/// \param[out] ext_symbols Externally visible (non-re-export) entries.
///
/// \return false if the trie is detectably corrupt, true otherwise.
bool ParseTrieEntries(DataExtractor &data, lldb::offset_t offset,
                      const bool is_arm, lldb::addr_t text_seg_base_addr,
                      std::set<lldb::addr_t> &resolver_addresses,
                      std::vector<TrieEntryWithOffset> &reexports,
                      std::vector<TrieEntryWithOffset> &ext_symbols);

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_OBJECTFILE_MACH_O_MACHOTRIE_H
