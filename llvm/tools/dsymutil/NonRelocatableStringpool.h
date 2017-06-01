//===-- NonRelocatableStringpool.h - A simple stringpool  -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_DSYMUTIL_NONRELOCATABLESTRINGPOOL_H
#define LLVM_TOOLS_DSYMUTIL_NONRELOCATABLESTRINGPOOL_H

#include "llvm/ADT/StringMap.h"

namespace llvm {
namespace dsymutil {

/// \brief A string table that doesn't need relocations.
///
/// We are doing a final link, no need for a string table that
/// has relocation entries for every reference to it. This class
/// provides this ablitity by just associating offsets with
/// strings.
class NonRelocatableStringpool {
public:
  /// \brief Entries are stored into the StringMap and simply linked
  /// together through the second element of this pair in order to
  /// keep track of insertion order.
  typedef StringMap<std::pair<uint32_t, StringMapEntryBase *>, BumpPtrAllocator>
      MapTy;

  NonRelocatableStringpool()
      : CurrentEndOffset(0), Sentinel(0), Last(&Sentinel) {
    // Legacy dsymutil puts an empty string at the start of the line
    // table.
    getStringOffset("");
  }

  /// \brief Get the offset of string \p S in the string table. This
  /// can insert a new element or return the offset of a preexisitng
  /// one.
  uint32_t getStringOffset(StringRef S);

  /// \brief Get permanent storage for \p S (but do not necessarily
  /// emit \p S in the output section).
  /// \returns The StringRef that points to permanent storage to use
  /// in place of \p S.
  StringRef internString(StringRef S);

  // \brief Return the first entry of the string table.
  const MapTy::MapEntryTy *getFirstEntry() const {
    return getNextEntry(&Sentinel);
  }

  // \brief Get the entry following \p E in the string table or null
  // if \p E was the last entry.
  const MapTy::MapEntryTy *getNextEntry(const MapTy::MapEntryTy *E) const {
    return static_cast<const MapTy::MapEntryTy *>(E->getValue().second);
  }

  uint64_t getSize() { return CurrentEndOffset; }

private:
  MapTy Strings;
  uint32_t CurrentEndOffset;
  MapTy::MapEntryTy Sentinel, *Last;
};
}
}

#endif
