//===- DWARFRelocMap.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFRELOCMAP_H
#define LLVM_DEBUGINFO_DWARF_DWARFRELOCMAP_H

#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <utility>

namespace llvm {

struct RelocAddrEntry {
  int64_t Value;
};

/// In place of applying the relocations to the data we've read from disk we use
/// a separate mapping table to the side and checking that at locations in the
/// dwarf where we expect relocated values. This adds a bit of complexity to the
/// dwarf parsing/extraction at the benefit of not allocating memory for the
/// entire size of the debug info sections.
typedef DenseMap<uint64_t, RelocAddrEntry> RelocAddrMap;

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFRELOCMAP_H
