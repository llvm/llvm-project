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

typedef DenseMap<uint64_t, std::pair<uint8_t, int64_t>> RelocAddrMap;

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFRELOCMAP_H
