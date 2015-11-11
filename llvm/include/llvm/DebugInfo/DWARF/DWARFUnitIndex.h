//===-- DWARFUnitIndex.h --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFUNITINDEX_H
#define LLVM_LIB_DEBUGINFO_DWARFUNITINDEX_H

#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace llvm {

class DWARFUnitIndex {
  class Header {
    uint32_t Version;
    uint32_t NumColumns;
    uint32_t NumUnits;
    uint32_t NumBuckets;

  public:
    bool parse(DataExtractor IndexData, uint32_t *OffsetPtr);
    void dump(raw_ostream &OS) const;
  };

  class Header Header;

public:
  bool parse(DataExtractor IndexData);
  void dump(raw_ostream &OS) const;
};
}

#endif
