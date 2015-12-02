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

enum DWARFSectionKind {
  DW_SECT_INFO = 1,
  DW_SECT_TYPES,
  DW_SECT_ABBREV,
  DW_SECT_LINE,
  DW_SECT_LOC,
  DW_SECT_STR_OFFSETS,
  DW_SECT_MACINFO,
  DW_SECT_MACRO,
};

class DWARFUnitIndex {
  struct Header {
    uint32_t Version;
    uint32_t NumColumns;
    uint32_t NumUnits;
    uint32_t NumBuckets = 0;

    bool parse(DataExtractor IndexData, uint32_t *OffsetPtr);
    void dump(raw_ostream &OS) const;
  };

public:
  class Entry {
  public:
    struct SectionContribution {
      uint32_t Offset;
      uint32_t Length;
    };

  private:
    const DWARFUnitIndex *Index;
    uint64_t Signature;
    std::unique_ptr<SectionContribution[]> Contributions;
    friend class DWARFUnitIndex;

  public:
    const SectionContribution *getOffset(DWARFSectionKind Sec) const;
    const SectionContribution *getOffset() const;
  };

private:
  struct Header Header;

  DWARFSectionKind InfoColumnKind;
  int InfoColumn = -1;
  std::unique_ptr<DWARFSectionKind[]> ColumnKinds;
  std::unique_ptr<Entry[]> Rows;

  static StringRef getColumnHeader(DWARFSectionKind DS);
  bool parseImpl(DataExtractor IndexData);

public:
  bool parse(DataExtractor IndexData);
  DWARFUnitIndex(DWARFSectionKind InfoColumnKind)
      : InfoColumnKind(InfoColumnKind) {}
  void dump(raw_ostream &OS) const;
  const Entry *getFromOffset(uint32_t Offset) const;
};
}

#endif
