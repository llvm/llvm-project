//===-- DWARFDebugAbbrev.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFDEBUGABBREV_H
#define LLVM_DEBUGINFO_DWARFDEBUGABBREV_H

#include "DWARFAbbreviationDeclaration.h"
#include <list>
#include <map>
#include <vector>

namespace llvm {

typedef std::vector<DWARFAbbreviationDeclaration>
  DWARFAbbreviationDeclarationColl;
typedef DWARFAbbreviationDeclarationColl::iterator
  DWARFAbbreviationDeclarationCollIter;
typedef DWARFAbbreviationDeclarationColl::const_iterator
  DWARFAbbreviationDeclarationCollConstIter;

class DWARFAbbreviationDeclarationSet {
  uint32_t Offset;
  uint32_t IdxOffset;
  std::vector<DWARFAbbreviationDeclaration> Decls;
  public:
  DWARFAbbreviationDeclarationSet()
    : Offset(0), IdxOffset(0) {}

  DWARFAbbreviationDeclarationSet(uint32_t offset, uint32_t idxOffset)
    : Offset(offset), IdxOffset(idxOffset) {}

  void clear() {
    IdxOffset = 0;
    Decls.clear();
  }
  uint32_t getOffset() const { return Offset; }
  void dump(raw_ostream &OS) const;
  bool extract(DataExtractor data, uint32_t* offset_ptr);

  const DWARFAbbreviationDeclaration *
    getAbbreviationDeclaration(uint32_t abbrCode) const;
};

class DWARFDebugAbbrev {
public:
  typedef std::map<uint64_t, DWARFAbbreviationDeclarationSet>
    DWARFAbbreviationDeclarationCollMap;
  typedef DWARFAbbreviationDeclarationCollMap::iterator
    DWARFAbbreviationDeclarationCollMapIter;
  typedef DWARFAbbreviationDeclarationCollMap::const_iterator
    DWARFAbbreviationDeclarationCollMapConstIter;

private:
  DWARFAbbreviationDeclarationCollMap AbbrevCollMap;
  mutable DWARFAbbreviationDeclarationCollMapConstIter PrevAbbrOffsetPos;

public:
  DWARFDebugAbbrev();
  const DWARFAbbreviationDeclarationSet *
    getAbbreviationDeclarationSet(uint64_t cu_abbr_offset) const;
  void dump(raw_ostream &OS) const;
  void parse(DataExtractor data);
};

}

#endif
