//===-- DWARFDebugAbbrev.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugAbbrev.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

bool DWARFAbbreviationDeclarationSet::extract(DataExtractor data,
                                              uint32_t* offset_ptr) {
  const uint32_t beginOffset = *offset_ptr;
  Offset = beginOffset;
  clear();
  DWARFAbbreviationDeclaration abbrevDeclaration;
  uint32_t prevAbbrAode = 0;
  while (abbrevDeclaration.extract(data, offset_ptr)) {
    Decls.push_back(abbrevDeclaration);
    if (IdxOffset == 0) {
      IdxOffset = abbrevDeclaration.getCode();
    } else {
      if (prevAbbrAode + 1 != abbrevDeclaration.getCode())
        IdxOffset = UINT32_MAX;// Out of order indexes, we can't do O(1) lookups
    }
    prevAbbrAode = abbrevDeclaration.getCode();
  }
  return beginOffset != *offset_ptr;
}

void DWARFAbbreviationDeclarationSet::dump(raw_ostream &OS) const {
  for (unsigned i = 0, e = Decls.size(); i != e; ++i)
    Decls[i].dump(OS);
}

const DWARFAbbreviationDeclaration*
DWARFAbbreviationDeclarationSet::getAbbreviationDeclaration(uint32_t abbrCode)
  const {
  if (IdxOffset == UINT32_MAX) {
    DWARFAbbreviationDeclarationCollConstIter pos;
    DWARFAbbreviationDeclarationCollConstIter end = Decls.end();
    for (pos = Decls.begin(); pos != end; ++pos) {
      if (pos->getCode() == abbrCode)
        return &(*pos);
    }
  } else {
    uint32_t idx = abbrCode - IdxOffset;
    if (idx < Decls.size())
      return &Decls[idx];
  }
  return NULL;
}

DWARFDebugAbbrev::DWARFDebugAbbrev() :
  AbbrevCollMap(),
  PrevAbbrOffsetPos(AbbrevCollMap.end()) {}


void DWARFDebugAbbrev::parse(DataExtractor data) {
  uint32_t offset = 0;

  while (data.isValidOffset(offset)) {
    uint32_t initial_cu_offset = offset;
    DWARFAbbreviationDeclarationSet abbrevDeclSet;

    if (abbrevDeclSet.extract(data, &offset))
      AbbrevCollMap[initial_cu_offset] = abbrevDeclSet;
    else
      break;
  }
  PrevAbbrOffsetPos = AbbrevCollMap.end();
}

void DWARFDebugAbbrev::dump(raw_ostream &OS) const {
  if (AbbrevCollMap.empty()) {
    OS << "< EMPTY >\n";
    return;
  }

  DWARFAbbreviationDeclarationCollMapConstIter pos;
  for (pos = AbbrevCollMap.begin(); pos != AbbrevCollMap.end(); ++pos) {
    OS << format("Abbrev table for offset: 0x%8.8" PRIx64 "\n", pos->first);
    pos->second.dump(OS);
  }
}

const DWARFAbbreviationDeclarationSet*
DWARFDebugAbbrev::getAbbreviationDeclarationSet(uint64_t cu_abbr_offset) const {
  DWARFAbbreviationDeclarationCollMapConstIter end = AbbrevCollMap.end();
  DWARFAbbreviationDeclarationCollMapConstIter pos;
  if (PrevAbbrOffsetPos != end &&
      PrevAbbrOffsetPos->first == cu_abbr_offset) {
    return &(PrevAbbrOffsetPos->second);
  } else {
    pos = AbbrevCollMap.find(cu_abbr_offset);
    PrevAbbrOffsetPos = pos;
  }

  if (pos != AbbrevCollMap.end())
    return &(pos->second);
  return NULL;
}
