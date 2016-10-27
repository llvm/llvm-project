//===-- DWARFAbbreviationDeclaration.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFABBREVIATIONDECLARATION_H
#define LLVM_LIB_DEBUGINFO_DWARFABBREVIATIONDECLARATION_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Dwarf.h"

namespace llvm {

class raw_ostream;

class DWARFAbbreviationDeclaration {
public:
  struct AttributeSpec {
    AttributeSpec(dwarf::Attribute A, dwarf::Form F) : Attr(A), Form(F) {}
    dwarf::Attribute Attr;
    dwarf::Form Form;
  };
  typedef SmallVector<AttributeSpec, 8> AttributeSpecVector;

  DWARFAbbreviationDeclaration();

  uint32_t getCode() const { return Code; }
  dwarf::Tag getTag() const { return Tag; }
  bool hasChildren() const { return HasChildren; }

  typedef iterator_range<AttributeSpecVector::const_iterator>
  attr_iterator_range;

  attr_iterator_range attributes() const {
    return attr_iterator_range(AttributeSpecs.begin(), AttributeSpecs.end());
  }

  dwarf::Form getFormByIndex(uint32_t idx) const {
    if (idx < AttributeSpecs.size())
      return AttributeSpecs[idx].Form;
    return dwarf::Form(0);
  }

  uint32_t findAttributeIndex(dwarf::Attribute attr) const;
  bool extract(DataExtractor Data, uint32_t* OffsetPtr);
  void dump(raw_ostream &OS) const;

private:
  void clear();

  uint32_t Code;
  dwarf::Tag Tag;
  bool HasChildren;

  AttributeSpecVector AttributeSpecs;
};

}

#endif
