//===-- DWARFAbbreviationDeclaration.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFABBREVIATIONDECLARATION_H
#define LLVM_DEBUGINFO_DWARFABBREVIATIONDECLARATION_H

#include "DWARFAttribute.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {

class raw_ostream;

class DWARFAbbreviationDeclaration {
  uint32_t Code;
  uint32_t Tag;
  bool HasChildren;
  SmallVector<DWARFAttribute, 8> Attributes;
public:
  enum { InvalidCode = 0 };
  DWARFAbbreviationDeclaration()
    : Code(InvalidCode), Tag(0), HasChildren(0) {}

  uint32_t getCode() const { return Code; }
  uint32_t getTag() const { return Tag; }
  bool hasChildren() const { return HasChildren; }
  uint32_t getNumAttributes() const { return Attributes.size(); }
  uint16_t getAttrByIndex(uint32_t idx) const {
    return Attributes.size() > idx ? Attributes[idx].getAttribute() : 0;
  }
  uint16_t getFormByIndex(uint32_t idx) const {
    return Attributes.size() > idx ? Attributes[idx].getForm() : 0;
  }

  uint32_t findAttributeIndex(uint16_t attr) const;
  bool extract(DataExtractor data, uint32_t* offset_ptr);
  bool extract(DataExtractor data, uint32_t* offset_ptr, uint32_t code);
  bool isValid() const { return Code != 0 && Tag != 0; }
  void dump(raw_ostream &OS) const;
  const SmallVectorImpl<DWARFAttribute> &getAttributes() const {
    return Attributes;
  }
};

}

#endif
