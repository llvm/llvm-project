//===-- DWARFAbbreviationDeclaration.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace dwarf;

void DWARFAbbreviationDeclaration::clear() {
  Code = 0;
  Tag = DW_TAG_null;
  HasChildren = false;
  AttributeSpecs.clear();
}

DWARFAbbreviationDeclaration::DWARFAbbreviationDeclaration() {
  clear();
}

bool
DWARFAbbreviationDeclaration::extract(DataExtractor Data, 
                                      uint32_t* OffsetPtr) {
  clear();
  Code = Data.getULEB128(OffsetPtr);
  if (Code == 0) {
    return false;
  }
  Tag = static_cast<llvm::dwarf::Tag>(Data.getULEB128(OffsetPtr));
  if (Tag == DW_TAG_null) {
    clear();
    return false;
  }
  uint8_t ChildrenByte = Data.getU8(OffsetPtr);
  HasChildren = (ChildrenByte == DW_CHILDREN_yes);

  while (true) {
    auto A = static_cast<Attribute>(Data.getULEB128(OffsetPtr));
    auto F = static_cast<Form>(Data.getULEB128(OffsetPtr));
    if (A && F) {
        AttributeSpecs.push_back(AttributeSpec(A, F));
    } else if (A == 0 && F == 0) {
      // We successfully reached the end of this abbreviation declaration
      // since both attribute and form are zero.
      break;
    } else {
      // Attribute and form pairs must either both be non-zero, in which case
      // they are added to the abbreviation declaration, or both be zero to
      // terminate the abbrevation declaration. In this case only one was
      // zero which is an error.
      clear();
      return false;
    }
  }
  return true;
}

void DWARFAbbreviationDeclaration::dump(raw_ostream &OS) const {
  auto tagString = TagString(getTag());
  OS << '[' << getCode() << "] ";
  if (!tagString.empty())
    OS << tagString;
  else
    OS << format("DW_TAG_Unknown_%x", getTag());
  OS << "\tDW_CHILDREN_" << (hasChildren() ? "yes" : "no") << '\n';
  for (const AttributeSpec &Spec : AttributeSpecs) {
    OS << '\t';
    auto attrString = AttributeString(Spec.Attr);
    if (!attrString.empty())
      OS << attrString;
    else
      OS << format("DW_AT_Unknown_%x", Spec.Attr);
    OS << '\t';
    auto formString = FormEncodingString(Spec.Form);
    if (!formString.empty())
      OS << formString;
    else
      OS << format("DW_FORM_Unknown_%x", Spec.Form);
    OS << '\n';
  }
  OS << '\n';
}

uint32_t
DWARFAbbreviationDeclaration::findAttributeIndex(dwarf::Attribute attr) const {
  for (uint32_t i = 0, e = AttributeSpecs.size(); i != e; ++i) {
    if (AttributeSpecs[i].Attr == attr)
      return i;
  }
  return -1U;
}
