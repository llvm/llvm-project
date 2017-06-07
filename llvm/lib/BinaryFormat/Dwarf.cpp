//===-- llvm/BinaryFormat/Dwarf.cpp - Dwarf Framework ------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for generic dwarf information.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace dwarf;

StringRef llvm::dwarf::TagString(unsigned Tag) {
  switch (Tag) {
  default:
    return StringRef();
#define HANDLE_DW_TAG(ID, NAME, VERSION, VENDOR)                               \
  case DW_TAG_##NAME:                                                          \
    return "DW_TAG_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::getTag(StringRef TagString) {
  return StringSwitch<unsigned>(TagString)
#define HANDLE_DW_TAG(ID, NAME, VERSION, VENDOR)                               \
  .Case("DW_TAG_" #NAME, DW_TAG_##NAME)
#include "llvm/BinaryFormat/Dwarf.def"
      .Default(DW_TAG_invalid);
}

unsigned llvm::dwarf::TagVersion(dwarf::Tag Tag) {
  switch (Tag) {
  default:
    return 0;
#define HANDLE_DW_TAG(ID, NAME, VERSION, VENDOR)                               \
  case DW_TAG_##NAME:                                                          \
    return VERSION;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::TagVendor(dwarf::Tag Tag) {
  switch (Tag) {
  default:
    return 0;
#define HANDLE_DW_TAG(ID, NAME, VERSION, VENDOR)                               \
  case DW_TAG_##NAME:                                                          \
    return DWARF_VENDOR_##VENDOR;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

StringRef llvm::dwarf::ChildrenString(unsigned Children) {
  switch (Children) {
  case DW_CHILDREN_no:
    return "DW_CHILDREN_no";
  case DW_CHILDREN_yes:
    return "DW_CHILDREN_yes";
  }
  return StringRef();
}

StringRef llvm::dwarf::AttributeString(unsigned Attribute) {
  switch (Attribute) {
  default:
    return StringRef();
#define HANDLE_DW_AT(ID, NAME, VERSION, VENDOR)                                \
  case DW_AT_##NAME:                                                           \
    return "DW_AT_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::AttributeVersion(dwarf::Attribute Attribute) {
  switch (Attribute) {
  default:
    return 0;
#define HANDLE_DW_AT(ID, NAME, VERSION, VENDOR)                                \
  case DW_AT_##NAME:                                                           \
    return VERSION;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::AttributeVendor(dwarf::Attribute Attribute) {
  switch (Attribute) {
  default:
    return 0;
#define HANDLE_DW_AT(ID, NAME, VERSION, VENDOR)                                \
  case DW_AT_##NAME:                                                           \
    return DWARF_VENDOR_##VENDOR;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

StringRef llvm::dwarf::FormEncodingString(unsigned Encoding) {
  switch (Encoding) {
  default:
    return StringRef();
#define HANDLE_DW_FORM(ID, NAME, VERSION, VENDOR)                              \
  case DW_FORM_##NAME:                                                         \
    return "DW_FORM_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::FormVersion(dwarf::Form Form) {
  switch (Form) {
  default:
    return 0;
#define HANDLE_DW_FORM(ID, NAME, VERSION, VENDOR)                              \
  case DW_FORM_##NAME:                                                         \
    return VERSION;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::FormVendor(dwarf::Form Form) {
  switch (Form) {
  default:
    return 0;
#define HANDLE_DW_FORM(ID, NAME, VERSION, VENDOR)                              \
  case DW_FORM_##NAME:                                                         \
    return DWARF_VENDOR_##VENDOR;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

StringRef llvm::dwarf::OperationEncodingString(unsigned Encoding) {
  switch (Encoding) {
  default:
    return StringRef();
#define HANDLE_DW_OP(ID, NAME, VERSION, VENDOR)                                \
  case DW_OP_##NAME:                                                           \
    return "DW_OP_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  case DW_OP_LLVM_fragment:
    return "DW_OP_LLVM_fragment";
  }
}

unsigned llvm::dwarf::getOperationEncoding(StringRef OperationEncodingString) {
  return StringSwitch<unsigned>(OperationEncodingString)
#define HANDLE_DW_OP(ID, NAME, VERSION, VENDOR)                                \
  .Case("DW_OP_" #NAME, DW_OP_##NAME)
#include "llvm/BinaryFormat/Dwarf.def"
      .Case("DW_OP_LLVM_fragment", DW_OP_LLVM_fragment)
      .Default(0);
}

unsigned llvm::dwarf::OperationVersion(dwarf::LocationAtom Op) {
  switch (Op) {
  default:
    return 0;
#define HANDLE_DW_OP(ID, NAME, VERSION, VENDOR)                                \
  case DW_OP_##NAME:                                                           \
    return VERSION;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::OperationVendor(dwarf::LocationAtom Op) {
  switch (Op) {
  default:
    return 0;
#define HANDLE_DW_OP(ID, NAME, VERSION, VENDOR)                                \
  case DW_OP_##NAME:                                                           \
    return DWARF_VENDOR_##VENDOR;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

StringRef llvm::dwarf::AttributeEncodingString(unsigned Encoding) {
  switch (Encoding) {
  default:
    return StringRef();
#define HANDLE_DW_ATE(ID, NAME, VERSION, VENDOR)                               \
  case DW_ATE_##NAME:                                                          \
    return "DW_ATE_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::getAttributeEncoding(StringRef EncodingString) {
  return StringSwitch<unsigned>(EncodingString)
#define HANDLE_DW_ATE(ID, NAME, VERSION, VENDOR)                               \
  .Case("DW_ATE_" #NAME, DW_ATE_##NAME)
#include "llvm/BinaryFormat/Dwarf.def"
      .Default(0);
}

unsigned llvm::dwarf::AttributeEncodingVersion(dwarf::TypeKind ATE) {
  switch (ATE) {
  default:
    return 0;
#define HANDLE_DW_ATE(ID, NAME, VERSION, VENDOR)                               \
  case DW_ATE_##NAME:                                                          \
    return VERSION;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::AttributeEncodingVendor(dwarf::TypeKind ATE) {
  switch (ATE) {
  default:
    return 0;
#define HANDLE_DW_ATE(ID, NAME, VERSION, VENDOR)                               \
  case DW_ATE_##NAME:                                                          \
    return DWARF_VENDOR_##VENDOR;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

StringRef llvm::dwarf::DecimalSignString(unsigned Sign) {
  switch (Sign) {
  case DW_DS_unsigned:
    return "DW_DS_unsigned";
  case DW_DS_leading_overpunch:
    return "DW_DS_leading_overpunch";
  case DW_DS_trailing_overpunch:
    return "DW_DS_trailing_overpunch";
  case DW_DS_leading_separate:
    return "DW_DS_leading_separate";
  case DW_DS_trailing_separate:
    return "DW_DS_trailing_separate";
  }
  return StringRef();
}

StringRef llvm::dwarf::EndianityString(unsigned Endian) {
  switch (Endian) {
  case DW_END_default:
    return "DW_END_default";
  case DW_END_big:
    return "DW_END_big";
  case DW_END_little:
    return "DW_END_little";
  case DW_END_lo_user:
    return "DW_END_lo_user";
  case DW_END_hi_user:
    return "DW_END_hi_user";
  }
  return StringRef();
}

StringRef llvm::dwarf::AccessibilityString(unsigned Access) {
  switch (Access) {
  // Accessibility codes
  case DW_ACCESS_public:
    return "DW_ACCESS_public";
  case DW_ACCESS_protected:
    return "DW_ACCESS_protected";
  case DW_ACCESS_private:
    return "DW_ACCESS_private";
  }
  return StringRef();
}

StringRef llvm::dwarf::VisibilityString(unsigned Visibility) {
  switch (Visibility) {
  case DW_VIS_local:
    return "DW_VIS_local";
  case DW_VIS_exported:
    return "DW_VIS_exported";
  case DW_VIS_qualified:
    return "DW_VIS_qualified";
  }
  return StringRef();
}

StringRef llvm::dwarf::VirtualityString(unsigned Virtuality) {
  switch (Virtuality) {
  default:
    return StringRef();
#define HANDLE_DW_VIRTUALITY(ID, NAME)                                         \
  case DW_VIRTUALITY_##NAME:                                                   \
    return "DW_VIRTUALITY_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::getVirtuality(StringRef VirtualityString) {
  return StringSwitch<unsigned>(VirtualityString)
#define HANDLE_DW_VIRTUALITY(ID, NAME)                                         \
  .Case("DW_VIRTUALITY_" #NAME, DW_VIRTUALITY_##NAME)
#include "llvm/BinaryFormat/Dwarf.def"
      .Default(DW_VIRTUALITY_invalid);
}

StringRef llvm::dwarf::LanguageString(unsigned Language) {
  switch (Language) {
  default:
    return StringRef();
#define HANDLE_DW_LANG(ID, NAME, VERSION, VENDOR)                              \
  case DW_LANG_##NAME:                                                         \
    return "DW_LANG_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::getLanguage(StringRef LanguageString) {
  return StringSwitch<unsigned>(LanguageString)
#define HANDLE_DW_LANG(ID, NAME, VERSION, VENDOR)                              \
  .Case("DW_LANG_" #NAME, DW_LANG_##NAME)
#include "llvm/BinaryFormat/Dwarf.def"
      .Default(0);
}

unsigned llvm::dwarf::LanguageVersion(dwarf::SourceLanguage Lang) {
  switch (Lang) {
  default:
    return 0;
#define HANDLE_DW_LANG(ID, NAME, VERSION, VENDOR)                              \
  case DW_LANG_##NAME:                                                         \
    return VERSION;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::LanguageVendor(dwarf::SourceLanguage Lang) {
  switch (Lang) {
  default:
    return 0;
#define HANDLE_DW_LANG(ID, NAME, VERSION, VENDOR)                              \
  case DW_LANG_##NAME:                                                         \
    return DWARF_VENDOR_##VENDOR;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

StringRef llvm::dwarf::CaseString(unsigned Case) {
  switch (Case) {
  case DW_ID_case_sensitive:
    return "DW_ID_case_sensitive";
  case DW_ID_up_case:
    return "DW_ID_up_case";
  case DW_ID_down_case:
    return "DW_ID_down_case";
  case DW_ID_case_insensitive:
    return "DW_ID_case_insensitive";
  }
  return StringRef();
}

StringRef llvm::dwarf::ConventionString(unsigned CC) {
  switch (CC) {
  default:
    return StringRef();
#define HANDLE_DW_CC(ID, NAME)                                                 \
  case DW_CC_##NAME:                                                           \
    return "DW_CC_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

unsigned llvm::dwarf::getCallingConvention(StringRef CCString) {
  return StringSwitch<unsigned>(CCString)
#define HANDLE_DW_CC(ID, NAME) .Case("DW_CC_" #NAME, DW_CC_##NAME)
#include "llvm/BinaryFormat/Dwarf.def"
      .Default(0);
}

StringRef llvm::dwarf::InlineCodeString(unsigned Code) {
  switch (Code) {
  case DW_INL_not_inlined:
    return "DW_INL_not_inlined";
  case DW_INL_inlined:
    return "DW_INL_inlined";
  case DW_INL_declared_not_inlined:
    return "DW_INL_declared_not_inlined";
  case DW_INL_declared_inlined:
    return "DW_INL_declared_inlined";
  }
  return StringRef();
}

StringRef llvm::dwarf::ArrayOrderString(unsigned Order) {
  switch (Order) {
  case DW_ORD_row_major:
    return "DW_ORD_row_major";
  case DW_ORD_col_major:
    return "DW_ORD_col_major";
  }
  return StringRef();
}

StringRef llvm::dwarf::DiscriminantString(unsigned Discriminant) {
  switch (Discriminant) {
  case DW_DSC_label:
    return "DW_DSC_label";
  case DW_DSC_range:
    return "DW_DSC_range";
  }
  return StringRef();
}

StringRef llvm::dwarf::LNStandardString(unsigned Standard) {
  switch (Standard) {
  default:
    return StringRef();
#define HANDLE_DW_LNS(ID, NAME)                                                \
  case DW_LNS_##NAME:                                                          \
    return "DW_LNS_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

StringRef llvm::dwarf::LNExtendedString(unsigned Encoding) {
  switch (Encoding) {
  default:
    return StringRef();
#define HANDLE_DW_LNE(ID, NAME)                                                \
  case DW_LNE_##NAME:                                                          \
    return "DW_LNE_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

StringRef llvm::dwarf::MacinfoString(unsigned Encoding) {
  switch (Encoding) {
  // Macinfo Type Encodings
  case DW_MACINFO_define:
    return "DW_MACINFO_define";
  case DW_MACINFO_undef:
    return "DW_MACINFO_undef";
  case DW_MACINFO_start_file:
    return "DW_MACINFO_start_file";
  case DW_MACINFO_end_file:
    return "DW_MACINFO_end_file";
  case DW_MACINFO_vendor_ext:
    return "DW_MACINFO_vendor_ext";
  case DW_MACINFO_invalid:
    return "DW_MACINFO_invalid";
  }
  return StringRef();
}

unsigned llvm::dwarf::getMacinfo(StringRef MacinfoString) {
  return StringSwitch<unsigned>(MacinfoString)
      .Case("DW_MACINFO_define", DW_MACINFO_define)
      .Case("DW_MACINFO_undef", DW_MACINFO_undef)
      .Case("DW_MACINFO_start_file", DW_MACINFO_start_file)
      .Case("DW_MACINFO_end_file", DW_MACINFO_end_file)
      .Case("DW_MACINFO_vendor_ext", DW_MACINFO_vendor_ext)
      .Default(DW_MACINFO_invalid);
}

StringRef llvm::dwarf::CallFrameString(unsigned Encoding) {
  switch (Encoding) {
  default:
    return StringRef();
#define HANDLE_DW_CFA(ID, NAME)                                                \
  case DW_CFA_##NAME:                                                          \
    return "DW_CFA_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

StringRef llvm::dwarf::ApplePropertyString(unsigned Prop) {
  switch (Prop) {
  default:
    return StringRef();
#define HANDLE_DW_APPLE_PROPERTY(ID, NAME)                                     \
  case DW_APPLE_PROPERTY_##NAME:                                               \
    return "DW_APPLE_PROPERTY_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

StringRef llvm::dwarf::UnitTypeString(unsigned UT) {
  switch (UT) {
  default:
    return StringRef();
#define HANDLE_DW_UT(ID, NAME)                                                 \
  case DW_UT_##NAME:                                                           \
    return "DW_UT_" #NAME;
#include "llvm/BinaryFormat/Dwarf.def"
  }
}

StringRef llvm::dwarf::AtomTypeString(unsigned AT) {
  switch (AT) {
  case dwarf::DW_ATOM_null:
    return "DW_ATOM_null";
  case dwarf::DW_ATOM_die_offset:
    return "DW_ATOM_die_offset";
  case DW_ATOM_cu_offset:
    return "DW_ATOM_cu_offset";
  case DW_ATOM_die_tag:
    return "DW_ATOM_die_tag";
  case DW_ATOM_type_flags:
    return "DW_ATOM_type_flags";
  }
  return StringRef();
}

StringRef llvm::dwarf::GDBIndexEntryKindString(GDBIndexEntryKind Kind) {
  switch (Kind) {
  case GIEK_NONE:
    return "NONE";
  case GIEK_TYPE:
    return "TYPE";
  case GIEK_VARIABLE:
    return "VARIABLE";
  case GIEK_FUNCTION:
    return "FUNCTION";
  case GIEK_OTHER:
    return "OTHER";
  case GIEK_UNUSED5:
    return "UNUSED5";
  case GIEK_UNUSED6:
    return "UNUSED6";
  case GIEK_UNUSED7:
    return "UNUSED7";
  }
  llvm_unreachable("Unknown GDBIndexEntryKind value");
}

StringRef
llvm::dwarf::GDBIndexEntryLinkageString(GDBIndexEntryLinkage Linkage) {
  switch (Linkage) {
  case GIEL_EXTERNAL:
    return "EXTERNAL";
  case GIEL_STATIC:
    return "STATIC";
  }
  llvm_unreachable("Unknown GDBIndexEntryLinkage value");
}

StringRef llvm::dwarf::AttributeValueString(uint16_t Attr, unsigned Val) {
  switch (Attr) {
  case DW_AT_accessibility:
    return AccessibilityString(Val);
  case DW_AT_virtuality:
    return VirtualityString(Val);
  case DW_AT_language:
    return LanguageString(Val);
  case DW_AT_encoding:
    return AttributeEncodingString(Val);
  case DW_AT_decimal_sign:
    return DecimalSignString(Val);
  case DW_AT_endianity:
    return EndianityString(Val);
  case DW_AT_visibility:
    return VisibilityString(Val);
  case DW_AT_identifier_case:
    return CaseString(Val);
  case DW_AT_calling_convention:
    return ConventionString(Val);
  case DW_AT_inline:
    return InlineCodeString(Val);
  case DW_AT_ordering:
    return ArrayOrderString(Val);
  case DW_AT_discr_value:
    return DiscriminantString(Val);
  }

  return StringRef();
}

bool llvm::dwarf::isValidFormForVersion(Form F, unsigned Version,
                                        bool ExtensionsOk) {
  if (FormVendor(F) == DWARF_VENDOR_DWARF) {
    unsigned FV = FormVersion(F);
    return FV > 0 && FV <= Version;
  }
  return ExtensionsOk;
}
