//===- unittest/BinaryFormat/DwarfTest.cpp - Dwarf support tests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::dwarf;

namespace {

TEST(DwarfTest, TagStringOnInvalid) {
  // This is invalid, so it shouldn't be stringified.
  EXPECT_EQ(StringRef(), TagString(DW_TAG_invalid));

  // These aren't really tags: they describe ranges within tags.  They
  // shouldn't be stringified either.
  EXPECT_EQ(StringRef(), TagString(DW_TAG_lo_user));
  EXPECT_EQ(StringRef(), TagString(DW_TAG_hi_user));
  EXPECT_EQ(StringRef(), TagString(DW_TAG_user_base));
}

TEST(DwarfTest, getTag) {
  // A couple of valid tags.
  EXPECT_EQ(DW_TAG_array_type, getTag("DW_TAG_array_type"));
  EXPECT_EQ(DW_TAG_module, getTag("DW_TAG_module"));

  // Invalid tags.
  EXPECT_EQ(DW_TAG_invalid, getTag("DW_TAG_invalid"));
  EXPECT_EQ(DW_TAG_invalid, getTag("DW_TAG_madeuptag"));
  EXPECT_EQ(DW_TAG_invalid, getTag("something else"));

  // Tag range markers should not be recognized.
  EXPECT_EQ(DW_TAG_invalid, getTag("DW_TAG_lo_user"));
  EXPECT_EQ(DW_TAG_invalid, getTag("DW_TAG_hi_user"));
  EXPECT_EQ(DW_TAG_invalid, getTag("DW_TAG_user_base"));
}

TEST(DwarfTest, getOperationEncoding) {
  // Some valid ops.
  EXPECT_EQ(DW_OP_deref, getOperationEncoding("DW_OP_deref"));
  EXPECT_EQ(DW_OP_bit_piece, getOperationEncoding("DW_OP_bit_piece"));

  // Invalid ops.
  EXPECT_EQ(0u, getOperationEncoding("DW_OP_otherthings"));
  EXPECT_EQ(0u, getOperationEncoding("other"));

  // Markers shouldn't be recognized.
  EXPECT_EQ(0u, getOperationEncoding("DW_OP_lo_user"));
  EXPECT_EQ(0u, getOperationEncoding("DW_OP_hi_user"));
}

TEST(DwarfTest, LanguageStringOnInvalid) {
  // This is invalid, so it shouldn't be stringified.
  EXPECT_EQ(StringRef(), LanguageString(0));

  // These aren't really tags: they describe ranges within tags.  They
  // shouldn't be stringified either.
  EXPECT_EQ(StringRef(), LanguageString(DW_LANG_lo_user));
  EXPECT_EQ(StringRef(), LanguageString(DW_LANG_hi_user));
}

TEST(DwarfTest, getLanguage) {
  // A couple of valid languages.
  EXPECT_EQ(DW_LANG_C89, getLanguage("DW_LANG_C89"));
  EXPECT_EQ(DW_LANG_C_plus_plus_11, getLanguage("DW_LANG_C_plus_plus_11"));
  EXPECT_EQ(DW_LANG_OCaml, getLanguage("DW_LANG_OCaml"));
  EXPECT_EQ(DW_LANG_Mips_Assembler, getLanguage("DW_LANG_Mips_Assembler"));

  // Invalid languages.
  EXPECT_EQ(0u, getLanguage("DW_LANG_invalid"));
  EXPECT_EQ(0u, getLanguage("DW_TAG_array_type"));
  EXPECT_EQ(0u, getLanguage("something else"));

  // Language range markers should not be recognized.
  EXPECT_EQ(0u, getLanguage("DW_LANG_lo_user"));
  EXPECT_EQ(0u, getLanguage("DW_LANG_hi_user"));
}

TEST(DwarfTest, AttributeEncodingStringOnInvalid) {
  // This is invalid, so it shouldn't be stringified.
  EXPECT_EQ(StringRef(), AttributeEncodingString(0));

  // These aren't really tags: they describe ranges within tags.  They
  // shouldn't be stringified either.
  EXPECT_EQ(StringRef(), AttributeEncodingString(DW_ATE_lo_user));
  EXPECT_EQ(StringRef(), AttributeEncodingString(DW_ATE_hi_user));
}

TEST(DwarfTest, getAttributeEncoding) {
  // A couple of valid languages.
  EXPECT_EQ(DW_ATE_boolean, getAttributeEncoding("DW_ATE_boolean"));
  EXPECT_EQ(DW_ATE_imaginary_float,
            getAttributeEncoding("DW_ATE_imaginary_float"));

  // Invalid languages.
  EXPECT_EQ(0u, getAttributeEncoding("DW_ATE_invalid"));
  EXPECT_EQ(0u, getAttributeEncoding("DW_TAG_array_type"));
  EXPECT_EQ(0u, getAttributeEncoding("something else"));

  // AttributeEncoding range markers should not be recognized.
  EXPECT_EQ(0u, getAttributeEncoding("DW_ATE_lo_user"));
  EXPECT_EQ(0u, getAttributeEncoding("DW_ATE_hi_user"));
}

TEST(DwarfTest, VirtualityString) {
  EXPECT_EQ(StringRef("DW_VIRTUALITY_none"),
            VirtualityString(DW_VIRTUALITY_none));
  EXPECT_EQ(StringRef("DW_VIRTUALITY_virtual"),
            VirtualityString(DW_VIRTUALITY_virtual));
  EXPECT_EQ(StringRef("DW_VIRTUALITY_pure_virtual"),
            VirtualityString(DW_VIRTUALITY_pure_virtual));

  // DW_VIRTUALITY_max should be pure virtual.
  EXPECT_EQ(StringRef("DW_VIRTUALITY_pure_virtual"),
            VirtualityString(DW_VIRTUALITY_max));

  // Invalid numbers shouldn't be stringified.
  EXPECT_EQ(StringRef(), VirtualityString(DW_VIRTUALITY_max + 1));
  EXPECT_EQ(StringRef(), VirtualityString(DW_VIRTUALITY_max + 77));
}

TEST(DwarfTest, getVirtuality) {
  EXPECT_EQ(DW_VIRTUALITY_none, getVirtuality("DW_VIRTUALITY_none"));
  EXPECT_EQ(DW_VIRTUALITY_virtual, getVirtuality("DW_VIRTUALITY_virtual"));
  EXPECT_EQ(DW_VIRTUALITY_pure_virtual,
            getVirtuality("DW_VIRTUALITY_pure_virtual"));

  // Invalid strings.
  EXPECT_EQ(DW_VIRTUALITY_invalid, getVirtuality("DW_VIRTUALITY_invalid"));
  EXPECT_EQ(DW_VIRTUALITY_invalid, getVirtuality("DW_VIRTUALITY_max"));
  EXPECT_EQ(DW_VIRTUALITY_invalid, getVirtuality("something else"));
}

} // end namespace
