//===- unittest/BinaryFormat/DwarfTest.cpp - Dwarf support tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
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

TEST(DwarfTest, FixedFormSizes) {
  std::optional<uint8_t> RefSize;
  std::optional<uint8_t> AddrSize;

  // Test 32 bit DWARF version 2 with 4 byte addresses.
  FormParams Params_2_4_32 = {2, 4, DWARF32};
  RefSize = getFixedFormByteSize(DW_FORM_ref_addr, Params_2_4_32);
  AddrSize = getFixedFormByteSize(DW_FORM_ref_addr, Params_2_4_32);
  EXPECT_TRUE(RefSize.has_value());
  EXPECT_TRUE(AddrSize.has_value());
  EXPECT_EQ(*RefSize, *AddrSize);

  // Test 32 bit DWARF version 2 with 8 byte addresses.
  FormParams Params_2_8_32 = {2, 8, DWARF32};
  RefSize = getFixedFormByteSize(DW_FORM_ref_addr, Params_2_8_32);
  AddrSize = getFixedFormByteSize(DW_FORM_ref_addr, Params_2_8_32);
  EXPECT_TRUE(RefSize.has_value());
  EXPECT_TRUE(AddrSize.has_value());
  EXPECT_EQ(*RefSize, *AddrSize);

  // DW_FORM_ref_addr is 4 bytes in DWARF 32 in DWARF version 3 and beyond.
  FormParams Params_3_4_32 = {3, 4, DWARF32};
  RefSize = getFixedFormByteSize(DW_FORM_ref_addr, Params_3_4_32);
  EXPECT_TRUE(RefSize.has_value());
  EXPECT_EQ(*RefSize, 4);

  FormParams Params_4_4_32 = {4, 4, DWARF32};
  RefSize = getFixedFormByteSize(DW_FORM_ref_addr, Params_4_4_32);
  EXPECT_TRUE(RefSize.has_value());
  EXPECT_EQ(*RefSize, 4);

  FormParams Params_5_4_32 = {5, 4, DWARF32};
  RefSize = getFixedFormByteSize(DW_FORM_ref_addr, Params_5_4_32);
  EXPECT_TRUE(RefSize.has_value());
  EXPECT_EQ(*RefSize, 4);

  // DW_FORM_ref_addr is 8 bytes in DWARF 64 in DWARF version 3 and beyond.
  FormParams Params_3_8_64 = {3, 8, DWARF64};
  RefSize = getFixedFormByteSize(DW_FORM_ref_addr, Params_3_8_64);
  EXPECT_TRUE(RefSize.has_value());
  EXPECT_EQ(*RefSize, 8);

  FormParams Params_4_8_64 = {4, 8, DWARF64};
  RefSize = getFixedFormByteSize(DW_FORM_ref_addr, Params_4_8_64);
  EXPECT_TRUE(RefSize.has_value());
  EXPECT_EQ(*RefSize, 8);

  FormParams Params_5_8_64 = {5, 8, DWARF64};
  RefSize = getFixedFormByteSize(DW_FORM_ref_addr, Params_5_8_64);
  EXPECT_TRUE(RefSize.has_value());
  EXPECT_EQ(*RefSize, 8);
}

TEST(DwarfTest, format_provider) {
  EXPECT_EQ("DW_AT_name", formatv("{0}", DW_AT_name).str());
  EXPECT_EQ("DW_AT_unknown_3fff", formatv("{0}", DW_AT_hi_user).str());
  EXPECT_EQ("DW_FORM_addr", formatv("{0}", DW_FORM_addr).str());
  EXPECT_EQ("DW_FORM_unknown_1f00", formatv("{0}", DW_FORM_lo_user).str());
  EXPECT_EQ("DW_IDX_compile_unit", formatv("{0}", DW_IDX_compile_unit).str());
  EXPECT_EQ("DW_IDX_unknown_3fff", formatv("{0}", DW_IDX_hi_user).str());
  EXPECT_EQ("DW_TAG_compile_unit", formatv("{0}", DW_TAG_compile_unit).str());
  EXPECT_EQ("DW_TAG_unknown_ffff", formatv("{0}", DW_TAG_hi_user).str());
  EXPECT_EQ("DW_OP_lit0", formatv("{0}", DW_OP_lit0).str());
  EXPECT_EQ("DW_OP_unknown_ff", formatv("{0}", DW_OP_hi_user).str());
}

TEST(DwarfTest, lname) {
  auto roundtrip = [](llvm::dwarf::SourceLanguage sl) {
    auto name_version = toDW_LNAME(sl);
    // Ignore ones without a defined mapping.
    if (sl == DW_LANG_Mips_Assembler || sl == DW_LANG_GOOGLE_RenderScript ||
        !name_version.has_value())
      return sl;
    return dwarf::toDW_LANG(name_version->first, name_version->second)
        .value_or(sl);
  };
#define HANDLE_DW_LANG(ID, NAME, LOWER_BOUND, VERSION, VENDOR)                 \
  EXPECT_EQ(roundtrip(DW_LANG_##NAME), DW_LANG_##NAME);
#include "llvm/BinaryFormat/Dwarf.def"
}

TEST(DwarfTest, lname_getSourceLanguageName) {
  // Some basics.
  EXPECT_EQ(getSourceLanguageName("DW_LNAME_Ada"), DW_LNAME_Ada);
  EXPECT_EQ(getSourceLanguageName("DW_LNAME_Metal"), DW_LNAME_Metal);

  // Test invalid input.
  EXPECT_EQ(getSourceLanguageName(""), 0U);
  EXPECT_EQ(getSourceLanguageName("blah"), 0U);
  EXPECT_EQ(getSourceLanguageName("DW_LNAME__something_unlikely"), 0U);
  EXPECT_EQ(getSourceLanguageName("DW_LANG_C"), 0U);

  // Test that we cover all DW_LNAME_ names.
#define xstr(X) #X
#define HANDLE_DW_LNAME(ID, NAME, DESC, LOWER_BOUND)                           \
  EXPECT_EQ(getSourceLanguageName(xstr(DW_LNAME_##NAME)), DW_LNAME_##NAME);
#include "llvm/BinaryFormat/Dwarf.def"
}

TEST(DwarfTest, lname_SourceLanguageNameString) {
  // Some basics.
  EXPECT_EQ(SourceLanguageNameString(DW_LNAME_C_plus_plus),
            "DW_LNAME_C_plus_plus");
  EXPECT_EQ(SourceLanguageNameString(DW_LNAME_CPP_for_OpenCL),
            "DW_LNAME_CPP_for_OpenCL");

  // Test invalid input.
  EXPECT_EQ(SourceLanguageNameString(static_cast<SourceLanguageName>(0)), "");

  // Test that we cover all DW_LNAME_ names.
#define xstr(X) #X
#define HANDLE_DW_LNAME(ID, NAME, DESC, LOWER_BOUND)                           \
  EXPECT_EQ(SourceLanguageNameString(DW_LNAME_##NAME), xstr(DW_LNAME_##NAME));
#include "llvm/BinaryFormat/Dwarf.def"
}

struct LanguageDescriptionTestCase {
  llvm::dwarf::SourceLanguageName LName;
  uint32_t LVersion;
  llvm::StringRef ExpectedDescription;
};

LanguageDescriptionTestCase LanguageDescriptionTestCases[] = {
    {static_cast<SourceLanguageName>(0), 0, "Unknown"},
    {static_cast<SourceLanguageName>(0), 1, "Unknown"},
    {DW_LNAME_Ada, 0, "Ada 83"},
    {DW_LNAME_Ada, 1982, "Ada 83"},
    {DW_LNAME_Ada, 1983, "Ada 83"},
    {DW_LNAME_Ada, 1994, "Ada 95"},
    {DW_LNAME_Ada, 1995, "Ada 95"},
    {DW_LNAME_Ada, 2004, "Ada 2005"},
    {DW_LNAME_Ada, 2005, "Ada 2005"},
    {DW_LNAME_Ada, 2011, "Ada 2012"},
    {DW_LNAME_Ada, 2012, "Ada 2012"},
    {DW_LNAME_Ada, 2013, "ISO Ada"},
    {DW_LNAME_Cobol, 0, "COBOL-74"},
    {DW_LNAME_Cobol, 1973, "COBOL-74"},
    {DW_LNAME_Cobol, 1974, "COBOL-74"},
    {DW_LNAME_Cobol, 1984, "COBOL-85"},
    {DW_LNAME_Cobol, 1985, "COBOL-85"},
    {DW_LNAME_Cobol, 1986, "ISO Cobol"},
    {DW_LNAME_Fortran, 0, "FORTRAN 77"},
    {DW_LNAME_Fortran, 1976, "FORTRAN 77"},
    {DW_LNAME_Fortran, 1977, "FORTRAN 77"},
    {DW_LNAME_Fortran, 1989, "FORTRAN 90"},
    {DW_LNAME_Fortran, 1990, "FORTRAN 90"},
    {DW_LNAME_Fortran, 1994, "Fortran 95"},
    {DW_LNAME_Fortran, 1995, "Fortran 95"},
    {DW_LNAME_Fortran, 2002, "Fortran 2003"},
    {DW_LNAME_Fortran, 2003, "Fortran 2003"},
    {DW_LNAME_Fortran, 2007, "Fortran 2008"},
    {DW_LNAME_Fortran, 2008, "Fortran 2008"},
    {DW_LNAME_Fortran, 2017, "Fortran 2018"},
    {DW_LNAME_Fortran, 2018, "Fortran 2018"},
    {DW_LNAME_Fortran, 2019, "ISO Fortran"},
    {DW_LNAME_C, 0, "C (K&R and ISO)"},
    {DW_LNAME_C, 198911, "C89"},
    {DW_LNAME_C, 198912, "C89"},
    {DW_LNAME_C, 199901, "C99"},
    {DW_LNAME_C, 199902, "C11"},
    {DW_LNAME_C, 201111, "C11"},
    {DW_LNAME_C, 201112, "C11"},
    {DW_LNAME_C, 201201, "C17"},
    {DW_LNAME_C, 201709, "C17"},
    {DW_LNAME_C, 201710, "C17"},
    {DW_LNAME_C, 201711, "C (K&R and ISO)"},
    {DW_LNAME_C_plus_plus, 0, "ISO C++"},
    {DW_LNAME_C_plus_plus, 199710, "C++98"},
    {DW_LNAME_C_plus_plus, 199711, "C++98"},
    {DW_LNAME_C_plus_plus, 199712, "C++03"},
    {DW_LNAME_C_plus_plus, 200310, "C++03"},
    {DW_LNAME_C_plus_plus, 200311, "C++11"},
    {DW_LNAME_C_plus_plus, 201102, "C++11"},
    {DW_LNAME_C_plus_plus, 201103, "C++11"},
    {DW_LNAME_C_plus_plus, 201104, "C++14"},
    {DW_LNAME_C_plus_plus, 201401, "C++14"},
    {DW_LNAME_C_plus_plus, 201402, "C++14"},
    {DW_LNAME_C_plus_plus, 201403, "C++17"},
    {DW_LNAME_C_plus_plus, 201702, "C++17"},
    {DW_LNAME_C_plus_plus, 201703, "C++17"},
    {DW_LNAME_C_plus_plus, 201704, "C++20"},
    {DW_LNAME_C_plus_plus, 202001, "C++20"},
    {DW_LNAME_C_plus_plus, 202002, "C++20"},
    {DW_LNAME_C_plus_plus, 202003, "ISO C++"},
    {DW_LNAME_ObjC_plus_plus, 0, LanguageDescription(DW_LNAME_ObjC_plus_plus)},
    {DW_LNAME_ObjC_plus_plus, 1, LanguageDescription(DW_LNAME_ObjC_plus_plus)},
    {DW_LNAME_ObjC, 0, LanguageDescription(DW_LNAME_ObjC)},
    {DW_LNAME_ObjC, 1, LanguageDescription(DW_LNAME_ObjC)},
    {DW_LNAME_Move, 0, LanguageDescription(DW_LNAME_Move)},
    {DW_LNAME_Move, 1, LanguageDescription(DW_LNAME_Move)},
    {DW_LNAME_SYCL, 0, LanguageDescription(DW_LNAME_SYCL)},
    {DW_LNAME_SYCL, 1, LanguageDescription(DW_LNAME_SYCL)},
    {DW_LNAME_BLISS, 0, LanguageDescription(DW_LNAME_BLISS)},
    {DW_LNAME_BLISS, 1, LanguageDescription(DW_LNAME_BLISS)},
    {DW_LNAME_Crystal, 0, LanguageDescription(DW_LNAME_Crystal)},
    {DW_LNAME_Crystal, 1, LanguageDescription(DW_LNAME_Crystal)},
    {DW_LNAME_D, 0, LanguageDescription(DW_LNAME_D)},
    {DW_LNAME_D, 1, LanguageDescription(DW_LNAME_D)},
    {DW_LNAME_Dylan, 0, LanguageDescription(DW_LNAME_Dylan)},
    {DW_LNAME_Dylan, 1, LanguageDescription(DW_LNAME_Dylan)},
    {DW_LNAME_Go, 0, LanguageDescription(DW_LNAME_Go)},
    {DW_LNAME_Go, 1, LanguageDescription(DW_LNAME_Go)},
    {DW_LNAME_Haskell, 0, LanguageDescription(DW_LNAME_Haskell)},
    {DW_LNAME_Haskell, 1, LanguageDescription(DW_LNAME_Haskell)},
    {DW_LNAME_HLSL, 0, LanguageDescription(DW_LNAME_HLSL)},
    {DW_LNAME_HLSL, 1, LanguageDescription(DW_LNAME_HLSL)},
    {DW_LNAME_Java, 0, LanguageDescription(DW_LNAME_Java)},
    {DW_LNAME_Java, 1, LanguageDescription(DW_LNAME_Java)},
    {DW_LNAME_Julia, 0, LanguageDescription(DW_LNAME_Julia)},
    {DW_LNAME_Julia, 1, LanguageDescription(DW_LNAME_Julia)},
    {DW_LNAME_Kotlin, 0, LanguageDescription(DW_LNAME_Kotlin)},
    {DW_LNAME_Kotlin, 1, LanguageDescription(DW_LNAME_Kotlin)},
    {DW_LNAME_Modula2, 0, LanguageDescription(DW_LNAME_Modula2)},
    {DW_LNAME_Modula2, 1, LanguageDescription(DW_LNAME_Modula2)},
    {DW_LNAME_Modula3, 0, LanguageDescription(DW_LNAME_Modula3)},
    {DW_LNAME_Modula3, 1, LanguageDescription(DW_LNAME_Modula3)},
    {DW_LNAME_OCaml, 0, LanguageDescription(DW_LNAME_OCaml)},
    {DW_LNAME_OCaml, 1, LanguageDescription(DW_LNAME_OCaml)},
    {DW_LNAME_OpenCL_C, 0, LanguageDescription(DW_LNAME_OpenCL_C)},
    {DW_LNAME_OpenCL_C, 1, LanguageDescription(DW_LNAME_OpenCL_C)},
    {DW_LNAME_Pascal, 0, LanguageDescription(DW_LNAME_Pascal)},
    {DW_LNAME_Pascal, 1, LanguageDescription(DW_LNAME_Pascal)},
    {DW_LNAME_PLI, 0, LanguageDescription(DW_LNAME_PLI)},
    {DW_LNAME_PLI, 1, LanguageDescription(DW_LNAME_PLI)},
    {DW_LNAME_Python, 0, LanguageDescription(DW_LNAME_Python)},
    {DW_LNAME_Python, 1, LanguageDescription(DW_LNAME_Python)},
    {DW_LNAME_RenderScript, 0, LanguageDescription(DW_LNAME_RenderScript)},
    {DW_LNAME_RenderScript, 1, LanguageDescription(DW_LNAME_RenderScript)},
    {DW_LNAME_Rust, 0, LanguageDescription(DW_LNAME_Rust)},
    {DW_LNAME_Rust, 1, LanguageDescription(DW_LNAME_Rust)},
    {DW_LNAME_Swift, 0, LanguageDescription(DW_LNAME_Swift)},
    {DW_LNAME_Swift, 1, LanguageDescription(DW_LNAME_Swift)},
    {DW_LNAME_UPC, 0, LanguageDescription(DW_LNAME_UPC)},
    {DW_LNAME_UPC, 1, LanguageDescription(DW_LNAME_UPC)},
    {DW_LNAME_Zig, 0, LanguageDescription(DW_LNAME_Zig)},
    {DW_LNAME_Zig, 1, LanguageDescription(DW_LNAME_Zig)},
    {DW_LNAME_Assembly, 0, LanguageDescription(DW_LNAME_Assembly)},
    {DW_LNAME_Assembly, 1, LanguageDescription(DW_LNAME_Assembly)},
    {DW_LNAME_C_sharp, 0, LanguageDescription(DW_LNAME_C_sharp)},
    {DW_LNAME_C_sharp, 1, LanguageDescription(DW_LNAME_C_sharp)},
    {DW_LNAME_Mojo, 0, LanguageDescription(DW_LNAME_Mojo)},
    {DW_LNAME_Mojo, 1, LanguageDescription(DW_LNAME_Mojo)},
    {DW_LNAME_GLSL, 0, LanguageDescription(DW_LNAME_GLSL)},
    {DW_LNAME_GLSL, 1, LanguageDescription(DW_LNAME_GLSL)},
    {DW_LNAME_GLSL_ES, 0, LanguageDescription(DW_LNAME_GLSL_ES)},
    {DW_LNAME_GLSL_ES, 1, LanguageDescription(DW_LNAME_GLSL_ES)},
    {DW_LNAME_OpenCL_CPP, 0, LanguageDescription(DW_LNAME_OpenCL_CPP)},
    {DW_LNAME_OpenCL_CPP, 1, LanguageDescription(DW_LNAME_OpenCL_CPP)},
    {DW_LNAME_CPP_for_OpenCL, 0, LanguageDescription(DW_LNAME_CPP_for_OpenCL)},
    {DW_LNAME_CPP_for_OpenCL, 1, LanguageDescription(DW_LNAME_CPP_for_OpenCL)},
    {DW_LNAME_Ruby, 0, LanguageDescription(DW_LNAME_Ruby)},
    {DW_LNAME_Ruby, 1, LanguageDescription(DW_LNAME_Ruby)},
    {DW_LNAME_Hylo, 0, LanguageDescription(DW_LNAME_Hylo)},
    {DW_LNAME_Hylo, 1, LanguageDescription(DW_LNAME_Hylo)},
    {DW_LNAME_Metal, 0, LanguageDescription(DW_LNAME_Metal)},
    {DW_LNAME_Metal, 1, LanguageDescription(DW_LNAME_Metal)}};

struct LanguageDescriptionTestFixture
    : public testing::Test,
      public testing::WithParamInterface<LanguageDescriptionTestCase> {};

TEST_P(LanguageDescriptionTestFixture, TestLanguageDescription) {
  auto [LName, LVersion, ExpectedDescription] = GetParam();

  // Basic test.
  EXPECT_EQ(llvm::dwarf::LanguageDescription(LName, LVersion),
            ExpectedDescription);

  // Now do the same test but roundtrip through the DW_LANG_ <-> DW_LNAME_
  // conversion APIs first.

  auto DWLang = llvm::dwarf::toDW_LANG(LName, LVersion);
  // Some languages are not 1-to-1 mapped. In which case there's nothing else
  // to test.
  if (!DWLang)
    return;

  std::optional<std::pair<SourceLanguageName, uint32_t>> DWLName =
      llvm::dwarf::toDW_LNAME(*DWLang);

  // We are roundtripping, so there definitely should be a mapping back to
  // DW_LNAME_.
  ASSERT_TRUE(DWLName);

  // There is no official DW_LANG_ code for C++98. So the roundtripping turns it
  // into a plain DW_LANG_C_plus_plus.
  if (DWLang == DW_LANG_C_plus_plus && LVersion <= 199711)
    EXPECT_EQ(llvm::dwarf::LanguageDescription(DWLName->first, DWLName->second),
              "ISO C++");
  else
    EXPECT_EQ(llvm::dwarf::LanguageDescription(DWLName->first, DWLName->second),
              ExpectedDescription);
}

INSTANTIATE_TEST_SUITE_P(LanguageDescriptionTests,
                         LanguageDescriptionTestFixture,
                         ::testing::ValuesIn(LanguageDescriptionTestCases));

} // end namespace
