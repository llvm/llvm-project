//===- llvm/unittest/DebugInfo/DWARFFormValueTest.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LEB128.h"
#include "llvm/TargetParser/Host.h"
#include "gtest/gtest.h"
#include <climits>
using namespace llvm;
using namespace dwarf;

namespace {

bool isFormClass(dwarf::Form Form, DWARFFormValue::FormClass FC) {
  return DWARFFormValue(Form).isFormClass(FC);
}

TEST(DWARFFormValue, FormClass) {
  EXPECT_TRUE(isFormClass(DW_FORM_addr, DWARFFormValue::FC_Address));
  EXPECT_FALSE(isFormClass(DW_FORM_data8, DWARFFormValue::FC_Address));
  EXPECT_TRUE(isFormClass(DW_FORM_data8, DWARFFormValue::FC_Constant));
  EXPECT_TRUE(isFormClass(DW_FORM_data8, DWARFFormValue::FC_SectionOffset));
  EXPECT_TRUE(
      isFormClass(DW_FORM_sec_offset, DWARFFormValue::FC_SectionOffset));
  EXPECT_TRUE(isFormClass(DW_FORM_GNU_str_index, DWARFFormValue::FC_String));
  EXPECT_TRUE(isFormClass(DW_FORM_GNU_addr_index, DWARFFormValue::FC_Address));
  EXPECT_FALSE(isFormClass(DW_FORM_ref_addr, DWARFFormValue::FC_Address));
  EXPECT_TRUE(isFormClass(DW_FORM_ref_addr, DWARFFormValue::FC_Reference));
  EXPECT_TRUE(isFormClass(DW_FORM_ref_sig8, DWARFFormValue::FC_Reference));
}

template<typename RawTypeT>
DWARFFormValue createDataXFormValue(dwarf::Form Form, RawTypeT Value) {
  char Raw[sizeof(RawTypeT)];
  memcpy(Raw, &Value, sizeof(RawTypeT));
  uint64_t Offset = 0;
  DWARFFormValue Result(Form);
  DWARFDataExtractor Data(StringRef(Raw, sizeof(RawTypeT)),
                          sys::IsLittleEndianHost, sizeof(void *));
  Result.extractValue(Data, &Offset, {0, 0, dwarf::DwarfFormat::DWARF32});
  return Result;
}

DWARFFormValue createULEBFormValue(uint64_t Value) {
  SmallString<10> RawData;
  raw_svector_ostream OS(RawData);
  encodeULEB128(Value, OS);
  uint64_t Offset = 0;
  DWARFFormValue Result(DW_FORM_udata);
  DWARFDataExtractor Data(OS.str(), sys::IsLittleEndianHost, sizeof(void *));
  Result.extractValue(Data, &Offset, {0, 0, dwarf::DwarfFormat::DWARF32});
  return Result;
}

DWARFFormValue createSLEBFormValue(int64_t Value) {
  SmallString<10> RawData;
  raw_svector_ostream OS(RawData);
  encodeSLEB128(Value, OS);
  uint64_t Offset = 0;
  DWARFFormValue Result(DW_FORM_sdata);
  DWARFDataExtractor Data(OS.str(), sys::IsLittleEndianHost, sizeof(void *));
  Result.extractValue(Data, &Offset, {0, 0, dwarf::DwarfFormat::DWARF32});
  return Result;
}

TEST(DWARFFormValue, SignedConstantForms) {
  // Check that we correctly sign extend fixed size forms.
  auto Sign1 = createDataXFormValue<uint8_t>(DW_FORM_data1, -123);
  auto Sign2 = createDataXFormValue<uint16_t>(DW_FORM_data2, -12345);
  auto Sign4 = createDataXFormValue<uint32_t>(DW_FORM_data4, -123456789);
  auto Sign8 = createDataXFormValue<uint64_t>(DW_FORM_data8, -1);
  EXPECT_EQ(*Sign1.getAsSignedConstant(), -123);
  EXPECT_EQ(*Sign2.getAsSignedConstant(), -12345);
  EXPECT_EQ(*Sign4.getAsSignedConstant(), -123456789);
  EXPECT_EQ(*Sign8.getAsSignedConstant(), -1);

  // Check that we can handle big positive values, but that we return
  // an error just over the limit.
  auto UMax = createULEBFormValue(LLONG_MAX);
  auto TooBig = createULEBFormValue(uint64_t(LLONG_MAX) + 1);
  EXPECT_EQ(*UMax.getAsSignedConstant(), LLONG_MAX);
  EXPECT_EQ(TooBig.getAsSignedConstant().has_value(), false);

  // Sanity check some other forms.
  auto Data1 = createDataXFormValue<uint8_t>(DW_FORM_data1, 120);
  auto Data2 = createDataXFormValue<uint16_t>(DW_FORM_data2, 32000);
  auto Data4 = createDataXFormValue<uint32_t>(DW_FORM_data4, 2000000000);
  auto Data8 = createDataXFormValue<uint64_t>(DW_FORM_data8, 0x1234567812345678LL);
  auto LEBMin = createSLEBFormValue(LLONG_MIN);
  auto LEBMax = createSLEBFormValue(LLONG_MAX);
  auto LEB1 = createSLEBFormValue(-42);
  auto LEB2 = createSLEBFormValue(42);
  EXPECT_EQ(*Data1.getAsSignedConstant(), 120);
  EXPECT_EQ(*Data2.getAsSignedConstant(), 32000);
  EXPECT_EQ(*Data4.getAsSignedConstant(), 2000000000);
  EXPECT_EQ(*Data8.getAsSignedConstant(), 0x1234567812345678LL);
  EXPECT_EQ(*LEBMin.getAsSignedConstant(), LLONG_MIN);
  EXPECT_EQ(*LEBMax.getAsSignedConstant(), LLONG_MAX);
  EXPECT_EQ(*LEB1.getAsSignedConstant(), -42);
  EXPECT_EQ(*LEB2.getAsSignedConstant(), 42);

  // Data16 is a little tricky.
  char Cksum[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  DWARFFormValue Data16(DW_FORM_data16);
  DWARFDataExtractor DE16(StringRef(Cksum, 16), sys::IsLittleEndianHost,
                          sizeof(void *));
  uint64_t Offset = 0;
  Data16.extractValue(DE16, &Offset, {0, 0, dwarf::DwarfFormat::DWARF32});
  SmallString<32> Str;
  raw_svector_ostream Res(Str);
  Data16.dump(Res, DIDumpOptions());
  EXPECT_EQ(memcmp(Str.data(), "000102030405060708090a0b0c0d0e0f", 32), 0);
}

using ParamType = std::tuple<Form, uint16_t, uint8_t, DwarfFormat,
                             ArrayRef<uint8_t>, uint64_t, bool>;

struct FormSkipValueFixtureBase : public testing::TestWithParam<ParamType> {
  void SetUp() override {
    std::tie(Fm, Version, AddrSize, Dwarf, InitialData, ExpectedSkipped,
             ExpectedResult) = GetParam();
  }

  void doSkipValueTest() {
    SCOPED_TRACE("Inputs: Form = " + std::to_string(Fm) +
                 ", Version = " + std::to_string(Version) +
                 ", AddrSize = " + std::to_string(uint32_t(AddrSize)) +
                 ", DwarfFormat = " + std::to_string(Dwarf));
    std::vector<uint8_t> Buf(InitialData.data(),
                             InitialData.data() + InitialData.size());
    // The data extractor only adjusts the offset to the end of the buffer when
    // attempting to read past the end, so the buffer must be bigger than the
    // expected amount to be skipped to identify cases where more data than
    // expected is skipped.
    Buf.resize(ExpectedSkipped + 1);
    DWARFDataExtractor Data(Buf, sys::IsLittleEndianHost, AddrSize);
    uint64_t Offset = 0;
    EXPECT_EQ(DWARFFormValue::skipValue(Fm, Data, &Offset,
                                        {Version, AddrSize, Dwarf}),
              ExpectedResult);
    EXPECT_EQ(Offset, ExpectedSkipped);
  }

  Form Fm;
  uint16_t Version;
  uint8_t AddrSize;
  DwarfFormat Dwarf;
  ArrayRef<uint8_t> InitialData;
  uint64_t ExpectedSkipped;
  bool ExpectedResult;
};

template <typename T> static ArrayRef<uint8_t> toBytes(const T &Input) {
  return ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&Input),
                           sizeof(Input));
}

const uint8_t LEBData[] = {0x80, 0x1};
ArrayRef<uint8_t> SampleLEB(LEBData, sizeof(LEBData));
const uint8_t SampleLength8 = 0x80;
const uint16_t SampleLength16 = 0x80;
const uint32_t SampleLength = 0x80;
ArrayRef<uint8_t> SampleU8 = toBytes(SampleLength8);
ArrayRef<uint8_t> SampleU16 = toBytes(SampleLength16);
ArrayRef<uint8_t> SampleU32 = toBytes(SampleLength);
const uint8_t StringData[] = "abcdef";
ArrayRef<uint8_t> SampleString(StringData, sizeof(StringData));
const uint8_t IndirectData8[] = {DW_FORM_data8};
const uint8_t IndirectData16[] = {DW_FORM_data16};
const uint8_t IndirectAddr[] = {DW_FORM_addr};
const uint8_t IndirectIndirectData1[] = {DW_FORM_indirect, DW_FORM_data1};
const uint8_t IndirectIndirectEnd[] = {DW_FORM_indirect};

// Gtest's paramterised tests only allow a maximum of 50 cases, so split the
// test into multiple identical parts to share the cases.
struct FormSkipValueFixture1 : FormSkipValueFixtureBase {};
struct FormSkipValueFixture2 : FormSkipValueFixtureBase {};
TEST_P(FormSkipValueFixture1, skipValuePart1) { doSkipValueTest(); }
TEST_P(FormSkipValueFixture2, skipValuePart2) { doSkipValueTest(); }

INSTANTIATE_TEST_SUITE_P(
    SkipValueTestParams1, FormSkipValueFixture1,
    testing::Values(
        // Form, Version, AddrSize, DwarfFormat, InitialData, ExpectedSize,
        // ExpectedResult.
        ParamType(DW_FORM_exprloc, 0, 0, DWARF32, SampleLEB,
                  SampleLength + SampleLEB.size(), true),
        ParamType(DW_FORM_block, 0, 0, DWARF32, SampleLEB,
                  SampleLength + SampleLEB.size(), true),
        ParamType(DW_FORM_block1, 0, 0, DWARF32, SampleU8, SampleLength8 + 1,
                  true),
        ParamType(DW_FORM_block2, 0, 0, DWARF32, SampleU16, SampleLength16 + 2,
                  true),
        ParamType(DW_FORM_block4, 0, 0, DWARF32, SampleU32, SampleLength + 4,
                  true),
        ParamType(DW_FORM_string, 0, 0, DWARF32, SampleString,
                  SampleString.size(), true),
        ParamType(DW_FORM_addr, 0, 42, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_addr, 4, 0, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_addr, 4, 42, DWARF32, SampleU32, 42, true),
        ParamType(DW_FORM_ref_addr, 0, 1, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_ref_addr, 1, 0, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_ref_addr, 1, 1, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_ref_addr, 1, 1, DWARF64, SampleU32, 8, true),
        ParamType(DW_FORM_ref_addr, 2, 42, DWARF32, SampleU32, 42, true),
        ParamType(DW_FORM_ref_addr, 2, 42, DWARF64, SampleU32, 42, true),
        ParamType(DW_FORM_ref_addr, 3, 3, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_ref_addr, 3, 3, DWARF64, SampleU32, 8, true),
        ParamType(DW_FORM_flag_present, 4, 4, DWARF32, SampleU32, 0, true),
        ParamType(DW_FORM_data1, 0, 0, DWARF32, SampleU32, 1, true),
        ParamType(DW_FORM_data2, 0, 0, DWARF32, SampleU32, 2, true),
        ParamType(DW_FORM_data4, 0, 0, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_data8, 0, 0, DWARF32, SampleU32, 8, true),
        ParamType(DW_FORM_data16, 0, 0, DWARF32, SampleU32, 16, true),
        ParamType(DW_FORM_flag, 0, 0, DWARF32, SampleU32, 1, true),
        ParamType(DW_FORM_ref1, 0, 0, DWARF32, SampleU32, 1, true),
        ParamType(DW_FORM_ref2, 0, 0, DWARF32, SampleU32, 2, true),
        ParamType(DW_FORM_ref4, 0, 0, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_ref8, 0, 0, DWARF32, SampleU32, 8, true),
        ParamType(DW_FORM_ref_sig8, 0, 0, DWARF32, SampleU32, 8, true),
        ParamType(DW_FORM_ref_sup4, 0, 0, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_ref_sup8, 0, 0, DWARF32, SampleU32, 8, true),
        ParamType(DW_FORM_strx1, 0, 0, DWARF32, SampleU32, 1, true),
        ParamType(DW_FORM_strx2, 0, 0, DWARF32, SampleU32, 2, true),
        ParamType(DW_FORM_strx4, 0, 0, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_addrx1, 0, 0, DWARF32, SampleU32, 1, true),
        ParamType(DW_FORM_addrx2, 0, 0, DWARF32, SampleU32, 2, true),
        ParamType(DW_FORM_addrx4, 0, 0, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_sec_offset, 0, 1, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_sec_offset, 1, 0, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_sec_offset, 1, 1, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_sec_offset, 1, 1, DWARF64, SampleU32, 8, true),
        ParamType(DW_FORM_strp, 0, 1, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_strp, 1, 0, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_strp, 1, 1, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_strp, 1, 1, DWARF64, SampleU32, 8, true),
        ParamType(DW_FORM_strp_sup, 0, 1, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_strp_sup, 1, 0, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_strp_sup, 1, 1, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_strp_sup, 1, 1, DWARF64, SampleU32, 8, true)));

INSTANTIATE_TEST_SUITE_P(
    SkipValueTestParams2, FormSkipValueFixture2,
    testing::Values(
        ParamType(DW_FORM_line_strp, 0, 1, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_line_strp, 1, 0, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_line_strp, 1, 1, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_line_strp, 1, 1, DWARF64, SampleU32, 8, true),
        ParamType(DW_FORM_GNU_ref_alt, 0, 1, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_GNU_ref_alt, 1, 0, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_GNU_ref_alt, 1, 1, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_GNU_ref_alt, 1, 1, DWARF64, SampleU32, 8, true),
        ParamType(DW_FORM_GNU_strp_alt, 0, 1, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_GNU_strp_alt, 1, 0, DWARF32, SampleU32, 0, false),
        ParamType(DW_FORM_GNU_strp_alt, 1, 1, DWARF32, SampleU32, 4, true),
        ParamType(DW_FORM_GNU_strp_alt, 1, 1, DWARF64, SampleU32, 8, true),
        ParamType(DW_FORM_sdata, 0, 0, DWARF32, SampleLEB, SampleLEB.size(),
                  true),
        ParamType(DW_FORM_udata, 0, 0, DWARF32, SampleLEB, SampleLEB.size(),
                  true),
        ParamType(DW_FORM_ref_udata, 0, 0, DWARF32, SampleLEB, SampleLEB.size(),
                  true),
        ParamType(DW_FORM_strx, 0, 0, DWARF32, SampleLEB, SampleLEB.size(),
                  true),
        ParamType(DW_FORM_addrx, 0, 0, DWARF32, SampleLEB, SampleLEB.size(),
                  true),
        ParamType(DW_FORM_loclistx, 0, 0, DWARF32, SampleLEB, SampleLEB.size(),
                  true),
        ParamType(DW_FORM_rnglistx, 0, 0, DWARF32, SampleLEB, SampleLEB.size(),
                  true),
        ParamType(DW_FORM_GNU_addr_index, 0, 0, DWARF32, SampleLEB,
                  SampleLEB.size(), true),
        ParamType(DW_FORM_GNU_str_index, 0, 0, DWARF32, SampleLEB,
                  SampleLEB.size(), true),
        ParamType(DW_FORM_indirect, 0, 0, DWARF32,
                  ArrayRef<uint8_t>(IndirectData8, sizeof(IndirectData8)), 9,
                  true),
        ParamType(DW_FORM_indirect, 0, 0, DWARF32,
                  ArrayRef<uint8_t>(IndirectData16, sizeof(IndirectData16)), 17,
                  true),
        ParamType(DW_FORM_indirect, 4, 0, DWARF32,
                  ArrayRef<uint8_t>(IndirectAddr, sizeof(IndirectAddr)), 1,
                  false),
        ParamType(DW_FORM_indirect, 4, 4, DWARF32,
                  ArrayRef<uint8_t>(IndirectAddr, sizeof(IndirectAddr)), 5,
                  true),
        ParamType(DW_FORM_indirect, 4, 4, DWARF32,
                  ArrayRef<uint8_t>(IndirectIndirectData1,
                                    sizeof(IndirectIndirectData1)),
                  3, true),
        ParamType(DW_FORM_indirect, 4, 4, DWARF32,
                  ArrayRef<uint8_t>(IndirectIndirectEnd,
                                    sizeof(IndirectIndirectEnd)),
                  2, false),
        ParamType(/*Unknown=*/Form(0xff), 4, 4, DWARF32, SampleU32, 0, false)));

using ErrorParams = std::tuple<Form, std::vector<uint8_t>>;
struct ExtractValueErrorFixture : public testing::TestWithParam<ErrorParams> {
  void SetUp() override { std::tie(Fm, InitialData) = GetParam(); }

  Form Fm;
  ArrayRef<uint8_t> InitialData;
};

TEST_P(ExtractValueErrorFixture, Test) {
  SCOPED_TRACE(formatv("Fm = {0}, InitialData = {1}", Fm,
                       make_range(InitialData.begin(), InitialData.end()))
                   .str());

  DWARFDataExtractor Data(InitialData, sys::IsLittleEndianHost, 4);
  DWARFFormValue Form(Fm);
  uint64_t Offset = 0;
  EXPECT_FALSE(Form.extractValue(Data, &Offset, {0, 0, DWARF32}));
}

INSTANTIATE_TEST_SUITE_P(
    ExtractValueErrorParams, ExtractValueErrorFixture,
    testing::Values(
        ErrorParams{DW_FORM_ref_addr, {}}, ErrorParams{DW_FORM_block, {}},
        ErrorParams{DW_FORM_block, {1}}, ErrorParams{DW_FORM_block, {2, 0}},
        ErrorParams{DW_FORM_block1, {}}, ErrorParams{DW_FORM_block2, {}},
        ErrorParams{DW_FORM_block4, {}}, ErrorParams{DW_FORM_data1, {}},
        ErrorParams{DW_FORM_data2, {}}, ErrorParams{DW_FORM_strx3, {}},
        ErrorParams{DW_FORM_data4, {}}, ErrorParams{DW_FORM_data8, {}},
        ErrorParams{DW_FORM_data16, {}}, ErrorParams{DW_FORM_sdata, {}},
        ErrorParams{DW_FORM_udata, {}}, ErrorParams{DW_FORM_string, {}},
        ErrorParams{DW_FORM_indirect, {}},
        ErrorParams{DW_FORM_indirect, {DW_FORM_data1}},
        ErrorParams{DW_FORM_strp_sup, {}}, ErrorParams{DW_FORM_ref_sig8, {}}));

using DumpValueParams =
    std::tuple<Form, ArrayRef<uint8_t>, DwarfFormat, StringRef>;
struct DumpValueFixture : public testing::TestWithParam<DumpValueParams> {
  void SetUp() override {
    std::tie(Fm, InitialData, Format, ExpectedResult) = GetParam();
  }

  Form Fm;
  ArrayRef<uint8_t> InitialData;
  DwarfFormat Format;
  StringRef ExpectedResult;
};

TEST_P(DumpValueFixture, Test) {
  SCOPED_TRACE(formatv("Fm = {0}, InitialData = [{1}], Format = {2}", Fm,
                       toHex(InitialData),
                       Format == DWARF64 ? "DWARF64" : "DWARF32"));
  DWARFDataExtractor Data(InitialData, sys::IsLittleEndianHost, 8);
  DWARFFormValue Form(Fm);
  uint64_t Offset = 0;
  Form.extractValue(Data, &Offset, {0, 0, Format});

  std::string Output;
  raw_string_ostream OS(Output);

  DIDumpOptions Opts;
  Opts.Verbose = true;
  Opts.ShowAddresses = true;

  Form.dump(OS, Opts);
  OS.flush();

  EXPECT_EQ(Output, ExpectedResult);
}

const uint32_t DumpTestSample32Val = 0x112233;
ArrayRef<uint8_t> DumpTestSample32 = toBytes(DumpTestSample32Val);
const uint64_t DumpTestSample64Val = 0x11223344556677;
ArrayRef<uint8_t> DumpTestSample64 = toBytes(DumpTestSample64Val);

INSTANTIATE_TEST_SUITE_P(
    DumpValueParams, DumpValueFixture,
    testing::Values(DumpValueParams{DW_FORM_strp, DumpTestSample32, DWARF32,
                                    " .debug_str[0x00112233] = "},
                    DumpValueParams{DW_FORM_strp, DumpTestSample64, DWARF64,
                                    " .debug_str[0x0011223344556677] = "},
                    DumpValueParams{DW_FORM_line_strp, DumpTestSample32,
                                    DWARF32, " .debug_line_str[0x00112233] = "},
                    DumpValueParams{DW_FORM_line_strp, DumpTestSample64,
                                    DWARF64,
                                    " .debug_line_str[0x0011223344556677] = "},
                    DumpValueParams{DW_FORM_sec_offset, DumpTestSample32,
                                    DWARF32, "0x00112233"},
                    DumpValueParams{DW_FORM_sec_offset, DumpTestSample64,
                                    DWARF64, "0x0011223344556677"}));

} // end anonymous namespace
