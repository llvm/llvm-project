//===- DWARFDataExtractorTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(DWARFDataExtractorTest, getRelocatedValue) {
  StringRef Yaml = R"(
!ELF
FileHeader:
  Class:    ELFCLASS32
  Data:     ELFDATA2LSB
  Type:     ET_REL
  Machine:  EM_386
Sections:
  - Name:     .text
    Type:     SHT_PROGBITS
    Size:     0x80
  - Name:     .debug_line
    Type:     SHT_PROGBITS
    Content:  '000000000000'
  - Name:     .rel.debug_line
    Type:     SHT_REL
    Info:     .debug_line
    Relocations:
      - Offset:   0
        Symbol:   f
        Type:     R_386_32
      - Offset:   4
        Symbol:   f
        Type:     R_386_32
Symbols:
  - Name:     f
    Type:     STT_SECTION
    Section:  .text
    Value:    0x42
)";
  SmallString<0> Storage;
  std::unique_ptr<object::ObjectFile> Obj = yaml::yaml2ObjectFile(
      Storage, Yaml, [](const Twine &Err) { errs() << Err; });
  ASSERT_TRUE(Obj);
  std::unique_ptr<DWARFContext> Ctx = DWARFContext::create(*Obj);
  const DWARFObject &DObj = Ctx->getDWARFObj();
  ASSERT_EQ(6u, DObj.getLineSection().Data.size());

  DWARFDataExtractor Data(DObj, DObj.getLineSection(), Obj->isLittleEndian(),
                          Obj->getBytesInAddress());
  DataExtractor::Cursor C(0);
  EXPECT_EQ(0x42u, Data.getRelocatedAddress(C));
  EXPECT_EQ(0u, Data.getRelocatedAddress(C));
  EXPECT_THAT_ERROR(
      C.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x6 while reading [0x4, 0x8)"));
}

TEST(DWARFDataExtractorTest, getInitialLength) {
  auto GetWithError = [](ArrayRef<uint8_t> Bytes)
      -> Expected<std::tuple<uint64_t, dwarf::DwarfFormat, uint64_t>> {
    DWARFDataExtractor Data(Bytes, /*IsLittleEndian=*/false, /*AddressSize=*/8);
    DWARFDataExtractor::Cursor C(0);
    uint64_t Length;
    dwarf::DwarfFormat Format;
    std::tie(Length, Format) = Data.getInitialLength(C);
    if (C)
      return std::make_tuple(Length, Format, C.tell());

    EXPECT_EQ(Length, 0u);
    EXPECT_EQ(Format, dwarf::DWARF32);
    EXPECT_EQ(C.tell(), 0u);
    return C.takeError();
  };
  auto GetWithoutError = [](ArrayRef<uint8_t> Bytes) {
    DWARFDataExtractor Data(Bytes, /*IsLittleEndian=*/false, /*AddressSize=*/8);
    uint64_t Offset = 0;
    uint64_t Length;
    dwarf::DwarfFormat Format;
    std::tie(Length, Format) = Data.getInitialLength(&Offset);
    return std::make_tuple(Length, Format, Offset);
  };
  auto ErrorResult = std::make_tuple(0, dwarf::DWARF32, 0);

  // Empty data.
  EXPECT_THAT_EXPECTED(
      GetWithError({}),
      FailedWithMessage(
          "unexpected end of data at offset 0x0 while reading [0x0, 0x4)"));
  EXPECT_EQ(GetWithoutError({}), ErrorResult);

  // Not long enough for the U32 field.
  EXPECT_THAT_EXPECTED(
      GetWithError({0x00, 0x01, 0x02}),
      FailedWithMessage(
          "unexpected end of data at offset 0x3 while reading [0x0, 0x4)"));
  EXPECT_EQ(GetWithoutError({0x00, 0x01, 0x02}), ErrorResult);

  EXPECT_THAT_EXPECTED(
      GetWithError({0x00, 0x01, 0x02, 0x03}),
      HasValue(std::make_tuple(0x00010203, dwarf::DWARF32, 4)));
  EXPECT_EQ(GetWithoutError({0x00, 0x01, 0x02, 0x03}),
            std::make_tuple(0x00010203, dwarf::DWARF32, 4));

  // Zeroes are not an error, but without the Error object it is hard to tell
  // them apart from a failed read.
  EXPECT_THAT_EXPECTED(
      GetWithError({0x00, 0x00, 0x00, 0x00}),
      HasValue(std::make_tuple(0x00000000, dwarf::DWARF32, 4)));
  EXPECT_EQ(GetWithoutError({0x00, 0x00, 0x00, 0x00}),
            std::make_tuple(0x00000000, dwarf::DWARF32, 4));

  // Smallest invalid value.
  EXPECT_THAT_EXPECTED(
      GetWithError({0xff, 0xff, 0xff, 0xf0}),
      FailedWithMessage(
          "unsupported reserved unit length of value 0xfffffff0"));
  EXPECT_EQ(GetWithoutError({0xff, 0xff, 0xff, 0xf0}), ErrorResult);

  // DWARF64 marker without the subsequent length field.
  EXPECT_THAT_EXPECTED(
      GetWithError({0xff, 0xff, 0xff, 0xff}),
      FailedWithMessage(
          "unexpected end of data at offset 0x4 while reading [0x4, 0xc)"));
  EXPECT_EQ(GetWithoutError({0xff, 0xff, 0xff, 0xff}), ErrorResult);

  // Not enough data for the U64 length.
  EXPECT_THAT_EXPECTED(
      GetWithError({0xff, 0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x03}),
      FailedWithMessage(
          "unexpected end of data at offset 0x8 while reading [0x4, 0xc)"));
  EXPECT_EQ(GetWithoutError({0xff, 0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x03}),
            ErrorResult);

  EXPECT_THAT_EXPECTED(
      GetWithError({0xff, 0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
                    0x06, 0x07}),
      HasValue(std::make_tuple(0x0001020304050607, dwarf::DWARF64, 12)));
  EXPECT_EQ(GetWithoutError({0xff, 0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x03,
                             0x04, 0x05, 0x06, 0x07}),
            std::make_tuple(0x0001020304050607, dwarf::DWARF64, 12));
}

TEST(DWARFDataExtractorTest, Truncation) {
  StringRef Yaml = R"(
!ELF
FileHeader:
  Class:    ELFCLASS32
  Data:     ELFDATA2LSB
  Type:     ET_REL
  Machine:  EM_386
Sections:
  - Name:     .text
    Type:     SHT_PROGBITS
    Size:     0x80
  - Name:     .debug_line
    Type:     SHT_PROGBITS
    Content:  '616263640000000065666768'
  - Name:     .rel.debug_line
    Type:     SHT_REL
    Info:     .debug_line
    Relocations:
      - Offset:   4
        Symbol:   f
        Type:     R_386_32
Symbols:
  - Name:     f
    Type:     STT_SECTION
    Section:  .text
    Value:    0x42
)";
  SmallString<0> Storage;
  std::unique_ptr<object::ObjectFile> Obj = yaml::yaml2ObjectFile(
      Storage, Yaml, [](const Twine &Err) { errs() << Err; });
  ASSERT_TRUE(Obj);
  std::unique_ptr<DWARFContext> Ctx = DWARFContext::create(*Obj);
  const DWARFObject &DObj = Ctx->getDWARFObj();
  ASSERT_EQ(12u, DObj.getLineSection().Data.size());

  DWARFDataExtractor Data(DObj, DObj.getLineSection(), Obj->isLittleEndian(),
                          Obj->getBytesInAddress());
  DataExtractor::Cursor C(0);
  EXPECT_EQ(0x64636261u, Data.getRelocatedAddress(C));
  EXPECT_EQ(0x42u, Data.getRelocatedAddress(C));
  EXPECT_EQ(0x68676665u, Data.getRelocatedAddress(C));
  EXPECT_THAT_ERROR(C.takeError(), Succeeded());

  C = DataExtractor::Cursor{0};
  DWARFDataExtractor Truncated8(Data, 8);
  EXPECT_EQ(0x64636261u, Truncated8.getRelocatedAddress(C));
  EXPECT_EQ(0x42u, Truncated8.getRelocatedAddress(C));
  EXPECT_EQ(0x0u, Truncated8.getRelocatedAddress(C));
  EXPECT_THAT_ERROR(
      C.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x8 while reading [0x8, 0xc)"));

  C = DataExtractor::Cursor{0};
  DWARFDataExtractor Truncated6(Data, 6);
  EXPECT_EQ(0x64636261u, Truncated6.getRelocatedAddress(C));
  EXPECT_EQ(0x0u, Truncated6.getRelocatedAddress(C));
  EXPECT_THAT_ERROR(
      C.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x6 while reading [0x4, 0x8)"));

  C = DataExtractor::Cursor{0};
  DWARFDataExtractor Truncated2(Data, 2);
  EXPECT_EQ(0x0u, Truncated2.getRelocatedAddress(C));
  EXPECT_THAT_ERROR(
      C.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x2 while reading [0x0, 0x4)"));
}

} // namespace
