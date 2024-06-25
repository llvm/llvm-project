//===- DWARFDebugLineTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DwarfGenerator.h"
#include "DwarfUtils.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

// AIX doesn't support the debug_addr section
#ifdef _AIX
#define NO_SUPPORT_DEBUG_ADDR
#endif

using namespace llvm;
using namespace dwarf;
using namespace dwarfgen;
using namespace object;
using namespace utils;
using namespace testing;

namespace {
struct CommonFixture {
  CommonFixture()
      : LineData("", true, 0), Recoverable(Error::success()),
        RecordRecoverable(std::bind(&CommonFixture::recordRecoverable, this,
                                    std::placeholders::_1)),
        Unrecoverable(Error::success()),
        RecordUnrecoverable(std::bind(&CommonFixture::recordUnrecoverable, this,
                                      std::placeholders::_1)){};

  ~CommonFixture() {
    EXPECT_FALSE(Recoverable);
    EXPECT_FALSE(Unrecoverable);
  }

  // Note: ASSERT_THAT_EXPECTED cannot be used in a non-void function, so
  // setupGenerator() is split into two.
  void setupGeneratorImpl(uint16_t Version, uint8_t AddrSize) {
    AddressSize = AddrSize;
    Triple T = getDefaultTargetTripleForAddrSize(AddressSize ? AddressSize : 8);
    if (!isConfigurationSupported(T))
      return;
    auto ExpectedGenerator = Generator::create(T, Version);
    ASSERT_THAT_EXPECTED(ExpectedGenerator, Succeeded());
    Gen = std::move(*ExpectedGenerator);
  }

  bool setupGenerator(uint16_t Version = 4, uint8_t AddrSize = 8) {
    setupGeneratorImpl(Version, AddrSize);
    return Gen != nullptr;
  }

  void generate() {
    Context = createContext();
    assert(Context != nullptr && "test state is not valid");
    const DWARFObject &Obj = Context->getDWARFObj();
    uint8_t TargetAddrSize = AddressSize == 0 ? 8 : AddressSize;
    LineData = DWARFDataExtractor(
        Obj, Obj.getLineSection(),
        getDefaultTargetTripleForAddrSize(TargetAddrSize).isLittleEndian(),
        AddressSize);
  }

  std::unique_ptr<DWARFContext> createContext() {
    assert(Gen != nullptr && "Generator is not set up");
    StringRef FileBytes = Gen->generate();
    MemoryBufferRef FileBuffer(FileBytes, "dwarf");
    auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
    if (Obj)
      return DWARFContext::create(**Obj);
    return nullptr;
  }

  DWARFDebugLine::SectionParser setupParser() {
    LineTable &LT = Gen->addLineTable(DWARF32);
    LT.addExtendedOpcode(9, DW_LNE_set_address, {{0xadd4e55, LineTable::Quad}});
    LT.addStandardOpcode(DW_LNS_copy, {});
    LT.addByte(0xaa);
    LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

    LineTable &LT2 = Gen->addLineTable(DWARF64);
    LT2.addExtendedOpcode(9, DW_LNE_set_address,
                          {{0x11223344, LineTable::Quad}});
    LT2.addStandardOpcode(DW_LNS_copy, {});
    LT2.addByte(0xbb);
    LT2.addExtendedOpcode(1, DW_LNE_end_sequence, {});

    generate();

    return DWARFDebugLine::SectionParser(LineData, *Context, Units);
  }

  void recordRecoverable(Error Err) {
    Recoverable = joinErrors(std::move(Recoverable), std::move(Err));
  }
  void recordUnrecoverable(Error Err) {
    Unrecoverable = joinErrors(std::move(Unrecoverable), std::move(Err));
  }

  Expected<const DWARFDebugLine::LineTable *>
  getOrParseLineTableFatalErrors(uint64_t Offset = 0) {
    auto ExpectedLineTable = Line.getOrParseLineTable(
        LineData, Offset, *Context, nullptr, RecordRecoverable);
    EXPECT_THAT_ERROR(std::move(Recoverable), Succeeded());
    return ExpectedLineTable;
  }

  uint8_t AddressSize;
  std::unique_ptr<Generator> Gen;
  std::unique_ptr<DWARFContext> Context;
  DWARFDataExtractor LineData;
  DWARFDebugLine Line;
  Error Recoverable;
  std::function<void(Error)> RecordRecoverable;
  Error Unrecoverable;
  std::function<void(Error)> RecordUnrecoverable;

  SmallVector<std::unique_ptr<DWARFUnit>, 2> Units;
};

// Fixtures must derive from "Test", but parameterised fixtures from
// "TestWithParam". It does not seem possible to inherit from both, so we share
// the common state in a separate class, inherited by the two fixture classes.
struct DebugLineBasicFixture : public Test, public CommonFixture {};

struct DebugLineParameterisedFixture
    : public TestWithParam<std::pair<uint16_t, DwarfFormat>>,
      public CommonFixture {
  void SetUp() override { std::tie(Version, Format) = GetParam(); }

  uint16_t Version;
  DwarfFormat Format;
};

void checkDefaultPrologue(uint16_t Version, DwarfFormat Format,
                          DWARFDebugLine::Prologue Prologue,
                          uint64_t BodyLength) {
  // Check version specific fields and values.
  uint64_t UnitLength;
  uint64_t PrologueLength;
  switch (Version) {
  case 4:
    PrologueLength = 36;
    UnitLength = PrologueLength + 2;
    EXPECT_EQ(Prologue.MaxOpsPerInst, 1u);
    break;
  case 2:
  case 3:
    PrologueLength = 35;
    UnitLength = PrologueLength + 2;
    break;
  case 5:
    PrologueLength = 42;
    UnitLength = PrologueLength + 4;
    EXPECT_EQ(Prologue.getAddressSize(), 8u);
    EXPECT_EQ(Prologue.SegSelectorSize, 0u);
    break;
  default:
    llvm_unreachable("unsupported DWARF version");
  }
  UnitLength += BodyLength + (Format == DWARF32 ? 4 : 8);

  EXPECT_EQ(Prologue.TotalLength, UnitLength);
  EXPECT_EQ(Prologue.PrologueLength, PrologueLength);
  EXPECT_EQ(Prologue.MinInstLength, 1u);
  EXPECT_EQ(Prologue.DefaultIsStmt, 1u);
  EXPECT_EQ(Prologue.LineBase, -5);
  EXPECT_EQ(Prologue.LineRange, 14u);
  EXPECT_EQ(Prologue.OpcodeBase, 13u);
  std::vector<uint8_t> ExpectedLengths = {0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1};
  EXPECT_EQ(Prologue.StandardOpcodeLengths, ExpectedLengths);
  ASSERT_EQ(Prologue.IncludeDirectories.size(), 1u);
  ASSERT_EQ(Prologue.IncludeDirectories[0].getForm(), DW_FORM_string);
  EXPECT_STREQ(*toString(Prologue.IncludeDirectories[0]), "a dir");
  ASSERT_EQ(Prologue.FileNames.size(), 1u);
  ASSERT_EQ(Prologue.FileNames[0].Name.getForm(), DW_FORM_string);
  ASSERT_EQ(Prologue.FileNames[0].DirIdx, 0u);
  EXPECT_STREQ(*toString(Prologue.FileNames[0].Name), "a file");
}

TEST_F(DebugLineBasicFixture, GetOrParseLineTableAtInvalidOffset) {
  if (!setupGenerator())
    GTEST_SKIP();
  generate();

  EXPECT_THAT_EXPECTED(
      getOrParseLineTableFatalErrors(0),
      FailedWithMessage(
          "offset 0x00000000 is not a valid debug line section offset"));
  // Repeat to show that an error is reported each time.
  EXPECT_THAT_EXPECTED(
      getOrParseLineTableFatalErrors(0),
      FailedWithMessage(
          "offset 0x00000000 is not a valid debug line section offset"));

  // Show that an error is reported for later offsets too.
  EXPECT_THAT_EXPECTED(
      getOrParseLineTableFatalErrors(1),
      FailedWithMessage(
          "offset 0x00000001 is not a valid debug line section offset"));
}

TEST_F(DebugLineBasicFixture, GetOrParseLineTableAtInvalidOffsetAfterData) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue({{0, LineTable::Byte}});

  generate();

  EXPECT_THAT_EXPECTED(
      getOrParseLineTableFatalErrors(0),
      FailedWithMessage(
          "parsing line table prologue at offset 0x00000000: "
          "unexpected end of data at offset 0x1 while reading [0x0, 0x4)"));

  EXPECT_THAT_EXPECTED(
      getOrParseLineTableFatalErrors(1),
      FailedWithMessage(
          "offset 0x00000001 is not a valid debug line section offset"));
}

#ifdef NO_SUPPORT_DEBUG_ADDR
TEST_P(DebugLineParameterisedFixture, DISABLED_PrologueGetLength) {
#else
TEST_P(DebugLineParameterisedFixture, PrologueGetLength) {
#endif
  if (!setupGenerator(Version))
    GTEST_SKIP();
  LineTable &LT = Gen->addLineTable(Format);
  DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();
  LT.setPrologue(Prologue);
  generate();

  // + 10 for sizes of DWARF-32 unit length, version, prologue length.
  uint64_t ExpectedLength = Prologue.PrologueLength + 10;
  if (Version == 5)
    // Add address and segment selector size fields.
    ExpectedLength += 2;
  if (Format == DWARF64)
    // Unit length grows by 8, prologue length by 4.
    ExpectedLength += 12;

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  EXPECT_EQ((*ExpectedLineTable)->Prologue.getLength(), ExpectedLength);
}

#ifdef NO_SUPPORT_DEBUG_ADDR
TEST_P(DebugLineParameterisedFixture, DISABLED_GetOrParseLineTableValidTable) {
#else
TEST_P(DebugLineParameterisedFixture, GetOrParseLineTableValidTable) {
#endif
  if (!setupGenerator(Version))
    GTEST_SKIP();

  SCOPED_TRACE("Checking Version " + std::to_string(Version) + ", Format " +
               (Format == DWARF64 ? "DWARF64" : "DWARF32"));

  LineTable &LT = Gen->addLineTable(Format);
  LT.addExtendedOpcode(9, DW_LNE_set_address, {{0xadd4e55, LineTable::Quad}});
  LT.addStandardOpcode(DW_LNS_copy, {});
  LT.addByte(0xaa);
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  LineTable &LT2 = Gen->addLineTable(Format);
  LT2.addExtendedOpcode(9, DW_LNE_set_address, {{0x11223344, LineTable::Quad}});
  LT2.addStandardOpcode(DW_LNS_copy, {});
  LT2.addByte(0xbb);
  LT2.addExtendedOpcode(1, DW_LNE_end_sequence, {});
  LT2.addExtendedOpcode(9, DW_LNE_set_address, {{0x55667788, LineTable::Quad}});
  LT2.addStandardOpcode(DW_LNS_copy, {});
  LT2.addByte(0xcc);
  LT2.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  ASSERT_TRUE(ExpectedLineTable.operator bool());
  EXPECT_FALSE(Recoverable);
  const DWARFDebugLine::LineTable *Expected = *ExpectedLineTable;
  checkDefaultPrologue(Version, Format, Expected->Prologue, 16);
  EXPECT_EQ(Expected->Sequences.size(), 1u);

  uint64_t SecondOffset =
      Expected->Prologue.sizeofTotalLength() + Expected->Prologue.TotalLength;
  Recoverable = Error::success();
  auto ExpectedLineTable2 = Line.getOrParseLineTable(
      LineData, SecondOffset, *Context, nullptr, RecordRecoverable);
  ASSERT_TRUE(ExpectedLineTable2.operator bool());
  EXPECT_FALSE(Recoverable);
  const DWARFDebugLine::LineTable *Expected2 = *ExpectedLineTable2;
  checkDefaultPrologue(Version, Format, Expected2->Prologue, 32);
  EXPECT_EQ(Expected2->Sequences.size(), 2u);

  EXPECT_NE(Expected, Expected2);

  // Check that if the same offset is requested, the exact same pointer is
  // returned.
  Recoverable = Error::success();
  auto ExpectedLineTable3 = Line.getOrParseLineTable(
      LineData, 0, *Context, nullptr, RecordRecoverable);
  ASSERT_TRUE(ExpectedLineTable3.operator bool());
  EXPECT_FALSE(Recoverable);
  EXPECT_EQ(Expected, *ExpectedLineTable3);

  Recoverable = Error::success();
  auto ExpectedLineTable4 = Line.getOrParseLineTable(
      LineData, SecondOffset, *Context, nullptr, RecordRecoverable);
  ASSERT_TRUE(ExpectedLineTable4.operator bool());
  EXPECT_FALSE(Recoverable);
  EXPECT_EQ(Expected2, *ExpectedLineTable4);

  // TODO: Add tests that show that the body of the programs have been read
  // correctly.
}

#ifdef NO_SUPPORT_DEBUG_ADDR
TEST_P(DebugLineParameterisedFixture, DISABLED_ClearLineValidTable) {
#else
TEST_P(DebugLineParameterisedFixture, ClearLineValidTable) {
#endif
  if (!setupGenerator(Version))
    GTEST_SKIP();

  SCOPED_TRACE("Checking Version " + std::to_string(Version) + ", Format " +
               (Format == DWARF64 ? "DWARF64" : "DWARF32"));

  LineTable &LT = Gen->addLineTable(Format);
  LT.addExtendedOpcode(9, DW_LNE_set_address, {{0xadd4e55, LineTable::Quad}});
  LT.addStandardOpcode(DW_LNS_copy, {});
  LT.addByte(0xaa);
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  LineTable &LT2 = Gen->addLineTable(Format);
  LT2.addExtendedOpcode(9, DW_LNE_set_address, {{0x11223344, LineTable::Quad}});
  LT2.addStandardOpcode(DW_LNS_copy, {});
  LT2.addByte(0xbb);
  LT2.addExtendedOpcode(1, DW_LNE_end_sequence, {});
  LT2.addExtendedOpcode(9, DW_LNE_set_address, {{0x55667788, LineTable::Quad}});
  LT2.addStandardOpcode(DW_LNS_copy, {});
  LT2.addByte(0xcc);
  LT2.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  generate();

  // Check that we have what we expect before calling clearLineTable().
  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  ASSERT_TRUE((bool)ExpectedLineTable);
  EXPECT_FALSE(Recoverable);
  const DWARFDebugLine::LineTable *Expected = *ExpectedLineTable;
  checkDefaultPrologue(Version, Format, Expected->Prologue, 16);
  EXPECT_EQ(Expected->Sequences.size(), 1u);

  uint64_t SecondOffset =
      Expected->Prologue.sizeofTotalLength() + Expected->Prologue.TotalLength;
  Recoverable = Error::success();
  auto ExpectedLineTable2 = Line.getOrParseLineTable(
      LineData, SecondOffset, *Context, nullptr, RecordRecoverable);
  ASSERT_TRUE((bool)ExpectedLineTable2);
  EXPECT_FALSE(Recoverable);
  const DWARFDebugLine::LineTable *Expected2 = *ExpectedLineTable2;
  checkDefaultPrologue(Version, Format, Expected2->Prologue, 32);
  EXPECT_EQ(Expected2->Sequences.size(), 2u);

  // Check that we no longer get the line tables after clearLineTable().
  Line.clearLineTable(0);
  Line.clearLineTable(SecondOffset);
  EXPECT_EQ(Line.getLineTable(0), nullptr);
  EXPECT_EQ(Line.getLineTable(SecondOffset), nullptr);

  // Check that if the same offset is requested, the contents match what we
  // had before.
  Recoverable = Error::success();
  auto ExpectedLineTable3 = Line.getOrParseLineTable(
      LineData, 0, *Context, nullptr, RecordRecoverable);
  ASSERT_TRUE((bool)ExpectedLineTable3);
  EXPECT_FALSE(Recoverable);
  const DWARFDebugLine::LineTable *Expected3 = *ExpectedLineTable3;
  checkDefaultPrologue(Version, Format, Expected3->Prologue, 16);
  EXPECT_EQ(Expected3->Sequences.size(), 1u);

  Recoverable = Error::success();
  auto ExpectedLineTable4 = Line.getOrParseLineTable(
      LineData, SecondOffset, *Context, nullptr, RecordRecoverable);
  ASSERT_TRUE((bool)ExpectedLineTable4);
  EXPECT_FALSE(Recoverable);
  const DWARFDebugLine::LineTable *Expected4 = *ExpectedLineTable4;
  checkDefaultPrologue(Version, Format, Expected4->Prologue, 32);
  EXPECT_EQ(Expected4->Sequences.size(), 2u);
}

TEST_F(DebugLineBasicFixture, ErrorForReservedLength) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue({{0xfffffff0, LineTable::Long}});

  generate();

  EXPECT_THAT_EXPECTED(
      getOrParseLineTableFatalErrors(),
      FailedWithMessage(
          "parsing line table prologue at offset 0x00000000: unsupported "
          "reserved unit length of value 0xfffffff0"));
}

struct DebugLineUnsupportedVersionFixture : public TestWithParam<uint16_t>,
                                            public CommonFixture {
  void SetUp() override { Version = GetParam(); }

  uint16_t Version;
};

TEST_P(DebugLineUnsupportedVersionFixture, ErrorForUnsupportedVersion) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue(
      {{LineTable::Half, LineTable::Long}, {Version, LineTable::Half}});

  generate();

  EXPECT_THAT_EXPECTED(
      getOrParseLineTableFatalErrors(),
      FailedWithMessage("parsing line table prologue at offset 0x00000000: "
                        "unsupported version " +
                        std::to_string(Version)));
}

INSTANTIATE_TEST_SUITE_P(UnsupportedVersionTestParams,
                         DebugLineUnsupportedVersionFixture,
                         Values(/*1 below min */ 1, /* 1 above max */ 6,
                                /* Maximum possible */ 0xffff));

#ifdef NO_SUPPORT_DEBUG_ADDR
TEST_F(DebugLineBasicFixture, DISABLED_ErrorForInvalidV5IncludeDirTable) {
#else
TEST_F(DebugLineBasicFixture, ErrorForInvalidV5IncludeDirTable) {
#endif
  if (!setupGenerator(5))
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue({
      {19, LineTable::Long}, // unit length
      {5, LineTable::Half},  // version
      {8, LineTable::Byte},  // addr size
      {0, LineTable::Byte},  // segment selector size
      {11, LineTable::Long}, // prologue length
      {1, LineTable::Byte},  // min instruction length
      {1, LineTable::Byte},  // max ops per instruction
      {1, LineTable::Byte},  // default is_stmt
      {0, LineTable::Byte},  // line base
      {14, LineTable::Byte}, // line range
      {2, LineTable::Byte},  // opcode base (small to reduce the amount of
                             // setup required).
      {0, LineTable::Byte},  // standard opcode lengths
      {0, LineTable::Byte},  // directory entry format count (should not be
                             // zero).
      {0, LineTable::ULEB},  // directories count
      {0, LineTable::Byte},  // file name entry format count
      {0, LineTable::ULEB}   // file name entry count
  });

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  EXPECT_THAT_EXPECTED(ExpectedLineTable, Succeeded());

  EXPECT_THAT_ERROR(
      std::move(Recoverable),
      FailedWithMessage(
          "parsing line table prologue at 0x00000000 found an invalid "
          "directory or file table description at 0x00000014",
          "failed to parse entry content descriptions because no path was "
          "found"));
}

#ifdef NO_SUPPORT_DEBUG_ADDR
TEST_P(DebugLineParameterisedFixture, DISABLED_ErrorForTooLargePrologueLength) {
#else
TEST_P(DebugLineParameterisedFixture, ErrorForTooLargePrologueLength) {
#endif
  if (!setupGenerator(Version))
    GTEST_SKIP();

  SCOPED_TRACE("Checking Version " + std::to_string(Version) + ", Format " +
               (Format == DWARF64 ? "DWARF64" : "DWARF32"));

  LineTable &LT = Gen->addLineTable(Format);
  DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();
  ++Prologue.PrologueLength;
  LT.setPrologue(Prologue);

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  DWARFDebugLine::LineTable Result(**ExpectedLineTable);
  // Undo the earlier modification so that it can be compared against a
  // "default" prologue.
  --Result.Prologue.PrologueLength;
  checkDefaultPrologue(Version, Format, Result.Prologue, 0);

  uint64_t ExpectedEnd =
      Prologue.TotalLength + 1 + Prologue.sizeofTotalLength();
  EXPECT_THAT_ERROR(
      std::move(Recoverable),
      FailedWithMessage(
          ("unknown data in line table prologue at offset 0x00000000: "
           "parsing ended (at offset 0x000000" +
           Twine::utohexstr(ExpectedEnd - 1) +
           ") before reaching the prologue end at offset 0x000000" +
           Twine::utohexstr(ExpectedEnd))
              .str()));
}

#ifdef NO_SUPPORT_DEBUG_ADDR
TEST_P(DebugLineParameterisedFixture, DISABLED_ErrorForTooShortPrologueLength) {
#else
TEST_P(DebugLineParameterisedFixture, ErrorForTooShortPrologueLength) {
#endif
  if (!setupGenerator(Version))
    GTEST_SKIP();

  SCOPED_TRACE("Checking Version " + std::to_string(Version) + ", Format " +
               (Format == DWARF64 ? "DWARF64" : "DWARF32"));

  LineTable &LT = Gen->addLineTable(Format);
  DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();
  Prologue.PrologueLength -= 2;
  LT.setPrologue(Prologue);

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  DWARFDebugLine::LineTable Result(**ExpectedLineTable);

  // Parsing will stop before reading a complete file entry.
  ASSERT_EQ(Result.Prologue.IncludeDirectories.size(), 1u);
  EXPECT_EQ(toStringRef(Result.Prologue.IncludeDirectories[0]), "a dir");
  EXPECT_EQ(Result.Prologue.FileNames.size(), 0u);

  // The exact place where the parsing will stop depends on the structure of the
  // prologue and the last complete field we are able to read. Before V5 we stop
  // before reading the file length. In V5, we stop before the filename.
  uint64_t ExpectedEnd = Prologue.TotalLength + Prologue.sizeofTotalLength() -
                         (Version < 5 ? 2 : 8);
  std::vector<std::string> Errs;
  Errs.emplace_back(
      (Twine("parsing line table prologue at 0x00000000 found an invalid "
             "directory or file table description at 0x000000") +
       Twine::utohexstr(ExpectedEnd))
          .str());
  if (Version < 5) {
    Errs.emplace_back("file names table was not null terminated before the end "
                      "of the prologue");
  } else {
    Errs.emplace_back(
        "failed to parse file entry because extracting the form value failed");
  }
  EXPECT_THAT_ERROR(std::move(Recoverable),
                    FailedWithMessageArray(testing::ElementsAreArray(Errs)));
}

INSTANTIATE_TEST_SUITE_P(
    LineTableTestParams, DebugLineParameterisedFixture,
    Values(std::make_pair(
               2, DWARF32), // Test lower-bound of v2-3 fields and DWARF32.
           std::make_pair(3, DWARF32), // Test upper-bound of v2-3 fields.
           std::make_pair(4, DWARF64), // Test v4 fields and DWARF64.
           std::make_pair(5, DWARF32), std::make_pair(5, DWARF64)));

TEST_F(DebugLineBasicFixture, ErrorForExtendedOpcodeLengthSmallerThanExpected) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.addByte(0xaa);
  // The Length should be 1 + sizeof(ULEB) for a set discriminator opcode.
  // The operand will be read for both the discriminator opcode and then parsed
  // again as DW_LNS_negate_stmt, to respect the claimed length.
  LT.addExtendedOpcode(1, DW_LNE_set_discriminator,
                       {{DW_LNS_negate_stmt, LineTable::ULEB}});
  LT.addByte(0xbb);
  LT.addStandardOpcode(DW_LNS_const_add_pc, {});
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  EXPECT_THAT_ERROR(std::move(Recoverable),
                    FailedWithMessage("unexpected line op length at offset "
                                      "0x00000031 expected 0x01 found 0x02"));
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  ASSERT_EQ((*ExpectedLineTable)->Rows.size(), 3u);
  EXPECT_EQ((*ExpectedLineTable)->Sequences.size(), 1u);
  EXPECT_EQ((*ExpectedLineTable)->Rows[1].IsStmt, 0u);
  EXPECT_EQ((*ExpectedLineTable)->Rows[1].Discriminator, DW_LNS_negate_stmt);
}

TEST_F(DebugLineBasicFixture, ErrorForExtendedOpcodeLengthLargerThanExpected) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.addByte(0xaa);
  LT.addStandardOpcode(DW_LNS_const_add_pc, {});
  // The Length should be 1 for an end sequence opcode.
  LT.addExtendedOpcode(2, DW_LNE_end_sequence, {});
  // The negate statement opcode will be skipped.
  LT.addStandardOpcode(DW_LNS_negate_stmt, {});
  LT.addByte(0xbb);
  LT.addStandardOpcode(DW_LNS_const_add_pc, {});
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  EXPECT_THAT_ERROR(std::move(Recoverable),
                    FailedWithMessage("unexpected line op length at offset "
                                      "0x00000032 expected 0x02 found 0x01"));
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  ASSERT_EQ((*ExpectedLineTable)->Rows.size(), 4u);
  EXPECT_EQ((*ExpectedLineTable)->Sequences.size(), 2u);
  ASSERT_EQ((*ExpectedLineTable)->Sequences[1].FirstRowIndex, 2u);
  EXPECT_EQ((*ExpectedLineTable)->Rows[2].IsStmt, 1u);
}

TEST_F(DebugLineBasicFixture, ErrorForUnitLengthTooLarge) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &Padding = Gen->addLineTable();
  // Add some padding to show that a non-zero offset is handled correctly.
  Padding.setCustomPrologue({{0, LineTable::Byte}});
  LineTable &LT = Gen->addLineTable();
  LT.addStandardOpcode(DW_LNS_copy, {});
  LT.addStandardOpcode(DW_LNS_const_add_pc, {});
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});
  DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();
  // Set the total length to 1 higher than the actual length.
  ++Prologue.TotalLength;
  LT.setPrologue(Prologue);

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 1, *Context,
                                                    nullptr, RecordRecoverable);
  EXPECT_THAT_ERROR(
      std::move(Recoverable),
      FailedWithMessage("line table program with offset 0x00000001 has length "
                        "0x00000034 but only 0x00000033 bytes are available"));
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  EXPECT_EQ((*ExpectedLineTable)->Rows.size(), 2u);
  EXPECT_EQ((*ExpectedLineTable)->Sequences.size(), 1u);
}

TEST_F(DebugLineBasicFixture, ErrorForMismatchedAddressSize) {
  if (!setupGenerator(4, 8))
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  // The line data extractor expects size 8 (Quad) addresses.
  uint64_t Addr1 = 0x11223344;
  LT.addExtendedOpcode(5, DW_LNE_set_address, {{Addr1, LineTable::Long}});
  LT.addStandardOpcode(DW_LNS_copy, {});
  // Show that the expected address size is unchanged, so later valid lines
  // don't cause a problem.
  uint64_t Addr2 = 0x1122334455667788;
  LT.addExtendedOpcode(9, DW_LNE_set_address, {{Addr2, LineTable::Quad}});
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  EXPECT_THAT_ERROR(std::move(Recoverable),
                    FailedWithMessage("mismatching address size at offset "
                                      "0x00000030 expected 0x08 found 0x04"));
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  ASSERT_EQ((*ExpectedLineTable)->Rows.size(), 2u);
  EXPECT_EQ((*ExpectedLineTable)->Sequences.size(), 1u);
  EXPECT_EQ((*ExpectedLineTable)->Rows[0].Address.Address, Addr1);
  EXPECT_EQ((*ExpectedLineTable)->Rows[1].Address.Address, Addr2);
}

TEST_F(DebugLineBasicFixture,
       ErrorForMismatchedAddressSizeUnsetInitialAddress) {
  if (!setupGenerator(4, 0))
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  uint64_t Addr1 = 0x11223344;
  LT.addExtendedOpcode(5, DW_LNE_set_address, {{Addr1, LineTable::Long}});
  LT.addStandardOpcode(DW_LNS_copy, {});
  uint64_t Addr2 = 0x1122334455667788;
  LT.addExtendedOpcode(9, DW_LNE_set_address, {{Addr2, LineTable::Quad}});
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  EXPECT_THAT_ERROR(std::move(Recoverable),
                    FailedWithMessage("mismatching address size at offset "
                                      "0x00000038 expected 0x04 found 0x08"));
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  ASSERT_EQ((*ExpectedLineTable)->Rows.size(), 2u);
  EXPECT_EQ((*ExpectedLineTable)->Sequences.size(), 1u);
  EXPECT_EQ((*ExpectedLineTable)->Rows[0].Address.Address, Addr1);
  EXPECT_EQ((*ExpectedLineTable)->Rows[1].Address.Address, Addr2);
}

TEST_F(DebugLineBasicFixture,
       ErrorForUnsupportedAddressSizeInSetAddressLength) {
  // Use DWARF v4, and 0 for data extractor address size so that the address
  // size is derived from the opcode length.
  if (!setupGenerator(4, 0))
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  // 4 == length of the extended opcode, i.e. 1 for the opcode itself and 3 for
  // the Half (2) + Byte (1) operand, representing the unsupported address size.
  LT.addExtendedOpcode(4, DW_LNE_set_address,
                       {{0x1234, LineTable::Half}, {0x56, LineTable::Byte}});
  LT.addStandardOpcode(DW_LNS_copy, {});
  // Special opcode to ensure the address has changed between the first and last
  // row in the sequence. Without this, the sequence will not be recorded.
  LT.addByte(0xaa);
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  EXPECT_THAT_ERROR(
      std::move(Recoverable),
      FailedWithMessage("address size 0x03 of DW_LNE_set_address opcode at "
                        "offset 0x00000030 is unsupported"));
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  ASSERT_EQ((*ExpectedLineTable)->Rows.size(), 3u);
  EXPECT_EQ((*ExpectedLineTable)->Sequences.size(), 1u);
  // Show that the set address opcode is ignored in this case.
  EXPECT_EQ((*ExpectedLineTable)->Rows[0].Address.Address, 0u);
}

TEST_F(DebugLineBasicFixture, ErrorForAddressSizeGreaterThanByteSize) {
  // Use DWARF v4, and 0 for data extractor address size so that the address
  // size is derived from the opcode length.
  if (!setupGenerator(4, 0))
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  // Specifically use an operand size that has a trailing byte of a supported
  // size (8), so that any potential truncation would result in a valid size.
  std::vector<LineTable::ValueAndLength> Operands(0x108);
  LT.addExtendedOpcode(Operands.size() + 1, DW_LNE_set_address, Operands);
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  EXPECT_THAT_ERROR(
      std::move(Recoverable),
      FailedWithMessage("address size 0x108 of DW_LNE_set_address opcode at "
                        "offset 0x00000031 is unsupported"));
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
}

#ifdef NO_SUPPORT_DEBUG_ADDR
TEST_F(DebugLineBasicFixture,
       DISABLED_ErrorForUnsupportedAddressSizeDefinedInHeader) {
#else
TEST_F(DebugLineBasicFixture, ErrorForUnsupportedAddressSizeDefinedInHeader) {
#endif
  // Use 0 for data extractor address size so that it does not clash with the
  // header address size.
  if (!setupGenerator(5, 0))
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  // AddressSize + 1 == length of the extended opcode, i.e. 1 for the opcode
  // itself and 9 for the Quad (8) + Byte (1) operand representing the
  // unsupported address size.
  uint8_t AddressSize = 9;
  LT.addExtendedOpcode(AddressSize + 1, DW_LNE_set_address,
                       {{0x12345678, LineTable::Quad}, {0, LineTable::Byte}});
  LT.addStandardOpcode(DW_LNS_copy, {});
  // Special opcode to ensure the address has changed between the first and last
  // row in the sequence. Without this, the sequence will not be recorded.
  LT.addByte(0xaa);
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});
  DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();
  Prologue.FormParams.AddrSize = AddressSize;
  LT.setPrologue(Prologue);

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  EXPECT_THAT_ERROR(
      std::move(Recoverable),
      FailedWithMessage("parsing line table prologue at offset 0x00000000: "
                        "invalid address size 9",
                        "address size 0x09 of DW_LNE_set_address opcode at "
                        "offset 0x00000038 is unsupported"));
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  ASSERT_EQ((*ExpectedLineTable)->Rows.size(), 3u);
  EXPECT_EQ((*ExpectedLineTable)->Sequences.size(), 1u);
  // Show that the set address opcode is ignored in this case.
  EXPECT_EQ((*ExpectedLineTable)->Rows[0].Address.Address, 0u);
}

TEST_F(DebugLineBasicFixture, CallbackUsedForUnterminatedSequence) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.addExtendedOpcode(9, DW_LNE_set_address,
                       {{0x1122334455667788, LineTable::Quad}});
  LT.addStandardOpcode(DW_LNS_copy, {});
  LT.addByte(0xaa);
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});
  LT.addExtendedOpcode(9, DW_LNE_set_address,
                       {{0x99aabbccddeeff00, LineTable::Quad}});
  LT.addStandardOpcode(DW_LNS_copy, {});
  LT.addByte(0xbb);
  LT.addByte(0xcc);

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  EXPECT_THAT_ERROR(std::move(Recoverable),
                    FailedWithMessage("last sequence in debug line table at "
                                      "offset 0x00000000 is not terminated"));
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  EXPECT_EQ((*ExpectedLineTable)->Rows.size(), 6u);
  // The unterminated sequence is not added to the sequence list.
  EXPECT_EQ((*ExpectedLineTable)->Sequences.size(), 1u);
}

struct AdjustAddressFixtureBase : public CommonFixture {
  virtual ~AdjustAddressFixtureBase() {}

  // Create and update the prologue as specified by the subclass, then return
  // the length of the table.
  virtual uint64_t editPrologue(LineTable &LT) = 0;

  virtual uint64_t getAdjustedAddr(uint64_t Base, uint64_t ConstIncrs,
                                   uint64_t SpecialIncrs,
                                   uint64_t AdvanceIncrs) {
    return Base + ConstIncrs + SpecialIncrs + AdvanceIncrs;
  }

  virtual uint64_t getAdjustedLine(uint64_t Base, uint64_t Incr) {
    return Base + Incr;
  }

  uint64_t setupNoProblemTable() {
    LineTable &NoProblem = Gen->addLineTable();
    NoProblem.addExtendedOpcode(9, DW_LNE_set_address,
                                {{0xabcd, LineTable::Quad}});
    NoProblem.addExtendedOpcode(1, DW_LNE_end_sequence, {});
    return editPrologue(NoProblem);
  }

  uint64_t setupConstAddPcFirstTable() {
    LineTable &ConstAddPCFirst = Gen->addLineTable();
    ConstAddPCFirst.addExtendedOpcode(9, DW_LNE_set_address,
                                      {{ConstAddPCAddr, LineTable::Quad}});
    ConstAddPCFirst.addStandardOpcode(DW_LNS_const_add_pc, {});
    ConstAddPCFirst.addStandardOpcode(DW_LNS_const_add_pc, {});
    ConstAddPCFirst.addStandardOpcode(DW_LNS_advance_pc,
                                      {{0x10, LineTable::ULEB}});
    ConstAddPCFirst.addByte(0x21); // Special opcode, +1 op, +1 line.
    ConstAddPCFirst.addExtendedOpcode(1, DW_LNE_end_sequence, {});
    return editPrologue(ConstAddPCFirst);
  }

  uint64_t setupSpecialFirstTable() {
    LineTable &SpecialFirst = Gen->addLineTable();
    SpecialFirst.addExtendedOpcode(9, DW_LNE_set_address,
                                   {{SpecialAddr, LineTable::Quad}});
    SpecialFirst.addByte(0x22); // Special opcode, +1 op, +2 line.
    SpecialFirst.addStandardOpcode(DW_LNS_const_add_pc, {});
    SpecialFirst.addStandardOpcode(DW_LNS_advance_pc,
                                   {{0x20, LineTable::ULEB}});
    SpecialFirst.addByte(0x23); // Special opcode, +1 op, +3 line.
    SpecialFirst.addExtendedOpcode(1, DW_LNE_end_sequence, {});
    return editPrologue(SpecialFirst);
  }

  uint64_t setupAdvancePcFirstTable() {
    LineTable &AdvancePCFirst = Gen->addLineTable();
    AdvancePCFirst.addExtendedOpcode(9, DW_LNE_set_address,
                                     {{AdvancePCAddr, LineTable::Quad}});
    AdvancePCFirst.addStandardOpcode(DW_LNS_advance_pc,
                                     {{0x30, LineTable::ULEB}});
    AdvancePCFirst.addStandardOpcode(DW_LNS_const_add_pc, {});
    AdvancePCFirst.addStandardOpcode(DW_LNS_advance_pc,
                                     {{0x40, LineTable::ULEB}});
    AdvancePCFirst.addByte(0x24); // Special opcode, +1 op, +4 line.
    AdvancePCFirst.addExtendedOpcode(1, DW_LNE_end_sequence, {});
    return editPrologue(AdvancePCFirst);
  }

  void setupTables(bool AddAdvancePCFirstTable) {
    LineTable &Padding = Gen->addLineTable();
    Padding.setCustomPrologue({{0, LineTable::Byte}});
    NoProblemOffset = 1;

    // Show that no warning is generated for the case where no
    // DW_LNS_const_add_pc or special opcode is used.
    ConstAddPCOffset = setupNoProblemTable() + NoProblemOffset;

    // Show that the warning is emitted for the first DW_LNS_const_add_pc opcode
    // and then not again.
    SpecialOffset = setupConstAddPcFirstTable() + ConstAddPCOffset;

    // Show that the warning is emitted for the first special opcode and then
    // not again.
    AdvancePCOffset = setupSpecialFirstTable() + SpecialOffset;

    // Show that the warning is emitted for the first DW_LNS_advance_pc opcode
    // (if requested) and then not again.
    if (AddAdvancePCFirstTable)
      setupAdvancePcFirstTable();
  }

  Expected<const DWARFDebugLine::LineTable *>
  checkTable(uint64_t Offset, StringRef OpcodeType, const Twine &MsgSuffix) {
    auto ExpectedTable = Line.getOrParseLineTable(LineData, Offset, *Context,
                                                  nullptr, RecordRecoverable);
    EXPECT_THAT_ERROR(std::move(Unrecoverable), Succeeded());
    if (!IsErrorExpected) {
      EXPECT_THAT_ERROR(std::move(Recoverable), Succeeded());
    } else {
      if (!ExpectedTable)
        return ExpectedTable;
      uint64_t ExpectedOffset = Offset +
                                (*ExpectedTable)->Prologue.getLength() +
                                11; // 11 == size of DW_LNE_set_address.
      std::string OffsetHex = Twine::utohexstr(Offset).str();
      std::string OffsetZeroes = std::string(8 - OffsetHex.size(), '0');
      std::string ExpectedHex = Twine::utohexstr(ExpectedOffset).str();
      std::string ExpectedZeroes = std::string(8 - ExpectedHex.size(), '0');
      EXPECT_THAT_ERROR(
          std::move(Recoverable),
          FailedWithMessage(("line table program at offset 0x" + OffsetZeroes +
                             OffsetHex + " contains a " + OpcodeType +
                             " opcode at offset 0x" + ExpectedZeroes +
                             ExpectedHex + ", " + MsgSuffix)
                                .str()));
    }
    return ExpectedTable;
  }

  void runTest(bool CheckAdvancePC, Twine MsgSuffix) {
    if (!setupGenerator(Version))
      GTEST_SKIP();

    setupTables(/*AddAdvancePCFirstTable=*/CheckAdvancePC);

    generate();

    auto ExpectedNoProblem = Line.getOrParseLineTable(
        LineData, NoProblemOffset, *Context, nullptr, RecordRecoverable);
    EXPECT_THAT_ERROR(std::move(Recoverable), Succeeded());
    EXPECT_THAT_ERROR(std::move(Unrecoverable), Succeeded());
    ASSERT_THAT_EXPECTED(ExpectedNoProblem, Succeeded());

    auto ExpectedConstAddPC =
        checkTable(ConstAddPCOffset, "DW_LNS_const_add_pc", MsgSuffix);
    ASSERT_THAT_EXPECTED(ExpectedConstAddPC, Succeeded());
    ASSERT_EQ((*ExpectedConstAddPC)->Rows.size(), 2u);
    EXPECT_EQ((*ExpectedConstAddPC)->Rows[0].Address.Address,
              getAdjustedAddr(ConstAddPCAddr, ConstIncr * 2, 0x1, 0x10));
    EXPECT_EQ((*ExpectedConstAddPC)->Rows[0].Line, getAdjustedLine(1, 1));
    EXPECT_THAT_ERROR(std::move(Unrecoverable), Succeeded());

    auto ExpectedSpecial = checkTable(SpecialOffset, "special", MsgSuffix);
    ASSERT_THAT_EXPECTED(ExpectedSpecial, Succeeded());
    ASSERT_EQ((*ExpectedSpecial)->Rows.size(), 3u);
    EXPECT_EQ((*ExpectedSpecial)->Rows[0].Address.Address,
              getAdjustedAddr(SpecialAddr, 0, 1, 0));
    EXPECT_EQ((*ExpectedSpecial)->Rows[0].Line, getAdjustedLine(1, 2));
    EXPECT_EQ((*ExpectedSpecial)->Rows[1].Address.Address,
              getAdjustedAddr(SpecialAddr, ConstIncr, 0x2, 0x20));
    EXPECT_EQ((*ExpectedSpecial)->Rows[1].Line, getAdjustedLine(1, 5));
    EXPECT_THAT_ERROR(std::move(Unrecoverable), Succeeded());

    if (!CheckAdvancePC)
      return;

    auto ExpectedAdvancePC =
        checkTable(AdvancePCOffset, "DW_LNS_advance_pc", MsgSuffix);
    ASSERT_THAT_EXPECTED(ExpectedAdvancePC, Succeeded());
    ASSERT_EQ((*ExpectedAdvancePC)->Rows.size(), 2u);
    EXPECT_EQ((*ExpectedAdvancePC)->Rows[0].Address.Address,
              getAdjustedAddr(AdvancePCAddr, ConstIncr, 0x1, 0x70));
    EXPECT_EQ((*ExpectedAdvancePC)->Rows[0].Line, getAdjustedLine(1, 4));
  }

  uint64_t ConstIncr = 0x11;
  uint64_t ConstAddPCAddr = 0x1234;
  uint64_t SpecialAddr = 0x5678;
  uint64_t AdvancePCAddr = 0xabcd;
  uint64_t NoProblemOffset;
  uint64_t ConstAddPCOffset;
  uint64_t SpecialOffset;
  uint64_t AdvancePCOffset;

  uint16_t Version = 4;
  bool IsErrorExpected;
};

struct OpIndexFixture : Test, CommonFixture {
  void createPrologue(LineTable &LT, uint8_t MaxOpsPerInst,
                      uint8_t MinInstLength) {
    DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();
    Prologue.MaxOpsPerInst = MaxOpsPerInst;
    Prologue.MinInstLength = MinInstLength;
    LT.setPrologue(Prologue);
  }
};

TEST_F(OpIndexFixture, OpIndexAdvance) {
  if (!setupGenerator(4, 4))
    GTEST_SKIP();

  uint8_t MaxOpsPerInst = 13;
  uint8_t MinInstLength = 4;

  LineTable &LT = Gen->addLineTable();

  // Row 0-2: Different locations for one bundle set up using special opcodes.
  LT.addExtendedOpcode(5, DW_LNE_set_address, {{0x20, LineTable::Long}});
  LT.addByte(0x13); // Special opcode, +1 line.
  LT.addByte(0x23); // Special opcode, +3 line, +1 op-index.
  LT.addByte(0x3a); // Special opcode, -2 line, +3 op-index.

  // Row 3: New bundle, set up using DW_LNS_advance pc.
  // Operation advance 0x84, which gives +40 addr, +2 op-index
  LT.addStandardOpcode(DW_LNS_advance_line, {{100, LineTable::SLEB}});
  LT.addStandardOpcode(DW_LNS_advance_pc, {{0x84, LineTable::ULEB}});
  LT.addStandardOpcode(DW_LNS_copy, {}); // Create new row.

  // Row 4: New bundle, set up using a single special opcode.
  LT.addByte(0x71); // Special opcode, +4 addr, -3 line, -6 op-index.

  // Row 5: New bundle, set up using using DW_LNS_const_add_pc.
  // Corresponds to advancing address and op-index using special opcode 255,
  // which gives +4 addr, +4 op-index.
  LT.addStandardOpcode(DW_LNS_advance_line, {{10, LineTable::SLEB}});
  LT.addStandardOpcode(DW_LNS_const_add_pc, {});
  LT.addStandardOpcode(DW_LNS_copy, {}); // Create new row.

  // Row 6: End sequence to have the input well-formed.
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  createPrologue(LT, MaxOpsPerInst, MinInstLength);

  generate();

  auto VerifyRow = [](const DWARFDebugLine::Row &Row, uint64_t Address,
                      uint8_t OpIndex, uint32_t Line) {
    EXPECT_EQ(Row.Address.Address, Address);
    EXPECT_EQ(Row.OpIndex, OpIndex);
    EXPECT_EQ(Row.Line, Line);
  };

  auto Table = Line.getOrParseLineTable(LineData, 0, *Context, nullptr,
                                        RecordRecoverable);
  EXPECT_THAT_ERROR(
      std::move(Recoverable),
      FailedWithMessage("line table program at offset 0x00000000 contains a "
                        "special opcode at offset 0x00000035, but the prologue "
                        "maximum_operations_per_instruction value is 13, which "
                        "is experimentally supported, so line number "
                        "information may be incorrect"));
  EXPECT_THAT_ERROR(std::move(Unrecoverable), Succeeded());
  ASSERT_THAT_EXPECTED(Table, Succeeded());

  ASSERT_EQ((*Table)->Rows.size(), 7u);

  VerifyRow((*Table)->Rows[0], 0x20, 0, 2);
  VerifyRow((*Table)->Rows[1], 0x20, 1, 5);
  VerifyRow((*Table)->Rows[2], 0x20, 4, 3);
  VerifyRow((*Table)->Rows[3], 0x48, 6, 103);
  VerifyRow((*Table)->Rows[4], 0x4c, 0, 100);
  VerifyRow((*Table)->Rows[5], 0x50, 4, 110);
}

TEST_F(OpIndexFixture, OpIndexReset) {
  if (!setupGenerator(4, 4))
    GTEST_SKIP();

  uint8_t MaxOpsPerInst = 13;
  uint8_t MinInstLength = 4;

  LineTable &LT = Gen->addLineTable();

  // Row 0: Just set op-index to some value > 0.
  LT.addExtendedOpcode(5, DW_LNE_set_address, {{0, LineTable::Long}});
  LT.addByte(0x20); // Special opcode, +1 op-index

  // Row 1: DW_LNE_fixed_advance_pc should set op-index to 0.
  LT.addStandardOpcode(DW_LNS_fixed_advance_pc, {{10, LineTable::Half}});
  LT.addStandardOpcode(DW_LNS_copy, {}); // Create new row.

  // Row 2: Just set op-index to some value > 0.
  LT.addByte(0x66); // Special opcode, +6 op-index

  // Row 3: DW_LNE_set_address should set op-index to 0.
  LT.addExtendedOpcode(5, DW_LNE_set_address, {{20, LineTable::Long}});
  LT.addStandardOpcode(DW_LNS_copy, {}); // Create new row.

  // Row 4: Just set op-index to some value > 0.
  LT.addByte(0xba); // Special opcode, +12 op-index

  // Row 5: End sequence (op-index unchanged for this row)...
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  // Row 6: ... but shall be reset after the DW_LNE_end_sequence row.
  LT.addStandardOpcode(DW_LNS_copy, {}); // Create new row.

  // Row 7: End sequence to have the input well-formed.
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  createPrologue(LT, MaxOpsPerInst, MinInstLength);

  generate();

  auto Table = Line.getOrParseLineTable(LineData, 0, *Context, nullptr,
                                        RecordRecoverable);
  EXPECT_THAT_ERROR(
      std::move(Recoverable),
      FailedWithMessage("line table program at offset 0x00000000 contains a "
                        "special opcode at offset 0x00000035, but the prologue "
                        "maximum_operations_per_instruction value is 13, which "
                        "is experimentally supported, so line number "
                        "information may be incorrect"));
  EXPECT_THAT_ERROR(std::move(Unrecoverable), Succeeded());
  ASSERT_THAT_EXPECTED(Table, Succeeded());

  ASSERT_EQ((*Table)->Rows.size(), 8u);
  EXPECT_EQ((*Table)->Rows[0].OpIndex, 1u);
  EXPECT_EQ((*Table)->Rows[1].OpIndex, 0u); // DW_LNS_fixed_advance_pc.
  EXPECT_EQ((*Table)->Rows[2].OpIndex, 6u);
  EXPECT_EQ((*Table)->Rows[3].OpIndex, 0u); // DW_LNE_set_address.
  EXPECT_EQ((*Table)->Rows[4].OpIndex, 12u);
  EXPECT_EQ((*Table)->Rows[5].OpIndex, 12u);
  EXPECT_EQ((*Table)->Rows[6].OpIndex, 0u); // row after DW_LNE_end_sequence.
  EXPECT_EQ((*Table)->Rows[7].OpIndex, 0u);
}

TEST_F(OpIndexFixture, MaxOpsZeroDwarf3) {
  if (!setupGenerator(3, 4))
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.addStandardOpcode(DW_LNS_const_add_pc, {}); // Just some opcode to advance.
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {}); //
  createPrologue(LT, /*MaxOpsPerInst=*/0, /*MinInstLength=*/1);
  generate();

  auto Table = Line.getOrParseLineTable(LineData, 0, *Context, nullptr,
                                        RecordRecoverable);
  EXPECT_THAT_ERROR(std::move(Recoverable), Succeeded());
  EXPECT_THAT_ERROR(std::move(Unrecoverable), Succeeded());
  ASSERT_THAT_EXPECTED(Table, Succeeded());
}

TEST_F(OpIndexFixture, MaxOpsZeroDwarf4) {
  if (!setupGenerator(4, 4))
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.addStandardOpcode(DW_LNS_const_add_pc, {}); // Just some opcode to advance.
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {}); //
  createPrologue(LT, /*MaxOpsPerInst=*/0, /*MinInstLength=*/1);
  generate();

  auto Table = Line.getOrParseLineTable(LineData, 0, *Context, nullptr,
                                        RecordRecoverable);
  EXPECT_THAT_ERROR(
      std::move(Recoverable),
      FailedWithMessage(
          "line table program at offset 0x00000000 contains a "
          "DW_LNS_const_add_pc opcode at offset 0x0000002e, but "
          "the prologue maximum_operations_per_instruction value "
          "is 0, which is invalid. Assuming a value of 1 instead"));
  EXPECT_THAT_ERROR(std::move(Unrecoverable), Succeeded());
  ASSERT_THAT_EXPECTED(Table, Succeeded());
}

struct LineRangeFixture : TestWithParam<std::tuple<uint8_t, bool>>,
                          AdjustAddressFixtureBase {
  void SetUp() override { std::tie(LineRange, IsErrorExpected) = GetParam(); }

  uint64_t editPrologue(LineTable &LT) override {
    DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();
    Prologue.LineRange = LineRange;
    LT.setPrologue(Prologue);
    return Prologue.TotalLength + Prologue.sizeofTotalLength();
  }

  uint64_t getAdjustedAddr(uint64_t Base, uint64_t ConstIncr,
                           uint64_t SpecialIncr,
                           uint64_t AdvanceIncr) override {
    if (LineRange == 0)
      return Base + AdvanceIncr;
    return AdjustAddressFixtureBase::getAdjustedAddr(Base, ConstIncr,
                                                     SpecialIncr, AdvanceIncr);
  }

  uint64_t getAdjustedLine(uint64_t Base, uint64_t Incr) override {
    return LineRange != 0
               ? AdjustAddressFixtureBase::getAdjustedLine(Base, Incr)
               : Base;
  }

  uint8_t LineRange;
};

TEST_P(LineRangeFixture, LineRangeProblemsReportedCorrectly) {
  runTest(/*CheckAdvancePC=*/false,
          "but the prologue line_range value is 0. The address and line will "
          "not be adjusted");
}

INSTANTIATE_TEST_SUITE_P(
    LineRangeParams, LineRangeFixture,
    Values(std::make_tuple(0, true),     // Test zero value (error).
           std::make_tuple(14, false))); // Test non-zero value (no error).

struct BadMinInstLenFixture : TestWithParam<std::tuple<uint8_t, bool>>,
                              AdjustAddressFixtureBase {
  void SetUp() override {
    std::tie(MinInstLength, IsErrorExpected) = GetParam();
  }

  uint64_t editPrologue(LineTable &LT) override {
    DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();
    Prologue.MinInstLength = MinInstLength;
    LT.setPrologue(Prologue);
    return Prologue.TotalLength + Prologue.sizeofTotalLength();
  }

  uint64_t getAdjustedAddr(uint64_t Base, uint64_t ConstIncr,
                           uint64_t SpecialIncr,
                           uint64_t AdvanceIncr) override {
    return MinInstLength != 0 ? AdjustAddressFixtureBase::getAdjustedAddr(
                                    Base, ConstIncr, SpecialIncr, AdvanceIncr)
                              : Base;
  }

  uint8_t MinInstLength;
};

TEST_P(BadMinInstLenFixture, MinInstLengthProblemsReportedCorrectly) {
  runTest(/*CheckAdvancePC=*/true,
          "but the prologue minimum_instruction_length value is 0, which "
          "prevents any address advancing");
}

INSTANTIATE_TEST_SUITE_P(
    BadMinInstLenParams, BadMinInstLenFixture,
    Values(std::make_tuple(0, true),    // Test zero value (error).
           std::make_tuple(1, false))); // Test non-zero value (no error).

TEST_F(DebugLineBasicFixture, ParserParsesCorrectly) {
  if (!setupGenerator())
    GTEST_SKIP();

  DWARFDebugLine::SectionParser Parser = setupParser();

  EXPECT_EQ(Parser.getOffset(), 0u);
  ASSERT_FALSE(Parser.done());

  DWARFDebugLine::LineTable Parsed =
      Parser.parseNext(RecordRecoverable, RecordUnrecoverable);
  checkDefaultPrologue(4, DWARF32, Parsed.Prologue, 16);
  EXPECT_EQ(Parsed.Sequences.size(), 1u);
  EXPECT_EQ(Parser.getOffset(), 62u);
  ASSERT_FALSE(Parser.done());

  DWARFDebugLine::LineTable Parsed2 =
      Parser.parseNext(RecordRecoverable, RecordUnrecoverable);
  checkDefaultPrologue(4, DWARF64, Parsed2.Prologue, 16);
  EXPECT_EQ(Parsed2.Sequences.size(), 1u);
  EXPECT_EQ(Parser.getOffset(), 136u);
  EXPECT_TRUE(Parser.done());

  EXPECT_FALSE(Recoverable);
  EXPECT_FALSE(Unrecoverable);
}

TEST_F(DebugLineBasicFixture, ParserSkipsCorrectly) {
  if (!setupGenerator())
    GTEST_SKIP();

  DWARFDebugLine::SectionParser Parser = setupParser();

  EXPECT_EQ(Parser.getOffset(), 0u);
  ASSERT_FALSE(Parser.done());

  Parser.skip(RecordRecoverable, RecordUnrecoverable);
  EXPECT_EQ(Parser.getOffset(), 62u);
  ASSERT_FALSE(Parser.done());

  Parser.skip(RecordRecoverable, RecordUnrecoverable);
  EXPECT_EQ(Parser.getOffset(), 136u);
  EXPECT_TRUE(Parser.done());

  EXPECT_FALSE(Recoverable);
  EXPECT_FALSE(Unrecoverable);
}

TEST_F(DebugLineBasicFixture, ParserAlwaysDoneForEmptySection) {
  if (!setupGenerator())
    GTEST_SKIP();

  generate();
  DWARFDebugLine::SectionParser Parser(LineData, *Context, Units);

  EXPECT_TRUE(Parser.done());
}

TEST_F(DebugLineBasicFixture, ParserMarkedAsDoneForBadLengthWhenParsing) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue({{0xfffffff0, LineTable::Long}});
  Gen->addLineTable();
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, Units);
  Parser.parseNext(RecordRecoverable, RecordUnrecoverable);

  EXPECT_EQ(Parser.getOffset(), 0u);
  EXPECT_TRUE(Parser.done());
  EXPECT_FALSE(Recoverable);

  EXPECT_THAT_ERROR(
      std::move(Unrecoverable),
      FailedWithMessage(
          "parsing line table prologue at offset 0x00000000: unsupported "
          "reserved unit length of value 0xfffffff0"));
}

TEST_F(DebugLineBasicFixture, ParserMarkedAsDoneForBadLengthWhenSkipping) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue({{0xfffffff0, LineTable::Long}});
  Gen->addLineTable();
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, Units);
  Parser.skip(RecordRecoverable, RecordUnrecoverable);

  EXPECT_EQ(Parser.getOffset(), 0u);
  EXPECT_TRUE(Parser.done());
  EXPECT_FALSE(Recoverable);

  EXPECT_THAT_ERROR(
      std::move(Unrecoverable),
      FailedWithMessage(
          "parsing line table prologue at offset 0x00000000: unsupported "
          "reserved unit length of value 0xfffffff0"));
}

TEST_F(DebugLineBasicFixture, ParserReportsFirstErrorInEachTableWhenParsing) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable(DWARF32);
  LT.setCustomPrologue({{2, LineTable::Long}, {0, LineTable::Half}});
  LineTable &LT2 = Gen->addLineTable(DWARF32);
  LT2.setCustomPrologue({{2, LineTable::Long}, {1, LineTable::Half}});
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, Units);
  Parser.parseNext(RecordRecoverable, RecordUnrecoverable);
  ASSERT_FALSE(Parser.done());
  Parser.parseNext(RecordRecoverable, RecordUnrecoverable);

  EXPECT_TRUE(Parser.done());
  EXPECT_THAT_ERROR(std::move(Recoverable), Succeeded());

  EXPECT_THAT_ERROR(
      std::move(Unrecoverable),
      FailedWithMessage("parsing line table prologue at offset 0x00000000: "
                        "unsupported version 0",
                        "parsing line table prologue at offset 0x00000006: "
                        "unsupported version 1"));
}

TEST_F(DebugLineBasicFixture, ParserReportsNonPrologueProblemsWhenParsing) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable(DWARF32);
  LT.addExtendedOpcode(0x42, DW_LNE_end_sequence, {});
  LineTable &LT2 = Gen->addLineTable(DWARF32);
  LT2.addExtendedOpcode(9, DW_LNE_set_address,
                        {{0x1234567890abcdef, LineTable::Quad}});
  LT2.addStandardOpcode(DW_LNS_copy, {});
  LT2.addByte(0xbb);
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, Units);
  Parser.parseNext(RecordRecoverable, RecordUnrecoverable);
  EXPECT_FALSE(Unrecoverable);
  ASSERT_FALSE(Parser.done());
  EXPECT_THAT_ERROR(std::move(Recoverable),
                    FailedWithMessage("unexpected line op length at offset "
                                      "0x00000030 expected 0x42 found 0x01"));

  // Reset the error state so that it does not confuse the next set of checks.
  Unrecoverable = Error::success();
  Parser.parseNext(RecordRecoverable, RecordUnrecoverable);

  EXPECT_TRUE(Parser.done());
  EXPECT_THAT_ERROR(std::move(Recoverable),
                    FailedWithMessage("last sequence in debug line table at "
                                      "offset 0x00000031 is not terminated"));
  EXPECT_FALSE(Unrecoverable);
}

TEST_F(DebugLineBasicFixture,
       ParserReportsPrologueErrorsInEachTableWhenSkipping) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable(DWARF32);
  LT.setCustomPrologue({{2, LineTable::Long}, {0, LineTable::Half}});
  LineTable &LT2 = Gen->addLineTable(DWARF32);
  LT2.setCustomPrologue({{2, LineTable::Long}, {1, LineTable::Half}});
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, Units);
  Parser.skip(RecordRecoverable, RecordUnrecoverable);
  ASSERT_FALSE(Parser.done());
  Parser.skip(RecordRecoverable, RecordUnrecoverable);

  EXPECT_TRUE(Parser.done());
  EXPECT_FALSE(Recoverable);

  EXPECT_THAT_ERROR(
      std::move(Unrecoverable),
      FailedWithMessage("parsing line table prologue at offset 0x00000000: "
                        "unsupported version 0",
                        "parsing line table prologue at offset 0x00000006: "
                        "unsupported version 1"));
}

TEST_F(DebugLineBasicFixture, ParserIgnoresNonPrologueErrorsWhenSkipping) {
  if (!setupGenerator())
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable(DWARF32);
  LT.addExtendedOpcode(42, DW_LNE_end_sequence, {});
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, Units);
  Parser.skip(RecordRecoverable, RecordUnrecoverable);

  EXPECT_TRUE(Parser.done());
  EXPECT_FALSE(Recoverable);
  EXPECT_FALSE(Unrecoverable);
}

#ifdef NO_SUPPORT_DEBUG_ADDR
TEST_F(DebugLineBasicFixture, DISABLED_VerboseOutput) {
#else
TEST_F(DebugLineBasicFixture, VerboseOutput) {
#endif
  if (!setupGenerator(5))
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  LT.addByte(0); // Extended opcode with zero length.
  LT.addByte(0);
  // Zero-value extended opcode.
  LT.addExtendedOpcode(2, 0, {{1, LineTable::Byte}});
  // Unknown extended opcode.
  LT.addExtendedOpcode(2, 0x42, {{1, LineTable::Byte}});
  LT.addExtendedOpcode(9, DW_LNE_set_address,
                       {{0x123456789abcdef, LineTable::Quad}});
  LT.addExtendedOpcode(6, DW_LNE_define_file,
                       {{'a', LineTable::Byte},
                        {'\0', LineTable::Byte},
                        {2, LineTable::ULEB},
                        {3, LineTable::ULEB},
                        {4, LineTable::ULEB}});
  LT.addExtendedOpcode(2, DW_LNE_set_discriminator, {{0x7f, LineTable::ULEB}});
  LT.addStandardOpcode(DW_LNS_copy, {});
  LT.addStandardOpcode(DW_LNS_advance_pc, {{11, LineTable::ULEB}});
  LT.addStandardOpcode(DW_LNS_advance_line, {{22, LineTable::SLEB}});
  LT.addStandardOpcode(DW_LNS_set_file, {{33, LineTable::ULEB}});
  LT.addStandardOpcode(DW_LNS_set_column, {{44, LineTable::ULEB}});
  LT.addStandardOpcode(DW_LNS_negate_stmt, {});
  LT.addStandardOpcode(DW_LNS_set_basic_block, {});
  LT.addStandardOpcode(DW_LNS_const_add_pc, {});
  LT.addStandardOpcode(DW_LNS_fixed_advance_pc, {{55, LineTable::Half}});
  LT.addStandardOpcode(DW_LNS_set_prologue_end, {});
  LT.addStandardOpcode(DW_LNS_set_epilogue_begin, {});
  LT.addStandardOpcode(DW_LNS_set_isa, {{66, LineTable::ULEB}});
  // Add unknown standard opcode with operands.
  LT.addStandardOpcode(
      0xd, {{1, LineTable::ULEB}, {0x123456789abcdef, LineTable::ULEB}});
  // Add unknown standard opcode without operands.
  LT.addStandardOpcode(0xe, {});
  LT.addByte(0xff); // Special opcode.
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});

  // Adjust the prologue to account for the extra standard opcode.
  DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();
  Prologue.TotalLength += 2;
  Prologue.PrologueLength += 2;
  Prologue.OpcodeBase += 2;
  Prologue.StandardOpcodeLengths.push_back(2);
  Prologue.StandardOpcodeLengths.push_back(0);
  LT.setPrologue(Prologue);

  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, Units);
  std::string Output;
  raw_string_ostream OS(Output);
  Parser.parseNext(RecordRecoverable, RecordUnrecoverable, &OS,
                   /*Verbose=*/true);
  OS.flush();
  StringRef OutputRef(Output);

  size_t Pos = 0;
  auto NextLine = [&Pos, &OutputRef]() {
    size_t EOL = OutputRef.find_first_of('\n', Pos);
    StringRef Line = OutputRef.substr(Pos, EOL - Pos);
    Pos = EOL + 1;
    return Line;
  };
  EXPECT_EQ(NextLine(), "Line table prologue:");
  EXPECT_EQ(NextLine(), "    total_length: 0x00000078");
  EXPECT_EQ(NextLine(), "          format: DWARF32");
  EXPECT_EQ(NextLine(), "         version: 5");
  EXPECT_EQ(NextLine(), "    address_size: 8");
  EXPECT_EQ(NextLine(), " seg_select_size: 0");
  EXPECT_EQ(NextLine(), " prologue_length: 0x0000002c");
  EXPECT_EQ(NextLine(), " min_inst_length: 1");
  EXPECT_EQ(NextLine(), "max_ops_per_inst: 1");
  EXPECT_EQ(NextLine(), " default_is_stmt: 1");
  EXPECT_EQ(NextLine(), "       line_base: -5");
  EXPECT_EQ(NextLine(), "      line_range: 14");
  EXPECT_EQ(NextLine(), "     opcode_base: 15");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_copy] = 0");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_advance_pc] = 1");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_advance_line] = 1");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_set_file] = 1");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_set_column] = 1");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_negate_stmt] = 0");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_set_basic_block] = 0");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_const_add_pc] = 0");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_fixed_advance_pc] = 1");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_set_prologue_end] = 0");
  EXPECT_EQ(NextLine(),
            "standard_opcode_lengths[DW_LNS_set_epilogue_begin] = 0");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_set_isa] = 1");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_unknown_d] = 2");
  EXPECT_EQ(NextLine(), "standard_opcode_lengths[DW_LNS_unknown_e] = 0");
  EXPECT_EQ(NextLine(), "include_directories[  0] = \"a dir\"");
  EXPECT_EQ(NextLine(), "file_names[  0]:");
  EXPECT_EQ(NextLine(), "           name: \"a file\"");
  EXPECT_EQ(NextLine(), "      dir_index: 0");
  EXPECT_EQ(NextLine(), "");
  EXPECT_EQ(NextLine(), "            Address            Line   Column File   "
                        "ISA Discriminator OpIndex Flags");
  EXPECT_EQ(NextLine(), "            ------------------ ------ ------ ------ "
                        "--- ------------- ------- -------------");
  EXPECT_EQ(NextLine(),
            "0x00000038: 00 Badly formed extended line op (length 0)");
  EXPECT_EQ(NextLine(),
            "0x0000003a: 00 Unrecognized extended op 0x00 length 2");
  EXPECT_EQ(NextLine(),
            "0x0000003e: 00 Unrecognized extended op 0x42 length 2");
  EXPECT_EQ(NextLine(),
            "0x00000042: 00 DW_LNE_set_address (0x0123456789abcdef)");
  EXPECT_EQ(NextLine(), "0x0000004d: 00 DW_LNE_define_file (a, dir=2, "
                        "mod_time=(0x0000000000000003), length=4)");
  EXPECT_EQ(NextLine(), "0x00000055: 00 DW_LNE_set_discriminator (127)");
  EXPECT_EQ(NextLine(), "0x00000059: 01 DW_LNS_copy");
  EXPECT_EQ(NextLine(), "            0x0123456789abcdef      1      0      1   "
                        "0           127       0  is_stmt");
  EXPECT_EQ(NextLine(), "0x0000005a: 02 DW_LNS_advance_pc (addr += 11, "
                        "op-index += 0)");
  EXPECT_EQ(NextLine(), "0x0000005c: 03 DW_LNS_advance_line (23)");
  EXPECT_EQ(NextLine(), "0x0000005e: 04 DW_LNS_set_file (33)");
  EXPECT_EQ(NextLine(), "0x00000060: 05 DW_LNS_set_column (44)");
  EXPECT_EQ(NextLine(), "0x00000062: 06 DW_LNS_negate_stmt");
  EXPECT_EQ(NextLine(), "0x00000063: 07 DW_LNS_set_basic_block");
  EXPECT_EQ(NextLine(),
            "0x00000064: 08 DW_LNS_const_add_pc (addr += 0x0000000000000011, "
            "op-index += 0)");
  EXPECT_EQ(NextLine(), "0x00000065: 09 DW_LNS_fixed_advance_pc (addr += 0x0037"
            ", op-index = 0)");
  EXPECT_EQ(NextLine(), "0x00000068: 0a DW_LNS_set_prologue_end");
  EXPECT_EQ(NextLine(), "0x00000069: 0b DW_LNS_set_epilogue_begin");
  EXPECT_EQ(NextLine(), "0x0000006a: 0c DW_LNS_set_isa (66)");
  EXPECT_EQ(NextLine(), "0x0000006c: 0d Unrecognized standard opcode "
                        "(operands: 0x0000000000000001, 0x0123456789abcdef)");
  EXPECT_EQ(NextLine(), "0x00000077: 0e Unrecognized standard opcode");
  EXPECT_EQ(NextLine(), "0x00000078: ff address += 17,  line += -3,  "
                        "op-index += 0");
  EXPECT_EQ(NextLine(),
            "            0x0123456789abce53     20     44     33  66           "
            "  0       0  basic_block prologue_end epilogue_begin");
  EXPECT_EQ(NextLine(), "0x00000079: 00 DW_LNE_end_sequence");
  EXPECT_EQ(NextLine(), "            0x0123456789abce53     20     44     33  "
                        "66             0       0  end_sequence");
  EXPECT_EQ(NextLine(), "");
  EXPECT_EQ(Output.size(), Pos);
}

struct TruncatedPrologueFixture
    : public TestWithParam<
          std::tuple<uint64_t, uint64_t, uint16_t, DwarfFormat, StringRef>>,
      public CommonFixture {
  void SetUp() override {
    std::tie(Length, ExpectedOffset, Version, Format, ExpectedErr) = GetParam();
  }

  uint64_t Length;
  uint64_t ExpectedOffset;
  uint16_t Version;
  DwarfFormat Format;
  StringRef ExpectedErr;
};

#ifdef NO_SUPPORT_DEBUG_ADDR
TEST_P(TruncatedPrologueFixture, DISABLED_ErrorForTruncatedPrologue) {
#else
TEST_P(TruncatedPrologueFixture, ErrorForTruncatedPrologue) {
#endif
  if (!setupGenerator(Version))
    GTEST_SKIP();

  LineTable &Padding = Gen->addLineTable();
  // Add some padding to show that a non-zero offset is handled correctly.
  Padding.setCustomPrologue({{0, LineTable::Byte}});

  // Add a table with only two standard opcodes - we don't need to test the full
  // set.
  LineTable &Table = Gen->addLineTable(Format);
  DWARFDebugLine::Prologue InputPrologue = Table.createBasicPrologue();
  InputPrologue.OpcodeBase = 3;
  InputPrologue.StandardOpcodeLengths.resize(2);
  Table.setPrologue(InputPrologue);

  generate();
  // Truncate the data extractor to the specified length.
  LineData = DWARFDataExtractor(LineData, Length);

  DWARFDebugLine::Prologue Prologue;
  uint64_t Offset = 1;
  Error Err = Prologue.parse(LineData, &Offset, RecordRecoverable, *Context);

  EXPECT_THAT_ERROR(std::move(Err), FailedWithMessage(ExpectedErr.str()));
  EXPECT_EQ(Offset, ExpectedOffset);
}

INSTANTIATE_TEST_SUITE_P(
    TruncatedPrologueParams, TruncatedPrologueFixture,
    Values(
        // Truncated length:
        std::make_tuple(
            4, 1, 4, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0x4 while reading [0x1, 0x5)"),
        std::make_tuple(
            4, 1, 4, DWARF64,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0x4 while reading [0x1, 0x5)"),
        std::make_tuple(
            0xc, 1, 4, DWARF64,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0xc while reading [0x5, 0xd)"),
        // Truncated version:
        std::make_tuple(
            6, 5, 4, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0x6 while reading [0x5, 0x7)"),
        // Truncated address size:
        std::make_tuple(
            7, 7, 5, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0x7 while reading [0x7, 0x8)"),
        // Truncated segment selector size:
        std::make_tuple(
            8, 8, 5, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0x8 while reading [0x8, 0x9)"),
        // Truncated prologue length:
        std::make_tuple(
            0xa, 7, 4, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0xa while reading [0x7, 0xb)"),
        std::make_tuple(
            0x16, 0xf, 4, DWARF64,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0x16 while reading [0xf, 0x17)"),
        // Truncated min instruction length:
        std::make_tuple(
            0xb, 0xb, 4, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0xb while reading [0xb, 0xc)"),
        // Truncated max ops per inst:
        std::make_tuple(
            0xc, 0xc, 4, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0xc while reading [0xc, 0xd)"),
        // Truncated default is stmt:
        std::make_tuple(
            0xd, 0xd, 4, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0xd while reading [0xd, 0xe)"),
        // Truncated line base:
        std::make_tuple(
            0xe, 0xe, 4, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0xe while reading [0xe, 0xf)"),
        // Truncated line range:
        std::make_tuple(
            0xf, 0xf, 4, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0xf while reading [0xf, 0x10)"),
        // Truncated opcode base:
        std::make_tuple(
            0x10, 0x10, 4, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0x10 while reading [0x10, 0x11)"),
        // Truncated first standard opcode:
        std::make_tuple(
            0x11, 0x11, 4, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0x11 while reading [0x11, 0x12)"),
        // Truncated second standard opcode:
        std::make_tuple(
            0x12, 0x12, 4, DWARF32,
            "parsing line table prologue at offset 0x00000001: unexpected end "
            "of data at offset 0x12 while reading [0x12, 0x13)")));

using ValueAndLengths = std::vector<LineTable::ValueAndLength>;

struct TruncatedOpcodeFixtureBase : public CommonFixture {
  LineTable &setupTable() {
    LineTable &LT = Gen->addLineTable();

    // Creating the prologue before adding any opcodes ensures that the unit
    // length does not include the table body.
    DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();

    // Add an unrecognised standard opcode, and adjust prologue properties
    // accordingly.
    Prologue.TotalLength += BodyLength + 1;
    ++Prologue.PrologueLength;
    ++Prologue.OpcodeBase;
    Prologue.StandardOpcodeLengths.push_back(2);
    LT.setPrologue(Prologue);

    return LT;
  }

  void runTest(uint8_t OpcodeValue) {
    generate();
    DWARFDebugLine::SectionParser Parser(LineData, *Context, Units);
    std::string Output;
    raw_string_ostream OS(Output);
    Parser.parseNext(RecordRecoverable, RecordUnrecoverable, &OS,
                     /*Verbose=*/true);
    OS.flush();

    std::string LinePrefix =
        ("0x0000002f: 0" + Twine::utohexstr(OpcodeValue) + " ").str();
    StringRef OutputRef(Output);
    StringRef OutputToCheck = OutputRef.split(LinePrefix).second;
    // Each extended opcode ends with a new line and then the table ends with an
    // additional blank line.
    EXPECT_EQ((ExpectedOutput + "\n\n").str(), OutputToCheck);
  }

  uint64_t BodyLength;
  uint8_t Opcode;
  ValueAndLengths Operands;
  StringRef ExpectedOutput;
  StringRef ExpectedErr;
};

struct TruncatedStandardOpcodeFixture
    : public TestWithParam<
          std::tuple<uint64_t, uint8_t, ValueAndLengths, StringRef, StringRef>>,
      public TruncatedOpcodeFixtureBase {
  void SetUp() override {
    std::tie(BodyLength, Opcode, Operands, ExpectedOutput, ExpectedErr) =
        GetParam();
  }
};

struct TruncatedExtendedOpcodeFixture
    : public TestWithParam<std::tuple<uint64_t, uint64_t, uint8_t,
                                      ValueAndLengths, StringRef, StringRef>>,
      public TruncatedOpcodeFixtureBase {
  void SetUp() override {
    std::tie(BodyLength, OpcodeLength, Opcode, Operands, ExpectedOutput,
             ExpectedErr) = GetParam();
  }

  uint64_t OpcodeLength;
};

TEST_P(TruncatedExtendedOpcodeFixture, ErrorForTruncatedExtendedOpcode) {
  if (!setupGenerator())
    GTEST_SKIP();
  LineTable &LT = setupTable();
  LT.addExtendedOpcode(OpcodeLength, Opcode, Operands);
  runTest(0);
  EXPECT_THAT_ERROR(std::move(Recoverable),
                    FailedWithMessage(ExpectedErr.str()));
}

INSTANTIATE_TEST_SUITE_P(
    TruncatedExtendedOpcodeParams, TruncatedExtendedOpcodeFixture,
    Values(
        // Truncated length:
        std::make_tuple(1, 1, /*ArbitraryOpcode=*/0x7f, ValueAndLengths(), "",
                        "unable to decode LEB128 at offset 0x00000030: "
                        "malformed uleb128, extends past end"),
        // Truncated opcode:
        std::make_tuple(
            2, 9, /*ArbitraryOpcode=*/0x7f, ValueAndLengths(), "",
            "unexpected end of data at offset 0x31 while reading [0x31, 0x32)"),
        // Truncated operands:
        std::make_tuple(
            3, 9, DW_LNE_set_address,
            ValueAndLengths{{0x1234567890abcdef, LineTable::Quad}},
            "DW_LNE_set_address",
            "unexpected end of data at offset 0x32 while reading [0x32, 0x3a)"),
        std::make_tuple(
            10, 9, DW_LNE_set_address,
            ValueAndLengths{{0x1234567878563412, LineTable::Quad}},
            "DW_LNE_set_address (<parsing error> 12 34 56 78 78 56 34)",
            "unexpected end of data at offset 0x39 while reading [0x32, 0x3a)"),
        std::make_tuple(3, 6, DW_LNE_define_file,
                        ValueAndLengths{{'a', LineTable::Byte},
                                        {'\0', LineTable::Byte},
                                        {1, LineTable::ULEB},
                                        {1, LineTable::ULEB},
                                        {1, LineTable::ULEB}},
                        "DW_LNE_define_file",
                        "no null terminated string at offset 0x32"),
        std::make_tuple(5, 6, DW_LNE_define_file,
                        ValueAndLengths{{'a', LineTable::Byte},
                                        {'\0', LineTable::Byte},
                                        {1, LineTable::ULEB},
                                        {1, LineTable::ULEB},
                                        {1, LineTable::ULEB}},
                        "DW_LNE_define_file (<parsing error> 61 00)",
                        "unable to decode LEB128 at offset 0x00000034: "
                        "malformed uleb128, extends past end"),
        std::make_tuple(6, 6, DW_LNE_define_file,
                        ValueAndLengths{{'a', LineTable::Byte},
                                        {'\0', LineTable::Byte},
                                        {1, LineTable::ULEB},
                                        {1, LineTable::ULEB},
                                        {1, LineTable::ULEB}},
                        "DW_LNE_define_file (<parsing error> 61 00 01)",
                        "unable to decode LEB128 at offset 0x00000035: "
                        "malformed uleb128, extends past end"),
        std::make_tuple(7, 6, DW_LNE_define_file,
                        ValueAndLengths{{'a', LineTable::Byte},
                                        {'\0', LineTable::Byte},
                                        {1, LineTable::ULEB},
                                        {1, LineTable::ULEB},
                                        {1, LineTable::ULEB}},
                        "DW_LNE_define_file (<parsing error> 61 00 01 01)",
                        "unable to decode LEB128 at offset 0x00000036: "
                        "malformed uleb128, extends past end"),
        std::make_tuple(3, 2, DW_LNE_set_discriminator,
                        ValueAndLengths{{1, LineTable::ULEB}},
                        "DW_LNE_set_discriminator",
                        "unable to decode LEB128 at offset 0x00000032: "
                        "malformed uleb128, extends past end"),
        std::make_tuple(
            6, 5, /*Unknown=*/0x7f,
            ValueAndLengths{{0x12343412, LineTable::Long}},
            "Unrecognized extended op 0x7f length 5 (<parsing error> 12 34 34)",
            "unexpected end of data at offset 0x35 while reading [0x32, "
            "0x36)")));

TEST_P(TruncatedStandardOpcodeFixture, ErrorForTruncatedStandardOpcode) {
  if (!setupGenerator())
    GTEST_SKIP();
  LineTable &LT = setupTable();
  LT.addStandardOpcode(Opcode, Operands);
  runTest(Opcode);
  EXPECT_THAT_ERROR(std::move(Unrecoverable),
                    FailedWithMessage(ExpectedErr.str()));
}

INSTANTIATE_TEST_SUITE_P(
    TruncatedStandardOpcodeParams, TruncatedStandardOpcodeFixture,
    Values(
        std::make_tuple(2, DW_LNS_advance_pc,
                        ValueAndLengths{{0x100, LineTable::ULEB}},
                        "DW_LNS_advance_pc",
                        "unable to decode LEB128 at offset 0x00000030: "
                        "malformed uleb128, extends past end"),
        std::make_tuple(2, DW_LNS_advance_line,
                        ValueAndLengths{{0x200, LineTable::SLEB}},
                        "DW_LNS_advance_line",
                        "unable to decode LEB128 at offset 0x00000030: "
                        "malformed sleb128, extends past end"),
        std::make_tuple(2, DW_LNS_set_file,
                        ValueAndLengths{{0x300, LineTable::ULEB}},
                        "DW_LNS_set_file",
                        "unable to decode LEB128 at offset 0x00000030: "
                        "malformed uleb128, extends past end"),
        std::make_tuple(2, DW_LNS_set_column,
                        ValueAndLengths{{0x400, LineTable::ULEB}},
                        "DW_LNS_set_column",
                        "unable to decode LEB128 at offset 0x00000030: "
                        "malformed uleb128, extends past end"),
        std::make_tuple(
            2, DW_LNS_fixed_advance_pc,
            ValueAndLengths{{0x500, LineTable::Half}},
            "DW_LNS_fixed_advance_pc",
            "unexpected end of data at offset 0x31 while reading [0x30, 0x32)"),
        std::make_tuple(2, DW_LNS_set_isa,
                        ValueAndLengths{{0x600, LineTable::ULEB}},
                        "DW_LNS_set_isa",
                        "unable to decode LEB128 at offset 0x00000030: "
                        "malformed uleb128, extends past end"),
        std::make_tuple(2, 0xd,
                        ValueAndLengths{{0x700, LineTable::ULEB},
                                        {0x800, LineTable::ULEB}},
                        "Unrecognized standard opcode",
                        "unable to decode LEB128 at offset 0x00000030: "
                        "malformed uleb128, extends past end"),
        std::make_tuple(
            4, 0xd,
            ValueAndLengths{{0x900, LineTable::ULEB}, {0xa00, LineTable::ULEB}},
            "Unrecognized standard opcode (operands: 0x0000000000000900)",
            "unable to decode LEB128 at offset 0x00000032: "
            "malformed uleb128, extends past end")));

#ifdef NO_SUPPORT_DEBUG_ADDR
TEST_F(DebugLineBasicFixture, DISABLED_PrintPathsProperly) {
#else
TEST_F(DebugLineBasicFixture, PrintPathsProperly) {
#endif
  if (!setupGenerator(5))
    GTEST_SKIP();

  LineTable &LT = Gen->addLineTable();
  DWARFDebugLine::Prologue P = LT.createBasicPrologue();
  P.IncludeDirectories.push_back(
      DWARFFormValue::createFromPValue(DW_FORM_string, "b dir"));
  P.FileNames.push_back(DWARFDebugLine::FileNameEntry());
  P.FileNames.back().Name =
      DWARFFormValue::createFromPValue(DW_FORM_string, "b file");
  P.FileNames.back().DirIdx = 1;
  P.TotalLength += 14;
  P.PrologueLength += 14;
  LT.setPrologue(P);
  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  EXPECT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  std::string Result;
  // DWARF 5 stores the compilation directory in two places: the Compilation
  // Unit and the directory table entry 0, and implementations are free to use
  // one or the other. This copy serves as the one stored in the CU.
  StringRef CompDir = "a dir";
  EXPECT_FALSE(
      (*ExpectedLineTable)
          ->Prologue.getFileNameByIndex(
              1, CompDir, DILineInfoSpecifier::FileLineInfoKind::None, Result));
  EXPECT_TRUE((*ExpectedLineTable)
                  ->Prologue.getFileNameByIndex(
                      1, CompDir,
                      DILineInfoSpecifier::FileLineInfoKind::RawValue, Result));
  EXPECT_TRUE((*ExpectedLineTable)
                  ->Prologue.getFileNameByIndex(
                      1, CompDir,
                      DILineInfoSpecifier::FileLineInfoKind::BaseNameOnly,
                      Result));
  EXPECT_STREQ(Result.c_str(), "b file");
  EXPECT_TRUE((*ExpectedLineTable)
                  ->Prologue.getFileNameByIndex(
                      1, CompDir,
                      DILineInfoSpecifier::FileLineInfoKind::RelativeFilePath,
                      Result));
  EXPECT_THAT(Result.c_str(), MatchesRegex("b dir.b file"));
  EXPECT_TRUE((*ExpectedLineTable)
                  ->Prologue.getFileNameByIndex(
                      1, CompDir,
                      DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath,
                      Result));
  EXPECT_THAT(Result.c_str(), MatchesRegex("a dir.b dir.b file"));
}

} // end anonymous namespace
