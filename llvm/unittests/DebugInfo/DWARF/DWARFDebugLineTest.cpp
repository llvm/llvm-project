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

  bool setupGenerator(uint16_t Version = 4, uint8_t AddrSize = 8) {
    AddressSize = AddrSize;
    Triple T =
        getDefaultTargetTripleForAddrSize(AddressSize == 0 ? 8 : AddressSize);
    if (!isConfigurationSupported(T))
      return false;
    auto ExpectedGenerator = Generator::create(T, Version);
    if (ExpectedGenerator)
      Gen.reset(ExpectedGenerator->release());
    return true;
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
    if (!Gen)
      return nullptr;
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

    return DWARFDebugLine::SectionParser(LineData, *Context, CUs, TUs);
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

  SmallVector<std::unique_ptr<DWARFUnit>, 2> CUs;
  SmallVector<std::unique_ptr<DWARFUnit>, 2> TUs;
};

// Fixtures must derive from "Test", but parameterised fixtures from
// "TestWithParam". It does not seem possible to inherit from both, so we share
// the common state in a separate class, inherited by the two fixture classes.
struct DebugLineBasicFixture : public Test, public CommonFixture {};

struct DebugLineParameterisedFixture
    : public TestWithParam<std::pair<uint16_t, DwarfFormat>>,
      public CommonFixture {
  void SetUp() { std::tie(Version, Format) = GetParam(); }

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
  EXPECT_STREQ(*Prologue.IncludeDirectories[0].getAsCString(), "a dir");
  ASSERT_EQ(Prologue.FileNames.size(), 1u);
  ASSERT_EQ(Prologue.FileNames[0].Name.getForm(), DW_FORM_string);
  ASSERT_EQ(Prologue.FileNames[0].DirIdx, 0u);
  EXPECT_STREQ(*Prologue.FileNames[0].Name.getAsCString(), "a file");
}

TEST_F(DebugLineBasicFixture, GetOrParseLineTableAtInvalidOffset) {
  if (!setupGenerator())
    return;
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
    return;

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

TEST_P(DebugLineParameterisedFixture, PrologueGetLength) {
  if (!setupGenerator(Version))
    return;
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

TEST_P(DebugLineParameterisedFixture, GetOrParseLineTableValidTable) {
  if (!setupGenerator(Version))
    return;

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

TEST_F(DebugLineBasicFixture, ErrorForReservedLength) {
  if (!setupGenerator())
    return;

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
  void SetUp() { Version = GetParam(); }

  uint16_t Version;
};

TEST_P(DebugLineUnsupportedVersionFixture, ErrorForUnsupportedVersion) {
  if (!setupGenerator())
    return;

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

INSTANTIATE_TEST_CASE_P(UnsupportedVersionTestParams,
                        DebugLineUnsupportedVersionFixture,
                        Values(/*1 below min */ 1, /* 1 above max */ 6,
                               /* Maximum possible */ 0xffff), );

TEST_F(DebugLineBasicFixture, ErrorForInvalidV5IncludeDirTable) {
  if (!setupGenerator(5))
    return;

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

TEST_P(DebugLineParameterisedFixture, ErrorForTooLargePrologueLength) {
  if (!setupGenerator(Version))
    return;

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
      FailedWithMessage(("parsing line table prologue at 0x00000000 should "
                         "have ended at 0x000000" +
                         Twine::utohexstr(ExpectedEnd) +
                         " but it ended at 0x000000" +
                         Twine::utohexstr(ExpectedEnd - 1))
                            .str()));
}

TEST_P(DebugLineParameterisedFixture, ErrorForTooShortPrologueLength) {
  if (!setupGenerator(Version))
    return;

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

  if (Version != 5) {
    // Parsing will stop before reading a complete file entry.
    ASSERT_EQ(Result.Prologue.IncludeDirectories.size(), 1u);
    EXPECT_EQ(toStringRef(Result.Prologue.IncludeDirectories[0]), "a dir");
    EXPECT_EQ(Result.Prologue.FileNames.size(), 0u);
  } else {
    // Parsing will continue past the presumed end of prologue.
    ASSERT_EQ(Result.Prologue.FileNames.size(), 1u);
    ASSERT_EQ(Result.Prologue.FileNames[0].Name.getForm(), DW_FORM_string);
    ASSERT_EQ(Result.Prologue.FileNames[0].DirIdx, 0u);
    EXPECT_EQ(toStringRef(Result.Prologue.FileNames[0].Name), "a file");
  }

  uint64_t ExpectedEnd =
      Prologue.TotalLength - 2 + Prologue.sizeofTotalLength();
  std::vector<std::string> Errs;
  if (Version != 5) {
    Errs.emplace_back(
        (Twine("parsing line table prologue at 0x00000000 found an invalid "
               "directory or file table description at 0x000000") +
         Twine::utohexstr(ExpectedEnd + 1))
            .str());
    Errs.emplace_back("file names table was not null terminated before the end "
                      "of the prologue");
  } else {
    Errs.emplace_back(
        (Twine("parsing line table prologue at 0x00000000 should have ended at "
               "0x000000") +
         Twine::utohexstr(ExpectedEnd) + " but it ended at 0x000000" +
         Twine::utohexstr(ExpectedEnd + 2))
            .str());
  }
  EXPECT_THAT_ERROR(std::move(Recoverable),
                    FailedWithMessageArray(testing::ElementsAreArray(Errs)));
}

INSTANTIATE_TEST_CASE_P(
    LineTableTestParams, DebugLineParameterisedFixture,
    Values(std::make_pair(
               2, DWARF32), // Test lower-bound of v2-3 fields and DWARF32.
           std::make_pair(3, DWARF32), // Test upper-bound of v2-3 fields.
           std::make_pair(4, DWARF64), // Test v4 fields and DWARF64.
           std::make_pair(5, DWARF32), std::make_pair(5, DWARF64)), );

TEST_F(DebugLineBasicFixture, ErrorForExtendedOpcodeLengthSmallerThanExpected) {
  if (!setupGenerator())
    return;

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
    return;

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
    return;

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
    return;

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
    return;

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
    return;

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
    return;

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

TEST_F(DebugLineBasicFixture, ErrorForUnsupportedAddressSizeDefinedInHeader) {
  // Use 0 for data extractor address size so that it does not clash with the
  // header address size.
  if (!setupGenerator(5, 0))
    return;

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
      FailedWithMessage("address size 0x09 of DW_LNE_set_address opcode at "
                        "offset 0x00000038 is unsupported"));
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  ASSERT_EQ((*ExpectedLineTable)->Rows.size(), 3u);
  EXPECT_EQ((*ExpectedLineTable)->Sequences.size(), 1u);
  // Show that the set address opcode is ignored in this case.
  EXPECT_EQ((*ExpectedLineTable)->Rows[0].Address.Address, 0u);
}

TEST_F(DebugLineBasicFixture, CallbackUsedForUnterminatedSequence) {
  if (!setupGenerator())
    return;

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
      return;

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

struct MaxOpsPerInstFixture
    : TestWithParam<std::tuple<uint16_t, uint8_t, bool>>,
      AdjustAddressFixtureBase {
  void SetUp() override {
    std::tie(Version, MaxOpsPerInst, IsErrorExpected) = GetParam();
  }

  uint64_t editPrologue(LineTable &LT) override {
    DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();
    Prologue.MaxOpsPerInst = MaxOpsPerInst;
    LT.setPrologue(Prologue);
    return Prologue.TotalLength + Prologue.sizeofTotalLength();
  }

  uint8_t MaxOpsPerInst;
};

TEST_P(MaxOpsPerInstFixture, MaxOpsPerInstProblemsReportedCorrectly) {
  runTest(/*CheckAdvancePC=*/true,
          "but the prologue maximum_operations_per_instruction value is " +
              Twine(unsigned(MaxOpsPerInst)) +
              ", which is unsupported. Assuming a value of 1 instead");
}

INSTANTIATE_TEST_CASE_P(
    MaxOpsPerInstParams, MaxOpsPerInstFixture,
    Values(std::make_tuple(3, 0, false), // Test for version < 4 (no error).
           std::make_tuple(4, 0, true),  // Test zero value for V4 (error).
           std::make_tuple(4, 1, false), // Test good value for V4 (no error).
           std::make_tuple(
               4, 2, true)), ); // Test one higher than permitted V4 (error).

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

INSTANTIATE_TEST_CASE_P(
    LineRangeParams, LineRangeFixture,
    Values(std::make_tuple(0, true),       // Test zero value (error).
           std::make_tuple(14, false)), ); // Test non-zero value (no error).

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

INSTANTIATE_TEST_CASE_P(
    BadMinInstLenParams, BadMinInstLenFixture,
    Values(std::make_tuple(0, true),      // Test zero value (error).
           std::make_tuple(1, false)), ); // Test non-zero value (no error).

TEST_F(DebugLineBasicFixture, ParserParsesCorrectly) {
  if (!setupGenerator())
    return;

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
    return;

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
    return;

  generate();
  DWARFDebugLine::SectionParser Parser(LineData, *Context, CUs, TUs);

  EXPECT_TRUE(Parser.done());
}

TEST_F(DebugLineBasicFixture, ParserMarkedAsDoneForBadLengthWhenParsing) {
  if (!setupGenerator())
    return;

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue({{0xfffffff0, LineTable::Long}});
  Gen->addLineTable();
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, CUs, TUs);
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
    return;

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue({{0xfffffff0, LineTable::Long}});
  Gen->addLineTable();
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, CUs, TUs);
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
    return;

  LineTable &LT = Gen->addLineTable(DWARF32);
  LT.setCustomPrologue({{2, LineTable::Long}, {0, LineTable::Half}});
  LineTable &LT2 = Gen->addLineTable(DWARF32);
  LT2.setCustomPrologue({{2, LineTable::Long}, {1, LineTable::Half}});
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, CUs, TUs);
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
    return;

  LineTable &LT = Gen->addLineTable(DWARF32);
  LT.addExtendedOpcode(0x42, DW_LNE_end_sequence, {});
  LineTable &LT2 = Gen->addLineTable(DWARF32);
  LT2.addExtendedOpcode(9, DW_LNE_set_address,
                        {{0x1234567890abcdef, LineTable::Quad}});
  LT2.addStandardOpcode(DW_LNS_copy, {});
  LT2.addByte(0xbb);
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, CUs, TUs);
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
    return;

  LineTable &LT = Gen->addLineTable(DWARF32);
  LT.setCustomPrologue({{2, LineTable::Long}, {0, LineTable::Half}});
  LineTable &LT2 = Gen->addLineTable(DWARF32);
  LT2.setCustomPrologue({{2, LineTable::Long}, {1, LineTable::Half}});
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, CUs, TUs);
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
    return;

  LineTable &LT = Gen->addLineTable(DWARF32);
  LT.addExtendedOpcode(42, DW_LNE_end_sequence, {});
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, CUs, TUs);
  Parser.skip(RecordRecoverable, RecordUnrecoverable);

  EXPECT_TRUE(Parser.done());
  EXPECT_FALSE(Recoverable);
  EXPECT_FALSE(Unrecoverable);
}

TEST_F(DebugLineBasicFixture, ParserPrintsStandardOpcodesWhenRequested) {
  if (!setupGenerator())
    return;

  using ValLen = dwarfgen::LineTable::ValueAndLength;
  LineTable &LT = Gen->addLineTable(DWARF32);
  LT.addStandardOpcode(DW_LNS_copy, {});
  LT.addStandardOpcode(DW_LNS_advance_pc, {ValLen{11, LineTable::ULEB}});
  LT.addStandardOpcode(DW_LNS_advance_line, {ValLen{22, LineTable::SLEB}});
  LT.addStandardOpcode(DW_LNS_set_file, {ValLen{33, LineTable::ULEB}});
  LT.addStandardOpcode(DW_LNS_set_column, {ValLen{44, LineTable::ULEB}});
  LT.addStandardOpcode(DW_LNS_negate_stmt, {});
  LT.addStandardOpcode(DW_LNS_set_basic_block, {});
  LT.addStandardOpcode(DW_LNS_const_add_pc, {});
  LT.addStandardOpcode(DW_LNS_fixed_advance_pc, {ValLen{55, LineTable::Half}});
  LT.addStandardOpcode(DW_LNS_set_prologue_end, {});
  LT.addStandardOpcode(DW_LNS_set_epilogue_begin, {});
  LT.addStandardOpcode(DW_LNS_set_isa, {ValLen{66, LineTable::ULEB}});
  LT.addExtendedOpcode(1, DW_LNE_end_sequence, {});
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, CUs, TUs);
  std::string Output;
  raw_string_ostream OS(Output);
  Parser.parseNext(RecordRecoverable, RecordUnrecoverable, &OS);
  OS.flush();

  EXPECT_FALSE(Recoverable);
  EXPECT_FALSE(Unrecoverable);
  auto InOutput = [&Output](char const *Str) {
    return Output.find(Str) != std::string::npos;
  };
  EXPECT_TRUE(InOutput("0x0000002e: 01 DW_LNS_copy\n")) << Output;
  EXPECT_TRUE(InOutput("0x0000002f: 02 DW_LNS_advance_pc (11)\n")) << Output;
  // FIXME: The value printed after DW_LNS_advance_line is currently the result
  // of the advance, but it should be the value being advanced by. See
  // https://bugs.llvm.org/show_bug.cgi?id=44261 for details.
  EXPECT_TRUE(InOutput("0x00000031: 03 DW_LNS_advance_line (23)\n")) << Output;
  EXPECT_TRUE(InOutput("0x00000033: 04 DW_LNS_set_file (33)\n")) << Output;
  EXPECT_TRUE(InOutput("0x00000035: 05 DW_LNS_set_column (44)\n")) << Output;
  EXPECT_TRUE(InOutput("0x00000037: 06 DW_LNS_negate_stmt\n")) << Output;
  EXPECT_TRUE(InOutput("0x00000038: 07 DW_LNS_set_basic_block\n")) << Output;
  EXPECT_TRUE(
      InOutput("0x00000039: 08 DW_LNS_const_add_pc (0x0000000000000011)\n"))
      << Output;
  EXPECT_TRUE(InOutput("0x0000003a: 09 DW_LNS_fixed_advance_pc (0x0037)\n"))
      << Output;
  EXPECT_TRUE(InOutput("0x0000003d: 0a DW_LNS_set_prologue_end\n")) << Output;
  EXPECT_TRUE(InOutput("0x0000003e: 0b DW_LNS_set_epilogue_begin\n")) << Output;
  EXPECT_TRUE(InOutput("0x0000003f: 0c DW_LNS_set_isa (66)\n")) << Output;
}

TEST_F(DebugLineBasicFixture, PrintPathsProperly) {
  if (!setupGenerator(5))
    return;

  LineTable &LT = Gen->addLineTable();
  DWARFDebugLine::Prologue P = LT.createBasicPrologue();
  P.IncludeDirectories.push_back(
      DWARFFormValue::createFromPValue(DW_FORM_string, "b dir"));
  P.FileNames.push_back(DWARFDebugLine::FileNameEntry());
  P.FileNames.back().Name =
      DWARFFormValue::createFromPValue(DW_FORM_string, "b file");
  P.FileNames.back().DirIdx = 1;
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
