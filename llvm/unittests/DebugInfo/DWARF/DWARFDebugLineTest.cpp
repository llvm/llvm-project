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

  void checkError(ArrayRef<StringRef> ExpectedMsgs, Error Err) {
    ASSERT_TRUE(Err.operator bool());
    size_t WhichMsg = 0;
    Error Remaining =
        handleErrors(std::move(Err), [&](const ErrorInfoBase &Actual) {
          ASSERT_LT(WhichMsg, ExpectedMsgs.size());
          // Use .str(), because googletest doesn't visualise a StringRef
          // properly.
          EXPECT_EQ(Actual.message(), ExpectedMsgs[WhichMsg++].str());
        });
    EXPECT_EQ(WhichMsg, ExpectedMsgs.size());
    EXPECT_FALSE(Remaining);
  }

  void checkError(StringRef ExpectedMsg, Error Err) {
    checkError(ArrayRef<StringRef>{ExpectedMsg}, std::move(Err));
  }

  void checkGetOrParseLineTableEmitsFatalError(StringRef ExpectedMsg,
                                               uint64_t Offset = 0) {
    auto ExpectedLineTable = Line.getOrParseLineTable(
        LineData, Offset, *Context, nullptr, RecordRecoverable);
    EXPECT_FALSE(ExpectedLineTable);
    EXPECT_FALSE(Recoverable);

    checkError(ExpectedMsg, ExpectedLineTable.takeError());
  }

  void checkGetOrParseLineTableEmitsFatalError(ArrayRef<StringRef> ExpectedMsgs,
                                               uint64_t Offset = 0) {
    auto ExpectedLineTable = Line.getOrParseLineTable(
        LineData, Offset, *Context, nullptr, RecordRecoverable);
    EXPECT_FALSE(ExpectedLineTable);
    EXPECT_FALSE(Recoverable);

    checkError(ExpectedMsgs, ExpectedLineTable.takeError());
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

  checkGetOrParseLineTableEmitsFatalError(
      "offset 0x00000000 is not a valid debug line section offset", 0);
  // Repeat to show that an error is reported each time.
  checkGetOrParseLineTableEmitsFatalError(
      "offset 0x00000000 is not a valid debug line section offset", 0);
  // Show that an error is reported for later offsets too.
  checkGetOrParseLineTableEmitsFatalError(
      "offset 0x00000001 is not a valid debug line section offset", 1);
}

TEST_F(DebugLineBasicFixture, GetOrParseLineTableAtInvalidOffsetAfterData) {
  if (!setupGenerator())
    return;

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue({{0, LineTable::Byte}});

  generate();

  checkGetOrParseLineTableEmitsFatalError(
      "offset 0x00000001 is not a valid debug line section offset", 1);
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

  checkGetOrParseLineTableEmitsFatalError(
      "parsing line table prologue at offset 0x00000000 unsupported reserved "
      "unit length found of value 0xfffffff0");
}

TEST_F(DebugLineBasicFixture, ErrorForLowVersion) {
  if (!setupGenerator())
    return;

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue(
      {{LineTable::Half, LineTable::Long}, {1, LineTable::Half}});

  generate();

  checkGetOrParseLineTableEmitsFatalError(
      "parsing line table prologue at offset "
      "0x00000000 found unsupported version "
      "0x01");
}

TEST_F(DebugLineBasicFixture, ErrorForHighVersion) {
  if (!setupGenerator())
    return;

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue(
      {{LineTable::Half, LineTable::Long}, {6, LineTable::Half}});

  generate();

  checkGetOrParseLineTableEmitsFatalError(
      "parsing line table prologue at offset 0x00000000 found unsupported "
      "version 0x06");
}

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

  checkError(
      {"parsing line table prologue at 0x00000000 found an invalid directory "
       "or file table description at 0x00000014",
       "failed to parse entry content descriptions because no path was found"},
      std::move(Recoverable));
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
  checkError(
      (Twine("parsing line table prologue at 0x00000000 should have ended at "
             "0x000000") +
       Twine::utohexstr(ExpectedEnd) + " but it ended at 0x000000" +
       Twine::utohexstr(ExpectedEnd - 1))
          .str(),
      std::move(Recoverable));
}

TEST_P(DebugLineParameterisedFixture, ErrorForTooShortPrologueLength) {
  if (!setupGenerator(Version))
    return;

  SCOPED_TRACE("Checking Version " + std::to_string(Version) + ", Format " +
               (Format == DWARF64 ? "DWARF64" : "DWARF32"));

  LineTable &LT = Gen->addLineTable(Format);
  DWARFDebugLine::Prologue Prologue = LT.createBasicPrologue();
  // FIXME: Ideally, we'd test for 1 less than expected, but the code does not
  // currently fail if missing only the terminator of a v2-4 file table.
  if (Version < 5)
    Prologue.PrologueLength -= 2;
  else
    Prologue.PrologueLength -= 1;
  LT.setPrologue(Prologue);

  generate();

  auto ExpectedLineTable = Line.getOrParseLineTable(LineData, 0, *Context,
                                                    nullptr, RecordRecoverable);
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  DWARFDebugLine::LineTable Result(**ExpectedLineTable);
  // Undo the earlier modification so that it can be compared against a
  // "default" prologue.
  if (Version < 5)
    Result.Prologue.PrologueLength += 2;
  else
    Result.Prologue.PrologueLength += 1;
  checkDefaultPrologue(Version, Format, Result.Prologue, 0);

  uint64_t ExpectedEnd =
      Prologue.TotalLength - 1 + Prologue.sizeofTotalLength();
  if (Version < 5)
    --ExpectedEnd;
  checkError(
      (Twine("parsing line table prologue at 0x00000000 should have ended at "
             "0x000000") +
       Twine::utohexstr(ExpectedEnd) + " but it ended at 0x000000" +
       Twine::utohexstr(ExpectedEnd + 1))
          .str(),
      std::move(Recoverable));
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
  checkError(
      "unexpected line op length at offset 0x00000031 expected 0x01 found 0x02",
      std::move(Recoverable));
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
  checkError(
      "unexpected line op length at offset 0x00000032 expected 0x02 found 0x01",
      std::move(Recoverable));
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
  checkError("line table program with offset 0x00000001 has length 0x00000034 "
             "but only 0x00000033 bytes are available",
             std::move(Recoverable));
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
  checkError(
      "mismatching address size at offset 0x00000030 expected 0x08 found 0x04",
      std::move(Recoverable));
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
  checkError(
      "mismatching address size at offset 0x00000038 expected 0x04 found 0x08",
      std::move(Recoverable));
  ASSERT_THAT_EXPECTED(ExpectedLineTable, Succeeded());
  ASSERT_EQ((*ExpectedLineTable)->Rows.size(), 2u);
  EXPECT_EQ((*ExpectedLineTable)->Sequences.size(), 1u);
  EXPECT_EQ((*ExpectedLineTable)->Rows[0].Address.Address, Addr1);
  EXPECT_EQ((*ExpectedLineTable)->Rows[1].Address.Address, Addr2);
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
  checkError("last sequence in debug line table at offset 0x00000000 is not "
             "terminated",
             std::move(Recoverable));
  ASSERT_TRUE(ExpectedLineTable.operator bool());
  EXPECT_EQ((*ExpectedLineTable)->Rows.size(), 6u);
  // The unterminated sequence is not added to the sequence list.
  EXPECT_EQ((*ExpectedLineTable)->Sequences.size(), 1u);
}

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

TEST_F(DebugLineBasicFixture, ParserMovesToEndForBadLengthWhenParsing) {
  if (!setupGenerator())
    return;

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue({{0xfffffff0, LineTable::Long}});
  Gen->addLineTable();
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, CUs, TUs);
  Parser.parseNext(RecordRecoverable, RecordUnrecoverable);

  EXPECT_EQ(Parser.getOffset(), 4u);
  EXPECT_TRUE(Parser.done());
  EXPECT_FALSE(Recoverable);

  checkError("parsing line table prologue at offset 0x00000000 unsupported "
             "reserved unit length found of value 0xfffffff0",
             std::move(Unrecoverable));
}

TEST_F(DebugLineBasicFixture, ParserMovesToEndForBadLengthWhenSkipping) {
  if (!setupGenerator())
    return;

  LineTable &LT = Gen->addLineTable();
  LT.setCustomPrologue({{0xfffffff0, LineTable::Long}});
  Gen->addLineTable();
  generate();

  DWARFDebugLine::SectionParser Parser(LineData, *Context, CUs, TUs);
  Parser.skip(RecordRecoverable, RecordUnrecoverable);

  EXPECT_EQ(Parser.getOffset(), 4u);
  EXPECT_TRUE(Parser.done());
  EXPECT_FALSE(Recoverable);

  checkError("parsing line table prologue at offset 0x00000000 unsupported "
             "reserved unit length found of value 0xfffffff0",
             std::move(Unrecoverable));
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
  EXPECT_FALSE(Recoverable);

  checkError({"parsing line table prologue at offset 0x00000000 found "
              "unsupported version 0x00",
              "parsing line table prologue at offset 0x00000006 found "
              "unsupported version 0x01"},
             std::move(Unrecoverable));
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
  checkError(
      "unexpected line op length at offset 0x00000030 expected 0x42 found 0x01",
      std::move(Recoverable));

  // Reset the error state so that it does not confuse the next set of checks.
  Unrecoverable = Error::success();
  Parser.parseNext(RecordRecoverable, RecordUnrecoverable);

  EXPECT_TRUE(Parser.done());
  checkError("last sequence in debug line table at offset 0x00000031 is not "
             "terminated",
             std::move(Recoverable));
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

  checkError({"parsing line table prologue at offset 0x00000000 found "
              "unsupported version 0x00",
              "parsing line table prologue at offset 0x00000006 found "
              "unsupported version 0x01"},
             std::move(Unrecoverable));
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

} // end anonymous namespace
