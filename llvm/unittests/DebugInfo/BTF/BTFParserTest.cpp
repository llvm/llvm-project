//===-- SourcePrinter.cpp -  source interleaving utilities ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/BTF/BTFContext.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace llvm::object;

#define LC(Line, Col) ((Line << 10u) | Col)

const char BTFEndOfData[] =
    "error while reading .BTF section: unexpected end of data";
const char BTFExtEndOfData[] =
    "error while reading .BTF.ext section: unexpected end of data";

static raw_ostream &operator<<(raw_ostream &OS, const yaml::BinaryRef &Ref) {
  Ref.writeAsHex(OS);
  return OS;
}

template <typename T>
static yaml::BinaryRef makeBinRef(const T *Ptr, size_t Size = sizeof(T)) {
  return yaml::BinaryRef(ArrayRef<uint8_t>((const uint8_t *)Ptr, Size));
}

namespace {
// This is a mockup for an ELF file containing .BTF and .BTF.ext sections.
// Binary content of these sections corresponds to the value of
// MockData1::BTF and MockData1::Ext fields.
//
// The yaml::yaml2ObjectFile() is used to generate actual ELF,
// see MockData1::makeObj().
//
// The `BTF` and `Ext` fields are initialized with correct values
// valid for a small example with a few sections, fields could be
// modified before a call to `makeObj()` to test parser with invalid
// input, etc.

struct MockData1 {
// Use "pragma pack" to model .BTF & .BTF.ext sections content using
// 'struct' objects. This pragma is supported by CLANG, GCC & MSVC,
// which matters for LLVM CI.
#pragma pack(push, 1)
  struct B {
    BTF::Header Header = {};
    // no types
    struct S {
      char Foo[4] = "foo";
      char Bar[4] = "bar";
      char Buz[4] = "buz";
      char Line1[11] = "first line";
      char Line2[12] = "second line";
      char File1[4] = "a.c";
      char File2[4] = "b.c";
    } Strings;

    B() {
      Header.Magic = BTF::MAGIC;
      Header.Version = 1;
      Header.HdrLen = sizeof(Header);
      Header.StrOff = offsetof(B, Strings) - sizeof(Header);
      Header.StrLen = sizeof(Strings);
    }
  } BTF;

  struct E {
    BTF::ExtHeader Header = {};
    // no func info
    struct {
      uint32_t LineRecSize = sizeof(BTF::BPFLineInfo);
      struct {
        BTF::SecLineInfo Sec = {offsetof(B::S, Foo), 2};
        BTF::BPFLineInfo Lines[2] = {
            {16, offsetof(B::S, File1), offsetof(B::S, Line1), LC(7, 1)},
            {32, offsetof(B::S, File1), offsetof(B::S, Line2), LC(14, 5)},
        };
      } Foo;
      struct {
        BTF::SecLineInfo Sec = {offsetof(B::S, Bar), 1};
        BTF::BPFLineInfo Lines[1] = {
            {0, offsetof(B::S, File2), offsetof(B::S, Line1), LC(42, 4)},
        };
      } Bar;
    } Lines;

    E() {
      Header.Magic = BTF::MAGIC;
      Header.Version = 1;
      Header.HdrLen = sizeof(Header);
      Header.LineInfoOff = offsetof(E, Lines) - sizeof(Header);
      Header.LineInfoLen = sizeof(Lines);
    }
  } Ext;
#pragma pack(pop)

  int BTFSectionLen = sizeof(BTF);
  int ExtSectionLen = sizeof(Ext);

  SmallString<0> Storage;
  std::unique_ptr<ObjectFile> Obj;

  ObjectFile &makeObj() {
    std::string Buffer;
    raw_string_ostream Yaml(Buffer);
    Yaml << R"(
!ELF
FileHeader:
  Class:    ELFCLASS64)";
    if (sys::IsBigEndianHost)
      Yaml << "\n  Data:     ELFDATA2MSB";
    else
      Yaml << "\n  Data:     ELFDATA2LSB";
    Yaml << R"(
  Type:     ET_REL
  Machine:  EM_BPF
Sections:
  - Name:     foo
    Type:     SHT_PROGBITS
    Size:     0x0
  - Name:     bar
    Type:     SHT_PROGBITS
    Size:     0x0)";

    if (BTFSectionLen >= 0)
      Yaml << R"(
  - Name:     .BTF
    Type:     SHT_PROGBITS
    Content: )"
           << makeBinRef(&BTF, BTFSectionLen);

    if (ExtSectionLen >= 0)
      Yaml << R"(
  - Name:     .BTF.ext
    Type:     SHT_PROGBITS
    Content: )"
           << makeBinRef(&Ext, ExtSectionLen);

    Obj = yaml::yaml2ObjectFile(Storage, Buffer,
                                [](const Twine &Err) { errs() << Err; });
    return *Obj.get();
  }
};

TEST(BTFParserTest, simpleCorrectInput) {
  BTFParser BTF;
  MockData1 Mock;
  Error Err = BTF.parse(Mock.makeObj());
  EXPECT_FALSE(Err);

  EXPECT_EQ(BTF.findString(offsetof(MockData1::B::S, Foo)), "foo");
  EXPECT_EQ(BTF.findString(offsetof(MockData1::B::S, Bar)), "bar");
  EXPECT_EQ(BTF.findString(offsetof(MockData1::B::S, Line1)), "first line");
  EXPECT_EQ(BTF.findString(offsetof(MockData1::B::S, Line2)), "second line");
  EXPECT_EQ(BTF.findString(offsetof(MockData1::B::S, File1)), "a.c");
  EXPECT_EQ(BTF.findString(offsetof(MockData1::B::S, File2)), "b.c");

  // Invalid offset
  EXPECT_EQ(BTF.findString(sizeof(MockData1::B::S)), StringRef());

  const BTF::BPFLineInfo *I1 = BTF.findLineInfo({16, 1});
  ASSERT_TRUE(I1);
  EXPECT_EQ(I1->getLine(), 7u);
  EXPECT_EQ(I1->getCol(), 1u);
  EXPECT_EQ(BTF.findString(I1->FileNameOff), "a.c");
  EXPECT_EQ(BTF.findString(I1->LineOff), "first line");

  const BTF::BPFLineInfo *I2 = BTF.findLineInfo({32, 1});
  ASSERT_TRUE(I2);
  EXPECT_EQ(I2->getLine(), 14u);
  EXPECT_EQ(I2->getCol(), 5u);
  EXPECT_EQ(BTF.findString(I2->FileNameOff), "a.c");
  EXPECT_EQ(BTF.findString(I2->LineOff), "second line");

  const BTF::BPFLineInfo *I3 = BTF.findLineInfo({0, 2});
  ASSERT_TRUE(I3);
  EXPECT_EQ(I3->getLine(), 42u);
  EXPECT_EQ(I3->getCol(), 4u);
  EXPECT_EQ(BTF.findString(I3->FileNameOff), "b.c");
  EXPECT_EQ(BTF.findString(I3->LineOff), "first line");

  // No info for insn address
  EXPECT_FALSE(BTF.findLineInfo({24, 1}));
  EXPECT_FALSE(BTF.findLineInfo({8, 2}));
  // No info for section number
  EXPECT_FALSE(BTF.findLineInfo({16, 3}));
}

TEST(BTFParserTest, badSectionNameOffset) {
  BTFParser BTF;
  MockData1 Mock;
  // "foo" is section #1, corrupting it's name offset will make impossible
  // to match section name with section index when BTF is parsed.
  Mock.Ext.Lines.Foo.Sec.SecNameOff = 100500;
  Error Err = BTF.parse(Mock.makeObj());
  EXPECT_FALSE(Err);
  // "foo" line info should be corrupted
  EXPECT_FALSE(BTF.findLineInfo({16, 1}));
  // "bar" line info should be ok
  EXPECT_TRUE(BTF.findLineInfo({0, 2}));
}

// Keep this as macro to preserve line number info
#define EXPECT_PARSE_ERROR(Mock, Message)                                      \
  do {                                                                         \
    BTFParser BTF;                                                             \
    EXPECT_THAT_ERROR(BTF.parse((Mock).makeObj()),                             \
                      FailedWithMessage(testing::HasSubstr(Message)));         \
  } while (false)

TEST(BTFParserTest, badBTFMagic) {
  MockData1 Mock;
  Mock.BTF.Header.Magic = 42;
  EXPECT_PARSE_ERROR(Mock, "invalid .BTF magic: 2a");
}

TEST(BTFParserTest, badBTFVersion) {
  MockData1 Mock;
  Mock.BTF.Header.Version = 42;
  EXPECT_PARSE_ERROR(Mock, "unsupported .BTF version: 42");
}

TEST(BTFParserTest, badBTFHdrLen) {
  MockData1 Mock;
  Mock.BTF.Header.HdrLen = 5;
  EXPECT_PARSE_ERROR(Mock, "unexpected .BTF header length: 5");
}

TEST(BTFParserTest, badBTFSectionLen) {
  MockData1 Mock1, Mock2;

  // Cut-off string section by one byte
  Mock1.BTFSectionLen =
      offsetof(MockData1::B, Strings) + sizeof(MockData1::B::S) - 1;
  EXPECT_PARSE_ERROR(Mock1, "invalid .BTF section size");

  // Cut-off header
  Mock2.BTFSectionLen = offsetof(BTF::Header, StrOff);
  EXPECT_PARSE_ERROR(Mock2, BTFEndOfData);
}

TEST(BTFParserTest, badBTFExtMagic) {
  MockData1 Mock;
  Mock.Ext.Header.Magic = 42;
  EXPECT_PARSE_ERROR(Mock, "invalid .BTF.ext magic: 2a");
}

TEST(BTFParserTest, badBTFExtVersion) {
  MockData1 Mock;
  Mock.Ext.Header.Version = 42;
  EXPECT_PARSE_ERROR(Mock, "unsupported .BTF.ext version: 42");
}

TEST(BTFParserTest, badBTFExtHdrLen) {
  MockData1 Mock1, Mock2;

  Mock1.Ext.Header.HdrLen = 5;
  EXPECT_PARSE_ERROR(Mock1, "unexpected .BTF.ext header length: 5");

  Mock2.Ext.Header.HdrLen = sizeof(Mock2.Ext);
  EXPECT_PARSE_ERROR(Mock2, BTFExtEndOfData);
}

TEST(BTFParserTest, badBTFExtSectionLen) {
  MockData1 Mock1, Mock2, Mock3;

  // Cut-off header before HdrLen
  Mock1.ExtSectionLen = offsetof(BTF::ExtHeader, HdrLen);
  EXPECT_PARSE_ERROR(Mock1, BTFExtEndOfData);

  // Cut-off header before LineInfoLen
  Mock2.ExtSectionLen = offsetof(BTF::ExtHeader, LineInfoLen);
  EXPECT_PARSE_ERROR(Mock2, BTFExtEndOfData);

  // Cut-off line-info section somewhere in the middle
  Mock3.ExtSectionLen = offsetof(MockData1::E, Lines) + 4;
  EXPECT_PARSE_ERROR(Mock3, BTFExtEndOfData);
}

TEST(BTFParserTest, badBTFExtLineInfoRecSize) {
  MockData1 Mock1, Mock2;

  Mock1.Ext.Lines.LineRecSize = 2;
  EXPECT_PARSE_ERROR(Mock1, "unexpected .BTF.ext line info record length: 2");

  Mock2.Ext.Lines.LineRecSize = sizeof(Mock2.Ext.Lines.Foo.Lines[0]) + 1;
  EXPECT_PARSE_ERROR(Mock2, BTFExtEndOfData);
}

TEST(BTFParserTest, badBTFExtLineSectionName) {
  MockData1 Mock1;

  Mock1.Ext.Lines.Foo.Sec.SecNameOff = offsetof(MockData1::B::S, Buz);
  EXPECT_PARSE_ERROR(
      Mock1, "can't find section 'buz' while parsing .BTF.ext line info");
}

TEST(BTFParserTest, missingSections) {
  MockData1 Mock1, Mock2, Mock3;

  Mock1.BTFSectionLen = -1;
  EXPECT_PARSE_ERROR(Mock1, "can't find .BTF section");
  EXPECT_FALSE(BTFParser::hasBTFSections(Mock1.makeObj()));

  Mock2.ExtSectionLen = -1;
  EXPECT_PARSE_ERROR(Mock2, "can't find .BTF.ext section");
  EXPECT_FALSE(BTFParser::hasBTFSections(Mock2.makeObj()));

  EXPECT_TRUE(BTFParser::hasBTFSections(Mock3.makeObj()));
}

// Check that BTFParser instance is reset when BTFParser::parse() is
// called several times.
TEST(BTFParserTest, parserReset) {
  BTFParser BTF;
  MockData1 Mock1, Mock2;

  EXPECT_FALSE(BTF.parse(Mock1.makeObj()));
  EXPECT_TRUE(BTF.findLineInfo({16, 1}));
  EXPECT_TRUE(BTF.findLineInfo({0, 2}));

  // Break the reference to "bar" section name, thus making
  // information about "bar" line numbers unavailable.
  Mock2.Ext.Lines.Bar.Sec.SecNameOff = 100500;

  EXPECT_FALSE(BTF.parse(Mock2.makeObj()));
  EXPECT_TRUE(BTF.findLineInfo({16, 1}));
  // Make sure that "bar" no longer available (its index is 2).
  EXPECT_FALSE(BTF.findLineInfo({0, 2}));
}

TEST(BTFParserTest, btfContext) {
  MockData1 Mock;
  BTFParser BTF;
  std::unique_ptr<BTFContext> Ctx = BTFContext::create(Mock.makeObj());

  DILineInfo I1 = Ctx->getLineInfoForAddress({16, 1});
  EXPECT_EQ(I1.Line, 7u);
  EXPECT_EQ(I1.Column, 1u);
  EXPECT_EQ(I1.FileName, "a.c");
  EXPECT_EQ(I1.LineSource, "first line");

  DILineInfo I2 = Ctx->getLineInfoForAddress({24, 1});
  EXPECT_EQ(I2.Line, 0u);
  EXPECT_EQ(I2.Column, 0u);
  EXPECT_EQ(I2.FileName, DILineInfo::BadString);
  EXPECT_EQ(I2.LineSource, std::nullopt);
}

} // namespace
