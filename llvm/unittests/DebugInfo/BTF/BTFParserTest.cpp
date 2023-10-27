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
#define ASSERT_SUCCEEDED(E) ASSERT_THAT_ERROR((E), Succeeded())

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
    // No types.
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
    // No func info.
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

  // Invalid offset.
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

  // No info for insn address.
  EXPECT_FALSE(BTF.findLineInfo({24, 1}));
  EXPECT_FALSE(BTF.findLineInfo({8, 2}));
  // No info for section number.
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
  // "foo" line info should be corrupted.
  EXPECT_FALSE(BTF.findLineInfo({16, 1}));
  // "bar" line info should be ok.
  EXPECT_TRUE(BTF.findLineInfo({0, 2}));
}

// Keep this as macro to preserve line number info.
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

  // Cut-off string section by one byte.
  Mock1.BTFSectionLen =
      offsetof(MockData1::B, Strings) + sizeof(MockData1::B::S) - 1;
  EXPECT_PARSE_ERROR(Mock1, "invalid .BTF section size");

  // Cut-off header.
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

  // Cut-off header before HdrLen.
  Mock1.ExtSectionLen = offsetof(BTF::ExtHeader, HdrLen);
  EXPECT_PARSE_ERROR(Mock1, BTFExtEndOfData);

  // Cut-off header before LineInfoLen.
  Mock2.ExtSectionLen = offsetof(BTF::ExtHeader, LineInfoLen);
  EXPECT_PARSE_ERROR(Mock2, BTFExtEndOfData);

  // Cut-off line-info section somewhere in the middle.
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

static uint32_t mkInfo(uint32_t Kind) { return Kind << 24; }

template <typename T> static void append(std::string &S, const T &What) {
  S.append((const char *)&What, sizeof(What));
}

class MockData2 {
  SmallString<0> ObjStorage;
  std::unique_ptr<ObjectFile> Obj;
  std::string Types;
  std::string Strings;
  std::string Relocs;
  std::string Lines;
  unsigned TotalTypes;
  int LastRelocSecIdx;
  unsigned NumRelocs;
  int LastLineSecIdx;
  unsigned NumLines;

public:
  MockData2() { reset(); }

  unsigned totalTypes() const { return TotalTypes; }

  uint32_t addString(StringRef S) {
    uint32_t Off = Strings.size();
    Strings.append(S.data(), S.size());
    Strings.append("\0", 1);
    return Off;
  };

  uint32_t addType(const BTF::CommonType &Tp) {
    append(Types, Tp);
    return ++TotalTypes;
  }

  template <typename T> void addTail(const T &Tp) { append(Types, Tp); }

  void resetTypes() {
    Types.resize(0);
    TotalTypes = 0;
  }

  void reset() {
    ObjStorage.clear();
    Types.resize(0);
    Strings.resize(0);
    Relocs.resize(0);
    Lines.resize(0);
    TotalTypes = 0;
    LastRelocSecIdx = -1;
    NumRelocs = 0;
    LastLineSecIdx = -1;
    NumLines = 0;
  }

  void finishRelocSec() {
    if (LastRelocSecIdx == -1)
      return;

    BTF::SecFieldReloc *SecInfo =
        (BTF::SecFieldReloc *)&Relocs[LastRelocSecIdx];
    SecInfo->NumFieldReloc = NumRelocs;
    LastRelocSecIdx = -1;
    NumRelocs = 0;
  }

  void finishLineSec() {
    if (LastLineSecIdx == -1)
      return;

    BTF::SecLineInfo *SecInfo = (BTF::SecLineInfo *)&Lines[LastLineSecIdx];
    SecInfo->NumLineInfo = NumLines;
    NumLines = 0;
    LastLineSecIdx = -1;
  }

  void addRelocSec(const BTF::SecFieldReloc &R) {
    finishRelocSec();
    LastRelocSecIdx = Relocs.size();
    append(Relocs, R);
  }

  void addReloc(const BTF::BPFFieldReloc &R) {
    append(Relocs, R);
    ++NumRelocs;
  }

  void addLinesSec(const BTF::SecLineInfo &R) {
    finishLineSec();
    LastLineSecIdx = Lines.size();
    append(Lines, R);
  }

  void addLine(const BTF::BPFLineInfo &R) {
    append(Lines, R);
    ++NumLines;
  }

  ObjectFile &makeObj() {
    finishRelocSec();
    finishLineSec();

    BTF::Header BTFHeader = {};
    BTFHeader.Magic = BTF::MAGIC;
    BTFHeader.Version = 1;
    BTFHeader.HdrLen = sizeof(BTFHeader);
    BTFHeader.StrOff = 0;
    BTFHeader.StrLen = Strings.size();
    BTFHeader.TypeOff = Strings.size();
    BTFHeader.TypeLen = Types.size();

    std::string BTFSec;
    append(BTFSec, BTFHeader);
    BTFSec.append(Strings);
    BTFSec.append(Types);

    BTF::ExtHeader ExtHeader = {};
    ExtHeader.Magic = BTF::MAGIC;
    ExtHeader.Version = 1;
    ExtHeader.HdrLen = sizeof(ExtHeader);
    ExtHeader.FieldRelocOff = 0;
    ExtHeader.FieldRelocLen = Relocs.size() + sizeof(uint32_t);
    ExtHeader.LineInfoOff = ExtHeader.FieldRelocLen;
    ExtHeader.LineInfoLen = Lines.size() + sizeof(uint32_t);

    std::string ExtSec;
    append(ExtSec, ExtHeader);
    append(ExtSec, (uint32_t)sizeof(BTF::BPFFieldReloc));
    ExtSec.append(Relocs);
    append(ExtSec, (uint32_t)sizeof(BTF::BPFLineInfo));
    ExtSec.append(Lines);

    std::string YamlBuffer;
    raw_string_ostream Yaml(YamlBuffer);
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
    Size:     0x80
  - Name:     bar
    Type:     SHT_PROGBITS
    Size:     0x80
  - Name:     .BTF
    Type:     SHT_PROGBITS
    Content: )"
         << makeBinRef(BTFSec.data(), BTFSec.size());
    Yaml << R"(
  - Name:     .BTF.ext
    Type:     SHT_PROGBITS
    Content: )"
         << makeBinRef(ExtSec.data(), ExtSec.size());

    Obj = yaml::yaml2ObjectFile(ObjStorage, YamlBuffer,
                                [](const Twine &Err) { errs() << Err; });
    return *Obj.get();
  }
};

TEST(BTFParserTest, allTypeKinds) {
  MockData2 D;
  D.addType({D.addString("1"), mkInfo(BTF::BTF_KIND_INT), {4}});
  D.addTail((uint32_t)0);
  D.addType({D.addString("2"), mkInfo(BTF::BTF_KIND_PTR), {1}});
  D.addType({D.addString("3"), mkInfo(BTF::BTF_KIND_ARRAY), {0}});
  D.addTail(BTF::BTFArray({1, 1, 2}));
  D.addType({D.addString("4"), mkInfo(BTF::BTF_KIND_STRUCT) | 2, {8}});
  D.addTail(BTF::BTFMember({D.addString("a"), 1, 0}));
  D.addTail(BTF::BTFMember({D.addString("b"), 1, 0}));
  D.addType({D.addString("5"), mkInfo(BTF::BTF_KIND_UNION) | 3, {8}});
  D.addTail(BTF::BTFMember({D.addString("a"), 1, 0}));
  D.addTail(BTF::BTFMember({D.addString("b"), 1, 0}));
  D.addTail(BTF::BTFMember({D.addString("c"), 1, 0}));
  D.addType({D.addString("6"), mkInfo(BTF::BTF_KIND_ENUM) | 2, {4}});
  D.addTail(BTF::BTFEnum({D.addString("U"), 1}));
  D.addTail(BTF::BTFEnum({D.addString("V"), 2}));
  D.addType({D.addString("7"), mkInfo(BTF::BTF_KIND_ENUM64) | 1, {4}});
  D.addTail(BTF::BTFEnum64({D.addString("W"), 0, 1}));
  D.addType(
      {D.addString("8"), BTF::FWD_UNION_FLAG | mkInfo(BTF::BTF_KIND_FWD), {0}});
  D.addType({D.addString("9"), mkInfo(BTF::BTF_KIND_TYPEDEF), {1}});
  D.addType({D.addString("10"), mkInfo(BTF::BTF_KIND_VOLATILE), {1}});
  D.addType({D.addString("11"), mkInfo(BTF::BTF_KIND_CONST), {1}});
  D.addType({D.addString("12"), mkInfo(BTF::BTF_KIND_RESTRICT), {1}});
  D.addType({D.addString("13"), mkInfo(BTF::BTF_KIND_FUNC_PROTO) | 1, {1}});
  D.addTail(BTF::BTFParam({D.addString("P"), 2}));
  D.addType({D.addString("14"), mkInfo(BTF::BTF_KIND_FUNC), {13}});
  D.addType({D.addString("15"), mkInfo(BTF::BTF_KIND_VAR), {2}});
  D.addTail((uint32_t)0);
  D.addType({D.addString("16"), mkInfo(BTF::BTF_KIND_DATASEC) | 3, {0}});
  D.addTail(BTF::BTFDataSec({1, 0, 4}));
  D.addTail(BTF::BTFDataSec({1, 4, 4}));
  D.addTail(BTF::BTFDataSec({1, 8, 4}));
  D.addType({D.addString("17"), mkInfo(BTF::BTF_KIND_FLOAT), {4}});
  D.addType({D.addString("18"), mkInfo(BTF::BTF_KIND_DECL_TAG), {0}});
  D.addTail((uint32_t)-1);
  D.addType({D.addString("19"), mkInfo(BTF::BTF_KIND_TYPE_TAG), {0}});

  BTFParser BTF;
  Error Err = BTF.parse(D.makeObj());
  EXPECT_FALSE(Err);

  EXPECT_EQ(D.totalTypes() + 1 /* +1 for void */, BTF.typesCount());
  for (unsigned Id = 1; Id < D.totalTypes() + 1; ++Id) {
    const BTF::CommonType *Tp = BTF.findType(Id);
    ASSERT_TRUE(Tp);
    std::string IdBuf;
    raw_string_ostream IdBufStream(IdBuf);
    IdBufStream << Id;
    EXPECT_EQ(BTF.findString(Tp->NameOff), IdBuf);
  }
}

TEST(BTFParserTest, bigStruct) {
  const uint32_t N = 1000u;
  MockData2 D;
  uint32_t FStr = D.addString("f");
  D.addType({D.addString("foo"), mkInfo(BTF::BTF_KIND_INT), {4}});
  D.addTail((uint32_t)0);
  D.addType({D.addString("big"), mkInfo(BTF::BTF_KIND_STRUCT) | N, {8}});
  for (unsigned I = 0; I < N; ++I)
    D.addTail(BTF::BTFMember({FStr, 1, 0}));
  D.addType({D.addString("bar"), mkInfo(BTF::BTF_KIND_INT), {4}});
  D.addTail((uint32_t)0);

  BTFParser BTF;
  ASSERT_SUCCEEDED(BTF.parse(D.makeObj()));
  ASSERT_EQ(BTF.typesCount(), 4u /* +1 for void */);
  const BTF::CommonType *Foo = BTF.findType(1);
  const BTF::CommonType *Big = BTF.findType(2);
  const BTF::CommonType *Bar = BTF.findType(3);
  ASSERT_TRUE(Foo);
  ASSERT_TRUE(Big);
  ASSERT_TRUE(Bar);
  EXPECT_EQ(BTF.findString(Foo->NameOff), "foo");
  EXPECT_EQ(BTF.findString(Big->NameOff), "big");
  EXPECT_EQ(BTF.findString(Bar->NameOff), "bar");
  EXPECT_EQ(Big->getVlen(), N);
}

TEST(BTFParserTest, incompleteTypes) {
  MockData2 D;
  auto IncompleteType = [&](const BTF::CommonType &Tp) {
    D.resetTypes();
    D.addType(Tp);
    EXPECT_PARSE_ERROR(D, "incomplete type definition in .BTF section");
  };

  // All kinds that need tail.
  IncompleteType({D.addString("a"), mkInfo(BTF::BTF_KIND_INT), {4}});
  IncompleteType({D.addString("b"), mkInfo(BTF::BTF_KIND_ARRAY), {0}});
  IncompleteType({D.addString("c"), mkInfo(BTF::BTF_KIND_VAR), {0}});
  IncompleteType({D.addString("d"), mkInfo(BTF::BTF_KIND_DECL_TAG), {0}});

  // All kinds with vlen.
  IncompleteType({D.addString("a"), mkInfo(BTF::BTF_KIND_STRUCT) | 2, {8}});
  IncompleteType({D.addString("b"), mkInfo(BTF::BTF_KIND_UNION) | 3, {8}});
  IncompleteType({D.addString("c"), mkInfo(BTF::BTF_KIND_ENUM) | 2, {4}});
  IncompleteType({D.addString("d"), mkInfo(BTF::BTF_KIND_ENUM64) | 1, {4}});
  IncompleteType({D.addString("e"), mkInfo(BTF::BTF_KIND_FUNC_PROTO) | 1, {1}});
  IncompleteType({D.addString("f"), mkInfo(BTF::BTF_KIND_DATASEC) | 3, {0}});

  // An unexpected tail.
  D.resetTypes();
  D.addTail((uint32_t)0);
  EXPECT_PARSE_ERROR(D, "incomplete type definition in .BTF section");
}

// Use macro to preserve line number in error message.
#define SYMBOLIZE(SecAddr, Expected)                                           \
  do {                                                                         \
    const BTF::BPFFieldReloc *R = BTF.findFieldReloc((SecAddr));               \
    ASSERT_TRUE(R);                                                            \
    SmallString<64> Symbolized;                                                \
    BTF.symbolize(R, Symbolized);                                              \
    EXPECT_EQ(Symbolized, (Expected));                                         \
  } while (false)

// Shorter name for initializers below.
using SA = SectionedAddress;

TEST(BTFParserTest, typeRelocs) {
  MockData2 D;
  uint32_t Zero = D.addString("0");
  // id 1: struct foo {}
  // id 2: union bar;
  // id 3: struct buz;
  D.addType({D.addString("foo"), mkInfo(BTF::BTF_KIND_STRUCT), {0}});
  D.addType({D.addString("bar"),
             mkInfo(BTF::BTF_KIND_FWD) | BTF::FWD_UNION_FLAG,
             {0}});
  D.addType({D.addString("buz"), mkInfo(BTF::BTF_KIND_FWD), {0}});
  D.addRelocSec({D.addString("foo"), 7});
  // List of all possible correct type relocations for type id #1.
  D.addReloc({0, 1, Zero, BTF::BTF_TYPE_ID_LOCAL});
  D.addReloc({8, 1, Zero, BTF::BTF_TYPE_ID_REMOTE});
  D.addReloc({16, 1, Zero, BTF::TYPE_EXISTENCE});
  D.addReloc({24, 1, Zero, BTF::TYPE_MATCH});
  D.addReloc({32, 1, Zero, BTF::TYPE_SIZE});
  // Forward declarations.
  D.addReloc({40, 2, Zero, BTF::TYPE_SIZE});
  D.addReloc({48, 3, Zero, BTF::TYPE_SIZE});
  // Incorrect type relocation: bad type id.
  D.addReloc({56, 42, Zero, BTF::TYPE_SIZE});
  // Incorrect type relocation: spec should be '0'.
  D.addReloc({64, 1, D.addString("10"), BTF::TYPE_SIZE});

  BTFParser BTF;
  Error E = BTF.parse(D.makeObj());
  EXPECT_FALSE(E);

  SYMBOLIZE(SA({0, 1}), "<local_type_id> [1] struct foo");
  SYMBOLIZE(SA({8, 1}), "<target_type_id> [1] struct foo");
  SYMBOLIZE(SA({16, 1}), "<type_exists> [1] struct foo");
  SYMBOLIZE(SA({24, 1}), "<type_matches> [1] struct foo");
  SYMBOLIZE(SA({32, 1}), "<type_size> [1] struct foo");
  SYMBOLIZE(SA({40, 1}), "<type_size> [2] fwd union bar");
  SYMBOLIZE(SA({48, 1}), "<type_size> [3] fwd struct buz");
  SYMBOLIZE(SA({56, 1}), "<type_size> [42] '0' <unknown type id: 42>");
  SYMBOLIZE(SA({64, 1}),
            "<type_size> [1] '10' "
            "<unexpected type-based relocation spec: should be '0'>");
}

TEST(BTFParserTest, enumRelocs) {
  MockData2 D;
  // id 1: enum { U, V }
  D.addType({D.addString("foo"), mkInfo(BTF::BTF_KIND_ENUM) | 2, {4}});
  D.addTail(BTF::BTFEnum({D.addString("U"), 1}));
  D.addTail(BTF::BTFEnum({D.addString("V"), 2}));
  // id 2: int
  D.addType({D.addString("int"), mkInfo(BTF::BTF_KIND_INT), {4}});
  D.addTail((uint32_t)0);
  // id 3: enum: uint64_t { A, B }
  D.addType({D.addString("bar"), mkInfo(BTF::BTF_KIND_ENUM64) | 2, {8}});
  D.addTail(BTF::BTFEnum64({D.addString("A"), 1, 0}));
  D.addTail(BTF::BTFEnum64({D.addString("B"), 2, 0}));

  D.addRelocSec({D.addString("foo"), 5});
  // An ok relocation accessing value #1: U.
  D.addReloc({0, 1, D.addString("0"), BTF::ENUM_VALUE_EXISTENCE});
  // An ok relocation accessing value #2: V.
  D.addReloc({8, 1, D.addString("1"), BTF::ENUM_VALUE});
  // Incorrect relocation: too many elements in string "1:0".
  D.addReloc({16, 1, D.addString("1:0"), BTF::ENUM_VALUE});
  // Incorrect relocation: type id "2" is not an enum.
  D.addReloc({24, 2, D.addString("1"), BTF::ENUM_VALUE});
  // Incorrect relocation: value #42 does not exist for enum "foo".
  D.addReloc({32, 1, D.addString("42"), BTF::ENUM_VALUE});
  // An ok relocation accessing value #1: A.
  D.addReloc({40, 3, D.addString("0"), BTF::ENUM_VALUE_EXISTENCE});
  // An ok relocation accessing value #2: B.
  D.addReloc({48, 3, D.addString("1"), BTF::ENUM_VALUE});

  BTFParser BTF;
  Error E = BTF.parse(D.makeObj());
  EXPECT_FALSE(E);

  SYMBOLIZE(SA({0, 1}), "<enumval_exists> [1] enum foo::U = 1");
  SYMBOLIZE(SA({8, 1}), "<enumval_value> [1] enum foo::V = 2");
  SYMBOLIZE(
      SA({16, 1}),
      "<enumval_value> [1] '1:0' <unexpected enumval relocation spec size>");
  SYMBOLIZE(
      SA({24, 1}),
      "<enumval_value> [2] '1' <unexpected type kind for enum relocation: 1>");
  SYMBOLIZE(SA({32, 1}), "<enumval_value> [1] '42' <bad value index: 42>");
  SYMBOLIZE(SA({40, 1}), "<enumval_exists> [3] enum bar::A = 1");
  SYMBOLIZE(SA({48, 1}), "<enumval_value> [3] enum bar::B = 2");
}

TEST(BTFParserTest, enumRelocsMods) {
  MockData2 D;
  // id 1: enum { U, V }
  D.addType({D.addString("foo"), mkInfo(BTF::BTF_KIND_ENUM) | 2, {4}});
  D.addTail(BTF::BTFEnum({D.addString("U"), 1}));
  D.addTail(BTF::BTFEnum({D.addString("V"), 2}));
  // id 2: typedef enum foo a;
  D.addType({D.addString("a"), mkInfo(BTF::BTF_KIND_TYPEDEF), {1}});
  // id 3: const enum foo;
  D.addType({D.addString(""), mkInfo(BTF::BTF_KIND_CONST), {1}});

  D.addRelocSec({D.addString("foo"), 0});
  D.addReloc({0, 2, D.addString("0"), BTF::ENUM_VALUE});
  D.addReloc({8, 3, D.addString("1"), BTF::ENUM_VALUE});

  BTFParser BTF;
  Error E = BTF.parse(D.makeObj());
  EXPECT_FALSE(E);

  SYMBOLIZE(SA({0, 1}), "<enumval_value> [2] typedef a::U = 1");
  SYMBOLIZE(SA({8, 1}), "<enumval_value> [3] const enum foo::V = 2");
}

TEST(BTFParserTest, fieldRelocs) {
  MockData2 D;
  // id 1: int
  D.addType({D.addString("int"), mkInfo(BTF::BTF_KIND_INT), {4}});
  D.addTail((uint32_t)0);
  // id 2: struct foo { int a; int b; }
  D.addType({D.addString("foo"), mkInfo(BTF::BTF_KIND_STRUCT) | 2, {8}});
  D.addTail(BTF::BTFMember({D.addString("a"), 1, 0}));
  D.addTail(BTF::BTFMember({D.addString("b"), 1, 0}));
  // id 3: array of struct foo.
  D.addType({D.addString(""), mkInfo(BTF::BTF_KIND_ARRAY), {0}});
  D.addTail(BTF::BTFArray({2, 1, 2}));
  // id 4: struct bar { struct foo u[2]; int v; }
  D.addType({D.addString("bar"), mkInfo(BTF::BTF_KIND_STRUCT) | 2, {8}});
  D.addTail(BTF::BTFMember({D.addString("u"), 3, 0}));
  D.addTail(BTF::BTFMember({D.addString("v"), 1, 0}));
  // id 5: array with bad element type id.
  D.addType({D.addString(""), mkInfo(BTF::BTF_KIND_ARRAY), {0}});
  D.addTail(BTF::BTFArray({42, 1, 2}));
  // id 6: struct buz { <bad type> u[2]; <bad type> v; }
  D.addType({D.addString("bar"), mkInfo(BTF::BTF_KIND_STRUCT) | 2, {8}});
  D.addTail(BTF::BTFMember({D.addString("u"), 5, 0}));
  D.addTail(BTF::BTFMember({D.addString("v"), 42, 0}));

  D.addRelocSec({D.addString("foo"), 0 /* patched automatically */});
  // All field relocations kinds for struct bar::v.
  D.addReloc({0, 4, D.addString("0:1"), BTF::FIELD_BYTE_OFFSET});
  D.addReloc({8, 4, D.addString("0:1"), BTF::FIELD_BYTE_SIZE});
  D.addReloc({16, 4, D.addString("0:1"), BTF::FIELD_EXISTENCE});
  D.addReloc({24, 4, D.addString("0:1"), BTF::FIELD_SIGNEDNESS});
  D.addReloc({32, 4, D.addString("0:1"), BTF::FIELD_LSHIFT_U64});
  D.addReloc({40, 4, D.addString("0:1"), BTF::FIELD_RSHIFT_U64});
  // Non-zero first idx.
  D.addReloc({48, 4, D.addString("7:1"), BTF::FIELD_BYTE_OFFSET});
  // Access through array and struct: struct bar::u[1].a.
  D.addReloc({56, 4, D.addString("0:0:1:0"), BTF::FIELD_BYTE_OFFSET});
  // Access through array and struct: struct bar::u[1].b.
  D.addReloc({64, 4, D.addString("0:0:1:1"), BTF::FIELD_BYTE_OFFSET});
  // Incorrect relocation: empty access string.
  D.addReloc({72, 4, D.addString(""), BTF::FIELD_BYTE_OFFSET});
  // Incorrect relocation: member index out of range (only two members in bar).
  D.addReloc({80, 4, D.addString("0:2"), BTF::FIELD_BYTE_OFFSET});
  // Incorrect relocation: unknown element type id (buz::u[0] access).
  D.addReloc({88, 6, D.addString("0:0:0"), BTF::FIELD_BYTE_OFFSET});
  // Incorrect relocation: unknown member type id (buz::v access).
  D.addReloc({96, 6, D.addString("0:1:0"), BTF::FIELD_BYTE_OFFSET});
  // Incorrect relocation: non structural type in the middle of access string
  //   struct bar::v.<something>.
  D.addReloc({104, 4, D.addString("0:1:0"), BTF::FIELD_BYTE_OFFSET});

  BTFParser BTF;
  Error E = BTF.parse(D.makeObj());
  EXPECT_FALSE(E);

  SYMBOLIZE(SA({0, 1}), "<byte_off> [4] struct bar::v (0:1)");
  SYMBOLIZE(SA({8, 1}), "<byte_sz> [4] struct bar::v (0:1)");
  SYMBOLIZE(SA({16, 1}), "<field_exists> [4] struct bar::v (0:1)");
  SYMBOLIZE(SA({24, 1}), "<signed> [4] struct bar::v (0:1)");
  SYMBOLIZE(SA({32, 1}), "<lshift_u64> [4] struct bar::v (0:1)");
  SYMBOLIZE(SA({40, 1}), "<rshift_u64> [4] struct bar::v (0:1)");
  SYMBOLIZE(SA({48, 1}), "<byte_off> [4] struct bar::[7].v (7:1)");
  SYMBOLIZE(SA({56, 1}), "<byte_off> [4] struct bar::u[1].a (0:0:1:0)");
  SYMBOLIZE(SA({64, 1}), "<byte_off> [4] struct bar::u[1].b (0:0:1:1)");
  SYMBOLIZE(SA({72, 1}), "<byte_off> [4] '' <field spec too short>");
  SYMBOLIZE(SA({80, 1}),
            "<byte_off> [4] '0:2' "
            "<member index 2 for spec sub-string 1 is out of range>");
  SYMBOLIZE(SA({88, 1}), "<byte_off> [6] '0:0:0' "
                         "<unknown element type id 42 for spec sub-string 2>");
  SYMBOLIZE(SA({96, 1}), "<byte_off> [6] '0:1:0' "
                         "<unknown member type id 42 for spec sub-string 1>");
  SYMBOLIZE(SA({104, 1}), "<byte_off> [4] '0:1:0' "
                          "<unexpected type kind 1 for spec sub-string 2>");
}

TEST(BTFParserTest, fieldRelocsMods) {
  MockData2 D;
  // struct foo {
  //   int u;
  // }
  // typedef struct foo bar;
  // struct buz {
  //   const bar v;
  // }
  // typedef buz quux;
  // const volatile restrict quux <some-var>;
  uint32_t Int =
      D.addType({D.addString("int"), mkInfo(BTF::BTF_KIND_INT), {4}});
  D.addTail((uint32_t)0);
  uint32_t Foo =
      D.addType({D.addString("foo"), mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});
  D.addTail(BTF::BTFMember({D.addString("u"), Int, 0}));
  uint32_t Bar =
      D.addType({D.addString("bar"), mkInfo(BTF::BTF_KIND_TYPEDEF), {Foo}});
  uint32_t CBar =
      D.addType({D.addString("bar"), mkInfo(BTF::BTF_KIND_CONST), {Bar}});
  uint32_t Buz =
      D.addType({D.addString("buz"), mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});
  D.addTail(BTF::BTFMember({D.addString("v"), CBar, 0}));
  uint32_t Quux =
      D.addType({D.addString("quux"), mkInfo(BTF::BTF_KIND_TYPEDEF), {Buz}});
  uint32_t RQuux =
      D.addType({D.addString(""), mkInfo(BTF::BTF_KIND_RESTRICT), {Quux}});
  uint32_t VRQuux =
      D.addType({D.addString(""), mkInfo(BTF::BTF_KIND_VOLATILE), {RQuux}});
  uint32_t CVRQuux =
      D.addType({D.addString(""), mkInfo(BTF::BTF_KIND_CONST), {VRQuux}});
  uint32_t CUnknown =
      D.addType({D.addString(""), mkInfo(BTF::BTF_KIND_CONST), {77}});
  uint32_t CVUnknown =
      D.addType({D.addString(""), mkInfo(BTF::BTF_KIND_VOLATILE), {CUnknown}});

  D.addRelocSec({D.addString("foo"), 0});
  D.addReloc({0, Bar, D.addString("0:0"), BTF::FIELD_BYTE_OFFSET});
  D.addReloc({8, CVRQuux, D.addString("0:0:0"), BTF::FIELD_BYTE_OFFSET});
  D.addReloc({16, CVUnknown, D.addString("0:1:2"), BTF::FIELD_BYTE_OFFSET});

  BTFParser BTF;
  Error E = BTF.parse(D.makeObj());
  EXPECT_FALSE(E);

  // Should show modifiers / name of typedef.
  SYMBOLIZE(SA({0, 1}), "<byte_off> [3] typedef bar::u (0:0)");
  SYMBOLIZE(SA({8, 1}),
            "<byte_off> [9] const volatile restrict typedef quux::v.u (0:0:0)");
  SYMBOLIZE(SA({16, 1}),
            "<byte_off> [11] '0:1:2' <unknown type id: 77 in modifiers chain>");
}

TEST(BTFParserTest, relocTypeTagAndVoid) {
  MockData2 D;
  // __attribute__((type_tag("tag"))) void
  uint32_t Tag =
      D.addType({D.addString("tag"), mkInfo(BTF::BTF_KIND_TYPE_TAG), {0}});

  D.addRelocSec({D.addString("foo"), 0});
  D.addReloc({0, Tag, D.addString("0"), BTF::TYPE_EXISTENCE});
  D.addReloc({8, 0 /* void */, D.addString("0"), BTF::TYPE_EXISTENCE});

  BTFParser BTF;
  Error E = BTF.parse(D.makeObj());
  EXPECT_FALSE(E);

  SYMBOLIZE(SA({0, 1}), "<type_exists> [1] type_tag(\"tag\") void");
  SYMBOLIZE(SA({8, 1}), "<type_exists> [0] void");
}

TEST(BTFParserTest, longRelocModifiersCycle) {
  MockData2 D;

  D.addType(
      {D.addString(""), mkInfo(BTF::BTF_KIND_CONST), {1 /* ourselves */}});
  D.addRelocSec({D.addString("foo"), 0});
  D.addReloc({0, 1, D.addString(""), BTF::TYPE_EXISTENCE});

  BTFParser BTF;
  Error E = BTF.parse(D.makeObj());
  EXPECT_FALSE(E);

  SYMBOLIZE(SA({0, 1}), "<type_exists> [1] '' <modifiers chain is too long>");
}

TEST(BTFParserTest, relocAnonFieldsAndTypes) {
  MockData2 D;

  // struct {
  //   int :32;
  // } v;
  uint32_t Int =
      D.addType({D.addString("int"), mkInfo(BTF::BTF_KIND_INT), {4}});
  D.addTail((uint32_t)0);
  uint32_t Anon =
      D.addType({D.addString(""), mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});
  D.addTail(BTF::BTFMember({D.addString(""), Int, 0}));

  D.addRelocSec({D.addString("foo"), 0});
  D.addReloc({0, Anon, D.addString("0"), BTF::TYPE_EXISTENCE});
  D.addReloc({8, Anon, D.addString("0:0"), BTF::FIELD_BYTE_OFFSET});

  BTFParser BTF;
  Error E = BTF.parse(D.makeObj());
  EXPECT_FALSE(E);

  SYMBOLIZE(SA({0, 1}), "<type_exists> [2] struct <anon 2>");
  SYMBOLIZE(SA({8, 1}), "<byte_off> [2] struct <anon 2>::<anon 0> (0:0)");
}

TEST(BTFParserTest, miscBadRelos) {
  MockData2 D;

  uint32_t S = D.addType({D.addString("S"), mkInfo(BTF::BTF_KIND_STRUCT), {0}});

  D.addRelocSec({D.addString("foo"), 0});
  D.addReloc({0, 0, D.addString(""), 777});
  D.addReloc({8, S, D.addString("abc"), BTF::FIELD_BYTE_OFFSET});
  D.addReloc({16, S, D.addString("0#"), BTF::FIELD_BYTE_OFFSET});

  BTFParser BTF;
  Error E = BTF.parse(D.makeObj());
  EXPECT_FALSE(E);

  SYMBOLIZE(SA({0, 1}),
            "<reloc kind #777> [0] '' <unknown relocation kind: 777>");
  SYMBOLIZE(SA({8, 1}), "<byte_off> [1] 'abc' <spec string is not a number>");
  SYMBOLIZE(SA({16, 1}),
            "<byte_off> [1] '0#' <unexpected spec string delimiter: '#'>");
}

TEST(BTFParserTest, relocsMultipleSections) {
  MockData2 D;

  uint32_t S = D.addType({D.addString("S"), mkInfo(BTF::BTF_KIND_STRUCT), {0}});
  uint32_t T = D.addType({D.addString("T"), mkInfo(BTF::BTF_KIND_STRUCT), {0}});

  D.addRelocSec({D.addString("foo"), 0});
  D.addReloc({0, S, D.addString(""), BTF::TYPE_EXISTENCE});
  D.addReloc({8, S, D.addString(""), BTF::TYPE_EXISTENCE});

  D.addRelocSec({D.addString("bar"), 0});
  D.addReloc({8, T, D.addString(""), BTF::TYPE_EXISTENCE});
  D.addReloc({16, T, D.addString(""), BTF::TYPE_EXISTENCE});

  BTFParser BTF;
  Error E = BTF.parse(D.makeObj());
  EXPECT_FALSE(E);

  EXPECT_TRUE(BTF.findFieldReloc({0, 1}));
  EXPECT_TRUE(BTF.findFieldReloc({8, 1}));
  EXPECT_FALSE(BTF.findFieldReloc({16, 1}));

  EXPECT_FALSE(BTF.findFieldReloc({0, 2}));
  EXPECT_TRUE(BTF.findFieldReloc({8, 2}));
  EXPECT_TRUE(BTF.findFieldReloc({16, 2}));

  EXPECT_FALSE(BTF.findFieldReloc({0, 3}));
  EXPECT_FALSE(BTF.findFieldReloc({8, 3}));
  EXPECT_FALSE(BTF.findFieldReloc({16, 3}));

  auto AssertReloType = [&](const SectionedAddress &A, const char *Name) {
    const BTF::BPFFieldReloc *Relo = BTF.findFieldReloc(A);
    ASSERT_TRUE(Relo);
    const BTF::CommonType *Type = BTF.findType(Relo->TypeID);
    ASSERT_TRUE(Type);
    EXPECT_EQ(BTF.findString(Type->NameOff), Name);
  };

  AssertReloType({8, 1}, "S");
  AssertReloType({8, 2}, "T");
}

TEST(BTFParserTest, parserResetReloAndTypes) {
  BTFParser BTF;
  MockData2 D;

  // First time: two types, two relocations.
  D.addType({D.addString("foo"), mkInfo(BTF::BTF_KIND_STRUCT), {0}});
  D.addType({D.addString("bar"), mkInfo(BTF::BTF_KIND_STRUCT), {0}});
  D.addRelocSec({D.addString("foo"), 0});
  D.addReloc({0, 1, D.addString(""), BTF::TYPE_EXISTENCE});
  D.addReloc({8, 2, D.addString(""), BTF::TYPE_EXISTENCE});

  Error E1 = BTF.parse(D.makeObj());
  EXPECT_FALSE(E1);

  ASSERT_TRUE(BTF.findType(1));
  EXPECT_EQ(BTF.findString(BTF.findType(1)->NameOff), "foo");
  EXPECT_TRUE(BTF.findType(2));
  EXPECT_TRUE(BTF.findFieldReloc({0, 1}));
  EXPECT_TRUE(BTF.findFieldReloc({8, 1}));

  // Second time: one type, one relocation.
  D.reset();
  D.addType({D.addString("buz"), mkInfo(BTF::BTF_KIND_STRUCT), {0}});
  D.addRelocSec({D.addString("foo"), 0});
  D.addReloc({0, 1, D.addString(""), BTF::TYPE_EXISTENCE});

  Error E2 = BTF.parse(D.makeObj());
  EXPECT_FALSE(E2);

  ASSERT_TRUE(BTF.findType(1));
  EXPECT_EQ(BTF.findString(BTF.findType(1)->NameOff), "buz");
  EXPECT_FALSE(BTF.findType(2));
  EXPECT_TRUE(BTF.findFieldReloc({0, 1}));
  EXPECT_FALSE(BTF.findFieldReloc({8, 1}));
}

TEST(BTFParserTest, selectiveLoad) {
  BTFParser BTF1, BTF2, BTF3;
  MockData2 D;

  D.addType({D.addString("foo"), mkInfo(BTF::BTF_KIND_STRUCT), {0}});
  D.addRelocSec({D.addString("foo"), 0});
  D.addReloc({0, 1, D.addString(""), BTF::TYPE_EXISTENCE});
  D.addLinesSec({D.addString("foo"), 0});
  D.addLine({0, D.addString("file.c"), D.addString("some line"), LC(2, 3)});

  BTFParser::ParseOptions Opts;

  ObjectFile &Obj1 = D.makeObj();
  Opts = {};
  Opts.LoadLines = true;
  ASSERT_SUCCEEDED(BTF1.parse(Obj1, Opts));

  Opts = {};
  Opts.LoadTypes = true;
  ASSERT_SUCCEEDED(BTF2.parse(Obj1, Opts));

  Opts = {};
  Opts.LoadRelocs = true;
  ASSERT_SUCCEEDED(BTF3.parse(Obj1, Opts));

  EXPECT_TRUE(BTF1.findLineInfo({0, 1}));
  EXPECT_FALSE(BTF2.findLineInfo({0, 1}));
  EXPECT_FALSE(BTF3.findLineInfo({0, 1}));

  EXPECT_FALSE(BTF1.findType(1));
  EXPECT_TRUE(BTF2.findType(1));
  EXPECT_FALSE(BTF3.findType(1));

  EXPECT_FALSE(BTF1.findFieldReloc({0, 1}));
  EXPECT_FALSE(BTF2.findFieldReloc({0, 1}));
  EXPECT_TRUE(BTF3.findFieldReloc({0, 1}));
}

} // namespace
