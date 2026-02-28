//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/BTF/BTFBuilder.h"
#include "llvm/DebugInfo/BTF/BTFParser.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace llvm::object;

#define ASSERT_SUCCEEDED(E) ASSERT_THAT_ERROR((E), Succeeded())

static uint32_t mkInfo(uint32_t Kind) { return Kind << 24; }

static raw_ostream &operator<<(raw_ostream &OS, const yaml::BinaryRef &Ref) {
  Ref.writeAsHex(OS);
  return OS;
}

static yaml::BinaryRef makeBinRef(const void *Ptr, size_t Size) {
  return yaml::BinaryRef(
      ArrayRef<uint8_t>(static_cast<const uint8_t *>(Ptr), Size));
}

// Wrap raw BTF bytes in an ELF ObjectFile for BTFParser verification.
// Includes a minimal empty .BTF.ext section (required by BTFParser).
static std::unique_ptr<ObjectFile>
makeELFWithBTF(const SmallVectorImpl<uint8_t> &BTFData,
               SmallString<0> &Storage) {
  // Build a minimal .BTF.ext section (just the header, no subsections).
  BTF::ExtHeader ExtHdr = {};
  ExtHdr.Magic = BTF::MAGIC;
  ExtHdr.Version = 1;
  ExtHdr.HdrLen = sizeof(BTF::ExtHeader);

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
  - Name:     .BTF
    Type:     SHT_PROGBITS
    Content: )"
       << makeBinRef(BTFData.data(), BTFData.size());
  Yaml << R"(
  - Name:     .BTF.ext
    Type:     SHT_PROGBITS
    Content: )"
       << makeBinRef(&ExtHdr, sizeof(ExtHdr));

  return yaml::yaml2ObjectFile(Storage, YamlBuffer,
                               [](const Twine &Err) { errs() << Err; });
}

namespace {

TEST(BTFBuilderTest, emptyBuilder) {
  BTFBuilder B;
  EXPECT_EQ(B.typesCount(), 0u);
  EXPECT_EQ(B.findType(0), nullptr);
  EXPECT_EQ(B.findType(1), nullptr);
  EXPECT_EQ(B.findString(0), "");
}

TEST(BTFBuilderTest, addStringAndType) {
  BTFBuilder B;

  uint32_t FooOff = B.addString("foo");
  uint32_t BarOff = B.addString("bar");
  EXPECT_EQ(B.findString(FooOff), "foo");
  EXPECT_EQ(B.findString(BarOff), "bar");
  EXPECT_EQ(B.findString(0), "");

  // Add INT type: int foo, 4 bytes.
  uint32_t Id = B.addType({FooOff, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0); // INT encoding
  EXPECT_EQ(Id, 1u);
  EXPECT_EQ(B.typesCount(), 1u);

  const BTF::CommonType *T = B.findType(1);
  ASSERT_TRUE(T);
  EXPECT_EQ(T->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(T->Size, 4u);
  EXPECT_EQ(B.findString(T->NameOff), "foo");

  // Add PTR type pointing to type 1.
  uint32_t Id2 = B.addType({BarOff, mkInfo(BTF::BTF_KIND_PTR), {1}});
  EXPECT_EQ(Id2, 2u);
  EXPECT_EQ(B.typesCount(), 2u);

  const BTF::CommonType *T2 = B.findType(2);
  ASSERT_TRUE(T2);
  EXPECT_EQ(T2->getKind(), BTF::BTF_KIND_PTR);
  EXPECT_EQ(T2->Type, 1u);
}

TEST(BTFBuilderTest, typeByteSize) {
  BTFBuilder B;
  uint32_t S = B.addString("s");

  // INT: CommonType + uint32_t = 12 + 4 = 16
  B.addType({S, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  EXPECT_EQ(B.getTypeBytes(1).size(), 16u);

  // PTR: CommonType only = 12
  B.addType({S, mkInfo(BTF::BTF_KIND_PTR), {1}});
  EXPECT_EQ(B.getTypeBytes(2).size(), 12u);

  // STRUCT with 2 members: 12 + 2*12 = 36
  B.addType({S, mkInfo(BTF::BTF_KIND_STRUCT) | 2, {8}});
  B.addTail(BTF::BTFMember({S, 1, 0}));
  B.addTail(BTF::BTFMember({S, 1, 32}));
  EXPECT_EQ(B.getTypeBytes(3).size(), 36u);

  // ARRAY: 12 + 12 = 24
  B.addType({S, mkInfo(BTF::BTF_KIND_ARRAY), {0}});
  B.addTail(BTF::BTFArray({1, 1, 10}));
  EXPECT_EQ(B.getTypeBytes(4).size(), 24u);
}

TEST(BTFBuilderTest, writeAndParseRoundtrip) {
  BTFBuilder B;

  // Build a small BTF with various types.
  uint32_t IntName = B.addString("int");
  uint32_t FooName = B.addString("foo");
  uint32_t AName = B.addString("a");
  uint32_t BName = B.addString("b");

  // Type 1: int, 4 bytes
  B.addType({IntName, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  // Type 2: struct foo { int a; int b; }
  B.addType({FooName, mkInfo(BTF::BTF_KIND_STRUCT) | 2, {8}});
  B.addTail(BTF::BTFMember({AName, 1, 0}));
  B.addTail(BTF::BTFMember({BName, 1, 32}));

  // Type 3: pointer to struct foo
  B.addType({0, mkInfo(BTF::BTF_KIND_PTR), {2}});

  // Write to binary.
  SmallVector<uint8_t, 0> Output;
  B.write(Output, !sys::IsBigEndianHost);

  // Parse with BTFParser to verify.
  SmallString<0> Storage;
  auto Obj = makeELFWithBTF(Output, Storage);
  ASSERT_TRUE(Obj);

  BTFParser Parser;
  BTFParser::ParseOptions Opts;
  Opts.LoadTypes = true;
  ASSERT_SUCCEEDED(Parser.parse(*Obj, Opts));

  ASSERT_EQ(Parser.typesCount(), 4u); // 3 types + void

  // Verify INT.
  const BTF::CommonType *IntType = Parser.findType(1);
  ASSERT_TRUE(IntType);
  EXPECT_EQ(IntType->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(IntType->Size, 4u);
  EXPECT_EQ(Parser.findString(IntType->NameOff), "int");

  // Verify STRUCT.
  const BTF::CommonType *StructType = Parser.findType(2);
  ASSERT_TRUE(StructType);
  EXPECT_EQ(StructType->getKind(), BTF::BTF_KIND_STRUCT);
  EXPECT_EQ(StructType->getVlen(), 2u);
  EXPECT_EQ(StructType->Size, 8u);
  EXPECT_EQ(Parser.findString(StructType->NameOff), "foo");

  // Verify struct members via cast.
  auto *ST = dyn_cast<BTF::StructType>(StructType);
  ASSERT_TRUE(ST);
  EXPECT_EQ(Parser.findString(ST->members()[0].NameOff), "a");
  EXPECT_EQ(ST->members()[0].Type, 1u);
  EXPECT_EQ(Parser.findString(ST->members()[1].NameOff), "b");
  EXPECT_EQ(ST->members()[1].Type, 1u);

  // Verify PTR.
  const BTF::CommonType *PtrType = Parser.findType(3);
  ASSERT_TRUE(PtrType);
  EXPECT_EQ(PtrType->getKind(), BTF::BTF_KIND_PTR);
  EXPECT_EQ(PtrType->Type, 2u);
}

TEST(BTFBuilderTest, mergeTwo) {
  // Build first BTF section.
  BTFBuilder B1;
  uint32_t IntName1 = B1.addString("int");
  B1.addType({IntName1, mkInfo(BTF::BTF_KIND_INT), {4}});
  B1.addTail((uint32_t)0);

  SmallVector<uint8_t, 0> Blob1;
  B1.write(Blob1, !sys::IsBigEndianHost);

  // Build second BTF section.
  BTFBuilder B2;
  uint32_t LongName2 = B2.addString("long");
  // Type 1 in B2: long, 8 bytes.
  B2.addType({LongName2, mkInfo(BTF::BTF_KIND_INT), {8}});
  B2.addTail((uint32_t)0);
  // Type 2 in B2: ptr to long (type 1).
  B2.addType({0, mkInfo(BTF::BTF_KIND_PTR), {1}});

  SmallVector<uint8_t, 0> Blob2;
  B2.write(Blob2, !sys::IsBigEndianHost);

  // Merge both into a new builder.
  BTFBuilder Merged;
  auto FirstId1 = Merged.merge(
      StringRef(reinterpret_cast<const char *>(Blob1.data()), Blob1.size()),
      !sys::IsBigEndianHost);
  ASSERT_SUCCEEDED(FirstId1.takeError());
  EXPECT_EQ(*FirstId1, 1u);
  EXPECT_EQ(Merged.typesCount(), 1u);

  auto FirstId2 = Merged.merge(
      StringRef(reinterpret_cast<const char *>(Blob2.data()), Blob2.size()),
      !sys::IsBigEndianHost);
  ASSERT_SUCCEEDED(FirstId2.takeError());
  EXPECT_EQ(*FirstId2, 2u);
  EXPECT_EQ(Merged.typesCount(), 3u);

  // Type 1: int from first blob.
  const BTF::CommonType *T1 = Merged.findType(1);
  ASSERT_TRUE(T1);
  EXPECT_EQ(T1->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(T1->Size, 4u);
  EXPECT_EQ(Merged.findString(T1->NameOff), "int");

  // Type 2: long from second blob.
  const BTF::CommonType *T2 = Merged.findType(2);
  ASSERT_TRUE(T2);
  EXPECT_EQ(T2->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(T2->Size, 8u);
  EXPECT_EQ(Merged.findString(T2->NameOff), "long");

  // Type 3: ptr to type 2 (remapped from type 1 in second blob).
  const BTF::CommonType *T3 = Merged.findType(3);
  ASSERT_TRUE(T3);
  EXPECT_EQ(T3->getKind(), BTF::BTF_KIND_PTR);
  EXPECT_EQ(T3->Type, 2u); // Remapped: was 1, now 1+1=2

  // Verify roundtrip through BTFParser.
  SmallVector<uint8_t, 0> MergedOutput;
  Merged.write(MergedOutput, !sys::IsBigEndianHost);

  SmallString<0> Storage;
  auto Obj = makeELFWithBTF(MergedOutput, Storage);
  ASSERT_TRUE(Obj);

  BTFParser Parser;
  BTFParser::ParseOptions Opts;
  Opts.LoadTypes = true;
  ASSERT_SUCCEEDED(Parser.parse(*Obj, Opts));
  EXPECT_EQ(Parser.typesCount(), 4u); // 3 types + void
}

TEST(BTFBuilderTest, mergeStructWithMembers) {
  // Build BTF with struct that references other types.
  BTFBuilder B1;
  uint32_t IntName = B1.addString("int");
  uint32_t FooName = B1.addString("foo");
  uint32_t XName = B1.addString("x");
  uint32_t YName = B1.addString("y");

  // Type 1: int
  B1.addType({IntName, mkInfo(BTF::BTF_KIND_INT), {4}});
  B1.addTail((uint32_t)0);
  // Type 2: struct foo { int x; int y; }
  B1.addType({FooName, mkInfo(BTF::BTF_KIND_STRUCT) | 2, {8}});
  B1.addTail(BTF::BTFMember({XName, 1, 0}));
  B1.addTail(BTF::BTFMember({YName, 1, 32}));
  // Type 3: ptr to void
  B1.addType({0, mkInfo(BTF::BTF_KIND_PTR), {0}});

  SmallVector<uint8_t, 0> Blob1;
  B1.write(Blob1, !sys::IsBigEndianHost);

  // Merge into a builder that already has one type.
  BTFBuilder Merged;
  uint32_t PreName = Merged.addString("pre");
  Merged.addType({PreName, mkInfo(BTF::BTF_KIND_FLOAT), {4}});
  EXPECT_EQ(Merged.typesCount(), 1u);

  auto FirstId = Merged.merge(
      StringRef(reinterpret_cast<const char *>(Blob1.data()), Blob1.size()),
      !sys::IsBigEndianHost);
  ASSERT_SUCCEEDED(FirstId.takeError());
  EXPECT_EQ(*FirstId, 2u); // First new type from merge
  EXPECT_EQ(Merged.typesCount(), 4u);

  // Type 1: pre-existing FLOAT
  EXPECT_EQ(Merged.findType(1)->getKind(), BTF::BTF_KIND_FLOAT);

  // Type 2: int (remapped from id 1)
  const BTF::CommonType *IntT = Merged.findType(2);
  ASSERT_TRUE(IntT);
  EXPECT_EQ(IntT->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(Merged.findString(IntT->NameOff), "int");

  // Type 3: struct foo with member types remapped to 2
  const BTF::CommonType *StructT = Merged.findType(3);
  ASSERT_TRUE(StructT);
  EXPECT_EQ(StructT->getKind(), BTF::BTF_KIND_STRUCT);
  EXPECT_EQ(Merged.findString(StructT->NameOff), "foo");
  auto *ST = dyn_cast<BTF::StructType>(StructT);
  ASSERT_TRUE(ST);
  EXPECT_EQ(Merged.findString(ST->members()[0].NameOff), "x");
  EXPECT_EQ(ST->members()[0].Type, 2u); // Was 1, remapped to 2
  EXPECT_EQ(Merged.findString(ST->members()[1].NameOff), "y");
  EXPECT_EQ(ST->members()[1].Type, 2u); // Was 1, remapped to 2

  // Type 4: ptr to void (type 0 stays 0, not remapped)
  const BTF::CommonType *PtrT = Merged.findType(4);
  ASSERT_TRUE(PtrT);
  EXPECT_EQ(PtrT->getKind(), BTF::BTF_KIND_PTR);
  EXPECT_EQ(PtrT->Type, 0u); // void stays 0
}

TEST(BTFBuilderTest, mergeAllKinds) {
  BTFBuilder B;

  // Build a BTF with all type kinds.
  uint32_t S = B.addString("t");
  uint32_t M = B.addString("m");

  B.addType({S, mkInfo(BTF::BTF_KIND_INT), {4}});       // 1
  B.addTail((uint32_t)0);
  B.addType({S, mkInfo(BTF::BTF_KIND_PTR), {1}});        // 2
  B.addType({S, mkInfo(BTF::BTF_KIND_ARRAY), {0}});      // 3
  B.addTail(BTF::BTFArray({1, 1, 10}));
  B.addType({S, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}}); // 4
  B.addTail(BTF::BTFMember({M, 1, 0}));
  B.addType({S, mkInfo(BTF::BTF_KIND_UNION) | 1, {4}});  // 5
  B.addTail(BTF::BTFMember({M, 1, 0}));
  B.addType({S, mkInfo(BTF::BTF_KIND_ENUM) | 1, {4}});   // 6
  B.addTail(BTF::BTFEnum({M, 42}));
  B.addType({S, mkInfo(BTF::BTF_KIND_FWD), {0}});        // 7
  B.addType({S, mkInfo(BTF::BTF_KIND_TYPEDEF), {1}});    // 8
  B.addType({S, mkInfo(BTF::BTF_KIND_VOLATILE), {1}});   // 9
  B.addType({S, mkInfo(BTF::BTF_KIND_CONST), {1}});      // 10
  B.addType({S, mkInfo(BTF::BTF_KIND_RESTRICT), {1}});   // 11
  B.addType({S, mkInfo(BTF::BTF_KIND_FUNC_PROTO) | 1, {1}}); // 12
  B.addTail(BTF::BTFParam({M, 1}));
  B.addType({S, mkInfo(BTF::BTF_KIND_FUNC), {12}});      // 13
  B.addType({S, mkInfo(BTF::BTF_KIND_VAR), {1}});        // 14
  B.addTail((uint32_t)0); // linkage
  B.addType({S, mkInfo(BTF::BTF_KIND_DATASEC) | 1, {0}}); // 15
  B.addTail(BTF::BTFDataSec({14, 0, 4}));
  B.addType({S, mkInfo(BTF::BTF_KIND_FLOAT), {4}});      // 16
  B.addType({S, mkInfo(BTF::BTF_KIND_DECL_TAG), {1}});   // 17
  B.addTail((uint32_t)-1); // component_idx
  B.addType({S, mkInfo(BTF::BTF_KIND_TYPE_TAG), {1}});   // 18
  B.addType({S, mkInfo(BTF::BTF_KIND_ENUM64) | 1, {8}}); // 19
  B.addTail(BTF::BTFEnum64({M, 1, 0}));

  EXPECT_EQ(B.typesCount(), 19u);

  // Write and parse back.
  SmallVector<uint8_t, 0> Output;
  B.write(Output, !sys::IsBigEndianHost);

  SmallString<0> Storage;
  auto Obj = makeELFWithBTF(Output, Storage);
  ASSERT_TRUE(Obj);

  BTFParser Parser;
  BTFParser::ParseOptions Opts;
  Opts.LoadTypes = true;
  ASSERT_SUCCEEDED(Parser.parse(*Obj, Opts));
  EXPECT_EQ(Parser.typesCount(), 20u); // 19 + void

  // Verify all types were parsed correctly.
  for (uint32_t Id = 1; Id <= 19; ++Id) {
    ASSERT_TRUE(Parser.findType(Id))
        << "Type " << Id << " not found after roundtrip";
  }

  // Spot-check a few.
  EXPECT_EQ(Parser.findType(1)->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(Parser.findType(2)->getKind(), BTF::BTF_KIND_PTR);
  EXPECT_EQ(Parser.findType(3)->getKind(), BTF::BTF_KIND_ARRAY);
  EXPECT_EQ(Parser.findType(6)->getKind(), BTF::BTF_KIND_ENUM);
  EXPECT_EQ(Parser.findType(12)->getKind(), BTF::BTF_KIND_FUNC_PROTO);
  EXPECT_EQ(Parser.findType(19)->getKind(), BTF::BTF_KIND_ENUM64);
}

TEST(BTFBuilderTest, mergeInvalidBTF) {
  BTFBuilder B;

  // Too small.
  EXPECT_THAT_ERROR(B.merge("", !sys::IsBigEndianHost).takeError(),
                    FailedWithMessage(testing::HasSubstr("too small")));

  // Bad magic.
  SmallVector<uint8_t, 0> BadMagic(sizeof(BTF::Header), 0);
  EXPECT_THAT_ERROR(
      B.merge(StringRef(reinterpret_cast<const char *>(BadMagic.data()),
                        BadMagic.size()),
              !sys::IsBigEndianHost)
          .takeError(),
      FailedWithMessage(testing::HasSubstr("invalid BTF magic")));
}

// Helper to build a raw BTF blob from a BTFBuilder.
static StringRef blobRef(const SmallVectorImpl<uint8_t> &V) {
  return StringRef(reinterpret_cast<const char *>(V.data()), V.size());
}

TEST(BTFBuilderTest, invalidTypeIdLookups) {
  BTFBuilder B;
  uint32_t S = B.addString("x");
  B.addType({S, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  // ID 0 (void) returns nullptr / empty.
  EXPECT_EQ(B.findType(0), nullptr);
  EXPECT_TRUE(B.getTypeBytes(0).empty());
  EXPECT_TRUE(B.getMutableTypeBytes(0).empty());

  // ID beyond range returns nullptr / empty.
  EXPECT_EQ(B.findType(2), nullptr);
  EXPECT_TRUE(B.getTypeBytes(2).empty());
  EXPECT_TRUE(B.getMutableTypeBytes(2).empty());
  EXPECT_EQ(B.findType(UINT32_MAX), nullptr);
}

TEST(BTFBuilderTest, getMutableTypeBytesIsWritable) {
  BTFBuilder B;
  uint32_t S = B.addString("int");
  B.addType({S, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0x00000020); // bits=32

  // Read the INT encoding via mutable bytes and change it.
  MutableArrayRef<uint8_t> Bytes = B.getMutableTypeBytes(1);
  ASSERT_EQ(Bytes.size(), 16u);
  auto *T = reinterpret_cast<BTF::CommonType *>(Bytes.data());
  EXPECT_EQ(T->Size, 4u);

  // Mutate: change size to 8.
  T->Size = 8;

  // Verify mutation is visible through findType.
  const BTF::CommonType *T2 = B.findType(1);
  EXPECT_EQ(T2->Size, 8u);
}

TEST(BTFBuilderTest, findStringOutOfBounds) {
  BTFBuilder B;
  B.addString("hello");
  // Offset 0 is always the empty string.
  EXPECT_EQ(B.findString(0), "");
  // Far past the end.
  EXPECT_TRUE(B.findString(99999).empty());
}

TEST(BTFBuilderTest, hasTypeRefAllKinds) {
  // Types that use Type (not Size) in the CommonType union.
  EXPECT_TRUE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_PTR));
  EXPECT_TRUE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_TYPEDEF));
  EXPECT_TRUE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_VOLATILE));
  EXPECT_TRUE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_CONST));
  EXPECT_TRUE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_RESTRICT));
  EXPECT_TRUE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_FUNC));
  EXPECT_TRUE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_FUNC_PROTO));
  EXPECT_TRUE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_VAR));
  EXPECT_TRUE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_DECL_TAG));
  EXPECT_TRUE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_TYPE_TAG));

  // Types that use Size (not a type reference).
  EXPECT_FALSE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_INT));
  EXPECT_FALSE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_ARRAY));
  EXPECT_FALSE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_STRUCT));
  EXPECT_FALSE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_UNION));
  EXPECT_FALSE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_ENUM));
  EXPECT_FALSE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_ENUM64));
  EXPECT_FALSE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_FWD));
  EXPECT_FALSE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_FLOAT));
  EXPECT_FALSE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_DATASEC));
  EXPECT_FALSE(BTFBuilder::hasTypeRef(BTF::BTF_KIND_UNKN));
}

TEST(BTFBuilderTest, typeByteSizeAllKinds) {
  BTFBuilder B;
  uint32_t S = B.addString("s");
  uint32_t M = B.addString("m");

  // ENUM with 2 values: 12 + 2*8 = 28
  B.addType({S, mkInfo(BTF::BTF_KIND_ENUM) | 2, {4}});  // 1
  B.addTail(BTF::BTFEnum({M, 0}));
  B.addTail(BTF::BTFEnum({M, 1}));
  EXPECT_EQ(B.getTypeBytes(1).size(), 28u);

  // ENUM64 with 1 value: 12 + 1*12 = 24
  B.addType({S, mkInfo(BTF::BTF_KIND_ENUM64) | 1, {8}});  // 2
  B.addTail(BTF::BTFEnum64({M, 0, 0}));
  EXPECT_EQ(B.getTypeBytes(2).size(), 24u);

  // FUNC_PROTO with 2 params: 12 + 2*8 = 28
  B.addType({0, mkInfo(BTF::BTF_KIND_FUNC_PROTO) | 2, {0}});  // 3
  B.addTail(BTF::BTFParam({M, 1}));
  B.addTail(BTF::BTFParam({M, 1}));
  EXPECT_EQ(B.getTypeBytes(3).size(), 28u);

  // DATASEC with 1 entry: 12 + 1*12 = 24
  B.addType({S, mkInfo(BTF::BTF_KIND_DATASEC) | 1, {0}});  // 4
  B.addTail(BTF::BTFDataSec({1, 0, 4}));
  EXPECT_EQ(B.getTypeBytes(4).size(), 24u);

  // VAR: 12 + 4 = 16
  B.addType({S, mkInfo(BTF::BTF_KIND_VAR), {1}});  // 5
  B.addTail((uint32_t)0);
  EXPECT_EQ(B.getTypeBytes(5).size(), 16u);

  // DECL_TAG: 12 + 4 = 16
  B.addType({S, mkInfo(BTF::BTF_KIND_DECL_TAG), {1}});  // 6
  B.addTail((uint32_t)-1);
  EXPECT_EQ(B.getTypeBytes(6).size(), 16u);

  // FLOAT: CommonType only = 12
  B.addType({S, mkInfo(BTF::BTF_KIND_FLOAT), {4}});  // 7
  EXPECT_EQ(B.getTypeBytes(7).size(), 12u);

  // FWD: CommonType only = 12
  B.addType({S, mkInfo(BTF::BTF_KIND_FWD), {0}});  // 8
  EXPECT_EQ(B.getTypeBytes(8).size(), 12u);

  // TYPEDEF: CommonType only = 12
  B.addType({S, mkInfo(BTF::BTF_KIND_TYPEDEF), {1}});  // 9
  EXPECT_EQ(B.getTypeBytes(9).size(), 12u);

  // VOLATILE: CommonType only = 12
  B.addType({S, mkInfo(BTF::BTF_KIND_VOLATILE), {1}});  // 10
  EXPECT_EQ(B.getTypeBytes(10).size(), 12u);

  // CONST: CommonType only = 12
  B.addType({S, mkInfo(BTF::BTF_KIND_CONST), {1}});  // 11
  EXPECT_EQ(B.getTypeBytes(11).size(), 12u);

  // RESTRICT: CommonType only = 12
  B.addType({S, mkInfo(BTF::BTF_KIND_RESTRICT), {1}});  // 12
  EXPECT_EQ(B.getTypeBytes(12).size(), 12u);

  // FUNC: CommonType only = 12
  B.addType({S, mkInfo(BTF::BTF_KIND_FUNC), {3}});  // 13
  EXPECT_EQ(B.getTypeBytes(13).size(), 12u);

  // TYPE_TAG: CommonType only = 12
  B.addType({S, mkInfo(BTF::BTF_KIND_TYPE_TAG), {1}});  // 14
  EXPECT_EQ(B.getTypeBytes(14).size(), 12u);

  // UNION with 1 member: 12 + 1*12 = 24
  B.addType({S, mkInfo(BTF::BTF_KIND_UNION) | 1, {4}});  // 15
  B.addTail(BTF::BTFMember({M, 1, 0}));
  EXPECT_EQ(B.getTypeBytes(15).size(), 24u);
}

TEST(BTFBuilderTest, mergeBadVersion) {
  // Build a valid BTF section but with version=2.
  BTFBuilder Tmp;
  uint32_t S = Tmp.addString("x");
  Tmp.addType({S, mkInfo(BTF::BTF_KIND_INT), {4}});
  Tmp.addTail((uint32_t)0);
  SmallVector<uint8_t, 0> Blob;
  Tmp.write(Blob, !sys::IsBigEndianHost);

  // Corrupt the version byte (offset 2 in the header).
  Blob[2] = 99;

  BTFBuilder B;
  EXPECT_THAT_ERROR(B.merge(blobRef(Blob), !sys::IsBigEndianHost).takeError(),
                    FailedWithMessage(testing::HasSubstr("unsupported BTF version")));
  EXPECT_EQ(B.typesCount(), 0u);
}

TEST(BTFBuilderTest, mergeStringBoundsExceeded) {
  // Build a valid BTF, then corrupt str_off+str_len to exceed section size.
  BTFBuilder Tmp;
  uint32_t S = Tmp.addString("x");
  Tmp.addType({S, mkInfo(BTF::BTF_KIND_INT), {4}});
  Tmp.addTail((uint32_t)0);
  SmallVector<uint8_t, 0> Blob;
  Tmp.write(Blob, !sys::IsBigEndianHost);

  // Corrupt StrLen: at offset 20 (uint32_t), set it to a huge value.
  uint32_t HugeLen = 0xFFFFFF;
  memcpy(&Blob[20], &HugeLen, sizeof(HugeLen));

  BTFBuilder B;
  EXPECT_THAT_ERROR(B.merge(blobRef(Blob), !sys::IsBigEndianHost).takeError(),
                    FailedWithMessage(testing::HasSubstr("exceeds section bounds")));
  EXPECT_EQ(B.typesCount(), 0u);
}

TEST(BTFBuilderTest, mergeTypeBoundsExceeded) {
  // Corrupt type_len to exceed section size.
  BTFBuilder Tmp;
  uint32_t S = Tmp.addString("x");
  Tmp.addType({S, mkInfo(BTF::BTF_KIND_INT), {4}});
  Tmp.addTail((uint32_t)0);
  SmallVector<uint8_t, 0> Blob;
  Tmp.write(Blob, !sys::IsBigEndianHost);

  // Corrupt TypeLen: at offset 12 in header.
  uint32_t HugeLen = 0xFFFFFF;
  memcpy(&Blob[12], &HugeLen, sizeof(HugeLen));

  BTFBuilder B;
  EXPECT_THAT_ERROR(B.merge(blobRef(Blob), !sys::IsBigEndianHost).takeError(),
                    FailedWithMessage(testing::HasSubstr("exceeds section bounds")));
  EXPECT_EQ(B.typesCount(), 0u);
}

TEST(BTFBuilderTest, mergeIncompleteTypeRollback) {
  // Manually build a raw BTF blob where the type section contains a STRUCT
  // header claiming vlen=2 (needs 36 bytes), but only 28 bytes of type data.
  BTF::Header Hdr;
  Hdr.Magic = BTF::MAGIC;
  Hdr.Version = BTF::VERSION;
  Hdr.Flags = 0;
  Hdr.HdrLen = sizeof(BTF::Header);
  Hdr.TypeOff = 0;
  Hdr.TypeLen = 28; // Struct with 2 members needs 36, only 28 given
  Hdr.StrOff = 28;
  Hdr.StrLen = 3; // "\0x\0"

  SmallVector<uint8_t, 0> Blob(sizeof(Hdr) + 28 + 3, 0);
  memcpy(Blob.data(), &Hdr, sizeof(Hdr));

  // STRUCT header at start of type section.
  uint8_t *TypePtr = Blob.data() + sizeof(Hdr);
  BTF::CommonType CT;
  CT.NameOff = 1; // "x"
  CT.Info = mkInfo(BTF::BTF_KIND_STRUCT) | 2; // vlen=2
  CT.Size = 8;
  memcpy(TypePtr, &CT, sizeof(CT));
  // Write 1 member (12 bytes) — struct claims 2 but only space for ~1.
  BTF::BTFMember Mem = {1, 0, 0};
  memcpy(TypePtr + sizeof(CT), &Mem, sizeof(Mem));

  // String table: "\0x\0"
  uint8_t *StrPtr = Blob.data() + sizeof(Hdr) + 28;
  StrPtr[0] = 0;
  StrPtr[1] = 'x';
  StrPtr[2] = 0;

  // Pre-populate the builder, then attempt merge.
  BTFBuilder B;
  uint32_t Pre = B.addString("pre");
  B.addType({Pre, mkInfo(BTF::BTF_KIND_FLOAT), {4}});
  EXPECT_EQ(B.typesCount(), 1u);

  EXPECT_THAT_ERROR(
      B.merge(blobRef(Blob), !sys::IsBigEndianHost).takeError(),
      FailedWithMessage(testing::HasSubstr("incomplete type")));
  // Builder state should be rolled back — still just 1 type.
  EXPECT_EQ(B.typesCount(), 1u);
}

TEST(BTFBuilderTest, mergeArrayRemapsElemAndIndexType) {
  BTFBuilder Src;
  uint32_t IntS = Src.addString("int");
  // Type 1: int
  Src.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  Src.addTail((uint32_t)0);
  // Type 2: array of 10 ints, indexed by int
  Src.addType({0, mkInfo(BTF::BTF_KIND_ARRAY), {0}});
  Src.addTail(BTF::BTFArray({1, 1, 10}));

  SmallVector<uint8_t, 0> Blob;
  Src.write(Blob, !sys::IsBigEndianHost);

  // Merge into a builder that already has 1 type.
  BTFBuilder B;
  uint32_t Pre = B.addString("pre");
  B.addType({Pre, mkInfo(BTF::BTF_KIND_FLOAT), {4}});

  ASSERT_SUCCEEDED(B.merge(blobRef(Blob), !sys::IsBigEndianHost).takeError());
  EXPECT_EQ(B.typesCount(), 3u);

  // Array is type 3, int is type 2. Array's ElemType and IndexType should
  // be remapped from 1 to 2.
  const BTF::CommonType *ArrT = B.findType(3);
  ASSERT_TRUE(ArrT);
  EXPECT_EQ(ArrT->getKind(), BTF::BTF_KIND_ARRAY);
  auto *Arr = reinterpret_cast<const BTF::BTFArray *>(
      reinterpret_cast<const uint8_t *>(ArrT) + sizeof(BTF::CommonType));
  EXPECT_EQ(Arr->ElemType, 2u);
  EXPECT_EQ(Arr->IndexType, 2u);
  EXPECT_EQ(Arr->Nelems, 10u);
}

TEST(BTFBuilderTest, mergeFuncProtoRemapsParamTypes) {
  BTFBuilder Src;
  uint32_t IntS = Src.addString("int");
  uint32_t XS = Src.addString("x");
  // Type 1: int
  Src.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  Src.addTail((uint32_t)0);
  // Type 2: func_proto(int x) -> int
  Src.addType({0, mkInfo(BTF::BTF_KIND_FUNC_PROTO) | 1, {1}});
  Src.addTail(BTF::BTFParam({XS, 1}));

  SmallVector<uint8_t, 0> Blob;
  Src.write(Blob, !sys::IsBigEndianHost);

  BTFBuilder B;
  uint32_t Pre = B.addString("pre");
  B.addType({Pre, mkInfo(BTF::BTF_KIND_FLOAT), {4}});

  ASSERT_SUCCEEDED(B.merge(blobRef(Blob), !sys::IsBigEndianHost).takeError());
  EXPECT_EQ(B.typesCount(), 3u);

  // func_proto is type 3, returns type 2 (remapped int).
  const BTF::CommonType *FP = B.findType(3);
  ASSERT_TRUE(FP);
  EXPECT_EQ(FP->getKind(), BTF::BTF_KIND_FUNC_PROTO);
  EXPECT_EQ(FP->Type, 2u); // Return type remapped
  auto *Params = reinterpret_cast<const BTF::BTFParam *>(
      reinterpret_cast<const uint8_t *>(FP) + sizeof(BTF::CommonType));
  EXPECT_EQ(Params[0].Type, 2u); // Param type remapped
}

TEST(BTFBuilderTest, mergeDataSecRemapsVarTypes) {
  BTFBuilder Src;
  uint32_t IntS = Src.addString("int");
  uint32_t DS = Src.addString(".data");
  // Type 1: int
  Src.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  Src.addTail((uint32_t)0);
  // Type 2: VAR referencing int
  Src.addType({IntS, mkInfo(BTF::BTF_KIND_VAR), {1}});
  Src.addTail((uint32_t)1); // global linkage
  // Type 3: DATASEC with 1 var
  Src.addType({DS, mkInfo(BTF::BTF_KIND_DATASEC) | 1, {4}});
  Src.addTail(BTF::BTFDataSec({2, 0, 4}));

  SmallVector<uint8_t, 0> Blob;
  Src.write(Blob, !sys::IsBigEndianHost);

  BTFBuilder B;
  uint32_t Pre = B.addString("pre");
  B.addType({Pre, mkInfo(BTF::BTF_KIND_FLOAT), {4}});

  ASSERT_SUCCEEDED(B.merge(blobRef(Blob), !sys::IsBigEndianHost).takeError());
  EXPECT_EQ(B.typesCount(), 4u);

  // DATASEC is type 4, its var entry should point to type 3 (remapped VAR).
  const BTF::CommonType *DSec = B.findType(4);
  ASSERT_TRUE(DSec);
  EXPECT_EQ(DSec->getKind(), BTF::BTF_KIND_DATASEC);
  auto *Vars = reinterpret_cast<const BTF::BTFDataSec *>(
      reinterpret_cast<const uint8_t *>(DSec) + sizeof(BTF::CommonType));
  EXPECT_EQ(Vars[0].Type, 3u); // Remapped from 2 to 3
}

TEST(BTFBuilderTest, writeEmptyBuilder) {
  BTFBuilder B;
  SmallVector<uint8_t, 0> Output;
  B.write(Output, !sys::IsBigEndianHost);

  // Should produce a valid header with empty type and string sections.
  EXPECT_GE(Output.size(), sizeof(BTF::Header));
  BTF::Header Hdr;
  memcpy(&Hdr, Output.data(), sizeof(Hdr));
  EXPECT_EQ(Hdr.Magic, BTF::MAGIC);
  EXPECT_EQ(Hdr.TypeLen, 0u);
  // String table always has at least the empty string (\0).
  EXPECT_EQ(Hdr.StrLen, 1u);
}

} // namespace
