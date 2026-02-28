//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/BTF/BTFDedup.h"
#include "llvm/DebugInfo/BTF/BTFBuilder.h"
#include "llvm/DebugInfo/BTF/BTFParser.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
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

static std::unique_ptr<ObjectFile>
makeELFWithBTF(const SmallVectorImpl<uint8_t> &BTFData,
               SmallString<0> &Storage) {
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

TEST(BTFDedupTest, emptyDedup) {
  BTFBuilder B;
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 0u);
}

TEST(BTFDedupTest, singleType) {
  BTFBuilder B;
  uint32_t S = B.addString("int");
  B.addType({S, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 1u);
  EXPECT_EQ(B.findType(1)->getKind(), BTF::BTF_KIND_INT);
}

TEST(BTFDedupTest, duplicateInts) {
  BTFBuilder B;
  uint32_t S1 = B.addString("int");
  uint32_t S2 = B.addString("int"); // Same string, different offset

  // Two identical INT types.
  B.addType({S1, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  B.addType({S2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  EXPECT_EQ(B.typesCount(), 2u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 1u);

  const BTF::CommonType *T = B.findType(1);
  ASSERT_TRUE(T);
  EXPECT_EQ(T->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(T->Size, 4u);
  EXPECT_EQ(B.findString(T->NameOff), "int");
}

TEST(BTFDedupTest, differentInts) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t LongS = B.addString("long");

  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  B.addType({LongS, mkInfo(BTF::BTF_KIND_INT), {8}});
  B.addTail((uint32_t)0);

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 2u);
}

TEST(BTFDedupTest, duplicateEnums) {
  BTFBuilder B;
  uint32_t EnumS = B.addString("color");
  uint32_t RedS = B.addString("RED");
  uint32_t BlueS = B.addString("BLUE");

  // First enum
  B.addType({EnumS, mkInfo(BTF::BTF_KIND_ENUM) | 2, {4}});
  B.addTail(BTF::BTFEnum({RedS, 0}));
  B.addTail(BTF::BTFEnum({BlueS, 1}));

  // Duplicate enum (same content, different string offsets)
  uint32_t EnumS2 = B.addString("color");
  uint32_t RedS2 = B.addString("RED");
  uint32_t BlueS2 = B.addString("BLUE");
  B.addType({EnumS2, mkInfo(BTF::BTF_KIND_ENUM) | 2, {4}});
  B.addTail(BTF::BTFEnum({RedS2, 0}));
  B.addTail(BTF::BTFEnum({BlueS2, 1}));

  EXPECT_EQ(B.typesCount(), 2u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 1u);
}

TEST(BTFDedupTest, duplicatePtrTypes) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 2: duplicate int
  uint32_t IntS2 = B.addString("int");
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 3: ptr to type 1
  B.addType({0, mkInfo(BTF::BTF_KIND_PTR), {1}});
  // Type 4: ptr to type 2
  B.addType({0, mkInfo(BTF::BTF_KIND_PTR), {2}});

  EXPECT_EQ(B.typesCount(), 4u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  // Should dedup to: int + ptr
  EXPECT_EQ(B.typesCount(), 2u);

  EXPECT_EQ(B.findType(1)->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(B.findType(2)->getKind(), BTF::BTF_KIND_PTR);
  EXPECT_EQ(B.findType(2)->Type, 1u);
}

TEST(BTFDedupTest, duplicateStructs) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t FooS = B.addString("foo");
  uint32_t AS = B.addString("a");
  uint32_t BS = B.addString("b");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  // Type 2: struct foo { int a; int b; }
  B.addType({FooS, mkInfo(BTF::BTF_KIND_STRUCT) | 2, {8}});
  B.addTail(BTF::BTFMember({AS, 1, 0}));
  B.addTail(BTF::BTFMember({BS, 1, 32}));

  // Duplicate types (from another compilation unit):
  // Type 3: int (duplicate)
  uint32_t IntS2 = B.addString("int");
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  // Type 4: struct foo { int a; int b; } (duplicate, refs type 3)
  uint32_t FooS2 = B.addString("foo");
  uint32_t AS2 = B.addString("a");
  uint32_t BS2 = B.addString("b");
  B.addType({FooS2, mkInfo(BTF::BTF_KIND_STRUCT) | 2, {8}});
  B.addTail(BTF::BTFMember({AS2, 3, 0}));   // refs type 3 (dup int)
  B.addTail(BTF::BTFMember({BS2, 3, 32}));  // refs type 3 (dup int)

  EXPECT_EQ(B.typesCount(), 4u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  // Should dedup to: int + struct foo
  EXPECT_EQ(B.typesCount(), 2u);

  // Verify struct members reference the deduped int.
  const BTF::CommonType *ST = B.findType(2);
  ASSERT_TRUE(ST);
  EXPECT_EQ(ST->getKind(), BTF::BTF_KIND_STRUCT);
  auto *Members = reinterpret_cast<const BTF::BTFMember *>(
      reinterpret_cast<const uint8_t *>(ST) + sizeof(BTF::CommonType));
  EXPECT_EQ(Members[0].Type, 1u); // Remapped to new int ID
  EXPECT_EQ(Members[1].Type, 1u);
}

TEST(BTFDedupTest, selfReferentialStruct) {
  BTFBuilder B;
  uint32_t NodeS = B.addString("node");
  uint32_t DataS = B.addString("data");
  uint32_t NextS = B.addString("next");
  uint32_t IntS = B.addString("int");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 2: ptr to type 3 (struct node)
  B.addType({0, mkInfo(BTF::BTF_KIND_PTR), {3}});
  // Type 3: struct node { int data; struct node *next; }
  B.addType({NodeS, mkInfo(BTF::BTF_KIND_STRUCT) | 2, {16}});
  B.addTail(BTF::BTFMember({DataS, 1, 0}));
  B.addTail(BTF::BTFMember({NextS, 2, 64}));

  // Duplicate self-referential struct:
  // Type 4: int (dup)
  uint32_t IntS2 = B.addString("int");
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 5: ptr to type 6
  B.addType({0, mkInfo(BTF::BTF_KIND_PTR), {6}});
  // Type 6: struct node (dup, refs types 4 and 5)
  uint32_t NodeS2 = B.addString("node");
  uint32_t DataS2 = B.addString("data");
  uint32_t NextS2 = B.addString("next");
  B.addType({NodeS2, mkInfo(BTF::BTF_KIND_STRUCT) | 2, {16}});
  B.addTail(BTF::BTFMember({DataS2, 4, 0}));
  B.addTail(BTF::BTFMember({NextS2, 5, 64}));

  EXPECT_EQ(B.typesCount(), 6u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  // Should dedup to: int + ptr + struct node = 3 types
  EXPECT_EQ(B.typesCount(), 3u);
}

TEST(BTFDedupTest, fwdDeclResolution) {
  BTFBuilder B;
  uint32_t FooS = B.addString("foo");

  // Type 1: forward declaration of struct foo
  B.addType({FooS, mkInfo(BTF::BTF_KIND_FWD), {0}});

  // Type 2: full struct foo {}
  uint32_t FooS2 = B.addString("foo");
  B.addType({FooS2, mkInfo(BTF::BTF_KIND_STRUCT), {0}});

  // FWD and STRUCT are different kinds, so they shouldn't be deduped
  // by the basic algorithm (FWD resolution is a separate concern).
  // Both should survive dedup.
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 2u);
}

TEST(BTFDedupTest, funcProtoDedup) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t XS = B.addString("x");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  // Type 2: func_proto(int) -> int
  B.addType({0, mkInfo(BTF::BTF_KIND_FUNC_PROTO) | 1, {1}});
  B.addTail(BTF::BTFParam({XS, 1}));

  // Duplicate func_proto:
  // Type 3: int (dup)
  uint32_t IntS2 = B.addString("int");
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  // Type 4: func_proto(int) -> int (dup, refs type 3)
  uint32_t XS2 = B.addString("x");
  B.addType({0, mkInfo(BTF::BTF_KIND_FUNC_PROTO) | 1, {3}});
  B.addTail(BTF::BTFParam({XS2, 3}));

  EXPECT_EQ(B.typesCount(), 4u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  // Should dedup to: int + func_proto
  EXPECT_EQ(B.typesCount(), 2u);
}

TEST(BTFDedupTest, roundtripAfterDedup) {
  BTFBuilder B;

  // Build types with duplicates.
  uint32_t IntS = B.addString("int");
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  uint32_t IntS2 = B.addString("int");
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  B.addType({0, mkInfo(BTF::BTF_KIND_PTR), {1}});
  B.addType({0, mkInfo(BTF::BTF_KIND_PTR), {2}});

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 2u);

  // Write to binary and verify with BTFParser.
  SmallVector<uint8_t, 0> Output;
  B.write(Output, !sys::IsBigEndianHost);

  SmallString<0> Storage;
  auto Obj = makeELFWithBTF(Output, Storage);
  ASSERT_TRUE(Obj);

  BTFParser Parser;
  BTFParser::ParseOptions Opts;
  Opts.LoadTypes = true;
  ASSERT_SUCCEEDED(Parser.parse(*Obj, Opts));
  EXPECT_EQ(Parser.typesCount(), 3u); // 2 types + void

  EXPECT_EQ(Parser.findType(1)->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(Parser.findType(2)->getKind(), BTF::BTF_KIND_PTR);
  EXPECT_EQ(Parser.findType(2)->Type, 1u);
}

TEST(BTFDedupTest, mergedBlobDedup) {
  // Simulate what lld would do: merge two .BTF sections, then dedup.

  // First "object file" BTF.
  BTFBuilder B1;
  uint32_t IntS1 = B1.addString("int");
  B1.addType({IntS1, mkInfo(BTF::BTF_KIND_INT), {4}});
  B1.addTail((uint32_t)0);
  uint32_t FooS1 = B1.addString("foo");
  uint32_t XS1 = B1.addString("x");
  B1.addType({FooS1, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});
  B1.addTail(BTF::BTFMember({XS1, 1, 0}));

  SmallVector<uint8_t, 0> Blob1;
  B1.write(Blob1, !sys::IsBigEndianHost);

  // Second "object file" BTF (same types, different IDs).
  BTFBuilder B2;
  uint32_t IntS2 = B2.addString("int");
  B2.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B2.addTail((uint32_t)0);
  uint32_t FooS2 = B2.addString("foo");
  uint32_t XS2 = B2.addString("x");
  B2.addType({FooS2, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});
  B2.addTail(BTF::BTFMember({XS2, 1, 0}));

  SmallVector<uint8_t, 0> Blob2;
  B2.write(Blob2, !sys::IsBigEndianHost);

  // Merge.
  BTFBuilder Merged;
  ASSERT_SUCCEEDED(
      Merged.merge(StringRef(reinterpret_cast<const char *>(Blob1.data()),
                             Blob1.size()),
                   !sys::IsBigEndianHost)
          .takeError());
  ASSERT_SUCCEEDED(
      Merged.merge(StringRef(reinterpret_cast<const char *>(Blob2.data()),
                             Blob2.size()),
                   !sys::IsBigEndianHost)
          .takeError());
  EXPECT_EQ(Merged.typesCount(), 4u); // 2 from each blob

  // Dedup.
  ASSERT_SUCCEEDED(BTF::dedup(Merged));
  EXPECT_EQ(Merged.typesCount(), 2u); // int + struct foo

  // Verify correctness.
  EXPECT_EQ(Merged.findType(1)->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(Merged.findType(2)->getKind(), BTF::BTF_KIND_STRUCT);

  auto *ST = Merged.findType(2);
  auto *Members = reinterpret_cast<const BTF::BTFMember *>(
      reinterpret_cast<const uint8_t *>(ST) + sizeof(BTF::CommonType));
  EXPECT_EQ(Members[0].Type, 1u); // Points to deduped int
}

TEST(BTFDedupTest, duplicateFloats) {
  BTFBuilder B;
  uint32_t S1 = B.addString("float");
  uint32_t S2 = B.addString("float");

  B.addType({S1, mkInfo(BTF::BTF_KIND_FLOAT), {4}});
  B.addType({S2, mkInfo(BTF::BTF_KIND_FLOAT), {4}});

  EXPECT_EQ(B.typesCount(), 2u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 1u);
  EXPECT_EQ(B.findType(1)->getKind(), BTF::BTF_KIND_FLOAT);
  EXPECT_EQ(B.findType(1)->Size, 4u);
}

TEST(BTFDedupTest, differentFloats) {
  BTFBuilder B;
  uint32_t S1 = B.addString("float");
  uint32_t S2 = B.addString("double");

  B.addType({S1, mkInfo(BTF::BTF_KIND_FLOAT), {4}});
  B.addType({S2, mkInfo(BTF::BTF_KIND_FLOAT), {8}});

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 2u);
}

TEST(BTFDedupTest, duplicateEnum64) {
  BTFBuilder B;
  uint32_t ES = B.addString("big_enum");
  uint32_t VS = B.addString("VAL");

  B.addType({ES, mkInfo(BTF::BTF_KIND_ENUM64) | 1, {8}});
  B.addTail(BTF::BTFEnum64({VS, 0xDEADBEEF, 0x12345678}));

  uint32_t ES2 = B.addString("big_enum");
  uint32_t VS2 = B.addString("VAL");
  B.addType({ES2, mkInfo(BTF::BTF_KIND_ENUM64) | 1, {8}});
  B.addTail(BTF::BTFEnum64({VS2, 0xDEADBEEF, 0x12345678}));

  EXPECT_EQ(B.typesCount(), 2u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 1u);
}

TEST(BTFDedupTest, differentEnum64Values) {
  BTFBuilder B;
  uint32_t ES = B.addString("big_enum");
  uint32_t VS = B.addString("VAL");

  B.addType({ES, mkInfo(BTF::BTF_KIND_ENUM64) | 1, {8}});
  B.addTail(BTF::BTFEnum64({VS, 1, 0}));

  uint32_t ES2 = B.addString("big_enum");
  uint32_t VS2 = B.addString("VAL");
  B.addType({ES2, mkInfo(BTF::BTF_KIND_ENUM64) | 1, {8}});
  B.addTail(BTF::BTFEnum64({VS2, 2, 0})); // Different value

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 2u);
}

TEST(BTFDedupTest, duplicateTypedefChain) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t MyIntS = B.addString("myint");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 2: typedef myint -> int
  B.addType({MyIntS, mkInfo(BTF::BTF_KIND_TYPEDEF), {1}});

  // Duplicate set:
  uint32_t IntS2 = B.addString("int");
  uint32_t MyIntS2 = B.addString("myint");
  // Type 3: int (dup)
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 4: typedef myint -> int (dup, refs type 3)
  B.addType({MyIntS2, mkInfo(BTF::BTF_KIND_TYPEDEF), {3}});

  EXPECT_EQ(B.typesCount(), 4u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 2u);

  EXPECT_EQ(B.findType(1)->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(B.findType(2)->getKind(), BTF::BTF_KIND_TYPEDEF);
  EXPECT_EQ(B.findType(2)->Type, 1u);
}

TEST(BTFDedupTest, duplicateVolatileConstRestrict) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 2: volatile int
  B.addType({0, mkInfo(BTF::BTF_KIND_VOLATILE), {1}});
  // Type 3: const int
  B.addType({0, mkInfo(BTF::BTF_KIND_CONST), {1}});
  // Type 4: restrict int
  B.addType({0, mkInfo(BTF::BTF_KIND_RESTRICT), {1}});

  // Duplicate set:
  uint32_t IntS2 = B.addString("int");
  // Type 5: int (dup)
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 6: volatile int (dup, refs 5)
  B.addType({0, mkInfo(BTF::BTF_KIND_VOLATILE), {5}});
  // Type 7: const int (dup, refs 5)
  B.addType({0, mkInfo(BTF::BTF_KIND_CONST), {5}});
  // Type 8: restrict int (dup, refs 5)
  B.addType({0, mkInfo(BTF::BTF_KIND_RESTRICT), {5}});

  EXPECT_EQ(B.typesCount(), 8u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  // int + volatile + const + restrict = 4 types
  EXPECT_EQ(B.typesCount(), 4u);
}

TEST(BTFDedupTest, duplicateArrays) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 2: int[10]
  B.addType({0, mkInfo(BTF::BTF_KIND_ARRAY), {0}});
  B.addTail(BTF::BTFArray({1, 1, 10}));

  // Duplicate set:
  uint32_t IntS2 = B.addString("int");
  // Type 3: int (dup)
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 4: int[10] (dup, refs type 3)
  B.addType({0, mkInfo(BTF::BTF_KIND_ARRAY), {0}});
  B.addTail(BTF::BTFArray({3, 3, 10}));

  EXPECT_EQ(B.typesCount(), 4u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 2u);

  // Verify array references are remapped.
  const BTF::CommonType *ArrT = B.findType(2);
  ASSERT_TRUE(ArrT);
  EXPECT_EQ(ArrT->getKind(), BTF::BTF_KIND_ARRAY);
  auto *Arr = reinterpret_cast<const BTF::BTFArray *>(
      reinterpret_cast<const uint8_t *>(ArrT) + sizeof(BTF::CommonType));
  EXPECT_EQ(Arr->ElemType, 1u);
  EXPECT_EQ(Arr->IndexType, 1u);
  EXPECT_EQ(Arr->Nelems, 10u);
}

TEST(BTFDedupTest, differentArrayNelems) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");

  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // int[10]
  B.addType({0, mkInfo(BTF::BTF_KIND_ARRAY), {0}});
  B.addTail(BTF::BTFArray({1, 1, 10}));
  // int[20] — different nelems, should NOT dedup.
  B.addType({0, mkInfo(BTF::BTF_KIND_ARRAY), {0}});
  B.addTail(BTF::BTFArray({1, 1, 20}));

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 3u);
}

TEST(BTFDedupTest, duplicateFuncAndFuncProto) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t XS = B.addString("x");
  uint32_t FnS = B.addString("myfunc");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 2: func_proto(int x) -> int
  B.addType({0, mkInfo(BTF::BTF_KIND_FUNC_PROTO) | 1, {1}});
  B.addTail(BTF::BTFParam({XS, 1}));
  // Type 3: func "myfunc" -> func_proto
  B.addType({FnS, mkInfo(BTF::BTF_KIND_FUNC), {2}});

  // Duplicate:
  uint32_t IntS2 = B.addString("int");
  uint32_t XS2 = B.addString("x");
  uint32_t FnS2 = B.addString("myfunc");
  // Type 4: int (dup)
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 5: func_proto (dup, refs type 4)
  B.addType({0, mkInfo(BTF::BTF_KIND_FUNC_PROTO) | 1, {4}});
  B.addTail(BTF::BTFParam({XS2, 4}));
  // Type 6: func (dup, refs type 5)
  B.addType({FnS2, mkInfo(BTF::BTF_KIND_FUNC), {5}});

  EXPECT_EQ(B.typesCount(), 6u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 3u);
}

TEST(BTFDedupTest, duplicateVar) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t VS = B.addString("myvar");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 2: VAR myvar -> int
  B.addType({VS, mkInfo(BTF::BTF_KIND_VAR), {1}});
  B.addTail((uint32_t)1); // global linkage

  // Duplicate:
  uint32_t IntS2 = B.addString("int");
  uint32_t VS2 = B.addString("myvar");
  // Type 3: int (dup)
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 4: VAR myvar (dup, refs type 3)
  B.addType({VS2, mkInfo(BTF::BTF_KIND_VAR), {3}});
  B.addTail((uint32_t)1);

  EXPECT_EQ(B.typesCount(), 4u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 2u);
}

TEST(BTFDedupTest, duplicateTypeTag) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t TagS = B.addString("user");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 2: TYPE_TAG "user" -> int
  B.addType({TagS, mkInfo(BTF::BTF_KIND_TYPE_TAG), {1}});

  // Duplicate:
  uint32_t IntS2 = B.addString("int");
  uint32_t TagS2 = B.addString("user");
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  B.addType({TagS2, mkInfo(BTF::BTF_KIND_TYPE_TAG), {3}});

  EXPECT_EQ(B.typesCount(), 4u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 2u);
}

TEST(BTFDedupTest, duplicateDeclTag) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t TagS = B.addString("mytag");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 2: DECL_TAG "mytag" on type 1, component_idx=-1
  B.addType({TagS, mkInfo(BTF::BTF_KIND_DECL_TAG), {1}});
  B.addTail((uint32_t)-1);

  // Duplicate:
  uint32_t IntS2 = B.addString("int");
  uint32_t TagS2 = B.addString("mytag");
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  B.addType({TagS2, mkInfo(BTF::BTF_KIND_DECL_TAG), {3}});
  B.addTail((uint32_t)-1);

  EXPECT_EQ(B.typesCount(), 4u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 2u);
}

TEST(BTFDedupTest, differentDeclTagComponentIdx) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t TagS = B.addString("mytag");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 2: DECL_TAG on type 1, component_idx=0
  B.addType({TagS, mkInfo(BTF::BTF_KIND_DECL_TAG), {1}});
  B.addTail((uint32_t)0);
  // Type 3: DECL_TAG on type 1, component_idx=1 — different, should NOT dedup
  uint32_t TagS2 = B.addString("mytag");
  B.addType({TagS2, mkInfo(BTF::BTF_KIND_DECL_TAG), {1}});
  B.addTail((uint32_t)1);

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 3u);
}

TEST(BTFDedupTest, structDifferentMemberOffsets) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t FooS = B.addString("foo");
  uint32_t AS = B.addString("a");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  // Type 2: struct foo { int a; } at offset 0
  B.addType({FooS, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});
  B.addTail(BTF::BTFMember({AS, 1, 0}));

  // Type 3: struct foo { int a; } at offset 32 — different offset
  uint32_t FooS2 = B.addString("foo");
  uint32_t AS2 = B.addString("a");
  B.addType({FooS2, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});
  B.addTail(BTF::BTFMember({AS2, 1, 32})); // Different offset

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 3u); // int + 2 different structs
}

TEST(BTFDedupTest, structDifferentMemberNames) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t FooS = B.addString("foo");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  // Type 2: struct foo { int a; }
  uint32_t AS = B.addString("a");
  B.addType({FooS, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});
  B.addTail(BTF::BTFMember({AS, 1, 0}));

  // Type 3: struct foo { int b; } — different member name
  uint32_t FooS2 = B.addString("foo");
  uint32_t BAS = B.addString("b");
  B.addType({FooS2, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});
  B.addTail(BTF::BTFMember({BAS, 1, 0}));

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 3u);
}

TEST(BTFDedupTest, structDifferentMemberCount) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t FooS = B.addString("foo");
  uint32_t AS = B.addString("a");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);

  // Type 2: struct foo { int a; } — 1 member
  B.addType({FooS, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});
  B.addTail(BTF::BTFMember({AS, 1, 0}));

  // Type 3: struct foo { int a; int b; } — 2 members
  uint32_t FooS2 = B.addString("foo");
  uint32_t AS2 = B.addString("a");
  uint32_t BS2 = B.addString("b");
  B.addType({FooS2, mkInfo(BTF::BTF_KIND_STRUCT) | 2, {8}});
  B.addTail(BTF::BTFMember({AS2, 1, 0}));
  B.addTail(BTF::BTFMember({BS2, 1, 32}));

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 3u);
}

TEST(BTFDedupTest, mutualRecursion) {
  // struct A { struct B *b; };
  // struct B { struct A *a; };
  BTFBuilder B;
  uint32_t AS = B.addString("A");
  uint32_t BS = B.addString("B");
  uint32_t MemA = B.addString("a");
  uint32_t MemB = B.addString("b");

  // Type 1: ptr to struct B (type 4)
  B.addType({0, mkInfo(BTF::BTF_KIND_PTR), {4}});
  // Type 2: ptr to struct A (type 3)
  B.addType({0, mkInfo(BTF::BTF_KIND_PTR), {3}});
  // Type 3: struct A { struct B *b; }
  B.addType({AS, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {8}});
  B.addTail(BTF::BTFMember({MemB, 1, 0}));
  // Type 4: struct B { struct A *a; }
  B.addType({BS, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {8}});
  B.addTail(BTF::BTFMember({MemA, 2, 0}));

  // Duplicate set:
  uint32_t AS2 = B.addString("A");
  uint32_t BS2 = B.addString("B");
  uint32_t MemA2 = B.addString("a");
  uint32_t MemB2 = B.addString("b");

  // Type 5: ptr to struct B (type 8)
  B.addType({0, mkInfo(BTF::BTF_KIND_PTR), {8}});
  // Type 6: ptr to struct A (type 7)
  B.addType({0, mkInfo(BTF::BTF_KIND_PTR), {7}});
  // Type 7: struct A (dup)
  B.addType({AS2, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {8}});
  B.addTail(BTF::BTFMember({MemB2, 5, 0}));
  // Type 8: struct B (dup)
  B.addType({BS2, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {8}});
  B.addTail(BTF::BTFMember({MemA2, 6, 0}));

  EXPECT_EQ(B.typesCount(), 8u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  // Should dedup to: ptr(B), ptr(A), struct A, struct B = 4 types
  EXPECT_EQ(B.typesCount(), 4u);
}

TEST(BTFDedupTest, diamondDependency) {
  // Two structs that share a base INT type.
  // struct S1 { int x; }; struct S2 { int y; };
  // Both reference the same int. After dedup from two CUs, we should get
  // 1 int + 2 structs (they're different) = 3 types.
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t S1S = B.addString("S1");
  uint32_t S2S = B.addString("S2");
  uint32_t XS = B.addString("x");
  uint32_t YS = B.addString("y");

  // CU 1:
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});  // 1
  B.addTail((uint32_t)0);
  B.addType({S1S, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});  // 2
  B.addTail(BTF::BTFMember({XS, 1, 0}));
  B.addType({S2S, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});  // 3
  B.addTail(BTF::BTFMember({YS, 1, 0}));

  // CU 2 (same types, different IDs):
  uint32_t IntS2 = B.addString("int");
  uint32_t S1S2 = B.addString("S1");
  uint32_t S2S2 = B.addString("S2");
  uint32_t XS2 = B.addString("x");
  uint32_t YS2 = B.addString("y");

  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});  // 4
  B.addTail((uint32_t)0);
  B.addType({S1S2, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});  // 5
  B.addTail(BTF::BTFMember({XS2, 4, 0}));
  B.addType({S2S2, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}});  // 6
  B.addTail(BTF::BTFMember({YS2, 4, 0}));

  EXPECT_EQ(B.typesCount(), 6u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  // 1 int + 2 different structs = 3
  EXPECT_EQ(B.typesCount(), 3u);
}

TEST(BTFDedupTest, manyDuplicateInts) {
  BTFBuilder B;
  const unsigned N = 100;
  for (unsigned I = 0; I < N; ++I) {
    uint32_t S = B.addString("int");
    B.addType({S, mkInfo(BTF::BTF_KIND_INT), {4}});
    B.addTail((uint32_t)0);
  }

  EXPECT_EQ(B.typesCount(), N);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 1u);
}

TEST(BTFDedupTest, enumMembersPreservedAfterDedup) {
  BTFBuilder B;
  uint32_t ES = B.addString("color");
  uint32_t RS = B.addString("RED");
  uint32_t GS = B.addString("GREEN");
  uint32_t BS2 = B.addString("BLUE");

  B.addType({ES, mkInfo(BTF::BTF_KIND_ENUM) | 3, {4}});
  B.addTail(BTF::BTFEnum({RS, 0}));
  B.addTail(BTF::BTFEnum({GS, 1}));
  B.addTail(BTF::BTFEnum({BS2, 2}));

  // Duplicate:
  uint32_t ES2 = B.addString("color");
  uint32_t RS2 = B.addString("RED");
  uint32_t GS2 = B.addString("GREEN");
  uint32_t BS3 = B.addString("BLUE");
  B.addType({ES2, mkInfo(BTF::BTF_KIND_ENUM) | 3, {4}});
  B.addTail(BTF::BTFEnum({RS2, 0}));
  B.addTail(BTF::BTFEnum({GS2, 1}));
  B.addTail(BTF::BTFEnum({BS3, 2}));

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 1u);

  // Verify enum members are intact.
  const BTF::CommonType *T = B.findType(1);
  ASSERT_TRUE(T);
  EXPECT_EQ(T->getKind(), BTF::BTF_KIND_ENUM);
  EXPECT_EQ(T->getVlen(), 3u);
  auto *Vals = reinterpret_cast<const BTF::BTFEnum *>(
      reinterpret_cast<const uint8_t *>(T) + sizeof(BTF::CommonType));
  EXPECT_EQ(B.findString(Vals[0].NameOff), "RED");
  EXPECT_EQ(Vals[0].Val, 0);
  EXPECT_EQ(B.findString(Vals[1].NameOff), "GREEN");
  EXPECT_EQ(Vals[1].Val, 1);
  EXPECT_EQ(B.findString(Vals[2].NameOff), "BLUE");
  EXPECT_EQ(Vals[2].Val, 2);
}

TEST(BTFDedupTest, allKindsDedup) {
  // Build a BTF with one of each type kind, then duplicate all of them.
  // After dedup, should end up with exactly 19 types.
  auto BuildAllKinds = [](BTFBuilder &B, uint32_t BaseId) {
    uint32_t S = B.addString("t");
    uint32_t M = B.addString("m");
    uint32_t IntId = BaseId + 1;
    uint32_t FuncProtoId = BaseId + 12;
    uint32_t VarId = BaseId + 14;

    B.addType({S, mkInfo(BTF::BTF_KIND_INT), {4}});        // +1
    B.addTail((uint32_t)0);
    B.addType({S, mkInfo(BTF::BTF_KIND_PTR), {IntId}});    // +2
    B.addType({S, mkInfo(BTF::BTF_KIND_ARRAY), {0}});      // +3
    B.addTail(BTF::BTFArray({IntId, IntId, 10}));
    B.addType({S, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}}); // +4
    B.addTail(BTF::BTFMember({M, IntId, 0}));
    B.addType({S, mkInfo(BTF::BTF_KIND_UNION) | 1, {4}});  // +5
    B.addTail(BTF::BTFMember({M, IntId, 0}));
    B.addType({S, mkInfo(BTF::BTF_KIND_ENUM) | 1, {4}});   // +6
    B.addTail(BTF::BTFEnum({M, 42}));
    B.addType({S, mkInfo(BTF::BTF_KIND_FWD), {0}});        // +7
    B.addType({S, mkInfo(BTF::BTF_KIND_TYPEDEF), {IntId}}); // +8
    B.addType({S, mkInfo(BTF::BTF_KIND_VOLATILE), {IntId}}); // +9
    B.addType({S, mkInfo(BTF::BTF_KIND_CONST), {IntId}});   // +10
    B.addType({S, mkInfo(BTF::BTF_KIND_RESTRICT), {IntId}}); // +11
    B.addType({S, mkInfo(BTF::BTF_KIND_FUNC_PROTO) | 1, {IntId}}); // +12
    B.addTail(BTF::BTFParam({M, IntId}));
    B.addType({S, mkInfo(BTF::BTF_KIND_FUNC), {FuncProtoId}}); // +13
    B.addType({S, mkInfo(BTF::BTF_KIND_VAR), {IntId}});     // +14
    B.addTail((uint32_t)0);
    B.addType({S, mkInfo(BTF::BTF_KIND_DATASEC) | 1, {0}}); // +15
    B.addTail(BTF::BTFDataSec({VarId, 0, 4}));
    B.addType({S, mkInfo(BTF::BTF_KIND_FLOAT), {4}});       // +16
    B.addType({S, mkInfo(BTF::BTF_KIND_DECL_TAG), {IntId}}); // +17
    B.addTail((uint32_t)-1);
    B.addType({S, mkInfo(BTF::BTF_KIND_TYPE_TAG), {IntId}}); // +18
    B.addType({S, mkInfo(BTF::BTF_KIND_ENUM64) | 1, {8}});  // +19
    B.addTail(BTF::BTFEnum64({M, 1, 0}));
  };

  BTFBuilder B;
  BuildAllKinds(B, 0);    // Types 1-19
  BuildAllKinds(B, 19);   // Types 20-38 (duplicates)

  EXPECT_EQ(B.typesCount(), 38u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 19u);

  // Verify all 19 type kinds survive.
  for (uint32_t Id = 1; Id <= 19; ++Id) {
    ASSERT_TRUE(B.findType(Id))
        << "Type " << Id << " missing after dedup";
  }
}

TEST(BTFDedupTest, unionDedup) {
  BTFBuilder B;
  uint32_t IntS = B.addString("int");
  uint32_t US = B.addString("myunion");
  uint32_t XS = B.addString("x");
  uint32_t YS = B.addString("y");

  // Type 1: int
  B.addType({IntS, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  // Type 2: union myunion { int x; int y; }
  B.addType({US, mkInfo(BTF::BTF_KIND_UNION) | 2, {4}});
  B.addTail(BTF::BTFMember({XS, 1, 0}));
  B.addTail(BTF::BTFMember({YS, 1, 0}));

  // Duplicate:
  uint32_t IntS2 = B.addString("int");
  uint32_t US2 = B.addString("myunion");
  uint32_t XS2 = B.addString("x");
  uint32_t YS2 = B.addString("y");
  B.addType({IntS2, mkInfo(BTF::BTF_KIND_INT), {4}});
  B.addTail((uint32_t)0);
  B.addType({US2, mkInfo(BTF::BTF_KIND_UNION) | 2, {4}});
  B.addTail(BTF::BTFMember({XS2, 3, 0}));
  B.addTail(BTF::BTFMember({YS2, 3, 0}));

  EXPECT_EQ(B.typesCount(), 4u);
  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 2u);
}

TEST(BTFDedupTest, dedupRoundtripAllKinds) {
  // Build, dedup, write, parse — verify the output is valid BTF.
  BTFBuilder B;
  uint32_t S = B.addString("t");
  uint32_t M = B.addString("m");

  B.addType({S, mkInfo(BTF::BTF_KIND_INT), {4}});        // 1
  B.addTail((uint32_t)0);
  B.addType({S, mkInfo(BTF::BTF_KIND_PTR), {1}});        // 2
  B.addType({S, mkInfo(BTF::BTF_KIND_STRUCT) | 1, {4}}); // 3
  B.addTail(BTF::BTFMember({M, 1, 0}));
  B.addType({S, mkInfo(BTF::BTF_KIND_ENUM) | 1, {4}});   // 4
  B.addTail(BTF::BTFEnum({M, 99}));
  B.addType({0, mkInfo(BTF::BTF_KIND_CONST), {1}});      // 5

  // Duplicate int + ptr:
  uint32_t S2 = B.addString("t");
  B.addType({S2, mkInfo(BTF::BTF_KIND_INT), {4}});       // 6
  B.addTail((uint32_t)0);
  B.addType({S2, mkInfo(BTF::BTF_KIND_PTR), {6}});       // 7

  ASSERT_SUCCEEDED(BTF::dedup(B));
  EXPECT_EQ(B.typesCount(), 5u);

  SmallVector<uint8_t, 0> Output;
  B.write(Output, !sys::IsBigEndianHost);

  SmallString<0> Storage;
  auto Obj = makeELFWithBTF(Output, Storage);
  ASSERT_TRUE(Obj);

  BTFParser Parser;
  BTFParser::ParseOptions Opts;
  Opts.LoadTypes = true;
  ASSERT_SUCCEEDED(Parser.parse(*Obj, Opts));
  EXPECT_EQ(Parser.typesCount(), 6u); // 5 types + void

  EXPECT_EQ(Parser.findType(1)->getKind(), BTF::BTF_KIND_INT);
  EXPECT_EQ(Parser.findType(2)->getKind(), BTF::BTF_KIND_PTR);
  EXPECT_EQ(Parser.findType(2)->Type, 1u);
  EXPECT_EQ(Parser.findType(3)->getKind(), BTF::BTF_KIND_STRUCT);
  EXPECT_EQ(Parser.findType(4)->getKind(), BTF::BTF_KIND_ENUM);
  EXPECT_EQ(Parser.findType(5)->getKind(), BTF::BTF_KIND_CONST);
  EXPECT_EQ(Parser.findType(5)->Type, 1u);
}

} // namespace
