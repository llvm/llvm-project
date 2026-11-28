//===-- COFFLinkGraphTests.cpp - Unit tests for COFF LinkGraph utils ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITLinkTestUtils.h"

#include "llvm/BinaryFormat/COFF.h"
#include "llvm/ExecutionEngine/JITLink/COFF.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Object/COFF.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;

TEST(COFFLinkGraphTest, GetImageBaseSymbolReturnsNullWhenMissing) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), 8, SubtargetFeatures(),
              getGenericEdgeKindName);

  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), nullptr);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolFindsExternal) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), 8, SubtargetFeatures(),
              getGenericEdgeKindName);

  auto &ExtSym = G.addExternalSymbol(G.intern("__ImageBase"), 0, false);
  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), &ExtSym);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolFindsAbsolute) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), 8, SubtargetFeatures(),
              getGenericEdgeKindName);

  auto &AbsSym =
      G.addAbsoluteSymbol(G.intern("__ImageBase"), orc::ExecutorAddr(0x1000), 0,
                          Linkage::Strong, Scope::Default, true);
  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), &AbsSym);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolFindsDefined) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), 8, SubtargetFeatures(),
              getGenericEdgeKindName);

  auto &Sec =
      G.createSection("__data", orc::MemProt::Read | orc::MemProt::Write);
  auto &B =
      G.createContentBlock(Sec, BlockContent, orc::ExecutorAddr(0x2000), 8, 0);
  auto &DefSym =
      G.addDefinedSymbol(B, 0, G.intern("__ImageBase"), 4, Linkage::Strong,
                         Scope::Default, false, false);
  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), &DefSym);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolCachesResult) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), 8, SubtargetFeatures(),
              getGenericEdgeKindName);

  auto &ExtSym = G.addExternalSymbol(G.intern("__ImageBase"), 0, false);
  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), &ExtSym);

  // Remove the symbol -- cached result should still be returned.
  G.removeExternalSymbol(ExtSym);
  EXPECT_EQ(GetIB(G), &ExtSym);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolReset) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), 8, SubtargetFeatures(),
              getGenericEdgeKindName);

  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), nullptr);

  // Add a symbol and reset -- should now find it.
  auto &ExtSym = G.addExternalSymbol(G.intern("__ImageBase"), 0, false);
  GetIB.reset();
  EXPECT_EQ(GetIB(G), &ExtSym);

  // Reset with an explicit cache value of nullptr -- should return nullptr
  // without searching.
  GetIB.reset(nullptr);
  EXPECT_EQ(GetIB(G), nullptr);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolCustomName) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), 8, SubtargetFeatures(),
              getGenericEdgeKindName);

  G.addExternalSymbol(G.intern("__ImageBase"), 0, false);
  auto &CustomSym = G.addExternalSymbol(G.intern("__CustomBase"), 0, false);

  GetImageBaseSymbol GetIB("__CustomBase");
  EXPECT_EQ(GetIB(G), &CustomSym);
}

TEST(COFFLinkGraphTest, IsCOFFWeakDefaultHelper) {
  constexpr uint32_t Alias = COFF::IMAGE_WEAK_EXTERN_SEARCH_ALIAS;
  EXPECT_TRUE(isCOFFWeakDefaultHelper(".weak.foo.default", "foo", Alias));
  EXPECT_TRUE(isCOFFWeakDefaultHelper(".weak.foo.default.inl", "foo", Alias));
  EXPECT_FALSE(isCOFFWeakDefaultHelper(".weak.foo.defaultx", "foo", Alias));
  EXPECT_FALSE(isCOFFWeakDefaultHelper(".weak.foobar.default", "foo", Alias));
  EXPECT_FALSE(isCOFFWeakDefaultHelper("bar", "foo", Alias));
  EXPECT_FALSE(isCOFFWeakDefaultHelper(
      ".weak.foo.default", "foo", COFF::IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY));
}

TEST(COFFLinkGraphTest, GetCOFFWeakDefaultHelper) {
  // foo is MC's weak definition, wbar is "wbar = bar", and bad has a bad
  // TagIndex. Aux records take symbol table slots.
  StringRef Yaml = R"(--- !COFF
header:
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: [  ]
sections:
  - Name:            .text
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    Alignment:       4
    SectionData:     C3C3
symbols:
  - Name:            foo
    Value:           0
    SectionNumber:   0
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_WEAK_EXTERNAL
    WeakExternal:
      TagIndex:        2
      Characteristics: IMAGE_WEAK_EXTERN_SEARCH_ALIAS
  - Name:            .weak.foo.default
    Value:           0
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            wbar
    Value:           0
    SectionNumber:   0
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_WEAK_EXTERNAL
    WeakExternal:
      TagIndex:        5
      Characteristics: IMAGE_WEAK_EXTERN_SEARCH_ALIAS
  - Name:            bar
    Value:           1
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            bad
    Value:           0
    SectionNumber:   0
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_WEAK_EXTERNAL
    WeakExternal:
      TagIndex:        100
      Characteristics: IMAGE_WEAK_EXTERN_SEARCH_ALIAS
...
)";
  SmallVector<char, 0> Storage;
  ASSERT_TRUE(yaml::yaml2ObjectFile(Storage, Yaml, [](const Twine &) {}));
  auto Obj = cantFail(object::COFFObjectFile::create(
      MemoryBufferRef(StringRef(Storage.data(), Storage.size()), "obj")));

  auto Helper = [&](uint32_t Index) {
    return getCOFFWeakDefaultHelper(*Obj, cantFail(Obj->getSymbol(Index)));
  };
  EXPECT_THAT_EXPECTED(Helper(0),
                       HasValue(std::optional<StringRef>(".weak.foo.default")));
  EXPECT_THAT_EXPECTED(Helper(2), HasValue(std::nullopt));
  EXPECT_THAT_EXPECTED(Helper(3), HasValue(std::nullopt));
  EXPECT_THAT_EXPECTED(Helper(6), Failed());
}
