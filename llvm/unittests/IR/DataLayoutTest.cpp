//===- ConstantRangeTest.cpp - ConstantRange tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class DataLayoutTest : public ::testing::Test {};

TEST(DataLayout, LayoutStringFormat) {
  for (StringRef Str : {"", "e", "m:e", "m:e-e"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str), Succeeded());

  for (StringRef Str : {"-", "e-", "-m:e", "m:e--e"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("empty specification is not allowed"));
}

TEST(DataLayoutTest, InvalidSpecifier) {
  EXPECT_THAT_EXPECTED(DataLayout::parse("^"),
                       FailedWithMessage("unknown specifier '^'"));
  EXPECT_THAT_EXPECTED(DataLayout::parse("I8:8"),
                       FailedWithMessage("unknown specifier 'I'"));
  EXPECT_THAT_EXPECTED(DataLayout::parse("e-X"),
                       FailedWithMessage("unknown specifier 'X'"));
  EXPECT_THAT_EXPECTED(DataLayout::parse("p0:32:32-64"),
                       FailedWithMessage("unknown specifier '6'"));
}

TEST(DataLayoutTest, ParseEndianness) {
  EXPECT_THAT_EXPECTED(DataLayout::parse("e"), Succeeded());
  EXPECT_THAT_EXPECTED(DataLayout::parse("E"), Succeeded());

  for (StringRef Str : {"ee", "e0", "e:0", "E0:E", "El", "E:B"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("malformed specification, must be just 'e' or 'E'"));
}

TEST(DataLayoutTest, ParseMangling) {
  for (StringRef Str : {"m:a", "m:e", "m:l", "m:m", "m:o", "m:w", "m:x"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str), Succeeded());

  for (StringRef Str : {"m", "ms:m", "m:"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage(
            "malformed specification, must be of the form \"m:<mangling>\""));

  for (StringRef Str : {"m:ms", "m:E", "m:0"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str),
                         FailedWithMessage("unknown mangling mode"));
}

TEST(DataLayoutTest, ParseStackNaturalAlign) {
  for (StringRef Str : {"S8", "S32768"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str), Succeeded());

  EXPECT_THAT_EXPECTED(
      DataLayout::parse("S"),
      FailedWithMessage(
          "malformed specification, must be of the form \"S<size>\""));

  for (StringRef Str : {"SX", "S0x20", "S65536"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("stack natural alignment must be a 16-bit integer"));

  EXPECT_THAT_EXPECTED(
      DataLayout::parse("S0"),
      FailedWithMessage("stack natural alignment must be non-zero"));

  for (StringRef Str : {"S1", "S7", "S24", "S65535"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("stack natural alignment must be a power of two "
                          "times the byte width"));
}

TEST(DataLayoutTest, ParseAddrSpace) {
  for (StringRef Str : {"P0", "A0", "G0", "P1", "A1", "G1", "P16777215",
                        "A16777215", "G16777215"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str), Succeeded());

  for (StringRef Str : {"P", "A", "G"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage(("malformed specification, must be of the form \"" +
                           Twine(Str.front()) + "<address space>\"")
                              .str()));

  for (StringRef Str : {"Px", "A0x1", "G16777216"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("address space must be a 24-bit integer"));
}

TEST(DataLayoutTest, ParseFuncPtrSpec) {
  for (StringRef Str : {"Fi8", "Fn16", "Fi32768", "Fn32768"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str), Succeeded());

  EXPECT_THAT_EXPECTED(
      DataLayout::parse("F"),
      FailedWithMessage(
          "malformed specification, must be of the form \"F<type><abi>\""));

  EXPECT_THAT_EXPECTED(
      DataLayout::parse("FN"),
      FailedWithMessage("unknown function pointer alignment type 'N'"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("F32"),
      FailedWithMessage("unknown function pointer alignment type '3'"));

  for (StringRef Str : {"Fi", "Fn"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("ABI alignment component cannot be empty"));

  for (StringRef Str : {"Fii", "Fn32x", "Fi65536", "Fn65536"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("ABI alignment must be a 16-bit integer"));

  for (StringRef Str : {"Fi0", "Fn0"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str),
                         FailedWithMessage("ABI alignment must be non-zero"));

  for (StringRef Str : {"Fi12", "Fn24"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage(
            "ABI alignment must be a power of two times the byte width"));
}

class DataLayoutPrimitiveSpecificationTest
    : public DataLayoutTest,
      public ::testing::WithParamInterface<char> {
  char Specifier;

public:
  DataLayoutPrimitiveSpecificationTest() : Specifier(GetParam()) {}

  std::string format(StringRef Str) const {
    std::string Res = Str.str();
    std::replace(Res.begin(), Res.end(), '!', Specifier);
    return Res;
  }
};

INSTANTIATE_TEST_SUITE_P(PrmitiveSpecifiers,
                         DataLayoutPrimitiveSpecificationTest,
                         ::testing::Values('i', 'f', 'v'));

TEST_P(DataLayoutPrimitiveSpecificationTest, ParsePrimitiveSpec) {
  for (StringRef Str :
       {"!1:16", "!8:8:8", "!16:32:64", "!16777215:32768:32768"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(format(Str)), Succeeded());

  for (StringRef Str : {"!", "!1", "!32:32:32:32", "!16:32:64:128"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(format(Str)),
        FailedWithMessage(format("malformed specification, must be of the form "
                                 "\"!<size>:<abi>[:<pref>]\"")));

  // size
  for (StringRef Str : {"!:8", "!:16:16", "!:32:64"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(format(Str)),
                         FailedWithMessage("size component cannot be empty"));

  for (StringRef Str :
       {"!0:8", "!0x8:8", "!x:8:8", "!0:16:32", "!16777216:64:64"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(format(Str)),
        FailedWithMessage("size must be a non-zero 24-bit integer"));

  // ABI alignment
  for (StringRef Str : {"!8:", "!16::16", "!32::64"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(format(Str)),
        FailedWithMessage("ABI alignment component cannot be empty"));

  for (StringRef Str : {"!1:x", "!8:8x:8", "!16:65536:65536"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(format(Str)),
        FailedWithMessage("ABI alignment must be a 16-bit integer"));

  for (StringRef Str : {"!8:0", "!16:0:16", "!32:0:64"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(format(Str)),
                         FailedWithMessage("ABI alignment must be non-zero"));

  for (StringRef Str : {"!1:1", "!8:4", "!16:6:16", "!32:24:64"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(format(Str)),
        FailedWithMessage(
            "ABI alignment must be a power of two times the byte width"));

  // preferred alignment
  for (StringRef Str : {"!1:8:", "!16:16:", "!64:32:"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(format(Str)),
        FailedWithMessage("preferred alignment component cannot be empty"));

  for (StringRef Str : {"!1:8:x", "!8:8:0x8", "!16:32:65536"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(format(Str)),
        FailedWithMessage("preferred alignment must be a 16-bit integer"));

  for (StringRef Str : {"!8:8:0", "!32:16:0"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(format(Str)),
        FailedWithMessage("preferred alignment must be non-zero"));

  for (StringRef Str : {"!1:8:12", "!8:8:17", "!16:32:40"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(format(Str)),
        FailedWithMessage(
            "preferred alignment must be a power of two times the byte width"));

  for (StringRef Str : {"!1:16:8", "!64:32:16"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(format(Str)),
        FailedWithMessage(
            "preferred alignment cannot be less than the ABI alignment"));

  // Additional check for byte-sized integer.
  if (GetParam() == 'i') {
    for (StringRef Str : {"!8:16", "!8:16:8", "!8:16:32"})
      EXPECT_THAT_EXPECTED(DataLayout::parse(format(Str)),
                           FailedWithMessage("i8 must be 8-bit aligned"));
  }
}

TEST(DataLayoutTest, ParseAggregateSpec) {
  for (StringRef Str : {"a:8", "a:0:16", "a0:32:64", "a:32768:32768"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str), Succeeded());

  for (StringRef Str : {"a", "a0", "a:32:32:32", "a0:32:64:128"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("malformed specification, must be of the form "
                          "\"a:<abi>[:<pref>]\""));

  // size
  for (StringRef Str : {"a1:8", "a0x0:8", "ax:16:32"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str),
                         FailedWithMessage("size must be zero"));

  // ABI alignment
  for (StringRef Str : {"a:", "a0:", "a::32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("ABI alignment component cannot be empty"));

  for (StringRef Str : {"a:x", "a0:0x0", "a:65536", "a0:65536:65536"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("ABI alignment must be a 16-bit integer"));

  for (StringRef Str : {"a:1", "a:4", "a:9:16", "a0:24:32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage(
            "ABI alignment must be a power of two times the byte width"));

  // preferred alignment
  for (StringRef Str : {"a:8:", "a0:16:", "a0:0:"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("preferred alignment component cannot be empty"));

  for (StringRef Str : {"a:16:x", "a0:8:0x8", "a:16:65536"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("preferred alignment must be a 16-bit integer"));

  for (StringRef Str : {"a:0:0", "a0:16:0"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("preferred alignment must be non-zero"));

  for (StringRef Str : {"a:8:12", "a:16:17", "a0:32:40"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage(
            "preferred alignment must be a power of two times the byte width"));

  for (StringRef Str : {"a:16:8", "a0:32:16"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage(
            "preferred alignment cannot be less than the ABI alignment"));
}

TEST(DataLayout, ParsePointerSpec) {
  for (StringRef Str :
       {"p:16:8", "p:16:16:64", "p:32:64:64:32", "p0:32:64", "p42:64:32:32",
        "p16777215:32:32:64:8", "p16777215:16777215:32768:32768:16777215"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str), Succeeded());

  for (StringRef Str :
       {"p", "p0", "p:32", "p0:32", "p:32:32:32:32:32", "p0:32:32:32:32:32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("malformed specification, must be of the form "
                          "\"p[<n>]:<size>:<abi>[:<pref>[:<idx>]]\""));

  // address space
  for (StringRef Str : {"p0x0:32:32", "px:32:32:32", "p16777216:32:32:32:32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("address space must be a 24-bit integer"));

  // pointer size
  for (StringRef Str : {"p::32", "p0::32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("pointer size component cannot be empty"));

  for (StringRef Str : {"p:0:32", "p0:0x1:32:32", "p42:16777216:32:32:32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("pointer size must be a non-zero 24-bit integer"));

  // ABI alignment
  for (StringRef Str : {"p:32:", "p0:32::32", "p42:32::32:32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("ABI alignment component cannot be empty"));

  for (StringRef Str : {"p:32:x", "p0:32:0x20:32", "p42:32:65536:32:32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("ABI alignment must be a 16-bit integer"));

  for (StringRef Str : {"p:32:0", "p0:32:0:32", "p42:32:0:32:32"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str),
                         FailedWithMessage("ABI alignment must be non-zero"));

  for (StringRef Str : {"p:32:4", "p42:32:24:32", "p0:32:65535:32:32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage(
            "ABI alignment must be a power of two times the byte width"));

  // preferred alignment
  for (StringRef Str : {"p:32:32:", "p0:32:32:", "p42:32:32::32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("preferred alignment component cannot be empty"));

  for (StringRef Str : {"p:32:32:x", "p0:32:32:0x20", "p42:32:32:65536:32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("preferred alignment must be a 16-bit integer"));

  for (StringRef Str : {"p:32:32:0", "p0:32:32:0", "p42:32:32:0:32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("preferred alignment must be non-zero"));

  for (StringRef Str : {"p:32:32:4", "p0:32:32:24", "p42:32:32:65535:32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage(
            "preferred alignment must be a power of two times the byte width"));

  for (StringRef Str : {"p:64:64:32", "p0:16:32:16:16"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage(
            "preferred alignment cannot be less than the ABI alignment"));

  // index size
  for (StringRef Str : {"p:32:32:32:", "p0:32:32:32:"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("index size component cannot be empty"));

  for (StringRef Str :
       {"p:32:32:32:0", "p0:32:32:32:0x20", "p42:32:32:32:16777216"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("index size must be a non-zero 24-bit integer"));

  for (StringRef Str : {"p:16:16:16:17", "p0:32:64:64:64", "p42:16:64:64:32"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("index size cannot be larger than the pointer size"));
}

TEST(DataLayoutTest, ParseNativeIntegersSpec) {
  for (StringRef Str : {"n1", "n1:8", "n24:12:16777215"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str), Succeeded());

  for (StringRef Str : {"n", "n1:", "n:8", "n16::32"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str),
                         FailedWithMessage("size component cannot be empty"));

  for (StringRef Str : {"n0", "n0x8:16", "n8:0", "n16:0:32", "n16777216",
                        "n16:16777216", "n32:64:16777216"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("size must be a non-zero 24-bit integer"));
}

TEST(DataLayout, ParseNonIntegralAddrSpace) {
  for (StringRef Str : {"ni:1", "ni:16777215", "ni:1:16777215"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str), Succeeded());

  for (StringRef Str : {"ni", "ni42", "nix"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("malformed specification, must be of the form "
                          "\"ni:<address space>[:<address space>]...\""));

  for (StringRef Str : {"ni:", "ni::42", "ni:42:"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("address space component cannot be empty"));

  for (StringRef Str : {"ni:x", "ni:42:0x1", "ni:16777216", "ni:42:16777216"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("address space must be a 24-bit integer"));

  for (StringRef Str : {"ni:0", "ni:42:0"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("address space 0 cannot be non-integral"));
}

TEST(DataLayout, GetStackAlignment) {
  DataLayout Default;
  EXPECT_FALSE(Default.getStackAlignment().has_value());

  std::pair<StringRef, Align> Cases[] = {
      {"S8", Align(1)},
      {"S64", Align(8)},
      {"S32768", Align(4096)},
  };
  for (auto [Layout, Val] : Cases) {
    DataLayout DL = cantFail(DataLayout::parse(Layout));
    EXPECT_EQ(DL.getStackAlignment(), Val) << Layout;
  }
}

TEST(DataLayout, GetPointerSizeInBits) {
  std::tuple<StringRef, unsigned, unsigned, unsigned> Cases[] = {
      {"", 64, 64, 64},
      {"p:16:32", 16, 16, 16},
      {"p0:32:64", 32, 32, 32},
      {"p1:16:32", 64, 16, 64},
      {"p1:31:32-p2:15:16:16:14", 64, 31, 15},
  };
  for (auto [Layout, V0, V1, V2] : Cases) {
    DataLayout DL = cantFail(DataLayout::parse(Layout));
    EXPECT_EQ(DL.getPointerSizeInBits(0), V0) << Layout;
    EXPECT_EQ(DL.getPointerSizeInBits(1), V1) << Layout;
    EXPECT_EQ(DL.getPointerSizeInBits(2), V2) << Layout;
  }
}

TEST(DataLayout, GetPointerSize) {
  std::tuple<StringRef, unsigned, unsigned, unsigned> Cases[] = {
      {"", 8, 8, 8},
      {"p:16:32", 2, 2, 2},
      {"p0:32:64", 4, 4, 4},
      {"p1:17:32", 8, 3, 8},
      {"p1:31:64-p2:23:8:16:9", 8, 4, 3},
  };
  for (auto [Layout, V0, V1, V2] : Cases) {
    DataLayout DL = cantFail(DataLayout::parse(Layout));
    EXPECT_EQ(DL.getPointerSize(0), V0) << Layout;
    EXPECT_EQ(DL.getPointerSize(1), V1) << Layout;
    EXPECT_EQ(DL.getPointerSize(2), V2) << Layout;
  }
}

TEST(DataLayout, GetIndexSizeInBits) {
  std::tuple<StringRef, unsigned, unsigned, unsigned> Cases[] = {
      {"", 64, 64, 64},
      {"p:16:32", 16, 16, 16},
      {"p0:32:64", 32, 32, 32},
      {"p1:16:32:32:10", 64, 10, 64},
      {"p1:31:32:64:20-p2:17:16:16:15", 64, 20, 15},
  };
  for (auto [Layout, V0, V1, V2] : Cases) {
    DataLayout DL = cantFail(DataLayout::parse(Layout));
    EXPECT_EQ(DL.getIndexSizeInBits(0), V0) << Layout;
    EXPECT_EQ(DL.getIndexSizeInBits(1), V1) << Layout;
    EXPECT_EQ(DL.getIndexSizeInBits(2), V2) << Layout;
  }
}

TEST(DataLayout, GetIndexSize) {
  std::tuple<StringRef, unsigned, unsigned, unsigned> Cases[] = {
      {"", 8, 8, 8},
      {"p:16:32", 2, 2, 2},
      {"p0:27:64", 4, 4, 4},
      {"p1:19:32:64:5", 8, 1, 8},
      {"p1:33:32:64:23-p2:21:8:16:13", 8, 3, 2},
  };
  for (auto [Layout, V0, V1, V2] : Cases) {
    DataLayout DL = cantFail(DataLayout::parse(Layout));
    EXPECT_EQ(DL.getIndexSize(0), V0) << Layout;
    EXPECT_EQ(DL.getIndexSize(1), V1) << Layout;
    EXPECT_EQ(DL.getIndexSize(2), V2) << Layout;
  }
}

TEST(DataLayout, GetPointerABIAlignment) {
  std::tuple<StringRef, unsigned, unsigned, unsigned> Cases[] = {
      {"", 8, 8, 8},
      {"p:16:32", 4, 4, 4},
      {"p0:16:32:64", 4, 4, 4},
      {"p1:32:16:64", 8, 2, 8},
      {"p1:33:16:32:15-p2:23:8:16:9", 8, 2, 1},
  };
  for (auto [Layout, V0, V1, V2] : Cases) {
    DataLayout DL = cantFail(DataLayout::parse(Layout));
    EXPECT_EQ(DL.getPointerABIAlignment(0).value(), V0) << Layout;
    EXPECT_EQ(DL.getPointerABIAlignment(1).value(), V1) << Layout;
    EXPECT_EQ(DL.getPointerABIAlignment(2).value(), V2) << Layout;
  }
}

TEST(DataLayout, GetPointerPrefAlignment) {
  std::tuple<StringRef, unsigned, unsigned, unsigned> Cases[] = {
      {"", 8, 8, 8},
      {"p:16:32", 4, 4, 4},
      {"p0:8:16:32", 4, 4, 4},
      {"p1:32:8:16", 8, 2, 8},
      {"p1:33:8:16:31-p2:23:8:32:17", 8, 2, 4},
  };
  for (auto [Layout, V0, V1, V2] : Cases) {
    DataLayout DL = cantFail(DataLayout::parse(Layout));
    EXPECT_EQ(DL.getPointerPrefAlignment(0).value(), V0) << Layout;
    EXPECT_EQ(DL.getPointerPrefAlignment(1).value(), V1) << Layout;
    EXPECT_EQ(DL.getPointerPrefAlignment(2).value(), V2) << Layout;
  }
}

TEST(DataLayout, IsNonIntegralAddressSpace) {
  DataLayout Default;
  EXPECT_THAT(Default.getNonIntegralAddressSpaces(), ::testing::SizeIs(0));
  EXPECT_FALSE(Default.isNonIntegralAddressSpace(0));
  EXPECT_FALSE(Default.isNonIntegralAddressSpace(1));

  DataLayout Custom = cantFail(DataLayout::parse("ni:2:16777215"));
  EXPECT_THAT(Custom.getNonIntegralAddressSpaces(),
              ::testing::ElementsAreArray({2U, 16777215U}));
  EXPECT_FALSE(Custom.isNonIntegralAddressSpace(0));
  EXPECT_FALSE(Custom.isNonIntegralAddressSpace(1));
  EXPECT_TRUE(Custom.isNonIntegralAddressSpace(2));
  EXPECT_TRUE(Custom.isNonIntegralAddressSpace(16777215));
}

TEST(DataLayoutTest, CopyAssignmentInvalidatesStructLayout) {
  DataLayout DL1 = cantFail(DataLayout::parse("p:32:32"));
  DataLayout DL2 = cantFail(DataLayout::parse("p:64:64"));

  LLVMContext Ctx;
  StructType *Ty = StructType::get(PointerType::getUnqual(Ctx));

  // Initialize struct layout caches.
  EXPECT_EQ(DL1.getStructLayout(Ty)->getSizeInBits(), 32U);
  EXPECT_EQ(DL1.getStructLayout(Ty)->getAlignment(), Align(4));
  EXPECT_EQ(DL2.getStructLayout(Ty)->getSizeInBits(), 64U);
  EXPECT_EQ(DL2.getStructLayout(Ty)->getAlignment(), Align(8));

  // The copy should invalidate DL1's cache.
  DL1 = DL2;
  EXPECT_EQ(DL1.getStructLayout(Ty)->getSizeInBits(), 64U);
  EXPECT_EQ(DL1.getStructLayout(Ty)->getAlignment(), Align(8));
  EXPECT_EQ(DL2.getStructLayout(Ty)->getSizeInBits(), 64U);
  EXPECT_EQ(DL2.getStructLayout(Ty)->getAlignment(), Align(8));
}

TEST(DataLayoutTest, FunctionPtrAlign) {
  EXPECT_EQ(MaybeAlign(0), DataLayout("").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(1), DataLayout("Fi8").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(2), DataLayout("Fi16").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(4), DataLayout("Fi32").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(8), DataLayout("Fi64").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(1), DataLayout("Fn8").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(2), DataLayout("Fn16").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(4), DataLayout("Fn32").getFunctionPtrAlign());
  EXPECT_EQ(MaybeAlign(8), DataLayout("Fn64").getFunctionPtrAlign());
  EXPECT_EQ(DataLayout::FunctionPtrAlignType::Independent,
            DataLayout("").getFunctionPtrAlignType());
  EXPECT_EQ(DataLayout::FunctionPtrAlignType::Independent,
            DataLayout("Fi8").getFunctionPtrAlignType());
  EXPECT_EQ(DataLayout::FunctionPtrAlignType::MultipleOfFunctionAlign,
            DataLayout("Fn8").getFunctionPtrAlignType());
  EXPECT_EQ(DataLayout("Fi8"), DataLayout("Fi8"));
  EXPECT_NE(DataLayout("Fi8"), DataLayout("Fi16"));
  EXPECT_NE(DataLayout("Fi8"), DataLayout("Fn8"));

  DataLayout a(""), b("Fi8"), c("Fn8");
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(b, c);

  a = b;
  EXPECT_EQ(a, b);
  a = c;
  EXPECT_EQ(a, c);
}

TEST(DataLayoutTest, ValueOrABITypeAlignment) {
  const DataLayout DL("Fi8");
  LLVMContext Context;
  Type *const FourByteAlignType = Type::getInt32Ty(Context);
  EXPECT_EQ(Align(16),
            DL.getValueOrABITypeAlignment(MaybeAlign(16), FourByteAlignType));
  EXPECT_EQ(Align(4),
            DL.getValueOrABITypeAlignment(MaybeAlign(), FourByteAlignType));
}

TEST(DataLayoutTest, GlobalsAddressSpace) {
  // When not explicitly defined the globals address space should be zero:
  EXPECT_EQ(DataLayout("").getDefaultGlobalsAddressSpace(), 0u);
  EXPECT_EQ(DataLayout("P1-A2").getDefaultGlobalsAddressSpace(), 0u);
  EXPECT_EQ(DataLayout("G2").getDefaultGlobalsAddressSpace(), 2u);
  // Check that creating a GlobalVariable without an explicit address space
  // in a module with a default globals address space respects that default:
  LLVMContext Context;
  std::unique_ptr<Module> M(new Module("MyModule", Context));
  // Default is globals in address space zero:
  auto *Int32 = Type::getInt32Ty(Context);
  auto *DefaultGlobal1 = new GlobalVariable(
      *M, Int32, false, GlobalValue::ExternalLinkage, nullptr);
  EXPECT_EQ(DefaultGlobal1->getAddressSpace(), 0u);
  auto *ExplicitGlobal1 = new GlobalVariable(
      *M, Int32, false, GlobalValue::ExternalLinkage, nullptr, "", nullptr,
      GlobalValue::NotThreadLocal, 123);
  EXPECT_EQ(ExplicitGlobal1->getAddressSpace(), 123u);

  // When using a datalayout with the global address space set to 200, global
  // variables should default to 200
  M->setDataLayout("G200");
  auto *DefaultGlobal2 = new GlobalVariable(
      *M, Int32, false, GlobalValue::ExternalLinkage, nullptr);
  EXPECT_EQ(DefaultGlobal2->getAddressSpace(), 200u);
  auto *ExplicitGlobal2 = new GlobalVariable(
      *M, Int32, false, GlobalValue::ExternalLinkage, nullptr, "", nullptr,
      GlobalValue::NotThreadLocal, 123);
  EXPECT_EQ(ExplicitGlobal2->getAddressSpace(), 123u);
}

TEST(DataLayoutTest, VectorAlign) {
  Expected<DataLayout> DL = DataLayout::parse("v64:64");
  EXPECT_THAT_EXPECTED(DL, Succeeded());

  LLVMContext Context;
  Type *const FloatTy = Type::getFloatTy(Context);
  Type *const V8F32Ty = FixedVectorType::get(FloatTy, 8);

  // The alignment for a vector type larger than any specified vector type uses
  // the natural alignment as a fallback.
  EXPECT_EQ(Align(4 * 8), DL->getABITypeAlign(V8F32Ty));
  EXPECT_EQ(Align(4 * 8), DL->getPrefTypeAlign(V8F32Ty));
}

TEST(DataLayoutTest, UEFI) {
  Triple TT = Triple("x86_64-unknown-uefi");

  // Test UEFI X86_64 Mangling Component.
  EXPECT_STREQ(DataLayout::getManglingComponent(TT), "-m:w");
}

} // anonymous namespace
