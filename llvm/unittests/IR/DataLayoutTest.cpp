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

// TODO: Split into multiple TESTs.
TEST(DataLayoutTest, ParseErrors) {
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("^"),
      FailedWithMessage("Unknown specifier in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("m:v"),
      FailedWithMessage("Unknown mangling in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("n0"),
      FailedWithMessage("Zero width native integer type in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("a1:64"),
      FailedWithMessage("Sized aggregate specification in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("a:"),
      FailedWithMessage("Trailing separator in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("m"),
      FailedWithMessage("Expected mangling specifier in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("m."),
      FailedWithMessage("Unexpected trailing characters after mangling "
                        "specifier in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("f"),
      FailedWithMessage(
          "Missing alignment specification in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse(":32"),
      FailedWithMessage(
          "Expected token before separator in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("i64:64:16"),
      FailedWithMessage(
          "Preferred alignment cannot be less than the ABI alignment"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("i64:16:16777216"),
      FailedWithMessage(
          "Invalid preferred alignment, must be a 16bit integer"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("i64:16777216:16777216"),
      FailedWithMessage("Invalid ABI alignment, must be a 16bit integer"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("i16777216:16:16"),
      FailedWithMessage("Invalid bit width, must be a 24-bit integer"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("v128:0:128"),
      FailedWithMessage(
          "ABI alignment specification must be >0 for non-aggregate types"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("i32:24:32"),
      FailedWithMessage("Invalid ABI alignment, must be a power of 2"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("i32:32:24"),
      FailedWithMessage("Invalid preferred alignment, must be a power of 2"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("A16777216"),
      FailedWithMessage("Invalid address space, must be a 24-bit integer"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("G16777216"),
      FailedWithMessage("Invalid address space, must be a 24-bit integer"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("P16777216"),
      FailedWithMessage("Invalid address space, must be a 24-bit integer"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("Fi24"),
      FailedWithMessage("Alignment is neither 0 nor a power of 2"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("i8:16"),
      FailedWithMessage("Invalid ABI alignment, i8 must be naturally aligned"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("S24"),
      FailedWithMessage("Alignment is neither 0 nor a power of 2"));
}

TEST(DataLayout, LayoutStringFormat) {
  for (StringRef Str : {"", "e", "m:e", "m:e-e"})
    EXPECT_THAT_EXPECTED(DataLayout::parse(Str), Succeeded());

  for (StringRef Str : {"-", "e-", "-m:e", "m:e--e"})
    EXPECT_THAT_EXPECTED(
        DataLayout::parse(Str),
        FailedWithMessage("empty specification is not allowed"));
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
  EXPECT_EQ(DataLayout::FunctionPtrAlignType::Independent, \
      DataLayout("").getFunctionPtrAlignType());
  EXPECT_EQ(DataLayout::FunctionPtrAlignType::Independent, \
      DataLayout("Fi8").getFunctionPtrAlignType());
  EXPECT_EQ(DataLayout::FunctionPtrAlignType::MultipleOfFunctionAlign, \
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
