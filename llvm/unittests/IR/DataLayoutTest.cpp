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
      DataLayout::parse("p16777216:64:64:64"),
      FailedWithMessage("Invalid address space, must be a 24-bit integer"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("a1:64"),
      FailedWithMessage("Sized aggregate specification in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("a:"),
      FailedWithMessage("Trailing separator in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("p:48:52"),
      FailedWithMessage("number of bits must be a byte width multiple"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("e-p"),
      FailedWithMessage(
          "Missing size specification for pointer in datalayout string"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("e-p:64"),
      FailedWithMessage(
          "Missing alignment specification for pointer in datalayout string"));
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
      DataLayout::parse("p:32:32:16"),
      FailedWithMessage(
          "Preferred alignment cannot be less than the ABI alignment"));
  EXPECT_THAT_EXPECTED(DataLayout::parse("p:0:32:32"),
                       FailedWithMessage("Invalid pointer size of 0 bytes"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("p:64:24:64"),
      FailedWithMessage("Pointer ABI alignment must be a power of 2"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("p:64:64:24"),
      FailedWithMessage("Pointer preferred alignment must be a power of 2"));
  EXPECT_THAT_EXPECTED(
      DataLayout::parse("p:64:64:64:128"),
      FailedWithMessage("Index width cannot be larger than pointer width"));
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
