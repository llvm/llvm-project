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
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

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

} // anonymous namespace
