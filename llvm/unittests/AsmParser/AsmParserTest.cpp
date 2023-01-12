//===- llvm/unittest/AsmParser/AsmParserTest.cpp - asm parser unittests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/AsmParser/SlotMapping.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(AsmParserTest, NullTerminatedInput) {
  LLVMContext Ctx;
  StringRef Source = "; Empty module \n";
  SMDiagnostic Error;
  auto Mod = parseAssemblyString(Source, Error, Ctx);

  EXPECT_TRUE(Mod != nullptr);
  EXPECT_TRUE(Error.getMessage().empty());
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG

TEST(AsmParserTest, NonNullTerminatedInput) {
  LLVMContext Ctx;
  StringRef Source = "; Empty module \n\1\2";
  SMDiagnostic Error;
  std::unique_ptr<Module> Mod;
  EXPECT_DEATH(Mod = parseAssemblyString(Source.substr(0, Source.size() - 2),
                                         Error, Ctx),
               "Buffer is not null terminated!");
}

#endif
#endif

TEST(AsmParserTest, SlotMappingTest) {
  LLVMContext Ctx;
  StringRef Source = "@0 = global i32 0\n !0 = !{}\n !42 = !{i32 42}";
  SMDiagnostic Error;
  SlotMapping Mapping;
  auto Mod = parseAssemblyString(Source, Error, Ctx, &Mapping);

  EXPECT_TRUE(Mod != nullptr);
  EXPECT_TRUE(Error.getMessage().empty());

  ASSERT_EQ(Mapping.GlobalValues.size(), 1u);
  EXPECT_TRUE(isa<GlobalVariable>(Mapping.GlobalValues[0]));

  EXPECT_EQ(Mapping.MetadataNodes.size(), 2u);
  EXPECT_EQ(Mapping.MetadataNodes.count(0), 1u);
  EXPECT_EQ(Mapping.MetadataNodes.count(42), 1u);
  EXPECT_EQ(Mapping.MetadataNodes.count(1), 0u);
}

TEST(AsmParserTest, TypeAndConstantValueParsing) {
  LLVMContext Ctx;
  SMDiagnostic Error;
  StringRef Source = "define void @test() {\n  entry:\n  ret void\n}";
  auto Mod = parseAssemblyString(Source, Error, Ctx);
  ASSERT_TRUE(Mod != nullptr);
  auto &M = *Mod;

  const Value *V;
  V = parseConstantValue("double 3.5", Error, M);
  ASSERT_TRUE(V);
  EXPECT_TRUE(V->getType()->isDoubleTy());
  ASSERT_TRUE(isa<ConstantFP>(V));
  EXPECT_TRUE(cast<ConstantFP>(V)->isExactlyValue(3.5));

  V = parseConstantValue("i32 42", Error, M);
  ASSERT_TRUE(V);
  EXPECT_TRUE(V->getType()->isIntegerTy());
  ASSERT_TRUE(isa<ConstantInt>(V));
  EXPECT_TRUE(cast<ConstantInt>(V)->equalsInt(42));

  V = parseConstantValue("<4 x i32> <i32 0, i32 1, i32 2, i32 3>", Error, M);
  ASSERT_TRUE(V);
  EXPECT_TRUE(V->getType()->isVectorTy());
  ASSERT_TRUE(isa<ConstantDataVector>(V));

  V = parseConstantValue("i32 add (i32 1, i32 2)", Error, M);
  ASSERT_TRUE(V);
  ASSERT_TRUE(isa<ConstantInt>(V));

  V = parseConstantValue("ptr blockaddress(@test, %entry)", Error, M);
  ASSERT_TRUE(V);
  ASSERT_TRUE(isa<BlockAddress>(V));

  V = parseConstantValue("ptr undef", Error, M);
  ASSERT_TRUE(V);
  ASSERT_TRUE(isa<UndefValue>(V));

  EXPECT_FALSE(parseConstantValue("duble 3.25", Error, M));
  EXPECT_EQ(Error.getMessage(), "expected type");

  EXPECT_FALSE(parseConstantValue("i32 3.25", Error, M));
  EXPECT_EQ(Error.getMessage(), "floating point constant invalid for type");

  EXPECT_FALSE(parseConstantValue("ptr @foo", Error, M));
  EXPECT_EQ(Error.getMessage(), "expected a constant value");

  EXPECT_FALSE(parseConstantValue("i32 3, ", Error, M));
  EXPECT_EQ(Error.getMessage(), "expected end of string");
}

TEST(AsmParserTest, TypeAndConstantValueWithSlotMappingParsing) {
  LLVMContext Ctx;
  SMDiagnostic Error;
  StringRef Source =
      "%st = type { i32, i32 }\n"
      "@v = common global [50 x %st] zeroinitializer, align 16\n"
      "%0 = type { i32, i32, i32, i32 }\n"
      "@g = common global [50 x %0] zeroinitializer, align 16\n"
      "define void @marker4(i64 %d) {\n"
      "entry:\n"
      "  %conv = trunc i64 %d to i32\n"
      "  store i32 %conv, ptr getelementptr inbounds "
      "    ([50 x %st], ptr @v, i64 0, i64 1, i32 0), align 16\n"
      "  store i32 %conv, ptr getelementptr inbounds "
      "    ([50 x %0], ptr @g, i64 0, i64 1, i32 0), align 16\n"
      "  ret void\n"
      "}";
  SlotMapping Mapping;
  auto Mod = parseAssemblyString(Source, Error, Ctx, &Mapping);
  ASSERT_TRUE(Mod != nullptr);
  auto &M = *Mod;

  const Value *V;
  V = parseConstantValue("ptr getelementptr inbounds ([50 x %st], ptr "
                         "@v, i64 0, i64 1, i32 0)",
                         Error, M, &Mapping);
  ASSERT_TRUE(V);
  ASSERT_TRUE(isa<ConstantExpr>(V));

  V = parseConstantValue("ptr getelementptr inbounds ([50 x %0], ptr "
                         "@g, i64 0, i64 1, i32 0)",
                         Error, M, &Mapping);
  ASSERT_TRUE(V);
  ASSERT_TRUE(isa<ConstantExpr>(V));
}

TEST(AsmParserTest, TypeWithSlotMappingParsing) {
  LLVMContext Ctx;
  SMDiagnostic Error;
  StringRef Source =
      "%st = type { i32, i32 }\n"
      "@v = common global [50 x %st] zeroinitializer, align 16\n"
      "%0 = type { i32, i32, i32, i32 }\n"
      "@g = common global [50 x %0] zeroinitializer, align 16\n"
      "define void @marker4(i64 %d) {\n"
      "entry:\n"
      "  %conv = trunc i64 %d to i32\n"
      "  store i32 %conv, ptr getelementptr inbounds "
      "    ([50 x %st], ptr @v, i64 0, i64 0, i32 0), align 16\n"
      "  store i32 %conv, ptr getelementptr inbounds "
      "    ([50 x %0], ptr @g, i64 0, i64 0, i32 0), align 16\n"
      "  ret void\n"
      "}";
  SlotMapping Mapping;
  auto Mod = parseAssemblyString(Source, Error, Ctx, &Mapping);
  ASSERT_TRUE(Mod != nullptr);
  auto &M = *Mod;

  // Check we properly parse integer types.
  Type *Ty;
  Ty = parseType("i32", Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isIntegerTy());
  ASSERT_TRUE(Ty->getPrimitiveSizeInBits() == 32);

  // Check we properly parse integer types with exotic size.
  Ty = parseType("i13", Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isIntegerTy());
  ASSERT_TRUE(Ty->getPrimitiveSizeInBits() == 13);

  // Check we properly parse floating point types.
  Ty = parseType("float", Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isFloatTy());

  Ty = parseType("double", Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isDoubleTy());

  // Check we properly parse struct types.
  // Named struct.
  Ty = parseType("%st", Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isStructTy());

  // Check the details of the struct.
  StructType *ST = cast<StructType>(Ty);
  ASSERT_TRUE(ST->getNumElements() == 2);
  for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
    Ty = ST->getElementType(i);
    ASSERT_TRUE(Ty->isIntegerTy());
    ASSERT_TRUE(Ty->getPrimitiveSizeInBits() == 32);
  }

  // Anonymous struct.
  Ty = parseType("%0", Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isStructTy());

  // Check the details of the struct.
  ST = cast<StructType>(Ty);
  ASSERT_TRUE(ST->getNumElements() == 4);
  for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
    Ty = ST->getElementType(i);
    ASSERT_TRUE(Ty->isIntegerTy());
    ASSERT_TRUE(Ty->getPrimitiveSizeInBits() == 32);
  }

  // Check we properly parse vector types.
  Ty = parseType("<5 x i32>", Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isVectorTy());

  // Check the details of the vector.
  auto *VT = cast<FixedVectorType>(Ty);
  ASSERT_TRUE(VT->getNumElements() == 5);
  ASSERT_TRUE(VT->getPrimitiveSizeInBits().getFixedValue() == 160);
  Ty = VT->getElementType();
  ASSERT_TRUE(Ty->isIntegerTy());
  ASSERT_TRUE(Ty->getPrimitiveSizeInBits() == 32);

  // Opaque struct.
  Ty = parseType("%opaque", Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isStructTy());

  ST = cast<StructType>(Ty);
  ASSERT_TRUE(ST->isOpaque());

  // Check we properly parse pointer types.
  Ty = parseType("ptr", Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isPointerTy());

  PointerType *PT = cast<PointerType>(Ty);
  ASSERT_TRUE(PT->isOpaque());

  // Check that we reject types with garbage.
  Ty = parseType("i32 garbage", Error, M, &Mapping);
  ASSERT_TRUE(!Ty);
}

TEST(AsmParserTest, TypeAtBeginningWithSlotMappingParsing) {
  LLVMContext Ctx;
  SMDiagnostic Error;
  StringRef Source =
      "%st = type { i32, i32 }\n"
      "@v = common global [50 x %st] zeroinitializer, align 16\n"
      "%0 = type { i32, i32, i32, i32 }\n"
      "@g = common global [50 x %0] zeroinitializer, align 16\n"
      "define void @marker4(i64 %d) {\n"
      "entry:\n"
      "  %conv = trunc i64 %d to i32\n"
      "  store i32 %conv, ptr getelementptr inbounds "
      "    ([50 x %st], ptr @v, i64 0, i64 0, i32 0), align 16\n"
      "  store i32 %conv, ptr getelementptr inbounds "
      "    ([50 x %0], ptr @g, i64 0, i64 0, i32 0), align 16\n"
      "  ret void\n"
      "}";
  SlotMapping Mapping;
  auto Mod = parseAssemblyString(Source, Error, Ctx, &Mapping);
  ASSERT_TRUE(Mod != nullptr);
  auto &M = *Mod;
  unsigned Read;

  // Check we properly parse integer types.
  Type *Ty;
  Ty = parseTypeAtBeginning("i32", Read, Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isIntegerTy());
  ASSERT_TRUE(Ty->getPrimitiveSizeInBits() == 32);
  ASSERT_TRUE(Read == 3);

  // Check we properly parse integer types with exotic size.
  Ty = parseTypeAtBeginning("i13", Read, Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isIntegerTy());
  ASSERT_TRUE(Ty->getPrimitiveSizeInBits() == 13);
  ASSERT_TRUE(Read == 3);

  // Check we properly parse floating point types.
  Ty = parseTypeAtBeginning("float", Read, Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isFloatTy());
  ASSERT_TRUE(Read == 5);

  Ty = parseTypeAtBeginning("double", Read, Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isDoubleTy());
  ASSERT_TRUE(Read == 6);

  // Check we properly parse struct types.
  // Named struct.
  Ty = parseTypeAtBeginning("%st", Read, Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isStructTy());
  ASSERT_TRUE(Read == 3);

  // Check the details of the struct.
  StructType *ST = cast<StructType>(Ty);
  ASSERT_TRUE(ST->getNumElements() == 2);
  for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
    Ty = ST->getElementType(i);
    ASSERT_TRUE(Ty->isIntegerTy());
    ASSERT_TRUE(Ty->getPrimitiveSizeInBits() == 32);
  }

  // Anonymous struct.
  Ty = parseTypeAtBeginning("%0", Read, Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isStructTy());
  ASSERT_TRUE(Read == 2);

  // Check the details of the struct.
  ST = cast<StructType>(Ty);
  ASSERT_TRUE(ST->getNumElements() == 4);
  for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
    Ty = ST->getElementType(i);
    ASSERT_TRUE(Ty->isIntegerTy());
    ASSERT_TRUE(Ty->getPrimitiveSizeInBits() == 32);
  }

  // Check we properly parse vector types.
  Ty = parseTypeAtBeginning("<5 x i32>", Read, Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isVectorTy());
  ASSERT_TRUE(Read == 9);

  // Check the details of the vector.
  auto *VT = cast<FixedVectorType>(Ty);
  ASSERT_TRUE(VT->getNumElements() == 5);
  ASSERT_TRUE(VT->getPrimitiveSizeInBits().getFixedValue() == 160);
  Ty = VT->getElementType();
  ASSERT_TRUE(Ty->isIntegerTy());
  ASSERT_TRUE(Ty->getPrimitiveSizeInBits() == 32);

  // Opaque struct.
  Ty = parseTypeAtBeginning("%opaque", Read, Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isStructTy());
  ASSERT_TRUE(Read == 7);

  ST = cast<StructType>(Ty);
  ASSERT_TRUE(ST->isOpaque());

  // Check we properly parse pointer types.
  // One indirection.
  Ty = parseTypeAtBeginning("ptr", Read, Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isPointerTy());
  ASSERT_TRUE(Read == 3);

  PointerType *PT = cast<PointerType>(Ty);
  ASSERT_TRUE(PT->isOpaque());

  // Check that we reject types with garbage.
  Ty = parseTypeAtBeginning("i32 garbage", Read, Error, M, &Mapping);
  ASSERT_TRUE(Ty);
  ASSERT_TRUE(Ty->isIntegerTy());
  ASSERT_TRUE(Ty->getPrimitiveSizeInBits() == 32);
  // We go to the next token, i.e., we read "i32" + ' '.
  ASSERT_TRUE(Read == 4);
}

TEST(AsmParserTest, InvalidDataLayoutStringCallback) {
  LLVMContext Ctx;
  SMDiagnostic Error;
  // Note the invalid i8:7 part
  // Overalign i32 as marker so we can check that indeed this DL was used,
  // and not some default.
  StringRef InvalidDLStr =
      "e-m:e-p:64:64-i8:7-i16:16-i32:64-i64:64-f80:128-n8:16:32:64";
  StringRef FixedDLStr =
      "e-m:e-p:64:64-i8:8-i16:16-i32:64-i64:64-f80:128-n8:16:32:64";
  Expected<DataLayout> ExpectedFixedDL = DataLayout::parse(FixedDLStr);
  ASSERT_TRUE(!ExpectedFixedDL.takeError());
  DataLayout FixedDL = ExpectedFixedDL.get();
  std::string Source = ("target datalayout = \"" + InvalidDLStr + "\"\n").str();
  MemoryBufferRef SourceBuffer(Source, "<string>");

  // Check that we reject the source without a DL override.
  SlotMapping Mapping1;
  auto Mod1 = parseAssembly(SourceBuffer, Error, Ctx, &Mapping1);
  EXPECT_TRUE(Mod1 == nullptr);

  // Check that we pass the correct DL str to the callback,
  // that fixing the DL str from the callback works,
  // and that the resulting module has the correct DL.
  SlotMapping Mapping2;
  auto Mod2 = parseAssembly(
      SourceBuffer, Error, Ctx, &Mapping2,
      [&](StringRef Triple, StringRef DLStr) -> std::optional<std::string> {
        EXPECT_EQ(DLStr, InvalidDLStr);
        return std::string{FixedDLStr};
      });
  ASSERT_TRUE(Mod2 != nullptr);
  EXPECT_EQ(Mod2->getDataLayout(), FixedDL);
}

} // end anonymous namespace
