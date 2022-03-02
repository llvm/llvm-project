//===- EnumsGenTest.cpp - TableGen EnumsGen Tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include "gmock/gmock.h"

#include <type_traits>

/// Pull in generated enum utility declarations and definitions.
#include "EnumsGenTest.h.inc"

#include "EnumsGenTest.cpp.inc"

/// Test namespaces and enum class/utility names.
using Outer::Inner::ConvertToEnum;
using Outer::Inner::ConvertToString;
using Outer::Inner::FooEnum;
using Outer::Inner::FooEnumAttr;

TEST(EnumsGenTest, GeneratedStrEnumDefinition) {
  EXPECT_EQ(0u, static_cast<uint64_t>(FooEnum::CaseA));
  EXPECT_EQ(1u, static_cast<uint64_t>(FooEnum::CaseB));
}

TEST(EnumsGenTest, GeneratedI32EnumDefinition) {
  EXPECT_EQ(5u, static_cast<uint64_t>(I32Enum::Case5));
  EXPECT_EQ(10u, static_cast<uint64_t>(I32Enum::Case10));
}

TEST(EnumsGenTest, GeneratedDenseMapInfo) {
  llvm::DenseMap<FooEnum, std::string> myMap;

  myMap[FooEnum::CaseA] = "zero";
  myMap[FooEnum::CaseB] = "one";

  EXPECT_EQ(myMap[FooEnum::CaseA], "zero");
  EXPECT_EQ(myMap[FooEnum::CaseB], "one");
}

TEST(EnumsGenTest, GeneratedSymbolToStringFn) {
  EXPECT_EQ(ConvertToString(FooEnum::CaseA), "CaseA");
  EXPECT_EQ(ConvertToString(FooEnum::CaseB), "CaseB");
}

TEST(EnumsGenTest, GeneratedStringToSymbolFn) {
  EXPECT_EQ(llvm::Optional<FooEnum>(FooEnum::CaseA), ConvertToEnum("CaseA"));
  EXPECT_EQ(llvm::Optional<FooEnum>(FooEnum::CaseB), ConvertToEnum("CaseB"));
  EXPECT_EQ(llvm::None, ConvertToEnum("X"));
}

TEST(EnumsGenTest, GeneratedUnderlyingType) {
  bool v = std::is_same<uint32_t, std::underlying_type<I32Enum>::type>::value;
  EXPECT_TRUE(v);
}

TEST(EnumsGenTest, GeneratedBitEnumDefinition) {
  EXPECT_EQ(0u, static_cast<uint32_t>(BitEnumWithNone::None));
  EXPECT_EQ(1u, static_cast<uint32_t>(BitEnumWithNone::Bit0));
  EXPECT_EQ(8u, static_cast<uint32_t>(BitEnumWithNone::Bit3));
}

TEST(EnumsGenTest, GeneratedSymbolToStringFnForBitEnum) {
  EXPECT_EQ(stringifyBitEnumWithNone(BitEnumWithNone::None), "None");
  EXPECT_EQ(stringifyBitEnumWithNone(BitEnumWithNone::Bit0), "Bit0");
  EXPECT_EQ(stringifyBitEnumWithNone(BitEnumWithNone::Bit3), "Bit3");
  EXPECT_EQ(
      stringifyBitEnumWithNone(BitEnumWithNone::Bit0 | BitEnumWithNone::Bit3),
      "Bit0|Bit3");
  EXPECT_EQ(2u, static_cast<uint64_t>(BitEnum64_Test::Bit1));
  EXPECT_EQ(144115188075855872u, static_cast<uint64_t>(BitEnum64_Test::Bit57));
}

TEST(EnumsGenTest, GeneratedStringToSymbolForBitEnum) {
  EXPECT_EQ(symbolizeBitEnumWithNone("None"), BitEnumWithNone::None);
  EXPECT_EQ(symbolizeBitEnumWithNone("Bit0"), BitEnumWithNone::Bit0);
  EXPECT_EQ(symbolizeBitEnumWithNone("Bit3"), BitEnumWithNone::Bit3);
  EXPECT_EQ(symbolizeBitEnumWithNone("Bit3|Bit0"),
            BitEnumWithNone::Bit3 | BitEnumWithNone::Bit0);

  EXPECT_EQ(symbolizeBitEnumWithNone("Bit2"), llvm::None);
  EXPECT_EQ(symbolizeBitEnumWithNone("Bit3|Bit4"), llvm::None);

  EXPECT_EQ(symbolizeBitEnumWithoutNone("None"), llvm::None);
}

TEST(EnumsGenTest, GeneratedSymbolToStringFnForGroupedBitEnum) {
  EXPECT_EQ(stringifyBitEnumWithGroup(BitEnumWithGroup::Bit0), "Bit0");
  EXPECT_EQ(stringifyBitEnumWithGroup(BitEnumWithGroup::Bit3), "Bit3");
  EXPECT_EQ(stringifyBitEnumWithGroup(BitEnumWithGroup::Bits0To3),
            "Bit0|Bit1|Bit2|Bit3|Bits0To3");
  EXPECT_EQ(stringifyBitEnumWithGroup(BitEnumWithGroup::Bit4), "Bit4");
  EXPECT_EQ(stringifyBitEnumWithGroup(
                BitEnumWithGroup::Bit0 | BitEnumWithGroup::Bit1 |
                BitEnumWithGroup::Bit2 | BitEnumWithGroup::Bit4),
            "Bit0|Bit1|Bit2|Bit4");
}

TEST(EnumsGenTest, GeneratedStringToSymbolForGroupedBitEnum) {
  EXPECT_EQ(symbolizeBitEnumWithGroup("Bit0"), BitEnumWithGroup::Bit0);
  EXPECT_EQ(symbolizeBitEnumWithGroup("Bit3"), BitEnumWithGroup::Bit3);
  EXPECT_EQ(symbolizeBitEnumWithGroup("Bit5"), llvm::None);
  EXPECT_EQ(symbolizeBitEnumWithGroup("Bit3|Bit0"),
            BitEnumWithGroup::Bit3 | BitEnumWithGroup::Bit0);
}

TEST(EnumsGenTest, GeneratedOperator) {
  EXPECT_TRUE(bitEnumContains(BitEnumWithNone::Bit0 | BitEnumWithNone::Bit3,
                              BitEnumWithNone::Bit0));
  EXPECT_FALSE(bitEnumContains(BitEnumWithNone::Bit0 & BitEnumWithNone::Bit3,
                               BitEnumWithNone::Bit0));
}

TEST(EnumsGenTest, GeneratedSymbolToCustomStringFn) {
  EXPECT_EQ(stringifyPrettyIntEnum(PrettyIntEnum::Case1), "case_one");
  EXPECT_EQ(stringifyPrettyIntEnum(PrettyIntEnum::Case2), "case_two");
}

TEST(EnumsGenTest, GeneratedCustomStringToSymbolFn) {
  auto one = symbolizePrettyIntEnum("case_one");
  EXPECT_TRUE(one);
  EXPECT_EQ(*one, PrettyIntEnum::Case1);

  auto two = symbolizePrettyIntEnum("case_two");
  EXPECT_TRUE(two);
  EXPECT_EQ(*two, PrettyIntEnum::Case2);

  auto none = symbolizePrettyIntEnum("Case1");
  EXPECT_FALSE(none);
}

TEST(EnumsGenTest, GeneratedIntAttributeClass) {
  mlir::MLIRContext ctx;
  I32Enum rawVal = I32Enum::Case5;

  I32EnumAttr enumAttr = I32EnumAttr::get(&ctx, rawVal);
  EXPECT_NE(enumAttr, nullptr);
  EXPECT_EQ(enumAttr.getValue(), rawVal);

  mlir::Type intType = mlir::IntegerType::get(&ctx, 32);
  mlir::Attribute intAttr = mlir::IntegerAttr::get(intType, 5);
  EXPECT_TRUE(intAttr.isa<I32EnumAttr>());
  EXPECT_EQ(intAttr, enumAttr);
}

TEST(EnumsGenTest, GeneratedBitAttributeClass) {
  mlir::MLIRContext ctx;

  mlir::Type intType = mlir::IntegerType::get(&ctx, 32);
  mlir::Attribute intAttr = mlir::IntegerAttr::get(
      intType,
      static_cast<uint32_t>(BitEnumWithNone::Bit0 | BitEnumWithNone::Bit3));
  EXPECT_TRUE(intAttr.isa<BitEnumWithNoneAttr>());
  EXPECT_TRUE(intAttr.isa<BitEnumWithoutNoneAttr>());

  intAttr = mlir::IntegerAttr::get(
      intType, static_cast<uint32_t>(BitEnumWithGroup::Bits0To3) | (1u << 6));
  EXPECT_FALSE(intAttr.isa<BitEnumWithGroupAttr>());
}
