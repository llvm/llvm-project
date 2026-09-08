//===------- Offload API tests - olIterateSymbols -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

// The 'global' program contains the 'read' and 'write' kernels as well as the
// 'global' global variable.
struct olIterateSymbolsTest : OffloadProgramTest {
  void SetUp() override { SetUpWith("global"); }
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olIterateSymbolsTest);

using SymbolVecT = std::vector<ol_symbol_handle_t>;

static bool collectSymbol(ol_symbol_handle_t Symbol, void *Data) {
  static_cast<SymbolVecT *>(Data)->push_back(Symbol);
  return true;
}

static std::string getName(ol_symbol_handle_t Symbol) {
  size_t Size = 0;
  if (olGetSymbolInfoSize(Symbol, OL_SYMBOL_INFO_NAME, &Size))
    return {};
  std::vector<char> Name(Size);
  if (olGetSymbolInfo(Symbol, OL_SYMBOL_INFO_NAME, Size, Name.data()))
    return {};
  return std::string{Name.data()};
}

static std::vector<std::string> getNames(const SymbolVecT &Symbols,
                                         ol_symbol_kind_t Expected) {
  std::vector<std::string> Names;
  for (auto *Symbol : Symbols) {
    ol_symbol_kind_t Kind;
    if (olGetSymbolInfo(Symbol, OL_SYMBOL_INFO_KIND, sizeof(Kind), &Kind))
      return {};
    EXPECT_EQ(Kind, Expected);
    Names.push_back(getName(Symbol));
  }
  return Names;
}

static bool contains(const std::vector<std::string> &Names,
                     const std::string &Name) {
  return std::find(Names.begin(), Names.end(), Name) != Names.end();
}

TEST_P(olIterateSymbolsTest, SuccessKernels) {
  SymbolVecT Symbols;
  ASSERT_SUCCESS(olIterateSymbols(Program, OL_SYMBOL_KIND_KERNEL, collectSymbol,
                                  &Symbols));

  auto Names = getNames(Symbols, OL_SYMBOL_KIND_KERNEL);
  ASSERT_TRUE(contains(Names, "read"));
  ASSERT_TRUE(contains(Names, "write"));
}

TEST_P(olIterateSymbolsTest, SuccessGlobals) {
  SymbolVecT Symbols;
  ASSERT_SUCCESS_OR_UNSUPPORTED(olIterateSymbols(
      Program, OL_SYMBOL_KIND_GLOBAL_VARIABLE, collectSymbol, &Symbols));

  auto Names = getNames(Symbols, OL_SYMBOL_KIND_GLOBAL_VARIABLE);
  ASSERT_TRUE(contains(Names, "global"));
}

TEST_P(olIterateSymbolsTest, SuccessSameSymbol) {
  SymbolVecT Symbols;
  ASSERT_SUCCESS(olIterateSymbols(Program, OL_SYMBOL_KIND_KERNEL, collectSymbol,
                                  &Symbols));
  ASSERT_FALSE(Symbols.empty());

  for (auto *Symbol : Symbols) {
    ol_symbol_handle_t Looked = nullptr;
    ASSERT_SUCCESS(olGetSymbol(Program, getName(Symbol).c_str(),
                               OL_SYMBOL_KIND_KERNEL, &Looked));
    ASSERT_EQ(Symbol, Looked);
  }
}

TEST_P(olIterateSymbolsTest, SuccessStopIteration) {
  size_t Count = 0;
  ASSERT_SUCCESS(olIterateSymbols(
      Program, OL_SYMBOL_KIND_KERNEL,
      [](ol_symbol_handle_t, void *Data) {
        (*static_cast<size_t *>(Data))++;
        return false;
      },
      &Count));
  ASSERT_EQ(Count, 1u);
}

TEST_P(olIterateSymbolsTest, InvalidNullProgram) {
  SymbolVecT Symbols;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olIterateSymbols(nullptr, OL_SYMBOL_KIND_KERNEL, collectSymbol,
                                &Symbols));
}

TEST_P(olIterateSymbolsTest, InvalidNullCallback) {
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olIterateSymbols(Program, OL_SYMBOL_KIND_KERNEL, nullptr, nullptr));
}

TEST_P(olIterateSymbolsTest, InvalidKind) {
  SymbolVecT Symbols;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olIterateSymbols(Program, OL_SYMBOL_KIND_FORCE_UINT32,
                                collectSymbol, &Symbols));
}
