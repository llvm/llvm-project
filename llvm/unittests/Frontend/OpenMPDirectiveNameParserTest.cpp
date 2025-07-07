//===- llvm/unittests/Frontend/OpenMPDirectiveNameParserTest.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/DirectiveNameParser.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "gtest/gtest.h"

#include <cctype>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace llvm;

static const omp::DirectiveNameParser &getParser() {
  static omp::DirectiveNameParser Parser(omp::SourceLanguage::C |
                                         omp::SourceLanguage::Fortran);
  return Parser;
}

static std::vector<std::string> tokenize(StringRef S) {
  std::vector<std::string> Tokens;

  using TokenIterator = std::istream_iterator<std::string>;
  std::string Copy = S.str();
  std::istringstream Stream(Copy);

  for (auto I = TokenIterator(Stream), E = TokenIterator(); I != E; ++I)
    Tokens.push_back(*I);
  return Tokens;
}

static std::string &prepareParamName(std::string &Name) {
  for (size_t I = 0, E = Name.size(); I != E; ++I) {
    // The parameter name must only have alphanumeric characters.
    if (!isalnum(Name[I]))
      Name[I] = 'X';
  }
  return Name;
}

namespace llvm {
template <> struct enum_iteration_traits<omp::Directive> {
  static constexpr bool is_iterable = true;
};
} // namespace llvm

// Test tokenizing.

class Tokenize : public testing::TestWithParam<omp::Directive> {};

static bool isEqual(const SmallVector<StringRef> &A,
                    const std::vector<std::string> &B) {
  if (A.size() != B.size())
    return false;

  for (size_t I = 0, E = A.size(); I != E; ++I) {
    if (A[I] != StringRef(B[I]))
      return false;
  }
  return true;
}

TEST_P(Tokenize, T) {
  omp::Directive DirId = GetParam();
  StringRef Name = omp::getOpenMPDirectiveName(DirId, omp::FallbackVersion);

  SmallVector<StringRef> tokens1 = omp::DirectiveNameParser::tokenize(Name);
  std::vector<std::string> tokens2 = tokenize(Name);
  ASSERT_TRUE(isEqual(tokens1, tokens2));
}

static std::string
getParamName1(const testing::TestParamInfo<Tokenize::ParamType> &Info) {
  omp::Directive DirId = Info.param;
  std::string Name =
      omp::getOpenMPDirectiveName(DirId, omp::FallbackVersion).str();
  return prepareParamName(Name);
}

INSTANTIATE_TEST_SUITE_P(
    DirectiveNameParserTest, Tokenize,
    testing::ValuesIn(
        llvm::enum_seq(static_cast<omp::Directive>(0),
                       static_cast<omp::Directive>(omp::Directive_enumSize))),
    getParamName1);

// Test parsing of valid names.

using ValueType = std::tuple<omp::Directive, unsigned>;

class ParseValid : public testing::TestWithParam<ValueType> {};

TEST_P(ParseValid, T) {
  auto [DirId, Version] = GetParam();
  if (DirId == omp::Directive::OMPD_unknown)
    return;

  std::string Name = omp::getOpenMPDirectiveName(DirId, Version).str();

  // Tokenize and parse
  auto &Parser = getParser();
  auto *State = Parser.initial();
  ASSERT_TRUE(State != nullptr);

  std::vector<std::string> Tokens = tokenize(Name);
  for (auto &Tok : Tokens) {
    State = Parser.consume(State, Tok);
    ASSERT_TRUE(State != nullptr);
  }

  ASSERT_EQ(State->Value, DirId);
}

static std::string
getParamName2(const testing::TestParamInfo<ParseValid::ParamType> &Info) {
  auto [DirId, Version] = Info.param;
  std::string Name = omp::getOpenMPDirectiveName(DirId, Version).str() + "v" +
                     std::to_string(Version);
  return prepareParamName(Name);
}

INSTANTIATE_TEST_SUITE_P(
    DirectiveNameParserTest, ParseValid,
    testing::Combine(testing::ValuesIn(llvm::enum_seq(
                         static_cast<omp::Directive>(0),
                         static_cast<omp::Directive>(omp::Directive_enumSize))),
                     testing::ValuesIn(omp::getOpenMPVersions())),
    getParamName2);

// Test parsing of invalid names

class ParseInvalid : public testing::TestWithParam<std::string> {};

TEST_P(ParseInvalid, T) {
  std::string Name = GetParam();

  auto &Parser = getParser();
  auto *State = Parser.initial();
  ASSERT_TRUE(State != nullptr);

  std::vector<std::string> Tokens = tokenize(Name);
  for (auto &Tok : Tokens)
    State = Parser.consume(State, Tok);

  ASSERT_TRUE(State == nullptr || State->Value == omp::Directive::OMPD_unknown);
}

namespace {
using namespace std;

INSTANTIATE_TEST_SUITE_P(DirectiveNameParserTest, ParseInvalid,
                         testing::Values(
                             // Names that contain invalid tokens
                             "bad"s, "target teams invalid"s,
                             "target sections parallel"s,
                             "target teams distribute parallel for wrong"s,
                             // Valid beginning, but not a complete name
                             "begin declare"s,
                             // Complete name with extra tokens
                             "distribute simd target"s));
} // namespace
