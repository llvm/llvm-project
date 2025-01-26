//===-- DILLexerTests.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILLexer.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <string>

using llvm::StringRef;

bool VerifyExpectedTokens(
    lldb_private::dil::DILLexer &lexer,
    std::vector<std::pair<lldb_private::dil::Token::Kind, std::string>>
        exp_tokens,
    uint32_t start_pos) {
  if (lexer.NumLexedTokens() - start_pos < exp_tokens.size())
    return false;

  if (start_pos > 0)
    lexer.ResetTokenIdx(start_pos -
                        1); // GetNextToken increments the idx first.
  for (const auto &pair : exp_tokens) {
    lldb_private::dil::Token token = lexer.GetNextToken();
    if (token.GetKind() != pair.first || token.GetSpelling() != pair.second)
      return false;
  }

  return true;
}

TEST(DILLexerTests, SimpleTest) {
  StringRef input_expr("simple_var");
  uint32_t tok_len = 10;
  lldb_private::dil::DILLexer lexer(input_expr);
  lldb_private::dil::Token token;
  token.SetKind(lldb_private::dil::Token::unknown);
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::unknown);
  auto success = lexer.LexAll();

  if (!success) {
    EXPECT_TRUE(false);
  }
  token = lexer.GetNextToken();
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::identifier);
  EXPECT_EQ(token.GetSpelling(), "simple_var");
  EXPECT_EQ(token.GetLength(), tok_len);
  token = lexer.GetNextToken();
  ;
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::eof);
}

TEST(DILLexerTests, TokenKindTest) {
  StringRef input_expr("namespace");
  lldb_private::dil::DILLexer lexer(input_expr);
  lldb_private::dil::Token token;
  token.SetKind(lldb_private::dil::Token::unknown);

  auto success = lexer.LexAll();
  if (!success) {
    EXPECT_TRUE(false);
  }
  token = lexer.GetNextToken();

  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::kw_namespace);
  EXPECT_TRUE(token.IsNot(lldb_private::dil::Token::identifier));
  EXPECT_FALSE(token.Is(lldb_private::dil::Token::l_paren));
  EXPECT_TRUE(token.IsOneOf(lldb_private::dil::Token::eof,
                            lldb_private::dil::Token::kw_namespace));
  EXPECT_FALSE(token.IsOneOf(
      lldb_private::dil::Token::l_paren, lldb_private::dil::Token::r_paren,
      lldb_private::dil::Token::coloncolon, lldb_private::dil::Token::eof));

  token.SetKind(lldb_private::dil::Token::identifier);
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::identifier);
}

TEST(DILLexerTests, LookAheadTest) {
  StringRef input_expr("(anonymous namespace)::some_var");
  lldb_private::dil::DILLexer lexer(input_expr);
  lldb_private::dil::Token token;
  token.SetKind(lldb_private::dil::Token::unknown);
  uint32_t expect_loc = 23;

  auto success = lexer.LexAll();
  if (!success) {
    EXPECT_TRUE(false);
  }
  token = lexer.GetNextToken();

  // Current token is '('; check the next 4 tokens, to make
  // sure they are the identifier 'anonymous', the namespace keyword,
  // ')' and '::', in that order.
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::l_paren);
  EXPECT_EQ(lexer.LookAhead(0).GetKind(), lldb_private::dil::Token::identifier);
  EXPECT_EQ(lexer.LookAhead(0).GetSpelling(), "anonymous");
  EXPECT_EQ(lexer.LookAhead(1).GetKind(),
            lldb_private::dil::Token::kw_namespace);
  EXPECT_EQ(lexer.LookAhead(2).GetKind(), lldb_private::dil::Token::r_paren);
  EXPECT_EQ(lexer.LookAhead(3).GetKind(), lldb_private::dil::Token::coloncolon);

  // Our current index should still be 0, as we only looked ahead; we are still
  // officially on the '('.
  EXPECT_EQ(lexer.GetCurrentTokenIdx(), (uint32_t)0);

  // Accept the 'lookahead', so our current token is '::', which has the index
  // 4 in our vector of tokens (which starts at zero).
  token = lexer.AcceptLookAhead(3);
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::coloncolon);
  EXPECT_EQ(lexer.GetCurrentTokenIdx(), (uint32_t)4);

  token = lexer.GetNextToken();
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::identifier);
  EXPECT_EQ(token.GetSpelling(), "some_var");
  EXPECT_EQ(lexer.GetCurrentTokenIdx(), (uint32_t)5);
  // Verify we've advanced our position counter (lexing location) in the
  // input 23 characters (the length of '(anonymous namespace)::'.
  EXPECT_EQ(token.GetLocation(), expect_loc);
  token = lexer.GetNextToken();
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::eof);
}

TEST(DILLexerTests, MultiTokenLexTest) {
  StringRef input_expr("This string has (several ) ::identifiers");
  lldb_private::dil::DILLexer lexer(input_expr);
  lldb_private::dil::Token token;
  token.SetKind(lldb_private::dil::Token::unknown);

  auto success = lexer.LexAll();
  if (!success) {
    EXPECT_TRUE(false);
  }

  std::vector<std::pair<lldb_private::dil::Token::Kind, std::string>>
      expected_tokens = {
          {lldb_private::dil::Token::identifier, "This"},
          {lldb_private::dil::Token::identifier, "string"},
          {lldb_private::dil::Token::identifier, "has"},
          {lldb_private::dil::Token::l_paren, "("},
          {lldb_private::dil::Token::identifier, "several"},
          {lldb_private::dil::Token::r_paren, ")"},
          {lldb_private::dil::Token::coloncolon, "::"},
          {lldb_private::dil::Token::identifier, "identifiers"},
      };

  EXPECT_TRUE(VerifyExpectedTokens(lexer, expected_tokens, 0));

  token = lexer.GetNextToken();
  EXPECT_EQ(token.GetSpelling(), "");
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::eof);
}

TEST(DILLexerTests, IdentifiersTest) {
  std::vector<std::string> valid_identifiers = {
    "$My_name1",
    "$pc",
    "abcd",
    "ab cd",
    "_",
    "_a",
    "_a_",
    "a_b",
    "this",
    "self",
    "a",
    "MyName"
  };
  std::vector<std::string> invalid_identifiers = {
    "234",
    "2a",
    "2",
    "$",
    "1MyName",
    "",
    "namespace"
  };

  // Verify that all of the valid identifiers come out as identifier tokens.
  for (auto &str : valid_identifiers) {
    SCOPED_TRACE(str);
    lldb_private::dil::DILLexer lexer(str);
    lldb_private::dil::Token token;
    token.SetKind(lldb_private::dil::Token::unknown);

    auto success = lexer.LexAll();
    if (!success) {
      EXPECT_TRUE(false);
    }
    token = lexer.GetNextToken();
    EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::identifier);
  }

  // Verify that none of the invalid identifiers come out as identifier tokens.
  for (auto &str : invalid_identifiers) {
    SCOPED_TRACE(str);
    lldb_private::dil::DILLexer lexer(str);
    lldb_private::dil::Token token;
    token.SetKind(lldb_private::dil::Token::unknown);

    auto success = lexer.LexAll();
    // In this case, it's ok for Lex() to return an error.
    if (!success) {
      llvm::consumeError(success.takeError());
    } else {
      // We didn't get an error; make sure we did not get an identifier token.
      token = lexer.GetNextToken();
      EXPECT_TRUE(token.IsNot(lldb_private::dil::Token::identifier));
      EXPECT_TRUE(token.IsOneOf(lldb_private::dil::Token::unknown,
                                lldb_private::dil::Token::none,
                                lldb_private::dil::Token::eof,
                                lldb_private::dil::Token::kw_namespace));
    }
  }
}
