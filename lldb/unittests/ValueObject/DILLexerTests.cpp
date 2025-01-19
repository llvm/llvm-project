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

TEST(DILLexerTests, SimpleTest) {
  StringRef dil_input_expr("simple_var");
  uint32_t tok_len = 10;
  lldb_private::dil::DILLexer dil_lexer(dil_input_expr);
  lldb_private::dil::DILToken dil_token;
  dil_token.setKind(lldb_private::dil::TokenKind::unknown);
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::unknown);
  dil_lexer.Lex(dil_token);
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::identifier);
  EXPECT_EQ(dil_token.getSpelling(), "simple_var");
  EXPECT_EQ(dil_token.getLength(), tok_len);
  dil_lexer.Lex(dil_token);
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::eof);
}

TEST(DILLexerTests, TokenKindTest) {
  StringRef dil_input_expr("namespace");
  lldb_private::dil::DILLexer dil_lexer(dil_input_expr);
  lldb_private::dil::DILToken dil_token;
  dil_token.setKind(lldb_private::dil::TokenKind::unknown);

  dil_lexer.Lex(dil_token);
  EXPECT_EQ(dil_lexer.GetCurrentTokenIdx(), UINT_MAX);
  dil_lexer.ResetTokenIdx(0);

  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::kw_namespace);
  EXPECT_TRUE(dil_token.isNot(lldb_private::dil::TokenKind::identifier));
  EXPECT_FALSE(dil_token.is(lldb_private::dil::TokenKind::l_paren));
  EXPECT_TRUE(dil_token.isOneOf(lldb_private::dil::TokenKind::eof,
                                lldb_private::dil::TokenKind::kw_namespace));
  EXPECT_FALSE(dil_token.isOneOf(lldb_private::dil::TokenKind::l_paren,
                                 lldb_private::dil::TokenKind::r_paren,
                                 lldb_private::dil::TokenKind::coloncolon,
                                 lldb_private::dil::TokenKind::eof));

  dil_token.setKind(lldb_private::dil::TokenKind::identifier);
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::identifier);
}

TEST(DILLexerTests, LookAheadTest) {
  StringRef dil_input_expr("(anonymous namespace)::some_var");
  lldb_private::dil::DILLexer dil_lexer(dil_input_expr);
  lldb_private::dil::DILToken dil_token;
  dil_token.setKind(lldb_private::dil::TokenKind::unknown);
  uint32_t expect_loc = 23;

  dil_lexer.Lex(dil_token);
  EXPECT_EQ(dil_lexer.GetCurrentTokenIdx(), UINT_MAX);
  dil_lexer.ResetTokenIdx(0);

  // Current token is '('; check the next 4 tokens, to make
  // sure they are the identifier 'anonymous', the namespace keyword,
  // ')' and '::', in that order.
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::l_paren);
  EXPECT_EQ(dil_lexer.LookAhead(0).getKind(),
            lldb_private::dil::TokenKind::identifier);
  EXPECT_EQ(dil_lexer.LookAhead(0).getSpelling(), "anonymous");
  EXPECT_EQ(dil_lexer.LookAhead(1).getKind(),
            lldb_private::dil::TokenKind::kw_namespace);
  EXPECT_EQ(dil_lexer.LookAhead(2).getKind(),
            lldb_private::dil::TokenKind::r_paren);
  EXPECT_EQ(dil_lexer.LookAhead(3).getKind(),
            lldb_private::dil::TokenKind::coloncolon);
  // Verify we've advanced our position counter (lexing location) in the
  // input 23 characters (the length of '(anonymous namespace)::'.
  EXPECT_EQ(dil_lexer.GetLocation(), expect_loc);

  // Our current index should still be 0, as we only looked ahead; we are still
  // officially on the '('.
  EXPECT_EQ(dil_lexer.GetCurrentTokenIdx(), 0);

  // Accept the 'lookahead', so our current token is '::', which has the index
  // 4 in our vector of tokens (which starts at zero).
  dil_token = dil_lexer.AcceptLookAhead(3);
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::coloncolon);
  EXPECT_EQ(dil_lexer.GetCurrentTokenIdx(), 4);

  // Lex the final variable name in the input string
  dil_lexer.Lex(dil_token);
  dil_lexer.IncrementTokenIdx();
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::identifier);
  EXPECT_EQ(dil_token.getSpelling(), "some_var");
  EXPECT_EQ(dil_lexer.GetCurrentTokenIdx(), 5);

  dil_lexer.Lex(dil_token);
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::eof);
}

TEST(DILLexerTests, MultiTokenLexTest) {
  StringRef dil_input_expr("This string has several identifiers");
  lldb_private::dil::DILLexer dil_lexer(dil_input_expr);
  lldb_private::dil::DILToken dil_token;
  dil_token.setKind(lldb_private::dil::TokenKind::unknown);

  dil_lexer.Lex(dil_token);
  EXPECT_EQ(dil_lexer.GetCurrentTokenIdx(), UINT_MAX);
  dil_lexer.ResetTokenIdx(0);

  EXPECT_EQ(dil_token.getSpelling(), "This");
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::identifier);
  dil_lexer.Lex(dil_token);
  dil_lexer.IncrementTokenIdx();

  EXPECT_EQ(dil_token.getSpelling(), "string");
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::identifier);
  dil_lexer.Lex(dil_token);
  dil_lexer.IncrementTokenIdx();

  EXPECT_EQ(dil_token.getSpelling(), "has");
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::identifier);
  dil_lexer.Lex(dil_token);
  dil_lexer.IncrementTokenIdx();

  EXPECT_EQ(dil_token.getSpelling(), "several");
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::identifier);
  dil_lexer.Lex(dil_token);
  dil_lexer.IncrementTokenIdx();

  EXPECT_EQ(dil_token.getSpelling(), "identifiers");
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::identifier);
  dil_lexer.Lex(dil_token);
  dil_lexer.IncrementTokenIdx();

  EXPECT_EQ(dil_token.getSpelling(), "");
  EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::eof);
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
  for (auto str : valid_identifiers) {
    StringRef dil_input_expr(str);
    lldb_private::dil::DILLexer dil_lexer(dil_input_expr);
    lldb_private::dil::DILToken dil_token;
    dil_token.setKind(lldb_private::dil::TokenKind::unknown);

    dil_lexer.Lex(dil_token);
    EXPECT_EQ(dil_token.getKind(), lldb_private::dil::TokenKind::identifier);
  }

  // Verify that none of the invalid identifiers come out as identifier tokens.
  for (auto str : invalid_identifiers) {
    StringRef dil_input_expr(str);
    lldb_private::dil::DILLexer dil_lexer(dil_input_expr);
    lldb_private::dil::DILToken dil_token;
    dil_token.setKind(lldb_private::dil::TokenKind::unknown);

    dil_lexer.Lex(dil_token);
    EXPECT_TRUE(dil_token.isNot(lldb_private::dil::TokenKind::identifier));
    EXPECT_TRUE(dil_token.isOneOf(lldb_private::dil::TokenKind::unknown,
                                  lldb_private::dil::TokenKind::none,
                                  lldb_private::dil::TokenKind::eof,
                                  lldb_private::dil::TokenKind::kw_namespace));
  }
}
