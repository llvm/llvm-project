//===-- DILLexerTests.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILLexer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <string>

using llvm::StringRef;

std::vector<std::pair<lldb_private::dil::Token::Kind, std::string>>
ExtractTokenData(lldb_private::dil::DILLexer &lexer) {
  std::vector<std::pair<lldb_private::dil::Token::Kind, std::string>> data;
  if (lexer.NumLexedTokens() == 0)
    return data;

  lexer.ResetTokenIdx(UINT_MAX);
  do {
    lexer.Advance();
    lldb_private::dil::Token tok = lexer.GetCurrentToken();
    data.push_back(std::make_pair(tok.GetKind(), tok.GetSpelling()));
  } while (data.back().first != lldb_private::dil::Token::eof);
  return data;
}

TEST(DILLexerTests, SimpleTest) {
  StringRef input_expr("simple_var");
  uint32_t tok_len = 10;
  llvm::Expected<lldb_private::dil::DILLexer> maybe_lexer =
      lldb_private::dil::DILLexer::Create(input_expr);
  ASSERT_THAT_EXPECTED(maybe_lexer, llvm::Succeeded());
  lldb_private::dil::DILLexer lexer(*maybe_lexer);
  lldb_private::dil::Token token =
      lldb_private::dil::Token(lldb_private::dil::Token::unknown, "", 0);
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::unknown);

  lexer.Advance();
  token = lexer.GetCurrentToken();
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::identifier);
  EXPECT_EQ(token.GetSpelling(), "simple_var");
  EXPECT_EQ(token.GetSpelling().size(), tok_len);
  lexer.Advance();
  token = lexer.GetCurrentToken();
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::eof);
}

TEST(DILLexerTests, TokenKindTest) {
  StringRef input_expr("namespace");
  llvm::Expected<lldb_private::dil::DILLexer> maybe_lexer =
      lldb_private::dil::DILLexer::Create(input_expr);
  ASSERT_THAT_EXPECTED(maybe_lexer, llvm::Succeeded());
  lldb_private::dil::DILLexer lexer(*maybe_lexer);
  lldb_private::dil::Token token =
      lldb_private::dil::Token(lldb_private::dil::Token::unknown, "", 0);
  lexer.Advance();
  token = lexer.GetCurrentToken();

  EXPECT_TRUE(token.Is(lldb_private::dil::Token::identifier));
  EXPECT_FALSE(token.Is(lldb_private::dil::Token::l_paren));
  EXPECT_TRUE(token.IsOneOf(lldb_private::dil::Token::eof,
                            lldb_private::dil::Token::identifier));
  EXPECT_FALSE(token.IsOneOf(
      lldb_private::dil::Token::l_paren, lldb_private::dil::Token::r_paren,
      lldb_private::dil::Token::coloncolon, lldb_private::dil::Token::eof));
}

TEST(DILLexerTests, LookAheadTest) {
  StringRef input_expr("(anonymous namespace)::some_var");
  llvm::Expected<lldb_private::dil::DILLexer> maybe_lexer =
      lldb_private::dil::DILLexer::Create(input_expr);
  ASSERT_THAT_EXPECTED(maybe_lexer, llvm::Succeeded());
  lldb_private::dil::DILLexer lexer(*maybe_lexer);
  lldb_private::dil::Token token =
      lldb_private::dil::Token(lldb_private::dil::Token::unknown, "", 0);
  uint32_t expect_loc = 23;
  lexer.Advance();
  token = lexer.GetCurrentToken();

  // Current token is '('; check the next 4 tokens, to make
  // sure they are the identifier 'anonymous', the identifier 'namespace'
  // ')' and '::', in that order.
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::l_paren);
  EXPECT_EQ(lexer.LookAhead(1).GetKind(), lldb_private::dil::Token::identifier);
  EXPECT_EQ(lexer.LookAhead(1).GetSpelling(), "anonymous");
  EXPECT_EQ(lexer.LookAhead(2).GetKind(), lldb_private::dil::Token::identifier);
  EXPECT_EQ(lexer.LookAhead(2).GetSpelling(), "namespace");
  EXPECT_EQ(lexer.LookAhead(3).GetKind(), lldb_private::dil::Token::r_paren);
  EXPECT_EQ(lexer.LookAhead(4).GetKind(), lldb_private::dil::Token::coloncolon);

  // Our current index should still be 0, as we only looked ahead; we are still
  // officially on the '('.
  EXPECT_EQ(lexer.GetCurrentTokenIdx(), (uint32_t)0);

  // Accept the 'lookahead', so our current token is '::', which has the index
  // 4 in our vector of tokens (which starts at zero).
  lexer.Advance(4);
  token = lexer.GetCurrentToken();
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::coloncolon);
  EXPECT_EQ(lexer.GetCurrentTokenIdx(), (uint32_t)4);

  lexer.Advance();
  token = lexer.GetCurrentToken();
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::identifier);
  EXPECT_EQ(token.GetSpelling(), "some_var");
  EXPECT_EQ(lexer.GetCurrentTokenIdx(), (uint32_t)5);
  // Verify we've advanced our position counter (lexing location) in the
  // input 23 characters (the length of '(anonymous namespace)::'.
  EXPECT_EQ(token.GetLocation(), expect_loc);

  lexer.Advance();
  token = lexer.GetCurrentToken();
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::eof);
}

TEST(DILLexerTests, MultiTokenLexTest) {
  StringRef input_expr("This string has (several ) ::identifiers");
  llvm::Expected<lldb_private::dil::DILLexer> maybe_lexer =
      lldb_private::dil::DILLexer::Create(input_expr);
  ASSERT_THAT_EXPECTED(maybe_lexer, llvm::Succeeded());
  lldb_private::dil::DILLexer lexer(*maybe_lexer);
  lldb_private::dil::Token token =
      lldb_private::dil::Token(lldb_private::dil::Token::unknown, "", 0);

  std::vector<std::pair<lldb_private::dil::Token::Kind, std::string>>
      lexer_tokens_data = ExtractTokenData(lexer);

  EXPECT_THAT(
      lexer_tokens_data,
      testing::ElementsAre(
          testing::Pair(lldb_private::dil::Token::identifier, "This"),
          testing::Pair(lldb_private::dil::Token::identifier, "string"),
          testing::Pair(lldb_private::dil::Token::identifier, "has"),
          testing::Pair(lldb_private::dil::Token::l_paren, "("),
          testing::Pair(lldb_private::dil::Token::identifier, "several"),
          testing::Pair(lldb_private::dil::Token::r_paren, ")"),
          testing::Pair(lldb_private::dil::Token::coloncolon, "::"),
          testing::Pair(lldb_private::dil::Token::identifier, "identifiers"),
          testing::Pair(lldb_private::dil::Token::eof, "")));
  lexer.Advance();
  token = lexer.GetCurrentToken();
  EXPECT_EQ(token.GetSpelling(), "");
  EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::eof);
}

TEST(DILLexerTests, IdentifiersTest) {
  std::vector<std::string> valid_identifiers = {
      "$My_name1", "$pc",  "abcd", "ab cd", "_",      "_a",       "_a_",
      "a_b",       "this", "self", "a",     "MyName", "namespace"};
  std::vector<std::string> invalid_identifiers = {"234", "2a",      "2",
                                                  "$",   "1MyName", ""};

  // Verify that all of the valid identifiers come out as identifier tokens.
  for (auto &str : valid_identifiers) {
    SCOPED_TRACE(str);
    llvm::Expected<lldb_private::dil::DILLexer> maybe_lexer =
        lldb_private::dil::DILLexer::Create(str);
    ASSERT_THAT_EXPECTED(maybe_lexer, llvm::Succeeded());
    lldb_private::dil::DILLexer lexer(*maybe_lexer);
    lldb_private::dil::Token token =
        lldb_private::dil::Token(lldb_private::dil::Token::unknown, "", 0);
    lexer.Advance();
    token = lexer.GetCurrentToken();
    EXPECT_EQ(token.GetKind(), lldb_private::dil::Token::identifier);
  }

  // Verify that none of the invalid identifiers come out as identifier tokens.
  for (auto &str : invalid_identifiers) {
    SCOPED_TRACE(str);
    llvm::Expected<lldb_private::dil::DILLexer> maybe_lexer =
        lldb_private::dil::DILLexer::Create(str);
    if (!maybe_lexer) {
      llvm::consumeError(maybe_lexer.takeError());
      // In this case, it's ok for lexing to return an error.
    } else {
      lldb_private::dil::DILLexer lexer(*maybe_lexer);
      lldb_private::dil::Token token =
          lldb_private::dil::Token(lldb_private::dil::Token::unknown, "", 0);
      // We didn't get an error; make sure we did not get an identifier token.
      lexer.Advance();
      token = lexer.GetCurrentToken();
      EXPECT_TRUE(token.IsNot(lldb_private::dil::Token::identifier));
      EXPECT_TRUE(token.IsOneOf(lldb_private::dil::Token::unknown,
                                lldb_private::dil::Token::eof));
    }
  }
}
