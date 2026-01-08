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

using namespace lldb_private::dil;

llvm::Expected<std::vector<std::pair<Token::Kind, std::string>>>
ExtractTokenData(llvm::StringRef input_expr) {

  llvm::Expected<DILLexer> maybe_lexer = DILLexer::Create(input_expr);
  if (!maybe_lexer)
    return maybe_lexer.takeError();
  DILLexer lexer(*maybe_lexer);

  std::vector<std::pair<Token::Kind, std::string>> data;
  do {
    Token tok = lexer.GetCurrentToken();
    data.push_back(std::make_pair(tok.GetKind(), tok.GetSpelling()));
    lexer.Advance();
  } while (data.back().first != Token::eof);
  // Don't return the eof token.
  data.pop_back();
  return data;
}

TEST(DILLexerTests, SimpleTest) {
  StringRef input_expr("simple_var");
  llvm::Expected<DILLexer> maybe_lexer = DILLexer::Create(input_expr);
  ASSERT_THAT_EXPECTED(maybe_lexer, llvm::Succeeded());
  DILLexer lexer(*maybe_lexer);
  Token token = lexer.GetCurrentToken();

  EXPECT_EQ(token.GetKind(), Token::identifier);
  EXPECT_EQ(token.GetSpelling(), "simple_var");
  lexer.Advance();
  token = lexer.GetCurrentToken();
  EXPECT_EQ(token.GetKind(), Token::eof);
}

TEST(DILLexerTests, TokenKindTest) {
  Token token = Token(Token::identifier, "ident", 0);

  EXPECT_TRUE(token.Is(Token::identifier));
  EXPECT_FALSE(token.Is(Token::l_paren));
  EXPECT_TRUE(token.IsOneOf({Token::eof, Token::identifier}));
  EXPECT_FALSE(token.IsOneOf(
      {Token::l_paren, Token::r_paren, Token::coloncolon, Token::eof}));
}

TEST(DILLexerTests, LookAheadTest) {
  StringRef input_expr("(anonymous namespace)::some_var");
  llvm::Expected<DILLexer> maybe_lexer = DILLexer::Create(input_expr);
  ASSERT_THAT_EXPECTED(maybe_lexer, llvm::Succeeded());
  DILLexer lexer(*maybe_lexer);
  Token token = lexer.GetCurrentToken();

  // Current token is '('; check the next 4 tokens, to make
  // sure they are the identifier 'anonymous', the identifier 'namespace'
  // ')' and '::', in that order.
  EXPECT_EQ(token.GetKind(), Token::l_paren);
  EXPECT_EQ(lexer.LookAhead(1).GetKind(), Token::identifier);
  EXPECT_EQ(lexer.LookAhead(1).GetSpelling(), "anonymous");
  EXPECT_EQ(lexer.LookAhead(2).GetKind(), Token::identifier);
  EXPECT_EQ(lexer.LookAhead(2).GetSpelling(), "namespace");
  EXPECT_EQ(lexer.LookAhead(3).GetKind(), Token::r_paren);
  EXPECT_EQ(lexer.LookAhead(4).GetKind(), Token::coloncolon);

  // Our current index should still be 0, as we only looked ahead; we are still
  // officially on the '('.
  EXPECT_EQ(lexer.GetCurrentTokenIdx(), 0u);

  // Accept the 'lookahead', so our current token is '::', which has the index
  // 4 in our vector of tokens (which starts at zero).
  lexer.Advance(4);
  token = lexer.GetCurrentToken();
  EXPECT_EQ(token.GetKind(), Token::coloncolon);
  EXPECT_EQ(lexer.GetCurrentTokenIdx(), 4u);

  lexer.Advance();
  token = lexer.GetCurrentToken();
  EXPECT_EQ(token.GetKind(), Token::identifier);
  EXPECT_EQ(token.GetSpelling(), "some_var");
  EXPECT_EQ(lexer.GetCurrentTokenIdx(), 5u);
  EXPECT_EQ(token.GetLocation(), strlen("(anonymous namespace)::"));

  lexer.Advance();
  token = lexer.GetCurrentToken();
  EXPECT_EQ(token.GetKind(), Token::eof);
}

TEST(DILLexerTests, MultiTokenLexTest) {
  EXPECT_THAT_EXPECTED(
      ExtractTokenData("This string has (several ) ::identifiers"),
      llvm::HasValue(testing::ElementsAre(
          testing::Pair(Token::identifier, "This"),
          testing::Pair(Token::identifier, "string"),
          testing::Pair(Token::identifier, "has"),
          testing::Pair(Token::l_paren, "("),
          testing::Pair(Token::identifier, "several"),
          testing::Pair(Token::r_paren, ")"),
          testing::Pair(Token::coloncolon, "::"),
          testing::Pair(Token::identifier, "identifiers"))));
}

TEST(DILLexerTests, IdentifiersTest) {
  // These strings should lex into identifier tokens.
  std::vector<std::string> valid_identifiers = {
      "$My_name1", "$pc",  "abcd", "_", "_a",     "_a_",      "$",
      "a_b",       "this", "self", "a", "MyName", "namespace"};

  // The lexer can lex these strings, but they should not be identifiers.
  std::vector<std::string> invalid_identifiers = {"", "::", "(", ")", "0abc"};

  // The lexer is expected to fail attempting to lex these strings (it cannot
  // create valid tokens out of them).
  std::vector<std::string> invalid_tok_strings = {"#include", "a@a"};

  // Verify that all of the valid identifiers come out as identifier tokens.
  for (auto &str : valid_identifiers) {
    SCOPED_TRACE(str);
    EXPECT_THAT_EXPECTED(ExtractTokenData(str),
                         llvm::HasValue(testing::ElementsAre(
                             testing::Pair(Token::identifier, str))));
  }

  // Verify that the lexer fails on invalid token strings.
  for (auto &str : invalid_tok_strings) {
    SCOPED_TRACE(str);
    auto maybe_lexer = DILLexer::Create(str);
    EXPECT_THAT_EXPECTED(maybe_lexer, llvm::Failed());
  }

  // Verify that none of the invalid identifiers come out as identifier tokens.
  for (auto &str : invalid_identifiers) {
    SCOPED_TRACE(str);
    llvm::Expected<DILLexer> maybe_lexer = DILLexer::Create(str);
    EXPECT_THAT_EXPECTED(maybe_lexer, llvm::Succeeded());
    DILLexer lexer(*maybe_lexer);
    Token token = lexer.GetCurrentToken();
    EXPECT_TRUE(token.IsNot(Token::identifier));
    EXPECT_TRUE(token.IsOneOf({Token::eof, Token::coloncolon, Token::l_paren,
                               Token::r_paren, Token::integer_constant}));
  }
}

TEST(DILLexerTests, NumbersTest) {
  // These strings should lex into number tokens.
  std::vector<std::string> valid_integers = {"123", "0x123", "0123", "0b101"};
  std::vector<std::string> valid_floats = {
      "1.2",    ".2",    "2.f",     "0x1.2",    "0x.2",      ".2e1f",
      "2.e+1f", "0x1.f", "0x1.2P1", "0x1.p-1f", "0x1.2P+3f", "1E1",
      "1E+1",   "0x1p1", "0x1p+1",  "0xf.fp1f"};

  // The lexer can lex these strings, but they should not be numbers.
  std::vector<std::string> invalid_numbers = {"", "x123", "b123", "a.b"};

  for (auto &str : valid_integers) {
    SCOPED_TRACE(str);
    EXPECT_THAT_EXPECTED(ExtractTokenData(str),
                         llvm::HasValue(testing::ElementsAre(
                             testing::Pair(Token::integer_constant, str))));
  }
  for (auto &str : valid_floats) {
    SCOPED_TRACE(str);
    EXPECT_THAT_EXPECTED(ExtractTokenData(str),
                         llvm::HasValue(testing::ElementsAre(
                             testing::Pair(Token::float_constant, str))));
  }
  // Verify that none of the invalid numbers come out as numeric tokens.
  for (auto &str : invalid_numbers) {
    SCOPED_TRACE(str);
    llvm::Expected<DILLexer> maybe_lexer = DILLexer::Create(str);
    EXPECT_THAT_EXPECTED(maybe_lexer, llvm::Succeeded());
    DILLexer lexer(*maybe_lexer);
    Token token = lexer.GetCurrentToken();
    EXPECT_TRUE(token.IsNot(Token::integer_constant));
    EXPECT_TRUE(token.IsOneOf({Token::eof, Token::identifier}));
  }

  // Verify that '-' and '+' are not lexed if they're not part of a number
  std::vector<std::string> expressions = {"1+e",     "0x1+p",      "1.1+e",
                                          "1.1e1+e", "0x1.1p-1-p", "1e-1+e",
                                          "1e1+e",   "0x1p-1-p",   "0xe+e"};
  for (auto &str : expressions) {
    SCOPED_TRACE(str);
    llvm::Expected<DILLexer> maybe_lexer = DILLexer::Create(str);
    EXPECT_THAT_EXPECTED(maybe_lexer, llvm::Succeeded());
    DILLexer lexer(*maybe_lexer);
    Token token = lexer.GetCurrentToken();
    EXPECT_TRUE(
        token.IsOneOf({Token::integer_constant, Token::float_constant}));
    lexer.Advance();
    token = lexer.GetCurrentToken();
    EXPECT_TRUE(token.IsOneOf({Token::plus, Token::minus}));
    lexer.Advance();
    token = lexer.GetCurrentToken();
    EXPECT_TRUE(token.Is(Token::identifier));
  }
}
