//===- TestOMPTraitLexer.cpp - Tests for OMP Trait Lexer -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp_traits.h"
#include "gtest/gtest.h"

#include <cstring>
#include <string>

using namespace lexer;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Compare a token's text against a C string.
static bool text_is(const token &tok, const char *expected) {
  size_t len = strlen(expected);
  if (tok.text.length() != len)
    return false;
  if (len == 0)
    return true;
  return memcmp(tok.text.begin(), expected, len) == 0;
}

// Lex the whole spec into a vector of tokens (excluding the trailing END).
static kmp_vector<token> tokenize(const char *spec) {
  kmp_lexer lex{kmp_str_ref(spec)};
  kmp_vector<token> tokens;
  for (token tok = lex.next(); tok.kind != token_kind::END; tok = lex.next())
    tokens.push_back(tok);
  return tokens;
}

// Assert a token has the expected kind and text.
static void expect_token(const token &tok, token_kind kind, const char *text) {
  EXPECT_EQ(tok.kind, kind);
  EXPECT_TRUE(text_is(tok, text))
      << "expected token text \"" << text << "\" but got \""
      << std::string(tok.text.begin(), tok.text.length()) << "\"";
}

//===----------------------------------------------------------------------===//
// Single Tokens
//===----------------------------------------------------------------------===//

TEST(LexerTest, EmptyInput) {
  kmp_lexer lex{kmp_str_ref("")};
  EXPECT_EQ(lex.next().kind, token_kind::END);
}

TEST(LexerTest, WhitespaceOnly) {
  kmp_lexer lex{kmp_str_ref("   \t  ")};
  EXPECT_EQ(lex.next().kind, token_kind::END);
}

TEST(LexerTest, EndIsSticky) {
  // Once the input is exhausted, next() keeps returning END.
  kmp_lexer lex{kmp_str_ref("*")};
  EXPECT_EQ(lex.next().kind, token_kind::STAR);
  EXPECT_EQ(lex.next().kind, token_kind::END);
  EXPECT_EQ(lex.next().kind, token_kind::END);
}

TEST(LexerTest, Punctuation) {
  kmp_vector<token> tokens = tokenize(",*!()[]:");

  ASSERT_EQ(tokens.size(), 8u);
  expect_token(tokens[0], token_kind::COMMA, ",");
  expect_token(tokens[1], token_kind::STAR, "*");
  expect_token(tokens[2], token_kind::NOT, "!");
  expect_token(tokens[3], token_kind::L_PAREN, "(");
  expect_token(tokens[4], token_kind::R_PAREN, ")");
  expect_token(tokens[5], token_kind::L_BRACKET, "[");
  expect_token(tokens[6], token_kind::R_BRACKET, "]");
  expect_token(tokens[7], token_kind::COLON, ":");
}

TEST(LexerTest, DoubleColon) {
  // "::" is lexed as two COLON tokens.
  kmp_vector<token> tokens = tokenize("::");

  ASSERT_EQ(tokens.size(), 2u);
  expect_token(tokens[0], token_kind::COLON, ":");
  expect_token(tokens[1], token_kind::COLON, ":");
}

TEST(LexerTest, AndOperator) {
  kmp_vector<token> tokens = tokenize("&&");

  ASSERT_EQ(tokens.size(), 1u);
  expect_token(tokens[0], token_kind::AND, "&&");
}

TEST(LexerTest, OrOperator) {
  kmp_vector<token> tokens = tokenize("||");

  ASSERT_EQ(tokens.size(), 1u);
  expect_token(tokens[0], token_kind::OR, "||");
}

//===----------------------------------------------------------------------===//
// Numbers
//
// The lexer does not have a dedicated number token: a run of digits is just a
// WORD. Recognizing a device number is the parser's job (see consume_clause).
//===----------------------------------------------------------------------===//

TEST(LexerTest, SingleDigitNumber) {
  kmp_vector<token> tokens = tokenize("0");

  ASSERT_EQ(tokens.size(), 1u);
  expect_token(tokens[0], token_kind::WORD, "0");
}

TEST(LexerTest, MultiDigitNumber) {
  kmp_vector<token> tokens = tokenize("12345");

  ASSERT_EQ(tokens.size(), 1u);
  expect_token(tokens[0], token_kind::WORD, "12345");
}

TEST(LexerTest, NumberList) {
  kmp_vector<token> tokens = tokenize("1,2,3");

  ASSERT_EQ(tokens.size(), 5u);
  expect_token(tokens[0], token_kind::WORD, "1");
  expect_token(tokens[1], token_kind::COMMA, ",");
  expect_token(tokens[2], token_kind::WORD, "2");
  expect_token(tokens[3], token_kind::COMMA, ",");
  expect_token(tokens[4], token_kind::WORD, "3");
}

//===----------------------------------------------------------------------===//
// Words
//===----------------------------------------------------------------------===//

TEST(LexerTest, SimpleWord) {
  kmp_vector<token> tokens = tokenize("uid");

  ASSERT_EQ(tokens.size(), 1u);
  expect_token(tokens[0], token_kind::WORD, "uid");
}

TEST(LexerTest, WordWithDash) {
  // A uid_value may contain '-'; it stays part of the word.
  kmp_vector<token> tokens = tokenize("device-0");

  ASSERT_EQ(tokens.size(), 1u);
  expect_token(tokens[0], token_kind::WORD, "device-0");
}

TEST(LexerTest, WordWithUnderscore) {
  kmp_vector<token> tokens = tokenize("my_device_123");

  ASSERT_EQ(tokens.size(), 1u);
  expect_token(tokens[0], token_kind::WORD, "my_device_123");
}

TEST(LexerTest, LeadingMinusIsWord) {
  // The grammar allows a leading "-" for device numbers, but since '-' is also
  // a uid_value symbol the lexer keeps the contiguous run together as a single
  // WORD. Distinguishing a negative device number is left to the parser.
  kmp_vector<token> tokens = tokenize("-5");

  ASSERT_EQ(tokens.size(), 1u);
  expect_token(tokens[0], token_kind::WORD, "-5");
}

TEST(LexerTest, WhitespaceBreaksWords) {
  // A uid_value may not contain whitespace, so spaces split the run.
  kmp_vector<token> tokens = tokenize("device - 0");

  ASSERT_EQ(tokens.size(), 3u);
  expect_token(tokens[0], token_kind::WORD, "device");
  expect_token(tokens[1], token_kind::WORD, "-");
  expect_token(tokens[2], token_kind::WORD, "0");
}

//===----------------------------------------------------------------------===//
// Full trait expressions
//===----------------------------------------------------------------------===//

TEST(LexerTest, UidTrait) {
  kmp_vector<token> tokens = tokenize("uid(device-0)");

  ASSERT_EQ(tokens.size(), 4u);
  expect_token(tokens[0], token_kind::WORD, "uid");
  expect_token(tokens[1], token_kind::L_PAREN, "(");
  expect_token(tokens[2], token_kind::WORD, "device-0");
  expect_token(tokens[3], token_kind::R_PAREN, ")");
}

TEST(LexerTest, NegatedUidTrait) {
  kmp_vector<token> tokens = tokenize("!uid(a)");

  ASSERT_EQ(tokens.size(), 5u);
  expect_token(tokens[0], token_kind::NOT, "!");
  expect_token(tokens[1], token_kind::WORD, "uid");
  expect_token(tokens[2], token_kind::L_PAREN, "(");
  expect_token(tokens[3], token_kind::WORD, "a");
  expect_token(tokens[4], token_kind::R_PAREN, ")");
}

TEST(LexerTest, AndGroup) {
  kmp_vector<token> tokens = tokenize("(uid(a) && uid(b))");

  ASSERT_EQ(tokens.size(), 11u);
  expect_token(tokens[0], token_kind::L_PAREN, "(");
  expect_token(tokens[1], token_kind::WORD, "uid");
  expect_token(tokens[2], token_kind::L_PAREN, "(");
  expect_token(tokens[3], token_kind::WORD, "a");
  expect_token(tokens[4], token_kind::R_PAREN, ")");
  expect_token(tokens[5], token_kind::AND, "&&");
  expect_token(tokens[6], token_kind::WORD, "uid");
  expect_token(tokens[7], token_kind::L_PAREN, "(");
  expect_token(tokens[8], token_kind::WORD, "b");
  expect_token(tokens[9], token_kind::R_PAREN, ")");
  expect_token(tokens[10], token_kind::R_PAREN, ")");
}

TEST(LexerTest, OrGroup) {
  kmp_vector<token> tokens = tokenize("uid(a)||uid(b)");

  ASSERT_EQ(tokens.size(), 9u);
  expect_token(tokens[0], token_kind::WORD, "uid");
  expect_token(tokens[1], token_kind::L_PAREN, "(");
  expect_token(tokens[2], token_kind::WORD, "a");
  expect_token(tokens[3], token_kind::R_PAREN, ")");
  expect_token(tokens[4], token_kind::OR, "||");
  expect_token(tokens[5], token_kind::WORD, "uid");
  expect_token(tokens[6], token_kind::L_PAREN, "(");
  expect_token(tokens[7], token_kind::WORD, "b");
  expect_token(tokens[8], token_kind::R_PAREN, ")");
}

TEST(LexerTest, WildcardWithLiterals) {
  kmp_vector<token> tokens = tokenize("1, *, 3");

  ASSERT_EQ(tokens.size(), 5u);
  expect_token(tokens[0], token_kind::WORD, "1");
  expect_token(tokens[1], token_kind::COMMA, ",");
  expect_token(tokens[2], token_kind::STAR, "*");
  expect_token(tokens[3], token_kind::COMMA, ",");
  expect_token(tokens[4], token_kind::WORD, "3");
}

TEST(LexerTest, IndexExpression) {
  kmp_vector<token> tokens = tokenize("[1:2:3]");

  ASSERT_EQ(tokens.size(), 7u);
  expect_token(tokens[0], token_kind::L_BRACKET, "[");
  expect_token(tokens[1], token_kind::WORD, "1");
  expect_token(tokens[2], token_kind::COLON, ":");
  expect_token(tokens[3], token_kind::WORD, "2");
  expect_token(tokens[4], token_kind::COLON, ":");
  expect_token(tokens[5], token_kind::WORD, "3");
  expect_token(tokens[6], token_kind::R_BRACKET, "]");
}

//===----------------------------------------------------------------------===//
// Whitespace handling
//===----------------------------------------------------------------------===//

TEST(LexerTest, LeadingAndTrailingWhitespace) {
  kmp_vector<token> tokens = tokenize("   uid ( a )   ");

  ASSERT_EQ(tokens.size(), 4u);
  expect_token(tokens[0], token_kind::WORD, "uid");
  expect_token(tokens[1], token_kind::L_PAREN, "(");
  expect_token(tokens[2], token_kind::WORD, "a");
  expect_token(tokens[3], token_kind::R_PAREN, ")");
}

TEST(LexerTest, WhitespaceAroundOperators) {
  kmp_vector<token> tokens = tokenize("uid(a)   &&   uid(b)");

  ASSERT_EQ(tokens.size(), 9u);
  expect_token(tokens[4], token_kind::AND, "&&");
}

//===----------------------------------------------------------------------===//
// Unknown / error characters
//===----------------------------------------------------------------------===//

TEST(LexerTest, UnknownCharacter) {
  kmp_vector<token> tokens = tokenize("@");

  ASSERT_EQ(tokens.size(), 1u);
  expect_token(tokens[0], token_kind::UNKNOWN, "@");
}

TEST(LexerTest, LoneAmpersandIsUnknown) {
  kmp_vector<token> tokens = tokenize("&");

  ASSERT_EQ(tokens.size(), 1u);
  expect_token(tokens[0], token_kind::UNKNOWN, "&");
}

TEST(LexerTest, LonePipeIsUnknown) {
  kmp_vector<token> tokens = tokenize("|");

  ASSERT_EQ(tokens.size(), 1u);
  expect_token(tokens[0], token_kind::UNKNOWN, "|");
}

TEST(LexerTest, TripleAmpersand) {
  // "&&&" is "&&" followed by a lone (unknown) '&'.
  kmp_vector<token> tokens = tokenize("&&&");

  ASSERT_EQ(tokens.size(), 2u);
  expect_token(tokens[0], token_kind::AND, "&&");
  expect_token(tokens[1], token_kind::UNKNOWN, "&");
}

//===----------------------------------------------------------------------===//
// peek()
//===----------------------------------------------------------------------===//

TEST(LexerTest, PeekDoesNotAdvance) {
  kmp_lexer lex{kmp_str_ref("uid(a)")};

  // Peek repeatedly returns the same token without consuming it.
  EXPECT_EQ(lex.peek().kind, token_kind::WORD);
  EXPECT_TRUE(text_is(lex.peek(), "uid"));
  EXPECT_EQ(lex.peek().kind, token_kind::WORD);

  // next() returns the peeked token, then advances.
  token tok = lex.next();
  expect_token(tok, token_kind::WORD, "uid");
  EXPECT_EQ(lex.peek().kind, token_kind::L_PAREN);
  expect_token(lex.next(), token_kind::L_PAREN, "(");
}

TEST(LexerTest, PeekAtEnd) {
  kmp_lexer lex{kmp_str_ref("")};

  EXPECT_EQ(lex.peek().kind, token_kind::END);
  EXPECT_EQ(lex.next().kind, token_kind::END);
  EXPECT_EQ(lex.peek().kind, token_kind::END);
}

} // namespace
