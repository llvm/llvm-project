//===-- DILLexer.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This implements the recursive descent parser for the Data Inspection
// Language (DIL), and its helper functions, which will eventually underlie the
// 'frame variable' command. The language that this parser recognizes is
// described in lldb/docs/dil-expr-lang.ebnf
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILLexer.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/DILParser.h"
#include "llvm/ADT/StringSwitch.h"

namespace lldb_private::dil {

llvm::StringRef Token::GetTokenName(Kind kind) {
  switch (kind) {
  case Kind::amp:
    return "amp";
  case Kind::arrow:
    return "arrow";
  case Kind::coloncolon:
    return "coloncolon";
  case Kind::eof:
    return "eof";
  case Kind::float_constant:
    return "float_constant";
  case Kind::identifier:
    return "identifier";
  case Kind::integer_constant:
    return "integer_constant";
  case Kind::kw_false:
    return "false";
  case Kind::kw_true:
    return "true";
  case Kind::l_paren:
    return "l_paren";
  case Kind::l_square:
    return "l_square";
  case Kind::minus:
    return "minus";
  case Kind::period:
    return "period";
  case Kind::plus:
    return "plus";
  case Kind::r_paren:
    return "r_paren";
  case Kind::r_square:
    return "r_square";
  case Token::star:
    return "star";
  }
  llvm_unreachable("Unknown token name");
}

static bool IsLetter(char c) {
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
}

static bool IsDigit(char c) { return '0' <= c && c <= '9'; }

// A word starts with a letter, underscore, or dollar sign, followed by
// letters ('a'..'z','A'..'Z'), digits ('0'..'9'), and/or  underscores.
static std::optional<llvm::StringRef> IsWord(llvm::StringRef expr,
                                             llvm::StringRef &remainder) {
  // Find the longest prefix consisting of letters, digits, underscors and
  // '$'. If it doesn't start with a digit, then it's a word.
  llvm::StringRef candidate = remainder.take_while(
      [](char c) { return IsDigit(c) || IsLetter(c) || c == '_' || c == '$'; });
  if (candidate.empty() || IsDigit(candidate[0]))
    return std::nullopt;
  remainder = remainder.drop_front(candidate.size());
  return candidate;
}

static bool IsNumberBodyChar(char ch) {
  return IsDigit(ch) || IsLetter(ch) || ch == '.';
}

static std::optional<llvm::StringRef> IsNumber(llvm::StringRef &remainder,
                                               bool &isFloat) {
  llvm::StringRef tail = remainder;
  llvm::StringRef body = tail.take_while(IsNumberBodyChar);
  size_t dots = body.count('.');
  if (dots > 1 || dots == body.size())
    return std::nullopt;
  if (IsDigit(body.front()) || (body[0] == '.' && IsDigit(body[1]))) {
    isFloat = dots == 1;
    tail = tail.drop_front(body.size());
    bool isHex = body.contains_insensitive('x');
    bool hasExp = !isHex && body.contains_insensitive('e');
    bool hasHexExp = isHex && body.contains_insensitive('p');
    if (hasExp || hasHexExp) {
      isFloat = true; // This marks numbers like 0x1p1 and 1e1 as float
      if (body.ends_with_insensitive("e") || body.ends_with_insensitive("p"))
        if (tail.consume_front("+") || tail.consume_front("-"))
          tail = tail.drop_while(IsNumberBodyChar);
    }
    size_t number_length = remainder.size() - tail.size();
    llvm::StringRef number = remainder.take_front(number_length);
    remainder = remainder.drop_front(number_length);
    return number;
  }
  return std::nullopt;
}

llvm::Expected<DILLexer> DILLexer::Create(llvm::StringRef expr) {
  std::vector<Token> tokens;
  llvm::StringRef remainder = expr;
  do {
    if (llvm::Expected<Token> t = Lex(expr, remainder)) {
      tokens.push_back(std::move(*t));
    } else {
      return t.takeError();
    }
  } while (tokens.back().GetKind() != Token::eof);
  return DILLexer(expr, std::move(tokens));
}

llvm::Expected<Token> DILLexer::Lex(llvm::StringRef expr,
                                    llvm::StringRef &remainder) {
  // Skip over whitespace (spaces).
  remainder = remainder.ltrim();
  llvm::StringRef::iterator cur_pos = remainder.begin();

  // Check to see if we've reached the end of our input string.
  if (remainder.empty())
    return Token(Token::eof, "", (uint32_t)expr.size());

  uint32_t position = cur_pos - expr.begin();
  bool isFloat = false;
  std::optional<llvm::StringRef> maybe_number = IsNumber(remainder, isFloat);
  if (maybe_number) {
    auto kind = isFloat ? Token::float_constant : Token::integer_constant;
    return Token(kind, maybe_number->str(), position);
  }
  std::optional<llvm::StringRef> maybe_word = IsWord(expr, remainder);
  if (maybe_word) {
    llvm::StringRef word = *maybe_word;
    Token::Kind kind = llvm::StringSwitch<Token::Kind>(word)
                           .Case("false", Token::kw_false)
                           .Case("true", Token::kw_true)
                           .Default(Token::identifier);
    return Token(kind, word.str(), position);
  }

  constexpr std::pair<Token::Kind, const char *> operators[] = {
      {Token::amp, "&"},      {Token::arrow, "->"},   {Token::coloncolon, "::"},
      {Token::l_paren, "("},  {Token::l_square, "["}, {Token::minus, "-"},
      {Token::period, "."},   {Token::plus, "+"},     {Token::r_paren, ")"},
      {Token::r_square, "]"}, {Token::star, "*"},
  };
  for (auto [kind, str] : operators) {
    if (remainder.consume_front(str))
      return Token(kind, str, position);
  }

  // Unrecognized character(s) in string; unable to lex it.
  return llvm::make_error<DILDiagnosticError>(expr, "unrecognized token",
                                              position);
}

} // namespace lldb_private::dil
