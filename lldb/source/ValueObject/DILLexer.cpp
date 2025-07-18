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
  case Kind::floating_constant:
    return "floating_constant";
  case Kind::identifier:
    return "identifier";
  case Kind::integer_constant:
    return "integer_constant";
  case Kind::l_paren:
    return "l_paren";
  case Kind::l_square:
    return "l_square";
  case Kind::minus:
    return "minus";
  case Kind::period:
    return "period";
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

static bool IsNumberBodyChar(char ch) { return IsDigit(ch) || IsLetter(ch); }

static std::optional<llvm::StringRef> IsNumber(llvm::StringRef &remainder,
                                               bool &isFloat) {
  llvm::StringRef::iterator cur_pos = remainder.begin();
  if (*cur_pos == '.') {
    auto next_pos = cur_pos + 1;
    if (next_pos == remainder.end() || !IsDigit(*next_pos))
      return std::nullopt;
  }
  if (IsDigit(*(cur_pos)) || *(cur_pos) == '.') {
    while (IsNumberBodyChar(*cur_pos))
      cur_pos++;

    if (*cur_pos == '.') {
      isFloat = true;
      cur_pos++;
      while (IsNumberBodyChar(*cur_pos))
        cur_pos++;

      // Check if there's an exponent
      char prev_ch = *(cur_pos - 1);
      if (prev_ch == 'e' || prev_ch == 'E' || prev_ch == 'p' ||
          prev_ch == 'P') {
        if (*(cur_pos) == '+' || *(cur_pos) == '-') {
          cur_pos++;
          while (IsNumberBodyChar(*cur_pos))
            cur_pos++;
        }
      }
    }

    llvm::StringRef number = remainder.substr(0, cur_pos - remainder.begin());
    if (remainder.consume_front(number))
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
    auto kind = isFloat ? Token::floating_constant : Token::integer_constant;
    return Token(kind, maybe_number->str(), position);
  }
  std::optional<llvm::StringRef> maybe_word = IsWord(expr, remainder);
  if (maybe_word)
    return Token(Token::identifier, maybe_word->str(), position);

  constexpr std::pair<Token::Kind, const char *> operators[] = {
      {Token::amp, "&"},     {Token::arrow, "->"},   {Token::coloncolon, "::"},
      {Token::l_paren, "("}, {Token::l_square, "["}, {Token::minus, "-"},
      {Token::period, "."},  {Token::r_paren, ")"},  {Token::r_square, "]"},
      {Token::star, "*"},
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
