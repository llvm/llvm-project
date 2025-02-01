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
#include "llvm/ADT/StringSwitch.h"

namespace lldb_private::dil {

llvm::StringRef Token::GetTokenName(Kind kind) {
  switch (kind) {
  case Kind::coloncolon:
    return "coloncolon";
  case Kind::eof:
    return "eof";
  case Kind::identifier:
    return "identifier";
  case Kind::l_paren:
    return "l_paren";
  case Kind::r_paren:
    return "r_paren";
  case Kind::unknown:
    return "unknown";
  }
}

static bool IsLetter(char c) {
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
}

static bool IsDigit(char c) { return '0' <= c && c <= '9'; }

// A word starts with a letter, underscore, or dollar sign, followed by
// letters ('a'..'z','A'..'Z'), digits ('0'..'9'), and/or  underscores.
static std::optional<llvm::StringRef> IsWord(llvm::StringRef expr,
                                             llvm::StringRef &remainder) {
  llvm::StringRef::iterator cur_pos = expr.end() - remainder.size();
  llvm::StringRef::iterator start = cur_pos;
  bool dollar_start = false;

  // Must not start with a digit.
  if (cur_pos == expr.end() || IsDigit(*cur_pos))
    return std::nullopt;

  // First character *may* be a '$', for a register name or convenience
  // variable.
  if (*cur_pos == '$') {
    dollar_start = true;
    ++cur_pos;
  }

  // Contains only letters, digits or underscores
  for (; cur_pos != expr.end(); ++cur_pos) {
    char c = *cur_pos;
    if (!IsLetter(c) && !IsDigit(c) && c != '_')
      break;
  }

  // If first char is '$', make sure there's at least one mare char, or it's
  // invalid.
  if (dollar_start && (cur_pos - start <= 1)) {
    cur_pos = start;
    return std::nullopt;
  }

  if (cur_pos == start)
    return std::nullopt;

  llvm::StringRef word = expr.substr(start - expr.begin(), cur_pos - start);
  if (remainder.consume_front(word))
    return word;

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
  llvm::StringRef::iterator cur_pos = expr.end() - remainder.size();

  // Check to see if we've reached the end of our input string.
  if (remainder.empty() || cur_pos == expr.end())
    return Token(Token::eof, "", (uint32_t)expr.size());

  uint32_t position = cur_pos - expr.begin();
  std::optional<llvm::StringRef> maybe_word = IsWord(expr, remainder);
  if (maybe_word) {
    llvm::StringRef word = *maybe_word;
    return Token(Token::identifier, word.str(), position);
  }

  constexpr std::pair<Token::Kind, const char *> operators[] = {
      {Token::l_paren, "("},
      {Token::r_paren, ")"},
      {Token::coloncolon, "::"},
  };
  for (auto [kind, str] : operators) {
    if (remainder.consume_front(str)) {
      cur_pos += strlen(str);
      return Token(kind, str, position);
    }
  }

  // Unrecognized character(s) in string; unable to lex it.
  return llvm::createStringError("Unable to lex input string");
}

} // namespace lldb_private::dil
