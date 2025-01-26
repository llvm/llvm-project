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

namespace lldb_private {

namespace dil {

llvm::StringRef Token::GetTokenName(Kind kind) {
  switch (kind) {
  case Kind::coloncolon:
    return "coloncolon";
  case Kind::eof:
    return "eof";
  case Kind::identifier:
    return "identifier";
  case Kind::invalid:
    return "invalid";
  case Kind::kw_namespace:
    return "namespace";
  case Kind::l_paren:
    return "l_paren";
  case Kind::none:
    return "none";
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
llvm::iterator_range<llvm::StringRef::iterator> DILLexer::IsWord() {
  llvm::StringRef::iterator start = m_cur_pos;
  bool dollar_start = false;

  // Must not start with a digit.
  if (m_cur_pos == m_expr.end() || IsDigit(*m_cur_pos))
    return llvm::make_range(m_cur_pos, m_cur_pos);

  // First character *may* be a '$', for a register name or convenience
  // variable.
  if (*m_cur_pos == '$') {
    dollar_start = true;
    ++m_cur_pos;
  }

  // Contains only letters, digits or underscores
  for (; m_cur_pos != m_expr.end(); ++m_cur_pos) {
    char c = *m_cur_pos;
    if (!IsLetter(c) && !IsDigit(c) && c != '_')
      break;
  }

  // If first char is '$', make sure there's at least one mare char, or it's
  // invalid.
  if (dollar_start && (m_cur_pos - start <= 1)) {
    m_cur_pos = start;
    return llvm::make_range(start, start); // Empty range
  }

  return llvm::make_range(start, m_cur_pos);
}

void DILLexer::UpdateLexedTokens(Token &result, Token::Kind tok_kind,
                                 std::string tok_str, uint32_t tok_pos) {
  Token new_token(tok_kind, tok_str, tok_pos);
  result = new_token;
  m_lexed_tokens.push_back(std::move(new_token));
}

llvm::Expected<bool> DILLexer::LexAll() {
  bool done = false;
  while (!done) {
    auto tok_or_err = Lex();
    if (!tok_or_err)
      return tok_or_err.takeError();
    Token token = *tok_or_err;
    if (token.GetKind() == Token::eof) {
      done = true;
    }
  }
  return true;
}

llvm::Expected<Token> DILLexer::Lex() {
  Token result;

  // Skip over whitespace (spaces).
  while (m_cur_pos != m_expr.end() && *m_cur_pos == ' ')
    m_cur_pos++;

  // Check to see if we've reached the end of our input string.
  if (m_cur_pos == m_expr.end()) {
    UpdateLexedTokens(result, Token::eof, "", (uint32_t)m_expr.size());
    return result;
  }

  uint32_t position = m_cur_pos - m_expr.begin();
  llvm::StringRef::iterator start = m_cur_pos;
  llvm::iterator_range<llvm::StringRef::iterator> word_range = IsWord();
  if (!word_range.empty()) {
    uint32_t length = word_range.end() - word_range.begin();
    llvm::StringRef word(m_expr.substr(position, length));
    // We will be adding more keywords here in the future...
    Token::Kind kind = llvm::StringSwitch<Token::Kind>(word)
                           .Case("namespace", Token::kw_namespace)
                           .Default(Token::identifier);
    UpdateLexedTokens(result, kind, word.str(), position);
    return result;
  }

  m_cur_pos = start;
  llvm::StringRef remainder(m_expr.substr(position, m_expr.end() - m_cur_pos));
  std::vector<std::pair<Token::Kind, const char *>> operators = {
      {Token::l_paren, "("},
      {Token::r_paren, ")"},
      {Token::coloncolon, "::"},
  };
  for (auto [kind, str] : operators) {
    if (remainder.consume_front(str)) {
      m_cur_pos += strlen(str);
      UpdateLexedTokens(result, kind, str, position);
      return result;
    }
  }

  // Unrecognized character(s) in string; unable to lex it.
  Status error("Unable to lex input string");
  return error.ToError();
}

const Token &DILLexer::LookAhead(uint32_t N) {
  if (m_tokens_idx + N + 1 < m_lexed_tokens.size())
    return m_lexed_tokens[m_tokens_idx + N + 1];

  return m_invalid_token;
}

const Token &DILLexer::AcceptLookAhead(uint32_t N) {
  if (m_tokens_idx + N + 1 > m_lexed_tokens.size())
    return m_invalid_token;

  m_tokens_idx += N + 1;
  return m_lexed_tokens[m_tokens_idx];
}

const Token &DILLexer::GetNextToken() {
  if (m_tokens_idx == UINT_MAX)
    m_tokens_idx = 0;
  else
    m_tokens_idx++;

  // Return the next token in the vector of lexed tokens.
  if (m_tokens_idx < m_lexed_tokens.size())
    return m_lexed_tokens[m_tokens_idx];

  // We're already at/beyond the end of our lexed tokens. If the last token
  // is an eof token, return it.
  if (m_lexed_tokens[m_lexed_tokens.size() - 1].GetKind() == Token::eof)
    return m_lexed_tokens[m_lexed_tokens.size() - 1];

  // Return the invalid token.
  return m_invalid_token;
}

} // namespace dil

} // namespace lldb_private
