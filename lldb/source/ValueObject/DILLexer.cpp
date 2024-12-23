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

namespace lldb_private {

namespace dil {

const std::string DILToken::getTokenName(dil::TokenKind kind) {
  std::string retval;
  switch (kind) {
  case dil::TokenKind::coloncolon:
    retval = "coloncolon";
    break;
  case dil::TokenKind::eof:
    retval = "eof";
    break;
  case dil::TokenKind::identifier:
    retval = "identifier";
    break;
  case dil::TokenKind::kw_namespace:
    retval = "namespace";
    break;
  case dil::TokenKind::kw_this:
    retval = "this";
    break;
  case dil::TokenKind::l_paren:
    retval = "l_paren";
    break;
  case dil::TokenKind::r_paren:
    retval = "r_paren";
    break;
  case dil::TokenKind::unknown:
    retval = "unknown";
    break;
  default:
    retval = "token_name";
    break;
  }
  return retval;
}

static bool Is_Letter(char c) {
  if (('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z'))
    return true;
  return false;
}

static bool Is_Digit(char c) { return ('0' <= c && c <= '9'); }

bool DILLexer::Is_Word(std::string::iterator start, uint32_t &length) {
  bool done = false;
  for (; m_cur_pos != m_expr.end() && !done; ++m_cur_pos) {
    char c = *m_cur_pos;
    if (!Is_Letter(c) && !Is_Digit(c) && c != '_') {
      done = true;
      break;
    } else
      length++;
  }
  if (length > 0)
    return true;
  else
    m_cur_pos = start;
  return false;
}

void DILLexer::UpdateLexedTokens(DILToken &result, dil::TokenKind tok_kind,
                                 std::string tok_str, uint32_t tok_pos,
                                 uint32_t tok_len) {
  DILToken new_token;
  result.setValues(tok_kind, tok_str, tok_pos, tok_len);
  new_token = result;
  m_lexed_tokens.push_back(std::move(new_token));
}

bool DILLexer::Lex(DILToken &result, bool look_ahead) {
  bool retval = true;

  if (!look_ahead) {
    // We're being asked for the 'next' token, and not a part of a LookAhead.
    // Check to see if we've already lexed it and pushed it onto our tokens
    // vector; if so, return the next token from the vector, rather than doing
    // more lexing.
    if ((m_tokens_idx != UINT_MAX) &&
        (m_tokens_idx < m_lexed_tokens.size() - 1)) {
      result = m_lexed_tokens[m_tokens_idx + 1];
      return retval;
    }
  }

  // Skip over whitespace (spaces).
  while (m_cur_pos != m_expr.end() && *m_cur_pos == ' ')
    m_cur_pos++;

  // Check to see if we've reached the end of our input string.
  if (m_cur_pos == m_expr.end()) {
    UpdateLexedTokens(result, dil::TokenKind::eof, "", m_expr.length(), 0);
    return retval;
  }

  uint32_t position = m_cur_pos - m_expr.begin();
  ;
  std::string::iterator start = m_cur_pos;
  uint32_t length = 0;
  if (Is_Word(start, length)) {
    dil::TokenKind kind;
    std::string word = m_expr.substr(position, length);
    if (word == "this")
      kind = dil::TokenKind::kw_this;
    else if (word == "namespace")
      kind = dil::TokenKind::kw_namespace;
    else
      kind = dil::TokenKind::identifier;

    UpdateLexedTokens(result, kind, word, position, length);
    return true;
  }

  switch (*m_cur_pos) {
  case '(':
    m_cur_pos++;
    UpdateLexedTokens(result, dil::TokenKind::l_paren, "(", position, 1);
    return true;
  case ')':
    m_cur_pos++;
    UpdateLexedTokens(result, dil::TokenKind::r_paren, ")", position, 1);
    return true;
  case ':':
    if (position + 1 < m_expr.size() && m_expr[position + 1] == ':') {
      m_cur_pos += 2;
      UpdateLexedTokens(result, dil::TokenKind::coloncolon, "::", position, 2);
      return true;
    }
    break;
  default:
    break;
  }
  // Empty Token
  result.setValues(dil::TokenKind::none, "", m_expr.length(), 0);
  return false;
}

const DILToken &DILLexer::LookAhead(uint32_t N) {
  uint32_t extra_lexed_tokens = m_lexed_tokens.size() - m_tokens_idx - 1;

  if (N + 1 < extra_lexed_tokens)
    return m_lexed_tokens[m_tokens_idx + N + 1];

  uint32_t remaining_tokens =
      (m_tokens_idx + N + 1) - m_lexed_tokens.size() + 1;

  bool done = false;
  bool look_ahead = true;
  while (!done && remaining_tokens > 0) {
    DILToken tok;
    Lex(tok, look_ahead);
    if (tok.getKind() == dil::TokenKind::eof)
      done = true;
    remaining_tokens--;
  };

  if (remaining_tokens > 0) {
    m_invalid_token.setValues(dil::TokenKind::invalid, "", 0, 0);
    return m_invalid_token;
  } else
    return m_lexed_tokens[m_tokens_idx + N + 1];
}

const DILToken &DILLexer::AcceptLookAhead(uint32_t N) {
  if (m_tokens_idx + N + 1 > m_lexed_tokens.size())
    return m_invalid_token;

  m_tokens_idx += N + 1;
  return m_lexed_tokens[m_tokens_idx];
}

} // namespace dil

} // namespace lldb_private
