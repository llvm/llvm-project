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
  switch (kind){
    case dil::TokenKind::amp: retval = "amp"; break;
    case dil::TokenKind::ampamp: retval = "ampamp"; break;
    case dil::TokenKind::ampequal: retval = "ampequal"; break;
    case dil::TokenKind::arrow: retval = "arrow"; break;
    case dil::TokenKind::caret: retval = "caret"; break;
    case dil::TokenKind::caretequal: retval = "caretequal"; break;
    case dil::TokenKind::colon: retval = "colon"; break;
    case dil::TokenKind::coloncolon: retval = "coloncolon"; break;
    case dil::TokenKind::comma: retval = "comma"; break;
    case dil::TokenKind::eof: retval = "eof"; break;
    case dil::TokenKind::equalequal: retval = "equalequal"; break;
    case dil::TokenKind::exclaim: retval = "exclaim"; break;
    case dil::TokenKind::exclaimequal: retval = "exclaimequal"; break;
    case dil::TokenKind::flt: retval = "flt"; break;
    case dil::TokenKind::greater: retval = "greater"; break;
    case dil::TokenKind::greaterequal: retval = "greaterequal"; break;
    case dil::TokenKind::greatergreater: retval = "greatergreater"; break;
    case dil::TokenKind::greatergreaterequal: retval = "greatergreaterequal";
      break;
    case dil::TokenKind::identifier: retval = "identifier"; break;
    case dil::TokenKind::integer: retval = "integer"; break;
    case dil::TokenKind::less: retval = "less"; break;
    case dil::TokenKind::lessequal: retval = "lessequal"; break;
    case dil::TokenKind::lessless: retval = "lessless"; break;
    case dil::TokenKind::lesslessequal: retval = "lesslessequal"; break;
    case dil::TokenKind::l_square: retval = "l_square"; break;
    case dil::TokenKind::l_paren: retval = "l_paren"; break;
    case dil::TokenKind::minus: retval = "minus"; break;
    case dil::TokenKind::minusequal: retval = "minusequal"; break;
    case dil::TokenKind::minusminus: retval = "minusminus"; break;
    case dil::TokenKind::numeric_constant: retval = "numeric_constant"; break;
    case dil::TokenKind::percent: retval = "percent"; break;
    case dil::TokenKind::percentequal: retval = "percentequal"; break;
    case dil::TokenKind::period: retval = "period"; break;
    case dil::TokenKind::pipe: retval = "pipe"; break;
    case dil::TokenKind::pipeequal: retval = "pipeequal"; break;
    case dil::TokenKind::pipepipe: retval = "pipepipe"; break;
    case dil::TokenKind::plus: retval = "plus"; break;
    case dil::TokenKind::plusequal: retval = "plusequal"; break;
    case dil::TokenKind::plusplus: retval = "plusplus"; break;
    case dil::TokenKind::question: retval = "question"; break;
    case dil::TokenKind::r_paren: retval = "r_paren"; break;
    case dil::TokenKind::r_square: retval = "r_square"; break;
    case dil::TokenKind::slash: retval = "slash"; break;
    case dil::TokenKind::slashequal: retval = "slashequal"; break;
    case dil::TokenKind::star: retval = "star"; break;
    case dil::TokenKind::starequal: retval = "starequal"; break;
    case dil::TokenKind::string_literal: retval = "string_literal"; break;
    case dil::TokenKind::tilde: retval = "tilde"; break;
    case dil::TokenKind::utf8_string_literal: retval = "utf8_string_literal";
      break;
    case dil::TokenKind::wide_string_literal: retval = "wide_string_literal";
      break;
    case dil::TokenKind::word: retval = "word"; break;
    case dil::TokenKind::char_constant: retval = "char_constant"; break;
    case dil::TokenKind::wide_char_constant: retval = "wide_char_constant";
      break;
    case dil::TokenKind::utf8_char_constant: retval = "utf8_char_constant";
      break;
    default:
    retval = "token_name";
    break;
  }
  return retval;
}

static bool Is_Letter (char c) {
  if (('a' <= c && c <= 'z') ||
      ('A' <= c && c <= 'Z'))
    return true;
  return false;
}

static bool Is_Digit (char c) {
  return ('0' <= c && c <= '9');
}

bool DILLexer::Is_Word(std::string::iterator start, uint32_t& length) {
  bool done = false;
  for ( ; m_cur_pos != m_expr.end() && !done; ++m_cur_pos) {
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

bool DILLexer::Is_Number(std::string::iterator start, uint32_t& length,
                         dil::NumberKind& kind) {

  if (Is_Digit(*start)) {
    while (m_cur_pos != m_expr.end() && Is_Digit(*m_cur_pos)) {
      length++;
      m_cur_pos++;
    }
    if (m_cur_pos == m_expr.end() || (*m_cur_pos != '.')) {
      kind = dil::NumberKind::eInteger;
      return true;
    }
    // We're not at the end of the string, and we should be looking at a '.'
    if (*m_cur_pos == '.') {
      length++;
      m_cur_pos++;
      while (m_cur_pos != m_expr.end() && Is_Digit(*m_cur_pos)) {
        length++;
        m_cur_pos++;
      }
      kind = dil::NumberKind::eFloat;
      return true;
    }
  }
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
  DILToken new_token;

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

  uint32_t position = m_cur_pos - m_expr.begin();;
  std::string::iterator start = m_cur_pos;
  uint32_t length = 0;
  dil::NumberKind kind = dil::NumberKind::eNone;
  if (Is_Number(start, length, kind)) {
    std::string number = m_expr.substr(position, length);
    if (kind == dil::NumberKind::eInteger) {
      UpdateLexedTokens(result, dil::TokenKind::numeric_constant, number,
                        position, length);
      return true;
    } else if (kind == dil::NumberKind::eFloat) {
      UpdateLexedTokens(result, dil::TokenKind::numeric_constant, number,
                        position, length);
      return true;
    }
  } else if (Is_Word(start, length)) {
    dil::TokenKind kind;
    std::string word = m_expr.substr(position, length);
    if (word == "bool")
      kind = dil::TokenKind::kw_bool;
    else if (word == "char")
      kind = dil::TokenKind::kw_char;
    else if (word == "double")
      kind = dil::TokenKind::kw_double;
    else if (word == "dynamic_cast")
      kind = dil::TokenKind::kw_dynamic_cast;
    else if (word == "false")
      kind = dil::TokenKind::kw_false;
    else if (word == "float")
      kind = dil::TokenKind::kw_float;
    else if (word == "int")
      kind = dil::TokenKind::kw_int;
    else if (word == "long")
      kind = dil::TokenKind::kw_long;
    else if (word == "nullptr")
      kind = dil::TokenKind::kw_nullptr;
    else if (word == "reinterpret_cast")
      kind = dil::TokenKind::kw_reinterpret_cast;
    else if (word == "short")
      kind = dil::TokenKind::kw_short;
    else if (word == "signed")
      kind = dil::TokenKind::kw_signed;
    else if (word == "static_cast")
      kind = dil::TokenKind::kw_static_cast;
    else if (word == "this")
      kind = dil::TokenKind::kw_this;
   else if (word == "true")
      kind = dil::TokenKind::kw_true;
    else if (word == "unsigned")
      kind = dil::TokenKind::kw_unsigned;
    else
      kind = dil::TokenKind::identifier;
    UpdateLexedTokens(result, kind, word, position,
                      length);
    return true;
  }

  m_cur_pos = start;
  switch (*m_cur_pos) {
    case '[':
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::l_square, "[", position, 1);
      return true;
    case ']':
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::r_square, "]", position, 1);
      return true;
    case '(':
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::l_paren, "(", position, 1);
      return true;
    case ')':
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::r_paren, ")", position, 1);
      return true;
    case '&':
      if (position+1 < m_expr.size() &&
          ((m_expr[position+1] == '=') || (m_expr[position+1] == '&'))) {
        m_cur_pos += 2;
        if (m_expr[position+1] == '&')
          UpdateLexedTokens(result, dil::TokenKind::ampamp, "&&", position, 2);
        else
          UpdateLexedTokens(result, dil::TokenKind::ampequal, "&=", position, 2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::amp, "&", position, 1);
      return true;
    case '|':
      if (position+1 < m_expr.size() &&
          ((m_expr[position+1] == '=') || (m_expr[position+1] == '|'))) {
        m_cur_pos += 2;
        if (m_expr[position+1] == '|')
          UpdateLexedTokens(result, dil::TokenKind::pipepipe, "||", position, 2);
        else
          UpdateLexedTokens(result, dil::TokenKind::pipeequal, "|=", position,
                            2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::pipe, "|", position, 1);
      return true;
    case '+':
      if (position+1 < m_expr.size() &&
          ((m_expr[position+1] == '=') || (m_expr[position+1] == '+'))) {
        m_cur_pos += 2;
        if (m_expr[position+1] == '+')
          UpdateLexedTokens(result, dil::TokenKind::plusplus, "++", position, 2);
        else
          UpdateLexedTokens(result, dil::TokenKind::plusequal, "+=", position,
                            2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::plus, "+", position, 1);
      return true;
    case '-':
      if (position+1 < m_expr.size() &&
          ((m_expr[position+1] == '=') || (m_expr[position+1] == '-') ||
           (m_expr[position+1] == '>'))) {
        m_cur_pos += 2;
        if (m_expr[position+1] == '-')
          UpdateLexedTokens(result, dil::TokenKind::minusminus, "--", position,
                            2);
        else if (m_expr[position+1] == '>')
          UpdateLexedTokens(result, dil::TokenKind::arrow, "->", position, 2);
        else
          UpdateLexedTokens(result, dil::TokenKind::minusequal, "-=", position,
                            2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::minus, "-", position, 1);
      return true;
    case '^':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        UpdateLexedTokens(result, dil::TokenKind::caretequal, "^=", position, 2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::caret, "^", position, 1);
      return true;
    case '=':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        UpdateLexedTokens(result, dil::TokenKind::equalequal, "==", position, 2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::equal, "=", position, 1);
      return true;
    case '!':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        UpdateLexedTokens(result, dil::TokenKind::exclaimequal, "!=", position,
                          2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::exclaim, "!", position, 1);
      return true;
    case '%':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        UpdateLexedTokens(result, dil::TokenKind::percentequal, "%=", position,
                          2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::percent, "%", position, 1);
      return true;
    case '/':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        UpdateLexedTokens(result, dil::TokenKind::slashequal, "/=", position, 2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::slash, "/", position, 1);
      return true;
    case '*':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        UpdateLexedTokens(result, dil::TokenKind::starequal, "*=", position, 2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::star, "*", position, 1);
      return true;
    case ':':
      if (position+1 < m_expr.size() && m_expr[position+1] == ':') {
        m_cur_pos += 2;
        UpdateLexedTokens(result, dil::TokenKind::coloncolon, "::", position, 2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::colon, ":", position, 1);
      return true;
    case '<':
      if (position+2 < m_expr.size() && m_expr[position+1] == '<' &&
          m_expr[position+2] == '=') {
        m_cur_pos += 3;
        UpdateLexedTokens(result, dil::TokenKind::lesslessequal, "<<=", position,
                          3);
        return true;
      } else if (position+1 < m_expr.size() &&
                 ((m_expr[position+1] == '<') || (m_expr[position+1] == '='))){
        m_cur_pos += 2;
        if (m_expr[position+1] == '<')
          UpdateLexedTokens(result, dil::TokenKind::lessless, "<<", position, 2);
        else
          UpdateLexedTokens(result, dil::TokenKind::lessequal, "<=", position,
                            2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::less, "<", position, 1);
      return true;
    case '>':
      if (position+2 < m_expr.size() && m_expr[position+1] == '>' &&
          m_expr[position+2] == '=') {
        m_cur_pos += 3;
        UpdateLexedTokens(result, dil::TokenKind::greatergreaterequal, ">>=",
                          position, 3);
        new_token = result;
        m_lexed_tokens.push_back(std::move(new_token));
        return true;
      } else if (position+1 < m_expr.size() &&
                 ((m_expr[position+1] == '>') || (m_expr[position+1] == '='))){
        m_cur_pos += 2;
        if (m_expr[position+1] == '>')
          UpdateLexedTokens(result, dil::TokenKind::greatergreater, ">>",
                            position, 2);
        else
          UpdateLexedTokens(result, dil::TokenKind::greaterequal, ">=", position,
                            2);
        return true;
      }
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::greater, ">", position, 1);
      return true;
    case '.':
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::period, ".", position, 1);
      return true;
    case '?':
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::question, "?", position, 1);
      return true;
    case ',':
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::comma, ",", position, 1);
      return true;
    case '~':
      m_cur_pos++;
      UpdateLexedTokens(result, dil::TokenKind::tilde, ",", position, 1);
      return true;
    default:
      break;
  }
  // Empty Token
  result.setValues(dil::TokenKind::none, "", m_expr.length(), 0);
  return false;
}

const DILToken &DILLexer::LookAhead(uint32_t N) {
  uint32_t extra_lexed_tokens = m_lexed_tokens.size() - m_tokens_idx - 1;

  if (N+1 < extra_lexed_tokens)
    return m_lexed_tokens[m_tokens_idx + N + 1];

  uint32_t remaining_tokens = (m_tokens_idx + N + 1) -
                              m_lexed_tokens.size() + 1;

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
