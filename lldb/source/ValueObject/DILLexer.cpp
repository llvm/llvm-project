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

const std::string DILToken::getTokenName(dil::TokenKind kind) {
  std::string retval;
  switch (kind){
    case dil::TokenKind::l_square: retval = "l_square"; break;
    case dil::TokenKind::r_square: retval = "r_square"; break;
    case dil::TokenKind::l_paren: retval = "l_paren"; break;
    case dil::TokenKind::r_paren: retval = "r_paren"; break;
    case dil::TokenKind::star: retval = "star"; break;
    case dil::TokenKind::amp: retval = "amp"; break;
    case dil::TokenKind::arrow: retval = "arrow"; break;
    case dil::TokenKind::colon: retval = "colon"; break;
    case dil::TokenKind::coloncolon: retval = "coloncolon"; break;
    case dil::TokenKind::minus: retval = "minus"; break;
    case dil::TokenKind::period: retval = "period"; break;
    case dil::TokenKind::word: retval = "word"; break;
    case dil::TokenKind::integer: retval = "integer"; break;
    case dil::TokenKind::flt: retval = "flt"; break;
    case dil::TokenKind::string_literal: retval = "string_literal"; break;
    case dil::TokenKind::wide_string_literal: retval = "wide_string_literal"; break;
    case dil::TokenKind::utf8_string_literal: retval = "utf8_string_literal"; break;
    case dil::TokenKind::exclaim: retval = "exclaim"; break;
    case dil::TokenKind::tilde: retval = "tilde"; break;
    case dil::TokenKind::numeric_constant: retval = "numeric_constant"; break;
    case dil::TokenKind::char_constant: retval = "char_constant"; break;
    case dil::TokenKind::wide_char_constant: retval = "wide_char_constant"; break;
    case dil::TokenKind::utf8_char_constant: retval = "utf8_char_constant"; break;
    case dil::TokenKind::identifier: retval = "identifier"; break;
    case dil::TokenKind::caret: retval = "caret"; break;
    case dil::TokenKind::caretequal: retval = "caretequal"; break;
    case dil::TokenKind::ampequal: retval = "ampequal"; break;
    case dil::TokenKind::ampamp: retval = "ampamp"; break;
    case dil::TokenKind::equal: retval = "equal"; break;
    case dil::TokenKind::equalequal: retval = "equalequal"; break;
    case dil::TokenKind::exclaimequal: retval = "exclaimequal"; break;
    case dil::TokenKind::pipe: retval = "pipe"; break;
    case dil::TokenKind::pipeequal: retval = "pipeequal"; break;
    case dil::TokenKind::pipepipe: retval = "pipepipe"; break;
    case dil::TokenKind::plus: retval = "plus"; break;
    case dil::TokenKind::plusequal: retval = "plusequal"; break;
    case dil::TokenKind::plusplus: retval = "plusplus"; break;
    case dil::TokenKind::minusequal: retval = "minusequal"; break;
    case dil::TokenKind::minusminus: retval = "minusminus"; break;
    case dil::TokenKind::slash: retval = "slash"; break;
    case dil::TokenKind::percent: retval = "percent"; break;
    case dil::TokenKind::percentequal: retval = "percentequal"; break;
    case dil::TokenKind::slashequal: retval = "slashequal"; break;
    case dil::TokenKind::starequal: retval = "starequal"; break;
    case dil::TokenKind::question: retval = "question"; break;
    case dil::TokenKind::less: retval = "less"; break;
    case dil::TokenKind::greater: retval = "greater"; break;
    case dil::TokenKind::greaterequal: retval = "greaterequal"; break;
    case dil::TokenKind::greatergreater: retval = "greatergreater"; break;
    case dil::TokenKind::greatergreaterequal: retval = "greatergreaterequal"; break;
    case dil::TokenKind::lessequal: retval = "lessequal"; break;
    case dil::TokenKind::lessless: retval = "lessless"; break;
    case dil::TokenKind::lesslessequal: retval = "lesslessequal"; break;
    case dil::TokenKind::comma: retval = "comma"; break;
    default:
    retval = "token_name";
    break;
  }
  return retval;
}

bool DILLexer::Lex(DILToken &result, bool look_ahead) {
  bool retval = true;

  if (!look_ahead)
    ClearTokenCache();

  if (m_cur_pos == m_expr.end()) {
    result.setValues(dil::TokenKind::eof, "", m_expr.length(), 0);
    return retval;
  }

  while (m_cur_pos != m_expr.end() && *m_cur_pos == ' ')
    m_cur_pos++;

  if (m_cur_pos == m_expr.end()) {
    result.setValues(dil::TokenKind::eof, "", m_expr.length(), 0);
    return retval;
  }

  uint32_t position = m_cur_pos - m_expr.begin();;
  std::string::iterator start = m_cur_pos;
  uint32_t length = 0;
  dil::NumberKind kind = dil::NumberKind::eNone;
  if (Is_Number(start, length, kind)) {
    std::string number = m_expr.substr(position, length);
    if (kind == dil::NumberKind::eInteger) {
      result.setValues(dil::TokenKind::numeric_constant, number, position,
                       length);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    } else if (kind == dil::NumberKind::eFloat) {
      result.setValues(dil::TokenKind::numeric_constant, number, position,
                       length);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    }
  } else if (Is_Word(start, length)) {
    std::string word = m_expr.substr(position, length);
    if (word == "true")
      result.setValues(dil::TokenKind::kw_true, word, position, length);
    else if (word == "false")
      result.setValues(dil::TokenKind::kw_false, word, position, length);
    else if (word == "this")
      result.setValues(dil::TokenKind::kw_this, word, position, length);
    else if (word == "nullptr")
      result.setValues(dil::TokenKind::kw_nullptr, word, position, length);
    else if (word == "bool")
      result.setValues(dil::TokenKind::kw_bool, word, position, length);
    else if (word == "char")
      result.setValues(dil::TokenKind::kw_char, word, position, length);
    else if (word == "short")
      result.setValues(dil::TokenKind::kw_short, word, position, length);
    else if (word == "int")
      result.setValues(dil::TokenKind::kw_int, word, position, length);
    else if (word == "long")
      result.setValues(dil::TokenKind::kw_long, word, position, length);
    else if (word == "float")
      result.setValues(dil::TokenKind::kw_float, word, position, length);
    else if (word == "double")
      result.setValues(dil::TokenKind::kw_double, word, position, length);
    else if (word == "signed")
      result.setValues(dil::TokenKind::kw_signed, word, position, length);
    else if (word == "unsigned")
      result.setValues(dil::TokenKind::kw_unsigned, word, position, length);
    else if (word == "static_cast")
      result.setValues(dil::TokenKind::kw_static_cast, word, position, length);
    else if (word == "dynamic_cast")
      result.setValues(dil::TokenKind::kw_dynamic_cast, word, position,
                       length);
    else if (word == "reinterpret_cast")
      result.setValues(dil::TokenKind::kw_reinterpret_cast, word, position,
                       length);
    else
      result.setValues(dil::TokenKind::identifier,
                       m_expr.substr(position, length),
                       position, length);
    if (IsBacktracking()) {
      m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
    }
    return true;
  }

  switch (*m_cur_pos) {
    case '[':
      m_cur_pos++;
      result.setValues(dil::TokenKind::l_square, "[", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case ']':
      m_cur_pos++;
      result.setValues(dil::TokenKind::r_square, "]", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '(':
      m_cur_pos++;
      result.setValues(dil::TokenKind::l_paren, "(", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case ')':
      m_cur_pos++;
      result.setValues(dil::TokenKind::r_paren, ")", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '&':
      if (position+1 < m_expr.size() &&
          ((m_expr[position+1] == '=') || (m_expr[position+1] == '&'))) {
        m_cur_pos += 2;
        if (m_expr[position+1] == '&')
          result.setValues(dil::TokenKind::ampamp, "&&", position, 2);
        else
          result.setValues(dil::TokenKind::ampequal, "&=", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::amp, "&", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '|':
      if (position+1 < m_expr.size() &&
          ((m_expr[position+1] == '=') || (m_expr[position+1] == '|'))) {
        m_cur_pos += 2;
        if (m_expr[position+1] == '|')
          result.setValues(dil::TokenKind::pipepipe, "||", position, 2);
        else
          result.setValues(dil::TokenKind::pipeequal, "|=", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::pipe, "|", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '+':
      if (position+1 < m_expr.size() &&
          ((m_expr[position+1] == '=') || (m_expr[position+1] == '+'))) {
        m_cur_pos += 2;
        if (m_expr[position+1] == '+')
          result.setValues(dil::TokenKind::plusplus, "++", position, 2);
        else
          result.setValues(dil::TokenKind::plusequal, "+=", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::plus, "+", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '-':
      if (position+1 < m_expr.size() &&
          ((m_expr[position+1] == '=') || (m_expr[position+1] == '-') ||
           (m_expr[position+1] == '>'))) {
        m_cur_pos += 2;
        if (m_expr[position+1] == '-')
          result.setValues(dil::TokenKind::minusminus, "--", position, 2);
        else if (m_expr[position+1] == '>')
          result.setValues(dil::TokenKind::arrow, "->", position, 2);
        else
          result.setValues(dil::TokenKind::minusequal, "-=", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::minus, "-", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '^':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        result.setValues(dil::TokenKind::caretequal, "^=", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::caret, "^", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '=':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        result.setValues(dil::TokenKind::equalequal, "==", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::equal, "=", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '!':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        result.setValues(dil::TokenKind::exclaimequal, "!=", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::exclaim, "!", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '%':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        result.setValues(dil::TokenKind::percentequal, "%=", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::percent, "%", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '/':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        result.setValues(dil::TokenKind::slashequal, "/=", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::slash, "/", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '*':
      if (position+1 < m_expr.size() && m_expr[position+1] == '=') {
        m_cur_pos += 2;
        result.setValues(dil::TokenKind::starequal, "*=", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::star, "*", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case ':':
      if (position+1 < m_expr.size() && m_expr[position+1] == ':') {
        m_cur_pos += 2;
        result.setValues(dil::TokenKind::coloncolon, "::", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::colon, ":", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '<':
      if (position+2 < m_expr.size() && m_expr[position+1] == '<' &&
          m_expr[position+2] == '=') {
        m_cur_pos += 3;
        result.setValues(dil::TokenKind::lesslessequal, "<<=", position, 3);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      } else if (position+1 < m_expr.size() &&
                 ((m_expr[position+1] == '<') || (m_expr[position+1] == '='))){
        m_cur_pos += 2;
        if (m_expr[position+1] == '<')
          result.setValues(dil::TokenKind::lessless, "<<", position, 2);
        else
          result.setValues(dil::TokenKind::lessequal, "<=", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::less, "<", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '>':
      if (position+2 < m_expr.size() && m_expr[position+1] == '>' &&
          m_expr[position+2] == '=') {
        m_cur_pos += 3;
        result.setValues(dil::TokenKind::greatergreaterequal, ">>=", position,
                         3);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      } else if (position+1 < m_expr.size() &&
                 ((m_expr[position+1] == '>') || (m_expr[position+1] == '='))){
        m_cur_pos += 2;
        if (m_expr[position+1] == '>')
          result.setValues(dil::TokenKind::greatergreater, ">>", position, 2);
        else
          result.setValues(dil::TokenKind::greaterequal, ">=", position, 2);
        if (IsBacktracking()) {
          m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
        }
        return true;
      }
      m_cur_pos++;
      result.setValues(dil::TokenKind::greater, ">", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '.':
      m_cur_pos++;
      result.setValues(dil::TokenKind::period, ".", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '?':
      m_cur_pos++;
      result.setValues(dil::TokenKind::question, "?", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case ',':
      m_cur_pos++;
      result.setValues(dil::TokenKind::question, ",", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    case '~':
      m_cur_pos++;
      result.setValues(dil::TokenKind::tilde, ",", position, 1);
      if (IsBacktracking()) {
        m_backtrack_tokens.push_back(std::make_pair(result, m_cur_pos));
      }
      return true;
    default:
      break;
  }
  // Empty Token
  result.setValues(dil::TokenKind::none, "", m_expr.length(), 0);
  return false;
}

const DILToken &DILLexer::LookAhead(uint32_t N) {
  if (N < m_cached_tokens.size())
    return m_cached_tokens[N].first;

  bool done = false;
  bool look_ahead = true;
  std::string::iterator save_cur_pos = m_cur_pos;
  uint32_t remaining_tokens = N - m_cached_tokens.size() + 1;
  while (!done && remaining_tokens > 0) {
    if (!m_cached_tokens.empty())
      m_cur_pos = m_cached_tokens.back().second;
    DILToken tok;
    Lex(tok, look_ahead);
    if (tok.getKind() == dil::TokenKind::eof)
      done = true;
    std::pair<DILToken, std::string::iterator> result(tok, m_cur_pos);
    m_cached_tokens.push_back(result);
    remaining_tokens--;
  };
  m_cur_pos = save_cur_pos;
  if (remaining_tokens > 0) {
    DILToken invalid_token = DILToken(dil::TokenKind::invalid, "", 0, 0);
    std::pair<DILToken, std::string::iterator> result(invalid_token,
                                                      m_cur_pos);
    m_cached_tokens.push_back(result);
    return m_cached_tokens.back().first;
  } else
    return m_cached_tokens[N].first;
}

void DILLexer::EnableBacktrackAtThisPos() {
  m_is_backtracking = true;
  m_backtracking_startpos = m_cur_pos;
}

void DILLexer::CommitBacktrackedTokens() {
  if (!IsBacktracking())
    return; //CAROLINE:: Should there be an error here?

  for (auto t : m_backtrack_tokens)
    m_cached_tokens.push_back(t);

  ClearBacktrackTokens();
  m_is_backtracking = false;
  return;
}

void DILLexer::Backtrack() {
  if (!IsBacktracking())
    return; // CAROLINE: Should there be an error here?

  ClearBacktrackTokens();
  m_cur_pos = m_backtracking_startpos;
  m_is_backtracking = false;
  return;
}

} // namespace dil

} // namespace lldb_private


/* JUST FOR TESTING!!
int main (int argc, char** argv) {
  std::string expr("foo->bar[0]");
  std::string expr2 = "(int) foo";
  std::string expr3 = "125 - 10.3";

  lldb_private::dil::DILLexer L1(expr);
  lldb_private::dil::DILLexer L2(expr2);
  lldb_private::dil::DILLexer L3(expr3);
  std::vector<lldb_private::dil::DILToken> l1_tokens;
  std::vector<lldb_private::dil::DILToken> l2_tokens;
  std::vector<lldb_private::dil::DILToken> l3_tokens;

  bool look_ahead_works = false;
  if (L1.LookAhead(0).Is(lldb_private::dil::TokenKind::word)) {
    if (L1.LookAhead(1).Is(lldb_private::dil::TokenKind::arrow)) {
      if (L1.LookAhead(4).Is(lldb_private::dil::TokenKind::integer)) {
        if (L1.LookAhead(3).Is(lldb_private::dil::TokenKind::l_square)) {
          look_ahead_works = true;
          L1.ClearTokenCache();
        }
      }
    }
  }

  if (! L1.LookAhead(8).Is(lldb_private::dil::TokenKind::invalid))
    look_ahead_works = false;

  L1.ClearTokenCache();
  lldb_private::dil::DILToken my_token = L1.Lex();
  l1_tokens.push_back(my_token);
  while (my_token.getKind() != lldb_private::dil::TokenKind::eof) {
    my_token = L1.Lex();
    l1_tokens.push_back(my_token);
  }

  my_token = L2.Lex();
  l2_tokens.push_back(my_token);
  while (my_token.getKind() != lldb_private::dil::TokenKind::eof) {
    my_token = L2.Lex();
    l2_tokens.push_back(my_token);
  }

  my_token = L3.Lex();
  l3_tokens.push_back(my_token);
  while (my_token.getKind() != lldb_private::dil::TokenKind::eof) {
    my_token = L3.Lex();
    l3_tokens.push_back(my_token);
  }

  return 0;
}
*/
