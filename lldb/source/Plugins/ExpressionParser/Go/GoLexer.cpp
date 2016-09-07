//===-- GoLexer.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string.h>

#include "GoLexer.h"

using namespace lldb_private;

llvm::StringMap<GoLexer::TokenType> *GoLexer::m_keywords;

GoLexer::GoLexer(const char *src)
    : m_src(src), m_end(src + strlen(src)), m_last_token(TOK_INVALID, "") {}

bool GoLexer::SkipWhitespace() {
  bool saw_newline = false;
  for (; m_src < m_end; ++m_src) {
    if (*m_src == '\n')
      saw_newline = true;
    if (*m_src == '/' && !SkipComment())
      return saw_newline;
    else if (!IsWhitespace(*m_src))
      return saw_newline;
  }
  return saw_newline;
}

bool GoLexer::SkipComment() {
  if (m_src[0] == '/' && m_src[1] == '/') {
    for (const char *c = m_src + 2; c < m_end; ++c) {
      if (*c == '\n') {
        m_src = c - 1;
        return true;
      }
    }
    return true;
  } else if (m_src[0] == '/' && m_src[1] == '*') {
    for (const char *c = m_src + 2; c < m_end; ++c) {
      if (c[0] == '*' && c[1] == '/') {
        m_src = c + 1;
        return true;
      }
    }
  }
  return false;
}

const GoLexer::Token &GoLexer::Lex() {
  bool newline = SkipWhitespace();
  const char *start = m_src;
  m_last_token.m_type = InternalLex(newline);
  m_last_token.m_value = llvm::StringRef(start, m_src - start);
  return m_last_token;
}

GoLexer::TokenType GoLexer::InternalLex(bool newline) {
  if (m_src >= m_end) {
    return TOK_EOF;
  }
  if (newline) {
    switch (m_last_token.m_type) {
    case TOK_IDENTIFIER:
    case LIT_FLOAT:
    case LIT_IMAGINARY:
    case LIT_INTEGER:
    case LIT_RUNE:
    case LIT_STRING:
    case KEYWORD_BREAK:
    case KEYWORD_CONTINUE:
    case KEYWORD_FALLTHROUGH:
    case KEYWORD_RETURN:
    case OP_PLUS_PLUS:
    case OP_MINUS_MINUS:
    case OP_RPAREN:
    case OP_RBRACK:
    case OP_RBRACE:
      return OP_SEMICOLON;
    default:
      break;
    }
  }
  char c = *m_src;
  switch (c) {
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    return DoNumber();
  case '+':
  case '-':
  case '*':
  case '/':
  case '%':
  case '&':
  case '|':
  case '^':
  case '<':
  case '>':
  case '!':
  case ':':
  case ';':
  case '(':
  case ')':
  case '[':
  case ']':
  case '{':
  case '}':
  case ',':
  case '=':
    return DoOperator();
  case '.':
    if (IsDecimal(m_src[1]))
      return DoNumber();
    return DoOperator();
  case '$':
    // For lldb persistent vars.
    return DoIdent();
  case '"':
  case '`':
    return DoString();
  case '\'':
    return DoRune();
  default:
    break;
  }
  if (IsLetterOrDigit(c))
    return DoIdent();
  ++m_src;
  return TOK_INVALID;
}

GoLexer::TokenType GoLexer::DoOperator() {
  TokenType t = TOK_INVALID;
  if (m_end - m_src > 2) {
    t = LookupKeyword(llvm::StringRef(m_src, 3));
    if (t != TOK_INVALID)
      m_src += 3;
  }
  if (t == TOK_INVALID && m_end - m_src > 1) {
    t = LookupKeyword(llvm::StringRef(m_src, 2));
    if (t != TOK_INVALID)
      m_src += 2;
  }
  if (t == TOK_INVALID) {
    t = LookupKeyword(llvm::StringRef(m_src, 1));
    ++m_src;
  }
  return t;
}

GoLexer::TokenType GoLexer::DoIdent() {
  const char *start = m_src++;
  while (m_src < m_end && IsLetterOrDigit(*m_src)) {
    ++m_src;
  }
  TokenType kw = LookupKeyword(llvm::StringRef(start, m_src - start));
  if (kw != TOK_INVALID)
    return kw;
  return TOK_IDENTIFIER;
}

GoLexer::TokenType GoLexer::DoNumber() {
  if (m_src[0] == '0' && (m_src[1] == 'x' || m_src[1] == 'X')) {
    m_src += 2;
    while (IsHexChar(*m_src))
      ++m_src;
    return LIT_INTEGER;
  }
  bool dot_ok = true;
  bool e_ok = true;
  while (true) {
    while (IsDecimal(*m_src))
      ++m_src;
    switch (*m_src) {
    case 'i':
      ++m_src;
      return LIT_IMAGINARY;
    case '.':
      if (!dot_ok)
        return LIT_FLOAT;
      ++m_src;
      dot_ok = false;
      break;
    case 'e':
    case 'E':
      if (!e_ok)
        return LIT_FLOAT;
      dot_ok = e_ok = false;
      ++m_src;
      if (*m_src == '+' || *m_src == '-')
        ++m_src;
      break;
    default:
      if (dot_ok)
        return LIT_INTEGER;
      return LIT_FLOAT;
    }
  }
}

GoLexer::TokenType GoLexer::DoRune() {
  while (++m_src < m_end) {
    switch (*m_src) {
    case '\'':
      ++m_src;
      return LIT_RUNE;
    case '\n':
      return TOK_INVALID;
    case '\\':
      if (m_src[1] == '\n')
        return TOK_INVALID;
      ++m_src;
    }
  }
  return TOK_INVALID;
}

GoLexer::TokenType GoLexer::DoString() {
  if (*m_src == '`') {
    while (++m_src < m_end) {
      if (*m_src == '`') {
        ++m_src;
        return LIT_STRING;
      }
    }
    return TOK_INVALID;
  }
  while (++m_src < m_end) {
    switch (*m_src) {
    case '"':
      ++m_src;
      return LIT_STRING;
    case '\n':
      return TOK_INVALID;
    case '\\':
      if (m_src[1] == '\n')
        return TOK_INVALID;
      ++m_src;
    }
  }
  return TOK_INVALID;
}

GoLexer::TokenType GoLexer::LookupKeyword(llvm::StringRef id) {
  if (m_keywords == nullptr)
    m_keywords = InitKeywords();
  const auto &it = m_keywords->find(id);
  if (it == m_keywords->end())
    return TOK_INVALID;
  return it->second;
}

llvm::StringRef GoLexer::LookupToken(TokenType t) {
  if (m_keywords == nullptr)
    m_keywords = InitKeywords();
  for (const auto &e : *m_keywords) {
    if (e.getValue() == t)
      return e.getKey();
  }
  return "";
}

llvm::StringMap<GoLexer::TokenType> *GoLexer::InitKeywords() {
  auto &result = *new llvm::StringMap<TokenType>(128);
  result["break"] = KEYWORD_BREAK;
  result["default"] = KEYWORD_DEFAULT;
  result["func"] = KEYWORD_FUNC;
  result["interface"] = KEYWORD_INTERFACE;
  result["select"] = KEYWORD_SELECT;
  result["case"] = KEYWORD_CASE;
  result["defer"] = KEYWORD_DEFER;
  result["go"] = KEYWORD_GO;
  result["map"] = KEYWORD_MAP;
  result["struct"] = KEYWORD_STRUCT;
  result["chan"] = KEYWORD_CHAN;
  result["else"] = KEYWORD_ELSE;
  result["goto"] = KEYWORD_GOTO;
  result["package"] = KEYWORD_PACKAGE;
  result["switch"] = KEYWORD_SWITCH;
  result["const"] = KEYWORD_CONST;
  result["fallthrough"] = KEYWORD_FALLTHROUGH;
  result["if"] = KEYWORD_IF;
  result["range"] = KEYWORD_RANGE;
  result["type"] = KEYWORD_TYPE;
  result["continue"] = KEYWORD_CONTINUE;
  result["for"] = KEYWORD_FOR;
  result["import"] = KEYWORD_IMPORT;
  result["return"] = KEYWORD_RETURN;
  result["var"] = KEYWORD_VAR;
  result["+"] = OP_PLUS;
  result["-"] = OP_MINUS;
  result["*"] = OP_STAR;
  result["/"] = OP_SLASH;
  result["%"] = OP_PERCENT;
  result["&"] = OP_AMP;
  result["|"] = OP_PIPE;
  result["^"] = OP_CARET;
  result["<<"] = OP_LSHIFT;
  result[">>"] = OP_RSHIFT;
  result["&^"] = OP_AMP_CARET;
  result["+="] = OP_PLUS_EQ;
  result["-="] = OP_MINUS_EQ;
  result["*="] = OP_STAR_EQ;
  result["/="] = OP_SLASH_EQ;
  result["%="] = OP_PERCENT_EQ;
  result["&="] = OP_AMP_EQ;
  result["|="] = OP_PIPE_EQ;
  result["^="] = OP_CARET_EQ;
  result["<<="] = OP_LSHIFT_EQ;
  result[">>="] = OP_RSHIFT_EQ;
  result["&^="] = OP_AMP_CARET_EQ;
  result["&&"] = OP_AMP_AMP;
  result["||"] = OP_PIPE_PIPE;
  result["<-"] = OP_LT_MINUS;
  result["++"] = OP_PLUS_PLUS;
  result["--"] = OP_MINUS_MINUS;
  result["=="] = OP_EQ_EQ;
  result["<"] = OP_LT;
  result[">"] = OP_GT;
  result["="] = OP_EQ;
  result["!"] = OP_BANG;
  result["!="] = OP_BANG_EQ;
  result["<="] = OP_LT_EQ;
  result[">="] = OP_GT_EQ;
  result[":="] = OP_COLON_EQ;
  result["..."] = OP_DOTS;
  result["("] = OP_LPAREN;
  result["["] = OP_LBRACK;
  result["{"] = OP_LBRACE;
  result[","] = OP_COMMA;
  result["."] = OP_DOT;
  result[")"] = OP_RPAREN;
  result["]"] = OP_RBRACK;
  result["}"] = OP_RBRACE;
  result[";"] = OP_SEMICOLON;
  result[":"] = OP_COLON;
  return &result;
}
