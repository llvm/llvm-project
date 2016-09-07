//===-- GoLexer.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GoLexer_h
#define liblldb_GoLexer_h

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

class GoLexer {
public:
  explicit GoLexer(const char *src);

  enum TokenType {
    TOK_EOF,
    TOK_INVALID,
    TOK_IDENTIFIER,
    LIT_INTEGER,
    LIT_FLOAT,
    LIT_IMAGINARY,
    LIT_RUNE,
    LIT_STRING,
    KEYWORD_BREAK,
    KEYWORD_DEFAULT,
    KEYWORD_FUNC,
    KEYWORD_INTERFACE,
    KEYWORD_SELECT,
    KEYWORD_CASE,
    KEYWORD_DEFER,
    KEYWORD_GO,
    KEYWORD_MAP,
    KEYWORD_STRUCT,
    KEYWORD_CHAN,
    KEYWORD_ELSE,
    KEYWORD_GOTO,
    KEYWORD_PACKAGE,
    KEYWORD_SWITCH,
    KEYWORD_CONST,
    KEYWORD_FALLTHROUGH,
    KEYWORD_IF,
    KEYWORD_RANGE,
    KEYWORD_TYPE,
    KEYWORD_CONTINUE,
    KEYWORD_FOR,
    KEYWORD_IMPORT,
    KEYWORD_RETURN,
    KEYWORD_VAR,
    OP_PLUS,
    OP_MINUS,
    OP_STAR,
    OP_SLASH,
    OP_PERCENT,
    OP_AMP,
    OP_PIPE,
    OP_CARET,
    OP_LSHIFT,
    OP_RSHIFT,
    OP_AMP_CARET,
    OP_PLUS_EQ,
    OP_MINUS_EQ,
    OP_STAR_EQ,
    OP_SLASH_EQ,
    OP_PERCENT_EQ,
    OP_AMP_EQ,
    OP_PIPE_EQ,
    OP_CARET_EQ,
    OP_LSHIFT_EQ,
    OP_RSHIFT_EQ,
    OP_AMP_CARET_EQ,
    OP_AMP_AMP,
    OP_PIPE_PIPE,
    OP_LT_MINUS,
    OP_PLUS_PLUS,
    OP_MINUS_MINUS,
    OP_EQ_EQ,
    OP_LT,
    OP_GT,
    OP_EQ,
    OP_BANG,
    OP_BANG_EQ,
    OP_LT_EQ,
    OP_GT_EQ,
    OP_COLON_EQ,
    OP_DOTS,
    OP_LPAREN,
    OP_LBRACK,
    OP_LBRACE,
    OP_COMMA,
    OP_DOT,
    OP_RPAREN,
    OP_RBRACK,
    OP_RBRACE,
    OP_SEMICOLON,
    OP_COLON,
  };

  struct Token {
    explicit Token(TokenType t, llvm::StringRef text)
        : m_type(t), m_value(text) {}
    TokenType m_type;
    llvm::StringRef m_value;
  };

  const Token &Lex();

  size_t BytesRemaining() const { return m_end - m_src; }
  llvm::StringRef GetString(int len) const {
    return llvm::StringRef(m_src, len);
  }

  static TokenType LookupKeyword(llvm::StringRef id);
  static llvm::StringRef LookupToken(TokenType t);

private:
  bool IsDecimal(char c) { return c >= '0' && c <= '9'; }
  bool IsHexChar(char c) {
    if (c >= '0' && c <= '9')
      return true;
    if (c >= 'A' && c <= 'F')
      return true;
    if (c >= 'a' && c <= 'f')
      return true;
    return false;
  }
  bool IsLetterOrDigit(char c) {
    if (c >= 'a' && c <= 'z')
      return true;
    if (c >= 'A' && c <= 'Z')
      return true;
    if (c == '_')
      return true;
    if (c >= '0' && c <= '9')
      return true;
    // Treat all non-ascii chars as letters for simplicity.
    return 0 != (c & 0x80);
  }
  bool IsWhitespace(char c) {
    switch (c) {
    case ' ':
    case '\t':
    case '\r':
      return true;
    }
    return false;
  }

  bool SkipWhitespace();
  bool SkipComment();

  TokenType InternalLex(bool newline);

  TokenType DoOperator();

  TokenType DoIdent();

  TokenType DoNumber();

  TokenType DoRune();

  TokenType DoString();

  static llvm::StringMap<TokenType> *InitKeywords();

  static llvm::StringMap<TokenType> *m_keywords;

  const char *m_src;
  const char *m_end;
  Token m_last_token;
};

} // namespace lldb_private

#endif
