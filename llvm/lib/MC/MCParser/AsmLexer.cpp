//===- AsmLexer.cpp - Lexer for Assembly Files ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the lexer for assembly files.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/MC/MCAsmInfo.h"
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
using namespace llvm;

AsmLexer::AsmLexer(const MCAsmInfo &_MAI) : MAI(_MAI)  {
  CurBuf = NULL;
  CurPtr = NULL;
  isAtStartOfLine = true;
}

AsmLexer::~AsmLexer() {
}

void AsmLexer::setBuffer(const MemoryBuffer *buf, const char *ptr) {
  CurBuf = buf;

  if (ptr)
    CurPtr = ptr;
  else
    CurPtr = CurBuf->getBufferStart();

  TokStart = 0;
}

/// ReturnError - Set the error to the specified string at the specified
/// location.  This is defined to always return AsmToken::Error.
AsmToken AsmLexer::ReturnError(const char *Loc, const std::string &Msg) {
  SetError(SMLoc::getFromPointer(Loc), Msg);

  return AsmToken(AsmToken::Error, StringRef(Loc, 0));
}

int AsmLexer::getNextChar() {
  char CurChar = *CurPtr++;
  switch (CurChar) {
  default:
    return (unsigned char)CurChar;
  case 0:
    // A nul character in the stream is either the end of the current buffer or
    // a random nul in the file.  Disambiguate that here.
    if (CurPtr-1 != CurBuf->getBufferEnd())
      return 0;  // Just whitespace.

    // Otherwise, return end of file.
    --CurPtr;  // Another call to lex will return EOF again.
    return EOF;
  }
}

/// LexFloatLiteral: [0-9]*[.][0-9]*([eE][+-]?[0-9]*)?
///
/// The leading integral digit sequence and dot should have already been
/// consumed, some or all of the fractional digit sequence *can* have been
/// consumed.
AsmToken AsmLexer::LexFloatLiteral() {
  // Skip the fractional digit sequence.
  while (isdigit(*CurPtr))
    ++CurPtr;

  // Check for exponent; we intentionally accept a slighlty wider set of
  // literals here and rely on the upstream client to reject invalid ones (e.g.,
  // "1e+").
  if (*CurPtr == 'e' || *CurPtr == 'E') {
    ++CurPtr;
    if (*CurPtr == '-' || *CurPtr == '+')
      ++CurPtr;
    while (isdigit(*CurPtr))
      ++CurPtr;
  }

  return AsmToken(AsmToken::Real,
                  StringRef(TokStart, CurPtr - TokStart));
}

/// LexIdentifier: [a-zA-Z_.][a-zA-Z0-9_$.@]*
static bool IsIdentifierChar(char c) {
  return isalnum(c) || c == '_' || c == '$' || c == '.' || c == '@';
}
AsmToken AsmLexer::LexIdentifier() {
  // Check for floating point literals.
  if (CurPtr[-1] == '.' && isdigit(*CurPtr)) {
    // Disambiguate a .1243foo identifier from a floating literal.
    while (isdigit(*CurPtr))
      ++CurPtr;
    if (*CurPtr == 'e' || *CurPtr == 'E' || !IsIdentifierChar(*CurPtr))
      return LexFloatLiteral();
  }

  while (IsIdentifierChar(*CurPtr))
    ++CurPtr;

  // Handle . as a special case.
  if (CurPtr == TokStart+1 && TokStart[0] == '.')
    return AsmToken(AsmToken::Dot, StringRef(TokStart, 1));

  return AsmToken(AsmToken::Identifier, StringRef(TokStart, CurPtr - TokStart));
}

/// LexSlash: Slash: /
///           C-Style Comment: /* ... */
AsmToken AsmLexer::LexSlash() {
  switch (*CurPtr) {
  case '*': break; // C style comment.
  case '/': return ++CurPtr, LexLineComment();
  default:  return AsmToken(AsmToken::Slash, StringRef(CurPtr-1, 1));
  }

  // C Style comment.
  ++CurPtr;  // skip the star.
  while (1) {
    int CurChar = getNextChar();
    switch (CurChar) {
    case EOF:
      return ReturnError(TokStart, "unterminated comment");
    case '*':
      // End of the comment?
      if (CurPtr[0] != '/') break;

      ++CurPtr;   // End the */.
      return LexToken();
    }
  }
}

/// LexLineComment: Comment: #[^\n]*
///                        : //[^\n]*
AsmToken AsmLexer::LexLineComment() {
  // FIXME: This is broken if we happen to a comment at the end of a file, which
  // was .included, and which doesn't end with a newline.
  int CurChar = getNextChar();
  while (CurChar != '\n' && CurChar != '\r' && CurChar != EOF)
    CurChar = getNextChar();

  if (CurChar == EOF)
    return AsmToken(AsmToken::Eof, StringRef(CurPtr, 0));
  return AsmToken(AsmToken::EndOfStatement, StringRef(CurPtr, 0));
}

static void SkipIgnoredIntegerSuffix(const char *&CurPtr) {
  if (CurPtr[0] == 'L' && CurPtr[1] == 'L')
    CurPtr += 2;
  if (CurPtr[0] == 'U' && CurPtr[1] == 'L' && CurPtr[2] == 'L')
    CurPtr += 3;
}

/// LexDigit: First character is [0-9].
///   Local Label: [0-9][:]
///   Forward/Backward Label: [0-9][fb]
///   Binary integer: 0b[01]+
///   Octal integer: 0[0-7]+
///   Hex integer: 0x[0-9a-fA-F]+
///   Decimal integer: [1-9][0-9]*
AsmToken AsmLexer::LexDigit() {
  // Decimal integer: [1-9][0-9]*
  if (CurPtr[-1] != '0' || CurPtr[0] == '.') {
    while (isdigit(*CurPtr))
      ++CurPtr;

    // Check for floating point literals.
    if (*CurPtr == '.' || *CurPtr == 'e') {
      ++CurPtr;
      return LexFloatLiteral();
    }

    StringRef Result(TokStart, CurPtr - TokStart);

    long long Value;
    if (Result.getAsInteger(10, Value)) {
      // Allow positive values that are too large to fit into a signed 64-bit
      // integer, but that do fit in an unsigned one, we just convert them over.
      unsigned long long UValue;
      if (Result.getAsInteger(10, UValue))
        return ReturnError(TokStart, "invalid decimal number");
      Value = (long long)UValue;
    }

    // The darwin/x86 (and x86-64) assembler accepts and ignores ULL and LL
    // suffixes on integer literals.
    SkipIgnoredIntegerSuffix(CurPtr);

    return AsmToken(AsmToken::Integer, Result, Value);
  }

  if (*CurPtr == 'b') {
    ++CurPtr;
    // See if we actually have "0b" as part of something like "jmp 0b\n"
    if (!isdigit(CurPtr[0])) {
      --CurPtr;
      StringRef Result(TokStart, CurPtr - TokStart);
      return AsmToken(AsmToken::Integer, Result, 0);
    }
    const char *NumStart = CurPtr;
    while (CurPtr[0] == '0' || CurPtr[0] == '1')
      ++CurPtr;

    // Requires at least one binary digit.
    if (CurPtr == NumStart)
      return ReturnError(TokStart, "invalid binary number");

    StringRef Result(TokStart, CurPtr - TokStart);

    long long Value;
    if (Result.substr(2).getAsInteger(2, Value))
      return ReturnError(TokStart, "invalid binary number");

    // The darwin/x86 (and x86-64) assembler accepts and ignores ULL and LL
    // suffixes on integer literals.
    SkipIgnoredIntegerSuffix(CurPtr);

    return AsmToken(AsmToken::Integer, Result, Value);
  }

  if (*CurPtr == 'x') {
    ++CurPtr;
    const char *NumStart = CurPtr;
    while (isxdigit(CurPtr[0]))
      ++CurPtr;

    // Requires at least one hex digit.
    if (CurPtr == NumStart)
      return ReturnError(CurPtr-2, "invalid hexadecimal number");

    unsigned long long Result;
    if (StringRef(TokStart, CurPtr - TokStart).getAsInteger(0, Result))
      return ReturnError(TokStart, "invalid hexadecimal number");

    // The darwin/x86 (and x86-64) assembler accepts and ignores ULL and LL
    // suffixes on integer literals.
    SkipIgnoredIntegerSuffix(CurPtr);

    return AsmToken(AsmToken::Integer, StringRef(TokStart, CurPtr - TokStart),
                    (int64_t)Result);
  }

  // Must be an octal number, it starts with 0.
  while (*CurPtr >= '0' && *CurPtr <= '9')
    ++CurPtr;

  StringRef Result(TokStart, CurPtr - TokStart);
  long long Value;
  if (Result.getAsInteger(8, Value))
    return ReturnError(TokStart, "invalid octal number");

  // The darwin/x86 (and x86-64) assembler accepts and ignores ULL and LL
  // suffixes on integer literals.
  SkipIgnoredIntegerSuffix(CurPtr);

  return AsmToken(AsmToken::Integer, Result, Value);
}

/// LexSingleQuote: Integer: 'b'
AsmToken AsmLexer::LexSingleQuote() {
  int CurChar = getNextChar();

  if (CurChar == '\\')
    CurChar = getNextChar();

  if (CurChar == EOF)
    return ReturnError(TokStart, "unterminated single quote");

  CurChar = getNextChar();

  if (CurChar != '\'')
    return ReturnError(TokStart, "single quote way too long");

  // The idea here being that 'c' is basically just an integral
  // constant.
  StringRef Res = StringRef(TokStart,CurPtr - TokStart);
  long long Value;

  if (Res.startswith("\'\\")) {
    char theChar = Res[2];
    switch (theChar) {
      default: Value = theChar; break;
      case '\'': Value = '\''; break;
      case 't': Value = '\t'; break;
      case 'n': Value = '\n'; break;
      case 'b': Value = '\b'; break;
    }
  } else
    Value = TokStart[1];

  return AsmToken(AsmToken::Integer, Res, Value);
}


/// LexQuote: String: "..."
AsmToken AsmLexer::LexQuote() {
  int CurChar = getNextChar();
  // TODO: does gas allow multiline string constants?
  while (CurChar != '"') {
    if (CurChar == '\\') {
      // Allow \", etc.
      CurChar = getNextChar();
    }

    if (CurChar == EOF)
      return ReturnError(TokStart, "unterminated string constant");

    CurChar = getNextChar();
  }

  return AsmToken(AsmToken::String, StringRef(TokStart, CurPtr - TokStart));
}

StringRef AsmLexer::LexUntilEndOfStatement() {
  TokStart = CurPtr;

  while (!isAtStartOfComment(*CurPtr) &&    // Start of line comment.
         !isAtStatementSeparator(CurPtr) && // End of statement marker.
         *CurPtr != '\n' &&
         *CurPtr != '\r' &&
         (*CurPtr != 0 || CurPtr != CurBuf->getBufferEnd())) {
    ++CurPtr;
  }
  return StringRef(TokStart, CurPtr-TokStart);
}

StringRef AsmLexer::LexUntilEndOfLine() {
  TokStart = CurPtr;

  while (*CurPtr != '\n' &&
         *CurPtr != '\r' &&
         (*CurPtr != 0 || CurPtr != CurBuf->getBufferEnd())) {
    ++CurPtr;
  }
  return StringRef(TokStart, CurPtr-TokStart);
}

bool AsmLexer::isAtStartOfComment(char Char) {
  // FIXME: This won't work for multi-character comment indicators like "//".
  return Char == *MAI.getCommentString();
}

bool AsmLexer::isAtStatementSeparator(const char *Ptr) {
  return strncmp(Ptr, MAI.getSeparatorString(),
                 strlen(MAI.getSeparatorString())) == 0;
}

AsmToken AsmLexer::LexToken() {
  TokStart = CurPtr;
  // This always consumes at least one character.
  int CurChar = getNextChar();

  if (isAtStartOfComment(CurChar)) {
    // If this comment starts with a '#', then return the Hash token and let
    // the assembler parser see if it can be parsed as a cpp line filename
    // comment. We do this only if we are at the start of a line.
    if (CurChar == '#' && isAtStartOfLine)
      return AsmToken(AsmToken::Hash, StringRef(TokStart, 1));
    isAtStartOfLine = true;
    return LexLineComment();
  }
  if (isAtStatementSeparator(TokStart)) {
    CurPtr += strlen(MAI.getSeparatorString()) - 1;
    return AsmToken(AsmToken::EndOfStatement,
                    StringRef(TokStart, strlen(MAI.getSeparatorString())));
  }

  // If we're missing a newline at EOF, make sure we still get an
  // EndOfStatement token before the Eof token.
  if (CurChar == EOF && !isAtStartOfLine) {
    isAtStartOfLine = true;
    return AsmToken(AsmToken::EndOfStatement, StringRef(TokStart, 1));
  }

  isAtStartOfLine = false;
  switch (CurChar) {
  default:
    // Handle identifier: [a-zA-Z_.][a-zA-Z0-9_$.@]*
    if (isalpha(CurChar) || CurChar == '_' || CurChar == '.')
      return LexIdentifier();

    // Unknown character, emit an error.
    return ReturnError(TokStart, "invalid character in input");
  case EOF: return AsmToken(AsmToken::Eof, StringRef(TokStart, 0));
  case 0:
  case ' ':
  case '\t':
    if (SkipSpace) {
      // Ignore whitespace.
      return LexToken();
    } else {
      int len = 1;
      while (*CurPtr==' ' || *CurPtr=='\t') {
        CurPtr++;
        len++;
      }
      return AsmToken(AsmToken::Space, StringRef(TokStart, len));
    }
  case '\n': // FALL THROUGH.
  case '\r':
    isAtStartOfLine = true;
    return AsmToken(AsmToken::EndOfStatement, StringRef(TokStart, 1));
  case ':': return AsmToken(AsmToken::Colon, StringRef(TokStart, 1));
  case '+': return AsmToken(AsmToken::Plus, StringRef(TokStart, 1));
  case '-': return AsmToken(AsmToken::Minus, StringRef(TokStart, 1));
  case '~': return AsmToken(AsmToken::Tilde, StringRef(TokStart, 1));
  case '(': return AsmToken(AsmToken::LParen, StringRef(TokStart, 1));
  case ')': return AsmToken(AsmToken::RParen, StringRef(TokStart, 1));
  case '[': return AsmToken(AsmToken::LBrac, StringRef(TokStart, 1));
  case ']': return AsmToken(AsmToken::RBrac, StringRef(TokStart, 1));
  case '{': return AsmToken(AsmToken::LCurly, StringRef(TokStart, 1));
  case '}': return AsmToken(AsmToken::RCurly, StringRef(TokStart, 1));
  case '*': return AsmToken(AsmToken::Star, StringRef(TokStart, 1));
  case ',': return AsmToken(AsmToken::Comma, StringRef(TokStart, 1));
  case '$': return AsmToken(AsmToken::Dollar, StringRef(TokStart, 1));
  case '@': return AsmToken(AsmToken::At, StringRef(TokStart, 1));
  case '\\': return AsmToken(AsmToken::BackSlash, StringRef(TokStart, 1));
  case '=':
    if (*CurPtr == '=')
      return ++CurPtr, AsmToken(AsmToken::EqualEqual, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Equal, StringRef(TokStart, 1));
  case '|':
    if (*CurPtr == '|')
      return ++CurPtr, AsmToken(AsmToken::PipePipe, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Pipe, StringRef(TokStart, 1));
  case '^': return AsmToken(AsmToken::Caret, StringRef(TokStart, 1));
  case '&':
    if (*CurPtr == '&')
      return ++CurPtr, AsmToken(AsmToken::AmpAmp, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Amp, StringRef(TokStart, 1));
  case '!':
    if (*CurPtr == '=')
      return ++CurPtr, AsmToken(AsmToken::ExclaimEqual, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Exclaim, StringRef(TokStart, 1));
  case '%': return AsmToken(AsmToken::Percent, StringRef(TokStart, 1));
  case '/': return LexSlash();
  case '#': return AsmToken(AsmToken::Hash, StringRef(TokStart, 1));
  case '\'': return LexSingleQuote();
  case '"': return LexQuote();
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
    return LexDigit();
  case '<':
    switch (*CurPtr) {
    case '<': return ++CurPtr, AsmToken(AsmToken::LessLess,
                                        StringRef(TokStart, 2));
    case '=': return ++CurPtr, AsmToken(AsmToken::LessEqual,
                                        StringRef(TokStart, 2));
    case '>': return ++CurPtr, AsmToken(AsmToken::LessGreater,
                                        StringRef(TokStart, 2));
    default: return AsmToken(AsmToken::Less, StringRef(TokStart, 1));
    }
  case '>':
    switch (*CurPtr) {
    case '>': return ++CurPtr, AsmToken(AsmToken::GreaterGreater,
                                        StringRef(TokStart, 2));
    case '=': return ++CurPtr, AsmToken(AsmToken::GreaterEqual,
                                        StringRef(TokStart, 2));
    default: return AsmToken(AsmToken::Greater, StringRef(TokStart, 1));
    }

  // TODO: Quoted identifiers (objc methods etc)
  // local labels: [0-9][:]
  // Forward/backward labels: [0-9][fb]
  // Integers, fp constants, character constants.
  }
}
