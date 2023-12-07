//===- Lexer.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Lexer.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/PDLL/AST/Diagnostic.h"
#include "mlir/Tools/PDLL/Parser/CodeComplete.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::pdll;

//===----------------------------------------------------------------------===//
// Token
//===----------------------------------------------------------------------===//

std::string Token::getStringValue() const {
  assert(getKind() == string || getKind() == string_block ||
         getKind() == code_complete_string);

  // Start by dropping the quotes.
  StringRef bytes = getSpelling();
  if (is(string))
    bytes = bytes.drop_front().drop_back();
  else if (is(string_block))
    bytes = bytes.drop_front(2).drop_back(2);

  std::string result;
  result.reserve(bytes.size());
  for (unsigned i = 0, e = bytes.size(); i != e;) {
    auto c = bytes[i++];
    if (c != '\\') {
      result.push_back(c);
      continue;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c1 = bytes[i++];
    switch (c1) {
    case '"':
    case '\\':
      result.push_back(c1);
      continue;
    case 'n':
      result.push_back('\n');
      continue;
    case 't':
      result.push_back('\t');
      continue;
    default:
      break;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c2 = bytes[i++];

    assert(llvm::isHexDigit(c1) && llvm::isHexDigit(c2) && "invalid escape");
    result.push_back((llvm::hexDigitValue(c1) << 4) | llvm::hexDigitValue(c2));
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

Lexer::Lexer(llvm::SourceMgr &mgr, ast::DiagnosticEngine &diagEngine,
             CodeCompleteContext *codeCompleteContext)
    : srcMgr(mgr), diagEngine(diagEngine), addedHandlerToDiagEngine(false),
      codeCompletionLocation(nullptr) {
  curBufferID = mgr.getMainFileID();
  curBuffer = srcMgr.getMemoryBuffer(curBufferID)->getBuffer();
  curPtr = curBuffer.begin();

  // Set the code completion location if necessary.
  if (codeCompleteContext) {
    codeCompletionLocation =
        codeCompleteContext->getCodeCompleteLoc().getPointer();
  }

  // If the diag engine has no handler, add a default that emits to the
  // SourceMgr.
  if (!diagEngine.getHandlerFn()) {
    diagEngine.setHandlerFn([&](const ast::Diagnostic &diag) {
      srcMgr.PrintMessage(diag.getLocation().Start, diag.getSeverity(),
                          diag.getMessage());
      for (const ast::Diagnostic &note : diag.getNotes())
        srcMgr.PrintMessage(note.getLocation().Start, note.getSeverity(),
                            note.getMessage());
    });
    addedHandlerToDiagEngine = true;
  }
}

Lexer::~Lexer() {
  if (addedHandlerToDiagEngine)
    diagEngine.setHandlerFn(nullptr);
}

LogicalResult Lexer::pushInclude(StringRef filename, SMRange includeLoc) {
  std::string includedFile;
  int bufferID =
      srcMgr.AddIncludeFile(filename.str(), includeLoc.End, includedFile);
  if (!bufferID)
    return failure();

  curBufferID = bufferID;
  curBuffer = srcMgr.getMemoryBuffer(curBufferID)->getBuffer();
  curPtr = curBuffer.begin();
  return success();
}

Token Lexer::emitError(SMRange loc, const Twine &msg) {
  diagEngine.emitError(loc, msg);
  return formToken(Token::error, loc.Start.getPointer());
}
Token Lexer::emitErrorAndNote(SMRange loc, const Twine &msg, SMRange noteLoc,
                              const Twine &note) {
  diagEngine.emitError(loc, msg)->attachNote(note, noteLoc);
  return formToken(Token::error, loc.Start.getPointer());
}
Token Lexer::emitError(const char *loc, const Twine &msg) {
  return emitError(
      SMRange(SMLoc::getFromPointer(loc), SMLoc::getFromPointer(loc + 1)), msg);
}

int Lexer::getNextChar() {
  char curChar = *curPtr++;
  switch (curChar) {
  default:
    return static_cast<unsigned char>(curChar);
  case 0: {
    // A nul character in the stream is either the end of the current buffer
    // or a random nul in the file. Disambiguate that here.
    if (curPtr - 1 != curBuffer.end())
      return 0;

    // Otherwise, return end of file.
    --curPtr;
    return EOF;
  }
  case '\n':
  case '\r':
    // Handle the newline character by ignoring it and incrementing the line
    // count. However, be careful about 'dos style' files with \n\r in them.
    // Only treat a \n\r or \r\n as a single line.
    if ((*curPtr == '\n' || (*curPtr == '\r')) && *curPtr != curChar)
      ++curPtr;
    return '\n';
  }
}

Token Lexer::lexToken() {
  while (true) {
    const char *tokStart = curPtr;

    // Check to see if this token is at the code completion location.
    if (tokStart == codeCompletionLocation)
      return formToken(Token::code_complete, tokStart);

    // This always consumes at least one character.
    int curChar = getNextChar();
    switch (curChar) {
    default:
      // Handle identifiers: [a-zA-Z_]
      if (isalpha(curChar) || curChar == '_')
        return lexIdentifier(tokStart);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");
    case EOF: {
      // Return EOF denoting the end of lexing.
      Token eof = formToken(Token::eof, tokStart);

      // Check to see if we are in an included file.
      SMLoc parentIncludeLoc = srcMgr.getParentIncludeLoc(curBufferID);
      if (parentIncludeLoc.isValid()) {
        curBufferID = srcMgr.FindBufferContainingLoc(parentIncludeLoc);
        curBuffer = srcMgr.getMemoryBuffer(curBufferID)->getBuffer();
        curPtr = parentIncludeLoc.getPointer();
      }

      return eof;
    }

    // Lex punctuation.
    case '-':
      if (*curPtr == '>') {
        ++curPtr;
        return formToken(Token::arrow, tokStart);
      }
      return emitError(tokStart, "unexpected character");
    case ':':
      return formToken(Token::colon, tokStart);
    case ',':
      return formToken(Token::comma, tokStart);
    case '.':
      return formToken(Token::dot, tokStart);
    case '=':
      if (*curPtr == '>') {
        ++curPtr;
        return formToken(Token::equal_arrow, tokStart);
      }
      return formToken(Token::equal, tokStart);
    case ';':
      return formToken(Token::semicolon, tokStart);
    case '[':
      if (*curPtr == '{') {
        ++curPtr;
        return lexString(tokStart, /*isStringBlock=*/true);
      }
      return formToken(Token::l_square, tokStart);
    case ']':
      return formToken(Token::r_square, tokStart);

    case '<':
      return formToken(Token::less, tokStart);
    case '>':
      return formToken(Token::greater, tokStart);
    case '{':
      return formToken(Token::l_brace, tokStart);
    case '}':
      return formToken(Token::r_brace, tokStart);
    case '(':
      return formToken(Token::l_paren, tokStart);
    case ')':
      return formToken(Token::r_paren, tokStart);
    case '/':
      if (*curPtr == '/') {
        lexComment();
        continue;
      }
      return emitError(tokStart, "unexpected character");

    // Ignore whitespace characters.
    case 0:
    case ' ':
    case '\t':
    case '\n':
      return lexToken();

    case '#':
      return lexDirective(tokStart);
    case '"':
      return lexString(tokStart, /*isStringBlock=*/false);

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
      return lexNumber(tokStart);
    }
  }
}

/// Skip a comment line, starting with a '//'.
void Lexer::lexComment() {
  // Advance over the second '/' in a '//' comment.
  assert(*curPtr == '/');
  ++curPtr;

  while (true) {
    switch (*curPtr++) {
    case '\n':
    case '\r':
      // Newline is end of comment.
      return;
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        return;
      }
      [[fallthrough]];
    default:
      // Skip over other characters.
      break;
    }
  }
}

Token Lexer::lexDirective(const char *tokStart) {
  // Match the rest with an identifier regex: [0-9a-zA-Z_]*
  while (isalnum(*curPtr) || *curPtr == '_')
    ++curPtr;

  StringRef str(tokStart, curPtr - tokStart);
  return Token(Token::directive, str);
}

Token Lexer::lexIdentifier(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_]*
  while (isalnum(*curPtr) || *curPtr == '_')
    ++curPtr;

  // Check to see if this identifier is a keyword.
  StringRef str(tokStart, curPtr - tokStart);
  Token::Kind kind = StringSwitch<Token::Kind>(str)
                         .Case("attr", Token::kw_attr)
                         .Case("Attr", Token::kw_Attr)
                         .Case("erase", Token::kw_erase)
                         .Case("let", Token::kw_let)
                         .Case("Constraint", Token::kw_Constraint)
                         .Case("not", Token::kw_not)
                         .Case("op", Token::kw_op)
                         .Case("Op", Token::kw_Op)
                         .Case("OpName", Token::kw_OpName)
                         .Case("Pattern", Token::kw_Pattern)
                         .Case("replace", Token::kw_replace)
                         .Case("return", Token::kw_return)
                         .Case("rewrite", Token::kw_rewrite)
                         .Case("Rewrite", Token::kw_Rewrite)
                         .Case("type", Token::kw_type)
                         .Case("Type", Token::kw_Type)
                         .Case("TypeRange", Token::kw_TypeRange)
                         .Case("Value", Token::kw_Value)
                         .Case("ValueRange", Token::kw_ValueRange)
                         .Case("with", Token::kw_with)
                         .Case("_", Token::underscore)
                         .Default(Token::identifier);
  return Token(kind, str);
}

Token Lexer::lexNumber(const char *tokStart) {
  assert(isdigit(curPtr[-1]));

  // Handle the normal decimal case.
  while (isdigit(*curPtr))
    ++curPtr;

  return formToken(Token::integer, tokStart);
}

Token Lexer::lexString(const char *tokStart, bool isStringBlock) {
  while (true) {
    // Check to see if there is a code completion location within the string. In
    // these cases we generate a completion location and place the currently
    // lexed string within the token (without the quotes). This allows for the
    // parser to use the partially lexed string when computing the completion
    // results.
    if (curPtr == codeCompletionLocation) {
      return formToken(Token::code_complete_string,
                       tokStart + (isStringBlock ? 2 : 1));
    }

    switch (*curPtr++) {
    case '"':
      // If this is a string block, we only end the string when we encounter a
      // `}]`.
      if (!isStringBlock)
        return formToken(Token::string, tokStart);
      continue;
    case '}':
      // If this is a string block, we only end the string when we encounter a
      // `}]`.
      if (!isStringBlock || *curPtr != ']')
        continue;
      ++curPtr;
      return formToken(Token::string_block, tokStart);
    case 0: {
      // If this is a random nul character in the middle of a string, just
      // include it. If it is the end of file, then it is an error.
      if (curPtr - 1 != curBuffer.end())
        continue;
      --curPtr;

      StringRef expectedEndStr = isStringBlock ? "}]" : "\"";
      return emitError(curPtr - 1,
                       "expected '" + expectedEndStr + "' in string literal");
    }

    case '\n':
    case '\v':
    case '\f':
      // String blocks allow multiple lines.
      if (!isStringBlock)
        return emitError(curPtr - 1, "expected '\"' in string literal");
      continue;

    case '\\':
      // Handle explicitly a few escapes.
      if (*curPtr == '"' || *curPtr == '\\' || *curPtr == 'n' ||
          *curPtr == 't') {
        ++curPtr;
      } else if (llvm::isHexDigit(*curPtr) && llvm::isHexDigit(curPtr[1])) {
        // Support \xx for two hex digits.
        curPtr += 2;
      } else {
        return emitError(curPtr - 1, "unknown escape in string literal");
      }
      continue;

    default:
      continue;
    }
  }
}
