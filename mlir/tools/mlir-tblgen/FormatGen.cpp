//===- FormatGen.cpp - Utilities for custom assembly formats ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatGen.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Error.h"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// FormatToken
//===----------------------------------------------------------------------===//

SMLoc FormatToken::getLoc() const {
  return SMLoc::getFromPointer(spelling.data());
}

//===----------------------------------------------------------------------===//
// FormatLexer
//===----------------------------------------------------------------------===//

FormatLexer::FormatLexer(llvm::SourceMgr &mgr, SMLoc loc)
    : mgr(mgr), loc(loc),
      curBuffer(mgr.getMemoryBuffer(mgr.getMainFileID())->getBuffer()),
      curPtr(curBuffer.begin()) {}

FormatToken FormatLexer::emitError(SMLoc loc, const Twine &msg) {
  mgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, msg);
  llvm::SrcMgr.PrintMessage(this->loc, llvm::SourceMgr::DK_Note,
                            "in custom assembly format for this operation");
  return formToken(FormatToken::error, loc.getPointer());
}

FormatToken FormatLexer::emitError(const char *loc, const Twine &msg) {
  return emitError(SMLoc::getFromPointer(loc), msg);
}

FormatToken FormatLexer::emitErrorAndNote(SMLoc loc, const Twine &msg,
                                          const Twine &note) {
  mgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, msg);
  llvm::SrcMgr.PrintMessage(this->loc, llvm::SourceMgr::DK_Note,
                            "in custom assembly format for this operation");
  mgr.PrintMessage(loc, llvm::SourceMgr::DK_Note, note);
  return formToken(FormatToken::error, loc.getPointer());
}

int FormatLexer::getNextChar() {
  char curChar = *curPtr++;
  switch (curChar) {
  default:
    return (unsigned char)curChar;
  case 0: {
    // A nul character in the stream is either the end of the current buffer or
    // a random nul in the file. Disambiguate that here.
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

FormatToken FormatLexer::lexToken() {
  const char *tokStart = curPtr;

  // This always consumes at least one character.
  int curChar = getNextChar();
  switch (curChar) {
  default:
    // Handle identifiers: [a-zA-Z_]
    if (isalpha(curChar) || curChar == '_')
      return lexIdentifier(tokStart);

    // Unknown character, emit an error.
    return emitError(tokStart, "unexpected character");
  case EOF:
    // Return EOF denoting the end of lexing.
    return formToken(FormatToken::eof, tokStart);

  // Lex punctuation.
  case '^':
    return formToken(FormatToken::caret, tokStart);
  case ':':
    return formToken(FormatToken::colon, tokStart);
  case ',':
    return formToken(FormatToken::comma, tokStart);
  case '=':
    return formToken(FormatToken::equal, tokStart);
  case '<':
    return formToken(FormatToken::less, tokStart);
  case '>':
    return formToken(FormatToken::greater, tokStart);
  case '?':
    return formToken(FormatToken::question, tokStart);
  case '(':
    return formToken(FormatToken::l_paren, tokStart);
  case ')':
    return formToken(FormatToken::r_paren, tokStart);
  case '*':
    return formToken(FormatToken::star, tokStart);

  // Ignore whitespace characters.
  case 0:
  case ' ':
  case '\t':
  case '\n':
    return lexToken();

  case '`':
    return lexLiteral(tokStart);
  case '$':
    return lexVariable(tokStart);
  }
}

FormatToken FormatLexer::lexLiteral(const char *tokStart) {
  assert(curPtr[-1] == '`');

  // Lex a literal surrounded by ``.
  while (const char curChar = *curPtr++) {
    if (curChar == '`')
      return formToken(FormatToken::literal, tokStart);
  }
  return emitError(curPtr - 1, "unexpected end of file in literal");
}

FormatToken FormatLexer::lexVariable(const char *tokStart) {
  if (!isalpha(curPtr[0]) && curPtr[0] != '_')
    return emitError(curPtr - 1, "expected variable name");

  // Otherwise, consume the rest of the characters.
  while (isalnum(*curPtr) || *curPtr == '_')
    ++curPtr;
  return formToken(FormatToken::variable, tokStart);
}

FormatToken FormatLexer::lexIdentifier(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_\-]*
  while (isalnum(*curPtr) || *curPtr == '_' || *curPtr == '-')
    ++curPtr;

  // Check to see if this identifier is a keyword.
  StringRef str(tokStart, curPtr - tokStart);
  auto kind =
      StringSwitch<FormatToken::Kind>(str)
          .Case("attr-dict", FormatToken::kw_attr_dict)
          .Case("attr-dict-with-keyword", FormatToken::kw_attr_dict_w_keyword)
          .Case("custom", FormatToken::kw_custom)
          .Case("functional-type", FormatToken::kw_functional_type)
          .Case("operands", FormatToken::kw_operands)
          .Case("params", FormatToken::kw_params)
          .Case("ref", FormatToken::kw_ref)
          .Case("regions", FormatToken::kw_regions)
          .Case("results", FormatToken::kw_results)
          .Case("struct", FormatToken::kw_struct)
          .Case("successors", FormatToken::kw_successors)
          .Case("type", FormatToken::kw_type)
          .Case("qualified", FormatToken::kw_qualified)
          .Default(FormatToken::identifier);
  return FormatToken(kind, str);
}

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

bool mlir::tblgen::shouldEmitSpaceBefore(StringRef value,
                                         bool lastWasPunctuation) {
  if (value.size() != 1 && value != "->")
    return true;
  if (lastWasPunctuation)
    return !StringRef(">)}],").contains(value.front());
  return !StringRef("<>(){}[],").contains(value.front());
}

bool mlir::tblgen::canFormatStringAsKeyword(
    StringRef value, function_ref<void(Twine)> emitError) {
  if (!isalpha(value.front()) && value.front() != '_') {
    if (emitError)
      emitError("valid keyword starts with a letter or '_'");
    return false;
  }
  if (!llvm::all_of(value.drop_front(), [](char c) {
        return isalnum(c) || c == '_' || c == '$' || c == '.';
      })) {
    if (emitError)
      emitError(
          "keywords should contain only alphanum, '_', '$', or '.' characters");
    return false;
  }
  return true;
}

bool mlir::tblgen::isValidLiteral(StringRef value,
                                  function_ref<void(Twine)> emitError) {
  if (value.empty()) {
    if (emitError)
      emitError("literal can't be empty");
    return false;
  }
  char front = value.front();

  // If there is only one character, this must either be punctuation or a
  // single character bare identifier.
  if (value.size() == 1) {
    StringRef bare = "_:,=<>()[]{}?+*";
    if (isalpha(front) || bare.contains(front))
      return true;
    if (emitError)
      emitError("single character literal must be a letter or one of '" + bare +
                "'");
    return false;
  }
  // Check the punctuation that are larger than a single character.
  if (value == "->")
    return true;

  // Otherwise, this must be an identifier.
  return canFormatStringAsKeyword(value, emitError);
}

//===----------------------------------------------------------------------===//
// Commandline Options
//===----------------------------------------------------------------------===//

llvm::cl::opt<bool> mlir::tblgen::formatErrorIsFatal(
    "asmformat-error-is-fatal",
    llvm::cl::desc("Emit a fatal error if format parsing fails"),
    llvm::cl::init(true));
