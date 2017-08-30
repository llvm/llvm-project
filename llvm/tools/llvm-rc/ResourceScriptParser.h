//===-- ResourceScriptParser.h ----------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This defines the RC scripts parser. It takes a sequence of RC tokens
// and then provides the method to parse the resources one by one.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMRC_RESOURCESCRIPTPARSER_H
#define LLVM_TOOLS_LLVMRC_RESOURCESCRIPTPARSER_H

#include "ResourceScriptStmt.h"
#include "ResourceScriptToken.h"

#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>
#include <vector>

namespace llvm {
namespace rc {

class RCParser {
public:
  using LocIter = std::vector<RCToken>::iterator;
  using ParseType = Expected<std::unique_ptr<RCResource>>;
  using ParseOptionType = Expected<std::unique_ptr<OptionalStmt>>;

  // Class describing a single failure of parser.
  class ParserError : public ErrorInfo<ParserError> {
  public:
    ParserError(Twine Expected, const LocIter CurLoc, const LocIter End);

    void log(raw_ostream &OS) const override { OS << CurMessage; }
    std::error_code convertToErrorCode() const override {
      return std::make_error_code(std::errc::invalid_argument);
    }
    const std::string &getMessage() const { return CurMessage; }

    static char ID; // Keep llvm::Error happy.

  private:
    std::string CurMessage;
    LocIter ErrorLoc, FileEnd;
  };

  RCParser(const std::vector<RCToken> &TokenList);
  RCParser(std::vector<RCToken> &&TokenList);

  // Reads and returns a single resource definition, or error message if any
  // occurred.
  ParseType parseSingleResource();

  bool isEof() const;

private:
  using Kind = RCToken::Kind;

  // Checks if the current parser state points to the token of type TokenKind.
  bool isNextTokenKind(Kind TokenKind) const;

  // These methods assume that the parser is not in EOF state.

  // Take a look at the current token. Do not fetch it.
  const RCToken &look() const;
  // Read the current token and advance the state by one token.
  const RCToken &read();
  // Advance the state by one token, discarding the current token.
  void consume();

  // The following methods try to read a single token, check if it has the
  // correct type and then parse it.
  Expected<uint32_t> readInt();            // Parse an integer.
  Expected<StringRef> readString();        // Parse a string.
  Expected<StringRef> readIdentifier();    // Parse an identifier.
  Expected<IntOrString> readIntOrString(); // Parse an integer or a string.
  Expected<IntOrString> readTypeOrName();  // Parse an integer or an identifier.

  // Advance the state by one, discarding the current token.
  // If the discarded token had an incorrect type, fail.
  Error consumeType(Kind TokenKind);

  // Check the current token type. If it's TokenKind, discard it.
  // Return true if the parser consumed this token successfully.
  bool consumeOptionalType(Kind TokenKind);

  // Read at least MinCount, and at most MaxCount integers separated by
  // commas. The parser stops reading after fetching MaxCount integers
  // or after an error occurs. Whenever the parser reads a comma, it
  // expects an integer to follow.
  Expected<SmallVector<uint32_t, 8>> readIntsWithCommas(size_t MinCount,
                                                        size_t MaxCount);

  // Read an unknown number of flags preceded by commas. Each correct flag
  // has an entry in FlagDesc array of length NumFlags. In case i-th
  // flag (0-based) has been read, the i-th bit of the result is set.
  // As long as parser has a comma to read, it expects to be fed with
  // a correct flag afterwards.
  Expected<uint32_t> parseFlags(ArrayRef<StringRef> FlagDesc);

  // Reads a set of optional statements. These can change the behavior of
  // a number of resource types (e.g. STRINGTABLE, MENU or DIALOG) if provided
  // before the main block with the contents of the resource.
  // Usually, resources use a basic set of optional statements:
  //    CHARACTERISTICS, LANGUAGE, VERSION
  // However, DIALOG and DIALOGEX extend this list by the following items:
  //    CAPTION, CLASS, EXSTYLE, FONT, MENU, STYLE
  // UseExtendedStatements flag (off by default) allows the parser to read
  // the additional types of statements.
  //
  // Ref (to the list of all optional statements):
  //    msdn.microsoft.com/en-us/library/windows/desktop/aa381002(v=vs.85).aspx
  Expected<OptionalStmtList>
  parseOptionalStatements(bool UseExtendedStatements = false);

  // Read a single optional statement.
  Expected<std::unique_ptr<OptionalStmt>>
  parseSingleOptionalStatement(bool UseExtendedStatements = false);

  // Top-level resource parsers.
  ParseType parseLanguageResource();
  ParseType parseAcceleratorsResource();
  ParseType parseCursorResource();
  ParseType parseDialogResource(bool IsExtended);
  ParseType parseIconResource();
  ParseType parseHTMLResource();
  ParseType parseMenuResource();
  ParseType parseStringTableResource();

  // Helper DIALOG parser - a single control.
  Expected<Control> parseControl();

  // Helper MENU parser.
  Expected<MenuDefinitionList> parseMenuItemsList();

  // Optional statement parsers.
  ParseOptionType parseLanguageStmt();
  ParseOptionType parseCharacteristicsStmt();
  ParseOptionType parseVersionStmt();
  ParseOptionType parseCaptionStmt();
  ParseOptionType parseFontStmt();
  ParseOptionType parseStyleStmt();

  // Raises an error. If IsAlreadyRead = false (default), this complains about
  // the token that couldn't be parsed. If the flag is on, this complains about
  // the correctly read token that makes no sense (that is, the current parser
  // state is beyond the erroneous token.)
  Error getExpectedError(const Twine Message, bool IsAlreadyRead = false);

  std::vector<RCToken> Tokens;
  LocIter CurLoc;
  const LocIter End;
};

} // namespace rc
} // namespace llvm

#endif
