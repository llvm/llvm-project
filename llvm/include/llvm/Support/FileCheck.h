//==-- llvm/Support/FileCheck.h ---------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file has some utilities to use FileCheck as an API
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILECHECK_H
#define LLVM_SUPPORT_FILECHECK_H

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include <vector>
#include <map>

namespace llvm {

/// Contains info about various FileCheck options.
struct FileCheckRequest {
  std::vector<std::string> CheckPrefixes;
  bool NoCanonicalizeWhiteSpace = false;
  std::vector<std::string> ImplicitCheckNot;
  std::vector<std::string> GlobalDefines;
  bool AllowEmptyInput = false;
  bool MatchFullLines = false;
  bool EnableVarScope = false;
  bool AllowDeprecatedDagOverlap = false;
  bool Verbose = false;
  bool VerboseVerbose = false;
};

//===----------------------------------------------------------------------===//
// Numeric expression handling code.
//===----------------------------------------------------------------------===//

/// Class representing a numeric expression.
class FileCheckNumExpr {
private:
  /// Value of the numeric expression.
  uint64_t Value;

public:
  /// Constructor for a numeric expression with a known value at parse time,
  /// e.g. the implicit numeric expression defining the @LINE numeric pseudo
  /// variable.
  explicit FileCheckNumExpr(uint64_t Value) : Value(Value) {}

  /// Return the value being matched against.
  uint64_t getValue() const { return Value; }
};

class FileCheckPatternContext;

/// Class representing a substitution to perform in the string to match.
class FileCheckPatternSubstitution {
private:
  /// Pointer to a class instance holding the table with the values of live
  /// pattern variables at the start of any given CHECK line. Used for
  /// substituting pattern variables (numeric variables have their value in the
  /// FileCheckNumExpr class instance pointed to by NumExpr).
  FileCheckPatternContext *Context;

  /// Whether this represents a numeric expression substitution.
  bool IsNumExpr;

  /// The string that needs to be substituted for something else. For a
  /// pattern variable this is its name, otherwise this is the whole numeric
  /// expression.
  StringRef FromStr;

  /// If this is a numeric expression substitution, this is the pointer to the
  /// class representing that numeric expression.
  FileCheckNumExpr *NumExpr = nullptr;

  // Index in RegExStr of where to do the substitution.
  size_t InsertIdx;

public:
  /// Constructor for a pattern variable substitution.
  FileCheckPatternSubstitution(FileCheckPatternContext *Context,
                               StringRef VarName, size_t InsertIdx)
      : Context(Context), IsNumExpr(false), FromStr(VarName),
        InsertIdx(InsertIdx) {}

  /// Constructor for a numeric expression substitution.
  FileCheckPatternSubstitution(FileCheckPatternContext *Context, StringRef Expr,
                               FileCheckNumExpr *NumExpr, size_t InsertIdx)
      : Context(Context), IsNumExpr(true), FromStr(Expr), NumExpr(NumExpr),
        InsertIdx(InsertIdx) {}

  /// Return whether this is a numeric expression substitution.
  bool isNumExpr() const { return IsNumExpr; }

  /// Return the string to be substituted.
  StringRef getFromString() const { return FromStr; }

  /// Return the index where the substitution is to be performed.
  size_t getIndex() const { return InsertIdx; }

  /// Return the result of the substitution represented by this class instance
  /// or None if substitution failed. For a numeric expression we substitute it
  /// by its value. For a pattern variable we simply replace it by the text its
  /// definition matched.
  llvm::Optional<std::string> getResult() const;

  /// Return the name of the undefined variable used in this substitution, if
  /// any, or an empty string otherwise.
  StringRef getUndefVarName() const;
};

//===----------------------------------------------------------------------===//
// Pattern handling code.
//===----------------------------------------------------------------------===//

namespace Check {

enum FileCheckKind {
  CheckNone = 0,
  CheckPlain,
  CheckNext,
  CheckSame,
  CheckNot,
  CheckDAG,
  CheckLabel,
  CheckEmpty,

  /// Indicates the pattern only matches the end of file. This is used for
  /// trailing CHECK-NOTs.
  CheckEOF,

  /// Marks when parsing found a -NOT check combined with another CHECK suffix.
  CheckBadNot,

  /// Marks when parsing found a -COUNT directive with invalid count value.
  CheckBadCount
};

class FileCheckType {
  FileCheckKind Kind;
  int Count; ///< optional Count for some checks

public:
  FileCheckType(FileCheckKind Kind = CheckNone) : Kind(Kind), Count(1) {}
  FileCheckType(const FileCheckType &) = default;

  operator FileCheckKind() const { return Kind; }

  int getCount() const { return Count; }
  FileCheckType &setCount(int C);

  std::string getDescription(StringRef Prefix) const;
};
} // namespace Check

struct FileCheckDiag;

/// Class holding the FileCheckPattern global state, shared by all patterns:
/// tables holding values of variables and whether they are defined or not at
/// any given time in the matching process.
class FileCheckPatternContext {
  friend class FileCheckPattern;

private:
  /// When matching a given pattern, this holds the value of all the FileCheck
  /// pattern variables defined in previous patterns. In a pattern, only the
  /// last definition for a given variable is recorded in this table.
  /// Back-references are used for uses after any the other definition.
  StringMap<StringRef> GlobalVariableTable;

  /// Vector holding pointers to all parsed numeric expressions. Used to
  /// automatically free the numeric expressions once they are guaranteed to no
  /// longer be used.
  std::vector<std::unique_ptr<FileCheckNumExpr>> NumExprs;

public:
  /// Return the value of pattern variable \p VarName or None if no such
  /// variable has been defined.
  llvm::Optional<StringRef> getPatternVarValue(StringRef VarName);

  /// Define pattern variables from definitions given on the command line,
  /// passed as a vector of VAR=VAL strings in \p CmdlineDefines. Report any
  /// error to \p SM and return whether an error occured.
  bool defineCmdlineVariables(std::vector<std::string> &CmdlineDefines,
                              SourceMgr &SM);

  /// Undefine local variables (variables whose name does not start with a '$'
  /// sign), i.e. remove them from GlobalVariableTable.
  void clearLocalVars();

private:
  /// Make a new numeric expression instance and register it for destruction
  /// when the context is destroyed.
  template <class... Types> FileCheckNumExpr *makeNumExpr(Types... Args);
};

class FileCheckPattern {
  SMLoc PatternLoc;

  /// A fixed string to match as the pattern or empty if this pattern requires
  /// a regex match.
  StringRef FixedStr;

  /// A regex string to match as the pattern or empty if this pattern requires
  /// a fixed string to match.
  std::string RegExStr;

  /// Entries in this vector represent uses of a pattern variable or a numeric
  /// expression in the pattern that need to be substituted in the regexp
  /// pattern at match time, e.g. "foo[[bar]]baz[[#@LINE+1]]". In this case,
  /// the RegExStr will contain "foobaz" and we'll get two entries in this
  /// vector that tells us to insert the value of pattern variable "bar" at
  /// offset 3 and the value of numeric expression "@LINE+1" at offset 6. Uses
  /// are represented by a FileCheckPatternSubstitution class to abstract
  /// whether it is a pattern variable or a numeric expression.
  std::vector<FileCheckPatternSubstitution> Substitutions;

  /// Maps names of pattern variables defined in a pattern to the parenthesized
  /// capture numbers of their last definition.
  ///
  /// E.g. for the pattern "foo[[bar:.*]]baz[[bar]]quux[[bar:.*]]",
  /// VariableDefs will map "bar" to 2 corresponding to the second definition
  /// of "bar".
  ///
  /// Note: uses std::map rather than StringMap to be able to get the key when
  /// iterating over values.
  std::map<StringRef, unsigned> VariableDefs;

  /// Pointer to the class instance shared by all patterns holding a table with
  /// the values of live variables at the start of any given CHECK line.
  FileCheckPatternContext *Context;

  Check::FileCheckType CheckTy;

  /// Contains the number of line this pattern is in.
  unsigned LineNumber;

public:
  explicit FileCheckPattern(Check::FileCheckType Ty,
                            FileCheckPatternContext *Context)
      : Context(Context), CheckTy(Ty) {}

  /// Returns the location in source code.
  SMLoc getLoc() const { return PatternLoc; }

  /// Returns the pointer to the global state for all patterns in this
  /// FileCheck instance.
  FileCheckPatternContext *getContext() const { return Context; }

  /// Return whether \p is a valid first character for a variable name.
  static bool isValidVarNameStart(char C);
  /// Verify that the string at the start of \p Str is a well formed variable.
  /// Return false if it is and set \p IsPseudo to indicate if it is a pseudo
  /// variable and \p TrailIdx to the position of the last character that is
  /// part of the variable name. Otherwise, only return true.
  static bool parseVariable(StringRef Str, bool &IsPseudo, unsigned &TrailIdx);
  /// Parse a numeric expression involving pseudo variable \p Name with the
  /// string corresponding to the operation being performed in \p Trailer.
  /// Return the class representing the numeric expression or nullptr if
  /// parsing fails in which case errors are reported on \p SM.
  FileCheckNumExpr *parseNumericExpression(StringRef Name, StringRef Trailer,
                                           const SourceMgr &SM) const;
  bool ParsePattern(StringRef PatternStr, StringRef Prefix, SourceMgr &SM,
                    unsigned LineNumber, const FileCheckRequest &Req);
  size_t match(StringRef Buffer, size_t &MatchLen) const;
  /// Print value of successful substitutions or name of undefined pattern or
  /// numeric variables preventing such a successful substitution.
  void printSubstitutions(const SourceMgr &SM, StringRef Buffer,
                          SMRange MatchRange = None) const;
  void printFuzzyMatch(const SourceMgr &SM, StringRef Buffer,
                       std::vector<FileCheckDiag> *Diags) const;

  bool hasVariable() const {
    return !(Substitutions.empty() && VariableDefs.empty());
  }

  Check::FileCheckType getCheckTy() const { return CheckTy; }

  int getCount() const { return CheckTy.getCount(); }

private:
  bool AddRegExToRegEx(StringRef RS, unsigned &CurParen, SourceMgr &SM);
  void AddBackrefToRegEx(unsigned BackrefNum);
  unsigned computeMatchDistance(StringRef Buffer) const;
  size_t FindRegexVarEnd(StringRef Str, SourceMgr &SM);
};

//===----------------------------------------------------------------------===//
/// Summary of a FileCheck diagnostic.
//===----------------------------------------------------------------------===//

struct FileCheckDiag {
  /// What is the FileCheck directive for this diagnostic?
  Check::FileCheckType CheckTy;
  /// Where is the FileCheck directive for this diagnostic?
  unsigned CheckLine, CheckCol;
  /// What type of match result does this diagnostic describe?
  ///
  /// A directive's supplied pattern is said to be either expected or excluded
  /// depending on whether the pattern must have or must not have a match in
  /// order for the directive to succeed.  For example, a CHECK directive's
  /// pattern is expected, and a CHECK-NOT directive's pattern is excluded.
  /// All match result types whose names end with "Excluded" are for excluded
  /// patterns, and all others are for expected patterns.
  ///
  /// There might be more than one match result for a single pattern.  For
  /// example, there might be several discarded matches
  /// (MatchFoundButDiscarded) before either a good match
  /// (MatchFoundAndExpected) or a failure to match (MatchNoneButExpected),
  /// and there might be a fuzzy match (MatchFuzzy) after the latter.
  enum MatchType {
    /// Indicates a good match for an expected pattern.
    MatchFoundAndExpected,
    /// Indicates a match for an excluded pattern.
    MatchFoundButExcluded,
    /// Indicates a match for an expected pattern, but the match is on the
    /// wrong line.
    MatchFoundButWrongLine,
    /// Indicates a discarded match for an expected pattern.
    MatchFoundButDiscarded,
    /// Indicates no match for an excluded pattern.
    MatchNoneAndExcluded,
    /// Indicates no match for an expected pattern, but this might follow good
    /// matches when multiple matches are expected for the pattern, or it might
    /// follow discarded matches for the pattern.
    MatchNoneButExpected,
    /// Indicates a fuzzy match that serves as a suggestion for the next
    /// intended match for an expected pattern with too few or no good matches.
    MatchFuzzy,
  } MatchTy;
  /// The search range if MatchTy is MatchNoneAndExcluded or
  /// MatchNoneButExpected, or the match range otherwise.
  unsigned InputStartLine;
  unsigned InputStartCol;
  unsigned InputEndLine;
  unsigned InputEndCol;
  FileCheckDiag(const SourceMgr &SM, const Check::FileCheckType &CheckTy,
                SMLoc CheckLoc, MatchType MatchTy, SMRange InputRange);
};

//===----------------------------------------------------------------------===//
// Check Strings.
//===----------------------------------------------------------------------===//

/// A check that we found in the input file.
struct FileCheckString {
  /// The pattern to match.
  FileCheckPattern Pat;

  /// Which prefix name this check matched.
  StringRef Prefix;

  /// The location in the match file that the check string was specified.
  SMLoc Loc;

  /// All of the strings that are disallowed from occurring between this match
  /// string and the previous one (or start of file).
  std::vector<FileCheckPattern> DagNotStrings;

  FileCheckString(const FileCheckPattern &P, StringRef S, SMLoc L)
      : Pat(P), Prefix(S), Loc(L) {}

  size_t Check(const SourceMgr &SM, StringRef Buffer, bool IsLabelScanMode,
               size_t &MatchLen, FileCheckRequest &Req,
               std::vector<FileCheckDiag> *Diags) const;

  bool CheckNext(const SourceMgr &SM, StringRef Buffer) const;
  bool CheckSame(const SourceMgr &SM, StringRef Buffer) const;
  bool CheckNot(const SourceMgr &SM, StringRef Buffer,
                const std::vector<const FileCheckPattern *> &NotStrings,
                const FileCheckRequest &Req,
                std::vector<FileCheckDiag> *Diags) const;
  size_t CheckDag(const SourceMgr &SM, StringRef Buffer,
                  std::vector<const FileCheckPattern *> &NotStrings,
                  const FileCheckRequest &Req,
                  std::vector<FileCheckDiag> *Diags) const;
};

/// FileCheck class takes the request and exposes various methods that
/// use information from the request.
class FileCheck {
  FileCheckRequest Req;
  FileCheckPatternContext PatternContext;

public:
  FileCheck(FileCheckRequest Req) : Req(Req) {}

  // Combines the check prefixes into a single regex so that we can efficiently
  // scan for any of the set.
  //
  // The semantics are that the longest-match wins which matches our regex
  // library.
  Regex buildCheckPrefixRegex();

  /// Read the check file, which specifies the sequence of expected strings.
  ///
  /// The strings are added to the CheckStrings vector. Returns true in case of
  /// an error, false otherwise.
  bool ReadCheckFile(SourceMgr &SM, StringRef Buffer, Regex &PrefixRE,
                     std::vector<FileCheckString> &CheckStrings);

  bool ValidateCheckPrefixes();

  /// Canonicalize whitespaces in the file. Line endings are replaced with
  /// UNIX-style '\n'.
  StringRef CanonicalizeFile(MemoryBuffer &MB,
                             SmallVectorImpl<char> &OutputBuffer);

  /// Check the input to FileCheck provided in the \p Buffer against the \p
  /// CheckStrings read from the check file.
  ///
  /// Returns false if the input fails to satisfy the checks.
  bool CheckInput(SourceMgr &SM, StringRef Buffer,
                  ArrayRef<FileCheckString> CheckStrings,
                  std::vector<FileCheckDiag> *Diags = nullptr);
};
} // namespace llvm
#endif
