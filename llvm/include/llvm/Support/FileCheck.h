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

/// Class representing a numeric variable with a given value in a numeric
/// expression.
class FileCheckNumericVariable {
private:
  /// Name of the numeric variable.
  StringRef Name;

  /// Value of numeric variable, if defined, or None otherwise.
  llvm::Optional<uint64_t> Value;

public:
  /// Constructor for numeric variable \p Name with a known \p Value at parse
  /// time (e.g. the @LINE numeric variable).
  FileCheckNumericVariable(StringRef Name, uint64_t Value)
      : Name(Name), Value(Value) {}

  /// \returns name of that numeric variable.
  StringRef getName() const { return Name; }

  /// \returns value of this numeric variable.
  llvm::Optional<uint64_t> getValue() const { return Value; }

  /// Sets value of this numeric variable if not defined. \returns whether the
  /// variable was already defined.
  bool setValue(uint64_t Value);

  /// Clears value of this numeric variable. \returns whether the variable was
  /// already undefined.
  bool clearValue();
};

/// Type of functions evaluating a given binary operation.
using binop_eval_t = uint64_t (*)(uint64_t, uint64_t);

/// Class representing a numeric expression consisting of either a single
/// numeric variable or a binary operation between a numeric variable and an
/// immediate.
class FileCheckNumExpr {
private:
  /// Left operand.
  FileCheckNumericVariable *LeftOp;

  /// Right operand.
  uint64_t RightOp;

  /// Pointer to function that can evaluate this binary operation.
  binop_eval_t EvalBinop;

public:
  FileCheckNumExpr(binop_eval_t EvalBinop,
                   FileCheckNumericVariable *OperandLeft, uint64_t OperandRight)
      : LeftOp(OperandLeft), RightOp(OperandRight), EvalBinop(EvalBinop) {}

  /// Evaluates the value of this numeric expression, using EvalBinop to
  /// perform the binary operation it consists of. \returns None if the numeric
  /// variable used is undefined, or the expression value otherwise.
  llvm::Optional<uint64_t> eval() const;

  /// \returns the name of the undefined variable used in this expression if
  /// any or an empty string otherwise.
  StringRef getUndefVarName() const;
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

  /// \returns whether this is a numeric expression substitution.
  bool isNumExpr() const { return IsNumExpr; }

  /// \returns the string to be substituted.
  StringRef getFromString() const { return FromStr; }

  /// \returns the index where the substitution is to be performed.
  size_t getIndex() const { return InsertIdx; }

  /// \returns the result of the substitution represented by this class
  /// instance or None if substitution failed. Numeric expressions are
  /// substituted by their values. Pattern variables are simply replaced by the
  /// text their definition matched.
  llvm::Optional<std::string> getResult() const;

  /// \returns the name of the undefined variable used in this substitution, if
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

  // \returns a description of \p Prefix.
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

  /// Map of all pattern variables defined so far. Used at parse time to detect
  /// a name conflict between a numeric variable and a pattern variable when
  /// the former is defined on a later line than the latter.
  StringMap<bool> DefinedVariableTable;

  /// When matching a given pattern, this holds the pointers to the classes
  /// representing the last definitions of numeric variables defined in
  /// previous patterns. Earlier definition of the variables, if any, have
  /// their own class instance not referenced by this table.
  StringMap<FileCheckNumericVariable *> GlobalNumericVariableTable;

  /// Vector holding pointers to all parsed numeric expressions. Used to
  /// automatically free the numeric expressions once they are guaranteed to no
  /// longer be used.
  std::vector<std::unique_ptr<FileCheckNumExpr>> NumExprs;

  /// Vector holding pointers to all parsed numeric variables. Used to
  /// automatically free them once they are guaranteed to no longer be used.
  std::vector<std::unique_ptr<FileCheckNumericVariable>> NumericVariables;

public:
  /// \returns the value of pattern variable \p VarName or None if no such
  /// variable has been defined.
  llvm::Optional<StringRef> getPatternVarValue(StringRef VarName);

  /// Defines pattern and numeric variables from definitions given on the
  /// command line, passed as a vector of [#]VAR=VAL strings in
  /// \p CmdlineDefines. Reports any error to \p SM and \returns whether an
  /// error occured.
  bool defineCmdlineVariables(std::vector<std::string> &CmdlineDefines,
                              SourceMgr &SM);

  /// Undefines local variables (variables whose name does not start with a '$'
  /// sign), i.e. removes them from GlobalVariableTable.
  void clearLocalVars();

private:
  /// Makes a new numeric expression instance and registers it for destruction
  /// when the context is destroyed.
  FileCheckNumExpr *makeNumExpr(binop_eval_t EvalBinop,
                                FileCheckNumericVariable *OperandLeft,
                                uint64_t OperandRight);

  /// Makes a new numeric variable and registers it for destruction when the
  /// context is destroyed.
  FileCheckNumericVariable *makeNumericVariable(StringRef Name, uint64_t Value);
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
  /// pattern at match time, e.g. "foo[[bar]]baz[[#N+1]]". In this case, the
  /// RegExStr will contain "foobaz" and we'll get two entries in this vector
  /// that tells us to insert the value of pattern variable "bar" at offset 3
  /// and the value of numeric expression "N+1" at offset 6. Uses are
  /// represented by a FileCheckPatternSubstitution class to abstract whether
  /// it is a pattern variable or a numeric expression.
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

  /// Pointer to a class instance holding the global state shared by all
  /// patterns:
  /// - separate tables with the values of live pattern and numeric variables
  ///   respectively at the start of any given CHECK line;
  /// - table holding whether a pattern variable has been defined at any given
  ///   point during the parsing phase.
  FileCheckPatternContext *Context;

  Check::FileCheckType CheckTy;

  /// Contains the number of line this pattern is in.
  unsigned LineNumber;

public:
  explicit FileCheckPattern(Check::FileCheckType Ty,
                            FileCheckPatternContext *Context)
      : Context(Context), CheckTy(Ty) {}

  /// \returns the location in source code.
  SMLoc getLoc() const { return PatternLoc; }

  /// \returns the pointer to the global state for all patterns in this
  /// FileCheck instance.
  FileCheckPatternContext *getContext() const { return Context; }

  /// \returns whether \p C is a valid first character for a variable name.
  static bool isValidVarNameStart(char C);
  /// Verifies that the string at the start of \p Str is a well formed
  /// variable. \returns false if it is and sets \p IsPseudo to indicate if it
  /// is a pseudo variable and \p TrailIdx to the position of the last
  /// character that is part of the variable name. Otherwise, only
  /// \returns true.
  static bool parseVariable(StringRef Str, bool &IsPseudo, unsigned &TrailIdx);
  /// Parses a numeric expression involving (pseudo if \p IsPseudo is true)
  /// variable \p Name with the string corresponding to the operation being
  /// performed in \p Trailer. \returns the class representing the numeric
  /// expression or nullptr if parsing fails in which case errors are reported
  /// on \p SM.
  FileCheckNumExpr *parseNumericExpression(StringRef Name, bool IsPseudo,
                                           StringRef Trailer,
                                           const SourceMgr &SM) const;
  /// Parses the pattern in \p PatternStr and initializes this FileCheckPattern
  /// instance accordingly.
  ///
  /// \p Prefix provides which prefix is being matched, \p Req describes the
  /// global options that influence the parsing such as whitespace
  /// canonicalization, \p SM provides the SourceMgr used for error reports,
  /// and \p LineNumber is the line number in the input file from which the
  /// pattern string was read. \returns true in case of an error, false
  /// otherwise.
  bool ParsePattern(StringRef PatternStr, StringRef Prefix, SourceMgr &SM,
                    unsigned LineNumber, const FileCheckRequest &Req);
  /// Matches the pattern string against the input buffer \p Buffer
  ///
  /// \returns the position that is matched or npos if there is no match. If
  /// there is a match, updates \p MatchLen with the size of the matched
  /// string.
  ///
  /// The GlobalVariableTable StringMap in the FileCheckPatternContext class
  /// instance provides the current values of FileCheck pattern variables and
  /// is updated if this match defines new values.
  size_t match(StringRef Buffer, size_t &MatchLen) const;
  /// Prints the value of successful substitutions or the name of the undefined
  /// pattern or numeric variable preventing such a successful substitution.
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
  /// Computes an arbitrary estimate for the quality of matching this pattern
  /// at the start of \p Buffer; a distance of zero should correspond to a
  /// perfect match.
  unsigned computeMatchDistance(StringRef Buffer) const;
  /// Finds the closing sequence of a regex variable usage or definition.
  ///
  /// \p Str has to point in the beginning of the definition (right after the
  /// opening sequence). \p SM holds the SourceMgr used for error repporting.
  ///  \returns the offset of the closing sequence within Str, or npos if it
  /// was not found.
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

  /// Matches check string and its "not strings" and/or "dag strings".
  size_t Check(const SourceMgr &SM, StringRef Buffer, bool IsLabelScanMode,
               size_t &MatchLen, FileCheckRequest &Req,
               std::vector<FileCheckDiag> *Diags) const;

  /// Verifies that there is a single line in the given \p Buffer. Errors are
  /// reported against \p SM.
  bool CheckNext(const SourceMgr &SM, StringRef Buffer) const;
  /// Verifies that there is no newline in the given \p Buffer. Errors are
  /// reported against \p SM.
  bool CheckSame(const SourceMgr &SM, StringRef Buffer) const;
  /// Verifies that none of the strings in \p NotStrings are found in the given
  /// \p Buffer. Errors are reported against \p SM and diagnostics recorded in
  /// \p Diags according to the verbosity level set in \p Req.
  bool CheckNot(const SourceMgr &SM, StringRef Buffer,
                const std::vector<const FileCheckPattern *> &NotStrings,
                const FileCheckRequest &Req,
                std::vector<FileCheckDiag> *Diags) const;
  /// Matches "dag strings" and their mixed "not strings".
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

  /// Reads the check file from \p Buffer and records the expected strings it
  /// contains in the \p CheckStrings vector. Errors are reported against
  /// \p SM.
  ///
  /// Only expected strings whose prefix is one of those listed in \p PrefixRE
  /// are recorded. \returns true in case of an error, false otherwise.
  bool ReadCheckFile(SourceMgr &SM, StringRef Buffer, Regex &PrefixRE,
                     std::vector<FileCheckString> &CheckStrings);

  bool ValidateCheckPrefixes();

  /// Canonicalizes whitespaces in the file. Line endings are replaced with
  /// UNIX-style '\n'.
  StringRef CanonicalizeFile(MemoryBuffer &MB,
                             SmallVectorImpl<char> &OutputBuffer);

  /// Checks the input to FileCheck provided in the \p Buffer against the
  /// \p CheckStrings read from the check file and record diagnostics emitted
  /// in \p Diags. Errors are recorded against \p SM.
  ///
  /// \returns false if the input fails to satisfy the checks.
  bool CheckInput(SourceMgr &SM, StringRef Buffer,
                  ArrayRef<FileCheckString> CheckStrings,
                  std::vector<FileCheckDiag> *Diags = nullptr);
};
} // namespace llvm
#endif
