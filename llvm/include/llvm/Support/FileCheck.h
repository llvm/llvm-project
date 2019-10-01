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
// Numeric substitution handling code.
//===----------------------------------------------------------------------===//

/// Class representing a numeric variable with a given value in a numeric
/// expression. Each definition of a variable gets its own instance of this
/// class. Variable uses share the same instance as their respective
/// definition.
class FileCheckNumericVariable {
private:
  /// Name of the numeric variable.
  StringRef Name;

  /// Value of numeric variable, if defined, or None otherwise.
  Optional<uint64_t> Value;

  /// Line number where this variable is defined. Used to determine whether a
  /// variable is defined on the same line as a given use.
  size_t DefLineNumber;

public:
  /// Constructor for a variable \p Name defined at line \p DefLineNumber.
  FileCheckNumericVariable(size_t DefLineNumber, StringRef Name)
      : Name(Name), DefLineNumber(DefLineNumber) {}

  /// Constructor for numeric variable \p Name with a known \p Value at parse
  /// time (e.g. the @LINE numeric variable).
  FileCheckNumericVariable(StringRef Name, uint64_t Value)
      : Name(Name), Value(Value), DefLineNumber(0) {}

  /// \returns name of that numeric variable.
  StringRef getName() const { return Name; }

  /// \returns value of this numeric variable.
  Optional<uint64_t> getValue() const { return Value; }

  /// Sets value of this numeric variable if not defined. \returns whether the
  /// variable was already defined.
  bool setValue(uint64_t Value);

  /// Clears value of this numeric variable. \returns whether the variable was
  /// already undefined.
  bool clearValue();

  /// \returns the line number where this variable is defined.
  size_t getDefLineNumber() { return DefLineNumber; }
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
  Optional<uint64_t> eval() const;

  /// \returns the name of the undefined variable used in this expression if
  /// any or an empty string otherwise.
  StringRef getUndefVarName() const;
};

class FileCheckPatternContext;

/// Class representing a substitution to perform in the RegExStr string.
class FileCheckSubstitution {
protected:
  /// Pointer to a class instance holding, among other things, the table with
  /// the values of live string variables at the start of any given CHECK line.
  /// Used for substituting string variables with the text they were defined
  /// as. Numeric expressions are linked to the numeric variables they use at
  /// parse time and directly access the value of the numeric variable to
  /// evaluate their value.
  FileCheckPatternContext *Context;

  /// The string that needs to be substituted for something else. For a
  /// string variable this is its name, otherwise this is the whole numeric
  /// expression.
  StringRef FromStr;

  // Index in RegExStr of where to do the substitution.
  size_t InsertIdx;

public:
  FileCheckSubstitution(FileCheckPatternContext *Context, StringRef VarName,
                        size_t InsertIdx)
      : Context(Context), FromStr(VarName), InsertIdx(InsertIdx) {}

  virtual ~FileCheckSubstitution() = default;

  /// \returns the string to be substituted for something else.
  StringRef getFromString() const { return FromStr; }

  /// \returns the index where the substitution is to be performed in RegExStr.
  size_t getIndex() const { return InsertIdx; }

  /// \returns a string containing the result of the substitution represented
  /// by this class instance or None if substitution failed.
  virtual Optional<std::string> getResult() const = 0;

  /// \returns the name of the variable used in this substitution if undefined,
  /// or an empty string otherwise.
  virtual StringRef getUndefVarName() const = 0;
};

class FileCheckStringSubstitution : public FileCheckSubstitution {
public:
  FileCheckStringSubstitution(FileCheckPatternContext *Context,
                              StringRef VarName, size_t InsertIdx)
      : FileCheckSubstitution(Context, VarName, InsertIdx) {}

  /// \returns the text that the string variable in this substitution matched
  /// when defined, or None if the variable is undefined.
  Optional<std::string> getResult() const override;

  /// \returns the name of the string variable used in this substitution if
  /// undefined, or an empty string otherwise.
  StringRef getUndefVarName() const override;
};

class FileCheckNumericSubstitution : public FileCheckSubstitution {
private:
  /// Pointer to the class representing the numeric expression whose value is
  /// to be substituted.
  FileCheckNumExpr *NumExpr;

public:
  FileCheckNumericSubstitution(FileCheckPatternContext *Context, StringRef Expr,
                               FileCheckNumExpr *NumExpr, size_t InsertIdx)
      : FileCheckSubstitution(Context, Expr, InsertIdx), NumExpr(NumExpr) {}

  /// \returns a string containing the result of evaluating the numeric
  /// expression in this substitution, or None if evaluation failed.
  Optional<std::string> getResult() const override;

  /// \returns the name of the numeric variable used in this substitution if
  /// undefined, or an empty string otherwise.
  StringRef getUndefVarName() const override;
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
  /// When matching a given pattern, this holds the value of all the string
  /// variables defined in previous patterns. In a pattern, only the last
  /// definition for a given variable is recorded in this table.
  /// Back-references are used for uses after any the other definition.
  StringMap<StringRef> GlobalVariableTable;

  /// Map of all string variables defined so far. Used at parse time to detect
  /// a name conflict between a numeric variable and a string variable when
  /// the former is defined on a later line than the latter.
  StringMap<bool> DefinedVariableTable;

  /// When matching a given pattern, this holds the pointers to the classes
  /// representing the last definitions of numeric variables defined in
  /// previous patterns. Earlier definitions of the variables, if any, have
  /// their own class instance not referenced by this table. When matching a
  /// pattern all definitions for that pattern are recorded in the
  /// NumericVariableDefs table in the FileCheckPattern instance of that
  /// pattern.
  StringMap<FileCheckNumericVariable *> GlobalNumericVariableTable;

  /// Vector holding pointers to all parsed numeric expressions. Used to
  /// automatically free the numeric expressions once they are guaranteed to no
  /// longer be used.
  std::vector<std::unique_ptr<FileCheckNumExpr>> NumExprs;

  /// Vector holding pointers to all parsed numeric variables. Used to
  /// automatically free them once they are guaranteed to no longer be used.
  std::vector<std::unique_ptr<FileCheckNumericVariable>> NumericVariables;

  /// Vector holding pointers to all substitutions. Used to automatically free
  /// them once they are guaranteed to no longer be used.
  std::vector<std::unique_ptr<FileCheckSubstitution>> Substitutions;

public:
  /// \returns the value of string variable \p VarName or None if no such
  /// variable has been defined.
  Optional<StringRef> getPatternVarValue(StringRef VarName);

  /// Defines string and numeric variables from definitions given on the
  /// command line, passed as a vector of [#]VAR=VAL strings in
  /// \p CmdlineDefines. Reports any error to \p SM and \returns whether an
  /// error occured.
  bool defineCmdlineVariables(std::vector<std::string> &CmdlineDefines,
                              SourceMgr &SM);

  /// Undefines local variables (variables whose name does not start with a '$'
  /// sign), i.e. removes them from GlobalVariableTable and from
  /// GlobalNumericVariableTable and also clears the value of numeric
  /// variables.
  void clearLocalVars();

private:
  /// Makes a new numeric expression instance and registers it for destruction
  /// when the context is destroyed.
  FileCheckNumExpr *makeNumExpr(binop_eval_t EvalBinop,
                                FileCheckNumericVariable *OperandLeft,
                                uint64_t OperandRight);

  /// Makes a new numeric variable and registers it for destruction when the
  /// context is destroyed.
  template <class... Types>
  FileCheckNumericVariable *makeNumericVariable(Types... args);

  /// Makes a new string substitution and registers it for destruction when the
  /// context is destroyed.
  FileCheckSubstitution *makeStringSubstitution(StringRef VarName,
                                                size_t InsertIdx);

  /// Makes a new numeric substitution and registers it for destruction when
  /// the context is destroyed.
  FileCheckSubstitution *makeNumericSubstitution(StringRef Expr,
                                                 FileCheckNumExpr *NumExpr,
                                                 size_t InsertIdx);
};

class FileCheckPattern {
  SMLoc PatternLoc;

  /// A fixed string to match as the pattern or empty if this pattern requires
  /// a regex match.
  StringRef FixedStr;

  /// A regex string to match as the pattern or empty if this pattern requires
  /// a fixed string to match.
  std::string RegExStr;

  /// Entries in this vector represent a substitution of a string variable or a
  /// numeric expression in the RegExStr regex at match time. For example, in
  /// the case of a CHECK directive with the pattern "foo[[bar]]baz[[#N+1]]",
  /// RegExStr will contain "foobaz" and we'll get two entries in this vector
  /// that tells us to insert the value of string variable "bar" at offset 3
  /// and the value of numeric expression "N+1" at offset 6.
  std::vector<FileCheckSubstitution *> Substitutions;

  /// Maps names of string variables defined in a pattern to the number of
  /// their parenthesis group in RegExStr capturing their last definition.
  ///
  /// E.g. for the pattern "foo[[bar:.*]]baz([[bar]][[QUUX]][[bar:.*]])",
  /// RegExStr will be "foo(.*)baz(\1<quux value>(.*))" where <quux value> is
  /// the value captured for QUUX on the earlier line where it was defined, and
  /// VariableDefs will map "bar" to the third parenthesis group which captures
  /// the second definition of "bar".
  ///
  /// Note: uses std::map rather than StringMap to be able to get the key when
  /// iterating over values.
  std::map<StringRef, unsigned> VariableDefs;

  /// Structure representing the definition of a numeric variable in a pattern.
  /// It holds the pointer to the class representing the numeric variable whose
  /// value is being defined and the number of the parenthesis group in
  /// RegExStr to capture that value.
  struct FileCheckNumExprMatch {
    /// Pointer to class representing the numeric variable whose value is being
    /// defined.
    FileCheckNumericVariable *DefinedNumericVariable;

    /// Number of the parenthesis group in RegExStr that captures the value of
    /// this numeric variable definition.
    unsigned CaptureParenGroup;
  };

  /// Holds the number of the parenthesis group in RegExStr and pointer to the
  /// corresponding FileCheckNumericVariable class instance of all numeric
  /// variable definitions. Used to set the matched value of all those
  /// variables.
  StringMap<FileCheckNumExprMatch> NumericVariableDefs;

  /// Pointer to a class instance holding the global state shared by all
  /// patterns:
  /// - separate tables with the values of live string and numeric variables
  ///   respectively at the start of any given CHECK line;
  /// - table holding whether a string variable has been defined at any given
  ///   point during the parsing phase.
  FileCheckPatternContext *Context;

  Check::FileCheckType CheckTy;

  /// Line number for this CHECK pattern. Used to determine whether a variable
  /// definition is made on an earlier line to the one with this CHECK.
  size_t LineNumber;

public:
  FileCheckPattern(Check::FileCheckType Ty, FileCheckPatternContext *Context,
                   size_t Line)
      : Context(Context), CheckTy(Ty), LineNumber(Line) {}

  /// \returns the location in source code.
  SMLoc getLoc() const { return PatternLoc; }

  /// \returns the pointer to the global state for all patterns in this
  /// FileCheck instance.
  FileCheckPatternContext *getContext() const { return Context; }

  /// \returns whether \p C is a valid first character for a variable name.
  static bool isValidVarNameStart(char C);
  /// Parses the string at the start of \p Str for a variable name and \returns
  /// whether the variable name is ill-formed. If parsing succeeded, sets
  /// \p IsPseudo to indicate if it is a pseudo variable, sets \p Name to the
  /// parsed variable name and strips \p Str from the variable name.
  static bool parseVariable(StringRef &Str, StringRef &Name, bool &IsPseudo);
  /// Parses \p Expr for the definition of a numeric variable, returning an
  /// error if \p Context already holds a string variable with the same name.
  /// \returns whether parsing fails, in which case errors are reported on
  /// \p SM. Otherwise, sets \p Name to the name of the parsed numeric
  /// variable.
  static bool parseNumericVariableDefinition(StringRef &Expr, StringRef &Name,
                                             FileCheckPatternContext *Context,
                                             const SourceMgr &SM);
  /// Parses \p Expr for a numeric substitution block. \returns the class
  /// representing the AST of the numeric expression whose value must be
  /// substituted, or nullptr if parsing fails, in which case errors are
  /// reported on \p SM. Sets \p DefinedNumericVariable to point to the class
  /// representing the numeric variable defined in this numeric substitution
  /// block, or nullptr if this block does not define any variable.
  FileCheckNumExpr *parseNumericSubstitutionBlock(
      StringRef Expr, FileCheckNumericVariable *&DefinedNumericVariable,
      const SourceMgr &SM) const;
  /// Parses the pattern in \p PatternStr and initializes this FileCheckPattern
  /// instance accordingly.
  ///
  /// \p Prefix provides which prefix is being matched, \p Req describes the
  /// global options that influence the parsing such as whitespace
  /// canonicalization, \p SM provides the SourceMgr used for error reports.
  /// \returns true in case of an error, false otherwise.
  bool parsePattern(StringRef PatternStr, StringRef Prefix, SourceMgr &SM,
                    const FileCheckRequest &Req);
  /// Matches the pattern string against the input buffer \p Buffer
  ///
  /// \returns the position that is matched or npos if there is no match. If
  /// there is a match, updates \p MatchLen with the size of the matched
  /// string.
  ///
  /// The GlobalVariableTable StringMap in the FileCheckPatternContext class
  /// instance provides the current values of FileCheck string variables and
  /// is updated if this match defines new values. Likewise, the
  /// GlobalNumericVariableTable StringMap in the same class provides the
  /// current values of FileCheck numeric variables and is updated if this
  /// match defines new numeric values.
  size_t match(StringRef Buffer, size_t &MatchLen, const SourceMgr &SM) const;
  /// Prints the value of successful substitutions or the name of the undefined
  /// string or numeric variable preventing a successful substitution.
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

  /// Parses \p Expr for the use of a numeric variable. \returns the pointer to
  /// the class instance representing that variable if successful, or nullptr
  /// otherwise, in which case errors are reported on \p SM.
  FileCheckNumericVariable *parseNumericVariableUse(StringRef &Expr,
                                                    const SourceMgr &SM) const;
  /// Parses \p Expr for a binary operation.
  /// \returns the class representing the binary operation of the numeric
  /// expression, or nullptr if parsing fails, in which case errors are
  /// reported on \p SM.
  FileCheckNumExpr *parseBinop(StringRef &Expr, const SourceMgr &SM) const;
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
