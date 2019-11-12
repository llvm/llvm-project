//===-- FileCheckImpl.h - Private FileCheck Interface ------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the private interfaces of FileCheck. Its purpose is to
// allow unit testing of FileCheck and to separate the interface from the
// implementation. It is only meant to be used by FileCheck.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_SUPPORT_FILECHECKIMPL_H
#define LLVM_LIB_SUPPORT_FILECHECKIMPL_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include <map>
#include <string>
#include <vector>

namespace llvm {

//===----------------------------------------------------------------------===//
// Numeric substitution handling code.
//===----------------------------------------------------------------------===//

/// Base class representing the AST of a given expression.
class FileCheckExpressionAST {
public:
  virtual ~FileCheckExpressionAST() = default;

  /// Evaluates and \returns the value of the expression represented by this
  /// AST or an error if evaluation fails.
  virtual Expected<uint64_t> eval() const = 0;
};

/// Class representing an unsigned literal in the AST of an expression.
class FileCheckExpressionLiteral : public FileCheckExpressionAST {
private:
  /// Actual value of the literal.
  uint64_t Value;

public:
  /// Constructs a literal with the specified value.
  FileCheckExpressionLiteral(uint64_t Val) : Value(Val) {}

  /// \returns the literal's value.
  Expected<uint64_t> eval() const override { return Value; }
};

/// Class to represent an undefined variable error, which quotes that
/// variable's name when printed.
class FileCheckUndefVarError : public ErrorInfo<FileCheckUndefVarError> {
private:
  StringRef VarName;

public:
  static char ID;

  FileCheckUndefVarError(StringRef VarName) : VarName(VarName) {}

  StringRef getVarName() const { return VarName; }

  std::error_code convertToErrorCode() const override {
    return inconvertibleErrorCode();
  }

  /// Print name of variable associated with this error.
  void log(raw_ostream &OS) const override {
    OS << "\"";
    OS.write_escaped(VarName) << "\"";
  }
};

/// Class representing a numeric variable and its associated current value.
class FileCheckNumericVariable {
private:
  /// Name of the numeric variable.
  StringRef Name;

  /// Value of numeric variable, if defined, or None otherwise.
  Optional<uint64_t> Value;

  /// Line number where this variable is defined, or None if defined before
  /// input is parsed. Used to determine whether a variable is defined on the
  /// same line as a given use.
  Optional<size_t> DefLineNumber;

public:
  /// Constructor for a variable \p Name defined at line \p DefLineNumber or
  /// defined before input is parsed if \p DefLineNumber is None.
  explicit FileCheckNumericVariable(StringRef Name,
                                    Optional<size_t> DefLineNumber = None)
      : Name(Name), DefLineNumber(DefLineNumber) {}

  /// \returns name of this numeric variable.
  StringRef getName() const { return Name; }

  /// \returns this variable's value.
  Optional<uint64_t> getValue() const { return Value; }

  /// Sets value of this numeric variable to \p NewValue.
  void setValue(uint64_t NewValue) { Value = NewValue; }

  /// Clears value of this numeric variable, regardless of whether it is
  /// currently defined or not.
  void clearValue() { Value = None; }

  /// \returns the line number where this variable is defined, if any, or None
  /// if defined before input is parsed.
  Optional<size_t> getDefLineNumber() const { return DefLineNumber; }
};

/// Class representing the use of a numeric variable in the AST of an
/// expression.
class FileCheckNumericVariableUse : public FileCheckExpressionAST {
private:
  /// Name of the numeric variable.
  StringRef Name;

  /// Pointer to the class instance for the variable this use is about.
  FileCheckNumericVariable *NumericVariable;

public:
  FileCheckNumericVariableUse(StringRef Name,
                              FileCheckNumericVariable *NumericVariable)
      : Name(Name), NumericVariable(NumericVariable) {}

  /// \returns the value of the variable referenced by this instance.
  Expected<uint64_t> eval() const override;
};

/// Type of functions evaluating a given binary operation.
using binop_eval_t = uint64_t (*)(uint64_t, uint64_t);

/// Class representing a single binary operation in the AST of an expression.
class FileCheckASTBinop : public FileCheckExpressionAST {
private:
  /// Left operand.
  std::unique_ptr<FileCheckExpressionAST> LeftOperand;

  /// Right operand.
  std::unique_ptr<FileCheckExpressionAST> RightOperand;

  /// Pointer to function that can evaluate this binary operation.
  binop_eval_t EvalBinop;

public:
  FileCheckASTBinop(binop_eval_t EvalBinop,
                    std::unique_ptr<FileCheckExpressionAST> LeftOp,
                    std::unique_ptr<FileCheckExpressionAST> RightOp)
      : EvalBinop(EvalBinop) {
    LeftOperand = std::move(LeftOp);
    RightOperand = std::move(RightOp);
  }

  /// Evaluates the value of the binary operation represented by this AST,
  /// using EvalBinop on the result of recursively evaluating the operands.
  /// \returns the expression value or an error if an undefined numeric
  /// variable is used in one of the operands.
  Expected<uint64_t> eval() const override;
};

class FileCheckPatternContext;

/// Class representing a substitution to perform in the RegExStr string.
class FileCheckSubstitution {
protected:
  /// Pointer to a class instance holding, among other things, the table with
  /// the values of live string variables at the start of any given CHECK line.
  /// Used for substituting string variables with the text they were defined
  /// as. Expressions are linked to the numeric variables they use at
  /// parse time and directly access the value of the numeric variable to
  /// evaluate their value.
  FileCheckPatternContext *Context;

  /// The string that needs to be substituted for something else. For a
  /// string variable this is its name, otherwise this is the whole expression.
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
  /// by this class instance or an error if substitution failed.
  virtual Expected<std::string> getResult() const = 0;
};

class FileCheckStringSubstitution : public FileCheckSubstitution {
public:
  FileCheckStringSubstitution(FileCheckPatternContext *Context,
                              StringRef VarName, size_t InsertIdx)
      : FileCheckSubstitution(Context, VarName, InsertIdx) {}

  /// \returns the text that the string variable in this substitution matched
  /// when defined, or an error if the variable is undefined.
  Expected<std::string> getResult() const override;
};

class FileCheckNumericSubstitution : public FileCheckSubstitution {
private:
  /// Pointer to the class representing the expression whose value is to be
  /// substituted.
  std::unique_ptr<FileCheckExpressionAST> ExpressionAST;

public:
  FileCheckNumericSubstitution(FileCheckPatternContext *Context, StringRef Expr,
                               std::unique_ptr<FileCheckExpressionAST> ExprAST,
                               size_t InsertIdx)
      : FileCheckSubstitution(Context, Expr, InsertIdx) {
    ExpressionAST = std::move(ExprAST);
  }

  /// \returns a string containing the result of evaluating the expression in
  /// this substitution, or an error if evaluation failed.
  Expected<std::string> getResult() const override;
};

//===----------------------------------------------------------------------===//
// Pattern handling code.
//===----------------------------------------------------------------------===//

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
  /// representing the numeric variables defined in previous patterns. When
  /// matching a pattern all definitions for that pattern are recorded in the
  /// NumericVariableDefs table in the FileCheckPattern instance of that
  /// pattern.
  StringMap<FileCheckNumericVariable *> GlobalNumericVariableTable;

  /// Pointer to the class instance representing the @LINE pseudo variable for
  /// easily updating its value.
  FileCheckNumericVariable *LineVariable = nullptr;

  /// Vector holding pointers to all parsed numeric variables. Used to
  /// automatically free them once they are guaranteed to no longer be used.
  std::vector<std::unique_ptr<FileCheckNumericVariable>> NumericVariables;

  /// Vector holding pointers to all substitutions. Used to automatically free
  /// them once they are guaranteed to no longer be used.
  std::vector<std::unique_ptr<FileCheckSubstitution>> Substitutions;

public:
  /// \returns the value of string variable \p VarName or an error if no such
  /// variable has been defined.
  Expected<StringRef> getPatternVarValue(StringRef VarName);

  /// Defines string and numeric variables from definitions given on the
  /// command line, passed as a vector of [#]VAR=VAL strings in
  /// \p CmdlineDefines. \returns an error list containing diagnostics against
  /// \p SM for all definition parsing failures, if any, or Success otherwise.
  Error defineCmdlineVariables(std::vector<std::string> &CmdlineDefines,
                               SourceMgr &SM);

  /// Create @LINE pseudo variable. Value is set when pattern are being
  /// matched.
  void createLineVariable();

  /// Undefines local variables (variables whose name does not start with a '$'
  /// sign), i.e. removes them from GlobalVariableTable and from
  /// GlobalNumericVariableTable and also clears the value of numeric
  /// variables.
  void clearLocalVars();

private:
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
  FileCheckSubstitution *
  makeNumericSubstitution(StringRef ExpressionStr,
                          std::unique_ptr<FileCheckExpressionAST> ExpressionAST,
                          size_t InsertIdx);
};

/// Class to represent an error holding a diagnostic with location information
/// used when printing it.
class FileCheckErrorDiagnostic : public ErrorInfo<FileCheckErrorDiagnostic> {
private:
  SMDiagnostic Diagnostic;

public:
  static char ID;

  FileCheckErrorDiagnostic(SMDiagnostic &&Diag) : Diagnostic(Diag) {}

  std::error_code convertToErrorCode() const override {
    return inconvertibleErrorCode();
  }

  /// Print diagnostic associated with this error when printing the error.
  void log(raw_ostream &OS) const override { Diagnostic.print(nullptr, OS); }

  static Error get(const SourceMgr &SM, SMLoc Loc, const Twine &ErrMsg) {
    return make_error<FileCheckErrorDiagnostic>(
        SM.GetMessage(Loc, SourceMgr::DK_Error, ErrMsg));
  }

  static Error get(const SourceMgr &SM, StringRef Buffer, const Twine &ErrMsg) {
    return get(SM, SMLoc::getFromPointer(Buffer.data()), ErrMsg);
  }
};

class FileCheckNotFoundError : public ErrorInfo<FileCheckNotFoundError> {
public:
  static char ID;

  std::error_code convertToErrorCode() const override {
    return inconvertibleErrorCode();
  }

  /// Print diagnostic associated with this error when printing the error.
  void log(raw_ostream &OS) const override {
    OS << "String not found in input";
  }
};

class FileCheckPattern {
  SMLoc PatternLoc;

  /// A fixed string to match as the pattern or empty if this pattern requires
  /// a regex match.
  StringRef FixedStr;

  /// A regex string to match as the pattern or empty if this pattern requires
  /// a fixed string to match.
  std::string RegExStr;

  /// Entries in this vector represent a substitution of a string variable or
  /// an expression in the RegExStr regex at match time. For example, in the
  /// case of a CHECK directive with the pattern "foo[[bar]]baz[[#N+1]]",
  /// RegExStr will contain "foobaz" and we'll get two entries in this vector
  /// that tells us to insert the value of string variable "bar" at offset 3
  /// and the value of expression "N+1" at offset 6.
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
  struct FileCheckNumericVariableMatch {
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
  StringMap<FileCheckNumericVariableMatch> NumericVariableDefs;

  /// Pointer to a class instance holding the global state shared by all
  /// patterns:
  /// - separate tables with the values of live string and numeric variables
  ///   respectively at the start of any given CHECK line;
  /// - table holding whether a string variable has been defined at any given
  ///   point during the parsing phase.
  FileCheckPatternContext *Context;

  Check::FileCheckType CheckTy;

  /// Line number for this CHECK pattern or None if it is an implicit pattern.
  /// Used to determine whether a variable definition is made on an earlier
  /// line to the one with this CHECK.
  Optional<size_t> LineNumber;

  /// Ignore case while matching if set to true.
  bool IgnoreCase = false;

public:
  FileCheckPattern(Check::FileCheckType Ty, FileCheckPatternContext *Context,
                   Optional<size_t> Line = None)
      : Context(Context), CheckTy(Ty), LineNumber(Line) {}

  /// \returns the location in source code.
  SMLoc getLoc() const { return PatternLoc; }

  /// \returns the pointer to the global state for all patterns in this
  /// FileCheck instance.
  FileCheckPatternContext *getContext() const { return Context; }

  /// \returns whether \p C is a valid first character for a variable name.
  static bool isValidVarNameStart(char C);

  /// Parsing information about a variable.
  struct VariableProperties {
    StringRef Name;
    bool IsPseudo;
  };

  /// Parses the string at the start of \p Str for a variable name. \returns
  /// a VariableProperties structure holding the variable name and whether it
  /// is the name of a pseudo variable, or an error holding a diagnostic
  /// against \p SM if parsing fail. If parsing was successful, also strips
  /// \p Str from the variable name.
  static Expected<VariableProperties> parseVariable(StringRef &Str,
                                                    const SourceMgr &SM);
  /// Parses \p Expr for a numeric substitution block at line \p LineNumber,
  /// or before input is parsed if \p LineNumber is None. Parameter
  /// \p IsLegacyLineExpr indicates whether \p Expr should be a legacy @LINE
  /// expression and \p Context points to the class instance holding the live
  /// string and numeric variables. \returns a pointer to the class instance
  /// representing the AST of the expression whose value must be substitued, or
  /// an error holding a diagnostic against \p SM if parsing fails. If
  /// substitution was successful, sets \p DefinedNumericVariable to point to
  /// the class representing the numeric variable defined in this numeric
  /// substitution block, or None if this block does not define any variable.
  static Expected<std::unique_ptr<FileCheckExpressionAST>>
  parseNumericSubstitutionBlock(
      StringRef Expr,
      Optional<FileCheckNumericVariable *> &DefinedNumericVariable,
      bool IsLegacyLineExpr, Optional<size_t> LineNumber,
      FileCheckPatternContext *Context, const SourceMgr &SM);
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
  /// \returns the position that is matched or an error indicating why matching
  /// failed. If there is a match, updates \p MatchLen with the size of the
  /// matched string.
  ///
  /// The GlobalVariableTable StringMap in the FileCheckPatternContext class
  /// instance provides the current values of FileCheck string variables and
  /// is updated if this match defines new values. Likewise, the
  /// GlobalNumericVariableTable StringMap in the same class provides the
  /// current values of FileCheck numeric variables and is updated if this
  /// match defines new numeric values.
  Expected<size_t> match(StringRef Buffer, size_t &MatchLen,
                         const SourceMgr &SM) const;
  /// Prints the value of successful substitutions or the name of the undefined
  /// string or numeric variables preventing a successful substitution.
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
  /// opening sequence). \p SM holds the SourceMgr used for error reporting.
  ///  \returns the offset of the closing sequence within Str, or npos if it
  /// was not found.
  static size_t FindRegexVarEnd(StringRef Str, SourceMgr &SM);

  /// Parses \p Expr for the name of a numeric variable to be defined at line
  /// \p LineNumber, or before input is parsed if \p LineNumber is None.
  /// \returns a pointer to the class instance representing that variable,
  /// creating it if needed, or an error holding a diagnostic against \p SM
  /// should defining such a variable be invalid.
  static Expected<FileCheckNumericVariable *> parseNumericVariableDefinition(
      StringRef &Expr, FileCheckPatternContext *Context,
      Optional<size_t> LineNumber, const SourceMgr &SM);
  /// Parses \p Name as a (pseudo if \p IsPseudo is true) numeric variable use
  /// at line \p LineNumber, or before input is parsed if \p LineNumber is
  /// None. Parameter \p Context points to the class instance holding the live
  /// string and numeric variables. \returns the pointer to the class instance
  /// representing that variable if successful, or an error holding a
  /// diagnostic against \p SM otherwise.
  static Expected<std::unique_ptr<FileCheckNumericVariableUse>>
  parseNumericVariableUse(StringRef Name, bool IsPseudo,
                          Optional<size_t> LineNumber,
                          FileCheckPatternContext *Context,
                          const SourceMgr &SM);
  enum class AllowedOperand { LineVar, Literal, Any };
  /// Parses \p Expr for use of a numeric operand at line \p LineNumber, or
  /// before input is parsed if \p LineNumber is None. Accepts both literal
  /// values and numeric variables, depending on the value of \p AO. Parameter
  /// \p Context points to the class instance holding the live string and
  /// numeric variables. \returns the class representing that operand in the
  /// AST of the expression or an error holding a diagnostic against \p SM
  /// otherwise.
  static Expected<std::unique_ptr<FileCheckExpressionAST>>
  parseNumericOperand(StringRef &Expr, AllowedOperand AO,
                      Optional<size_t> LineNumber,
                      FileCheckPatternContext *Context, const SourceMgr &SM);
  /// Parses \p Expr for a binary operation at line \p LineNumber, or before
  /// input is parsed if \p LineNumber is None. The left operand of this binary
  /// operation is given in \p LeftOp and \p IsLegacyLineExpr indicates whether
  /// we are parsing a legacy @LINE expression. Parameter \p Context points to
  /// the class instance holding the live string and numeric variables.
  /// \returns the class representing the binary operation in the AST of the
  /// expression, or an error holding a diagnostic against \p SM otherwise.
  static Expected<std::unique_ptr<FileCheckExpressionAST>>
  parseBinop(StringRef &Expr, std::unique_ptr<FileCheckExpressionAST> LeftOp,
             bool IsLegacyLineExpr, Optional<size_t> LineNumber,
             FileCheckPatternContext *Context, const SourceMgr &SM);
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

} // namespace llvm

#endif
