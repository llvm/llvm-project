//==-- llvm/Support/FileCheck.h ---------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
// Pattern Handling Code.
//===----------------------------------------------------------------------===//

namespace Check {
enum FileCheckType {
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
  CheckBadNot
};
}

class FileCheckPattern {
  SMLoc PatternLoc;

  /// A fixed string to match as the pattern or empty if this pattern requires
  /// a regex match.
  StringRef FixedStr;

  /// A regex string to match as the pattern or empty if this pattern requires
  /// a fixed string to match.
  std::string RegExStr;

  /// Entries in this vector map to uses of a variable in the pattern, e.g.
  /// "foo[[bar]]baz".  In this case, the RegExStr will contain "foobaz" and
  /// we'll get an entry in this vector that tells us to insert the value of
  /// bar at offset 3.
  std::vector<std::pair<StringRef, unsigned>> VariableUses;

  /// Maps definitions of variables to their parenthesized capture numbers.
  /// 
  /// E.g. for the pattern "foo[[bar:.*]]baz", VariableDefs will map "bar" to
  /// 1.
  std::map<StringRef, unsigned> VariableDefs;

  Check::FileCheckType CheckTy;

  /// Contains the number of line this pattern is in.
  unsigned LineNumber;

public:
  explicit FileCheckPattern(Check::FileCheckType Ty)
      : CheckTy(Ty) {}

  /// Returns the location in source code.
  SMLoc getLoc() const { return PatternLoc; }

  bool ParsePattern(StringRef PatternStr, StringRef Prefix, SourceMgr &SM,
                    unsigned LineNumber, const FileCheckRequest &Req);
  size_t Match(StringRef Buffer, size_t &MatchLen,
               StringMap<StringRef> &VariableTable) const;
  void PrintVariableUses(const SourceMgr &SM, StringRef Buffer,
                         const StringMap<StringRef> &VariableTable,
                         SMRange MatchRange = None) const;
  void PrintFuzzyMatch(const SourceMgr &SM, StringRef Buffer,
                       const StringMap<StringRef> &VariableTable) const;

  bool hasVariable() const {
    return !(VariableUses.empty() && VariableDefs.empty());
  }

  Check::FileCheckType getCheckTy() const { return CheckTy; }

private:
  bool AddRegExToRegEx(StringRef RS, unsigned &CurParen, SourceMgr &SM);
  void AddBackrefToRegEx(unsigned BackrefNum);
  unsigned
  ComputeMatchDistance(StringRef Buffer,
                       const StringMap<StringRef> &VariableTable) const;
  bool EvaluateExpression(StringRef Expr, std::string &Value) const;
  size_t FindRegexVarEnd(StringRef Str, SourceMgr &SM);
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
               size_t &MatchLen, StringMap<StringRef> &VariableTable,
               FileCheckRequest &Req) const;

  bool CheckNext(const SourceMgr &SM, StringRef Buffer) const;
  bool CheckSame(const SourceMgr &SM, StringRef Buffer) const;
  bool CheckNot(const SourceMgr &SM, StringRef Buffer,
                const std::vector<const FileCheckPattern *> &NotStrings,
                StringMap<StringRef> &VariableTable,
                const FileCheckRequest &Req) const;
  size_t CheckDag(const SourceMgr &SM, StringRef Buffer,
                  std::vector<const FileCheckPattern *> &NotStrings,
                  StringMap<StringRef> &VariableTable,
                  const FileCheckRequest &Req) const;
};

/// FileCheck class takes the request and exposes various methods that
/// use information from the request.
class FileCheck {
  FileCheckRequest Req;

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
                  ArrayRef<FileCheckString> CheckStrings);
};
} // namespace llvm
#endif
