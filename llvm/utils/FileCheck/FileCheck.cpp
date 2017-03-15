//===- FileCheck.cpp - Check that File's Contents match what is expected --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FileCheck does a line-by line check of a file that validates whether it
// contains the expected content.  This is useful for regression tests etc.
//
// This program exits with an exit status of 2 on error, exit status of 0 if
// the file matched the expected contents, and exit status of 1 if it did not
// contain the expected contents.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <map>
#include <string>
#include <system_error>
#include <vector>
using namespace llvm;

static cl::opt<std::string>
    CheckFilename(cl::Positional, cl::desc("<check-file>"), cl::Required);

static cl::opt<std::string>
    InputFilename("input-file", cl::desc("File to check (defaults to stdin)"),
                  cl::init("-"), cl::value_desc("filename"));

static cl::list<std::string> CheckPrefixes(
    "check-prefix",
    cl::desc("Prefix to use from check file (defaults to 'CHECK')"));
static cl::alias CheckPrefixesAlias(
    "check-prefixes", cl::aliasopt(CheckPrefixes), cl::CommaSeparated,
    cl::NotHidden,
    cl::desc(
        "Alias for -check-prefix permitting multiple comma separated values"));

static cl::opt<bool> NoCanonicalizeWhiteSpace(
    "strict-whitespace",
    cl::desc("Do not treat all horizontal whitespace as equivalent"));

static cl::list<std::string> ImplicitCheckNot(
    "implicit-check-not",
    cl::desc("Add an implicit negative check with this pattern to every\n"
             "positive check. This can be used to ensure that no instances of\n"
             "this pattern occur which are not matched by a positive pattern"),
    cl::value_desc("pattern"));

static cl::opt<bool> AllowEmptyInput(
    "allow-empty", cl::init(false),
    cl::desc("Allow the input file to be empty. This is useful when making\n"
             "checks that some error message does not occur, for example."));

static cl::opt<bool> MatchFullLines(
    "match-full-lines", cl::init(false),
    cl::desc("Require all positive matches to cover an entire input line.\n"
             "Allows leading and trailing whitespace if --strict-whitespace\n"
             "is not also passed."));

static cl::opt<bool> EnableVarScope(
    "enable-var-scope", cl::init(false),
    cl::desc("Enables scope for regex variables. Variables with names that\n"
             "do not start with '$' will be reset at the beginning of\n"
             "each CHECK-LABEL block."));

typedef cl::list<std::string>::const_iterator prefix_iterator;

//===----------------------------------------------------------------------===//
// Pattern Handling Code.
//===----------------------------------------------------------------------===//

namespace Check {
enum CheckType {
  CheckNone = 0,
  CheckPlain,
  CheckNext,
  CheckSame,
  CheckNot,
  CheckDAG,
  CheckLabel,

  /// Indicates the pattern only matches the end of file. This is used for
  /// trailing CHECK-NOTs.
  CheckEOF,

  /// Marks when parsing found a -NOT check combined with another CHECK suffix.
  CheckBadNot
};
}

class Pattern {
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

  Check::CheckType CheckTy;

  /// Contains the number of line this pattern is in.
  unsigned LineNumber;

public:
  explicit Pattern(Check::CheckType Ty) : CheckTy(Ty) {}

  /// Returns the location in source code.
  SMLoc getLoc() const { return PatternLoc; }

  bool ParsePattern(StringRef PatternStr, StringRef Prefix, SourceMgr &SM,
                    unsigned LineNumber);
  size_t Match(StringRef Buffer, size_t &MatchLen,
               StringMap<StringRef> &VariableTable) const;
  void PrintFailureInfo(const SourceMgr &SM, StringRef Buffer,
                        const StringMap<StringRef> &VariableTable) const;

  bool hasVariable() const {
    return !(VariableUses.empty() && VariableDefs.empty());
  }

  Check::CheckType getCheckTy() const { return CheckTy; }

private:
  bool AddRegExToRegEx(StringRef RS, unsigned &CurParen, SourceMgr &SM);
  void AddBackrefToRegEx(unsigned BackrefNum);
  unsigned
  ComputeMatchDistance(StringRef Buffer,
                       const StringMap<StringRef> &VariableTable) const;
  bool EvaluateExpression(StringRef Expr, std::string &Value) const;
  size_t FindRegexVarEnd(StringRef Str, SourceMgr &SM);
};

/// Parses the given string into the Pattern.
///
/// \p Prefix provides which prefix is being matched, \p SM provides the
/// SourceMgr used for error reports, and \p LineNumber is the line number in
/// the input file from which the pattern string was read. Returns true in
/// case of an error, false otherwise.
bool Pattern::ParsePattern(StringRef PatternStr, StringRef Prefix,
                           SourceMgr &SM, unsigned LineNumber) {
  bool MatchFullLinesHere = MatchFullLines && CheckTy != Check::CheckNot;

  this->LineNumber = LineNumber;
  PatternLoc = SMLoc::getFromPointer(PatternStr.data());

  if (!(NoCanonicalizeWhiteSpace && MatchFullLines))
    // Ignore trailing whitespace.
    while (!PatternStr.empty() &&
           (PatternStr.back() == ' ' || PatternStr.back() == '\t'))
      PatternStr = PatternStr.substr(0, PatternStr.size() - 1);

  // Check that there is something on the line.
  if (PatternStr.empty()) {
    SM.PrintMessage(PatternLoc, SourceMgr::DK_Error,
                    "found empty check string with prefix '" + Prefix + ":'");
    return true;
  }

  // Check to see if this is a fixed string, or if it has regex pieces.
  if (!MatchFullLinesHere &&
      (PatternStr.size() < 2 || (PatternStr.find("{{") == StringRef::npos &&
                                 PatternStr.find("[[") == StringRef::npos))) {
    FixedStr = PatternStr;
    return false;
  }

  if (MatchFullLinesHere) {
    RegExStr += '^';
    if (!NoCanonicalizeWhiteSpace)
      RegExStr += " *";
  }

  // Paren value #0 is for the fully matched string.  Any new parenthesized
  // values add from there.
  unsigned CurParen = 1;

  // Otherwise, there is at least one regex piece.  Build up the regex pattern
  // by escaping scary characters in fixed strings, building up one big regex.
  while (!PatternStr.empty()) {
    // RegEx matches.
    if (PatternStr.startswith("{{")) {
      // This is the start of a regex match.  Scan for the }}.
      size_t End = PatternStr.find("}}");
      if (End == StringRef::npos) {
        SM.PrintMessage(SMLoc::getFromPointer(PatternStr.data()),
                        SourceMgr::DK_Error,
                        "found start of regex string with no end '}}'");
        return true;
      }

      // Enclose {{}} patterns in parens just like [[]] even though we're not
      // capturing the result for any purpose.  This is required in case the
      // expression contains an alternation like: CHECK:  abc{{x|z}}def.  We
      // want this to turn into: "abc(x|z)def" not "abcx|zdef".
      RegExStr += '(';
      ++CurParen;

      if (AddRegExToRegEx(PatternStr.substr(2, End - 2), CurParen, SM))
        return true;
      RegExStr += ')';

      PatternStr = PatternStr.substr(End + 2);
      continue;
    }

    // Named RegEx matches.  These are of two forms: [[foo:.*]] which matches .*
    // (or some other regex) and assigns it to the FileCheck variable 'foo'. The
    // second form is [[foo]] which is a reference to foo.  The variable name
    // itself must be of the form "[a-zA-Z_][0-9a-zA-Z_]*", otherwise we reject
    // it.  This is to catch some common errors.
    if (PatternStr.startswith("[[")) {
      // Find the closing bracket pair ending the match.  End is going to be an
      // offset relative to the beginning of the match string.
      size_t End = FindRegexVarEnd(PatternStr.substr(2), SM);

      if (End == StringRef::npos) {
        SM.PrintMessage(SMLoc::getFromPointer(PatternStr.data()),
                        SourceMgr::DK_Error,
                        "invalid named regex reference, no ]] found");
        return true;
      }

      StringRef MatchStr = PatternStr.substr(2, End);
      PatternStr = PatternStr.substr(End + 4);

      // Get the regex name (e.g. "foo").
      size_t NameEnd = MatchStr.find(':');
      StringRef Name = MatchStr.substr(0, NameEnd);

      if (Name.empty()) {
        SM.PrintMessage(SMLoc::getFromPointer(Name.data()), SourceMgr::DK_Error,
                        "invalid name in named regex: empty name");
        return true;
      }

      // Verify that the name/expression is well formed. FileCheck currently
      // supports @LINE, @LINE+number, @LINE-number expressions. The check here
      // is relaxed, more strict check is performed in \c EvaluateExpression.
      bool IsExpression = false;
      for (unsigned i = 0, e = Name.size(); i != e; ++i) {
        if (i == 0) {
          if (Name[i] == '$')  // Global vars start with '$'
            continue;
          if (Name[i] == '@') {
            if (NameEnd != StringRef::npos) {
              SM.PrintMessage(SMLoc::getFromPointer(Name.data()),
                              SourceMgr::DK_Error,
                              "invalid name in named regex definition");
              return true;
            }
            IsExpression = true;
            continue;
          }
        }
        if (Name[i] != '_' && !isalnum(Name[i]) &&
            (!IsExpression || (Name[i] != '+' && Name[i] != '-'))) {
          SM.PrintMessage(SMLoc::getFromPointer(Name.data() + i),
                          SourceMgr::DK_Error, "invalid name in named regex");
          return true;
        }
      }

      // Name can't start with a digit.
      if (isdigit(static_cast<unsigned char>(Name[0]))) {
        SM.PrintMessage(SMLoc::getFromPointer(Name.data()), SourceMgr::DK_Error,
                        "invalid name in named regex");
        return true;
      }

      // Handle [[foo]].
      if (NameEnd == StringRef::npos) {
        // Handle variables that were defined earlier on the same line by
        // emitting a backreference.
        if (VariableDefs.find(Name) != VariableDefs.end()) {
          unsigned VarParenNum = VariableDefs[Name];
          if (VarParenNum < 1 || VarParenNum > 9) {
            SM.PrintMessage(SMLoc::getFromPointer(Name.data()),
                            SourceMgr::DK_Error,
                            "Can't back-reference more than 9 variables");
            return true;
          }
          AddBackrefToRegEx(VarParenNum);
        } else {
          VariableUses.push_back(std::make_pair(Name, RegExStr.size()));
        }
        continue;
      }

      // Handle [[foo:.*]].
      VariableDefs[Name] = CurParen;
      RegExStr += '(';
      ++CurParen;

      if (AddRegExToRegEx(MatchStr.substr(NameEnd + 1), CurParen, SM))
        return true;

      RegExStr += ')';
    }

    // Handle fixed string matches.
    // Find the end, which is the start of the next regex.
    size_t FixedMatchEnd = PatternStr.find("{{");
    FixedMatchEnd = std::min(FixedMatchEnd, PatternStr.find("[["));
    RegExStr += Regex::escape(PatternStr.substr(0, FixedMatchEnd));
    PatternStr = PatternStr.substr(FixedMatchEnd);
  }

  if (MatchFullLinesHere) {
    if (!NoCanonicalizeWhiteSpace)
      RegExStr += " *";
    RegExStr += '$';
  }

  return false;
}

bool Pattern::AddRegExToRegEx(StringRef RS, unsigned &CurParen, SourceMgr &SM) {
  Regex R(RS);
  std::string Error;
  if (!R.isValid(Error)) {
    SM.PrintMessage(SMLoc::getFromPointer(RS.data()), SourceMgr::DK_Error,
                    "invalid regex: " + Error);
    return true;
  }

  RegExStr += RS.str();
  CurParen += R.getNumMatches();
  return false;
}

void Pattern::AddBackrefToRegEx(unsigned BackrefNum) {
  assert(BackrefNum >= 1 && BackrefNum <= 9 && "Invalid backref number");
  std::string Backref = std::string("\\") + std::string(1, '0' + BackrefNum);
  RegExStr += Backref;
}

/// Evaluates expression and stores the result to \p Value.
///
/// Returns true on success and false when the expression has invalid syntax.
bool Pattern::EvaluateExpression(StringRef Expr, std::string &Value) const {
  // The only supported expression is @LINE([\+-]\d+)?
  if (!Expr.startswith("@LINE"))
    return false;
  Expr = Expr.substr(StringRef("@LINE").size());
  int Offset = 0;
  if (!Expr.empty()) {
    if (Expr[0] == '+')
      Expr = Expr.substr(1);
    else if (Expr[0] != '-')
      return false;
    if (Expr.getAsInteger(10, Offset))
      return false;
  }
  Value = llvm::itostr(LineNumber + Offset);
  return true;
}

/// Matches the pattern string against the input buffer \p Buffer
///
/// This returns the position that is matched or npos if there is no match. If
/// there is a match, the size of the matched string is returned in \p
/// MatchLen.
///
/// The \p VariableTable StringMap provides the current values of filecheck
/// variables and is updated if this match defines new values.
size_t Pattern::Match(StringRef Buffer, size_t &MatchLen,
                      StringMap<StringRef> &VariableTable) const {
  // If this is the EOF pattern, match it immediately.
  if (CheckTy == Check::CheckEOF) {
    MatchLen = 0;
    return Buffer.size();
  }

  // If this is a fixed string pattern, just match it now.
  if (!FixedStr.empty()) {
    MatchLen = FixedStr.size();
    return Buffer.find(FixedStr);
  }

  // Regex match.

  // If there are variable uses, we need to create a temporary string with the
  // actual value.
  StringRef RegExToMatch = RegExStr;
  std::string TmpStr;
  if (!VariableUses.empty()) {
    TmpStr = RegExStr;

    unsigned InsertOffset = 0;
    for (const auto &VariableUse : VariableUses) {
      std::string Value;

      if (VariableUse.first[0] == '@') {
        if (!EvaluateExpression(VariableUse.first, Value))
          return StringRef::npos;
      } else {
        StringMap<StringRef>::iterator it =
            VariableTable.find(VariableUse.first);
        // If the variable is undefined, return an error.
        if (it == VariableTable.end())
          return StringRef::npos;

        // Look up the value and escape it so that we can put it into the regex.
        Value += Regex::escape(it->second);
      }

      // Plop it into the regex at the adjusted offset.
      TmpStr.insert(TmpStr.begin() + VariableUse.second + InsertOffset,
                    Value.begin(), Value.end());
      InsertOffset += Value.size();
    }

    // Match the newly constructed regex.
    RegExToMatch = TmpStr;
  }

  SmallVector<StringRef, 4> MatchInfo;
  if (!Regex(RegExToMatch, Regex::Newline).match(Buffer, &MatchInfo))
    return StringRef::npos;

  // Successful regex match.
  assert(!MatchInfo.empty() && "Didn't get any match");
  StringRef FullMatch = MatchInfo[0];

  // If this defines any variables, remember their values.
  for (const auto &VariableDef : VariableDefs) {
    assert(VariableDef.second < MatchInfo.size() && "Internal paren error");
    VariableTable[VariableDef.first] = MatchInfo[VariableDef.second];
  }

  MatchLen = FullMatch.size();
  return FullMatch.data() - Buffer.data();
}


/// Computes an arbitrary estimate for the quality of matching this pattern at
/// the start of \p Buffer; a distance of zero should correspond to a perfect
/// match.
unsigned
Pattern::ComputeMatchDistance(StringRef Buffer,
                              const StringMap<StringRef> &VariableTable) const {
  // Just compute the number of matching characters. For regular expressions, we
  // just compare against the regex itself and hope for the best.
  //
  // FIXME: One easy improvement here is have the regex lib generate a single
  // example regular expression which matches, and use that as the example
  // string.
  StringRef ExampleString(FixedStr);
  if (ExampleString.empty())
    ExampleString = RegExStr;

  // Only compare up to the first line in the buffer, or the string size.
  StringRef BufferPrefix = Buffer.substr(0, ExampleString.size());
  BufferPrefix = BufferPrefix.split('\n').first;
  return BufferPrefix.edit_distance(ExampleString);
}

/// Prints additional information about a failure to match involving this
/// pattern.
void Pattern::PrintFailureInfo(
    const SourceMgr &SM, StringRef Buffer,
    const StringMap<StringRef> &VariableTable) const {
  // If this was a regular expression using variables, print the current
  // variable values.
  if (!VariableUses.empty()) {
    for (const auto &VariableUse : VariableUses) {
      SmallString<256> Msg;
      raw_svector_ostream OS(Msg);
      StringRef Var = VariableUse.first;
      if (Var[0] == '@') {
        std::string Value;
        if (EvaluateExpression(Var, Value)) {
          OS << "with expression \"";
          OS.write_escaped(Var) << "\" equal to \"";
          OS.write_escaped(Value) << "\"";
        } else {
          OS << "uses incorrect expression \"";
          OS.write_escaped(Var) << "\"";
        }
      } else {
        StringMap<StringRef>::const_iterator it = VariableTable.find(Var);

        // Check for undefined variable references.
        if (it == VariableTable.end()) {
          OS << "uses undefined variable \"";
          OS.write_escaped(Var) << "\"";
        } else {
          OS << "with variable \"";
          OS.write_escaped(Var) << "\" equal to \"";
          OS.write_escaped(it->second) << "\"";
        }
      }

      SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()), SourceMgr::DK_Note,
                      OS.str());
    }
  }

  // Attempt to find the closest/best fuzzy match.  Usually an error happens
  // because some string in the output didn't exactly match. In these cases, we
  // would like to show the user a best guess at what "should have" matched, to
  // save them having to actually check the input manually.
  size_t NumLinesForward = 0;
  size_t Best = StringRef::npos;
  double BestQuality = 0;

  // Use an arbitrary 4k limit on how far we will search.
  for (size_t i = 0, e = std::min(size_t(4096), Buffer.size()); i != e; ++i) {
    if (Buffer[i] == '\n')
      ++NumLinesForward;

    // Patterns have leading whitespace stripped, so skip whitespace when
    // looking for something which looks like a pattern.
    if (Buffer[i] == ' ' || Buffer[i] == '\t')
      continue;

    // Compute the "quality" of this match as an arbitrary combination of the
    // match distance and the number of lines skipped to get to this match.
    unsigned Distance = ComputeMatchDistance(Buffer.substr(i), VariableTable);
    double Quality = Distance + (NumLinesForward / 100.);

    if (Quality < BestQuality || Best == StringRef::npos) {
      Best = i;
      BestQuality = Quality;
    }
  }

  // Print the "possible intended match here" line if we found something
  // reasonable and not equal to what we showed in the "scanning from here"
  // line.
  if (Best && Best != StringRef::npos && BestQuality < 50) {
    SM.PrintMessage(SMLoc::getFromPointer(Buffer.data() + Best),
                    SourceMgr::DK_Note, "possible intended match here");

    // FIXME: If we wanted to be really friendly we would show why the match
    // failed, as it can be hard to spot simple one character differences.
  }
}

/// Finds the closing sequence of a regex variable usage or definition.
///
/// \p Str has to point in the beginning of the definition (right after the
/// opening sequence). Returns the offset of the closing sequence within Str,
/// or npos if it was not found.
size_t Pattern::FindRegexVarEnd(StringRef Str, SourceMgr &SM) {
  // Offset keeps track of the current offset within the input Str
  size_t Offset = 0;
  // [...] Nesting depth
  size_t BracketDepth = 0;

  while (!Str.empty()) {
    if (Str.startswith("]]") && BracketDepth == 0)
      return Offset;
    if (Str[0] == '\\') {
      // Backslash escapes the next char within regexes, so skip them both.
      Str = Str.substr(2);
      Offset += 2;
    } else {
      switch (Str[0]) {
      default:
        break;
      case '[':
        BracketDepth++;
        break;
      case ']':
        if (BracketDepth == 0) {
          SM.PrintMessage(SMLoc::getFromPointer(Str.data()),
                          SourceMgr::DK_Error,
                          "missing closing \"]\" for regex variable");
          exit(1);
        }
        BracketDepth--;
        break;
      }
      Str = Str.substr(1);
      Offset++;
    }
  }

  return StringRef::npos;
}

//===----------------------------------------------------------------------===//
// Check Strings.
//===----------------------------------------------------------------------===//

/// A check that we found in the input file.
struct CheckString {
  /// The pattern to match.
  Pattern Pat;

  /// Which prefix name this check matched.
  StringRef Prefix;

  /// The location in the match file that the check string was specified.
  SMLoc Loc;

  /// All of the strings that are disallowed from occurring between this match
  /// string and the previous one (or start of file).
  std::vector<Pattern> DagNotStrings;

  CheckString(const Pattern &P, StringRef S, SMLoc L)
      : Pat(P), Prefix(S), Loc(L) {}

  size_t Check(const SourceMgr &SM, StringRef Buffer, bool IsLabelScanMode,
               size_t &MatchLen, StringMap<StringRef> &VariableTable) const;

  bool CheckNext(const SourceMgr &SM, StringRef Buffer) const;
  bool CheckSame(const SourceMgr &SM, StringRef Buffer) const;
  bool CheckNot(const SourceMgr &SM, StringRef Buffer,
                const std::vector<const Pattern *> &NotStrings,
                StringMap<StringRef> &VariableTable) const;
  size_t CheckDag(const SourceMgr &SM, StringRef Buffer,
                  std::vector<const Pattern *> &NotStrings,
                  StringMap<StringRef> &VariableTable) const;
};

/// Canonicalize whitespaces in the file. Line endings are replaced with
/// UNIX-style '\n'.
static StringRef CanonicalizeFile(MemoryBuffer &MB,
                                  SmallVectorImpl<char> &OutputBuffer) {
  OutputBuffer.reserve(MB.getBufferSize());

  for (const char *Ptr = MB.getBufferStart(), *End = MB.getBufferEnd();
       Ptr != End; ++Ptr) {
    // Eliminate trailing dosish \r.
    if (Ptr <= End - 2 && Ptr[0] == '\r' && Ptr[1] == '\n') {
      continue;
    }

    // If current char is not a horizontal whitespace or if horizontal
    // whitespace canonicalization is disabled, dump it to output as is.
    if (NoCanonicalizeWhiteSpace || (*Ptr != ' ' && *Ptr != '\t')) {
      OutputBuffer.push_back(*Ptr);
      continue;
    }

    // Otherwise, add one space and advance over neighboring space.
    OutputBuffer.push_back(' ');
    while (Ptr + 1 != End && (Ptr[1] == ' ' || Ptr[1] == '\t'))
      ++Ptr;
  }

  // Add a null byte and then return all but that byte.
  OutputBuffer.push_back('\0');
  return StringRef(OutputBuffer.data(), OutputBuffer.size() - 1);
}

static bool IsPartOfWord(char c) {
  return (isalnum(c) || c == '-' || c == '_');
}

// Get the size of the prefix extension.
static size_t CheckTypeSize(Check::CheckType Ty) {
  switch (Ty) {
  case Check::CheckNone:
  case Check::CheckBadNot:
    return 0;

  case Check::CheckPlain:
    return sizeof(":") - 1;

  case Check::CheckNext:
    return sizeof("-NEXT:") - 1;

  case Check::CheckSame:
    return sizeof("-SAME:") - 1;

  case Check::CheckNot:
    return sizeof("-NOT:") - 1;

  case Check::CheckDAG:
    return sizeof("-DAG:") - 1;

  case Check::CheckLabel:
    return sizeof("-LABEL:") - 1;

  case Check::CheckEOF:
    llvm_unreachable("Should not be using EOF size");
  }

  llvm_unreachable("Bad check type");
}

static Check::CheckType FindCheckType(StringRef Buffer, StringRef Prefix) {
  char NextChar = Buffer[Prefix.size()];

  // Verify that the : is present after the prefix.
  if (NextChar == ':')
    return Check::CheckPlain;

  if (NextChar != '-')
    return Check::CheckNone;

  StringRef Rest = Buffer.drop_front(Prefix.size() + 1);
  if (Rest.startswith("NEXT:"))
    return Check::CheckNext;

  if (Rest.startswith("SAME:"))
    return Check::CheckSame;

  if (Rest.startswith("NOT:"))
    return Check::CheckNot;

  if (Rest.startswith("DAG:"))
    return Check::CheckDAG;

  if (Rest.startswith("LABEL:"))
    return Check::CheckLabel;

  // You can't combine -NOT with another suffix.
  if (Rest.startswith("DAG-NOT:") || Rest.startswith("NOT-DAG:") ||
      Rest.startswith("NEXT-NOT:") || Rest.startswith("NOT-NEXT:") ||
      Rest.startswith("SAME-NOT:") || Rest.startswith("NOT-SAME:"))
    return Check::CheckBadNot;

  return Check::CheckNone;
}

// From the given position, find the next character after the word.
static size_t SkipWord(StringRef Str, size_t Loc) {
  while (Loc < Str.size() && IsPartOfWord(Str[Loc]))
    ++Loc;
  return Loc;
}

/// Search the buffer for the first prefix in the prefix regular expression.
///
/// This searches the buffer using the provided regular expression, however it
/// enforces constraints beyond that:
/// 1) The found prefix must not be a suffix of something that looks like
///    a valid prefix.
/// 2) The found prefix must be followed by a valid check type suffix using \c
///    FindCheckType above.
///
/// The first match of the regular expression to satisfy these two is returned,
/// otherwise an empty StringRef is returned to indicate failure.
///
/// If this routine returns a valid prefix, it will also shrink \p Buffer to
/// start at the beginning of the returned prefix, increment \p LineNumber for
/// each new line consumed from \p Buffer, and set \p CheckTy to the type of
/// check found by examining the suffix.
///
/// If no valid prefix is found, the state of Buffer, LineNumber, and CheckTy
/// is unspecified.
static StringRef FindFirstMatchingPrefix(Regex &PrefixRE, StringRef &Buffer,
                                         unsigned &LineNumber,
                                         Check::CheckType &CheckTy) {
  SmallVector<StringRef, 2> Matches;

  while (!Buffer.empty()) {
    // Find the first (longest) match using the RE.
    if (!PrefixRE.match(Buffer, &Matches))
      // No match at all, bail.
      return StringRef();

    StringRef Prefix = Matches[0];
    Matches.clear();

    assert(Prefix.data() >= Buffer.data() &&
           Prefix.data() < Buffer.data() + Buffer.size() &&
           "Prefix doesn't start inside of buffer!");
    size_t Loc = Prefix.data() - Buffer.data();
    StringRef Skipped = Buffer.substr(0, Loc);
    Buffer = Buffer.drop_front(Loc);
    LineNumber += Skipped.count('\n');

    // Check that the matched prefix isn't a suffix of some other check-like
    // word.
    // FIXME: This is a very ad-hoc check. it would be better handled in some
    // other way. Among other things it seems hard to distinguish between
    // intentional and unintentional uses of this feature.
    if (Skipped.empty() || !IsPartOfWord(Skipped.back())) {
      // Now extract the type.
      CheckTy = FindCheckType(Buffer, Prefix);

      // If we've found a valid check type for this prefix, we're done.
      if (CheckTy != Check::CheckNone)
        return Prefix;
    }

    // If we didn't successfully find a prefix, we need to skip this invalid
    // prefix and continue scanning. We directly skip the prefix that was
    // matched and any additional parts of that check-like word.
    Buffer = Buffer.drop_front(SkipWord(Buffer, Prefix.size()));
  }

  // We ran out of buffer while skipping partial matches so give up.
  return StringRef();
}

/// Read the check file, which specifies the sequence of expected strings.
///
/// The strings are added to the CheckStrings vector. Returns true in case of
/// an error, false otherwise.
static bool ReadCheckFile(SourceMgr &SM, StringRef Buffer, Regex &PrefixRE,
                          std::vector<CheckString> &CheckStrings) {
  std::vector<Pattern> ImplicitNegativeChecks;
  for (const auto &PatternString : ImplicitCheckNot) {
    // Create a buffer with fake command line content in order to display the
    // command line option responsible for the specific implicit CHECK-NOT.
    std::string Prefix = (Twine("-") + ImplicitCheckNot.ArgStr + "='").str();
    std::string Suffix = "'";
    std::unique_ptr<MemoryBuffer> CmdLine = MemoryBuffer::getMemBufferCopy(
        Prefix + PatternString + Suffix, "command line");

    StringRef PatternInBuffer =
        CmdLine->getBuffer().substr(Prefix.size(), PatternString.size());
    SM.AddNewSourceBuffer(std::move(CmdLine), SMLoc());

    ImplicitNegativeChecks.push_back(Pattern(Check::CheckNot));
    ImplicitNegativeChecks.back().ParsePattern(PatternInBuffer,
                                               "IMPLICIT-CHECK", SM, 0);
  }

  std::vector<Pattern> DagNotMatches = ImplicitNegativeChecks;

  // LineNumber keeps track of the line on which CheckPrefix instances are
  // found.
  unsigned LineNumber = 1;

  while (1) {
    Check::CheckType CheckTy;

    // See if a prefix occurs in the memory buffer.
    StringRef UsedPrefix = FindFirstMatchingPrefix(PrefixRE, Buffer, LineNumber,
                                                   CheckTy);
    if (UsedPrefix.empty())
      break;
    assert(UsedPrefix.data() == Buffer.data() &&
           "Failed to move Buffer's start forward, or pointed prefix outside "
           "of the buffer!");

    // Location to use for error messages.
    const char *UsedPrefixStart = UsedPrefix.data();

    // Skip the buffer to the end.
    Buffer = Buffer.drop_front(UsedPrefix.size() + CheckTypeSize(CheckTy));

    // Complain about useful-looking but unsupported suffixes.
    if (CheckTy == Check::CheckBadNot) {
      SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()), SourceMgr::DK_Error,
                      "unsupported -NOT combo on prefix '" + UsedPrefix + "'");
      return true;
    }

    // Okay, we found the prefix, yay. Remember the rest of the line, but ignore
    // leading whitespace.
    if (!(NoCanonicalizeWhiteSpace && MatchFullLines))
      Buffer = Buffer.substr(Buffer.find_first_not_of(" \t"));

    // Scan ahead to the end of line.
    size_t EOL = Buffer.find_first_of("\n\r");

    // Remember the location of the start of the pattern, for diagnostics.
    SMLoc PatternLoc = SMLoc::getFromPointer(Buffer.data());

    // Parse the pattern.
    Pattern P(CheckTy);
    if (P.ParsePattern(Buffer.substr(0, EOL), UsedPrefix, SM, LineNumber))
      return true;

    // Verify that CHECK-LABEL lines do not define or use variables
    if ((CheckTy == Check::CheckLabel) && P.hasVariable()) {
      SM.PrintMessage(
          SMLoc::getFromPointer(UsedPrefixStart), SourceMgr::DK_Error,
          "found '" + UsedPrefix + "-LABEL:'"
                                   " with variable definition or use");
      return true;
    }

    Buffer = Buffer.substr(EOL);

    // Verify that CHECK-NEXT lines have at least one CHECK line before them.
    if ((CheckTy == Check::CheckNext || CheckTy == Check::CheckSame) &&
        CheckStrings.empty()) {
      StringRef Type = CheckTy == Check::CheckNext ? "NEXT" : "SAME";
      SM.PrintMessage(SMLoc::getFromPointer(UsedPrefixStart),
                      SourceMgr::DK_Error,
                      "found '" + UsedPrefix + "-" + Type +
                          "' without previous '" + UsedPrefix + ": line");
      return true;
    }

    // Handle CHECK-DAG/-NOT.
    if (CheckTy == Check::CheckDAG || CheckTy == Check::CheckNot) {
      DagNotMatches.push_back(P);
      continue;
    }

    // Okay, add the string we captured to the output vector and move on.
    CheckStrings.emplace_back(P, UsedPrefix, PatternLoc);
    std::swap(DagNotMatches, CheckStrings.back().DagNotStrings);
    DagNotMatches = ImplicitNegativeChecks;
  }

  // Add an EOF pattern for any trailing CHECK-DAG/-NOTs, and use the first
  // prefix as a filler for the error message.
  if (!DagNotMatches.empty()) {
    CheckStrings.emplace_back(Pattern(Check::CheckEOF), *CheckPrefixes.begin(),
                              SMLoc::getFromPointer(Buffer.data()));
    std::swap(DagNotMatches, CheckStrings.back().DagNotStrings);
  }

  if (CheckStrings.empty()) {
    errs() << "error: no check strings found with prefix"
           << (CheckPrefixes.size() > 1 ? "es " : " ");
    prefix_iterator I = CheckPrefixes.begin();
    prefix_iterator E = CheckPrefixes.end();
    if (I != E) {
      errs() << "\'" << *I << ":'";
      ++I;
    }
    for (; I != E; ++I)
      errs() << ", \'" << *I << ":'";

    errs() << '\n';
    return true;
  }

  return false;
}

static void PrintCheckFailed(const SourceMgr &SM, SMLoc Loc, const Pattern &Pat,
                             StringRef Buffer,
                             StringMap<StringRef> &VariableTable) {
  // Otherwise, we have an error, emit an error message.
  SM.PrintMessage(Loc, SourceMgr::DK_Error,
                  "expected string not found in input");

  // Print the "scanning from here" line.  If the current position is at the
  // end of a line, advance to the start of the next line.
  Buffer = Buffer.substr(Buffer.find_first_not_of(" \t\n\r"));

  SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()), SourceMgr::DK_Note,
                  "scanning from here");

  // Allow the pattern to print additional information if desired.
  Pat.PrintFailureInfo(SM, Buffer, VariableTable);
}

static void PrintCheckFailed(const SourceMgr &SM, const CheckString &CheckStr,
                             StringRef Buffer,
                             StringMap<StringRef> &VariableTable) {
  PrintCheckFailed(SM, CheckStr.Loc, CheckStr.Pat, Buffer, VariableTable);
}

/// Count the number of newlines in the specified range.
static unsigned CountNumNewlinesBetween(StringRef Range,
                                        const char *&FirstNewLine) {
  unsigned NumNewLines = 0;
  while (1) {
    // Scan for newline.
    Range = Range.substr(Range.find_first_of("\n\r"));
    if (Range.empty())
      return NumNewLines;

    ++NumNewLines;

    // Handle \n\r and \r\n as a single newline.
    if (Range.size() > 1 && (Range[1] == '\n' || Range[1] == '\r') &&
        (Range[0] != Range[1]))
      Range = Range.substr(1);
    Range = Range.substr(1);

    if (NumNewLines == 1)
      FirstNewLine = Range.begin();
  }
}

/// Match check string and its "not strings" and/or "dag strings".
size_t CheckString::Check(const SourceMgr &SM, StringRef Buffer,
                          bool IsLabelScanMode, size_t &MatchLen,
                          StringMap<StringRef> &VariableTable) const {
  size_t LastPos = 0;
  std::vector<const Pattern *> NotStrings;

  // IsLabelScanMode is true when we are scanning forward to find CHECK-LABEL
  // bounds; we have not processed variable definitions within the bounded block
  // yet so cannot handle any final CHECK-DAG yet; this is handled when going
  // over the block again (including the last CHECK-LABEL) in normal mode.
  if (!IsLabelScanMode) {
    // Match "dag strings" (with mixed "not strings" if any).
    LastPos = CheckDag(SM, Buffer, NotStrings, VariableTable);
    if (LastPos == StringRef::npos)
      return StringRef::npos;
  }

  // Match itself from the last position after matching CHECK-DAG.
  StringRef MatchBuffer = Buffer.substr(LastPos);
  size_t MatchPos = Pat.Match(MatchBuffer, MatchLen, VariableTable);
  if (MatchPos == StringRef::npos) {
    PrintCheckFailed(SM, *this, MatchBuffer, VariableTable);
    return StringRef::npos;
  }

  // Similar to the above, in "label-scan mode" we can't yet handle CHECK-NEXT
  // or CHECK-NOT
  if (!IsLabelScanMode) {
    StringRef SkippedRegion = Buffer.substr(LastPos, MatchPos);

    // If this check is a "CHECK-NEXT", verify that the previous match was on
    // the previous line (i.e. that there is one newline between them).
    if (CheckNext(SM, SkippedRegion))
      return StringRef::npos;

    // If this check is a "CHECK-SAME", verify that the previous match was on
    // the same line (i.e. that there is no newline between them).
    if (CheckSame(SM, SkippedRegion))
      return StringRef::npos;

    // If this match had "not strings", verify that they don't exist in the
    // skipped region.
    if (CheckNot(SM, SkippedRegion, NotStrings, VariableTable))
      return StringRef::npos;
  }

  return LastPos + MatchPos;
}

/// Verify there is a single line in the given buffer.
bool CheckString::CheckNext(const SourceMgr &SM, StringRef Buffer) const {
  if (Pat.getCheckTy() != Check::CheckNext)
    return false;

  // Count the number of newlines between the previous match and this one.
  assert(Buffer.data() !=
             SM.getMemoryBuffer(SM.FindBufferContainingLoc(
                                    SMLoc::getFromPointer(Buffer.data())))
                 ->getBufferStart() &&
         "CHECK-NEXT can't be the first check in a file");

  const char *FirstNewLine = nullptr;
  unsigned NumNewLines = CountNumNewlinesBetween(Buffer, FirstNewLine);

  if (NumNewLines == 0) {
    SM.PrintMessage(Loc, SourceMgr::DK_Error,
                    Prefix + "-NEXT: is on the same line as previous match");
    SM.PrintMessage(SMLoc::getFromPointer(Buffer.end()), SourceMgr::DK_Note,
                    "'next' match was here");
    SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()), SourceMgr::DK_Note,
                    "previous match ended here");
    return true;
  }

  if (NumNewLines != 1) {
    SM.PrintMessage(Loc, SourceMgr::DK_Error,
                    Prefix +
                        "-NEXT: is not on the line after the previous match");
    SM.PrintMessage(SMLoc::getFromPointer(Buffer.end()), SourceMgr::DK_Note,
                    "'next' match was here");
    SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()), SourceMgr::DK_Note,
                    "previous match ended here");
    SM.PrintMessage(SMLoc::getFromPointer(FirstNewLine), SourceMgr::DK_Note,
                    "non-matching line after previous match is here");
    return true;
  }

  return false;
}

/// Verify there is no newline in the given buffer.
bool CheckString::CheckSame(const SourceMgr &SM, StringRef Buffer) const {
  if (Pat.getCheckTy() != Check::CheckSame)
    return false;

  // Count the number of newlines between the previous match and this one.
  assert(Buffer.data() !=
             SM.getMemoryBuffer(SM.FindBufferContainingLoc(
                                    SMLoc::getFromPointer(Buffer.data())))
                 ->getBufferStart() &&
         "CHECK-SAME can't be the first check in a file");

  const char *FirstNewLine = nullptr;
  unsigned NumNewLines = CountNumNewlinesBetween(Buffer, FirstNewLine);

  if (NumNewLines != 0) {
    SM.PrintMessage(Loc, SourceMgr::DK_Error,
                    Prefix +
                        "-SAME: is not on the same line as the previous match");
    SM.PrintMessage(SMLoc::getFromPointer(Buffer.end()), SourceMgr::DK_Note,
                    "'next' match was here");
    SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()), SourceMgr::DK_Note,
                    "previous match ended here");
    return true;
  }

  return false;
}

/// Verify there's no "not strings" in the given buffer.
bool CheckString::CheckNot(const SourceMgr &SM, StringRef Buffer,
                           const std::vector<const Pattern *> &NotStrings,
                           StringMap<StringRef> &VariableTable) const {
  for (const Pattern *Pat : NotStrings) {
    assert((Pat->getCheckTy() == Check::CheckNot) && "Expect CHECK-NOT!");

    size_t MatchLen = 0;
    size_t Pos = Pat->Match(Buffer, MatchLen, VariableTable);

    if (Pos == StringRef::npos)
      continue;

    SM.PrintMessage(SMLoc::getFromPointer(Buffer.data() + Pos),
                    SourceMgr::DK_Error, Prefix + "-NOT: string occurred!");
    SM.PrintMessage(Pat->getLoc(), SourceMgr::DK_Note,
                    Prefix + "-NOT: pattern specified here");
    return true;
  }

  return false;
}

/// Match "dag strings" and their mixed "not strings".
size_t CheckString::CheckDag(const SourceMgr &SM, StringRef Buffer,
                             std::vector<const Pattern *> &NotStrings,
                             StringMap<StringRef> &VariableTable) const {
  if (DagNotStrings.empty())
    return 0;

  size_t LastPos = 0;
  size_t StartPos = LastPos;

  for (const Pattern &Pat : DagNotStrings) {
    assert((Pat.getCheckTy() == Check::CheckDAG ||
            Pat.getCheckTy() == Check::CheckNot) &&
           "Invalid CHECK-DAG or CHECK-NOT!");

    if (Pat.getCheckTy() == Check::CheckNot) {
      NotStrings.push_back(&Pat);
      continue;
    }

    assert((Pat.getCheckTy() == Check::CheckDAG) && "Expect CHECK-DAG!");

    size_t MatchLen = 0, MatchPos;

    // CHECK-DAG always matches from the start.
    StringRef MatchBuffer = Buffer.substr(StartPos);
    MatchPos = Pat.Match(MatchBuffer, MatchLen, VariableTable);
    // With a group of CHECK-DAGs, a single mismatching means the match on
    // that group of CHECK-DAGs fails immediately.
    if (MatchPos == StringRef::npos) {
      PrintCheckFailed(SM, Pat.getLoc(), Pat, MatchBuffer, VariableTable);
      return StringRef::npos;
    }
    // Re-calc it as the offset relative to the start of the original string.
    MatchPos += StartPos;

    if (!NotStrings.empty()) {
      if (MatchPos < LastPos) {
        // Reordered?
        SM.PrintMessage(SMLoc::getFromPointer(Buffer.data() + MatchPos),
                        SourceMgr::DK_Error,
                        Prefix + "-DAG: found a match of CHECK-DAG"
                                 " reordering across a CHECK-NOT");
        SM.PrintMessage(SMLoc::getFromPointer(Buffer.data() + LastPos),
                        SourceMgr::DK_Note,
                        Prefix + "-DAG: the farthest match of CHECK-DAG"
                                 " is found here");
        SM.PrintMessage(NotStrings[0]->getLoc(), SourceMgr::DK_Note,
                        Prefix + "-NOT: the crossed pattern specified"
                                 " here");
        SM.PrintMessage(Pat.getLoc(), SourceMgr::DK_Note,
                        Prefix + "-DAG: the reordered pattern specified"
                                 " here");
        return StringRef::npos;
      }
      // All subsequent CHECK-DAGs should be matched from the farthest
      // position of all precedent CHECK-DAGs (including this one.)
      StartPos = LastPos;
      // If there's CHECK-NOTs between two CHECK-DAGs or from CHECK to
      // CHECK-DAG, verify that there's no 'not' strings occurred in that
      // region.
      StringRef SkippedRegion = Buffer.substr(LastPos, MatchPos);
      if (CheckNot(SM, SkippedRegion, NotStrings, VariableTable))
        return StringRef::npos;
      // Clear "not strings".
      NotStrings.clear();
    }

    // Update the last position with CHECK-DAG matches.
    LastPos = std::max(MatchPos + MatchLen, LastPos);
  }

  return LastPos;
}

// A check prefix must contain only alphanumeric, hyphens and underscores.
static bool ValidateCheckPrefix(StringRef CheckPrefix) {
  Regex Validator("^[a-zA-Z0-9_-]*$");
  return Validator.match(CheckPrefix);
}

static bool ValidateCheckPrefixes() {
  StringSet<> PrefixSet;

  for (StringRef Prefix : CheckPrefixes) {
    // Reject empty prefixes.
    if (Prefix == "")
      return false;

    if (!PrefixSet.insert(Prefix).second)
      return false;

    if (!ValidateCheckPrefix(Prefix))
      return false;
  }

  return true;
}

// Combines the check prefixes into a single regex so that we can efficiently
// scan for any of the set.
//
// The semantics are that the longest-match wins which matches our regex
// library.
static Regex buildCheckPrefixRegex() {
  // I don't think there's a way to specify an initial value for cl::list,
  // so if nothing was specified, add the default
  if (CheckPrefixes.empty())
    CheckPrefixes.push_back("CHECK");

  // We already validated the contents of CheckPrefixes so just concatenate
  // them as alternatives.
  SmallString<32> PrefixRegexStr;
  for (StringRef Prefix : CheckPrefixes) {
    if (Prefix != CheckPrefixes.front())
      PrefixRegexStr.push_back('|');

    PrefixRegexStr.append(Prefix);
  }

  return Regex(PrefixRegexStr);
}

static void DumpCommandLine(int argc, char **argv) {
  errs() << "FileCheck command line: ";
  for (int I = 0; I < argc; I++)
    errs() << " " << argv[I];
  errs() << "\n";
}

// Remove local variables from \p VariableTable. Global variables
// (start with '$') are preserved.
static void ClearLocalVars(StringMap<StringRef> &VariableTable) {
  SmallVector<StringRef, 16> LocalVars;
  for (const auto &Var : VariableTable)
    if (Var.first()[0] != '$')
      LocalVars.push_back(Var.first());

  for (const auto &Var : LocalVars)
    VariableTable.erase(Var);
}

/// Check the input to FileCheck provided in the \p Buffer against the \p
/// CheckStrings read from the check file.
///
/// Returns false if the input fails to satisfy the checks.
bool CheckInput(SourceMgr &SM, StringRef Buffer,
                ArrayRef<CheckString> CheckStrings) {
  bool ChecksFailed = false;

  /// VariableTable - This holds all the current filecheck variables.
  StringMap<StringRef> VariableTable;

  unsigned i = 0, j = 0, e = CheckStrings.size();
  while (true) {
    StringRef CheckRegion;
    if (j == e) {
      CheckRegion = Buffer;
    } else {
      const CheckString &CheckLabelStr = CheckStrings[j];
      if (CheckLabelStr.Pat.getCheckTy() != Check::CheckLabel) {
        ++j;
        continue;
      }

      // Scan to next CHECK-LABEL match, ignoring CHECK-NOT and CHECK-DAG
      size_t MatchLabelLen = 0;
      size_t MatchLabelPos =
          CheckLabelStr.Check(SM, Buffer, true, MatchLabelLen, VariableTable);
      if (MatchLabelPos == StringRef::npos)
        // Immediately bail of CHECK-LABEL fails, nothing else we can do.
        return false;

      CheckRegion = Buffer.substr(0, MatchLabelPos + MatchLabelLen);
      Buffer = Buffer.substr(MatchLabelPos + MatchLabelLen);
      ++j;
    }

    if (EnableVarScope)
      ClearLocalVars(VariableTable);

    for (; i != j; ++i) {
      const CheckString &CheckStr = CheckStrings[i];

      // Check each string within the scanned region, including a second check
      // of any final CHECK-LABEL (to verify CHECK-NOT and CHECK-DAG)
      size_t MatchLen = 0;
      size_t MatchPos =
          CheckStr.Check(SM, CheckRegion, false, MatchLen, VariableTable);

      if (MatchPos == StringRef::npos) {
        ChecksFailed = true;
        i = j;
        break;
      }

      CheckRegion = CheckRegion.substr(MatchPos + MatchLen);
    }

    if (j == e)
      break;
  }

  // Success if no checks failed.
  return !ChecksFailed;
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  if (!ValidateCheckPrefixes()) {
    errs() << "Supplied check-prefix is invalid! Prefixes must be unique and "
              "start with a letter and contain only alphanumeric characters, "
              "hyphens and underscores\n";
    return 2;
  }

  Regex PrefixRE = buildCheckPrefixRegex();
  std::string REError;
  if (!PrefixRE.isValid(REError)) {
    errs() << "Unable to combine check-prefix strings into a prefix regular "
              "expression! This is likely a bug in FileCheck's verification of "
              "the check-prefix strings. Regular expression parsing failed "
              "with the following error: "
           << REError << "\n";
    return 2;
  }

  SourceMgr SM;

  // Read the expected strings from the check file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> CheckFileOrErr =
      MemoryBuffer::getFileOrSTDIN(CheckFilename);
  if (std::error_code EC = CheckFileOrErr.getError()) {
    errs() << "Could not open check file '" << CheckFilename
           << "': " << EC.message() << '\n';
    return 2;
  }
  MemoryBuffer &CheckFile = *CheckFileOrErr.get();

  SmallString<4096> CheckFileBuffer;
  StringRef CheckFileText = CanonicalizeFile(CheckFile, CheckFileBuffer);

  SM.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(
                            CheckFileText, CheckFile.getBufferIdentifier()),
                        SMLoc());

  std::vector<CheckString> CheckStrings;
  if (ReadCheckFile(SM, CheckFileText, PrefixRE, CheckStrings))
    return 2;

  // Open the file to check and add it to SourceMgr.
  ErrorOr<std::unique_ptr<MemoryBuffer>> InputFileOrErr =
      MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (std::error_code EC = InputFileOrErr.getError()) {
    errs() << "Could not open input file '" << InputFilename
           << "': " << EC.message() << '\n';
    return 2;
  }
  MemoryBuffer &InputFile = *InputFileOrErr.get();

  if (InputFile.getBufferSize() == 0 && !AllowEmptyInput) {
    errs() << "FileCheck error: '" << InputFilename << "' is empty.\n";
    DumpCommandLine(argc, argv);
    return 2;
  }

  SmallString<4096> InputFileBuffer;
  StringRef InputFileText = CanonicalizeFile(InputFile, InputFileBuffer);

  SM.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(
                            InputFileText, InputFile.getBufferIdentifier()),
                        SMLoc());

  return CheckInput(SM, InputFileText, CheckStrings) ? EXIT_SUCCESS : 1;
}
