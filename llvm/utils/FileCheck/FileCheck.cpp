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
// This program exits with an error status of 2 on error, exit status of 0 if
// the file matched the expected contents, and exit status of 1 if it did not
// contain the expected contents.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include <algorithm>
using namespace llvm;

static cl::opt<std::string>
CheckFilename(cl::Positional, cl::desc("<check-file>"), cl::Required);

static cl::opt<std::string>
InputFilename("input-file", cl::desc("File to check (defaults to stdin)"),
              cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string>
CheckPrefix("check-prefix", cl::init("CHECK"),
            cl::desc("Prefix to use from check file (defaults to 'CHECK')"));

static cl::opt<bool>
NoCanonicalizeWhiteSpace("strict-whitespace",
              cl::desc("Do not treat all horizontal whitespace as equivalent"));

//===----------------------------------------------------------------------===//
// Pattern Handling Code.
//===----------------------------------------------------------------------===//

class Pattern {
  SMLoc PatternLoc;

  /// MatchEOF - When set, this pattern only matches the end of file. This is
  /// used for trailing CHECK-NOTs.
  bool MatchEOF;

  /// FixedStr - If non-empty, this pattern is a fixed string match with the
  /// specified fixed string.
  StringRef FixedStr;

  /// RegEx - If non-empty, this is a regex pattern.
  std::string RegExStr;

  /// VariableUses - Entries in this vector map to uses of a variable in the
  /// pattern, e.g. "foo[[bar]]baz".  In this case, the RegExStr will contain
  /// "foobaz" and we'll get an entry in this vector that tells us to insert the
  /// value of bar at offset 3.
  std::vector<std::pair<StringRef, unsigned> > VariableUses;

  /// VariableDefs - Entries in this vector map to definitions of a variable in
  /// the pattern, e.g. "foo[[bar:.*]]baz".  In this case, the RegExStr will
  /// contain "foo(.*)baz" and VariableDefs will contain the pair "bar",1.  The
  /// index indicates what parenthesized value captures the variable value.
  std::vector<std::pair<StringRef, unsigned> > VariableDefs;

public:

  Pattern(bool matchEOF = false) : MatchEOF(matchEOF) { }

  bool ParsePattern(StringRef PatternStr, SourceMgr &SM);

  /// Match - Match the pattern string against the input buffer Buffer.  This
  /// returns the position that is matched or npos if there is no match.  If
  /// there is a match, the size of the matched string is returned in MatchLen.
  ///
  /// The VariableTable StringMap provides the current values of filecheck
  /// variables and is updated if this match defines new values.
  size_t Match(StringRef Buffer, size_t &MatchLen,
               StringMap<StringRef> &VariableTable) const;

  /// PrintFailureInfo - Print additional information about a failure to match
  /// involving this pattern.
  void PrintFailureInfo(const SourceMgr &SM, StringRef Buffer,
                        const StringMap<StringRef> &VariableTable) const;

private:
  static void AddFixedStringToRegEx(StringRef FixedStr, std::string &TheStr);
  bool AddRegExToRegEx(StringRef RegExStr, unsigned &CurParen, SourceMgr &SM);

  /// ComputeMatchDistance - Compute an arbitrary estimate for the quality of
  /// matching this pattern at the start of \arg Buffer; a distance of zero
  /// should correspond to a perfect match.
  unsigned ComputeMatchDistance(StringRef Buffer,
                               const StringMap<StringRef> &VariableTable) const;
};


bool Pattern::ParsePattern(StringRef PatternStr, SourceMgr &SM) {
  PatternLoc = SMLoc::getFromPointer(PatternStr.data());

  // Ignore trailing whitespace.
  while (!PatternStr.empty() &&
         (PatternStr.back() == ' ' || PatternStr.back() == '\t'))
    PatternStr = PatternStr.substr(0, PatternStr.size()-1);

  // Check that there is something on the line.
  if (PatternStr.empty()) {
    SM.PrintMessage(PatternLoc, SourceMgr::DK_Error,
                    "found empty check string with prefix '" +
                    CheckPrefix+":'");
    return true;
  }

  // Check to see if this is a fixed string, or if it has regex pieces.
  if (PatternStr.size() < 2 ||
      (PatternStr.find("{{") == StringRef::npos &&
       PatternStr.find("[[") == StringRef::npos)) {
    FixedStr = PatternStr;
    return false;
  }

  // Paren value #0 is for the fully matched string.  Any new parenthesized
  // values add from there.
  unsigned CurParen = 1;

  // Otherwise, there is at least one regex piece.  Build up the regex pattern
  // by escaping scary characters in fixed strings, building up one big regex.
  while (!PatternStr.empty()) {
    // RegEx matches.
    if (PatternStr.startswith("{{")) {

      // Otherwise, this is the start of a regex match.  Scan for the }}.
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

      if (AddRegExToRegEx(PatternStr.substr(2, End-2), CurParen, SM))
        return true;
      RegExStr += ')';

      PatternStr = PatternStr.substr(End+2);
      continue;
    }

    // Named RegEx matches.  These are of two forms: [[foo:.*]] which matches .*
    // (or some other regex) and assigns it to the FileCheck variable 'foo'. The
    // second form is [[foo]] which is a reference to foo.  The variable name
    // itself must be of the form "[a-zA-Z_][0-9a-zA-Z_]*", otherwise we reject
    // it.  This is to catch some common errors.
    if (PatternStr.startswith("[[")) {
      // Verify that it is terminated properly.
      size_t End = PatternStr.find("]]");
      if (End == StringRef::npos) {
        SM.PrintMessage(SMLoc::getFromPointer(PatternStr.data()),
                        SourceMgr::DK_Error,
                        "invalid named regex reference, no ]] found");
        return true;
      }

      StringRef MatchStr = PatternStr.substr(2, End-2);
      PatternStr = PatternStr.substr(End+2);

      // Get the regex name (e.g. "foo").
      size_t NameEnd = MatchStr.find(':');
      StringRef Name = MatchStr.substr(0, NameEnd);

      if (Name.empty()) {
        SM.PrintMessage(SMLoc::getFromPointer(Name.data()), SourceMgr::DK_Error,
                        "invalid name in named regex: empty name");
        return true;
      }

      // Verify that the name is well formed.
      for (unsigned i = 0, e = Name.size(); i != e; ++i)
        if (Name[i] != '_' && !isalnum(Name[i])) {
          SM.PrintMessage(SMLoc::getFromPointer(Name.data()+i),
                          SourceMgr::DK_Error, "invalid name in named regex");
          return true;
        }

      // Name can't start with a digit.
      if (isdigit(Name[0])) {
        SM.PrintMessage(SMLoc::getFromPointer(Name.data()), SourceMgr::DK_Error,
                        "invalid name in named regex");
        return true;
      }

      // Handle [[foo]].
      if (NameEnd == StringRef::npos) {
        VariableUses.push_back(std::make_pair(Name, RegExStr.size()));
        continue;
      }

      // Handle [[foo:.*]].
      VariableDefs.push_back(std::make_pair(Name, CurParen));
      RegExStr += '(';
      ++CurParen;

      if (AddRegExToRegEx(MatchStr.substr(NameEnd+1), CurParen, SM))
        return true;

      RegExStr += ')';
    }

    // Handle fixed string matches.
    // Find the end, which is the start of the next regex.
    size_t FixedMatchEnd = PatternStr.find("{{");
    FixedMatchEnd = std::min(FixedMatchEnd, PatternStr.find("[["));
    AddFixedStringToRegEx(PatternStr.substr(0, FixedMatchEnd), RegExStr);
    PatternStr = PatternStr.substr(FixedMatchEnd);
    continue;
  }

  return false;
}

void Pattern::AddFixedStringToRegEx(StringRef FixedStr, std::string &TheStr) {
  // Add the characters from FixedStr to the regex, escaping as needed.  This
  // avoids "leaning toothpicks" in common patterns.
  for (unsigned i = 0, e = FixedStr.size(); i != e; ++i) {
    switch (FixedStr[i]) {
    // These are the special characters matched in "p_ere_exp".
    case '(':
    case ')':
    case '^':
    case '$':
    case '|':
    case '*':
    case '+':
    case '?':
    case '.':
    case '[':
    case '\\':
    case '{':
      TheStr += '\\';
      // FALL THROUGH.
    default:
      TheStr += FixedStr[i];
      break;
    }
  }
}

bool Pattern::AddRegExToRegEx(StringRef RegexStr, unsigned &CurParen,
                              SourceMgr &SM) {
  Regex R(RegexStr);
  std::string Error;
  if (!R.isValid(Error)) {
    SM.PrintMessage(SMLoc::getFromPointer(RegexStr.data()), SourceMgr::DK_Error,
                    "invalid regex: " + Error);
    return true;
  }

  RegExStr += RegexStr.str();
  CurParen += R.getNumMatches();
  return false;
}

/// Match - Match the pattern string against the input buffer Buffer.  This
/// returns the position that is matched or npos if there is no match.  If
/// there is a match, the size of the matched string is returned in MatchLen.
size_t Pattern::Match(StringRef Buffer, size_t &MatchLen,
                      StringMap<StringRef> &VariableTable) const {
  // If this is the EOF pattern, match it immediately.
  if (MatchEOF) {
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
    for (unsigned i = 0, e = VariableUses.size(); i != e; ++i) {
      StringMap<StringRef>::iterator it =
        VariableTable.find(VariableUses[i].first);
      // If the variable is undefined, return an error.
      if (it == VariableTable.end())
        return StringRef::npos;

      // Look up the value and escape it so that we can plop it into the regex.
      std::string Value;
      AddFixedStringToRegEx(it->second, Value);

      // Plop it into the regex at the adjusted offset.
      TmpStr.insert(TmpStr.begin()+VariableUses[i].second+InsertOffset,
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
  for (unsigned i = 0, e = VariableDefs.size(); i != e; ++i) {
    assert(VariableDefs[i].second < MatchInfo.size() &&
           "Internal paren error");
    VariableTable[VariableDefs[i].first] = MatchInfo[VariableDefs[i].second];
  }

  MatchLen = FullMatch.size();
  return FullMatch.data()-Buffer.data();
}

unsigned Pattern::ComputeMatchDistance(StringRef Buffer,
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

void Pattern::PrintFailureInfo(const SourceMgr &SM, StringRef Buffer,
                               const StringMap<StringRef> &VariableTable) const{
  // If this was a regular expression using variables, print the current
  // variable values.
  if (!VariableUses.empty()) {
    for (unsigned i = 0, e = VariableUses.size(); i != e; ++i) {
      StringRef Var = VariableUses[i].first;
      StringMap<StringRef>::const_iterator it = VariableTable.find(Var);
      SmallString<256> Msg;
      raw_svector_ostream OS(Msg);

      // Check for undefined variable references.
      if (it == VariableTable.end()) {
        OS << "uses undefined variable \"";
        OS.write_escaped(Var) << "\"";;
      } else {
        OS << "with variable \"";
        OS.write_escaped(Var) << "\" equal to \"";
        OS.write_escaped(it->second) << "\"";
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

//===----------------------------------------------------------------------===//
// Check Strings.
//===----------------------------------------------------------------------===//

/// CheckString - This is a check that we found in the input file.
struct CheckString {
  /// Pat - The pattern to match.
  Pattern Pat;

  /// Loc - The location in the match file that the check string was specified.
  SMLoc Loc;

  /// IsCheckNext - This is true if this is a CHECK-NEXT: directive (as opposed
  /// to a CHECK: directive.
  bool IsCheckNext;

  /// NotStrings - These are all of the strings that are disallowed from
  /// occurring between this match string and the previous one (or start of
  /// file).
  std::vector<std::pair<SMLoc, Pattern> > NotStrings;

  CheckString(const Pattern &P, SMLoc L, bool isCheckNext)
    : Pat(P), Loc(L), IsCheckNext(isCheckNext) {}
};

/// CanonicalizeInputFile - Remove duplicate horizontal space from the specified
/// memory buffer, free it, and return a new one.
static MemoryBuffer *CanonicalizeInputFile(MemoryBuffer *MB) {
  SmallString<128> NewFile;
  NewFile.reserve(MB->getBufferSize());

  for (const char *Ptr = MB->getBufferStart(), *End = MB->getBufferEnd();
       Ptr != End; ++Ptr) {
    // Eliminate trailing dosish \r.
    if (Ptr <= End - 2 && Ptr[0] == '\r' && Ptr[1] == '\n') {
      continue;
    }

    // If current char is not a horizontal whitespace, dump it to output as is.
    if (*Ptr != ' ' && *Ptr != '\t') {
      NewFile.push_back(*Ptr);
      continue;
    }

    // Otherwise, add one space and advance over neighboring space.
    NewFile.push_back(' ');
    while (Ptr+1 != End &&
           (Ptr[1] == ' ' || Ptr[1] == '\t'))
      ++Ptr;
  }

  // Free the old buffer and return a new one.
  MemoryBuffer *MB2 =
    MemoryBuffer::getMemBufferCopy(NewFile.str(), MB->getBufferIdentifier());

  delete MB;
  return MB2;
}


/// ReadCheckFile - Read the check file, which specifies the sequence of
/// expected strings.  The strings are added to the CheckStrings vector.
static bool ReadCheckFile(SourceMgr &SM,
                          std::vector<CheckString> &CheckStrings) {
  // Open the check file, and tell SourceMgr about it.
  OwningPtr<MemoryBuffer> File;
  if (error_code ec =
        MemoryBuffer::getFileOrSTDIN(CheckFilename.c_str(), File)) {
    errs() << "Could not open check file '" << CheckFilename << "': "
           << ec.message() << '\n';
    return true;
  }
  MemoryBuffer *F = File.take();

  // If we want to canonicalize whitespace, strip excess whitespace from the
  // buffer containing the CHECK lines.
  if (!NoCanonicalizeWhiteSpace)
    F = CanonicalizeInputFile(F);

  SM.AddNewSourceBuffer(F, SMLoc());

  // Find all instances of CheckPrefix followed by : in the file.
  StringRef Buffer = F->getBuffer();

  std::vector<std::pair<SMLoc, Pattern> > NotMatches;

  while (1) {
    // See if Prefix occurs in the memory buffer.
    Buffer = Buffer.substr(Buffer.find(CheckPrefix));

    // If we didn't find a match, we're done.
    if (Buffer.empty())
      break;

    const char *CheckPrefixStart = Buffer.data();

    // When we find a check prefix, keep track of whether we find CHECK: or
    // CHECK-NEXT:
    bool IsCheckNext = false, IsCheckNot = false;

    // Verify that the : is present after the prefix.
    if (Buffer[CheckPrefix.size()] == ':') {
      Buffer = Buffer.substr(CheckPrefix.size()+1);
    } else if (Buffer.size() > CheckPrefix.size()+6 &&
               memcmp(Buffer.data()+CheckPrefix.size(), "-NEXT:", 6) == 0) {
      Buffer = Buffer.substr(CheckPrefix.size()+6);
      IsCheckNext = true;
    } else if (Buffer.size() > CheckPrefix.size()+5 &&
               memcmp(Buffer.data()+CheckPrefix.size(), "-NOT:", 5) == 0) {
      Buffer = Buffer.substr(CheckPrefix.size()+5);
      IsCheckNot = true;
    } else {
      Buffer = Buffer.substr(1);
      continue;
    }

    // Okay, we found the prefix, yay.  Remember the rest of the line, but
    // ignore leading and trailing whitespace.
    Buffer = Buffer.substr(Buffer.find_first_not_of(" \t"));

    // Scan ahead to the end of line.
    size_t EOL = Buffer.find_first_of("\n\r");

    // Remember the location of the start of the pattern, for diagnostics.
    SMLoc PatternLoc = SMLoc::getFromPointer(Buffer.data());

    // Parse the pattern.
    Pattern P;
    if (P.ParsePattern(Buffer.substr(0, EOL), SM))
      return true;

    Buffer = Buffer.substr(EOL);


    // Verify that CHECK-NEXT lines have at least one CHECK line before them.
    if (IsCheckNext && CheckStrings.empty()) {
      SM.PrintMessage(SMLoc::getFromPointer(CheckPrefixStart),
                      SourceMgr::DK_Error,
                      "found '"+CheckPrefix+"-NEXT:' without previous '"+
                      CheckPrefix+ ": line");
      return true;
    }

    // Handle CHECK-NOT.
    if (IsCheckNot) {
      NotMatches.push_back(std::make_pair(SMLoc::getFromPointer(Buffer.data()),
                                          P));
      continue;
    }


    // Okay, add the string we captured to the output vector and move on.
    CheckStrings.push_back(CheckString(P,
                                       PatternLoc,
                                       IsCheckNext));
    std::swap(NotMatches, CheckStrings.back().NotStrings);
  }

  // Add an EOF pattern for any trailing CHECK-NOTs.
  if (!NotMatches.empty()) {
    CheckStrings.push_back(CheckString(Pattern(true),
                                       SMLoc::getFromPointer(Buffer.data()),
                                       false));
    std::swap(NotMatches, CheckStrings.back().NotStrings);
  }

  if (CheckStrings.empty()) {
    errs() << "error: no check strings found with prefix '" << CheckPrefix
           << ":'\n";
    return true;
  }

  return false;
}

static void PrintCheckFailed(const SourceMgr &SM, const CheckString &CheckStr,
                             StringRef Buffer,
                             StringMap<StringRef> &VariableTable) {
  // Otherwise, we have an error, emit an error message.
  SM.PrintMessage(CheckStr.Loc, SourceMgr::DK_Error,
                  "expected string not found in input");

  // Print the "scanning from here" line.  If the current position is at the
  // end of a line, advance to the start of the next line.
  Buffer = Buffer.substr(Buffer.find_first_not_of(" \t\n\r"));

  SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()), SourceMgr::DK_Note,
                  "scanning from here");

  // Allow the pattern to print additional information if desired.
  CheckStr.Pat.PrintFailureInfo(SM, Buffer, VariableTable);
}

/// CountNumNewlinesBetween - Count the number of newlines in the specified
/// range.
static unsigned CountNumNewlinesBetween(StringRef Range) {
  unsigned NumNewLines = 0;
  while (1) {
    // Scan for newline.
    Range = Range.substr(Range.find_first_of("\n\r"));
    if (Range.empty()) return NumNewLines;

    ++NumNewLines;

    // Handle \n\r and \r\n as a single newline.
    if (Range.size() > 1 &&
        (Range[1] == '\n' || Range[1] == '\r') &&
        (Range[0] != Range[1]))
      Range = Range.substr(1);
    Range = Range.substr(1);
  }
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  SourceMgr SM;

  // Read the expected strings from the check file.
  std::vector<CheckString> CheckStrings;
  if (ReadCheckFile(SM, CheckStrings))
    return 2;

  // Open the file to check and add it to SourceMgr.
  OwningPtr<MemoryBuffer> File;
  if (error_code ec =
        MemoryBuffer::getFileOrSTDIN(InputFilename.c_str(), File)) {
    errs() << "Could not open input file '" << InputFilename << "': "
           << ec.message() << '\n';
    return true;
  }
  MemoryBuffer *F = File.take();

  if (F->getBufferSize() == 0) {
    errs() << "FileCheck error: '" << InputFilename << "' is empty.\n";
    return 1;
  }
  
  // Remove duplicate spaces in the input file if requested.
  if (!NoCanonicalizeWhiteSpace)
    F = CanonicalizeInputFile(F);

  SM.AddNewSourceBuffer(F, SMLoc());

  /// VariableTable - This holds all the current filecheck variables.
  StringMap<StringRef> VariableTable;

  // Check that we have all of the expected strings, in order, in the input
  // file.
  StringRef Buffer = F->getBuffer();

  const char *LastMatch = Buffer.data();

  for (unsigned StrNo = 0, e = CheckStrings.size(); StrNo != e; ++StrNo) {
    const CheckString &CheckStr = CheckStrings[StrNo];

    StringRef SearchFrom = Buffer;

    // Find StrNo in the file.
    size_t MatchLen = 0;
    size_t MatchPos = CheckStr.Pat.Match(Buffer, MatchLen, VariableTable);
    Buffer = Buffer.substr(MatchPos);

    // If we didn't find a match, reject the input.
    if (MatchPos == StringRef::npos) {
      PrintCheckFailed(SM, CheckStr, SearchFrom, VariableTable);
      return 1;
    }

    StringRef SkippedRegion(LastMatch, Buffer.data()-LastMatch);

    // If this check is a "CHECK-NEXT", verify that the previous match was on
    // the previous line (i.e. that there is one newline between them).
    if (CheckStr.IsCheckNext) {
      // Count the number of newlines between the previous match and this one.
      assert(LastMatch != F->getBufferStart() &&
             "CHECK-NEXT can't be the first check in a file");

      unsigned NumNewLines = CountNumNewlinesBetween(SkippedRegion);
      if (NumNewLines == 0) {
        SM.PrintMessage(CheckStr.Loc, SourceMgr::DK_Error,
                    CheckPrefix+"-NEXT: is on the same line as previous match");
        SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()),
                        SourceMgr::DK_Note, "'next' match was here");
        SM.PrintMessage(SMLoc::getFromPointer(LastMatch), SourceMgr::DK_Note,
                        "previous match was here");
        return 1;
      }

      if (NumNewLines != 1) {
        SM.PrintMessage(CheckStr.Loc, SourceMgr::DK_Error, CheckPrefix+
                        "-NEXT: is not on the line after the previous match");
        SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()),
                        SourceMgr::DK_Note, "'next' match was here");
        SM.PrintMessage(SMLoc::getFromPointer(LastMatch), SourceMgr::DK_Note,
                        "previous match was here");
        return 1;
      }
    }

    // If this match had "not strings", verify that they don't exist in the
    // skipped region.
    for (unsigned ChunkNo = 0, e = CheckStr.NotStrings.size();
         ChunkNo != e; ++ChunkNo) {
      size_t MatchLen = 0;
      size_t Pos = CheckStr.NotStrings[ChunkNo].second.Match(SkippedRegion,
                                                             MatchLen,
                                                             VariableTable);
      if (Pos == StringRef::npos) continue;

      SM.PrintMessage(SMLoc::getFromPointer(LastMatch+Pos), SourceMgr::DK_Error,
                      CheckPrefix+"-NOT: string occurred!");
      SM.PrintMessage(CheckStr.NotStrings[ChunkNo].first, SourceMgr::DK_Note,
                      CheckPrefix+"-NOT: pattern specified here");
      return 1;
    }


    // Otherwise, everything is good.  Step over the matched text and remember
    // the position after the match as the end of the last match.
    Buffer = Buffer.substr(MatchLen);
    LastMatch = Buffer.data();
  }

  return 0;
}
