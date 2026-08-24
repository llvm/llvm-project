//===- FileCheck.cpp - Check that File's Contents match what is expected --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "llvm/FileCheck/FileCheck.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
using namespace llvm;

static cl::extrahelp FileCheckOptsEnv(
    "\nOptions are parsed from the environment variable FILECHECK_OPTS and\n"
    "from the command line.\n");

static cl::opt<std::string>
    CheckFilename(cl::Positional, cl::desc("<check-file>"), cl::Optional);

static cl::opt<std::string>
    InputFilename("input-file", cl::desc("File to check (defaults to stdin)"),
                  cl::init("-"), cl::value_desc("filename"));

static cl::list<std::string>
    CheckPrefixes("check-prefixes", cl::CommaSeparated,
                  cl::desc("Comma separated list of prefixes to use from check "
                           "file\n(defaults to 'CHECK')"));
static cl::alias CheckPrefixesAlias("check-prefix", cl::aliasopt(CheckPrefixes),
                                    cl::CommaSeparated, cl::NotHidden,
                                    cl::desc("Alias for -check-prefixes"));

static cl::list<std::string> CommentPrefixes(
    "comment-prefixes", cl::CommaSeparated, cl::Hidden,
    cl::desc("Comma-separated list of comment prefixes to use from check file\n"
             "(defaults to 'COM,RUN'). Please avoid using this feature in\n"
             "LLVM's LIT-based test suites, which should be easier to\n"
             "maintain if they all follow a consistent comment style. This\n"
             "feature is meant for non-LIT test suites using FileCheck."));

static cl::opt<bool> NoCanonicalizeWhiteSpace(
    "strict-whitespace",
    cl::desc("Do not treat all horizontal whitespace as equivalent"));

static cl::opt<bool> IgnoreCase(
    "ignore-case",
    cl::desc("Use case-insensitive matching"));

static cl::list<std::string> ImplicitCheckNot(
    "implicit-check-not",
    cl::desc("Add an implicit negative check with this pattern to every\n"
             "positive check. This can be used to ensure that no instances of\n"
             "this pattern occur which are not matched by a positive pattern"),
    cl::value_desc("pattern"));

static cl::list<std::string>
    GlobalDefines("D", cl::AlwaysPrefix,
                  cl::desc("Define a variable to be used in capture patterns."),
                  cl::value_desc("VAR=VALUE"));

static cl::opt<bool> AllowEmptyInput(
    "allow-empty", cl::init(false),
    cl::desc("Allow the input file to be empty. This is useful when making\n"
             "checks that some error message does not occur, for example."));

static cl::opt<bool> AllowUnusedPrefixes(
    "allow-unused-prefixes",
    cl::desc("Allow prefixes to be specified but not appear in the test."));

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

static cl::opt<bool> AllowDeprecatedDagOverlap(
    "allow-deprecated-dag-overlap", cl::init(false),
    cl::desc("Enable overlapping among matches in a group of consecutive\n"
             "CHECK-DAG directives.  This option is deprecated and is only\n"
             "provided for convenience as old tests are migrated to the new\n"
             "non-overlapping CHECK-DAG implementation.\n"));

static cl::opt<bool> Verbose(
    "v",
    cl::desc("Print directive pattern matches, or add them to the input dump\n"
             "if enabled.\n"));

static cl::opt<bool> VerboseVerbose(
    "vv",
    cl::desc("Print information helpful in diagnosing internal FileCheck\n"
             "issues, or add it to the input dump if enabled.  Implies\n"
             "-v.\n"));

// The order of DumpInputValue members affects their precedence, as documented
// for -dump-input below.
enum DumpInputValue {
  DumpInputNever,
  DumpInputFail,
  DumpInputAlways,
  DumpInputHelp
};

static cl::list<DumpInputValue> DumpInputs(
    "dump-input",
    cl::desc("Dump input to stderr, adding annotations representing\n"
             "currently enabled diagnostics.  When there are multiple\n"
             "occurrences of this option, the <value> that appears earliest\n"
             "in the list below has precedence.  The default is 'fail'.\n"),
    cl::value_desc("mode"),
    cl::values(clEnumValN(DumpInputHelp, "help", "Explain input dump and quit"),
               clEnumValN(DumpInputAlways, "always", "Always dump input"),
               clEnumValN(DumpInputFail, "fail", "Dump input on failure"),
               clEnumValN(DumpInputNever, "never", "Never dump input")));

// The order of DumpInputFilterValue members affects their precedence, as
// documented for -dump-input-filter below.
enum DumpInputFilterValue {
  DumpInputFilterError,
  DumpInputFilterAnnotation,
  DumpInputFilterAnnotationFull,
  DumpInputFilterAll
};

static cl::list<DumpInputFilterValue> DumpInputFilters(
    "dump-input-filter",
    cl::desc("In the dump requested by -dump-input, print only input lines of\n"
             "kind <value> plus any context specified by -dump-input-context.\n"
             "When there are multiple occurrences of this option, the <value>\n"
             "that appears earliest in the list below has precedence.  The\n"
             "default is 'error' when -dump-input=fail, and it's 'all' when\n"
             "-dump-input=always.\n"),
    cl::values(clEnumValN(DumpInputFilterAll, "all", "All input lines"),
               clEnumValN(DumpInputFilterAnnotationFull, "annotation-full",
                          "Input lines with annotations"),
               clEnumValN(DumpInputFilterAnnotation, "annotation",
                          "Input lines with starting points of annotations"),
               clEnumValN(DumpInputFilterError, "error",
                          "Input lines with starting points of error "
                          "annotations")));

static cl::list<unsigned> DumpInputContexts(
    "dump-input-context", cl::value_desc("N"),
    cl::desc("In the dump requested by -dump-input, print <N> input lines\n"
             "before and <N> input lines after any lines specified by\n"
             "-dump-input-filter.  When there are multiple occurrences of\n"
             "this option, the largest specified <N> has precedence.  The\n"
             "default is 5.\n"));

static cl::opt<unsigned> DumpInputLabelWidth(
    "dump-input-label-width", cl::value_desc("N"), cl::init(0), cl::Hidden,
    cl::desc("In the dump requested by -dump-input, set <N> as the minimum\n"
             "width for the initial label column.  When there are multiple\n"
             "occurrences of this option, the last specified has precedence.\n"
             "The default is 0, meaning that the actual labels fully\n"
             "determine the width.  FileCheck's own test suite uses this\n"
             "option to avoid a fluctuating column width when checking input\n"
             "dumps.  This option is not expected to be useful elsewhere.\n"));

typedef cl::list<std::string>::const_iterator prefix_iterator;







static void DumpCommandLine(int argc, char **argv) {
  errs() << "FileCheck command line: ";
  for (int I = 0; I < argc; I++)
    errs() << " " << argv[I];
  errs() << "\n";
}

struct MarkerStyle {
  /// The first char for marking the input line.
  char Head;
  /// Every character for marking the input line between \c Head and \c Tail.
  /// Normally it is a tilde.
  char Mid;
  /// The final char for marking the input line.  Normally it is a tilde.
  char Tail;
  /// What color to use for this annotation.
  raw_ostream::Colors Color;
  /// A note to follow the marker, or empty string if none.
  std::string Note;
  /// Does this marker indicate inclusion by -dump-input-filter=error?
  bool FiltersAsError;
};

static MarkerStyle getMarker(const FileCheckDiag &Diag) {
  // By default, the marker is based on whether the diagnostic is an error or is
  // a MatchNoteDiag on a MatchResultDiag that is an error.
  //
  // It's less confusing if diagnostics that don't actually have match ranges
  // don't have markers.  For example, a marker for the MatchNoteDiag
  // 'with "VAR" equal to "5"' would seem to indicate where "VAR" matches, but
  // we don't actually have that location.  Instead, we just place the note
  // after the start of the associated MatchResultDiag.  Search ranges are
  // indicated separately.
  MarkerStyle Res;
  bool IsError = Diag.isError() || Diag.getMatchResultDiag().isError();
  if (Diag.getMatchRange()) {
    Res.Head = IsError ? '!' : '^';
    Res.Mid = Res.Tail = '~';
  } else {
    Res.Head = Res.Mid = Res.Tail = ' ';
  }
  Res.Color = IsError ? raw_ostream::RED : raw_ostream::GREEN;
  Res.FiltersAsError = IsError;

  // Add Note.  Override the default Head and Color for some diagnostic kinds.
  switch (Diag.getKind()) {
  case FileCheckDiag::MatchFoundDiag:
    switch (cast<MatchFoundDiag>(Diag).getStatus()) {
    case MatchFoundDiag::Success:
      break;
    case MatchFoundDiag::Excluded:
      Res.Note = "no match expected";
      break;
    case MatchFoundDiag::WrongLine:
      Res.Note = "match on wrong line";
      break;
    case MatchFoundDiag::Discarded:
      Res.Head = '!'; // Not an error, but not a successful match either.
      Res.Color = raw_ostream::CYAN;
      Res.Note = "discard: overlaps earlier match";
      break;
    }
    break;
  case FileCheckDiag::MatchNoneDiag:
    switch (cast<MatchNoneDiag>(Diag).getStatus()) {
    case MatchNoneDiag::Success:
      break;
    case MatchNoneDiag::InvalidPattern:
      Res.Note = "match failed for invalid pattern";
      break;
    case MatchNoneDiag::Expected:
      Res.Note = "no match found in search range";
      break;
    }
    break;
  case FileCheckDiag::MatchFuzzyDiag:
    Res.Head = '?';
    Res.Color = raw_ostream::MAGENTA;
    Res.Note = "possible intended match";
    break;
  case FileCheckDiag::MatchCustomNoteDiag:
    Res.Note = cast<MatchCustomNoteDiag>(Diag).getNote();
    break;
  }
  if (Diag.isError()) {
    assert(!Res.Note.empty() && "expected error diagnostic to have note");
    Res.Note = "error: " + Res.Note;
  }
  return Res;
}

static void DumpInputAnnotationHelp(raw_ostream &OS) {
  OS << "The following description was requested by -dump-input=help to\n"
     << "explain the input dump printed by FileCheck.\n"
     << "\n"
     << "Related command-line options:\n"
     << "\n"
     << "  - -dump-input=<value> enables or disables the input dump\n"
     << "  - -dump-input-filter=<value> filters the input lines\n"
     << "  - -dump-input-context=<N> adjusts the context of filtered lines\n"
     << "  - -v and -vv add more annotations\n"
     << "  - -color forces colors to be enabled both in the dump and below\n"
     << "  - -help documents the above options in more detail\n"
     << "\n"
     << "These options can also be set via FILECHECK_OPTS.  For example, for\n"
     << "maximum debugging output on failures:\n"
     << "\n"
     << "  $ FILECHECK_OPTS='-dump-input-filter=all -vv -color' ninja check\n"
     << "\n"
     << "Input dump annotation format:\n"
     << "\n";

  // Labels for input lines.
  OS << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "L:";
  OS << "     labels line number L of the input file\n"
     << "           An extra space is added after each input line to represent"
     << " the\n"
     << "           newline character\n";

  // Labels for annotation lines.
  OS << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "T:L";
  OS << "    labels the only match result for either (1) a pattern of type T"
     << " from\n"
     << "           line L of the check file if L is an integer or (2) the"
     << " I-th implicit\n"
     << "           pattern if L is \"imp\" followed by an integer "
     << "I (index origin one)\n";
  OS << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "T:L'N";
  OS << "  labels the Nth match result for such a pattern\n";

  // Markers on annotation lines.
  OS << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "^~~";
  OS << "    marks good match (reported if -v)\n"
     << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "!~~";
  OS << "    marks bad match, such as:\n"
     << "           - CHECK-NEXT on same line as previous match (error)\n"
     << "           - CHECK-NOT found (error)\n"
     << "           - CHECK-DAG overlapping match (discarded, reported if "
     << "-vv)\n"
     << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "{ }";
  OS << "    encloses search range (exclusive bounds) when no match is found "
     << "or\n"
     << "           there is an error, such as:\n"
     << "           - the errors mentioned above\n"
     << "           - CHECK-NEXT not found (error)\n"
     << "           - CHECK-NOT not found (success, reported if -vv)\n"
     << "           - CHECK-DAG not found after discarded matches (error)\n"
     << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "?";
  OS << "      marks fuzzy match when no match is found\n";

  // Elided lines.
  OS << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "...";
  OS << "    indicates elided input lines and annotations, as specified by\n"
     << "           -dump-input-filter and -dump-input-context\n";

  // Colors.
  OS << "  - colors ";
  WithColor(OS, raw_ostream::GREEN, true) << "success";
  OS << ", ";
  WithColor(OS, raw_ostream::RED, true) << "error";
  OS << ", ";
  WithColor(OS, raw_ostream::MAGENTA, true) << "fuzzy match";
  OS << ", ";
  WithColor(OS, raw_ostream::CYAN, true, false) << "discarded match";
  OS << ", ";
  WithColor(OS, raw_ostream::CYAN, true, true) << "unmatched input";
  OS << "\n";
}

/// An annotation for a single input line.
struct InputAnnotation {
  /// A globally unique index for this annotation before it was broken into
  /// multiple lines.
  unsigned LabelIndexGlobal;
  /// The globally unique label for this annotation before it was broken into
  /// multiple lines.  There is one \c Label per \c LabelIndexGlobal and
  /// vice-versa.
  std::string Label;
  /// Is this the initial (possibly only) fragment of an annotation, which has
  /// been broken across multiple lines if necessary?
  bool IsFirstLine;
  /// What input line (one-origin indexing) this annotation marks.  This might
  /// be different from the starting line of the original diagnostic if
  /// !IsFirstLine.
  unsigned InputLine;
  /// The column range (inclusive boundaries) in which to mark the input line.
  /// A value of one indicates the first column of the actual input, and a
  /// value of zero indicates the left margin.  If \c InputLastCol is
  /// \c UINT_MAX, the rest of the input line should be marked, and another
  /// \c InputAnnotation will continue it on the next line.
  unsigned InputFirstCol, InputLastCol;
  /// The marker to use.
  MarkerStyle Marker;
  /// Whether this annotation represents a good match for an expected pattern.
  bool FoundAndExpectedMatch;
};

/// Get an abbreviation for the check type.
static std::string GetCheckTypeAbbreviation(Check::FileCheckType Ty) {
  switch (Ty) {
  case Check::CheckPlain:
    if (Ty.getCount() > 1)
      return "count";
    return "check";
  case Check::CheckNext:
    return "next";
  case Check::CheckSame:
    return "same";
  case Check::CheckNot:
    return "not";
  case Check::CheckDAG:
    return "dag";
  case Check::CheckLabel:
    return "label";
  case Check::CheckEmpty:
    return "empty";
  case Check::CheckComment:
    return "com";
  case Check::CheckEOF:
    return "eof";
  case Check::CheckBadNot:
    return "bad-not";
  case Check::CheckBadCount:
    return "bad-count";
  case Check::CheckMisspelled:
    return "misspelled";
  case Check::CheckNone:
    llvm_unreachable("invalid FileCheckType");
  }
  llvm_unreachable("unknown FileCheckType");
}

template <> struct llvm::DenseMapInfo<SMLoc> {
  static unsigned getHashValue(const SMLoc &Loc) {
    return DenseMapInfo<const char *>::getHashValue(Loc.getPointer());
  }
  static bool isEqual(const SMLoc &LHS, SMLoc &RHS) { return LHS == RHS; }
};

namespace {
/// Stores all information needed to generate \c InputAnnotation labels.
class InputAnnotationLabeler {
private:
  const SourceMgr &SM;
  const unsigned CheckFileBufferID;
  const std::pair<unsigned, unsigned> ImpPatBufferIDRange;

  /// How many unique input annotation labels does each check pattern need?
  /// Each check pattern can have multiple \c MatchResultDiag's, each followed
  /// by a series of zero or more \c MatchNoteDiag's.  Each such
  /// \c MatchResultDiag and its \c MatchNoteDiag series can require multiple
  /// labels.
  DenseMap<SMLoc, unsigned> LabelCountPerPattern;
  /// For each check pattern, how many labels have we generated so far?
  DenseMap<SMLoc, unsigned> LabelIndexPerPattern;
  /// For each check pattern, what is the common prefix for all its labels?
  DenseMap<SMLoc, std::string> LabelPrefixPerPattern;
  /// How many total labels have we generated so far over all check patterns?
  unsigned LabelIndexGlobal;
  /// The widest label generated so far over all check patterns.
  unsigned LabelWidthGlobal;

public:
  /// - \p CheckFileBufferID is the buffer ID for the check file.
  /// - \p ImpPatBufferIDRange is the buffer ID range for all implicit patterns.
  InputAnnotationLabeler(const SourceMgr &SM, unsigned CheckFileBufferID,
                         std::pair<unsigned, unsigned> ImpPatBufferIDRange)
      : SM(SM), CheckFileBufferID(CheckFileBufferID),
        ImpPatBufferIDRange(ImpPatBufferIDRange), LabelIndexGlobal(0),
        LabelWidthGlobal(0) {}
  /// Add \c C to the number of expected \c makeLabel calls for \c Diag.
  /// \c expect must not be called after the first \c makeLabel call.
  void expect(const FileCheckDiag &Diag, unsigned C) {
    LabelCountPerPattern[Diag.getMatchResultDiag().getCheckLoc()] += C;
  }
  /// Write a new globally unique label for \c Diag into \p Label, and write its
  /// globally unique index into LabelIndexGlobal.  All \c makeLabel calls must
  /// have already been predicted by \c expect calls.
  void makeLabel(const FileCheckDiag &Diag, std::string &Label,
                 unsigned &LabelIndexGlobal) {
    const MatchResultDiag &MRD = Diag.getMatchResultDiag();
    SMLoc CheckLoc = MRD.getCheckLoc();
    std::string &LabelPrefix = LabelPrefixPerPattern[CheckLoc];
    if (LabelPrefix.empty()) {
      llvm::raw_string_ostream LabelStrm(LabelPrefix);
      LabelStrm << GetCheckTypeAbbreviation(MRD.getCheckTy()) << ":";
      unsigned CheckBufferID = SM.FindBufferContainingLoc(CheckLoc);
      if (CheckBufferID == CheckFileBufferID)
        LabelStrm << SM.getLineAndColumn(CheckLoc, CheckBufferID).first;
      else if (ImpPatBufferIDRange.first <= CheckBufferID &&
               CheckBufferID < ImpPatBufferIDRange.second)
        LabelStrm << "imp" << (CheckBufferID - ImpPatBufferIDRange.first + 1);
      else
        llvm_unreachable(
            "expected check location to be either in the check file or for an "
            "implicit pattern");
    }
    assert(Label.empty() && "expected empty string for writing label");
    llvm::raw_string_ostream LabelStrm(Label);
    LabelStrm << LabelPrefix;
    unsigned LabelCount = LabelCountPerPattern[CheckLoc];
    unsigned LabelIndex = LabelIndexPerPattern[CheckLoc]++;
    assert(LabelIndex < LabelCount &&
           "expected all makeLabel calls to be predicted by expect calls");
    if (LabelCount > 1)
      LabelStrm << "'" << LabelIndex;
    LabelWidthGlobal =
        std::max((std::string::size_type)LabelWidthGlobal, Label.size());
    LabelIndexGlobal = this->LabelIndexGlobal++;
  }
  /// Get the widest label generated.
  unsigned getLabelWidthGlobal() const { return LabelWidthGlobal; }
};

/// A range specifying where annotation markers are physically \a drawn in the
/// input dump.
struct MarkerRange {
public:
  /// An inclusive \c MarkerRange boundary.  Both line and column use a 1-based
  /// index origin.
  struct Loc {
    unsigned Line;
    unsigned Col;
    /// Make an invalid location to be overwritten before being used.
    Loc() : Line(0), Col(0) {}
    /// Make a valid location.
    Loc(const std::pair<unsigned, unsigned> &LineAndCol)
        : Line(LineAndCol.first), Col(LineAndCol.second) {}
  };

private:
  /// Location of the first marked character.
  Loc First;
  /// Location of the last marked character.
  Loc Last;

public:
  /// Make an invalid range to be overwritten before being used.
  MarkerRange() = default;
  /// \p Range specifies the \a logical input range to be depicted by annotation
  /// markers \a drawn at the resulting \c MarkerRange.
  ///
  /// \a how that drawing depicts that logical input range is determined by
  /// \p ShowExclusive.  The drawing specifies either:
  /// - The \a inclusive start and end bounds of the logical input range if
  ///   \p !ShowExclusive.  In this case:
  ///   - If the logical input range is empty, then the resulting \c MarkerRange
  ///     is expanded to a single character.  This avoids a missing marker, but
  ///     it means the markers for a single-character range are
  ///     indistinguishable from markers for an empty range.
  ///   - The first and last location of the \c MarkerRange are always real
  ///     locations in the input (never, for example, column 0).
  /// - The \a exclusive start and end bounds of the logical input range if
  ///   \p ShowExclusive.  In this case:
  ///   - The \c MarkerRange length is then always at least two because
  ///     exclusive boundaries never occupy the same location.
  ///   - If a \p Range boundary is an input line boundary, the corresponding
  ///     \c MarkerRange column might be in the line's margin (e.g., column 0)
  ///     to avoid placing a marker on an adjacent line.  That decision can make
  ///     input annotations more concise (more line-liners) and easier to read.
  ///     It also avoids non-existent adjacent lines (e.g., line 0) that are not
  ///     depicted in the input dump.
  MarkerRange(const SourceMgr &SM, SMRange Range, bool ShowExclusive = false) {
    // Given an SMRange representing the range of text "range of text", the
    // following example compares how it and the resulting MarkerRange encode
    // the same start (s) and end (e) bounds:
    //
    //     ....range of text....
    //         s            e    SMRange
    //         s           e     MarkerRange with ShowExclusive=false
    //        s             e    MarkerRange with ShowExclusive=true
    if (ShowExclusive) {
      // Range has inclusive start, but ShowExclusive requires exclusive start.
      First = SM.getLineAndColumn(Range.Start);
      --First.Col;
      // Range has an exclusive end as ShowExclusive requires.  If it is at a
      // line boundary, it is at the start of the next line, so normally move it
      // to the end of the previous line.  For an empty range, do not do that as
      // we do not want an end marker on the line before the start marker.
      if (Range.Start == Range.End) {
        Last = SM.getLineAndColumn(Range.End);
      } else {
        SMLoc EndLoc = SMLoc::getFromPointer(Range.End.getPointer() - 1);
        Last = SM.getLineAndColumn(EndLoc);
        ++Last.Col;
      }
      return;
    }
    // Range has an inclusive start as !ShowExclusive requires.
    First = SM.getLineAndColumn(Range.Start);
    // Range has an exclusive end, but !ShowExclusive requires an inclusive end.
    if (Range.Start == Range.End) {
      // Convert the empty range to a one-character range.
      Last = First;
    } else {
      // We cannot simply subtract one from the end column number because that
      // might result in column 0, which does not exist and is thus incorrect
      // for an inclusive boundary.
      SMLoc EndLoc = SMLoc::getFromPointer(Range.End.getPointer() - 1);
      Last = SM.getLineAndColumn(EndLoc);
    }
  }
  /// \p Loc specifies a single input character to be marked by a single
  /// annotation marker character.
  MarkerRange(Loc OneChar) : First(OneChar), Last(OneChar) {}
  /// Is the marker range contained on a single line?
  bool isSingleLine() const { return First.Line == Last.Line; }
  /// Get the location of the first marked character.
  Loc getFirstLoc() const { return First; }
  /// Get the location of the last marked character.
  Loc getLastLoc() const { return Last; }
};

/// Emits search range annotations for each \c MatchResultDiag as it is
/// encountered.
///
/// In some cases, it emits a single, one-line annotation.  Otherwise, it emits
/// separate annotations for the start and end of the search range.  The logic
/// for making this determination is encapsulated in static member functions.
class SearchRangeAnnotator {
private:
  const SourceMgr &SM;
  InputAnnotationLabeler &Labeler;
  std::vector<InputAnnotation> &Annotations;

  /// The most recent \c MatchResultDiag, or \c nullptr if all search range
  /// annotations have been added already for the most recent
  /// \c MatchResultDiag.
  const MatchResultDiag *MRD;
  /// Assuming \c makesAnnotationsFor(MRD), would a \c SearchRangeAnnotator make
  /// a one-line search range annotation for \p MRD?  Either way, the search
  /// range computed for \p MRD is stored in \p SearchRange.
  static bool makesOneLinerFor(const SourceMgr &SM, const MatchResultDiag &MRD,
                               MarkerRange &SearchRange) {
    assert(makesAnnotationsFor(MRD) &&
           "expected makesAnnotationsFor to be checked first");
    SearchRange = {SM, MRD.getSearchRange(), /*ShowExclusive=*/true};
    return SearchRange.isSingleLine();
  }
  /// Make the next annotation for the current \c MatchResultDiag.
  void makeAnnotation(bool Start) {
    InputAnnotation &A = Annotations.emplace_back();
    Labeler.makeLabel(*MRD, A.Label, A.LabelIndexGlobal);
    A.IsFirstLine = true;
    A.FoundAndExpectedMatch = false;
    MarkerRange SearchRange;
    if (makesOneLinerFor(SM, *MRD, SearchRange)) {
      assert(Start && "expected no search range end annotation for one-liner");
      A.InputLine = SearchRange.getFirstLoc().Line;
      A.InputFirstCol = SearchRange.getFirstLoc().Col;
      A.InputLastCol = SearchRange.getLastLoc().Col;
      MatchCustomNoteDiag NoteDiag("search range (exclusive bounds)");
      NoteDiag.setMatchResultDiag(MRD);
      A.Marker = getMarker(NoteDiag);
      A.Marker.Head = '{';
      A.Marker.Mid = ' ';
      A.Marker.Tail = '}';
      MRD = nullptr;
      return;
    }
    // We have separate annotations for start and end.
    MarkerRange::Loc Loc =
        Start ? SearchRange.getFirstLoc() : SearchRange.getLastLoc();
    A.InputLine = Loc.Line;
    A.InputFirstCol = A.InputLastCol = Loc.Col;
    MatchCustomNoteDiag NoteDiag(std::string("search range ") +
                                 (Start ? "start" : "end") + " (exclusive)");
    NoteDiag.setMatchResultDiag(MRD);
    A.Marker = getMarker(NoteDiag);
    A.Marker.Head = Start ? '{' : '}';
  }

public:
  /// Would a \c SearchRangeAnnotator make any search range annotations for
  /// \p MRD?
  static bool makesAnnotationsFor(const MatchResultDiag &MRD) {
    return !MRD.getMatchRange() || MRD.isError();
  }
  /// Tell the labeler how many times this will call
  /// \c InputAnnotationLabeler::makeLabel for \c MRD.
  void predictLabelsFor(const MatchResultDiag &MRD) {
    if (!makesAnnotationsFor(MRD))
      return;
    MarkerRange SearchRange;
    Labeler.expect(MRD, makesOneLinerFor(SM, MRD, SearchRange) ? 1 : 2);
  }
  /// \p Annotations is where this annotator should append search range
  /// annotations.
  SearchRangeAnnotator(const SourceMgr &SM, InputAnnotationLabeler &Labeler,
                       std::vector<InputAnnotation> &Annotations)
      : SM(SM), Labeler(Labeler), Annotations(Annotations), MRD(nullptr) {}
  /// Emit any search range start annotation or one-line search range annotation
  /// for \p MRDNew.  Emit any search range end annotation for \p MRDNew at the
  /// next call to \c newMatchResultDiag or \c endDiags.
  void newMatchResultDiag(const MatchResultDiag &MRDNew) {
    if (MRD) {
      makeAnnotation(/*Start=*/false);
      MRD = nullptr;
    }
    if (makesAnnotationsFor(MRDNew)) {
      MRD = &MRDNew;
      makeAnnotation(/*Start=*/true);
    }
  }
  /// Emit any search range end annotation for the final \c MatchResultDiag
  /// passed to \c newMatchResultDiag.
  void endDiags() {
    if (MRD)
      makeAnnotation(/*Start=*/false);
  }
};

/// Emits the main annotations for each \c FileCheckDiag as it is encountered.
class FileCheckDiagAnnotator {
private:
  const SourceMgr &SM;
  InputAnnotationLabeler &Labeler;
  std::vector<InputAnnotation> &Annotations;
  /// Would a \c FileCheckDiagAnnotator make any annotations for \p Diag?
  static bool makesAnnotationsFor(const FileCheckDiag &Diag) {
    const MatchResultDiag *MRD = dyn_cast<MatchResultDiag>(&Diag);
    if (!MRD)
      return true;
    if (SearchRangeAnnotator::makesAnnotationsFor(*MRD) &&
        getMarker(*MRD).Note.empty())
      return false;
    return true;
  }

public:
  /// Tell the labeler how many times this will call
  /// \c InputAnnotationLabeler::makeLabel for \c Diag.
  void predictLabelsFor(const FileCheckDiag &Diag) {
    Labeler.expect(Diag, makesAnnotationsFor(Diag) ? 1 : 0);
  }
  /// \p Annotations is where this annotator should append annotations.
  FileCheckDiagAnnotator(const SourceMgr &SM, InputAnnotationLabeler &Labeler,
                         std::vector<InputAnnotation> &Annotations)
      : SM(SM), Labeler(Labeler), Annotations(Annotations) {}
  /// Emit any annotations for \c Diag.
  void makeAnnotations(const FileCheckDiag &Diag) {
    if (!makesAnnotationsFor(Diag))
      return;

    // Build label that is unique for this input annotation before it is
    // potentially broken across multiple lines.
    InputAnnotation A;
    Labeler.makeLabel(Diag, A.Label, A.LabelIndexGlobal);

    // Build the input marker.
    A.Marker = getMarker(Diag);

    // Does this diagnostic mark text that has been successfully matched?
    A.FoundAndExpectedMatch = false;
    if (const MatchFoundDiag *Found = dyn_cast<MatchFoundDiag>(&Diag)) {
      if (Found->getStatus() == MatchFoundDiag::Success)
        A.FoundAndExpectedMatch = true;
    }

    // If Diag has a match range, position the marker there.  Otherwise,
    // position the marker at the start of the most recent MatchResultDiag, with
    // which it is associated.
    MarkerRange InputRange;
    if (Diag.getMatchRange()) {
      InputRange = MarkerRange(SM, *Diag.getMatchRange());
    } else {
      const MatchResultDiag &MRD = Diag.getMatchResultDiag();
      InputRange = MRD.getMatchRange() ? MarkerRange(SM, *MRD.getMatchRange())
                                       : MarkerRange(SM, MRD.getSearchRange());
      InputRange = MarkerRange(InputRange.getFirstLoc());
      assert(A.Marker.Head == ' ' && "expected no marker for no match range");
    }

    // Compute the marker location, and break annotation into multiple
    // annotations if it spans multiple lines.
    A.IsFirstLine = true;
    A.InputLine = InputRange.getFirstLoc().Line;
    A.InputFirstCol = InputRange.getFirstLoc().Col;
    if (InputRange.isSingleLine()) {
      A.InputLastCol = InputRange.getLastLoc().Col;
      Annotations.push_back(A);
    } else {
      A.InputLastCol = UINT_MAX;
      char MarkerTail = A.Marker.Tail;
      A.Marker.Tail = A.Marker.Mid;
      Annotations.push_back(A);
      for (unsigned L = InputRange.getFirstLoc().Line + 1,
                    E = InputRange.getLastLoc().Line;
           L <= E; ++L) {
        InputAnnotation B;
        B.LabelIndexGlobal = A.LabelIndexGlobal;
        B.Label = A.Label;
        B.IsFirstLine = false;
        B.InputLine = L;
        B.Marker = A.Marker;
        B.Marker.Head = B.Marker.Mid = A.Marker.Mid;
        B.Marker.Tail = L != E ? A.Marker.Mid : MarkerTail;
        B.Marker.Note = "";
        B.InputFirstCol = 1;
        B.InputLastCol = L != E ? UINT_MAX : InputRange.getLastLoc().Col;
        B.FoundAndExpectedMatch = A.FoundAndExpectedMatch;
        Annotations.push_back(B);
      }
    }
  }
};
} // namespace

static void
buildInputAnnotations(const SourceMgr &SM, unsigned CheckFileBufferID,
                      const std::pair<unsigned, unsigned> &ImpPatBufferIDRange,
                      const FileCheckDiagList &Diags,
                      std::vector<InputAnnotation> &Annotations,
                      unsigned &LabelWidthGlobal) {
  InputAnnotationLabeler Labeler(SM, CheckFileBufferID, ImpPatBufferIDRange);
  SearchRangeAnnotator TheSearchRangeAnnotator(SM, Labeler, Annotations);
  FileCheckDiagAnnotator TheFileCheckDiagAnnotator(SM, Labeler, Annotations);
  for (const FileCheckDiag &Diag : Diags) {
    if (const MatchResultDiag *MRD = dyn_cast<MatchResultDiag>(&Diag))
      TheSearchRangeAnnotator.predictLabelsFor(*MRD);
    TheFileCheckDiagAnnotator.predictLabelsFor(Diag);
  }
  for (const FileCheckDiag &Diag : Diags) {
    if (const MatchResultDiag *MRD = dyn_cast<MatchResultDiag>(&Diag))
      TheSearchRangeAnnotator.newMatchResultDiag(*MRD);
    TheFileCheckDiagAnnotator.makeAnnotations(Diag);
  }
  TheSearchRangeAnnotator.endDiags();
  LabelWidthGlobal = Labeler.getLabelWidthGlobal();
}

static unsigned FindInputLineInFilter(
    DumpInputFilterValue DumpInputFilter, unsigned CurInputLine,
    const std::vector<InputAnnotation>::iterator &AnnotationBeg,
    const std::vector<InputAnnotation>::iterator &AnnotationEnd) {
  if (DumpInputFilter == DumpInputFilterAll)
    return CurInputLine;
  for (auto AnnotationItr = AnnotationBeg; AnnotationItr != AnnotationEnd;
       ++AnnotationItr) {
    switch (DumpInputFilter) {
    case DumpInputFilterAll:
      llvm_unreachable("unexpected DumpInputFilterAll");
      break;
    case DumpInputFilterAnnotationFull:
      return AnnotationItr->InputLine;
    case DumpInputFilterAnnotation:
      if (AnnotationItr->IsFirstLine)
        return AnnotationItr->InputLine;
      break;
    case DumpInputFilterError:
      if (AnnotationItr->IsFirstLine && AnnotationItr->Marker.FiltersAsError)
        return AnnotationItr->InputLine;
      break;
    }
  }
  return UINT_MAX;
}

/// To OS, print a vertical ellipsis (right-justified at LabelWidthGlobal) if it
/// would occupy less lines than ElidedLines, but print ElidedLines otherwise.
/// Either way, clear ElidedLines.  Thus, if ElidedLines is empty, do nothing.
static void DumpEllipsisOrElidedLines(raw_ostream &OS, std::string &ElidedLines,
                                      unsigned LabelWidthGlobal) {
  if (ElidedLines.empty())
    return;
  unsigned EllipsisLines = 3;
  if (EllipsisLines < StringRef(ElidedLines).count('\n')) {
    for (unsigned i = 0; i < EllipsisLines; ++i) {
      WithColor(OS, raw_ostream::BRIGHT_BLACK, /*Bold=*/true)
          << right_justify(".", LabelWidthGlobal);
      OS << '\n';
    }
  } else
    OS << ElidedLines;
  ElidedLines.clear();
}

static void DumpAnnotatedInput(raw_ostream &OS, const FileCheckRequest &Req,
                               DumpInputFilterValue DumpInputFilter,
                               unsigned DumpInputContext,
                               StringRef InputFileText,
                               std::vector<InputAnnotation> &Annotations,
                               unsigned LabelWidthGlobal) {
  OS << "Input was:\n<<<<<<\n";

  // Sort annotations.
  llvm::sort(Annotations,
             [](const InputAnnotation &A, const InputAnnotation &B) {
               // 1. Sort annotations in the order of the input lines.
               //
               // This makes it easier to find relevant annotations while
               // iterating input lines in the implementation below.  FileCheck
               // does not always produce diagnostics in the order of input
               // lines due to, for example, CHECK-DAG and CHECK-NOT.
               if (A.InputLine != B.InputLine)
                 return A.InputLine < B.InputLine;
               // 2. Sort annotations in the temporal order FileCheck produced
               // their associated diagnostics.
               //
               // This sort offers several benefits:
               //
               // A. On a single input line, the order of annotations reflects
               //    the FileCheck logic for processing directives/patterns.
               //    This can be helpful in understanding cases in which the
               //    order of the associated directives/patterns in the check
               //    file or on the command line either (i) does not match the
               //    temporal order in which FileCheck looks for matches for the
               //    directives/patterns (due to, for example, CHECK-LABEL,
               //    CHECK-NOT, or `--implicit-check-not`) or (ii) does match
               //    that order but does not match the order of those
               //    diagnostics along an input line (due to, for example,
               //    CHECK-DAG).
               //
               //    On the other hand, because our presentation format presents
               //    input lines in order, there's no clear way to offer the
               //    same benefit across input lines.  For consistency, it might
               //    then seem worthwhile to have annotations on a single line
               //    also sorted in input order (that is, by input column).
               //    However, in practice, this appears to be more confusing
               //    than helpful.  Perhaps it's intuitive to expect annotations
               //    to be listed in the temporal order in which they were
               //    produced except in cases the presentation format obviously
               //    and inherently cannot support it (that is, across input
               //    lines).
               //
               // B. When diagnostics' annotations are split among multiple
               //    input lines, the user must track them from one input line
               //    to the next.  One property of the sort chosen here is that
               //    it facilitates the user in this regard by ensuring the
               //    following: when comparing any two input lines, a
               //    diagnostic's annotations are sorted in the same position
               //    relative to all other diagnostics' annotations.
               return A.LabelIndexGlobal < B.LabelIndexGlobal;
             });

  // Compute the width of the label column.
  const unsigned char *InputFilePtr = InputFileText.bytes_begin(),
                      *InputFileEnd = InputFileText.bytes_end();
  unsigned LineCount = InputFileText.count('\n');
  if (InputFileEnd[-1] != '\n')
    ++LineCount;
  unsigned LineNoWidth = NumDigitsBase10(LineCount);
  // +3 below adds spaces (1) to the left of the (right-aligned) line numbers
  // on input lines and (2) to the right of the (left-aligned) labels on
  // annotation lines so that input lines and annotation lines are more
  // visually distinct.  For example, the spaces on the annotation lines ensure
  // that input line numbers and check directive line numbers never align
  // horizontally.  Those line numbers might not even be for the same file.
  // One space would be enough to achieve that, but more makes it even easier
  // to see.
  LabelWidthGlobal = std::max(LabelWidthGlobal, LineNoWidth) + 3;
  LabelWidthGlobal = std::max(LabelWidthGlobal, DumpInputLabelWidth.getValue());

  // Print annotated input lines.
  unsigned PrevLineInFilter = 0; // 0 means none so far
  unsigned NextLineInFilter = 0; // 0 means uncomputed, UINT_MAX means none
  std::string ElidedLines;
  raw_string_ostream ElidedLinesOS(ElidedLines);
  ColorMode TheColorMode =
      WithColor(OS).colorsEnabled() ? ColorMode::Enable : ColorMode::Disable;
  if (TheColorMode == ColorMode::Enable)
    ElidedLinesOS.enable_colors(true);
  auto AnnotationItr = Annotations.begin(), AnnotationEnd = Annotations.end();
  for (unsigned Line = 1;
       InputFilePtr != InputFileEnd || AnnotationItr != AnnotationEnd;
       ++Line) {
    const unsigned char *InputFileLine = InputFilePtr;

    // Compute the previous and next line included by the filter.
    if (NextLineInFilter < Line)
      NextLineInFilter = FindInputLineInFilter(DumpInputFilter, Line,
                                               AnnotationItr, AnnotationEnd);
    assert(NextLineInFilter && "expected NextLineInFilter to be computed");
    if (NextLineInFilter == Line)
      PrevLineInFilter = Line;

    // Elide this input line and its annotations if it's not within the
    // context specified by -dump-input-context of an input line included by
    // -dump-input-filter.  However, in case the resulting ellipsis would occupy
    // more lines than the input lines and annotations it elides, buffer the
    // elided lines and annotations so we can print them instead.
    raw_ostream *LineOS;
    if ((!PrevLineInFilter || PrevLineInFilter + DumpInputContext < Line) &&
        (NextLineInFilter == UINT_MAX ||
         Line + DumpInputContext < NextLineInFilter))
      LineOS = &ElidedLinesOS;
    else {
      LineOS = &OS;
      DumpEllipsisOrElidedLines(OS, ElidedLines, LabelWidthGlobal);
    }

    // Print right-aligned line number.
    WithColor(*LineOS, raw_ostream::BRIGHT_BLACK, /*Bold=*/true, /*BG=*/false,
              TheColorMode)
        << format_decimal(Line, LabelWidthGlobal) << ": ";

    // For the case where -v and colors are enabled, find the annotations for
    // good matches for expected patterns in order to highlight everything
    // else in the line.  There are no such annotations if -v is disabled.
    std::vector<InputAnnotation> FoundAndExpectedMatches;
    if (Req.Verbose && TheColorMode == ColorMode::Enable) {
      for (auto I = AnnotationItr; I != AnnotationEnd && I->InputLine == Line;
           ++I) {
        if (I->FoundAndExpectedMatch)
          FoundAndExpectedMatches.push_back(*I);
      }
    }

    // Print numbered line with highlighting where there are no matches for
    // expected patterns.
    bool Newline = false;
    {
      WithColor COS(*LineOS, raw_ostream::SAVEDCOLOR, /*Bold=*/false,
                    /*BG=*/false, TheColorMode);
      bool InMatch = false;
      if (Req.Verbose) {
        COS.changeColor(raw_ostream::CYAN, /*Bold=*/true, /*BG=*/true);
      } else {
        // Our goal is to use the output streams's default color so that input
        // text is legibile in both light and dark themes.  SAVEDCOLOR above
        // currently ignores the Bold=false there, so we override it with
        // resetColor here, which ensures consistent colors with the resetColor
        // below anyway.
        COS.resetColor();
      }
      for (unsigned Col = 1; InputFilePtr != InputFileEnd && !Newline; ++Col) {
        bool WasInMatch = InMatch;
        InMatch = false;
        for (const InputAnnotation &M : FoundAndExpectedMatches) {
          if (M.InputFirstCol <= Col && Col <= M.InputLastCol) {
            InMatch = true;
            break;
          }
        }
        // If !Req.Verbose, FoundAndExpectedMatches is empty, so InMatch and
        // WasInMatch remain false, so these color transitions never happen.
        if (!WasInMatch && InMatch)
          COS.resetColor();
        else if (WasInMatch && !InMatch)
          COS.changeColor(raw_ostream::CYAN, true, true);
        if (*InputFilePtr == '\n') {
          Newline = true;
          COS << ' ';
        } else
          COS << *InputFilePtr;
        ++InputFilePtr;
      }
    }
    *LineOS << '\n';
    unsigned InputLineWidth = InputFilePtr - InputFileLine;

    // Print any annotations.
    while (AnnotationItr != AnnotationEnd &&
           AnnotationItr->InputLine == Line) {
      WithColor COS(*LineOS, AnnotationItr->Marker.Color, /*Bold=*/true,
                    /*BG=*/false, TheColorMode);
      // The space below aligns with the ":" on the input line.
      COS << left_justify(AnnotationItr->Label, LabelWidthGlobal) << " ";
      unsigned Col;
      // A search range annotation at the beginning of the line starts at column
      // 0 because it is an exclusive boundary.
      for (Col = 0; Col < AnnotationItr->InputFirstCol; ++Col)
        COS << ' ';
      COS << AnnotationItr->Marker.Head;
      // If InputLastCol==UINT_MAX, stop at InputLineWidth.
      for (++Col; Col < AnnotationItr->InputLastCol && Col <= InputLineWidth;
           ++Col)
        COS << AnnotationItr->Marker.Mid;
      if (Col <= AnnotationItr->InputLastCol &&
          AnnotationItr->InputLastCol != UINT_MAX) {
        COS << AnnotationItr->Marker.Tail;
        ++Col;
      }
      const std::string &Note = AnnotationItr->Marker.Note;
      if (!Note.empty()) {
        // Put the note at the end of the input line.  If we were to instead
        // put the note right after the marker, subsequent annotations for the
        // same input line might appear to mark this note instead of the input
        // line.
        for (; Col <= InputLineWidth + 1; ++Col)
          COS << ' ';
        COS << ' ' << Note;
      }
      COS << '\n';
      ++AnnotationItr;
    }
  }
  DumpEllipsisOrElidedLines(OS, ElidedLines, LabelWidthGlobal);

  OS << ">>>>>>\n";
}

int main(int argc, char **argv) {
  // Enable use of ANSI color codes because FileCheck is using them to
  // highlight text.
  llvm::sys::Process::UseANSIEscapeCodes(true);

  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, /*Overview*/ "", /*Errs*/ nullptr,
                              /*VFS*/ nullptr, "FILECHECK_OPTS");

  // Select -dump-input* values.  The -help documentation specifies the default
  // value and which value to choose if an option is specified multiple times.
  // In the latter case, the general rule of thumb is to choose the value that
  // provides the most information.
  DumpInputValue DumpInput =
      DumpInputs.empty() ? DumpInputFail : *llvm::max_element(DumpInputs);
  DumpInputFilterValue DumpInputFilter;
  if (DumpInputFilters.empty())
    DumpInputFilter = DumpInput == DumpInputAlways ? DumpInputFilterAll
                                                   : DumpInputFilterError;
  else
    DumpInputFilter = *llvm::max_element(DumpInputFilters);
  unsigned DumpInputContext =
      DumpInputContexts.empty() ? 5 : *llvm::max_element(DumpInputContexts);

  if (DumpInput == DumpInputHelp) {
    DumpInputAnnotationHelp(outs());
    return 0;
  }
  if (CheckFilename.empty()) {
    errs() << "<check-file> not specified\n";
    return 2;
  }

  FileCheckRequest Req;
  append_range(Req.CheckPrefixes, CheckPrefixes);

  append_range(Req.CommentPrefixes, CommentPrefixes);

  append_range(Req.ImplicitCheckNot, ImplicitCheckNot);

  bool GlobalDefineError = false;
  for (StringRef G : GlobalDefines) {
    size_t EqIdx = G.find('=');
    if (EqIdx == std::string::npos) {
      errs() << "Missing equal sign in command-line definition '-D" << G
             << "'\n";
      GlobalDefineError = true;
      continue;
    }
    if (EqIdx == 0) {
      errs() << "Missing variable name in command-line definition '-D" << G
             << "'\n";
      GlobalDefineError = true;
      continue;
    }
    Req.GlobalDefines.push_back(G);
  }
  if (GlobalDefineError)
    return 2;

  Req.AllowEmptyInput = AllowEmptyInput;
  Req.AllowUnusedPrefixes = AllowUnusedPrefixes;
  Req.EnableVarScope = EnableVarScope;
  Req.AllowDeprecatedDagOverlap = AllowDeprecatedDagOverlap;
  Req.Verbose = Verbose;
  Req.VerboseVerbose = VerboseVerbose;
  Req.NoCanonicalizeWhiteSpace = NoCanonicalizeWhiteSpace;
  Req.MatchFullLines = MatchFullLines;
  Req.IgnoreCase = IgnoreCase;

  if (VerboseVerbose)
    Req.Verbose = true;

  FileCheck FC(Req);
  if (!FC.ValidateCheckPrefixes())
    return 2;

  SourceMgr SM;

  // Read the expected strings from the check file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> CheckFileOrErr =
      MemoryBuffer::getFileOrSTDIN(CheckFilename, /*IsText=*/true);
  if (std::error_code EC = CheckFileOrErr.getError()) {
    errs() << "Could not open check file '" << CheckFilename
           << "': " << EC.message() << '\n';
    return 2;
  }
  MemoryBuffer &CheckFile = *CheckFileOrErr.get();

  SmallString<4096> CheckFileBuffer;
  StringRef CheckFileText = FC.CanonicalizeFile(CheckFile, CheckFileBuffer);

  unsigned CheckFileBufferID =
      SM.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(
                                CheckFileText, CheckFile.getBufferIdentifier()),
                            SMLoc());

  std::pair<unsigned, unsigned> ImpPatBufferIDRange;
  if (FC.readCheckFile(SM, CheckFileText, &ImpPatBufferIDRange))
    return 2;

  // Open the file to check and add it to SourceMgr.
  ErrorOr<std::unique_ptr<MemoryBuffer>> InputFileOrErr =
      MemoryBuffer::getFileOrSTDIN(InputFilename, /*IsText=*/true);
  if (InputFilename == "-")
    InputFilename = "<stdin>"; // Overwrite for improved diagnostic messages
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
  StringRef InputFileText = FC.CanonicalizeFile(InputFile, InputFileBuffer);

  SM.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(
                            InputFileText, InputFile.getBufferIdentifier()),
                        SMLoc());

  FileCheckDiagList Diags;
  int ExitCode = FC.checkInput(SM, InputFileText,
                               DumpInput == DumpInputNever ? nullptr : &Diags)
                     ? EXIT_SUCCESS
                     : 1;
  if (DumpInput == DumpInputAlways ||
      (ExitCode == 1 && DumpInput == DumpInputFail)) {
    errs() << "\n"
           << "Input file: " << InputFilename << "\n"
           << "Check file: " << CheckFilename << "\n"
           << "\n"
           << "-dump-input=help explains the following input dump.\n"
           << "\n";
    std::vector<InputAnnotation> Annotations;
    unsigned LabelWidthGlobal;
    buildInputAnnotations(SM, CheckFileBufferID, ImpPatBufferIDRange, Diags,
                          Annotations, LabelWidthGlobal);
    DumpAnnotatedInput(errs(), Req, DumpInputFilter, DumpInputContext,
                       InputFileText, Annotations, LabelWidthGlobal);
  }

  return ExitCode;
}
