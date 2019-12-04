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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileCheck.h"
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
    "v", cl::init(false),
    cl::desc("Print directive pattern matches, or add them to the input dump\n"
             "if enabled.\n"));

static cl::opt<bool> VerboseVerbose(
    "vv", cl::init(false),
    cl::desc("Print information helpful in diagnosing internal FileCheck\n"
             "issues, or add it to the input dump if enabled.  Implies\n"
             "-v.\n"));
static const char * DumpInputEnv = "FILECHECK_DUMP_INPUT_ON_FAILURE";

static cl::opt<bool> DumpInputOnFailure(
    "dump-input-on-failure",
    cl::init(std::getenv(DumpInputEnv) && *std::getenv(DumpInputEnv)),
    cl::desc("Dump original input to stderr before failing.\n"
             "The value can be also controlled using\n"
             "FILECHECK_DUMP_INPUT_ON_FAILURE environment variable.\n"
             "This option is deprecated in favor of -dump-input=fail.\n"));

// The order of DumpInputValue members affects their precedence, as documented
// for -dump-input below.
enum DumpInputValue {
  DumpInputDefault,
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
             "in the list below has precedence.\n"),
    cl::value_desc("mode"),
    cl::values(clEnumValN(DumpInputHelp, "help",
                          "Explain dump format and quit"),
               clEnumValN(DumpInputAlways, "always", "Always dump input"),
               clEnumValN(DumpInputFail, "fail", "Dump input on failure"),
               clEnumValN(DumpInputNever, "never", "Never dump input")));

typedef cl::list<std::string>::const_iterator prefix_iterator;







static void DumpCommandLine(int argc, char **argv) {
  errs() << "FileCheck command line: ";
  for (int I = 0; I < argc; I++)
    errs() << " " << argv[I];
  errs() << "\n";
}

struct MarkerStyle {
  /// The starting char (before tildes) for marking the line.
  char Lead;
  /// What color to use for this annotation.
  raw_ostream::Colors Color;
  /// A note to follow the marker, or empty string if none.
  std::string Note;
  MarkerStyle() {}
  MarkerStyle(char Lead, raw_ostream::Colors Color,
              const std::string &Note = "")
      : Lead(Lead), Color(Color), Note(Note) {}
};

static MarkerStyle GetMarker(FileCheckDiag::MatchType MatchTy) {
  switch (MatchTy) {
  case FileCheckDiag::MatchFoundAndExpected:
    return MarkerStyle('^', raw_ostream::GREEN);
  case FileCheckDiag::MatchFoundButExcluded:
    return MarkerStyle('!', raw_ostream::RED, "error: no match expected");
  case FileCheckDiag::MatchFoundButWrongLine:
    return MarkerStyle('!', raw_ostream::RED, "error: match on wrong line");
  case FileCheckDiag::MatchFoundButDiscarded:
    return MarkerStyle('!', raw_ostream::CYAN,
                       "discard: overlaps earlier match");
  case FileCheckDiag::MatchNoneAndExcluded:
    return MarkerStyle('X', raw_ostream::GREEN);
  case FileCheckDiag::MatchNoneButExpected:
    return MarkerStyle('X', raw_ostream::RED, "error: no match found");
  case FileCheckDiag::MatchFuzzy:
    return MarkerStyle('?', raw_ostream::MAGENTA, "possible intended match");
  }
  llvm_unreachable_internal("unexpected match type");
}

static void DumpInputAnnotationHelp(raw_ostream &OS) {
  OS << "The following description was requested by -dump-input=help to\n"
     << "explain the input annotations printed by -dump-input=always and\n"
     << "-dump-input=fail:\n\n";

  // Labels for input lines.
  OS << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "L:";
  OS << "     labels line number L of the input file\n";

  // Labels for annotation lines.
  OS << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "T:L";
  OS << "    labels the only match result for a pattern of type T from "
     << "line L of\n"
     << "           the check file\n";
  OS << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "T:L'N";
  OS << "  labels the Nth match result for a pattern of type T from line "
     << "L of\n"
     << "           the check file\n";

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
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "X~~";
  OS << "    marks search range when no match is found, such as:\n"
     << "           - CHECK-NEXT not found (error)\n"
     << "           - CHECK-NOT not found (success, reported if -vv)\n"
     << "           - CHECK-DAG not found after discarded matches (error)\n"
     << "  - ";
  WithColor(OS, raw_ostream::SAVEDCOLOR, true) << "?";
  OS << "      marks fuzzy match when no match is found\n";

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
  OS << "\n\n"
     << "If you are not seeing color above or in input dumps, try: -color\n";
}

/// An annotation for a single input line.
struct InputAnnotation {
  /// The check file line (one-origin indexing) where the directive that
  /// produced this annotation is located.
  unsigned CheckLine;
  /// The index of the match result for this check.
  unsigned CheckDiagIndex;
  /// The label for this annotation.
  std::string Label;
  /// What input line (one-origin indexing) this annotation marks.  This might
  /// be different from the starting line of the original diagnostic if this is
  /// a non-initial fragment of a diagnostic that has been broken across
  /// multiple lines.
  unsigned InputLine;
  /// The column range (one-origin indexing, open end) in which to to mark the
  /// input line.  If InputEndCol is UINT_MAX, treat it as the last column
  /// before the newline.
  unsigned InputStartCol, InputEndCol;
  /// The marker to use.
  MarkerStyle Marker;
  /// Whether this annotation represents a good match for an expected pattern.
  bool FoundAndExpectedMatch;
};

/// Get an abbreviation for the check type.
std::string GetCheckTypeAbbreviation(Check::FileCheckType Ty) {
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
  case Check::CheckEOF:
    return "eof";
  case Check::CheckBadNot:
    return "bad-not";
  case Check::CheckBadCount:
    return "bad-count";
  case Check::CheckNone:
    llvm_unreachable("invalid FileCheckType");
  }
  llvm_unreachable("unknown FileCheckType");
}

static void BuildInputAnnotations(const std::vector<FileCheckDiag> &Diags,
                                  std::vector<InputAnnotation> &Annotations,
                                  unsigned &LabelWidth) {
  // How many diagnostics has the current check seen so far?
  unsigned CheckDiagCount = 0;
  // What's the widest label?
  LabelWidth = 0;
  for (auto DiagItr = Diags.begin(), DiagEnd = Diags.end(); DiagItr != DiagEnd;
       ++DiagItr) {
    InputAnnotation A;

    // Build label, which uniquely identifies this check result.
    A.CheckLine = DiagItr->CheckLine;
    llvm::raw_string_ostream Label(A.Label);
    Label << GetCheckTypeAbbreviation(DiagItr->CheckTy) << ":"
          << DiagItr->CheckLine;
    A.CheckDiagIndex = UINT_MAX;
    auto DiagNext = std::next(DiagItr);
    if (DiagNext != DiagEnd && DiagItr->CheckTy == DiagNext->CheckTy &&
        DiagItr->CheckLine == DiagNext->CheckLine)
      A.CheckDiagIndex = CheckDiagCount++;
    else if (CheckDiagCount) {
      A.CheckDiagIndex = CheckDiagCount;
      CheckDiagCount = 0;
    }
    if (A.CheckDiagIndex != UINT_MAX)
      Label << "'" << A.CheckDiagIndex;
    else
      A.CheckDiagIndex = 0;
    Label.flush();
    LabelWidth = std::max((std::string::size_type)LabelWidth, A.Label.size());

    A.Marker = GetMarker(DiagItr->MatchTy);
    A.FoundAndExpectedMatch =
        DiagItr->MatchTy == FileCheckDiag::MatchFoundAndExpected;

    // Compute the mark location, and break annotation into multiple
    // annotations if it spans multiple lines.
    A.InputLine = DiagItr->InputStartLine;
    A.InputStartCol = DiagItr->InputStartCol;
    if (DiagItr->InputStartLine == DiagItr->InputEndLine) {
      // Sometimes ranges are empty in order to indicate a specific point, but
      // that would mean nothing would be marked, so adjust the range to
      // include the following character.
      A.InputEndCol =
          std::max(DiagItr->InputStartCol + 1, DiagItr->InputEndCol);
      Annotations.push_back(A);
    } else {
      assert(DiagItr->InputStartLine < DiagItr->InputEndLine &&
             "expected input range not to be inverted");
      A.InputEndCol = UINT_MAX;
      Annotations.push_back(A);
      for (unsigned L = DiagItr->InputStartLine + 1, E = DiagItr->InputEndLine;
           L <= E; ++L) {
        // If a range ends before the first column on a line, then it has no
        // characters on that line, so there's nothing to render.
        if (DiagItr->InputEndCol == 1 && L == E)
          break;
        InputAnnotation B;
        B.CheckLine = A.CheckLine;
        B.CheckDiagIndex = A.CheckDiagIndex;
        B.Label = A.Label;
        B.InputLine = L;
        B.Marker = A.Marker;
        B.Marker.Lead = '~';
        B.Marker.Note = "";
        B.InputStartCol = 1;
        if (L != E)
          B.InputEndCol = UINT_MAX;
        else
          B.InputEndCol = DiagItr->InputEndCol;
        B.FoundAndExpectedMatch = A.FoundAndExpectedMatch;
        Annotations.push_back(B);
      }
    }
  }
}

static void DumpAnnotatedInput(raw_ostream &OS, const FileCheckRequest &Req,
                               StringRef InputFileText,
                               std::vector<InputAnnotation> &Annotations,
                               unsigned LabelWidth) {
  OS << "Full input was:\n<<<<<<\n";

  // Sort annotations.
  //
  // First, sort in the order of input lines to make it easier to find relevant
  // annotations while iterating input lines in the implementation below.
  // FileCheck diagnostics are not always reported and recorded in the order of
  // input lines due to, for example, CHECK-DAG and CHECK-NOT.
  //
  // Second, for annotations for the same input line, sort in the order of the
  // FileCheck directive's line in the check file (where there's at most one
  // directive per line) and then by the index of the match result for that
  // directive.  The rationale of this choice is that, for any input line, this
  // sort establishes a total order of annotations that, with respect to match
  // results, is consistent across multiple lines, thus making match results
  // easier to track from one line to the next when they span multiple lines.
  std::sort(Annotations.begin(), Annotations.end(),
            [](const InputAnnotation &A, const InputAnnotation &B) {
              if (A.InputLine != B.InputLine)
                return A.InputLine < B.InputLine;
              if (A.CheckLine != B.CheckLine)
                return A.CheckLine < B.CheckLine;
              // FIXME: Sometimes CHECK-LABEL reports its match twice with
              // other diagnostics in between, and then diag index incrementing
              // fails to work properly, and then this assert fails.  We should
              // suppress one of those diagnostics or do a better job of
              // computing this index.  For now, we just produce a redundant
              // CHECK-LABEL annotation.
              // assert(A.CheckDiagIndex != B.CheckDiagIndex &&
              //        "expected diagnostic indices to be unique within a "
              //        " check line");
              return A.CheckDiagIndex < B.CheckDiagIndex;
            });

  // Compute the width of the label column.
  const unsigned char *InputFilePtr = InputFileText.bytes_begin(),
                      *InputFileEnd = InputFileText.bytes_end();
  unsigned LineCount = InputFileText.count('\n');
  if (InputFileEnd[-1] != '\n')
    ++LineCount;
  unsigned LineNoWidth = std::log10(LineCount) + 1;
  // +3 below adds spaces (1) to the left of the (right-aligned) line numbers
  // on input lines and (2) to the right of the (left-aligned) labels on
  // annotation lines so that input lines and annotation lines are more
  // visually distinct.  For example, the spaces on the annotation lines ensure
  // that input line numbers and check directive line numbers never align
  // horizontally.  Those line numbers might not even be for the same file.
  // One space would be enough to achieve that, but more makes it even easier
  // to see.
  LabelWidth = std::max(LabelWidth, LineNoWidth) + 3;

  // Print annotated input lines.
  auto AnnotationItr = Annotations.begin(), AnnotationEnd = Annotations.end();
  for (unsigned Line = 1;
       InputFilePtr != InputFileEnd || AnnotationItr != AnnotationEnd;
       ++Line) {
    const unsigned char *InputFileLine = InputFilePtr;

    // Print right-aligned line number.
    WithColor(OS, raw_ostream::BLACK, true)
        << format_decimal(Line, LabelWidth) << ": ";

    // For the case where -v and colors are enabled, find the annotations for
    // good matches for expected patterns in order to highlight everything
    // else in the line.  There are no such annotations if -v is disabled.
    std::vector<InputAnnotation> FoundAndExpectedMatches;
    if (Req.Verbose && WithColor(OS).colorsEnabled()) {
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
      WithColor COS(OS);
      bool InMatch = false;
      if (Req.Verbose)
        COS.changeColor(raw_ostream::CYAN, true, true);
      for (unsigned Col = 1; InputFilePtr != InputFileEnd && !Newline; ++Col) {
        bool WasInMatch = InMatch;
        InMatch = false;
        for (auto M : FoundAndExpectedMatches) {
          if (M.InputStartCol <= Col && Col < M.InputEndCol) {
            InMatch = true;
            break;
          }
        }
        if (!WasInMatch && InMatch)
          COS.resetColor();
        else if (WasInMatch && !InMatch)
          COS.changeColor(raw_ostream::CYAN, true, true);
        if (*InputFilePtr == '\n')
          Newline = true;
        else
          COS << *InputFilePtr;
        ++InputFilePtr;
      }
    }
    OS << '\n';
    unsigned InputLineWidth = InputFilePtr - InputFileLine - Newline;

    // Print any annotations.
    while (AnnotationItr != AnnotationEnd &&
           AnnotationItr->InputLine == Line) {
      WithColor COS(OS, AnnotationItr->Marker.Color, true);
      // The two spaces below are where the ": " appears on input lines.
      COS << left_justify(AnnotationItr->Label, LabelWidth) << "  ";
      unsigned Col;
      for (Col = 1; Col < AnnotationItr->InputStartCol; ++Col)
        COS << ' ';
      COS << AnnotationItr->Marker.Lead;
      // If InputEndCol=UINT_MAX, stop at InputLineWidth.
      for (++Col; Col < AnnotationItr->InputEndCol && Col <= InputLineWidth;
           ++Col)
        COS << '~';
      const std::string &Note = AnnotationItr->Marker.Note;
      if (!Note.empty()) {
        // Put the note at the end of the input line.  If we were to instead
        // put the note right after the marker, subsequent annotations for the
        // same input line might appear to mark this note instead of the input
        // line.
        for (; Col <= InputLineWidth; ++Col)
          COS << ' ';
        COS << ' ' << Note;
      }
      COS << '\n';
      ++AnnotationItr;
    }
  }

  OS << ">>>>>>\n";
}

int main(int argc, char **argv) {
  // Enable use of ANSI color codes because FileCheck is using them to
  // highlight text.
  llvm::sys::Process::UseANSIEscapeCodes(true);

  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, /*Overview*/ "", /*Errs*/ nullptr,
                              "FILECHECK_OPTS");
  DumpInputValue DumpInput =
      DumpInputs.empty()
          ? DumpInputDefault
          : *std::max_element(DumpInputs.begin(), DumpInputs.end());
  if (DumpInput == DumpInputHelp) {
    DumpInputAnnotationHelp(outs());
    return 0;
  }
  if (CheckFilename.empty()) {
    errs() << "<check-file> not specified\n";
    return 2;
  }

  FileCheckRequest Req;
  for (auto Prefix : CheckPrefixes)
    Req.CheckPrefixes.push_back(Prefix);

  for (auto CheckNot : ImplicitCheckNot)
    Req.ImplicitCheckNot.push_back(CheckNot);

  bool GlobalDefineError = false;
  for (auto G : GlobalDefines) {
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
  if (!FC.ValidateCheckPrefixes()) {
    errs() << "Supplied check-prefix is invalid! Prefixes must be unique and "
              "start with a letter and contain only alphanumeric characters, "
              "hyphens and underscores\n";
    return 2;
  }

  Regex PrefixRE = FC.buildCheckPrefixRegex();
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
  StringRef CheckFileText = FC.CanonicalizeFile(CheckFile, CheckFileBuffer);

  SM.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(
                            CheckFileText, CheckFile.getBufferIdentifier()),
                        SMLoc());

  if (FC.readCheckFile(SM, CheckFileText, PrefixRE))
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
  StringRef InputFileText = FC.CanonicalizeFile(InputFile, InputFileBuffer);

  SM.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(
                            InputFileText, InputFile.getBufferIdentifier()),
                        SMLoc());

  if (DumpInput == DumpInputDefault)
    DumpInput = DumpInputOnFailure ? DumpInputFail : DumpInputNever;

  std::vector<FileCheckDiag> Diags;
  int ExitCode = FC.checkInput(SM, InputFileText,
                               DumpInput == DumpInputNever ? nullptr : &Diags)
                     ? EXIT_SUCCESS
                     : 1;
  if (DumpInput == DumpInputAlways ||
      (ExitCode == 1 && DumpInput == DumpInputFail)) {
    errs() << "\n"
           << "Input file: "
           << (InputFilename == "-" ? "<stdin>" : InputFilename.getValue())
           << "\n"
           << "Check file: " << CheckFilename << "\n"
           << "\n"
           << "-dump-input=help describes the format of the following dump.\n"
           << "\n";
    std::vector<InputAnnotation> Annotations;
    unsigned LabelWidth;
    BuildInputAnnotations(Diags, Annotations, LabelWidth);
    DumpAnnotatedInput(errs(), Req, InputFileText, Annotations, LabelWidth);
  }

  return ExitCode;
}
