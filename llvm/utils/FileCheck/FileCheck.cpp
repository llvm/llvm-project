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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileCheck.h"
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

static cl::list<std::string> GlobalDefines("D", cl::Prefix,
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

static cl::opt<bool> Verbose("v", cl::init(false),
                             cl::desc("Print directive pattern matches.\n"));

static cl::opt<bool> VerboseVerbose(
    "vv", cl::init(false),
    cl::desc("Print information helpful in diagnosing internal FileCheck\n"
             "issues.  Implies -v.\n"));
static const char * DumpInputEnv = "FILECHECK_DUMP_INPUT_ON_FAILURE";

static cl::opt<bool> DumpInputOnFailure(
    "dump-input-on-failure", cl::init(std::getenv(DumpInputEnv)),
    cl::desc("Dump original input to stderr before failing.\n"
             "The value can be also controlled using\n"
             "FILECHECK_DUMP_INPUT_ON_FAILURE environment variable.\n"));

typedef cl::list<std::string>::const_iterator prefix_iterator;







static void DumpCommandLine(int argc, char **argv) {
  errs() << "FileCheck command line: ";
  for (int I = 0; I < argc; I++)
    errs() << " " << argv[I];
  errs() << "\n";
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  FileCheckRequest Req;
  for (auto Prefix : CheckPrefixes)
    Req.CheckPrefixes.push_back(Prefix);

  for (auto CheckNot : ImplicitCheckNot)
    Req.ImplicitCheckNot.push_back(CheckNot);

  for (auto G : GlobalDefines)
    Req.GlobalDefines.push_back(G);

  Req.AllowEmptyInput = AllowEmptyInput;
  Req.EnableVarScope = EnableVarScope;
  Req.AllowDeprecatedDagOverlap = AllowDeprecatedDagOverlap;
  Req.Verbose = Verbose;
  Req.VerboseVerbose = VerboseVerbose;
  Req.NoCanonicalizeWhiteSpace = NoCanonicalizeWhiteSpace;
  Req.MatchFullLines = MatchFullLines;

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

  std::vector<FileCheckString> CheckStrings;
  if (FC.ReadCheckFile(SM, CheckFileText, PrefixRE, CheckStrings))
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

  int ExitCode =
      FC.CheckInput(SM, InputFileText, CheckStrings) ? EXIT_SUCCESS : 1;
  if (ExitCode == 1 && DumpInputOnFailure)
    errs() << "Full input was:\n<<<<<<\n" << InputFileText << "\n>>>>>>\n";

  return ExitCode;
}
