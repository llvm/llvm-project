//===- PrintPasses.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PrintPasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

using namespace llvm;

// Print IR out before/after specified passes.
static cl::list<std::string>
    PrintBefore("print-before",
                llvm::cl::desc("Print IR before specified passes"),
                cl::CommaSeparated, cl::Hidden);

static cl::list<std::string>
    PrintAfter("print-after", llvm::cl::desc("Print IR after specified passes"),
               cl::CommaSeparated, cl::Hidden);

static cl::opt<bool> PrintBeforeAll("print-before-all",
                                    llvm::cl::desc("Print IR before each pass"),
                                    cl::init(false), cl::Hidden);
static cl::opt<bool> PrintAfterAll("print-after-all",
                                   llvm::cl::desc("Print IR after each pass"),
                                   cl::init(false), cl::Hidden);

// Print out the IR after passes, similar to -print-after-all except that it
// only prints the IR after passes that change the IR. Those passes that do not
// make changes to the IR are reported as not making any changes. In addition,
// the initial IR is also reported.  Other hidden options affect the output from
// this option. -filter-passes will limit the output to the named passes that
// actually change the IR and other passes are reported as filtered out. The
// specified passes will either be reported as making no changes (with no IR
// reported) or the changed IR will be reported. Also, the -filter-print-funcs,
// -filter-print-source-locs and -print-module-scope options will do similar
// filtering based on function name or source location, reporting changed IRs as
// functions(or modules if -print-module-scope is specified) for a particular
// function or indicating that the IR has been filtered out. The extra options
// can be combined, allowing only changed IRs for certain passes on certain
// functions or source locations to be reported in different formats, with the
// rest being reported as filtered out.  The -print-before-changed
// option will print the IR as it was before each pass that changed it. The
// optional value of quiet will only report when the IR changes, suppressing all
// other messages, including the initial IR. The values "diff" and "diff-quiet"
// will present the changes in a form similar to a patch, in either verbose or
// quiet mode, respectively. The lines that are removed and added are prefixed
// with '-' and '+', respectively. The -filter-print-funcs,
// -filter-print-source-locs and -filter-passes can be used to filter the
// output. This reporter relies on the linux diff utility to do comparisons and
// insert the prefixes. For systems that do not have the necessary facilities,
// the error message will be shown in place of the expected output.
cl::opt<ChangePrinter> llvm::PrintChanged(
    "print-changed", cl::desc("Print changed IRs"), cl::Hidden,
    cl::ValueOptional, cl::init(ChangePrinter::None),
    cl::values(
        clEnumValN(ChangePrinter::Quiet, "quiet", "Run in quiet mode"),
        clEnumValN(ChangePrinter::DiffVerbose, "diff",
                   "Display patch-like changes"),
        clEnumValN(ChangePrinter::DiffQuiet, "diff-quiet",
                   "Display patch-like changes in quiet mode"),
        clEnumValN(ChangePrinter::ColourDiffVerbose, "cdiff",
                   "Display patch-like changes with color"),
        clEnumValN(ChangePrinter::ColourDiffQuiet, "cdiff-quiet",
                   "Display patch-like changes in quiet mode with color"),
        clEnumValN(ChangePrinter::DotCfgVerbose, "dot-cfg",
                   "Create a website with graphical changes"),
        clEnumValN(ChangePrinter::DotCfgQuiet, "dot-cfg-quiet",
                   "Create a website with graphical changes in quiet mode"),
        // Sentinel value for unspecified option.
        clEnumValN(ChangePrinter::Verbose, "", "")));

// An option for specifying the diff used by print-changed=[diff | diff-quiet]
static cl::opt<std::string>
    DiffBinary("print-changed-diff-path", cl::Hidden, cl::init("diff"),
               cl::desc("system diff used by change reporters"));

static cl::opt<bool>
    PrintModuleScope("print-module-scope",
                     cl::desc("When printing IR for print-[before|after]{-all} "
                              "always print a module IR"),
                     cl::init(false), cl::Hidden);

static cl::opt<bool> LoopPrintFuncScope(
    "print-loop-func-scope",
    cl::desc("When printing IR for print-[before|after]{-all} "
             "for a loop pass, always print function IR"),
    cl::init(false), cl::Hidden);

// See the description for -print-changed for an explanation of the use
// of this option.
static cl::list<std::string> FilterPasses(
    "filter-passes", cl::value_desc("pass names"),
    cl::desc("Only consider IR changes for passes whose names "
             "match the specified value. No-op without -print-changed"),
    cl::CommaSeparated, cl::Hidden);

static cl::list<std::string>
    PrintFuncsList("filter-print-funcs", cl::value_desc("function names"),
                   cl::desc("Only print IR for functions whose name "
                            "match this for all print-[before|after][-all] "
                            "options"),
                   cl::CommaSeparated, cl::Hidden);

static cl::list<std::string> PrintSourceLocs(
    "filter-print-source-locs", cl::value_desc("file:line[,line-line][,line]"),
    cl::desc("Only print IR containing matching source locations"), cl::Hidden);

/// This is a helper to determine whether to print IR before or
/// after a pass.

bool llvm::shouldPrintBeforeSomePass() {
  return PrintBeforeAll || !PrintBefore.empty();
}

bool llvm::shouldPrintAfterSomePass() {
  return PrintAfterAll || !PrintAfter.empty();
}

static bool shouldPrintBeforeOrAfterPass(StringRef PassID,
                                         ArrayRef<std::string> PassesToPrint) {
  return llvm::is_contained(PassesToPrint, PassID);
}

bool llvm::shouldPrintBeforeAll() { return PrintBeforeAll; }

bool llvm::shouldPrintAfterAll() { return PrintAfterAll; }

bool llvm::shouldPrintBeforePass(StringRef PassID) {
  return PrintBeforeAll || shouldPrintBeforeOrAfterPass(PassID, PrintBefore);
}

bool llvm::shouldPrintAfterPass(StringRef PassID) {
  return PrintAfterAll || shouldPrintBeforeOrAfterPass(PassID, PrintAfter);
}

std::vector<std::string> llvm::printBeforePasses() {
  return std::vector<std::string>(PrintBefore);
}

std::vector<std::string> llvm::printAfterPasses() {
  return std::vector<std::string>(PrintAfter);
}

bool llvm::forcePrintModuleIR() { return PrintModuleScope; }

bool llvm::forcePrintFuncIR() { return LoopPrintFuncScope; }

bool llvm::isPassInPrintList(StringRef PassName) {
  static const StringSet<> Set(llvm::from_range, FilterPasses);
  return Set.empty() || Set.contains(PassName);
}

bool llvm::isFilterPassesEmpty() { return FilterPasses.empty(); }

bool llvm::isFunctionInPrintList(StringRef FunctionName) {
  static const StringSet<> PrintFuncNames(llvm::from_range, PrintFuncsList);
  return PrintFuncNames.empty() || PrintFuncNames.contains(FunctionName) ||
         PrintFuncNames.contains("*");
}

namespace {

struct PrintLineRange {
  unsigned First;
  unsigned Last;
};

struct PrintSourceLocSpec {
  std::string File;
  SmallVector<PrintLineRange, 4> Lines;
};

[[noreturn]] void reportBadLocSpec(StringRef Spec) {
  report_fatal_error(Twine("Invalid -filter-print-source-locs value '") + Spec +
                     "'. Expected file:line[,line-line][,line].");
}

std::string normalizeSlashes(StringRef Path) {
  std::string Result = Path.str();
  for (char &C : Result)
    if (C == '\\')
      C = '/';
  return Result;
}

bool parseLine(StringRef LineText, unsigned &Line) {
  return !LineText.empty() && !LineText.getAsInteger(10, Line);
}

PrintLineRange parseLineRange(StringRef RangeText, StringRef FullSpec) {
  auto [FirstText, LastText] = RangeText.split('-');

  unsigned First;
  if (!parseLine(FirstText, First))
    reportBadLocSpec(FullSpec);

  if (LastText.empty())
    return {First, First};

  unsigned Last;
  if (!parseLine(LastText, Last) || Last < First)
    reportBadLocSpec(FullSpec);

  return {First, Last};
}

std::vector<PrintSourceLocSpec> parseSourceLocSpecs() {
  std::vector<PrintSourceLocSpec> Result;
  for (const std::string &RawSpec : PrintSourceLocs) {
    StringRef Spec(RawSpec);
    auto [File, LineSpec] = Spec.rsplit(':');
    if (File.empty() || LineSpec.empty())
      reportBadLocSpec(Spec);

    PrintSourceLocSpec Parsed;
    Parsed.File = normalizeSlashes(File);
    for (StringRef RangeText : llvm::split(LineSpec, ",")) {
      Parsed.Lines.push_back(parseLineRange(RangeText, Spec));
    }
    Result.push_back(std::move(Parsed));
  }
  return Result;
}

ArrayRef<PrintSourceLocSpec> getSourceLocSpecs() {
  static const std::vector<PrintSourceLocSpec> Specs = parseSourceLocSpecs();
  return Specs;
}

std::string makeDebugLocPath(StringRef Directory, StringRef Filename) {
  std::string NormalizedFilename = normalizeSlashes(Filename);
  if (Directory.empty() || sys::path::is_absolute(NormalizedFilename))
    return NormalizedFilename;

  std::string NormalizedDirectory = normalizeSlashes(Directory);
  if (NormalizedDirectory.empty())
    return NormalizedFilename;
  if (NormalizedDirectory.back() == '/')
    return NormalizedDirectory + NormalizedFilename;
  return NormalizedDirectory + "/" + NormalizedFilename;
}

bool matchesFile(StringRef SpecFile, StringRef Directory, StringRef Filename) {
  std::string LocFile = normalizeSlashes(Filename);
  std::string LocPath = makeDebugLocPath(Directory, Filename);

  if (SpecFile == LocFile || SpecFile == LocPath)
    return true;

  StringRef LocFileRef(LocFile);
  StringRef LocPathRef(LocPath);
  if (sys::path::filename(LocFileRef) == SpecFile)
    return true;

  std::string Suffix = (Twine("/") + SpecFile).str();
  return LocFileRef.ends_with(Suffix) || LocPathRef.ends_with(Suffix);
}

bool matchesLine(ArrayRef<PrintLineRange> Ranges, unsigned Line) {
  return any_of(Ranges, [Line](const PrintLineRange &Range) {
    return Range.First <= Line && Line <= Range.Last;
  });
}

bool matchesSourceLocSpec(const DebugLoc &Loc, const PrintSourceLocSpec &Spec) {
  auto *Scope = dyn_cast_or_null<DIScope>(Loc.getScope());
  return Scope &&
         matchesFile(Spec.File, Scope->getDirectory(), Scope->getFilename()) &&
         matchesLine(Spec.Lines, Loc.getLine());
}

} // namespace

bool llvm::isSourceLocInPrintList(const DebugLoc &Loc) {
  ArrayRef<PrintSourceLocSpec> Specs = getSourceLocSpecs();
  if (Specs.empty())
    return true;

  for (DebugLoc CurLoc = Loc; CurLoc; CurLoc = CurLoc.getInlinedAt()) {
    if (any_of(Specs, [&CurLoc](const PrintSourceLocSpec &Spec) {
          return matchesSourceLocSpec(CurLoc, Spec);
        }))
      return true;
  }
  return false;
}

bool llvm::isSourceLocFilterEmpty() { return PrintSourceLocs.empty(); }

bool llvm::shouldPrintFunction(const Function &F) {
  if (!isFunctionInPrintList(F.getName()))
    return false;

  if (isSourceLocFilterEmpty())
    return true;

  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB)
      if (isSourceLocInPrintList(I.getDebugLoc()))
        return true;
  return false;
}

std::error_code cleanUpTempFilesImpl(ArrayRef<std::string> FileName,
                                     unsigned N) {
  std::error_code RC;
  for (unsigned I = 0; I < N; ++I) {
    std::error_code EC = sys::fs::remove(FileName[I]);
    if (EC)
      RC = EC;
  }
  return RC;
}

std::error_code llvm::prepareTempFiles(SmallVector<int> &FD,
                                       ArrayRef<StringRef> SR,
                                       SmallVector<std::string> &FileName) {
  assert(FD.size() >= SR.size() && FileName.size() == FD.size() &&
         "Unexpected array sizes");
  std::error_code EC;
  unsigned I = 0;
  for (; I < FD.size(); ++I) {
    if (FD[I] == -1) {
      SmallVector<char, 200> SV;
      EC = sys::fs::createTemporaryFile("tmpfile", "txt", FD[I], SV);
      if (EC)
        break;
      FileName[I] = Twine(SV).str();
    }
    if (I < SR.size()) {
      EC = sys::fs::openFileForWrite(FileName[I], FD[I]);
      if (EC)
        break;
      raw_fd_ostream OutStream(FD[I], /*shouldClose=*/true);
      if (FD[I] == -1) {
        EC = make_error_code(errc::io_error);
        break;
      }
      OutStream << SR[I];
    }
  }
  if (EC && I > 0)
    // clean up created temporary files
    cleanUpTempFilesImpl(FileName, I);
  return EC;
}

std::error_code llvm::cleanUpTempFiles(ArrayRef<std::string> FileName) {
  return cleanUpTempFilesImpl(FileName, FileName.size());
}

std::string llvm::doSystemDiff(StringRef Before, StringRef After,
                               StringRef OldLineFormat, StringRef NewLineFormat,
                               StringRef UnchangedLineFormat) {
  auto BypassSandbox = sys::sandbox::scopedDisable();

  // Store the 2 bodies into temporary files and call diff on them
  // to get the body of the node.
  static SmallVector<int> FD{-1, -1, -1};
  SmallVector<StringRef> SR{Before, After};
  static SmallVector<std::string> FileName{"", "", ""};
  if (prepareTempFiles(FD, SR, FileName))
    return "Unable to create temporary file.";

  static ErrorOr<std::string> DiffExe = sys::findProgramByName(DiffBinary);
  if (!DiffExe)
    return "Unable to find diff executable.";

  SmallString<128> OLF, NLF, ULF;
  ("--old-line-format=" + OldLineFormat).toVector(OLF);
  ("--new-line-format=" + NewLineFormat).toVector(NLF);
  ("--unchanged-line-format=" + UnchangedLineFormat).toVector(ULF);

  StringRef Args[] = {DiffBinary, "-w", "-d",        OLF,
                      NLF,        ULF,  FileName[0], FileName[1]};
  std::optional<StringRef> Redirects[] = {std::nullopt, StringRef(FileName[2]),
                                          std::nullopt};
  int Result = sys::ExecuteAndWait(*DiffExe, Args, std::nullopt, Redirects);
  if (Result < 0)
    return "Error executing system diff.";
  std::string Diff;
  auto B = MemoryBuffer::getFile(FileName[2]);
  if (B && *B)
    Diff = (*B)->getBuffer().str();
  else
    return "Unable to read result.";

  if (cleanUpTempFiles(FileName))
    return "Unable to remove temporary file.";

  return Diff;
}

void llvm::reportChangedIR(StringRef Before, StringRef After,
                           StringRef PassName, StringRef PassID,
                           StringRef IRName, bool IsInteresting,
                           bool ShouldReport) {
  if (!ShouldReport && IsInteresting)
    return;

  if (IsInteresting && Before != After) {
    errs() << ("*** IR Dump After " + PassName + " (" + PassID + ") on " +
               IRName + " ***\n");
    switch (PrintChanged) {
    case ChangePrinter::None:
      llvm_unreachable("");
    case ChangePrinter::Quiet:
    case ChangePrinter::Verbose:
    case ChangePrinter::DotCfgQuiet:   // unimplemented
    case ChangePrinter::DotCfgVerbose: // unimplemented
      errs() << After;
      break;
    case ChangePrinter::DiffQuiet:
    case ChangePrinter::DiffVerbose:
    case ChangePrinter::ColourDiffQuiet:
    case ChangePrinter::ColourDiffVerbose: {
      bool Color = llvm::is_contained(
          {ChangePrinter::ColourDiffQuiet, ChangePrinter::ColourDiffVerbose},
          PrintChanged.getValue());
      StringRef Removed = Color ? "\033[31m-%l\033[0m\n" : "-%l\n";
      StringRef Added = Color ? "\033[32m+%l\033[0m\n" : "+%l\n";
      StringRef NoChange = " %l\n";
      errs() << doSystemDiff(Before, After, Removed, Added, NoChange);
      break;
    }
    }
  } else if (llvm::is_contained({ChangePrinter::Verbose,
                                 ChangePrinter::DiffVerbose,
                                 ChangePrinter::ColourDiffVerbose},
                                PrintChanged.getValue())) {
    const char *Reason =
        IsInteresting ? " omitted because no change" : " filtered out";
    errs() << "*** IR Dump After " << PassName;
    if (!PassID.empty())
      errs() << " (" << PassID << ")";
    errs() << " on " << IRName + Reason + " ***\n";
  }
}
