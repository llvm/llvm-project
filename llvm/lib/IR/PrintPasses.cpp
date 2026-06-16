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
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

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
// reported) or the changed IR will be reported. Also, the -filter-print-funcs
// and -print-module-scope options will do similar filtering based on function
// name, reporting changed IRs as functions(or modules if -print-module-scope is
// specified) for a particular function or indicating that the IR has been
// filtered out. The extra options can be combined, allowing only changed IRs
// for certain passes on certain functions to be reported in different formats,
// with the rest being reported as filtered out.  The -print-before-changed
// option will print the IR as it was before each pass that changed it. The
// optional value of quiet will only report when the IR changes, suppressing all
// other messages, including the initial IR. The optional values are separated
// by commas. The value "attrs-only" will only print function attribute changes
// when only function attributes changed. Without "attrs-only", attribute-only
// changes keep the historical behavior: -print-changed reports the change by
// printing the changed IR. With "attrs-only", passes that only change function
// attributes print a compact before/after attribute report instead. Other IR
// changes still use the normal print-changed output.
// The hash modes use structural hashes to avoid printing and comparing the full
// IR after every pass. This is faster, but it reports changes according to what
// the hash encodes. The default text mode remains the reference mode for
// faithfully detecting changes in the printed IR. The "hash-func" mode reports
// changed functions, while "hash-bb" can narrow reports to changed basic
// blocks when block hashes are available. This avoids printing unchanged basic
// blocks, which has been effective in empirical measurements when most blocks
// are unchanged.
// The values "diff" and "diff-quiet"
// will present the changes in a form similar to a patch, in either verbose or
// quiet mode, respectively. The lines that are removed and added are prefixed
// with '-' and '+', respectively. The -filter-print-funcs and -filter-passes
// can be used to filter the output. This reporter relies on the linux diff
// utility to do comparisons and insert the prefixes. For systems that do not
// have the necessary facilities, the error message will be shown in place of
// the expected output.
ChangePrinter llvm::PrintChanged = ChangePrinter::None;
static bool PrintChangedAttributeDiffs = false;
static ChangePrinterHashMode PrintChangedHashMode = ChangePrinterHashMode::None;

namespace {
enum class ChangePrinterFormat { Text, Diff, ColourDiff, DotCfg };

struct PrintChangedOption {
  void reportError(StringRef Value) const {
    reportFatalUsageError(
        Twine("invalid argument '") + Value +
        "' to -print-changed=; expected a comma-separated list of quiet, "
        "hash, hash-func, hash-bb, attrs-only, diff, diff-quiet, cdiff, "
        "cdiff-quiet, dot-cfg, or dot-cfg-quiet");
  }

  void operator=(const std::string &Value) const {
    PrintChanged = ChangePrinter::Verbose;
    PrintChangedAttributeDiffs = false;
    PrintChangedHashMode = ChangePrinterHashMode::None;

    bool Quiet = false;
    std::optional<ChangePrinterFormat> Format;
    std::optional<std::string> HashModeValue;

    SmallVector<StringRef> Values;
    StringRef(Value).split(Values, ',', -1, false);
    if (Values.empty())
      Values.push_back("");

    auto SetFormat = [&](StringRef V, ChangePrinterFormat F) {
      if (Format && *Format != F)
        reportError(V);
      Format = F;
    };
    auto SetHashMode = [&](StringRef V, ChangePrinterHashMode H) {
      if (PrintChangedHashMode != ChangePrinterHashMode::None &&
          PrintChangedHashMode != H)
        reportError(V);
      PrintChangedHashMode = H;
      if (!HashModeValue)
        HashModeValue = V.str();
    };

    for (StringRef V : Values) {
      if (V.empty())
        continue;
      if (V == "quiet") {
        Quiet = true;
      } else if (V == "hash" || V == "hash-func") {
        SetHashMode(V, ChangePrinterHashMode::Function);
      } else if (V == "hash-bb") {
        SetHashMode(V, ChangePrinterHashMode::BasicBlock);
      } else if (V == "attrs-only") {
        PrintChangedAttributeDiffs = true;
      } else if (V == "diff") {
        SetFormat(V, ChangePrinterFormat::Diff);
      } else if (V == "diff-quiet") {
        SetFormat(V, ChangePrinterFormat::Diff);
        Quiet = true;
      } else if (V == "cdiff") {
        SetFormat(V, ChangePrinterFormat::ColourDiff);
      } else if (V == "cdiff-quiet") {
        SetFormat(V, ChangePrinterFormat::ColourDiff);
        Quiet = true;
      } else if (V == "dot-cfg") {
        SetFormat(V, ChangePrinterFormat::DotCfg);
      } else if (V == "dot-cfg-quiet") {
        SetFormat(V, ChangePrinterFormat::DotCfg);
        Quiet = true;
      } else {
        reportError(V);
      }
    }

    switch (Format.value_or(ChangePrinterFormat::Text)) {
    case ChangePrinterFormat::Text:
      PrintChanged = Quiet ? ChangePrinter::Quiet : ChangePrinter::Verbose;
      break;
    case ChangePrinterFormat::Diff:
      PrintChanged =
          Quiet ? ChangePrinter::DiffQuiet : ChangePrinter::DiffVerbose;
      break;
    case ChangePrinterFormat::ColourDiff:
      PrintChanged = Quiet ? ChangePrinter::ColourDiffQuiet
                           : ChangePrinter::ColourDiffVerbose;
      break;
    case ChangePrinterFormat::DotCfg:
      PrintChanged =
          Quiet ? ChangePrinter::DotCfgQuiet : ChangePrinter::DotCfgVerbose;
      break;
    }

    if (PrintChangedAttributeDiffs && Format &&
        *Format != ChangePrinterFormat::Text)
      reportError("attrs-only");
    if (PrintChangedHashMode != ChangePrinterHashMode::None && Format &&
        *Format != ChangePrinterFormat::Text)
      reportError(*HashModeValue);
  }
};
} // namespace

static PrintChangedOption PrintChangedOptionLoc;

static cl::opt<PrintChangedOption, true, cl::parser<std::string>>
    PrintChangedOpt("print-changed", cl::desc("Print changed IRs"), cl::Hidden,
                    cl::ValueOptional, cl::location(PrintChangedOptionLoc));

bool llvm::shouldPrintChangedAttributeDiffs() {
  return PrintChangedAttributeDiffs;
}

bool llvm::shouldUsePrintChangedHash() {
  return PrintChangedHashMode != ChangePrinterHashMode::None;
}

ChangePrinterHashMode llvm::getPrintChangedHashMode() {
  return PrintChangedHashMode;
}

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
  return PrintFuncNames.empty() || PrintFuncNames.contains(FunctionName);
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
          PrintChanged);
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
                                PrintChanged)) {
    const char *Reason =
        IsInteresting ? " omitted because no change" : " filtered out";
    errs() << "*** IR Dump After " << PassName;
    if (!PassID.empty())
      errs() << " (" << PassID << ")";
    errs() << " on " << IRName + Reason + " ***\n";
  }
}
