//===- PrintPasses.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PrintPasses.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Threading.h"
#include <atomic>
#include <unordered_set>

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

static cl::opt<std::string> IRDumpDirectory(
    "ir-dump-directory",
    cl::desc("If specified, IR printed using the "
             "-print-[before|after]{-all} options will be dumped into "
             "files in this directory rather than written to stderr"),
    cl::Hidden, cl::value_desc("filename"));

StringRef llvm::irDumpDirectory() { return IRDumpDirectory; }

static cl::opt<std::string> IRDumpFilenameFormat(
    "ir-dump-filename-format",
    cl::desc("Specifies how filenames are generated when dumping IR to files."
             " Supported values are 'default' and 'sortable'."),
    cl::Hidden, cl::init("default"));

static cl::opt<bool> IRDumpFilenamePrependThreadId(
    "ir-dump-filename-prepend-thread-id",
    cl::desc("Prepend the filename with the current thread id. Usefule for"
             " multi-threaded LTO compiles"),
    cl::Hidden);

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
// other messages, including the initial IR. The values "diff" and "diff-quiet"
// will present the changes in a form similar to a patch, in either verbose or
// quiet mode, respectively. The lines that are removed and added are prefixed
// with '-' and '+', respectively. The -filter-print-funcs and -filter-passes
// can be used to filter the output.  This reporter relies on the linux diff
// utility to do comparisons and insert the prefixes. For systems that do not
// have the necessary facilities, the error message will be shown in place of
// the expected output.
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
  static std::unordered_set<std::string> Set(FilterPasses.begin(),
                                             FilterPasses.end());
  return Set.empty() || Set.count(std::string(PassName));
}

bool llvm::isFilterPassesEmpty() { return FilterPasses.empty(); }

bool llvm::isFunctionInPrintList(StringRef FunctionName) {
  static std::unordered_set<std::string> PrintFuncNames(PrintFuncsList.begin(),
                                                        PrintFuncsList.end());
  return PrintFuncNames.empty() ||
         PrintFuncNames.count(std::string(FunctionName));
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

namespace {
const char *getFileSuffix(IRDumpFileSuffixType Type) {
  if (Type == IRDumpFileSuffixType::Before)
    return "before";
  if (Type == IRDumpFileSuffixType::After)
    return "after";
  if (Type == IRDumpFileSuffixType::Invalidated)
    return "invalidated";
  return "unknown";
}
} // namespace

std::string llvm::irDumpFilename(StringRef Kind, StringRef PassNameIn,
                                 std::optional<unsigned> PassNumber,
                                 IRDumpFileSuffixType SuffixType) {

  // sanitize PassName
  std::string PassName = PassNameIn.str();
  for (char &c : PassName) {
    if (!isAlnum(c) && c != '-')
      c = '_';
  }

  // One-time initialization of format string and generation of placeholders
  static std::string Fmt;
  static std::once_flag InitFmtFlag;
  static auto InitFmt = []() {
    static const char *FmtTidPrefix = "{6,0+8}-";
    static const char *FmtDefault = "{1}-{2}-{3}-{5}.ll";
    static const char *FmtSortable = "{0,0+8}-{1,0+8}-{2}-{3}-{4}-{5}.ll";
    if (IRDumpFilenamePrependThreadId)
      Fmt += FmtTidPrefix;
    if (IRDumpFilenameFormat == "sortable")
      Fmt += FmtSortable;
    else if (IRDumpFilenameFormat == "default")
      Fmt += FmtDefault;
    else {
      Fmt += FmtDefault;
      errs() << "warning: invalid value for -ir-dump-filename-format '"
             << IRDumpFilenameFormat << "'; using 'default'\n";
    }
  };
  std::call_once(InitFmtFlag, InitFmt);

  // Generate filename
  static std::atomic<int> Ordinal;
  int Ord = Ordinal++;
  std::string Filename = formatv(false, Fmt.c_str(),
                                 /* 0 */ Ord,
                                 /* 1 */ PassNumber.value_or(Ord),
                                 /* 2 */ Kind,
                                 /* 3 */ PassName,
                                 /* 4 */ static_cast<size_t>(SuffixType),
                                 /* 5 */ getFileSuffix(SuffixType),
                                 /* 6 */ llvm::get_threadid());

  // Generate path
  const StringRef DumpDir = irDumpDirectory();
  assert(!DumpDir.empty() &&
         "The flag -ir-dump-directory must be passed to dump IR to files");
  SmallString<128> ResultPath;
  sys::path::append(ResultPath, DumpDir, Filename);
  return std::string(ResultPath);
}

std::string llvm::IRDumpStream::buildBanner(llvm::StringRef PassName,
                                            llvm::StringRef PassID,
                                            IRDumpFileSuffixType SuffixType) {

  const char *Suffix = "Unknown";
  if (SuffixType == IRDumpFileSuffixType::Before)
    Suffix = "Before";
  else if (SuffixType == IRDumpFileSuffixType::After)
    Suffix = "After";
  else if (SuffixType == IRDumpFileSuffixType::Invalidated)
    Suffix = "Invalidated";

  return formatv("*** IR Dump {0} {1} ({2}) ***", Suffix, PassName, PassID)
      .str();
}

static std::string extractPassName(StringRef Banner,
                                   IRDumpFileSuffixType &PhaseOut) {
  if (Banner.consume_front("*** IR Dump Before ")) {
    PhaseOut = IRDumpFileSuffixType::Before;
  } else if (Banner.consume_front("*** IR Dump After ")) {
    PhaseOut = IRDumpFileSuffixType::After;
  } else {
    return std::string();
  }

  size_t open = Banner.find(" (");
  if (open == StringRef::npos) {
    return std::string();
  }

  open += 2;
  size_t close = Banner.find(')', open);
  if (close == StringRef::npos) {
    return std::string();
  }

  return Banner.substr(open, close - open).str();
}

llvm::IRDumpStream::IRDumpStream(StringRef Kind, StringRef Banner,
                                 raw_ostream &fallback)
    : fstream(nullptr), fallback(fallback) {

  IRDumpFileSuffixType Phase;
  std::string PassName = extractPassName(Banner, Phase);

  StringRef Dir = irDumpDirectory();
  if (Dir.empty() || PassName.empty())
    return;

  std::string DumpIRFilename =
      irDumpFilename(Kind, PassName, std::nullopt, Phase);

  std::error_code EC = llvm::sys::fs::create_directories(Dir);
  if (EC) {
    report_fatal_error(Twine("Failed to create directory ") + Dir +
                       " to support -ir-dump-directory: " + EC.message());
  }
  if (sys::fs::exists(DumpIRFilename)) {
    errs() << "warning: overwriting existing file '" << DumpIRFilename << "'\n";
  }
  int FD = 0;
  EC = sys::fs::openFile(DumpIRFilename, FD, sys::fs::CD_OpenAlways,
                         sys::fs::FA_Write, sys::fs::OF_Text);
  if (EC) {
    report_fatal_error(Twine("Failed to open ") + DumpIRFilename +
                       " to support -ir-dump-directory: " + EC.message());
  }
  // return FD;
  fstream = new raw_fd_ostream(FD, /* shouldClose */ true);
}

llvm::IRDumpStream::~IRDumpStream() {
  if (fstream) {
    fstream->close();
    delete fstream;
  }
}

raw_ostream &llvm::IRDumpStream::os() { return fstream ? *fstream : fallback; }
