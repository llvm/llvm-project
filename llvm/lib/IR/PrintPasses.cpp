//===- PrintPasses.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PrintPasses.h"
#include "llvm/Support/CommandLine.h"
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

static cl::opt<bool>
    PrintModuleScope("print-module-scope",
                     cl::desc("When printing IR for print-[before|after]{-all} "
                              "always print a module IR"),
                     cl::init(false), cl::Hidden);

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

bool llvm::isFunctionInPrintList(StringRef FunctionName) {
  static std::unordered_set<std::string> PrintFuncNames(PrintFuncsList.begin(),
                                                        PrintFuncsList.end());
  return PrintFuncNames.empty() ||
         PrintFuncNames.count(std::string(FunctionName));
}
