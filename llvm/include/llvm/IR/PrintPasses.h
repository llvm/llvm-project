//===- PrintPasses.h - Determining whether/when to print IR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PRINTPASSES_H
#define LLVM_IR_PRINTPASSES_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include <vector>

namespace llvm {

enum class ChangePrinter {
  None,
  Verbose,
  Quiet,
  DiffVerbose,
  DiffQuiet,
  ColourDiffVerbose,
  ColourDiffQuiet,
  DotCfgVerbose,
  DotCfgQuiet
};

extern cl::opt<ChangePrinter> PrintChanged;

// Returns true if printing before/after some pass is enabled, whether all
// passes or a specific pass.
bool shouldPrintBeforeSomePass();
bool shouldPrintAfterSomePass();

// Returns true if we should print before/after a specific pass. The argument
// should be the pass ID, e.g. "instcombine".
bool shouldPrintBeforePass(StringRef PassID);
bool shouldPrintAfterPass(StringRef PassID);

// Returns true if we should print before/after all passes.
bool shouldPrintBeforeAll();
bool shouldPrintAfterAll();

// The list of passes to print before/after, if we only want to print
// before/after specific passes.
std::vector<std::string> printBeforePasses();
std::vector<std::string> printAfterPasses();

// Returns true if we should always print the entire module.
bool forcePrintModuleIR();

// Return true if -filter-passes is empty or contains the pass name.
bool isPassInPrintList(StringRef PassName);

// Returns true if we should print the function.
bool isFunctionInPrintList(StringRef FunctionName);

// Perform a system based diff between \p Before and \p After, using \p
// OldLineFormat, \p NewLineFormat, and \p UnchangedLineFormat to control the
// formatting of the output. Return an error message for any failures instead
// of the diff.
std::string doSystemDiff(StringRef Before, StringRef After,
                         StringRef OldLineFormat, StringRef NewLineFormat,
                         StringRef UnchangedLineFormat);

} // namespace llvm

#endif // LLVM_IR_PRINTPASSES_H
