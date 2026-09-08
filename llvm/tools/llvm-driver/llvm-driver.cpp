//===-- llvm-driver.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define LLVM_DRIVER_TOOL(tool, entry)                                          \
  int entry##_main(int argc, char **argv, const llvm::ToolContext &);
#include "LLVMDriverTools.def"

constexpr char subcommands[] =
#define LLVM_DRIVER_TOOL(tool, entry) "  " tool "\n"
#include "LLVMDriverTools.def"
    ;

static void printHelpMessage() {
  llvm::outs() << "OVERVIEW: llvm toolchain driver\n\n"
               << "USAGE: llvm [subcommand] [options]\n\n"
               << "SUBCOMMANDS:\n\n"
               << subcommands
               << "\n  Type \"llvm <subcommand> --help\" to get more help on a "
                  "specific subcommand\n\n"
               << "OPTIONS:\n\n  --help - Display this message\n";
}

int main(int Argc, char **Argv) {
  const CallableTool Tools[] = {
#define LLVM_DRIVER_TOOL(tool, entry) {tool, entry##_main},
#include "LLVMDriverTools.def"
  };

  LLVMToolSession Session(Argc, Argv, Tools);

  StringRef Stem = sys::path::stem(Argv[0]);
  if (Stem.equals_insensitive("llvm") &&
      (Argc == 1 || (Argc == 2 && StringRef(Argv[1]) == "--help"))) {
    printHelpMessage();
    return Argc == 1 ? 1 : 0;
  }

  SmallVector<const char *, 16> Args(Argv, Argv + Argc);
  int Result = Session.callTool(Args);
  if (Result == -1) {
    printHelpMessage();
    return 1;
  }
  return Result;
}
