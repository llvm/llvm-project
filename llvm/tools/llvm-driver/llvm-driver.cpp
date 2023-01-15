//===-- llvm-driver.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

#define LLVM_DRIVER_TOOL(tool, entry) int entry##_main(int argc, char **argv);
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
               << "OPTIONS:\n\n  --help - Display this message";
}

static int findTool(int Argc, char **Argv) {
  if (!Argc) {
    printHelpMessage();
    return 1;
  }

  StringRef ToolName = Argv[0];

  if (ToolName == "--help") {
    printHelpMessage();
    return 0;
  }

  StringRef Stem = sys::path::stem(ToolName);
  auto Is = [=](StringRef Tool) {
    auto IsImpl = [=](StringRef Stem) {
      auto I = Stem.rfind_insensitive(Tool);
      return I != StringRef::npos && (I + Tool.size() == Stem.size() ||
                                      !llvm::isAlnum(Stem[I + Tool.size()]));
    };
    for (StringRef S : {Stem, sys::path::filename(ToolName)})
      if (IsImpl(S))
        return true;
    return false;
  };

#define LLVM_DRIVER_TOOL(tool, entry)                                          \
  if (Is(tool))                                                                \
    return entry##_main(Argc, Argv);
#include "LLVMDriverTools.def"

  if (Is("llvm"))
    return findTool(Argc - 1, Argv + 1);

  printHelpMessage();
  return 1;
}

extern bool IsLLVMDriver;

int main(int Argc, char **Argv) {
  IsLLVMDriver = true;
  return findTool(Argc, Argv);
}
