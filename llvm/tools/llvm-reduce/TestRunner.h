//===-- tools/llvm-reduce/TestRunner.h ---------------------------*- C++ -*-===/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMREDUCE_TESTRUNNER_H
#define LLVM_TOOLS_LLVMREDUCE_TESTRUNNER_H

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <vector>

namespace llvm {

// This class contains all the info necessary for running the provided
// interesting-ness test, as well as the most reduced module and its
// respective filename.
class TestRunner {
public:
  TestRunner(StringRef TestName, std::vector<std::string> TestArgs,
             StringRef ReducedFilepath);

  /// Runs the interesting-ness test for the specified file
  /// @returns 0 if test was successful, 1 if otherwise
  int run(StringRef Filename);

  /// Filename to the most reduced testcase
  StringRef getReducedFilepath() const { return ReducedFilepath; }
  /// Directory where tmp files are created
  StringRef getTmpDir() const { return TmpDirectory; }
  /// Returns the most reduced version of the original testcase
  Module *getProgram() const { return Program.get(); }

  void setReducedFilepath(SmallString<128> F) {
    ReducedFilepath = std::move(F);
  }
  void setProgram(std::unique_ptr<Module> P) { Program = std::move(P); }

private:
  SmallString<128> TestName;
  std::vector<std::string> TestArgs;
  SmallString<128> ReducedFilepath;
  SmallString<128> TmpDirectory;
  std::unique_ptr<Module> Program;
};

} // namespace llvm

#endif
