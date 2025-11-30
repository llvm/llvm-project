//===------------------- CommandAnalyzer.h - LLVM Advisor -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the CommandAnalyzer code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ADVISOR_CORE_COMMANDANALYZER_H
#define LLVM_ADVISOR_CORE_COMMANDANALYZER_H

#include "BuildContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace llvm {
namespace advisor {

class CommandAnalyzer {
public:
  CommandAnalyzer(llvm::StringRef command,
                  const llvm::SmallVectorImpl<std::string> &args);

  BuildContext analyze() const;

private:
  BuildTool detectBuildTool() const;
  BuildPhase detectBuildPhase(BuildTool tool) const;
  void detectBuildFeatures(BuildContext &context) const;
  llvm::SmallVector<std::string, 8> extractInputFiles() const;
  llvm::SmallVector<std::string, 8> extractOutputFiles() const;

  std::string command_;
  llvm::SmallVector<std::string, 8> args_;
};

} // namespace advisor
} // namespace llvm

#endif // LLVM_ADVISOR_CORE_COMMANDANALYZER_H
