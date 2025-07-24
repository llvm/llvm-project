//===---------------- CompilationManager.h - LLVM Advisor -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the CompilationManager code generator driver. It provides a
// convenient command-line interface for generating an assembly file or a
// relocatable file, given LLVM bitcode.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ADVISOR_CORE_COMPILATIONMANAGER_H
#define LLVM_ADVISOR_CORE_COMPILATIONMANAGER_H

#include "../Config/AdvisorConfig.h"
#include "../Utils/FileClassifier.h"
#include "BuildExecutor.h"
#include "CompilationUnit.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <string>
#include <unordered_set>

namespace llvm {
namespace advisor {

class CompilationManager {
public:
  explicit CompilationManager(const AdvisorConfig &config);
  ~CompilationManager();

  llvm::Expected<int>
  executeWithDataCollection(const std::string &compiler,
                            const llvm::SmallVectorImpl<std::string> &args);

private:
  std::unordered_set<std::string> scanDirectory(llvm::StringRef dir) const;

  void collectGeneratedFiles(
      const std::unordered_set<std::string> &existingFiles,
      llvm::SmallVectorImpl<std::unique_ptr<CompilationUnit>> &units);

  llvm::Error organizeOutput(
      const llvm::SmallVectorImpl<std::unique_ptr<CompilationUnit>> &units);

  void cleanupLeakedFiles();

  const AdvisorConfig &config_;
  BuildExecutor buildExecutor_;
  std::string tempDir_;
  std::string initialWorkingDir_;
};

} // namespace advisor
} // namespace llvm

#endif // LLVM_ADVISOR_CORE_COMPILATIONMANAGER_H
