//===------------------- CompilationUnit.h - LLVM Advisor -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the CompilationUnit code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ADVISOR_CORE_COMPILATIONUNIT_H
#define LLVM_ADVISOR_CORE_COMPILATIONUNIT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>
#include <unordered_map>

namespace llvm {
namespace advisor {

struct SourceFile {
  std::string path;
  std::string language;
  bool isHeader = false;
  llvm::SmallVector<std::string, 8> dependencies;
};

struct CompilationUnitInfo {
  std::string name;
  llvm::SmallVector<SourceFile, 4> sources;
  llvm::SmallVector<std::string, 8> compileFlags;
  std::string targetArch;
  bool hasOffloading = false;
  std::string outputObject;
  std::string outputExecutable;
};

class CompilationUnit {
public:
  CompilationUnit(const CompilationUnitInfo &info, const std::string &workDir);

  const std::string &getName() const { return info_.name; }
  const CompilationUnitInfo &getInfo() const { return info_; }
  const std::string &getWorkDir() const { return workDir_; }
  std::string getPrimarySource() const;

  std::string getDataDir() const;
  std::string getExecutablePath() const;

  void addGeneratedFile(llvm::StringRef type, llvm::StringRef path);

  bool hasGeneratedFiles(llvm::StringRef type) const;
  llvm::SmallVector<std::string, 8>
  getGeneratedFiles(llvm::StringRef type = "") const;
  const std::unordered_map<std::string, llvm::SmallVector<std::string, 8>> &
  getAllGeneratedFiles() const;

private:
  CompilationUnitInfo info_;
  std::string workDir_;
  std::unordered_map<std::string, llvm::SmallVector<std::string, 8>>
      generatedFiles_;
};

} // namespace advisor
} // namespace llvm

#endif // LLVM_ADVISOR_CORE_COMPILATIONUNIT_H