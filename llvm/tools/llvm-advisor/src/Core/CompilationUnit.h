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
#ifndef LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_COMPILATIONUNIT_H
#define LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_COMPILATIONUNIT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>
#include <unordered_map>

namespace llvm::advisor {

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

  auto getName() const -> const std::string & { return info.name; }
  auto getInfo() const -> const CompilationUnitInfo & { return info; }
  auto getWorkDir() const -> const std::string & { return workDir; }
  auto getPrimarySource() const -> std::string;

  auto getDataDir() const -> std::string;
  auto getExecutablePath() const -> std::string;

  void addGeneratedFile(llvm::StringRef type, llvm::StringRef path);

  auto hasGeneratedFiles(llvm::StringRef type) const -> bool;
  auto
  getGeneratedFiles(llvm::StringRef type = "") const -> llvm::SmallVector<std::string, 8>;
  auto
  getAllGeneratedFiles() const -> const std::unordered_map<std::string, llvm::SmallVector<std::string, 8>> &;

private:
  CompilationUnitInfo info;
  std::string workDir;
  std::unordered_map<std::string, llvm::SmallVector<std::string, 8>>
      generatedFiles;
};

} // namespace llvm::advisor

#endif // LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_COMPILATIONUNIT_H
