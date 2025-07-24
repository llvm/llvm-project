//===---------------- CompilationUnit.cpp - LLVM Advisor ------------------===//
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

#include "CompilationUnit.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <unordered_map>

namespace llvm {
namespace advisor {

CompilationUnit::CompilationUnit(const CompilationUnitInfo &info,
                                 const std::string &workDir)
    : info_(info), workDir_(workDir) {
  // Create unit-specific data directory
  llvm::SmallString<128> dataDir;
  llvm::sys::path::append(dataDir, workDir, "units", info.name);
  llvm::sys::fs::create_directories(dataDir);
}

std::string CompilationUnit::getPrimarySource() const {
  if (info_.sources.empty()) {
    return "";
  }
  return info_.sources[0].path;
}

std::string CompilationUnit::getDataDir() const {
  llvm::SmallString<128> dataDir;
  llvm::sys::path::append(dataDir, workDir_, "units", info_.name);
  return dataDir.str().str();
}

std::string CompilationUnit::getExecutablePath() const {
  return info_.outputExecutable;
}

void CompilationUnit::addGeneratedFile(llvm::StringRef type,
                                       llvm::StringRef path) {
  generatedFiles_[type.str()].push_back(path.str());
}

bool CompilationUnit::hasGeneratedFiles(llvm::StringRef type) const {
  if (type.empty()) {
    return !generatedFiles_.empty();
  }
  auto it = generatedFiles_.find(type.str());
  return it != generatedFiles_.end() && !it->second.empty();
}

llvm::SmallVector<std::string, 8>
CompilationUnit::getGeneratedFiles(llvm::StringRef type) const {
  if (type.empty()) {
    llvm::SmallVector<std::string, 8> allFiles;
    for (const auto &pair : generatedFiles_) {
      allFiles.append(pair.second.begin(), pair.second.end());
    }
    return allFiles;
  }
  auto it = generatedFiles_.find(type.str());
  return it != generatedFiles_.end() ? it->second
                                     : llvm::SmallVector<std::string, 8>();
}

const std::unordered_map<std::string, llvm::SmallVector<std::string, 8>> &
CompilationUnit::getAllGeneratedFiles() const {
  return generatedFiles_;
}

} // namespace advisor
} // namespace llvm