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
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include <unordered_map>

namespace llvm {
namespace advisor {

CompilationUnit::CompilationUnit(const CompilationUnitInfo &Info,
                                 const std::string &WorkDir)
    : info(Info), workDir(WorkDir) {
  // Create unit-specific data directory
  llvm::SmallString<128> DataDir;
  llvm::sys::path::append(DataDir, WorkDir, "units", Info.name);
  llvm::sys::fs::create_directories(DataDir);
}

std::string CompilationUnit::getPrimarySource() const {
  if (info.sources.empty())
    return "";
  return info.sources[0].path;
}

std::string CompilationUnit::getDataDir() const {
  llvm::SmallString<128> DataDir;
  llvm::sys::path::append(DataDir, workDir, "units", info.name);
  return std::string(DataDir.str());
}

std::string CompilationUnit::getExecutablePath() const {
  return info.outputExecutable;
}

std::string CompilationUnit::getScratchDir() const {
  llvm::SmallString<128> ScratchDir;
  llvm::sys::path::append(ScratchDir, workDir, "units", info.name, "scratch");
  llvm::sys::fs::create_directories(ScratchDir);
  return std::string(ScratchDir.str());
}

void CompilationUnit::addGeneratedFile(llvm::StringRef Type,
                                       llvm::StringRef Path) {
  generatedFiles[Type.str()].push_back(Path.str());
}

bool CompilationUnit::hasGeneratedFiles(llvm::StringRef Type) const {
  if (Type.empty())
    return !generatedFiles.empty();
  auto It = generatedFiles.find(Type.str());
  return It != generatedFiles.end() && !It->second.empty();
}

llvm::SmallVector<std::string, 8>
CompilationUnit::getGeneratedFiles(llvm::StringRef Type) const {
  if (Type.empty()) {
    llvm::SmallVector<std::string, 8> AllFiles;
    for (const auto &Pair : generatedFiles)
      AllFiles.append(Pair.second.begin(), Pair.second.end());
    return AllFiles;
  }
  auto It = generatedFiles.find(Type.str());
  return It != generatedFiles.end() ? It->second
                                    : llvm::SmallVector<std::string, 8>();
}

const std::unordered_map<std::string, llvm::SmallVector<std::string, 8>> &
CompilationUnit::getAllGeneratedFiles() const {
  return generatedFiles;
}

std::string CompilationUnit::buildCategoryDir(llvm::StringRef Category) const {
  llvm::SmallString<128> Dir;
  llvm::sys::path::append(Dir, workDir, "units", info.name, Category);
  llvm::sys::fs::create_directories(Dir);
  return std::string(Dir.str());
}

std::string CompilationUnit::makeUniqueStem(llvm::StringRef SourcePath) const {
  llvm::hash_code Hash = llvm::hash_value(SourcePath);
  std::string HashSuffix =
      llvm::formatv("{0:x}", static_cast<uint64_t>(Hash)).str();
  return (llvm::Twine(llvm::sys::path::stem(SourcePath)) + "_" + HashSuffix)
      .str();
}

std::string CompilationUnit::makeArtifactPath(llvm::StringRef Category,
                                              llvm::StringRef SourcePath,
                                              llvm::StringRef Extension) const {
  std::string CategoryDir = buildCategoryDir(Category);
  llvm::SmallString<128> FilePath(CategoryDir);
  std::string Stem = makeUniqueStem(SourcePath);
  llvm::SmallString<32> FileName(Stem);
  FileName += Extension;
  llvm::sys::path::append(FilePath, FileName);
  return std::string(FilePath.str());
}

} // namespace advisor
} // namespace llvm
