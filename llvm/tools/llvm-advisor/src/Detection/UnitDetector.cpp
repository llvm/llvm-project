//===--------------------- UnitDetector.cpp - LLVM Advisor ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the UnitDetector code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "UnitDetector.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace advisor {

UnitDetector::UnitDetector(const AdvisorConfig &config) : config_(config) {}

llvm::Expected<llvm::SmallVector<CompilationUnitInfo, 4>>
UnitDetector::detectUnits(llvm::StringRef compiler,
                          const llvm::SmallVectorImpl<std::string> &args) {

  auto sources = findSourceFiles(args);
  if (sources.empty()) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "No source files found");
  }

  CompilationUnitInfo unit;
  unit.name = generateUnitName(sources);
  unit.sources = sources;

  // Store original args but filter out source files for the compile flags
  for (const auto &arg : args) {
    // Skip source files when adding to compile flags
    llvm::StringRef extension = llvm::sys::path::extension(arg);
    if (!arg.empty() && arg[0] != '-' &&
        (extension == ".c" || extension == ".cpp" || extension == ".cc" ||
         extension == ".cxx" || extension == ".C")) {
      continue;
    }
    unit.compileFlags.push_back(arg);
  }

  // Extract output files and features
  extractBuildInfo(args, unit);

  return llvm::SmallVector<CompilationUnitInfo, 4>{unit};
}

llvm::SmallVector<SourceFile, 4> UnitDetector::findSourceFiles(
    const llvm::SmallVectorImpl<std::string> &args) const {
  llvm::SmallVector<SourceFile, 4> sources;

  for (const auto &arg : args) {
    if (arg.empty() || arg[0] == '-')
      continue;

    llvm::StringRef extension = llvm::sys::path::extension(arg);
    if (extension == ".c" || extension == ".cpp" || extension == ".cc" ||
        extension == ".cxx" || extension == ".C") {

      SourceFile source;
      source.path = arg;
      source.language = classifier_.getLanguage(arg);
      source.isHeader = false;
      sources.push_back(source);
    }
  }

  return sources;
}

void UnitDetector::extractBuildInfo(
    const llvm::SmallVectorImpl<std::string> &args, CompilationUnitInfo &unit) {
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];

    if (arg == "-o" && i + 1 < args.size()) {
      llvm::StringRef output = args[i + 1];
      llvm::StringRef ext = llvm::sys::path::extension(output);
      if (ext == ".o") {
        unit.outputObject = args[i + 1];
      } else {
        unit.outputExecutable = args[i + 1];
      }
    }

    llvm::StringRef argRef(arg);
    if (argRef.contains("openmp") || argRef.contains("offload") ||
        argRef.contains("cuda")) {
      unit.hasOffloading = true;
    }

    if (llvm::StringRef(arg).starts_with("-march=")) {
      unit.targetArch = arg.substr(7);
    }
  }
}

std::string UnitDetector::generateUnitName(
    const llvm::SmallVectorImpl<SourceFile> &sources) const {
  if (sources.empty())
    return "unknown";

  // Use first source file name as base
  std::string baseName = llvm::sys::path::stem(sources[0].path).str();

  // Add hash for uniqueness when multiple sources
  if (sources.size() > 1) {
    std::string combined;
    for (const auto &source : sources) {
      combined += source.path;
    }
    auto hash = llvm::hash_value(combined);
    baseName += "_" + std::to_string(static_cast<size_t>(hash) % 10000);
  }

  return baseName;
}

} // namespace advisor
} // namespace llvm
