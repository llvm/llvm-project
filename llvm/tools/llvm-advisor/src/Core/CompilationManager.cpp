//===---------------- CompilationManager.cpp - LLVM Advisor ---------------===//
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

#include "CompilationManager.h"
#include "../Detection/UnitDetector.h"
#include "../Utils/FileManager.h"
#include "CommandAnalyzer.h"
#include "DataExtractor.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <chrono>
#include <cstdlib>
#include <unordered_set>

namespace llvm {
namespace advisor {

CompilationManager::CompilationManager(const AdvisorConfig &config)
    : config_(config), buildExecutor_(config) {

  // Get current working directory first
  llvm::SmallString<256> currentDir;
  llvm::sys::fs::current_path(currentDir);
  initialWorkingDir_ = currentDir.str().str();

  // Create temp directory with proper error handling
  llvm::SmallString<128> tempDirPath;
  if (auto EC =
          llvm::sys::fs::createUniqueDirectory("llvm-advisor", tempDirPath)) {
    // Use timestamp for temp folder naming
    auto now = std::chrono::system_clock::now();
    auto timestamp =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
            .count();
    tempDir_ = "/tmp/llvm-advisor-" + std::to_string(timestamp);
    llvm::sys::fs::create_directories(tempDir_);
  } else {
    tempDir_ = tempDirPath.str().str();
  }

  // Ensure the directory actually exists
  if (!llvm::sys::fs::exists(tempDir_)) {
    llvm::sys::fs::create_directories(tempDir_);
  }

  if (config_.getVerbose()) {
    llvm::outs() << "Using temporary directory: " << tempDir_ << "\n";
  }
}

CompilationManager::~CompilationManager() {
  if (!config_.getKeepTemps() && llvm::sys::fs::exists(tempDir_)) {
    llvm::sys::fs::remove_directories(tempDir_);
  }
}

llvm::Expected<int> CompilationManager::executeWithDataCollection(
    const std::string &compiler,
    const llvm::SmallVectorImpl<std::string> &args) {

  // Analyze the build command
  BuildContext buildContext = CommandAnalyzer(compiler, args).analyze();

  if (config_.getVerbose()) {
    llvm::outs() << "Build phase: " << static_cast<int>(buildContext.phase)
                 << "\n";
  }

  // Skip data collection for linking/archiving phases
  if (buildContext.phase == BuildPhase::Linking ||
      buildContext.phase == BuildPhase::Archiving) {
    return buildExecutor_.execute(compiler, args, buildContext, tempDir_);
  }

  // Detect compilation units
  UnitDetector detector(config_);
  auto detectedUnits = detector.detectUnits(compiler, args);
  if (!detectedUnits) {
    return detectedUnits.takeError();
  }

  llvm::SmallVector<std::unique_ptr<CompilationUnit>, 4> units;
  for (auto &unitInfo : *detectedUnits) {
    units.push_back(std::make_unique<CompilationUnit>(unitInfo, tempDir_));
  }

  // Scan existing files before compilation
  auto existingFiles = scanDirectory(initialWorkingDir_);

  // Execute compilation with instrumentation
  auto execResult =
      buildExecutor_.execute(compiler, args, buildContext, tempDir_);
  if (!execResult) {
    return execResult;
  }
  int exitCode = *execResult;

  // Collect generated files (even if compilation failed for analysis)
  collectGeneratedFiles(existingFiles, units);

  // Extract additional data
  DataExtractor extractor(config_);
  for (auto &unit : units) {
    if (auto Err = extractor.extractAllData(*unit, tempDir_)) {
      if (config_.getVerbose()) {
        llvm::errs() << "Data extraction failed: "
                     << llvm::toString(std::move(Err)) << "\n";
      }
    }
  }

  // Organize output
  if (auto Err = organizeOutput(units)) {
    if (config_.getVerbose()) {
      llvm::errs() << "Output organization failed: "
                   << llvm::toString(std::move(Err)) << "\n";
    }
  }

  // Clean up leaked files from source directory
  cleanupLeakedFiles();

  return exitCode;
}

std::unordered_set<std::string>
CompilationManager::scanDirectory(llvm::StringRef dir) const {
  std::unordered_set<std::string> files;
  std::error_code EC;
  for (llvm::sys::fs::directory_iterator DI(dir, EC), DE; DI != DE && !EC;
       DI.increment(EC)) {
    if (DI->type() != llvm::sys::fs::file_type::directory_file) {
      files.insert(DI->path());
    }
  }
  return files;
}

void CompilationManager::collectGeneratedFiles(
    const std::unordered_set<std::string> &existingFiles,
    llvm::SmallVectorImpl<std::unique_ptr<CompilationUnit>> &units) {
  FileClassifier classifier;

  // Collect files from temp directory
  std::error_code EC;
  for (llvm::sys::fs::recursive_directory_iterator DI(tempDir_, EC), DE;
       DI != DE && !EC; DI.increment(EC)) {
    if (DI->type() != llvm::sys::fs::file_type::directory_file) {
      std::string filePath = DI->path();
      if (classifier.shouldCollect(filePath)) {
        auto classification = classifier.classifyFile(filePath);

        // Add to appropriate unit
        if (!units.empty()) {
          units[0]->addGeneratedFile(classification.category, filePath);
        }
      }
    }
  }

  // Also check for files that leaked into source directory
  auto currentFiles = scanDirectory(initialWorkingDir_);
  for (const auto &file : currentFiles) {
    if (existingFiles.find(file) == existingFiles.end()) {
      if (classifier.shouldCollect(file)) {
        auto classification = classifier.classifyFile(file);

        // Move leaked file to temp directory
        std::string destPath =
            tempDir_ + "/" + llvm::sys::path::filename(file).str();
        if (!FileManager::moveFile(file, destPath)) {
          if (!units.empty()) {
            units[0]->addGeneratedFile(classification.category, destPath);
          }
        }
      }
    }
  }
}

llvm::Error CompilationManager::organizeOutput(
    const llvm::SmallVectorImpl<std::unique_ptr<CompilationUnit>> &units) {
  // Resolve output directory as absolute path from initial working directory
  llvm::SmallString<256> outputDirPath;
  if (llvm::sys::path::is_absolute(config_.getOutputDir())) {
    outputDirPath = config_.getOutputDir();
  } else {
    outputDirPath = initialWorkingDir_;
    llvm::sys::path::append(outputDirPath, config_.getOutputDir());
  }

  std::string outputDir = outputDirPath.str().str();

  if (config_.getVerbose()) {
    llvm::outs() << "Output directory: " << outputDir << "\n";
  }

  // Move collected files to organized structure
  for (const auto &unit : units) {
    std::string unitDir = outputDir + "/" + unit->getName();

    // Remove existing unit directory if it exists
    if (llvm::sys::fs::exists(unitDir)) {
      if (auto EC = llvm::sys::fs::remove_directories(unitDir)) {
        if (config_.getVerbose()) {
          llvm::errs() << "Warning: Could not remove existing unit directory: "
                       << unitDir << "\n";
        }
      }
    }

    // Create fresh unit directory
    if (auto EC = llvm::sys::fs::create_directories(unitDir)) {
      continue; // Skip if we can't create the directory
    }

    const auto &generatedFiles = unit->getAllGeneratedFiles();
    for (const auto &category : generatedFiles) {
      std::string categoryDir = unitDir + "/" + category.first;
      llvm::sys::fs::create_directories(categoryDir);

      for (const auto &file : category.second) {
        std::string destFile =
            categoryDir + "/" + llvm::sys::path::filename(file).str();
        if (auto Err = FileManager::copyFile(file, destFile)) {
          if (config_.getVerbose()) {
            llvm::errs() << "Failed to copy " << file << " to " << destFile
                         << "\n";
          }
        }
      }
    }
  }

  return llvm::Error::success();
}

void CompilationManager::cleanupLeakedFiles() {
  // Clean up any remaining leaked files in source directory
  auto currentFiles = scanDirectory(initialWorkingDir_);
  for (const auto &file : currentFiles) {
    llvm::StringRef filename = llvm::sys::path::filename(file);

    // Remove optimization remarks files that leaked
    if (filename.ends_with(".opt.yaml") || filename.ends_with(".opt.yml")) {
      llvm::sys::fs::remove(file);
      if (config_.getVerbose()) {
        llvm::outs() << "Cleaned up leaked file: " << file << "\n";
      }
    }

    // Remove profile files that leaked
    if (filename.ends_with(".profraw") || filename.ends_with(".profdata")) {
      llvm::sys::fs::remove(file);
      if (config_.getVerbose()) {
        llvm::outs() << "Cleaned up leaked file: " << file << "\n";
      }
    }
  }
}

} // namespace advisor
} // namespace llvm
