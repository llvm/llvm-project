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
#include "../Utils/FileClassifier.h"
#include "../Utils/FileManager.h"
#include "../Utils/UnitMetadata.h"
#include "CommandAnalyzer.h"
#include "DataExtractor.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <optional>
#include <unordered_set>

namespace llvm {
namespace advisor {

CompilationManager::CompilationManager(const AdvisorConfig &config)
    : config(config), buildExecutor(config) {

  // Get current working directory first
  llvm::SmallString<256> currentDir;
  llvm::sys::fs::current_path(currentDir);
  initialWorkingDir = std::string(currentDir.str());

  // Create temp directory with proper error handling
  llvm::SmallString<128> tempDirPath;
  if (auto EC =
          llvm::sys::fs::createUniqueDirectory("llvm-advisor", tempDirPath)) {
    // Use timestamp for temp folder naming
    auto now = std::chrono::system_clock::now();
    auto timestamp =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
            .count();
    tempDir = "/tmp/llvm-advisor-" + std::to_string(timestamp);
    llvm::sys::fs::create_directories(tempDir);
  } else
    tempDir = std::string(tempDirPath.str());

  // Ensure the directory actually exists
  if (!llvm::sys::fs::exists(tempDir))
    llvm::sys::fs::create_directories(tempDir);

  if (config.getVerbose())
    llvm::outs() << "Using temporary directory: " << tempDir << "\n";

  // Initialize unit metadata tracking
  llvm::SmallString<256> outputDirPath;
  if (llvm::sys::path::is_absolute(config.getOutputDir())) {
    outputDirPath = config.getOutputDir();
  } else {
    outputDirPath = initialWorkingDir;
    llvm::sys::path::append(outputDirPath, config.getOutputDir());
  }

  unitMetadata = std::make_unique<utils::UnitMetadata>(outputDirPath.str());
  if (auto Err = unitMetadata->loadMetadata()) {
    if (config.getVerbose()) {
      llvm::errs() << "Failed to load metadata: "
                   << llvm::toString(std::move(Err)) << "\n";
    }
  }
}

CompilationManager::~CompilationManager() {
  if (!config.getKeepTemps() && llvm::sys::fs::exists(tempDir))
    llvm::sys::fs::remove_directories(tempDir);
}

llvm::Expected<int> CompilationManager::executeWithDataCollection(
    const std::string &compiler,
    const llvm::SmallVectorImpl<std::string> &args) {

  // Analyze the build command
  CommandAnalyzer Analyzer(compiler, args);
  BuildContext buildContext = Analyzer.analyze();

  if (config.getVerbose())
    llvm::outs() << "Build phase: " << static_cast<int>(buildContext.phase)
                 << "\n";

  // Skip data collection for linking/archiving phases
  if (buildContext.phase == BuildPhase::Linking ||
      buildContext.phase == BuildPhase::Archiving)
    return buildExecutor.execute(compiler, args, buildContext, tempDir);

  std::string ArtifactRoot = tempDir;

  auto PreparedBuildOrErr = buildExecutor.prepareBuild(
      compiler, args, buildContext, tempDir, ArtifactRoot);
  if (!PreparedBuildOrErr)
    return PreparedBuildOrErr.takeError();

  auto PreparedBuild = std::move(*PreparedBuildOrErr);

  // ── Analysis phase (pre-instrumentation)
  // ──────────────────────────────────── Build a non-instrumented compilation
  // from the original user args so that phase refinement and unit detection see
  // clean, unmodified flags.  The result is used for analysis only and must
  // never be executed.
  std::optional<BuildExecutor::PreparedBuild> OriginalBuild;
  {
    auto OrErr = buildExecutor.buildOriginalCompilation(compiler, args);
    if (OrErr)
      OriginalBuild = std::move(*OrErr);
    else
      llvm::consumeError(OrErr.takeError()); // non-fatal; fall back to nullptr
  }
  const clang::driver::Compilation *OriginalCompilation =
      (OriginalBuild && OriginalBuild->UsesDriver)
          ? OriginalBuild->Compilation.get()
          : nullptr;

  Analyzer.refineWithCompilation(buildContext, OriginalCompilation);
  if (!buildContext.outputFiles.empty()) {
    const std::string &PrimaryOutput = buildContext.outputFiles.front();
    for (auto &Site : buildContext.coverageSites)
      if (Site.instrumentedBinary.empty())
        Site.instrumentedBinary = PrimaryOutput;
  }
  if (!buildContext.coverageSites.empty())
    coverageIngestion.registerSites(buildContext.coverageSites);

  UnitDetector detector(config);
  auto detectedUnits =
      detector.detectUnits(compiler, args, OriginalCompilation);

  // The original compilation is no longer needed — release it before executing
  // the (potentially large) instrumented build.
  OriginalBuild.reset();

  if (!detectedUnits)
    return detectedUnits.takeError();

  llvm::SmallVector<std::unique_ptr<CompilationUnit>, 4> units;
  for (auto &unitInfo : *detectedUnits) {
    units.push_back(std::make_unique<CompilationUnit>(unitInfo, tempDir));

    unitMetadata->registerUnit(unitInfo.name);
  }

  auto execResult = buildExecutor.executePreparedBuild(PreparedBuild);
  if (!execResult)
    return execResult;
  int exitCode = *execResult;

  coverageIngestion.processOnce();
  coverageIngestion.startWatching();

  registerBuildArtifacts(buildContext, units);

  // Extract additional data
  DataExtractor extractor(config);
  for (auto &unit : units) {
    if (auto Err = extractor.extractAllData(*unit, tempDir)) {
      if (config.getVerbose())
        llvm::errs() << "Data extraction failed: "
                     << llvm::toString(std::move(Err)) << "\n";
      // Mark unit as failed if data extraction fails
      unitMetadata->updateUnitStatus(unit->getName(), "failed");
    } else {
      // Update unit metadata with file counts and artifact types
      const auto &generatedFiles = unit->getAllGeneratedFiles();
      size_t totalFiles = 0;
      for (const auto &category : generatedFiles) {
        totalFiles += category.second.size();
        unitMetadata->addArtifactType(unit->getName(), category.first);
      }
      unitMetadata->updateUnitFileCount(unit->getName(), totalFiles);
    }
  }

  // Organize output
  if (auto Err = organizeOutput(units)) {
    if (config.getVerbose())
      llvm::errs() << "Output organization failed: "
                   << llvm::toString(std::move(Err)) << "\n";
    // Mark units as failed if output organization fails
    for (auto &unit : units) {
      unitMetadata->updateUnitStatus(unit->getName(), "failed");
    }
  } else {
    // Mark units as completed on successful organization
    for (auto &unit : units) {
      unitMetadata->updateUnitStatus(unit->getName(), "completed");
    }
  }

  // Save metadata to disk
  if (auto Err = unitMetadata->saveMetadata()) {
    if (config.getVerbose())
      llvm::errs() << "Failed to save metadata: "
                   << llvm::toString(std::move(Err)) << "\n";
  }

  // Clean up leaked files from source directory
  cleanupLeakedFiles();

  return exitCode;
}

void CompilationManager::registerBuildArtifacts(
    const BuildContext &Ctx,
    llvm::SmallVectorImpl<std::unique_ptr<CompilationUnit>> &units) {
  if (units.empty())
    return;

  FileClassifier classifier;
  auto *unit = units.front().get();

  for (const auto &path : Ctx.expectedGeneratedFiles) {
    if (!llvm::sys::fs::exists(path))
      continue;
    if (!classifier.shouldCollect(path))
      continue;

    auto classification = classifier.classifyFile(path);
    unit->addGeneratedFile(classification.category, path);
  }
}

llvm::Error CompilationManager::organizeOutput(
    const llvm::SmallVectorImpl<std::unique_ptr<CompilationUnit>> &units) {
  // Resolve output directory as absolute path from initial working directory
  llvm::SmallString<256> outputDirPath;
  if (llvm::sys::path::is_absolute(config.getOutputDir())) {
    outputDirPath = config.getOutputDir();
  } else {
    outputDirPath = initialWorkingDir;
    llvm::sys::path::append(outputDirPath, config.getOutputDir());
  }

  std::string outputDir = std::string(outputDirPath.str());

  if (config.getVerbose()) {
    llvm::outs() << "Output directory: " << outputDir << "\n";
  }

  // Generate timestamp for this compilation run
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto tm = *std::localtime(&time_t);

  char timestampStr[20];
  std::strftime(timestampStr, sizeof(timestampStr), "%Y%m%d_%H%M%S", &tm);

  // Move collected files to organized structure
  for (const auto &unit : units) {
    // Create base unit directory if it doesn't exist
    std::string baseUnitDir = outputDir + "/" + unit->getName();
    llvm::sys::fs::create_directories(baseUnitDir);

    // Create timestamped run directory
    std::string runDirName = unit->getName() + "_" + std::string(timestampStr);
    std::string unitDir = baseUnitDir + "/" + runDirName;

    if (config.getVerbose()) {
      llvm::outs() << "Creating run directory: " << unitDir << "\n";
    }

    // Create timestamped run directory
    if (auto EC = llvm::sys::fs::create_directories(unitDir)) {
      if (config.getVerbose()) {
        llvm::errs() << "Warning: Could not create run directory: " << unitDir
                     << "\n";
      }
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
          if (config.getVerbose()) {
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
  std::error_code EC;
  for (llvm::sys::fs::directory_iterator I(initialWorkingDir, EC), E;
       I != E && !EC; I.increment(EC)) {
    if (I->type() != llvm::sys::fs::file_type::regular_file)
      continue;

    llvm::StringRef filename = llvm::sys::path::filename(I->path());
    if (filename.ends_with(".opt.yaml") || filename.ends_with(".opt.yml") ||
        filename.ends_with(".profraw") || filename.ends_with(".profdata")) {
      llvm::sys::fs::remove(I->path());
      if (config.getVerbose())
        llvm::outs() << "Cleaned up leaked file: " << I->path() << "\n";
    }
  }
}

} // namespace advisor
} // namespace llvm
