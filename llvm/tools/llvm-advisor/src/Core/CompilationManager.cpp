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

CompilationManager::CompilationManager(const AdvisorConfig &Config)
    : config(Config), buildExecutor(Config) {

  // Get current working directory first
  llvm::SmallString<256> CurrentDir;
  llvm::sys::fs::current_path(CurrentDir);
  initialWorkingDir = std::string(CurrentDir.str());

  // Create temp directory with proper error handling
  llvm::SmallString<128> TempDirPath;
  if (auto EC =
          llvm::sys::fs::createUniqueDirectory("llvm-advisor", TempDirPath)) {
    // Use timestamp for temp folder naming
    auto Now = std::chrono::system_clock::now();
    auto Timestamp =
        std::chrono::duration_cast<std::chrono::seconds>(Now.time_since_epoch())
            .count();
    tempDir = "/tmp/llvm-advisor-" + std::to_string(Timestamp);
    llvm::sys::fs::create_directories(tempDir);
  } else
    tempDir = std::string(TempDirPath.str());

  // Ensure the directory actually exists
  if (!llvm::sys::fs::exists(tempDir))
    llvm::sys::fs::create_directories(tempDir);

  if (Config.getVerbose())
    llvm::outs() << "Using temporary directory: " << tempDir << "\n";

  // Initialize unit metadata tracking
  llvm::SmallString<256> OutputDirPath;
  if (llvm::sys::path::is_absolute(Config.getOutputDir())) {
    OutputDirPath = Config.getOutputDir();
  } else {
    OutputDirPath = initialWorkingDir;
    llvm::sys::path::append(OutputDirPath, Config.getOutputDir());
  }

  unitMetadata = std::make_unique<utils::UnitMetadata>(OutputDirPath.str());
  if (auto Err = unitMetadata->loadMetadata()) {
    if (Config.getVerbose()) {
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
    const std::string &Compiler,
    const llvm::SmallVectorImpl<std::string> &Args) {

  // Analyze the build command
  CommandAnalyzer Analyzer(Compiler, Args);
  BuildContext BuildContext = Analyzer.analyze();

  if (config.getVerbose())
    llvm::outs() << "Build phase: " << static_cast<int>(BuildContext.phase)
                 << "\n";

  // Skip data collection for linking/archiving phases
  if (BuildContext.phase == BuildPhase::Linking ||
      BuildContext.phase == BuildPhase::Archiving)
    return buildExecutor.execute(Compiler, Args, BuildContext, tempDir);

  std::string ArtifactRoot = tempDir;

  auto PreparedBuildOrErr = buildExecutor.prepareBuild(
      Compiler, Args, BuildContext, tempDir, ArtifactRoot);
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
    auto OrErr = buildExecutor.buildOriginalCompilation(Compiler, Args);
    if (OrErr)
      OriginalBuild = std::move(*OrErr);
    else
      llvm::consumeError(OrErr.takeError()); // non-fatal; fall back to nullptr
  }
  const clang::driver::Compilation *OriginalCompilation =
      (OriginalBuild && OriginalBuild->UsesDriver)
          ? OriginalBuild->Compilation.get()
          : nullptr;

  Analyzer.refineWithCompilation(BuildContext, OriginalCompilation);
  if (!BuildContext.outputFiles.empty()) {
    const std::string &PrimaryOutput = BuildContext.outputFiles.front();
    for (auto &Site : BuildContext.coverageSites)
      if (Site.instrumentedBinary.empty())
        Site.instrumentedBinary = PrimaryOutput;
  }
  if (!BuildContext.coverageSites.empty())
    coverageIngestion.registerSites(BuildContext.coverageSites);

  UnitDetector Detector(config);
  auto DetectedUnits =
      Detector.detectUnits(Compiler, Args, OriginalCompilation);

  // The original compilation is no longer needed — release it before executing
  // the (potentially large) instrumented build.
  OriginalBuild.reset();

  if (!DetectedUnits)
    return DetectedUnits.takeError();

  llvm::SmallVector<std::unique_ptr<CompilationUnit>, 4> Units;
  for (auto &UnitInfo : *DetectedUnits) {
    Units.push_back(std::make_unique<CompilationUnit>(UnitInfo, tempDir));

    unitMetadata->registerUnit(UnitInfo.name);
  }

  auto ExecResult = buildExecutor.executePreparedBuild(PreparedBuild);
  if (!ExecResult)
    return ExecResult;
  int ExitCode = *ExecResult;

  coverageIngestion.processOnce();
  coverageIngestion.startWatching();

  registerBuildArtifacts(BuildContext, Units);

  // Extract additional data
  DataExtractor Extractor(config);
  for (auto &Unit : Units) {
    if (auto Err = Extractor.extractAllData(*Unit, tempDir)) {
      if (config.getVerbose())
        llvm::errs() << "Data extraction failed: "
                     << llvm::toString(std::move(Err)) << "\n";
      // Mark unit as failed if data extraction fails
      unitMetadata->updateUnitStatus(Unit->getName(), "failed");
    } else {
      // Update unit metadata with file counts and artifact types
      const auto &GeneratedFiles = Unit->getAllGeneratedFiles();
      size_t TotalFiles = 0;
      for (const auto &Category : GeneratedFiles) {
        TotalFiles += Category.second.size();
        unitMetadata->addArtifactType(Unit->getName(), Category.first);
      }
      unitMetadata->updateUnitFileCount(Unit->getName(), TotalFiles);
    }
  }

  // Organize output
  if (auto Err = organizeOutput(Units)) {
    if (config.getVerbose())
      llvm::errs() << "Output organization failed: "
                   << llvm::toString(std::move(Err)) << "\n";
    // Mark units as failed if output organization fails
    for (auto &Unit : Units)
      unitMetadata->updateUnitStatus(Unit->getName(), "failed");
  } else {
    // Mark units as completed on successful organization
    for (auto &Unit : Units)
      unitMetadata->updateUnitStatus(Unit->getName(), "completed");
  }

  // Save metadata to disk
  if (auto Err = unitMetadata->saveMetadata()) {
    if (config.getVerbose())
      llvm::errs() << "Failed to save metadata: "
                   << llvm::toString(std::move(Err)) << "\n";
  }

  // Clean up leaked files from source directory
  cleanupLeakedFiles();

  return ExitCode;
}

void CompilationManager::registerBuildArtifacts(
    const BuildContext &Ctx,
    llvm::SmallVectorImpl<std::unique_ptr<CompilationUnit>> &Units) {
  if (Units.empty())
    return;

  FileClassifier Classifier;
  auto *Unit = Units.front().get();

  for (const auto &Path : Ctx.expectedGeneratedFiles) {
    if (!llvm::sys::fs::exists(Path))
      continue;
    if (!Classifier.shouldCollect(Path))
      continue;

    auto Classification = Classifier.classifyFile(Path);
    Unit->addGeneratedFile(Classification.category, Path);
  }
}

llvm::Error CompilationManager::organizeOutput(
    const llvm::SmallVectorImpl<std::unique_ptr<CompilationUnit>> &Units) {
  // Resolve output directory as absolute path from initial working directory
  llvm::SmallString<256> OutputDirPath;
  if (llvm::sys::path::is_absolute(config.getOutputDir())) {
    OutputDirPath = config.getOutputDir();
  } else {
    OutputDirPath = initialWorkingDir;
    llvm::sys::path::append(OutputDirPath, config.getOutputDir());
  }

  std::string OutputDir = std::string(OutputDirPath.str());

  if (config.getVerbose())
    llvm::outs() << "Output directory: " << OutputDir << "\n";

  // Generate timestamp for this compilation run
  auto Now = std::chrono::system_clock::now();
  auto TimeT = std::chrono::system_clock::to_time_t(Now);
  auto Tm = *std::localtime(&TimeT);

  char TimestampStr[20];
  std::strftime(TimestampStr, sizeof(TimestampStr), "%Y%m%d_%H%M%S", &Tm);

  // Move collected files to organized structure
  for (const auto &Unit : Units) {
    // Create base unit directory if it doesn't exist
    std::string BaseUnitDir = OutputDir + "/" + Unit->getName();
    llvm::sys::fs::create_directories(BaseUnitDir);

    // Create timestamped run directory
    std::string RunDirName = Unit->getName() + "_" + std::string(TimestampStr);
    std::string UnitDir = BaseUnitDir + "/" + RunDirName;

    if (config.getVerbose())
      llvm::outs() << "Creating run directory: " << UnitDir << "\n";

    // Create timestamped run directory
    if (auto EC = llvm::sys::fs::create_directories(UnitDir)) {
      if (config.getVerbose()) {
        llvm::errs() << "Warning: Could not create run directory: " << UnitDir
                     << "\n";
      }
      continue; // Skip if we can't create the directory
    }

    const auto &GeneratedFiles = Unit->getAllGeneratedFiles();
    for (const auto &Category : GeneratedFiles) {
      std::string CategoryDir = UnitDir + "/" + Category.first;
      llvm::sys::fs::create_directories(CategoryDir);

      for (const auto &File : Category.second) {
        std::string DestFile =
            CategoryDir + "/" + llvm::sys::path::filename(File).str();
        if (auto Err = FileManager::copyFile(File, DestFile)) {
          if (config.getVerbose()) {
            llvm::errs() << "Failed to copy " << File << " to " << DestFile
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

    llvm::StringRef Filename = llvm::sys::path::filename(I->path());
    if (Filename.ends_with(".opt.yaml") || Filename.ends_with(".opt.yml") ||
        Filename.ends_with(".profraw") || Filename.ends_with(".profdata")) {
      llvm::sys::fs::remove(I->path());
      if (config.getVerbose())
        llvm::outs() << "Cleaned up leaked file: " << I->path() << "\n";
    }
  }
}

} // namespace advisor
} // namespace llvm
