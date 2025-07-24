//===------------------ DataExtractor.cpp - LLVM Advisor ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the DataExtractor code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "DataExtractor.h"
#include "../Utils/ProcessRunner.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace llvm {
namespace advisor {

DataExtractor::DataExtractor(const AdvisorConfig &config) : config_(config) {}

Error DataExtractor::extractAllData(CompilationUnit &unit,
                                    llvm::StringRef tempDir) {
  if (config_.getVerbose()) {
    outs() << "Extracting data for unit: " << unit.getName() << "\n";
  }

  // Create extraction subdirectories
  sys::fs::create_directories(tempDir + "/ir");
  sys::fs::create_directories(tempDir + "/assembly");
  sys::fs::create_directories(tempDir + "/ast");
  sys::fs::create_directories(tempDir + "/preprocessed");
  sys::fs::create_directories(tempDir + "/include-tree");
  sys::fs::create_directories(tempDir + "/debug");
  sys::fs::create_directories(tempDir + "/static-analyzer");

  if (auto Err = extractIR(unit, tempDir))
    return Err;
  if (auto Err = extractAssembly(unit, tempDir))
    return Err;
  if (auto Err = extractAST(unit, tempDir))
    return Err;
  if (auto Err = extractPreprocessed(unit, tempDir))
    return Err;
  if (auto Err = extractIncludeTree(unit, tempDir))
    return Err;
  if (auto Err = extractDebugInfo(unit, tempDir))
    return Err;
  if (auto Err = extractStaticAnalysis(unit, tempDir))
    return Err;
  if (auto Err = extractMacroExpansion(unit, tempDir))
    return Err;
  if (auto Err = extractCompilationPhases(unit, tempDir))
    return Err;

  return Error::success();
}

llvm::SmallVector<std::string, 8>
DataExtractor::getBaseCompilerArgs(const CompilationUnitInfo &unitInfo) const {
  llvm::SmallVector<std::string, 8> baseArgs;

  // Copy include paths and defines
  for (const auto &arg : unitInfo.compileFlags) {
    if (StringRef(arg).starts_with("-I") || StringRef(arg).starts_with("-D") ||
        StringRef(arg).starts_with("-U") ||
        StringRef(arg).starts_with("-std=") ||
        StringRef(arg).starts_with("-m") || StringRef(arg).starts_with("-f") ||
        StringRef(arg).starts_with("-W") || StringRef(arg).starts_with("-O")) {
      // Skip problematic flags for extraction
      if (StringRef(arg).starts_with("-fsave-optimization-record") ||
          StringRef(arg).starts_with("-fprofile-instr-generate") ||
          StringRef(arg).starts_with("-fcoverage-mapping") ||
          StringRef(arg).starts_with("-foptimization-record-file")) {
        continue;
      }
      baseArgs.push_back(arg);
    }
  }

  return baseArgs;
}

Error DataExtractor::extractIR(CompilationUnit &unit, llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile =
        (tempDir + "/ir/" + sys::path::stem(source.path).str() + ".ll").str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-emit-llvm");
    baseArgs.push_back("-S");
    baseArgs.push_back("-o");
    baseArgs.push_back(outputFile);
    baseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(baseArgs)) {
      if (config_.getVerbose()) {
        errs() << "Failed to extract IR for " << source.path << "\n";
      }
      continue;
    }

    if (sys::fs::exists(outputFile)) {
      unit.addGeneratedFile("ir", outputFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractAssembly(CompilationUnit &unit,
                                     llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile =
        (tempDir + "/assembly/" + sys::path::stem(source.path).str() + ".s")
            .str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-S");
    baseArgs.push_back("-o");
    baseArgs.push_back(outputFile);
    baseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(baseArgs)) {
      if (config_.getVerbose()) {
        errs() << "Failed to extract assembly for " << source.path << "\n";
      }
      continue;
    }

    if (sys::fs::exists(outputFile)) {
      unit.addGeneratedFile("assembly", outputFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractAST(CompilationUnit &unit,
                                llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile =
        (tempDir + "/ast/" + sys::path::stem(source.path).str() + ".ast").str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-ast-dump");
    baseArgs.push_back("-fsyntax-only");
    baseArgs.push_back(source.path);

    auto result = ProcessRunner::run(config_.getToolPath("clang"), baseArgs,
                                     config_.getTimeout());
    if (result && result->exitCode == 0) {
      std::error_code EC;
      raw_fd_ostream OS(outputFile, EC);
      if (!EC) {
        OS << result->stdout;
        unit.addGeneratedFile("ast", outputFile);
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractPreprocessed(CompilationUnit &unit,
                                         llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string ext = (source.language == "C++") ? ".ii" : ".i";
    std::string outputFile =
        (tempDir + "/preprocessed/" + sys::path::stem(source.path).str() + ext)
            .str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-E");
    baseArgs.push_back("-o");
    baseArgs.push_back(outputFile);
    baseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(baseArgs)) {
      if (config_.getVerbose()) {
        errs() << "Failed to extract preprocessed for " << source.path << "\n";
      }
      continue;
    }

    if (sys::fs::exists(outputFile)) {
      unit.addGeneratedFile("preprocessed", outputFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractIncludeTree(CompilationUnit &unit,
                                        llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile =
        (tempDir + "/include-tree/" + sys::path::stem(source.path).str() +
         ".include.txt")
            .str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-H");
    baseArgs.push_back("-fsyntax-only");
    baseArgs.push_back(source.path);

    auto result = ProcessRunner::run(config_.getToolPath("clang"), baseArgs,
                                     config_.getTimeout());
    if (result && !result->stderr.empty()) {
      std::error_code EC;
      raw_fd_ostream OS(outputFile, EC);
      if (!EC) {
        OS << result->stderr; // Include tree goes to stderr
        unit.addGeneratedFile("include-tree", outputFile);
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractDebugInfo(CompilationUnit &unit,
                                      llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile = (tempDir + "/debug/" +
                              sys::path::stem(source.path).str() + ".debug.txt")
                                 .str();
    std::string objectFile =
        (tempDir + "/debug/" + sys::path::stem(source.path).str() + ".o").str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-g");
    baseArgs.push_back("-c");
    baseArgs.push_back("-o");
    baseArgs.push_back(objectFile);
    baseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(baseArgs)) {
      if (config_.getVerbose()) {
        errs() << "Failed to extract debug info for " << source.path << "\n";
      }
      continue;
    }

    // Extract DWARF info using llvm-dwarfdump
    if (sys::fs::exists(objectFile)) {
      llvm::SmallVector<std::string, 8> dwarfArgs = {objectFile};
      auto result =
          ProcessRunner::run("llvm-dwarfdump", dwarfArgs, config_.getTimeout());
      if (result && result->exitCode == 0) {
        std::error_code EC;
        raw_fd_ostream OS(outputFile, EC);
        if (!EC) {
          OS << result->stdout;
          unit.addGeneratedFile("debug", outputFile);
        }
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractStaticAnalysis(CompilationUnit &unit,
                                           llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile =
        (tempDir + "/static-analyzer/" + sys::path::stem(source.path).str() +
         ".analysis.txt")
            .str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("--analyze");
    baseArgs.push_back("-Xanalyzer");
    baseArgs.push_back("-analyzer-output=text");
    baseArgs.push_back(source.path);

    auto result = ProcessRunner::run(config_.getToolPath("clang"), baseArgs,
                                     config_.getTimeout());
    if (result) {
      std::error_code EC;
      raw_fd_ostream OS(outputFile, EC);
      if (!EC) {
        OS << "STDOUT:\n" << result->stdout << "\nSTDERR:\n" << result->stderr;
        unit.addGeneratedFile("static-analyzer", outputFile);
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractMacroExpansion(CompilationUnit &unit,
                                           llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile =
        (tempDir + "/preprocessed/" + sys::path::stem(source.path).str() +
         ".macro-expanded" + ((source.language == "C++") ? ".ii" : ".i"))
            .str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-E");
    baseArgs.push_back("-dM"); // Show macro definitions
    baseArgs.push_back("-o");
    baseArgs.push_back(outputFile);
    baseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(baseArgs)) {
      if (config_.getVerbose()) {
        errs() << "Failed to extract macro expansion for " << source.path
               << "\n";
      }
      continue;
    }

    if (sys::fs::exists(outputFile)) {
      unit.addGeneratedFile("macro-expansion", outputFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractCompilationPhases(CompilationUnit &unit,
                                              llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile =
        (tempDir + "/debug/" + sys::path::stem(source.path).str() +
         ".phases.txt")
            .str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-v"); // Verbose compilation phases
    baseArgs.push_back("-fsyntax-only");
    baseArgs.push_back(source.path);

    auto result = ProcessRunner::run(config_.getToolPath("clang"), baseArgs,
                                     config_.getTimeout());
    if (result) {
      std::error_code EC;
      raw_fd_ostream OS(outputFile, EC);
      if (!EC) {
        OS << "COMPILATION PHASES:\n"
           << result->stderr; // Verbose output goes to stderr
        unit.addGeneratedFile("compilation-phases", outputFile);
      }
    }
  }
  return Error::success();
}

Error DataExtractor::runCompilerWithFlags(
    const llvm::SmallVector<std::string, 8> &args) {
  auto result = ProcessRunner::run(config_.getToolPath("clang"), args,
                                   config_.getTimeout());
  if (!result || result->exitCode != 0) {
    return createStringError(std::make_error_code(std::errc::io_error),
                             "Compiler failed");
  }
  return Error::success();
}

} // namespace advisor
} // namespace llvm