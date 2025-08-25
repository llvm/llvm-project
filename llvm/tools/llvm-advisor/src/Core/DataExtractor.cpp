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
#include "../Utils/FileManager.h"
#include "../Utils/ProcessRunner.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
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
  sys::fs::create_directories(tempDir + "/dependencies");
  sys::fs::create_directories(tempDir + "/debug");
  sys::fs::create_directories(tempDir + "/static-analyzer");
  sys::fs::create_directories(tempDir + "/diagnostics");
  sys::fs::create_directories(tempDir + "/coverage");
  sys::fs::create_directories(tempDir + "/time-trace");
  sys::fs::create_directories(tempDir + "/runtime-trace");
  sys::fs::create_directories(tempDir + "/binary-analysis");
  sys::fs::create_directories(tempDir + "/pgo");
  sys::fs::create_directories(tempDir + "/ftime-report");
  sys::fs::create_directories(tempDir + "/version-info");
  sys::fs::create_directories(tempDir + "/sources");

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
  if (auto Err = extractDependencies(unit, tempDir))
    return Err;
  if (auto Err = extractDebugInfo(unit, tempDir))
    return Err;
  if (auto Err = extractStaticAnalysis(unit, tempDir))
    return Err;
  if (auto Err = extractMacroExpansion(unit, tempDir))
    return Err;
  if (auto Err = extractCompilationPhases(unit, tempDir))
    return Err;
  if (auto Err = extractFTimeReport(unit, tempDir))
    return Err;
  if (auto Err = extractVersionInfo(unit, tempDir))
    return Err;
  if (auto Err = extractSources(unit, tempDir))
    return Err;

  // Run additional extractors
  for (size_t i = 0; i < numExtractors_; ++i) {
    const auto &extractor = extractors_[i];
    if (auto Err = (this->*extractor.method)(unit, tempDir)) {
      if (config_.getVerbose()) {
        errs() << extractor.name
               << " extraction failed: " << toString(std::move(Err)) << "\n";
      }
    }
  }

  return Error::success();
}

llvm::SmallVector<std::string, 8>
DataExtractor::getBaseCompilerArgs(const CompilationUnitInfo &unitInfo) const {
  llvm::SmallVector<std::string, 8> baseArgs;

  // Preserve relevant compile flags and handle paired flags that forward
  // arguments to specific toolchains (e.g. OpenMP target flags).
  for (size_t i = 0; i < unitInfo.compileFlags.size(); ++i) {
    const std::string &flag = unitInfo.compileFlags[i];

    // Handle paired forwarding flags that must precede their next argument.
    // Example: -Xopenmp-target -march=sm_70
    if (StringRef(flag) == "-Xopenmp-target" ||
        StringRef(flag).starts_with("-Xopenmp-target=")) {
      baseArgs.push_back(flag);
      // If the flag is the two-argument form, also copy the next arg if
      // present.
      if (StringRef(flag) == "-Xopenmp-target" &&
          i + 1 < unitInfo.compileFlags.size()) {
        baseArgs.push_back(unitInfo.compileFlags[i + 1]);
        ++i; // consume the next argument
      }
      continue;
    }

    // Commonly needed flags for reproducing preprocessing/IR/ASM
    if (StringRef(flag).starts_with("-I") ||
        StringRef(flag).starts_with("-D") ||
        StringRef(flag).starts_with("-U") ||
        StringRef(flag).starts_with("-std=") ||
        StringRef(flag).starts_with("-m") ||
        StringRef(flag).starts_with("-f") ||
        StringRef(flag).starts_with("-W") ||
        StringRef(flag).starts_with("-O")) {
      // Skip instrumentation/file-emission flags added by the executor
      if (StringRef(flag).starts_with("-fsave-optimization-record") ||
          StringRef(flag).starts_with("-fprofile-instr-generate") ||
          StringRef(flag).starts_with("-fcoverage-mapping") ||
          StringRef(flag).starts_with("-foptimization-record-file")) {
        continue;
      }
      baseArgs.push_back(flag);
      continue;
    }

    // Preserve explicit target specification when present
    if (StringRef(flag).starts_with("--target=") ||
        StringRef(flag) == "-target") {
      baseArgs.push_back(flag);
      if (StringRef(flag) == "-target" &&
          i + 1 < unitInfo.compileFlags.size()) {
        baseArgs.push_back(unitInfo.compileFlags[i + 1]);
        ++i;
      }
      continue;
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

Error DataExtractor::extractDependencies(CompilationUnit &unit,
                                         llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile = (tempDir + "/dependencies/" +
                              sys::path::stem(source.path).str() + ".deps.txt")
                                 .str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-MM"); // Generate dependencies in Makefile format
    baseArgs.push_back(source.path);

    auto result = ProcessRunner::run(config_.getToolPath("clang"), baseArgs,
                                     config_.getTimeout());
    if (result && result->exitCode == 0 && !result->stdout.empty()) {
      std::error_code EC;
      raw_fd_ostream OS(outputFile, EC);
      if (!EC) {
        OS << result->stdout; // Dependencies go to stdout
        unit.addGeneratedFile("dependencies", outputFile);
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
    std::string bindingsFile =
        (tempDir + "/debug/" + sys::path::stem(source.path).str() +
         ".bindings.txt")
            .str();

    // First: Extract compilation bindings with -ccc-print-bindings
    llvm::SmallVector<std::string, 8> bindingsArgs =
        getBaseCompilerArgs(unit.getInfo());
    bindingsArgs.push_back(
        "-ccc-print-bindings"); // Print compilation bindings/phases
    bindingsArgs.push_back("-fsyntax-only");
    bindingsArgs.push_back(source.path);

    auto bindingsResult = ProcessRunner::run(
        config_.getToolPath("clang"), bindingsArgs, config_.getTimeout());
    if (bindingsResult) {
      std::error_code EC;
      raw_fd_ostream bindingsOS(bindingsFile, EC);
      if (!EC) {
        bindingsOS << bindingsResult->stderr; // Bindings output goes to stderr
        unit.addGeneratedFile("compilation-phases", bindingsFile);
      }
    }

    // Second: Extract verbose compiler info with -v
    llvm::SmallVector<std::string, 8> verboseArgs =
        getBaseCompilerArgs(unit.getInfo());
    verboseArgs.push_back("-v"); // Verbose compilation phases
    verboseArgs.push_back("-fsyntax-only");
    verboseArgs.push_back(source.path);

    auto verboseResult = ProcessRunner::run(config_.getToolPath("clang"),
                                            verboseArgs, config_.getTimeout());
    if (verboseResult) {
      std::error_code EC;
      raw_fd_ostream verboseOS(outputFile, EC);
      if (!EC) {
        verboseOS << "COMPILATION PHASES:\n"
                  << verboseResult->stderr; // Verbose output goes to stderr
        unit.addGeneratedFile("compilation-phases", outputFile);
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractFTimeReport(CompilationUnit &unit,
                                        llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile = (tempDir + "/ftime-report/" +
                              sys::path::stem(source.path).str() + ".ftime.txt")
                                 .str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-ftime-report");
    baseArgs.push_back("-fsyntax-only");
    baseArgs.push_back(source.path);

    auto result = ProcessRunner::run(config_.getToolPath("clang"), baseArgs,
                                     config_.getTimeout());
    if (result) {
      std::error_code EC;
      raw_fd_ostream OS(outputFile, EC);
      if (!EC) {
        OS << "FTIME REPORT:\n"
           << result->stderr; // ftime-report output goes to stderr
        unit.addGeneratedFile("ftime-report", outputFile);
        if (config_.getVerbose()) {
          outs() << "FTime Report: " << outputFile << "\n";
        }
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractVersionInfo(CompilationUnit &unit,
                                        llvm::StringRef tempDir) {
  std::string outputFile = (tempDir + "/version-info/clang-version.txt").str();

  llvm::SmallVector<std::string, 8> args;
  args.push_back("--version");

  auto result = ProcessRunner::run(config_.getToolPath("clang"), args,
                                   config_.getTimeout());
  if (result) {
    std::error_code EC;
    raw_fd_ostream OS(outputFile, EC);
    if (!EC) {
      OS << result->stdout;
      unit.addGeneratedFile("version-info", outputFile);
      if (config_.getVerbose()) {
        outs() << "Version Info: " << outputFile << "\n";
      }
    }
  }
  return Error::success();
}

Error DataExtractor::runCompilerWithFlags(
    const llvm::SmallVector<std::string, 8> &args) {
  auto result = ProcessRunner::run(config_.getToolPath("clang"), args,
                                   config_.getTimeout());
  if (result && result->exitCode == 0) {
    return Error::success();
  }

  // Fallback: retry without offloading-specific flags to at least produce
  // host-side artifacts when device toolchains are unavailable.
  llvm::SmallVector<std::string, 8> sanitizedArgs;
  for (size_t i = 0; i < args.size(); ++i) {
    llvm::StringRef a(args[i]);

    if (a.starts_with("-fopenmp-targets")) {
      continue; // drop device list
    }
    if (a == "-Xopenmp-target") {
      // drop the flag and its immediate argument, if present
      if (i + 1 < args.size()) {
        ++i;
      }
      continue;
    }
    if (a.starts_with("-Xopenmp-target=")) {
      continue; // drop single-arg form
    }
    sanitizedArgs.push_back(args[i]);
  }

  auto retry = ProcessRunner::run(config_.getToolPath("clang"), sanitizedArgs,
                                  config_.getTimeout());
  if (!retry || retry->exitCode != 0) {
    return createStringError(std::make_error_code(std::errc::io_error),
                             "Compiler failed");
  }
  return Error::success();
}

Error DataExtractor::extractASTJSON(CompilationUnit &unit,
                                    llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile =
        (tempDir + "/ast/" + sys::path::stem(source.path).str() + ".ast.json")
            .str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-Xclang");
    baseArgs.push_back("-ast-dump=json");
    baseArgs.push_back("-fsyntax-only");
    baseArgs.push_back(source.path);

    auto result = ProcessRunner::run(config_.getToolPath("clang"), baseArgs,
                                     config_.getTimeout());
    if (result && result->exitCode == 0) {
      std::error_code EC;
      raw_fd_ostream OS(outputFile, EC);
      if (!EC) {
        OS << result->stdout;
        unit.addGeneratedFile("ast-json", outputFile);
        if (config_.getVerbose()) {
          outs() << "AST JSON: " << outputFile << "\n";
        }
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractDiagnostics(CompilationUnit &unit,
                                        llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile =
        (tempDir + "/diagnostics/" + sys::path::stem(source.path).str() +
         ".diagnostics.txt")
            .str();

    std::error_code EC;
    raw_fd_ostream OS(outputFile, EC);
    if (!EC) {
      OS << "Diagnostics for: " << source.path << "\n";

      // Run basic diagnostics
      llvm::SmallVector<std::string, 8> baseArgs =
          getBaseCompilerArgs(unit.getInfo());
      baseArgs.push_back("-fdiagnostics-parseable-fixits");
      baseArgs.push_back("-fdiagnostics-absolute-paths");
      baseArgs.push_back("-Wall");
      baseArgs.push_back("-Wextra");
      baseArgs.push_back("-fsyntax-only");
      baseArgs.push_back(source.path);

      auto result = ProcessRunner::run(config_.getToolPath("clang"), baseArgs,
                                       config_.getTimeout());
      if (result) {
        OS << "Exit code: " << result->exitCode << "\n";
        if (!result->stderr.empty()) {
          OS << result->stderr << "\n";
        }
        if (!result->stdout.empty()) {
          OS << result->stdout << "\n";
        }
      }

      // Run additional diagnostics with more flags
      llvm::SmallVector<std::string, 8> extraArgs =
          getBaseCompilerArgs(unit.getInfo());
      extraArgs.push_back("-Weverything");
      extraArgs.push_back("-Wno-c++98-compat");
      extraArgs.push_back("-Wno-c++98-compat-pedantic");
      extraArgs.push_back("-fsyntax-only");
      extraArgs.push_back(source.path);

      auto extraResult = ProcessRunner::run(config_.getToolPath("clang"),
                                            extraArgs, config_.getTimeout());
      if (extraResult) {
        OS << "\nExtended diagnostics:\n";
        OS << "Exit code: " << extraResult->exitCode << "\n";
        if (!extraResult->stderr.empty()) {
          OS << extraResult->stderr << "\n";
        }
        if (!extraResult->stdout.empty()) {
          OS << extraResult->stdout << "\n";
        }
      }

      unit.addGeneratedFile("diagnostics", outputFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractCoverage(CompilationUnit &unit,
                                     llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string objectFile = (tempDir + "/" + sourceStem + "_cov.o").str();
    std::string executableFile = (tempDir + "/" + sourceStem + "_cov").str();
    std::string profrawFile = (tempDir + "/" + sourceStem + ".profraw").str();
    std::string profdataFile = (tempDir + "/" + sourceStem + ".profdata").str();
    std::string coverageFile =
        (tempDir + "/coverage/" + sourceStem + ".coverage.json").str();

    // Compile with coverage instrumentation to create executable
    llvm::SmallVector<std::string, 8> compileArgs =
        getBaseCompilerArgs(unit.getInfo());
    compileArgs.push_back("-fprofile-instr-generate=" + profrawFile);
    compileArgs.push_back("-fcoverage-mapping");
    compileArgs.push_back("-o");
    compileArgs.push_back(executableFile);
    compileArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(compileArgs)) {
      continue;
    }

    if (sys::fs::exists(executableFile)) {
      // Run the executable to generate profile data (if it doesn't require
      // input)
      auto runResult =
          ProcessRunner::run(executableFile, {}, config_.getTimeout());

      // Convert raw profile to indexed format if profraw exists
      if (sys::fs::exists(profrawFile)) {
        llvm::SmallVector<std::string, 8> mergeArgs = {
            "merge", "-sparse", "-o", profdataFile, profrawFile};
        auto mergeResult = ProcessRunner::run("llvm-profdata", mergeArgs,
                                              config_.getTimeout());

        if (mergeResult && mergeResult->exitCode == 0 &&
            sys::fs::exists(profdataFile)) {
          // Generate coverage report in JSON format
          llvm::SmallVector<std::string, 8> covArgs = {
              "export", executableFile, "-instr-profile=" + profdataFile,
              "-format=json"};
          auto covResult =
              ProcessRunner::run("llvm-cov", covArgs, config_.getTimeout());

          if (covResult && covResult->exitCode == 0) {
            std::error_code EC;
            raw_fd_ostream OS(coverageFile, EC);
            if (!EC) {
              OS << covResult->stdout;
              unit.addGeneratedFile("coverage", coverageFile);
              if (config_.getVerbose()) {
                outs() << "Coverage: " << coverageFile << "\n";
              }
            }
          }
        }
        // Clean up temporary files
        sys::fs::remove(profrawFile);
        sys::fs::remove(profdataFile);
      }
      // Clean up temporary executable
      sys::fs::remove(executableFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractTimeTrace(CompilationUnit &unit,
                                      llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string traceFile =
        (tempDir + "/time-trace/" + sourceStem + ".trace.json").str();

    // Create a temporary object file in temp directory
    std::string tempObject = (tempDir + "/" + sourceStem + "_trace.o").str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-ftime-trace");
    baseArgs.push_back("-ftime-trace-granularity=0");
    baseArgs.push_back("-c");
    baseArgs.push_back("-o");
    baseArgs.push_back(tempObject);
    baseArgs.push_back(source.path);

    auto result = ProcessRunner::run(config_.getToolPath("clang"), baseArgs,
                                     config_.getTimeout());

    // The trace file is generated next to the object file with .json extension
    std::string expectedTraceFile =
        (tempDir + "/" + sourceStem + "_trace.json").str();

    // Also check the working directory in case trace went there
    std::string workingDirTrace = sourceStem + ".json";

    if (sys::fs::exists(expectedTraceFile)) {
      if (auto EC = sys::fs::rename(expectedTraceFile, traceFile)) {
        if (config_.getVerbose()) {
          errs() << "Failed to move trace file: " << EC.message() << "\n";
        }
      } else {
        unit.addGeneratedFile("time-trace", traceFile);
        if (config_.getVerbose()) {
          outs() << "Time trace: " << traceFile << "\n";
        }
      }
    } else if (sys::fs::exists(workingDirTrace)) {
      if (auto EC = sys::fs::rename(workingDirTrace, traceFile)) {
        if (config_.getVerbose()) {
          errs() << "Failed to move trace file from working dir: "
                 << EC.message() << "\n";
        }
      } else {
        unit.addGeneratedFile("time-trace", traceFile);
        if (config_.getVerbose()) {
          outs() << "Time trace: " << traceFile << "\n";
        }
      }
    } else if (config_.getVerbose()) {
      errs() << "Time trace file not found for " << source.path << "\n";
    }

    // Clean up temporary object file
    sys::fs::remove(tempObject);
  }
  return Error::success();
}

Error DataExtractor::extractRuntimeTrace(CompilationUnit &unit,
                                         llvm::StringRef tempDir) {

  // Check for OpenMP offloading flags
  bool hasOffloading = false;
  for (const auto &flag : unit.getInfo().compileFlags) {
    StringRef flagStr(flag);
    // Check for various OpenMP offloading indicators
    if (flagStr.contains("offload") ||          // Generic offload flags
        flagStr.contains("-fopenmp-targets") || // OpenMP target specification
        flagStr.contains("-Xopenmp-target") ||  // OpenMP target-specific flags
        flagStr.contains("nvptx") ||            // NVIDIA GPU targets
        flagStr.contains("amdgcn") ||           // AMD GPU targets
        flagStr.contains("-fopenmp")) { // Basic OpenMP (may have runtime)
      hasOffloading = true;
      break;
    }
  }

  if (!hasOffloading) {
    if (config_.getVerbose())
      outs() << "Runtime trace skipped - no OpenMP offloading flags detected\n";
    return Error::success();
  }

  if (config_.getVerbose())
    outs()
        << "OpenMP offloading detected, attempting runtime trace extraction\n";

  // Find executable name from compile flags
  std::string executableStr = "a.out"; // Default executable name
  for (size_t i = 0; i < unit.getInfo().compileFlags.size(); ++i) {
    if (unit.getInfo().compileFlags[i] == "-o" &&
        i + 1 < unit.getInfo().compileFlags.size()) {
      executableStr = unit.getInfo().compileFlags[i + 1];
      break;
    }
  }

  StringRef executable = executableStr;

  if (config_.getVerbose())
    outs() << "Looking for executable: " << executable << "\n";

  if (!sys::fs::exists(executable)) {
    if (config_.getVerbose()) {
      outs() << "Runtime trace skipped - executable not found: " << executable
             << "\n";
      outs() << "Note: Executable is needed to generate runtime profile data\n";
      outs() << "Checked current directory for: " << executable << "\n";

      // List current directory contents for debugging
      outs() << "Current directory contents:\n";
      std::error_code EC;
      for (sys::fs::directory_iterator I(".", EC), E; I != E && !EC;
           I.increment(EC)) {
        outs() << "  " << I->path() << "\n";
      }
    }

    return Error::success();
  }

  // Create a trace directory
  SmallString<256> traceDir;
  sys::path::append(traceDir, tempDir, "runtime-trace");
  if (auto ec = sys::fs::create_directories(traceDir, true)) {
    if (config_.getVerbose())
      outs() << "Warning: Failed to create runtime trace directory: "
             << ec.message() << "\n";
  }

  // Prepare trace file
  SmallString<256> traceFile;
  sys::path::append(traceFile, traceDir, "profile.json");

  if (config_.getVerbose()) {
    outs() << "Running runtime trace: " << executable << "\n";
    outs() << "Trace file: " << traceFile << "\n";
  }

  // Set environment variable for OpenMP target profiling
  SmallVector<std::string, 8> env;
  env.push_back("LIBOMPTARGET_PROFILE=" + traceFile.str().str());

  if (config_.getVerbose()) {
    outs() << "Setting environment: LIBOMPTARGET_PROFILE=" << traceFile << "\n";
    outs() << "Executing: " << executable << "\n";
  }

  // Run executable with profiling environment
  auto result =
      ProcessRunner::runWithEnv(executable, {}, env, config_.getTimeout());

  if (config_.getVerbose()) {
    if (result) {
      outs() << "Runtime trace completed with exit code: " << result->exitCode
             << "\n";

      if (!result->stdout.empty())
        outs() << "STDOUT: " << result->stdout << "\n";
      if (!result->stderr.empty())
        outs() << "STDERR: " << result->stderr << "\n";
    } else {
      outs() << "Runtime trace failed: " << toString(result.takeError())
             << "\n";
    }
  }

  // Register trace file if generated
  if (sys::fs::exists(traceFile)) {
    unit.addGeneratedFile("runtime-trace", traceFile.str());
    if (config_.getVerbose())
      outs() << "Runtime trace saved: " << traceFile << "\n";
  } else {
    if (config_.getVerbose()) {
      outs() << "Runtime trace failed - no trace file generated at: "
             << traceFile << "\n";
      outs() << "This may happen if:\n";
      outs() << "  1. The program didn't use OpenMP target offloading\n";
      outs() << "  2. The runtime doesn't support LIBOMPTARGET_PROFILE\n";
      outs() << "  3. The program crashed before generating profile data\n";
    }
  }

  return Error::success();
}

Error DataExtractor::extractSARIF(CompilationUnit &unit,
                                  llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string sarifFile =
        (tempDir + "/static-analyzer/" + sourceStem + ".sarif").str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("--analyze");
    baseArgs.push_back("-Xanalyzer");
    baseArgs.push_back("-analyzer-output=sarif");
    baseArgs.push_back("-o");
    baseArgs.push_back(sarifFile);
    baseArgs.push_back(source.path);

    auto result = ProcessRunner::run(config_.getToolPath("clang"), baseArgs,
                                     config_.getTimeout());

    if (result && sys::fs::exists(sarifFile)) {
      // Check if the SARIF file has content
      uint64_t fileSize;
      auto fileSizeErr = sys::fs::file_size(sarifFile, fileSize);
      if (!fileSizeErr && fileSize > 0) {
        unit.addGeneratedFile("static-analysis-sarif", sarifFile);
        if (config_.getVerbose()) {
          outs() << "SARIF static analysis extracted: " << sarifFile << "\n";
        }
      } else if (config_.getVerbose()) {
        outs() << "SARIF file created but empty for " << source.path << "\n";
      }
    } else if (config_.getVerbose()) {
      errs() << "Failed to extract SARIF static analysis for " << source.path
             << "\n";
    }
  }
  return Error::success();
}

Error DataExtractor::extractBinarySize(CompilationUnit &unit,
                                       llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string objectFile = (tempDir + "/" + sourceStem + "_size.o").str();
    std::string sizeFile =
        (tempDir + "/binary-analysis/" + sourceStem + ".size.txt").str();

    // Compile object file
    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-c");
    baseArgs.push_back("-o");
    baseArgs.push_back(objectFile);
    baseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(baseArgs)) {
      continue;
    }

    if (sys::fs::exists(objectFile)) {
      std::error_code EC;
      raw_fd_ostream OS(sizeFile, EC);
      if (!EC) {
        OS << "Binary size analysis for: " << source.path << "\n";
        OS << "Generated from object file: " << objectFile << "\n\n";

        // Try different llvm-size formats
        llvm::SmallVector<std::string, 8> berkeleyArgs = {objectFile};
        auto berkeleyResult =
            ProcessRunner::run("llvm-size", berkeleyArgs, config_.getTimeout());
        if (berkeleyResult && berkeleyResult->exitCode == 0) {
          OS << "Berkeley format (default):\n"
             << berkeleyResult->stdout << "\n";
        }

        // Also try -A format for more details
        llvm::SmallVector<std::string, 8> sysVArgs = {"-A", objectFile};
        auto sysVResult =
            ProcessRunner::run("llvm-size", sysVArgs, config_.getTimeout());
        if (sysVResult && sysVResult->exitCode == 0) {
          OS << "System V format:\n" << sysVResult->stdout << "\n";
        }

        unit.addGeneratedFile("binary-size", sizeFile);
        if (config_.getVerbose()) {
          outs() << "Binary size: " << sizeFile << "\n";
        }
      }

      // Clean up temporary object file
      sys::fs::remove(objectFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractPGO(CompilationUnit &unit,
                                llvm::StringRef tempDir) {
  // Look for existing profile raw data file from compilation
  std::string profrawFile = tempDir.str() + "/profile.profraw";

  if (sys::fs::exists(profrawFile)) {
    std::string profileFile = (tempDir + "/pgo/merged.profdata").str();
    std::string profileText = (tempDir + "/pgo/profile.txt").str();

    // Convert raw profile to indexed format
    llvm::SmallVector<std::string, 8> mergeArgs = {"merge", "-sparse", "-o",
                                                   profileFile, profrawFile};
    auto mergeResult =
        ProcessRunner::run("llvm-profdata", mergeArgs, config_.getTimeout());

    if (mergeResult && mergeResult->exitCode == 0 &&
        sys::fs::exists(profileFile)) {
      // Generate human-readable profile summary
      llvm::SmallVector<std::string, 8> showArgs = {
          "show", "-all-functions", "-text", "-detailed-summary", profileFile};
      auto showResult =
          ProcessRunner::run("llvm-profdata", showArgs, config_.getTimeout());

      if (showResult && showResult->exitCode == 0) {
        std::error_code EC;
        raw_fd_ostream OS(profileText, EC);
        if (!EC) {
          OS << "PGO Profile Data Summary\n";
          OS << "========================\n";
          OS << "Source profile: " << profrawFile << "\n";
          OS << "Indexed profile: " << profileFile << "\n\n";
          OS << showResult->stdout;
          unit.addGeneratedFile("pgo-profile", profileText);
          if (config_.getVerbose()) {
            outs() << "PGO profile data extracted: " << profileText << "\n";
          }
        }
      }

      // Also create a JSON-like summary if possible
      llvm::SmallVector<std::string, 8> jsonArgs = {"show", "-json",
                                                    profileFile};
      auto jsonResult =
          ProcessRunner::run("llvm-profdata", jsonArgs, config_.getTimeout());

      if (jsonResult && jsonResult->exitCode == 0) {
        std::string jsonFile = (tempDir + "/pgo/profile.json").str();
        std::error_code EC;
        raw_fd_ostream jsonOS(jsonFile, EC);
        if (!EC) {
          jsonOS << jsonResult->stdout;
          unit.addGeneratedFile("pgo-profile-json", jsonFile);
          if (config_.getVerbose()) {
            outs() << "PGO profile JSON extracted: " << jsonFile << "\n";
          }
        }
      }
      sys::fs::remove(profileFile);
    } else if (config_.getVerbose()) {
      errs() << "Failed to merge PGO profile data\n";
    }
  } else if (config_.getVerbose()) {
    outs() << "No PGO profile data found to extract\n";
  }

  return Error::success();
}

Error DataExtractor::extractSymbols(CompilationUnit &unit,
                                    llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string objectFile = (tempDir + "/" + sourceStem + "_symbols.o").str();
    std::string symbolsFile =
        (tempDir + "/binary-analysis/" + sourceStem + ".symbols.txt").str();

    // Compile object file
    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-c");
    baseArgs.push_back("-o");
    baseArgs.push_back(objectFile);
    baseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(baseArgs)) {
      continue;
    }

    if (sys::fs::exists(objectFile)) {
      std::error_code EC;
      raw_fd_ostream OS(symbolsFile, EC);
      if (!EC) {
        OS << "Symbol table for: " << source.path << "\n";
        OS << "Generated from object file: " << objectFile << "\n\n";

        // Extract symbols using llvm-nm
        llvm::SmallVector<std::string, 8> nmArgs = {"-C", "-a", objectFile};
        auto nmResult =
            ProcessRunner::run("llvm-nm", nmArgs, config_.getTimeout());
        if (nmResult && nmResult->exitCode == 0) {
          OS << "Symbols:\n" << nmResult->stdout << "\n";
        }

        unit.addGeneratedFile("symbols", symbolsFile);
        if (config_.getVerbose()) {
          outs() << "Symbols: " << symbolsFile << "\n";
        }
      }

      // Clean up temporary object file
      sys::fs::remove(objectFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractObjdump(CompilationUnit &unit,
                                    llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string objectFile = (tempDir + "/" + sourceStem + "_objdump.o").str();
    std::string objdumpFile =
        (tempDir + "/binary-analysis/" + sourceStem + ".objdump.txt").str();

    // Compile object file
    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-c");
    baseArgs.push_back("-o");
    baseArgs.push_back(objectFile);
    baseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(baseArgs)) {
      continue;
    }

    if (sys::fs::exists(objectFile)) {
      std::error_code EC;
      raw_fd_ostream OS(objdumpFile, EC);
      if (!EC) {
        OS << "Object dump for: " << source.path << "\n";
        OS << "Generated from object file: " << objectFile << "\n\n";

        // Disassemble using llvm-objdump
        llvm::SmallVector<std::string, 8> objdumpArgs = {"-d", "-t", "-r",
                                                         objectFile};
        auto objdumpResult = ProcessRunner::run("llvm-objdump", objdumpArgs,
                                                config_.getTimeout());
        if (objdumpResult && objdumpResult->exitCode == 0) {
          OS << objdumpResult->stdout << "\n";
        }

        unit.addGeneratedFile("objdump", objdumpFile);
        if (config_.getVerbose()) {
          outs() << "Objdump: " << objdumpFile << "\n";
        }
      }

      // Clean up temporary object file
      sys::fs::remove(objectFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractXRay(CompilationUnit &unit,
                                 llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string executableFile = (tempDir + "/" + sourceStem + "_xray").str();
    std::string xrayFile =
        (tempDir + "/binary-analysis/" + sourceStem + ".xray.txt").str();

    // Compile with XRay instrumentation
    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-fxray-instrument");
    baseArgs.push_back("-fxray-instruction-threshold=1");
    baseArgs.push_back("-o");
    baseArgs.push_back(executableFile);
    baseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(baseArgs)) {
      continue;
    }

    if (sys::fs::exists(executableFile)) {
      std::error_code EC;
      raw_fd_ostream OS(xrayFile, EC);
      if (!EC) {
        OS << "XRay analysis for: " << source.path << "\n";
        OS << "Generated from executable: " << executableFile << "\n\n";

        // Extract XRay instrumentation info
        llvm::SmallVector<std::string, 8> xrayArgs = {"extract",
                                                      executableFile};
        auto xrayResult =
            ProcessRunner::run("llvm-xray", xrayArgs, config_.getTimeout());
        if (xrayResult && xrayResult->exitCode == 0) {
          OS << "XRay instrumentation:\n" << xrayResult->stdout << "\n";
        }

        unit.addGeneratedFile("xray", xrayFile);
        if (config_.getVerbose()) {
          outs() << "XRay: " << xrayFile << "\n";
        }
      }

      // Clean up temporary executable
      sys::fs::remove(executableFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractOptDot(CompilationUnit &unit,
                                   llvm::StringRef tempDir) {
  for (const auto &source : unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string bitcodeFile = (tempDir + "/" + sourceStem + ".bc").str();
    std::string dotFile = (tempDir + "/ir/" + sourceStem + ".cfg.dot").str();

    // Compile to bitcode
    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(unit.getInfo());
    baseArgs.push_back("-emit-llvm");
    baseArgs.push_back("-c");
    baseArgs.push_back("-o");
    baseArgs.push_back(bitcodeFile);
    baseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(baseArgs)) {
      continue;
    }

    if (sys::fs::exists(bitcodeFile)) {
      // Generate CFG dot file using opt
      llvm::SmallVector<std::string, 8> optArgs = {
          "-dot-cfg", "-disable-output", bitcodeFile};
      auto optResult = ProcessRunner::run("opt", optArgs, config_.getTimeout());

      // The dot file is typically generated in current directory
      std::string expectedDotFile = "." + sourceStem + ".dot";
      if (sys::fs::exists(expectedDotFile)) {
        if (auto EC = sys::fs::rename(expectedDotFile, dotFile)) {
          if (config_.getVerbose()) {
            errs() << "Failed to move dot file: " << EC.message() << "\n";
          }
        } else {
          unit.addGeneratedFile("cfg-dot", dotFile);
          if (config_.getVerbose()) {
            outs() << "CFG DOT: " << dotFile << "\n";
          }
        }
      }

      // Clean up temporary bitcode file
      sys::fs::remove(bitcodeFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractSources(CompilationUnit &unit,
                                    llvm::StringRef tempDir) {
  if (config_.getVerbose()) {
    outs() << "Extracting source files based on dependencies...\n";
  }

  // Create sources directory
  SmallString<256> sourcesDir;
  sys::path::append(sourcesDir, tempDir, "sources");
  if (auto EC = sys::fs::create_directories(sourcesDir)) {
    if (config_.getVerbose()) {
      outs() << "Warning: Failed to create sources directory: " << EC.message()
             << "\n";
    }
    return Error::success(); // Continue even if we can't create directory
  }

  // Find and parse dependencies files
  SmallString<256> depsDir;
  sys::path::append(depsDir, tempDir, "dependencies");

  if (!sys::fs::exists(depsDir)) {
    if (config_.getVerbose()) {
      outs() << "No dependencies directory found, skipping source extraction\n";
    }
    return Error::success();
  }

  std::error_code EC;
  for (sys::fs::directory_iterator I(depsDir, EC), E; I != E && !EC;
       I.increment(EC)) {
    StringRef filePath = I->path();
    if (!filePath.ends_with(".deps.txt")) {
      continue;
    }

    if (config_.getVerbose()) {
      outs() << "Processing dependencies file: " << filePath << "\n";
    }

    // Read and parse dependencies file
    auto bufferOrErr = MemoryBuffer::getFile(filePath);
    if (!bufferOrErr) {
      if (config_.getVerbose()) {
        outs() << "Warning: Failed to read dependencies file: " << filePath
               << "\n";
      }
      continue;
    }

    StringRef content = bufferOrErr.get()->getBuffer();
    SmallVector<StringRef, 16> lines;
    content.split(lines, '\n');

    for (StringRef line : lines) {
      line = line.trim();
      if (line.empty() || !line.contains(":")) {
        continue;
      }

      // Parse line format: "target.o: source1.c source2.h ..."
      auto colonPos = line.find(':');
      if (colonPos == StringRef::npos) {
        continue;
      }

      StringRef sourcesStr = line.substr(colonPos + 1).trim();
      SmallVector<StringRef, 8> sourceFiles;
      sourcesStr.split(sourceFiles, ' ', -1, false);

      for (StringRef sourceFile : sourceFiles) {
        sourceFile = sourceFile.trim();
        if (sourceFile.empty()) {
          continue;
        }

        // Convert relative paths to absolute paths
        SmallString<256> absoluteSourcePath;
        if (sys::path::is_absolute(sourceFile)) {
          absoluteSourcePath = sourceFile;
        } else {
          // Get current working directory
          SmallString<256> currentDir;
          sys::fs::current_path(currentDir);
          absoluteSourcePath = currentDir;
          sys::path::append(absoluteSourcePath, sourceFile);
        }

        // Check if source file exists
        if (!sys::fs::exists(absoluteSourcePath)) {
          if (config_.getVerbose()) {
            outs() << "Warning: Source file not found: " << absoluteSourcePath
                   << "\n";
          }
          continue;
        }

        // Create destination path preserving directory structure
        SmallString<256> destPath;
        sys::path::append(destPath, sourcesDir,
                          sys::path::filename(sourceFile));

        // Copy source file to sources directory
        if (auto copyErr = FileManager::copyFile(absoluteSourcePath.str().str(),
                                                 destPath.str().str())) {
          if (config_.getVerbose()) {
            outs() << "Warning: Failed to copy source file "
                   << absoluteSourcePath << " to " << destPath << "\n";
          }
        } else {
          unit.addGeneratedFile("sources", destPath.str().str());
          if (config_.getVerbose()) {
            outs() << "Copied source: " << sourceFile << " -> " << destPath
                   << "\n";
          }
        }
      }
    }
  }

  return Error::success();
}

const DataExtractor::ExtractorInfo DataExtractor::extractors_[] = {
    {&DataExtractor::extractASTJSON, "AST JSON"},
    {&DataExtractor::extractDiagnostics, "Diagnostics"},
    {&DataExtractor::extractCoverage, "Coverage"},
    {&DataExtractor::extractTimeTrace, "Time trace"},
    {&DataExtractor::extractRuntimeTrace, "Runtime trace"},
    {&DataExtractor::extractSARIF, "SARIF"},
    {&DataExtractor::extractBinarySize, "Binary size"},
    {&DataExtractor::extractPGO, "PGO"},
    {&DataExtractor::extractSymbols, "Symbols"},
    {&DataExtractor::extractObjdump, "Objdump"},
    {&DataExtractor::extractXRay, "XRay"},
    {&DataExtractor::extractOptDot, "Opt DOT"}};

const size_t DataExtractor::numExtractors_ =
    sizeof(DataExtractor::extractors_) / sizeof(DataExtractor::ExtractorInfo);

} // namespace advisor
} // namespace llvm
