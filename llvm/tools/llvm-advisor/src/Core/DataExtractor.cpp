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

DataExtractor::DataExtractor(const AdvisorConfig &Config) : Config(Config) {}

Error DataExtractor::extractAllData(CompilationUnit &Unit,
                                    llvm::StringRef TempDir) {
  if (Config.getVerbose()) {
    outs() << "Extracting data for unit: " << Unit.getName() << "\n";
  }

  // Create extraction subdirectories
  sys::fs::create_directories(TempDir + "/ir");
  sys::fs::create_directories(TempDir + "/assembly");
  sys::fs::create_directories(TempDir + "/ast");
  sys::fs::create_directories(TempDir + "/preprocessed");
  sys::fs::create_directories(TempDir + "/include-tree");
  sys::fs::create_directories(TempDir + "/dependencies");
  sys::fs::create_directories(TempDir + "/debug");
  sys::fs::create_directories(TempDir + "/static-analyzer");
  sys::fs::create_directories(TempDir + "/diagnostics");
  sys::fs::create_directories(TempDir + "/coverage");
  sys::fs::create_directories(TempDir + "/time-trace");
  sys::fs::create_directories(TempDir + "/runtime-trace");
  sys::fs::create_directories(TempDir + "/binary-analysis");
  sys::fs::create_directories(TempDir + "/pgo");
  sys::fs::create_directories(TempDir + "/ftime-report");
  sys::fs::create_directories(TempDir + "/version-info");
  sys::fs::create_directories(TempDir + "/sources");

  if (auto Err = extractIR(Unit, TempDir))
    return Err;
  if (auto Err = extractAssembly(Unit, TempDir))
    return Err;
  if (auto Err = extractAST(Unit, TempDir))
    return Err;
  if (auto Err = extractPreprocessed(Unit, TempDir))
    return Err;
  if (auto Err = extractIncludeTree(Unit, TempDir))
    return Err;
  if (auto Err = extractDependencies(Unit, TempDir))
    return Err;
  if (auto Err = extractDebugInfo(Unit, TempDir))
    return Err;
  if (auto Err = extractStaticAnalysis(Unit, TempDir))
    return Err;
  if (auto Err = extractMacroExpansion(Unit, TempDir))
    return Err;
  if (auto Err = extractCompilationPhases(Unit, TempDir))
    return Err;
  if (auto Err = extractFTimeReport(Unit, TempDir))
    return Err;
  if (auto Err = extractVersionInfo(Unit, TempDir))
    return Err;
  if (auto Err = extractSources(Unit, TempDir))
    return Err;

  // Run additional extractors
  for (size_t i = 0; i < numExtractors; ++i) {
    const auto &extractor = extractors[i];
    if (auto Err = (this->*extractor.method)(Unit, TempDir)) {
      if (Config.getVerbose()) {
        errs() << extractor.name
               << " extraction failed: " << toString(std::move(Err)) << "\n";
      }
    }
  }

  return Error::success();
}

llvm::SmallVector<std::string, 8>
DataExtractor::getBaseCompilerArgs(const CompilationUnitInfo &UnitInfo) const {
  llvm::SmallVector<std::string, 8> BaseArgs;

  // Preserve relevant compile flags and handle paired flags that forward
  // arguments to specific toolchains (e.g. OpenMP target flags).
  for (size_t I = 0; I < UnitInfo.compileFlags.size(); ++I) {
    const std::string &Flag = UnitInfo.compileFlags[I];

    // Handle paired forwarding flags that must precede their next argument.
    // Example: -Xopenmp-target -march=sm_70
    if (StringRef(Flag) == "-Xopenmp-target" ||
        StringRef(Flag).starts_with("-Xopenmp-target=")) {
      BaseArgs.push_back(Flag);
      // If the flag is the two-argument form, also copy the next arg if
      // present.
      if (StringRef(Flag) == "-Xopenmp-target" &&
          I + 1 < UnitInfo.compileFlags.size()) {
        BaseArgs.push_back(UnitInfo.compileFlags[I + 1]);
        ++I; // consume the next argument
      }
      continue;
    }

    // Commonly needed flags for reproducing preprocessing/IR/ASM
    if (StringRef(Flag).starts_with("-I") ||
        StringRef(Flag).starts_with("-D") ||
        StringRef(Flag).starts_with("-U") ||
        StringRef(Flag).starts_with("-std=") ||
        StringRef(Flag).starts_with("-m") ||
        StringRef(Flag).starts_with("-f") ||
        StringRef(Flag).starts_with("-W") ||
        StringRef(Flag).starts_with("-O")) {
      // Skip instrumentation/file-emission flags added by the executor
      if (StringRef(Flag).starts_with("-fsave-optimization-record") ||
          StringRef(Flag).starts_with("-fprofile-instr-generate") ||
          StringRef(Flag).starts_with("-fcoverage-mapping") ||
          StringRef(Flag).starts_with("-foptimization-record-file")) {
        continue;
      }
      BaseArgs.push_back(Flag);
      continue;
    }

    // Preserve explicit target specification when present
    if (StringRef(Flag).starts_with("--target=") ||
        StringRef(Flag) == "-target") {
      BaseArgs.push_back(Flag);
      if (StringRef(Flag) == "-target" &&
          I + 1 < UnitInfo.compileFlags.size()) {
        BaseArgs.push_back(UnitInfo.compileFlags[I + 1]);
        ++I;
      }
      continue;
    }
  }

  return BaseArgs;
}

Error DataExtractor::extractIR(CompilationUnit &Unit, llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile =
        std::string(TempDir) + "/ir/" + sys::path::stem(source.path).str() + ".ll";

    llvm::SmallVector<std::string, 8> BaseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    BaseArgs.push_back("-emit-llvm");
    BaseArgs.push_back("-S");
    BaseArgs.push_back("-o");
    BaseArgs.push_back(OutputFile);
    BaseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(BaseArgs)) {
      if (Config.getVerbose())
        errs() << "Failed to extract IR for " << source.path << "\n";
      continue;
    }

    if (sys::fs::exists(OutputFile))
      Unit.addGeneratedFile("ir", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractAssembly(CompilationUnit &Unit,
                                     llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile =
        std::string(TempDir) + "/assembly/" + sys::path::stem(source.path).str() + ".s";

    llvm::SmallVector<std::string, 8> BaseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    BaseArgs.push_back("-S");
    BaseArgs.push_back("-o");
    BaseArgs.push_back(OutputFile);
    BaseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(BaseArgs)) {
      if (Config.getVerbose())
        errs() << "Failed to extract assembly for " << source.path << "\n";
      continue;
    }

    if (sys::fs::exists(OutputFile))
      Unit.addGeneratedFile("assembly", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractAST(CompilationUnit &Unit,
                                llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile =
        std::string(TempDir) + "/ast/" + sys::path::stem(source.path).str() + ".ast";

    llvm::SmallVector<std::string, 8> BaseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    BaseArgs.push_back("-ast-dump");
    BaseArgs.push_back("-fsyntax-only");
    BaseArgs.push_back(source.path);

    auto Result = ProcessRunner::run(Config.getToolPath("clang"), BaseArgs,
                                     Config.getTimeout());
    if (Result && Result->exitCode == 0) {
      std::error_code EC;
      raw_fd_ostream OS(OutputFile, EC);
      if (!EC) {
        OS << Result->stdout;
        Unit.addGeneratedFile("ast", OutputFile);
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractPreprocessed(CompilationUnit &Unit,
                                         llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string ext = (source.language == "C++") ? ".ii" : ".i";
    std::string OutputFile = std::string(TempDir) + "/preprocessed/" +
                             sys::path::stem(source.path).str() + ext;

    llvm::SmallVector<std::string, 8> BaseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    BaseArgs.push_back("-E");
    BaseArgs.push_back("-o");
    BaseArgs.push_back(OutputFile);
    BaseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(BaseArgs)) {
      if (Config.getVerbose())
        errs() << "Failed to extract preprocessed for " << source.path << "\n";
      continue;
    }

    if (sys::fs::exists(OutputFile))
      Unit.addGeneratedFile("preprocessed", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractIncludeTree(CompilationUnit &Unit,
                                        llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile = std::string(TempDir) + "/include-tree/" +
                             sys::path::stem(source.path).str() +
                             ".include.txt";

    llvm::SmallVector<std::string, 8> BaseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    BaseArgs.push_back("-H");
    BaseArgs.push_back("-fsyntax-only");
    BaseArgs.push_back(source.path);

    auto Result = ProcessRunner::run(Config.getToolPath("clang"), BaseArgs,
                                     Config.getTimeout());
    if (Result && !Result->stderr.empty()) {
      std::error_code EC;
      raw_fd_ostream OS(OutputFile, EC);
      if (!EC) {
        OS << Result->stderr; // Include tree goes to stderr
        Unit.addGeneratedFile("include-tree", OutputFile);
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractDependencies(CompilationUnit &Unit,
                                         llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile =
        std::string(TempDir) + "/dependencies/" +
        sys::path::stem(source.path).str() + ".deps.txt";

    llvm::SmallVector<std::string, 8> BaseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    BaseArgs.push_back("-MM"); // Generate dependencies in Makefile format
    BaseArgs.push_back(source.path);

    auto Result = ProcessRunner::run(Config.getToolPath("clang"), BaseArgs,
                                     Config.getTimeout());
    if (Result && Result->exitCode == 0 && !Result->stdout.empty()) {
      std::error_code EC;
      raw_fd_ostream OS(OutputFile, EC);
      if (!EC) {
        OS << Result->stdout; // Dependencies go to stdout
        Unit.addGeneratedFile("dependencies", OutputFile);
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractDebugInfo(CompilationUnit &Unit,
                                      llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile = std::string(TempDir) + "/debug/" +
                             sys::path::stem(source.path).str() +
                             ".debug.txt";
    std::string ObjectFile =
        std::string(TempDir) + "/debug/" +
        sys::path::stem(source.path).str() + ".o";

    llvm::SmallVector<std::string, 8> BaseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    BaseArgs.push_back("-g");
    BaseArgs.push_back("-c");
    BaseArgs.push_back("-o");
    BaseArgs.push_back(ObjectFile);
    BaseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(BaseArgs)) {
      if (Config.getVerbose())
        errs() << "Failed to extract debug info for " << source.path << "\n";
      continue;
    }

    // Extract DWARF info using llvm-dwarfdump
    if (sys::fs::exists(ObjectFile)) {
      llvm::SmallVector<std::string, 8> DwarfArgs = {ObjectFile};
      auto DwarfResult = ProcessRunner::run("llvm-dwarfdump", DwarfArgs, Config.getTimeout());
      if (DwarfResult && DwarfResult->exitCode == 0) {
        std::error_code EC;
        raw_fd_ostream OS(OutputFile, EC);
        if (!EC) {
          OS << DwarfResult->stdout;
          Unit.addGeneratedFile("debug", OutputFile);
        }
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractStaticAnalysis(CompilationUnit &Unit,
                                           llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile = std::string(TempDir) + "/static-analyzer/" +
                             sys::path::stem(source.path).str() +
                             ".analysis.txt";

    llvm::SmallVector<std::string, 8> BaseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    BaseArgs.push_back("--analyze");
    BaseArgs.push_back("-Xanalyzer");
    BaseArgs.push_back("-analyzer-output=text");
    BaseArgs.push_back(source.path);

    auto Result = ProcessRunner::run(Config.getToolPath("clang"), BaseArgs,
                                     Config.getTimeout());
    if (Result) {
      std::error_code EC;
      raw_fd_ostream OS(OutputFile, EC);
      if (!EC) {
        OS << "STDOUT:\n" << Result->stdout << "\nSTDERR:\n" << Result->stderr;
        Unit.addGeneratedFile("static-analyzer", OutputFile);
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractMacroExpansion(CompilationUnit &Unit,
                                           llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile =
        std::string(TempDir) + "/preprocessed/" +
        sys::path::stem(source.path).str() + ".macro-expanded" +
        ((source.language == "C++") ? ".ii" : ".i");

    llvm::SmallVector<std::string, 8> BaseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    BaseArgs.push_back("-E");
    BaseArgs.push_back("-dM"); // Show macro definitions
    BaseArgs.push_back("-o");
    BaseArgs.push_back(OutputFile);
    BaseArgs.push_back(source.path);

    if (auto Err = runCompilerWithFlags(BaseArgs)) {
      if (Config.getVerbose()) {
        errs() << "Failed to extract macro expansion for " << source.path
               << "\n";
      }
      continue;
    }

    if (sys::fs::exists(OutputFile)) {
      Unit.addGeneratedFile("macro-expansion", OutputFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractCompilationPhases(CompilationUnit &Unit,
                                              llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile =
        (TempDir + "/debug/" + sys::path::stem(source.path).str() +
         ".phases.txt")
            .str();
    std::string bindingsFile =
        (TempDir + "/debug/" + sys::path::stem(source.path).str() +
         ".bindings.txt")
            .str();

    // First: Extract compilation bindings with -ccc-print-bindings
    llvm::SmallVector<std::string, 8> bindingsArgs =
        getBaseCompilerArgs(Unit.getInfo());
    bindingsArgs.push_back(
        "-ccc-print-bindings"); // Print compilation bindings/phases
    bindingsArgs.push_back("-fsyntax-only");
    bindingsArgs.push_back(source.path);

    auto bindingsResult = ProcessRunner::run(
        Config.getToolPath("clang"), bindingsArgs, Config.getTimeout());
    if (bindingsResult) {
      std::error_code EC;
      raw_fd_ostream bindingsOS(bindingsFile, EC);
      if (!EC) {
        bindingsOS << bindingsResult->stderr; // Bindings output goes to stderr
        Unit.addGeneratedFile("compilation-phases", bindingsFile);
      }
    }

    // Second: Extract verbose compiler info with -v
    llvm::SmallVector<std::string, 8> verboseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    verboseArgs.push_back("-v"); // Verbose compilation phases
    verboseArgs.push_back("-fsyntax-only");
    verboseArgs.push_back(source.path);

    auto verboseResult = ProcessRunner::run(Config.getToolPath("clang"),
                                            verboseArgs, Config.getTimeout());
    if (verboseResult) {
      std::error_code EC;
      raw_fd_ostream verboseOS(outputFile, EC);
      if (!EC) {
        verboseOS << "COMPILATION PHASES:\n"
                  << verboseResult->stderr; // Verbose output goes to stderr
        Unit.addGeneratedFile("compilation-phases", outputFile);
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractFTimeReport(CompilationUnit &Unit,
                                        llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile = std::string(TempDir) + "/ftime-report/" +
                             sys::path::stem(source.path).str() + ".ftime.txt";

    llvm::SmallVector<std::string, 8> baseArgs = getBaseCompilerArgs(Unit.getInfo());
    baseArgs.push_back("-ftime-report");
    baseArgs.push_back("-fsyntax-only");
    baseArgs.push_back(source.path);

    auto result = ProcessRunner::run(Config.getToolPath("clang"), baseArgs,
                                     Config.getTimeout());
    if (result) {
      std::error_code EC;
      raw_fd_ostream OS(outputFile, EC);
      if (!EC) {
        OS << "FTIME REPORT:\n"
           << result->stderr; // ftime-report output goes to stderr
        Unit.addGeneratedFile("ftime-report", outputFile);
        if (Config.getVerbose()) {
          outs() << "FTime Report: " << outputFile << "\n";
        }
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractVersionInfo(CompilationUnit &Unit,
                                        llvm::StringRef TempDir) {
  std::string OutputFile = std::string(TempDir) + "/version-info/clang-version.txt";

  llvm::SmallVector<std::string, 8> Args;
  Args.push_back("--version");

  auto Result = ProcessRunner::run(Config.getToolPath("clang"), Args,
                                   Config.getTimeout());
  if (Result) {
    std::error_code EC;
    raw_fd_ostream OS(OutputFile, EC);
    if (!EC) {
      OS << Result->stdout;
      Unit.addGeneratedFile("version-info", OutputFile);
      if (Config.getVerbose()) {
        outs() << "Version Info: " << OutputFile << "\n";
      }
    }
  }
  return Error::success();
}

Error DataExtractor::runCompilerWithFlags(
    const llvm::SmallVector<std::string, 8> &Args) {
  auto Result = ProcessRunner::run(Config.getToolPath("clang"), Args,
                                   Config.getTimeout());
  if (Result && Result->exitCode == 0)
    return Error::success();

  // Fallback: retry without offloading-specific flags to at least produce
  // host-side artifacts when device toolchains are unavailable.
  llvm::SmallVector<std::string, 8> SanitizedArgs;
  for (size_t I = 0; I < Args.size(); ++I) {
    llvm::StringRef A(Args[I]);

    if (A.starts_with("-fopenmp-targets")) {
      continue; // drop device list
    }
    if (A == "-Xopenmp-target") {
      // drop the flag and its immediate argument, if present
      if (I + 1 < Args.size()) {
        ++I;
      }
      continue;
    }
    if (A.starts_with("-Xopenmp-target=")) {
      continue; // drop single-arg form
    }
    SanitizedArgs.push_back(Args[I]);
  }

  auto Retry = ProcessRunner::run(Config.getToolPath("clang"), SanitizedArgs,
                                  Config.getTimeout());
  if (!Retry || Retry->exitCode != 0)
    return createStringError(std::make_error_code(std::errc::io_error),
                             "Compiler failed");
  return Error::success();
}

Error DataExtractor::extractASTJSON(CompilationUnit &Unit,
                                    llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile = std::string(TempDir) + "/ast/" +
                             sys::path::stem(source.path).str() + ".ast.json";

    llvm::SmallVector<std::string, 8> BaseArgs = getBaseCompilerArgs(Unit.getInfo());
    BaseArgs.push_back("-Xclang");
    BaseArgs.push_back("-ast-dump=json");
    BaseArgs.push_back("-fsyntax-only");
    BaseArgs.push_back(source.path);

    auto Result = ProcessRunner::run(Config.getToolPath("clang"), BaseArgs,
                                     Config.getTimeout());
    if (Result && Result->exitCode == 0) {
      std::error_code EC;
      raw_fd_ostream OS(OutputFile, EC);
      if (!EC) {
        OS << Result->stdout;
        Unit.addGeneratedFile("ast-json", OutputFile);
        if (Config.getVerbose()) {
          outs() << "AST JSON: " << OutputFile << "\n";
        }
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractDiagnostics(CompilationUnit &Unit,
                                        llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile = std::string(TempDir) + "/diagnostics/" +
                             sys::path::stem(source.path).str() +
                             ".diagnostics.txt";

    std::error_code EC;
    raw_fd_ostream OS(outputFile, EC);
    if (!EC) {
      OS << "Diagnostics for: " << source.path << "\n";

      // Run basic diagnostics
      llvm::SmallVector<std::string, 8> baseArgs = getBaseCompilerArgs(Unit.getInfo());
      baseArgs.push_back("-fdiagnostics-parseable-fixits");
      baseArgs.push_back("-fdiagnostics-absolute-paths");
      baseArgs.push_back("-Wall");
      baseArgs.push_back("-Wextra");
      baseArgs.push_back("-fsyntax-only");
      baseArgs.push_back(source.path);

      auto result = ProcessRunner::run(Config.getToolPath("clang"), baseArgs,
                                       Config.getTimeout());
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
      llvm::SmallVector<std::string, 8> extraArgs = getBaseCompilerArgs(Unit.getInfo());
      extraArgs.push_back("-Weverything");
      extraArgs.push_back("-Wno-c++98-compat");
      extraArgs.push_back("-Wno-c++98-compat-pedantic");
      extraArgs.push_back("-fsyntax-only");
      extraArgs.push_back(source.path);

      auto extraResult = ProcessRunner::run(Config.getToolPath("clang"),
                                            extraArgs, Config.getTimeout());
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

      Unit.addGeneratedFile("diagnostics", outputFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractCoverage(CompilationUnit &Unit,
                                     llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string objectFile = std::string(TempDir) + "/" + sourceStem + "_cov.o";
    std::string executableFile = std::string(TempDir) + "/" + sourceStem + "_cov";
    std::string profrawFile = std::string(TempDir) + "/" + sourceStem + ".profraw";
    std::string profdataFile = std::string(TempDir) + "/" + sourceStem + ".profdata";
    std::string coverageFile = std::string(TempDir) + "/coverage/" + sourceStem + ".coverage.json";

    // Compile with coverage instrumentation to create executable
    llvm::SmallVector<std::string, 8> compileArgs =
        getBaseCompilerArgs(Unit.getInfo());
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
          ProcessRunner::run(executableFile, {}, Config.getTimeout());

      // Convert raw profile to indexed format if profraw exists
      if (sys::fs::exists(profrawFile)) {
        llvm::SmallVector<std::string, 8> mergeArgs = {
            "merge", "-sparse", "-o", profdataFile, profrawFile};
        auto mergeResult = ProcessRunner::run("llvm-profdata", mergeArgs,
                                              Config.getTimeout());

        if (mergeResult && mergeResult->exitCode == 0 &&
            sys::fs::exists(profdataFile)) {
          // Generate coverage report in JSON format
          llvm::SmallVector<std::string, 8> covArgs = {
              "export", executableFile, "-instr-profile=" + profdataFile,
              "-format=json"};
          auto covResult =
              ProcessRunner::run("llvm-cov", covArgs, Config.getTimeout());

          if (covResult && covResult->exitCode == 0) {
            std::error_code EC;
            raw_fd_ostream OS(coverageFile, EC);
            if (!EC) {
              OS << covResult->stdout;
              Unit.addGeneratedFile("coverage", coverageFile);
              if (Config.getVerbose()) {
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

Error DataExtractor::extractTimeTrace(CompilationUnit &Unit,
                                      llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string SourceStem = sys::path::stem(source.path).str();
    std::string TraceFile = std::string(TempDir) + "/time-trace/" + SourceStem + ".trace.json";

    // Create a temporary object file in temp directory
    std::string TempObject = std::string(TempDir) + "/" + SourceStem + "_trace.o";

    llvm::SmallVector<std::string, 8> BaseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    BaseArgs.push_back("-ftime-trace");
    BaseArgs.push_back("-ftime-trace-granularity=0");
    BaseArgs.push_back("-c");
    BaseArgs.push_back("-o");
    BaseArgs.push_back(TempObject);
    BaseArgs.push_back(source.path);

    auto Result = ProcessRunner::run(Config.getToolPath("clang"), BaseArgs,
                                     Config.getTimeout());

    // The trace file is generated next to the object file with .json extension
    std::string ExpectedTraceFile = std::string(TempDir) + "/" + SourceStem + "_trace.json";

    // Also check the working directory in case trace went there
    std::string WorkingDirTrace = SourceStem + ".json";

    if (sys::fs::exists(ExpectedTraceFile)) {
      if (auto EC = sys::fs::rename(ExpectedTraceFile, TraceFile)) {
        if (Config.getVerbose()) {
          errs() << "Failed to move trace file: " << EC.message() << "\n";
        }
      } else {
        Unit.addGeneratedFile("time-trace", TraceFile);
        if (Config.getVerbose()) {
          outs() << "Time trace: " << TraceFile << "\n";
        }
      }
    } else if (sys::fs::exists(WorkingDirTrace)) {
      if (auto EC = sys::fs::rename(WorkingDirTrace, TraceFile)) {
        if (Config.getVerbose()) {
          errs() << "Failed to move trace file from working dir: " << EC.message() << "\n";
        }
      } else {
        Unit.addGeneratedFile("time-trace", TraceFile);
        if (Config.getVerbose()) {
          outs() << "Time trace: " << TraceFile << "\n";
        }
      }
    } else if (Config.getVerbose()) {
      errs() << "Time trace file not found for " << source.path << "\n";
    }

    // Clean up temporary object file
    sys::fs::remove(TempObject);
  }
  return Error::success();


}

Error DataExtractor::extractRuntimeTrace(CompilationUnit &Unit,
                                         llvm::StringRef TempDir) {

  // Check for OpenMP offloading flags
  bool HasOffloading = false;
  for (const auto &Flag : Unit.getInfo().compileFlags) {
    StringRef FlagStr(Flag);
    // Check for various OpenMP offloading indicators
    if (FlagStr.contains("offload") ||          // Generic offload flags
        FlagStr.contains("-fopenmp-targets") || // OpenMP target specification
        FlagStr.contains("-Xopenmp-target") ||  // OpenMP target-specific flags
        FlagStr.contains("nvptx") ||            // NVIDIA GPU targets
        FlagStr.contains("amdgcn") ||           // AMD GPU targets
        FlagStr.contains("-fopenmp")) {         // Basic OpenMP (may have runtime)
      HasOffloading = true;
      break;
    }
  }

  if (!HasOffloading) {
    if (Config.getVerbose())
      outs() << "Runtime trace skipped - no OpenMP offloading flags detected\n";
    return Error::success();
  }

  if (Config.getVerbose())
    outs() << "OpenMP offloading detected, attempting runtime trace extraction\n";

  // Find executable name from compile flags
  std::string ExecutableStr = "a.out"; // Default executable name
  for (size_t I = 0; I < Unit.getInfo().compileFlags.size(); ++I) {
    if (Unit.getInfo().compileFlags[I] == "-o" &&
        I + 1 < Unit.getInfo().compileFlags.size()) {
      ExecutableStr = Unit.getInfo().compileFlags[I + 1];
      break;
    }
  }

  llvm::StringRef Executable = ExecutableStr;

  if (Config.getVerbose())
    outs() << "Looking for executable: " << Executable << "\n";

  if (!sys::fs::exists(Executable)) {
    if (Config.getVerbose()) {
      outs() << "Runtime trace skipped - executable not found: " << Executable
             << "\n";
      outs() << "Note: Executable is needed to generate runtime profile data\n";
      outs() << "Checked current directory for: " << Executable << "\n";

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
  SmallString<256> TraceDir;
  sys::path::append(TraceDir, TempDir, "runtime-trace");
  if (auto ec = sys::fs::create_directories(TraceDir, true)) {
    if (Config.getVerbose())
      outs() << "Warning: Failed to create runtime trace directory: "
             << ec.message() << "\n";
  }

  // Prepare trace file
  SmallString<256> TraceFile;
  sys::path::append(TraceFile, TraceDir, "profile.json");

  if (Config.getVerbose()) {
    outs() << "Running runtime trace: " << Executable << "\n";
    outs() << "Trace file: " << TraceFile << "\n";
  }

  // Set environment variable for OpenMP target profiling
  SmallVector<std::string, 8> Env;
  Env.push_back((Twine("LIBOMPTARGET_PROFILE=") + TraceFile).str());

  if (Config.getVerbose()) {
    outs() << "Setting environment: LIBOMPTARGET_PROFILE=" << TraceFile << "\n";
    outs() << "Executing: " << Executable << "\n";
  }

  // Run executable with profiling environment
  auto Result = ProcessRunner::runWithEnv(Executable, {}, Env, Config.getTimeout());

  if (Config.getVerbose()) {
    if (Result) {
      outs() << "Runtime trace completed with exit code: " << Result->exitCode
             << "\n";

      if (!Result->stdout.empty())
        outs() << "STDOUT: " << Result->stdout << "\n";
      if (!Result->stderr.empty())
        outs() << "STDERR: " << Result->stderr << "\n";
    } else {
      outs() << "Runtime trace failed: " << toString(Result.takeError()) << "\n";
    }
  }

  // Register trace file if generated
  if (sys::fs::exists(TraceFile)) {
    Unit.addGeneratedFile("runtime-trace", TraceFile.str());
    if (Config.getVerbose())
      outs() << "Runtime trace saved: " << TraceFile << "\n";
  } else {
    if (Config.getVerbose()) {
      outs() << "Runtime trace failed - no trace file generated at: "
             << TraceFile << "\n";
      outs() << "This may happen if:\n";
      outs() << "  1. The program didn't use OpenMP target offloading\n";
      outs() << "  2. The runtime doesn't support LIBOMPTARGET_PROFILE\n";
      outs() << "  3. The program crashed before generating profile data\n";
    }
  }

  return Error::success();
}

Error DataExtractor::extractSARIF(CompilationUnit &Unit,
                                  llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string sarifFile =
        (TempDir + "/static-analyzer/" + sourceStem + ".sarif").str();

    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(Unit.getInfo());
    baseArgs.push_back("--analyze");
    baseArgs.push_back("-Xanalyzer");
    baseArgs.push_back("-analyzer-output=sarif");
    baseArgs.push_back("-o");
    baseArgs.push_back(sarifFile);
    baseArgs.push_back(source.path);

    auto result = ProcessRunner::run(Config.getToolPath("clang"), baseArgs,
                                     Config.getTimeout());

    if (result && sys::fs::exists(sarifFile)) {
      // Check if the SARIF file has content
      uint64_t fileSize;
      auto fileSizeErr = sys::fs::file_size(sarifFile, fileSize);
      if (!fileSizeErr && fileSize > 0) {
        Unit.addGeneratedFile("static-analysis-sarif", sarifFile);
        if (Config.getVerbose()) {
          outs() << "SARIF static analysis extracted: " << sarifFile << "\n";
        }
      } else if (Config.getVerbose()) {
        outs() << "SARIF file created but empty for " << source.path << "\n";
      }
    } else if (Config.getVerbose()) {
      errs() << "Failed to extract SARIF static analysis for " << source.path
             << "\n";
    }
  }
  return Error::success();
}

Error DataExtractor::extractBinarySize(CompilationUnit &Unit,
                                       llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string objectFile = std::string(TempDir) + "/" + sourceStem + "_size.o";
    std::string sizeFile =
        std::string(TempDir) + "/binary-analysis/" + sourceStem + ".size.txt";

    // Compile object file
    llvm::SmallVector<std::string, 8> baseArgs = getBaseCompilerArgs(Unit.getInfo());
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
            ProcessRunner::run("llvm-size", berkeleyArgs, Config.getTimeout());
        if (berkeleyResult && berkeleyResult->exitCode == 0) {
          OS << "Berkeley format (default):\n"
             << berkeleyResult->stdout << "\n";
        }

        // Also try -A format for more details
        llvm::SmallVector<std::string, 8> sysVArgs = {"-A", objectFile};
        auto sysVResult =
            ProcessRunner::run("llvm-size", sysVArgs, Config.getTimeout());
        if (sysVResult && sysVResult->exitCode == 0)
          OS << "System V format:\n" << sysVResult->stdout << "\n";

        Unit.addGeneratedFile("binary-size", sizeFile);
        if (Config.getVerbose()) {
          outs() << "Binary size: " << sizeFile << "\n";
        }
      }

      // Clean up temporary object file
      sys::fs::remove(objectFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractPGO(CompilationUnit &Unit,
                                llvm::StringRef TempDir) {
  // Look for existing profile raw data file from compilation
  std::string profrawFile = std::string(TempDir) + "/profile.profraw";

  if (sys::fs::exists(profrawFile)) {
    std::string profileFile = (TempDir + "/pgo/merged.profdata").str();
    std::string profileText = (TempDir + "/pgo/profile.txt").str();

    // Convert raw profile to indexed format
    llvm::SmallVector<std::string, 8> mergeArgs = {"merge", "-sparse", "-o",
                                                   profileFile, profrawFile};
    auto mergeResult =
        ProcessRunner::run("llvm-profdata", mergeArgs, Config.getTimeout());

    if (mergeResult && mergeResult->exitCode == 0 &&
        sys::fs::exists(profileFile)) {
      // Generate human-readable profile summary
      llvm::SmallVector<std::string, 8> showArgs = {
          "show", "-all-functions", "-text", "-detailed-summary", profileFile};
      auto showResult =
          ProcessRunner::run("llvm-profdata", showArgs, Config.getTimeout());

      if (showResult && showResult->exitCode == 0) {
        std::error_code EC;
        raw_fd_ostream OS(profileText, EC);
        if (!EC) {
          OS << "PGO Profile Data Summary\n";
          OS << "========================\n";
          OS << "Source profile: " << profrawFile << "\n";
          OS << "Indexed profile: " << profileFile << "\n\n";
          OS << showResult->stdout;
          Unit.addGeneratedFile("pgo-profile", profileText);
          if (Config.getVerbose()) {
            outs() << "PGO profile data extracted: " << profileText << "\n";
          }
        }
      }

      // Also create a JSON-like summary if possible
      llvm::SmallVector<std::string, 8> jsonArgs = {"show", "-json",
                                                    profileFile};
      auto jsonResult =
          ProcessRunner::run("llvm-profdata", jsonArgs, Config.getTimeout());

      if (jsonResult && jsonResult->exitCode == 0) {
        std::string jsonFile = (TempDir + "/pgo/profile.json").str();
        std::error_code EC;
        raw_fd_ostream jsonOS(jsonFile, EC);
        if (!EC) {
          jsonOS << jsonResult->stdout;
          Unit.addGeneratedFile("pgo-profile-json", jsonFile);
          if (Config.getVerbose()) {
            outs() << "PGO profile JSON extracted: " << jsonFile << "\n";
          }
        }
      }
      sys::fs::remove(profileFile);
    } else if (Config.getVerbose()) {
      errs() << "Failed to merge PGO profile data\n";
    }
  } else if (Config.getVerbose()) {
    outs() << "No PGO profile data found to extract\n";
  }

  return Error::success();
}

Error DataExtractor::extractSymbols(CompilationUnit &Unit,
                                    llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string objectFile = std::string(TempDir) + "/" + sourceStem + "_symbols.o";
    std::string symbolsFile =
        std::string(TempDir) + "/binary-analysis/" + sourceStem + ".symbols.txt";

    // Compile object file
    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(Unit.getInfo());
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
            ProcessRunner::run("llvm-nm", nmArgs, Config.getTimeout());
        if (nmResult && nmResult->exitCode == 0)
          OS << "Symbols:\n" << nmResult->stdout << "\n";

        Unit.addGeneratedFile("symbols", symbolsFile);
        if (Config.getVerbose()) {
          outs() << "Symbols: " << symbolsFile << "\n";
        }
      }

      // Clean up temporary object file
      sys::fs::remove(objectFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractObjdump(CompilationUnit &Unit,
                                    llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string objectFile = std::string(TempDir) + "/" + sourceStem + "_objdump.o";
    std::string objdumpFile =
        std::string(TempDir) + "/binary-analysis/" + sourceStem + ".objdump.txt";

    // Compile object file
    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(Unit.getInfo());
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
                                                Config.getTimeout());
        if (objdumpResult && objdumpResult->exitCode == 0)
          OS << objdumpResult->stdout << "\n";

        Unit.addGeneratedFile("objdump", objdumpFile);
        if (Config.getVerbose()) {
          outs() << "Objdump: " << objdumpFile << "\n";
        }
      }

      // Clean up temporary object file
      sys::fs::remove(objectFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractXRay(CompilationUnit &Unit,
                                 llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string executableFile = std::string(TempDir) + "/" + sourceStem + "_xray";
    std::string xrayFile =
        std::string(TempDir) + "/binary-analysis/" + sourceStem + ".xray.txt";

    // Compile with XRay instrumentation
    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(Unit.getInfo());
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
        llvm::SmallVector<std::string, 8> xrayArgs = {"extract", executableFile};
        auto xrayResult =
            ProcessRunner::run("llvm-xray", xrayArgs, Config.getTimeout());
        if (xrayResult && xrayResult->exitCode == 0)
          OS << "XRay instrumentation:\n" << xrayResult->stdout << "\n";

        Unit.addGeneratedFile("xray", xrayFile);
        if (Config.getVerbose()) {
          outs() << "XRay: " << xrayFile << "\n";
        }
      }

      // Clean up temporary executable
      sys::fs::remove(executableFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractOptDot(CompilationUnit &Unit,
                                   llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string sourceStem = sys::path::stem(source.path).str();
    std::string bitcodeFile = std::string(TempDir) + "/" + sourceStem + ".bc";
    std::string dotFile = std::string(TempDir) + "/ir/" + sourceStem + ".cfg.dot";

    // Compile to bitcode
    llvm::SmallVector<std::string, 8> baseArgs =
        getBaseCompilerArgs(Unit.getInfo());
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
      auto optResult = ProcessRunner::run("opt", optArgs, Config.getTimeout());

      // The dot file is typically generated in current directory
      std::string expectedDotFile = "." + sourceStem + ".dot";
      if (sys::fs::exists(expectedDotFile)) {
        if (auto EC = sys::fs::rename(expectedDotFile, dotFile)) {
          if (Config.getVerbose()) {
            errs() << "Failed to move dot file: " << EC.message() << "\n";
          }
        } else {
          Unit.addGeneratedFile("cfg-dot", dotFile);
          if (Config.getVerbose()) {
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

Error DataExtractor::extractSources(CompilationUnit &Unit,
                                    llvm::StringRef TempDir) {
  if (Config.getVerbose()) {
    outs() << "Extracting source files based on dependencies...\n";
  }

  // Create sources directory
  SmallString<256> SourcesDir;
  sys::path::append(SourcesDir, TempDir, "sources");
  if (auto EC = sys::fs::create_directories(SourcesDir)) {
    if (Config.getVerbose()) {
      outs() << "Warning: Failed to create sources directory: " << EC.message()
             << "\n";
    }
    return Error::success(); // Continue even if we can't create directory
  }

  // Find and parse dependencies files
  SmallString<256> depsDir;
  sys::path::append(depsDir, TempDir, "dependencies");

  if (!sys::fs::exists(depsDir)) {
    if (Config.getVerbose()) {
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

    if (Config.getVerbose()) {
      outs() << "Processing dependencies file: " << filePath << "\n";
    }

    // Read and parse dependencies file
    auto bufferOrErr = MemoryBuffer::getFile(filePath);
    if (!bufferOrErr) {
      if (Config.getVerbose()) {
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
      if (line.empty() || !line.contains(":"))
        continue;

      // Parse line format: "target.o: source1.c source2.h ..."
      auto colonPos = line.find(':');
      if (colonPos == StringRef::npos)
        continue;

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
          if (Config.getVerbose()) {
            outs() << "Warning: Source file not found: " << absoluteSourcePath
                   << "\n";
          }
          continue;
        }

        // Create destination path preserving directory structure
        SmallString<256> destPath;
        sys::path::append(destPath, SourcesDir,
                          sys::path::filename(sourceFile));

        // Copy source file to sources directory
        if (auto copyErr = FileManager::copyFile(absoluteSourcePath.str(),
                                                 destPath.str())) {
          if (Config.getVerbose()) {
            outs() << "Warning: Failed to copy source file " << absoluteSourcePath
                   << " to " << destPath.str() << "\n";
          }
        } else {
          Unit.addGeneratedFile("sources", destPath.str());
          if (Config.getVerbose()) {
            outs() << "Copied source: " << sourceFile << " -> " << destPath.str()
                   << "\n";
          }
        }
      }
    }
  }

  return Error::success();
}

const DataExtractor::ExtractorInfo DataExtractor::extractors[] = {
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

const size_t DataExtractor::numExtractors =
    sizeof(DataExtractor::extractors) / sizeof(DataExtractor::ExtractorInfo);

} // namespace advisor
} // namespace llvm
