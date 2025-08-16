//===-------------- llvm-advisor.cpp - LLVM Advisor -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the llvm-advisor code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "Config/AdvisorConfig.h"
#include "Core/CompilationManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string>
    ConfigFile("config", llvm::cl::desc("Configuration file"),
               llvm::cl::value_desc("filename"));
static llvm::cl::opt<std::string> OutputDir("output-dir",
                                            llvm::cl::desc("Output directory"),
                                            llvm::cl::value_desc("directory"));
static llvm::cl::opt<bool> Verbose("verbose", llvm::cl::desc("Verbose output"));
static llvm::cl::opt<bool> KeepTemps("keep-temps",
                                     llvm::cl::desc("Keep temporary files"));
static llvm::cl::opt<bool> NoProfiler("no-profiler",
                                      llvm::cl::desc("Disable profiler"));

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);

  // Parse llvm-advisor options until we find the compiler
  llvm::SmallVector<const char *, 8> advisorArgs;
  advisorArgs.push_back(argv[0]);

  int compilerArgStart = 1;
  bool foundCompiler = false;

  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg.starts_with("--") ||
        (arg.starts_with("-") && arg.size() > 1 && arg != "-")) {
      advisorArgs.push_back(argv[i]);
      if (arg == "--config" || arg == "--output-dir") {
        if (i + 1 < argc && !llvm::StringRef(argv[i + 1]).starts_with("-")) {
          advisorArgs.push_back(argv[++i]);
        }
      }
    } else {
      compilerArgStart = i;
      foundCompiler = true;
      break;
    }
  }

  if (!foundCompiler) {
    llvm::errs() << "Error: No compiler command provided.\n";
    llvm::errs()
        << "Usage: llvm-advisor [options] <compiler> [compiler-args...]\n";
    return 1;
  }

  // Parse llvm-advisor options
  int advisorArgc = static_cast<int>(advisorArgs.size());
  llvm::cl::ParseCommandLineOptions(advisorArgc,
                                    const_cast<char **>(advisorArgs.data()),
                                    "LLVM Compilation Advisor");

  // Extract compiler and arguments
  std::string compiler = argv[compilerArgStart];
  llvm::SmallVector<std::string, 8> compilerArgs;
  for (int i = compilerArgStart + 1; i < argc; ++i) {
    compilerArgs.push_back(argv[i]);
  }

  // Configure advisor
  llvm::advisor::AdvisorConfig config;
  if (!ConfigFile.empty()) {
    if (auto Err = config.loadFromFile(ConfigFile).takeError()) {
      llvm::errs() << "Error loading config: " << llvm::toString(std::move(Err))
                   << "\n";
      return 1;
    }
  }

  if (!OutputDir.empty()) {
    config.setOutputDir(OutputDir);
  } else {
    config.setOutputDir(".llvm-advisor"); // Default hidden directory
  }

  config.setVerbose(Verbose);
  config.setKeepTemps(KeepTemps);
  config.setRunProfiler(!NoProfiler);

  // Create output directory
  if (auto EC = llvm::sys::fs::create_directories(config.getOutputDir())) {
    llvm::errs() << "Error creating output directory: " << EC.message() << "\n";
    return 1;
  }

  if (config.getVerbose()) {
    llvm::outs() << "LLVM Compilation Advisor\n";
    llvm::outs() << "Compiler: " << compiler << "\n";
    llvm::outs() << "Output: " << config.getOutputDir() << "\n";
  }

  // Execute with data collection
  llvm::advisor::CompilationManager manager(config);
  auto result = manager.executeWithDataCollection(compiler, compilerArgs);

  if (result) {
    if (config.getVerbose()) {
      llvm::outs() << "Compilation completed (exit code: " << *result << ")\n";
    }
    return *result;
  } else {
    llvm::errs() << "Error: " << llvm::toString(result.takeError()) << "\n";
    return 1;
  }
}
