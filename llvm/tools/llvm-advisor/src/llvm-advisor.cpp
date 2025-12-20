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
#include "Core/ViewerLauncher.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string>
    g_ConfigFile("config", llvm::cl::desc("Configuration file"),
               llvm::cl::value_desc("filename"));
static llvm::cl::opt<std::string> g_OutputDir("output-dir",
                                            llvm::cl::desc("Output directory"),
                                            llvm::cl::value_desc("directory"));
static llvm::cl::opt<bool> g_Verbose("verbose", llvm::cl::desc("Verbose output"));
static llvm::cl::opt<bool> g_KeepTemps("keep-temps",
                                     llvm::cl::desc("Keep temporary files"));
static llvm::cl::opt<bool> g_NoProfiler("no-profiler",
                                      llvm::cl::desc("Disable profiler"));
static llvm::cl::opt<int>
    g_Port("port", llvm::cl::desc("Web server port (for view command)"),
         llvm::cl::value_desc("port"), llvm::cl::init(8000));

auto main(int argc, char **argv) -> int {
  llvm::InitLLVM X(argc, argv);

  // Handle help and subcommands before argument parsing
  if (argc > 1) {
    llvm::StringRef firstArg(argv[1]);
    if (firstArg == "--help" || firstArg == "-h") {
      llvm::outs() << "LLVM Advisor - Compilation analysis tool\n\n";
      llvm::outs() << "Usage:\n";
      llvm::outs() << "  llvm-advisor [options] <compiler> [compiler-args...]  "
                      "   - Compile with data collection\n";
      llvm::outs() << "  llvm-advisor view [options] <compiler> "
                      "[compiler-args...] - Compile and launch web viewer\n\n";
      llvm::outs() << "Examples:\n";
      llvm::outs() << "  llvm-advisor clang -O2 -g main.c\n";
      llvm::outs() << "  llvm-advisor view --port 8080 clang++ -O3 app.cpp\n\n";
      llvm::outs() << "Options:\n";
      llvm::outs() << "  --config <file>      Configuration file\n";
      llvm::outs() << "  --output-dir <dir>   Output directory (default: "
                      ".llvm-advisor)\n";
      llvm::outs() << "  --verbose            Verbose output\n";
      llvm::outs() << "  --keep-temps         Keep temporary files\n";
      llvm::outs() << "  --no-profiler        Disable profiler\n";
      llvm::outs() << "  --port <port>        Web server port for view command "
                      "(default: 8000)\n";
      return 0;
    }
  }

  // Check for 'view' subcommand
  bool isViewCommand = false;
  int argOffset = 0;

  if (argc > 1 && llvm::StringRef(argv[1]) == "view") {
    isViewCommand = true;
    argOffset = 1;
  }

  // Parse llvm-advisor options until we find the compiler
  llvm::SmallVector<const char *, 8> advisorArgs;
  advisorArgs.push_back(argv[0]);

  int compilerArgStart = 1 + argOffset;
  bool foundCompiler = false;

  for (int i = 1 + argOffset; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg.starts_with("--") ||
        (arg.starts_with("-") && arg.size() > 1 && arg != "-")) {
      advisorArgs.push_back(argv[i]);
      if (arg == "--config" || arg == "--output-dir" || arg == "--port") {
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
    if (isViewCommand) {
      llvm::errs() << "Usage: llvm-advisor view [options] <compiler> "
                      "[compiler-args...]\n";
    } else {
      llvm::errs()
          << "Usage: llvm-advisor [options] <compiler> [compiler-args...]\n";
    }
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
  if (!g_ConfigFile.empty()) {
    if (auto Err = config.loadFromFile(g_ConfigFile).takeError()) {
      llvm::errs() << "Error loading config: " << llvm::toString(std::move(Err))
                   << "\n";
      return 1;
    }
  }

  if (!g_OutputDir.empty()) {
    config.setOutputDir(g_OutputDir);
  } else {
    config.setOutputDir(".llvm-advisor"); // Default hidden directory
  }

  config.setVerbose(g_Verbose);
  config.setKeepTemps(g_KeepTemps ||
                      isViewCommand); // Keep temps for view command
  config.setRunProfiler(!g_NoProfiler);

  // Create output directory
  if (auto EC = llvm::sys::fs::create_directories(config.getOutputDir())) {
    llvm::errs() << "Error creating output directory: " << EC.message() << "\n";
    return 1;
  }

  if (config.getVerbose()) {
    llvm::outs() << "LLVM Compilation Advisor\n";
    llvm::outs() << "Compiler: " << compiler << "\n";
    llvm::outs() << "Output: " << config.getOutputDir() << "\n";
    if (isViewCommand) {
      llvm::outs() << "Mode: Compile and launch web viewer\n";
    }
  }

  // Execute with data collection
  llvm::advisor::CompilationManager manager(config);
  auto result = manager.executeWithDataCollection(compiler, compilerArgs);

  if (!result) {
    llvm::errs() << "Error: " << llvm::toString(result.takeError()) << "\n";
    return 1;
  }

  if (config.getVerbose()) {
    llvm::outs() << "Compilation completed (exit code: " << *result << ")\n";
  }

  // If this is a view command and compilation succeeded, launch the web viewer
  if (isViewCommand && *result == 0) {
    if (config.getVerbose()) {
      llvm::outs() << "Launching web viewer...\n";
    }

    // Convert output directory to absolute path for web viewer
    llvm::SmallString<256> absoluteOutputDir;
    if (llvm::sys::path::is_absolute(config.getOutputDir())) {
      absoluteOutputDir = config.getOutputDir();
    } else {
      llvm::sys::fs::current_path(absoluteOutputDir);
      llvm::sys::path::append(absoluteOutputDir, config.getOutputDir());
    }

    auto viewerResult = llvm::advisor::ViewerLauncher::launch(
        std::string(absoluteOutputDir.str()), g_Port);
    if (!viewerResult) {
      llvm::errs() << "Error launching web viewer: "
                   << llvm::toString(viewerResult.takeError()) << "\n";
      llvm::errs() << "Compilation data is still available in: "
                   << config.getOutputDir() << "\n";
      return 1;
    }

    return *viewerResult;
  }

  return *result;
}
