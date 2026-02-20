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

namespace {

llvm::cl::OptionCategory AdvisorCategory("llvm-advisor options");

llvm::cl::opt<std::string> g_ConfigFile("config",
                                        llvm::cl::desc("Configuration file"),
                                        llvm::cl::value_desc("filename"),
                                        llvm::cl::cat(AdvisorCategory));
llvm::cl::opt<std::string> g_OutputDir("output-dir",
                                       llvm::cl::desc("Output directory"),
                                       llvm::cl::value_desc("directory"),
                                       llvm::cl::cat(AdvisorCategory));
llvm::cl::opt<bool> g_Verbose("verbose", llvm::cl::desc("Verbose output"),
                              llvm::cl::cat(AdvisorCategory));
llvm::cl::opt<bool> g_KeepTemps("keep-temps",
                                llvm::cl::desc("Keep temporary files"),
                                llvm::cl::cat(AdvisorCategory));
llvm::cl::opt<bool> g_NoProfiler("no-profiler",
                                 llvm::cl::desc("Disable profiler"),
                                 llvm::cl::cat(AdvisorCategory));

llvm::cl::SubCommand ViewCmd("view", "Compile and launch web viewer");

llvm::cl::opt<int> g_Port("port",
                          llvm::cl::desc("Web server port (for view command)"),
                          llvm::cl::value_desc("port"), llvm::cl::init(8000),
                          llvm::cl::sub(ViewCmd));

llvm::cl::list<std::string> g_DefaultCompileCommand(
    llvm::cl::Positional, llvm::cl::desc("<compiler> [compiler-args...]"),
    llvm::cl::ZeroOrMore, llvm::cl::cat(AdvisorCategory));
llvm::cl::list<std::string>
    g_ViewCompileCommand(llvm::cl::Positional,
                         llvm::cl::desc("<compiler> [compiler-args...]"),
                         llvm::cl::ZeroOrMore, llvm::cl::sub(ViewCmd));

} // namespace

auto main(int argc, char **argv) -> int {
  llvm::InitLLVM X(argc, argv);

  llvm::cl::HideUnrelatedOptions({&AdvisorCategory});
  llvm::cl::ParseCommandLineOptions(argc, argv, "LLVM Compilation Advisor");

  const bool isViewCommand = ViewCmd;
  const auto &CommandLine =
      isViewCommand ? g_ViewCompileCommand : g_DefaultCompileCommand;
  if (CommandLine.empty()) {
    llvm::errs() << "error: missing compiler command\n";
    llvm::cl::PrintHelpMessage();
    return 1;
  }

  std::string compiler = CommandLine.front();
  llvm::SmallVector<std::string, 8> compilerArgs;
  for (size_t i = 1; i < CommandLine.size(); ++i)
    compilerArgs.push_back(CommandLine[i]);

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
