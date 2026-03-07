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

llvm::cl::opt<std::string> GConfigFile("config",
                                       llvm::cl::desc("Configuration file"),
                                       llvm::cl::value_desc("filename"),
                                       llvm::cl::cat(AdvisorCategory));
llvm::cl::opt<std::string> GOutputDir("output-dir",
                                      llvm::cl::desc("Output directory"),
                                      llvm::cl::value_desc("directory"),
                                      llvm::cl::cat(AdvisorCategory));
llvm::cl::opt<bool> GVerbose("verbose", llvm::cl::desc("Verbose output"),
                             llvm::cl::cat(AdvisorCategory));
llvm::cl::opt<bool> GKeepTemps("keep-temps",
                               llvm::cl::desc("Keep temporary files"),
                               llvm::cl::cat(AdvisorCategory));
llvm::cl::opt<bool> GNoProfiler("no-profiler",
                                llvm::cl::desc("Disable profiler"),
                                llvm::cl::cat(AdvisorCategory));

llvm::cl::SubCommand ViewCmd("view", "Compile and launch web viewer");

llvm::cl::opt<int> GPort("port",
                         llvm::cl::desc("Web server port (for view command)"),
                         llvm::cl::value_desc("port"), llvm::cl::init(8000),
                         llvm::cl::sub(ViewCmd));

llvm::cl::list<std::string> GDefaultCompileCommand(
    llvm::cl::Positional, llvm::cl::desc("<compiler> [compiler-args...]"),
    llvm::cl::ZeroOrMore, llvm::cl::cat(AdvisorCategory));
llvm::cl::list<std::string>
    GViewCompileCommand(llvm::cl::Positional,
                        llvm::cl::desc("<compiler> [compiler-args...]"),
                        llvm::cl::ZeroOrMore, llvm::cl::sub(ViewCmd));

} // namespace

auto main(int argc, char **argv) -> int {
  llvm::InitLLVM X(argc, argv);

  llvm::cl::HideUnrelatedOptions({&AdvisorCategory});
  llvm::cl::ParseCommandLineOptions(argc, argv, "LLVM Compilation Advisor");

  const bool IsViewCommand = static_cast<bool>(ViewCmd);
  const auto &CommandLine =
      IsViewCommand ? GViewCompileCommand : GDefaultCompileCommand;
  if (CommandLine.empty()) {
    llvm::errs() << "error: missing compiler command\n";
    llvm::cl::PrintHelpMessage();
    return 1;
  }

  std::string Compiler = CommandLine.front();
  llvm::SmallVector<std::string, 8> CompilerArgs;
  for (size_t I = 1; I < CommandLine.size(); ++I)
    CompilerArgs.push_back(CommandLine[I]);

  // Configure advisor
  llvm::advisor::AdvisorConfig Config;
  if (!GConfigFile.empty()) {
    if (auto Err = Config.loadFromFile(GConfigFile).takeError()) {
      llvm::errs() << "Error loading config: " << llvm::toString(std::move(Err))
                   << "\n";
      return 1;
    }
  }

  if (!GOutputDir.empty()) {
    Config.setOutputDir(GOutputDir);
  } else {
    Config.setOutputDir(".llvm-advisor"); // Default hidden directory
  }

  Config.setVerbose(GVerbose);
  Config.setKeepTemps(GKeepTemps ||
                      IsViewCommand); // Keep temps for view command
  Config.setRunProfiler(!GNoProfiler);

  // Create output directory
  if (auto EC = llvm::sys::fs::create_directories(Config.getOutputDir())) {
    llvm::errs() << "Error creating output directory: " << EC.message() << "\n";
    return 1;
  }

  if (Config.getVerbose()) {
    llvm::outs() << "LLVM Compilation Advisor\n";
    llvm::outs() << "Compiler: " << Compiler << "\n";
    llvm::outs() << "Output: " << Config.getOutputDir() << "\n";
    if (IsViewCommand)
      llvm::outs() << "Mode: Compile and launch web viewer\n";
  }

  // Execute with data collection
  llvm::advisor::CompilationManager Manager(Config);
  auto Result = Manager.executeWithDataCollection(Compiler, CompilerArgs);

  if (!Result) {
    llvm::errs() << "Error: " << llvm::toString(Result.takeError()) << "\n";
    return 1;
  }

  if (Config.getVerbose())
    llvm::outs() << "Compilation completed (exit code: " << *Result << ")\n";

  // If this is a view command and compilation succeeded, launch the web viewer
  if (IsViewCommand && *Result == 0) {
    if (Config.getVerbose())
      llvm::outs() << "Launching web viewer...\n";

    // Convert output directory to absolute path for web viewer
    llvm::SmallString<256> AbsoluteOutputDir;
    if (llvm::sys::path::is_absolute(Config.getOutputDir())) {
      AbsoluteOutputDir = Config.getOutputDir();
    } else {
      llvm::sys::fs::current_path(AbsoluteOutputDir);
      llvm::sys::path::append(AbsoluteOutputDir, Config.getOutputDir());
    }

    auto ViewerResult = llvm::advisor::ViewerLauncher::launch(
        std::string(AbsoluteOutputDir.str()), GPort);
    if (!ViewerResult) {
      llvm::errs() << "Error launching web viewer: "
                   << llvm::toString(ViewerResult.takeError()) << "\n";
      llvm::errs() << "Compilation data is still available in: "
                   << Config.getOutputDir() << "\n";
      return 1;
    }

    return *ViewerResult;
  }

  return *Result;
}
