//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LLVM Advisor Tool - Unified C++ wrapper for LLVM optimization analysis
// and performance guidance. Provides compiler wrapper, web viewer launcher,
// and automated profiling capabilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

extern char **environ;

namespace {

struct RemarksConfig {
  std::string tempDir;
  std::string outputDir;
  bool verbose = false;
  bool keepTempFiles = false;
};

class CompilerWrapper {
public:
  CompilerWrapper(const std::string &compiler) : compiler_(compiler) {}

  Expected<int> execute(const std::vector<std::string> &args,
                        RemarksConfig &config);

private:
  std::string compiler_;

  std::vector<std::string>
  buildRemarksArgs(const std::vector<std::string> &originalArgs,
                   const RemarksConfig &config);
  Expected<std::string> createTempDir(const RemarksConfig &config);
  void cleanup(const std::string &tempDir, bool keepFiles);
  Expected<int>
  tryExecuteForProfiling(const std::vector<std::string> &originalArgs,
                         const std::string &tempDir,
                         const RemarksConfig &config);
  std::string findExecutableFromArgs(const std::vector<std::string> &args);
  bool looksLikeOpenMPProgram(const std::vector<std::string> &args);
};

class ViewerLauncher {
public:
  static Expected<int> launch(const std::string &remarksDir, int port = 8080);

private:
  static Expected<std::string> findPythonExecutable();
  static Expected<std::string> getViewerScript();
};

class SubcommandHandler {
public:
  virtual ~SubcommandHandler() = default;
  virtual Expected<int> execute(const std::vector<std::string> &args) = 0;
  virtual const char *getName() const = 0;
  virtual const char *getDescription() const = 0;
};

class ViewSubcommand : public SubcommandHandler {
public:
  Expected<int> execute(const std::vector<std::string> &args) override;
  const char *getName() const override { return "view"; }
  const char *getDescription() const override {
    return "Compile with remarks and launch web viewer";
  }
};

class CompileSubcommand : public SubcommandHandler {
public:
  Expected<int> execute(const std::vector<std::string> &args) override;
  const char *getName() const override { return "compile"; }
  const char *getDescription() const override {
    return "Compile with remarks generation";
  }
};

} // anonymous namespace

Expected<std::string>
CompilerWrapper::createTempDir(const RemarksConfig &config) {
  if (!config.tempDir.empty()) {
    if (sys::fs::create_directories(config.tempDir)) {
      return createStringError(
          std::make_error_code(std::errc::io_error),
          ("Failed to create temp directory: " + config.tempDir).c_str());
    }
    return config.tempDir;
  }

  SmallString<128> tempDir;
  if (sys::fs::createUniqueDirectory("llvm-advisor", tempDir)) {
    return createStringError(std::make_error_code(std::errc::io_error),
                             "Failed to create unique temp directory");
  }

  return tempDir.str().str();
}

std::vector<std::string>
CompilerWrapper::buildRemarksArgs(const std::vector<std::string> &originalArgs,
                                  const RemarksConfig &config) {

  std::vector<std::string> newArgs = originalArgs;

  // Add optimization remarks flags
  newArgs.push_back("-Rpass=.*");
  newArgs.push_back("-Rpass-missed=.*");
  newArgs.push_back("-Rpass-analysis=.*");

  // Add YAML output flags - use the newer format
  newArgs.push_back("-fsave-optimization-record");

  // Set output directory for YAML files in temp directory
  if (!config.tempDir.empty()) {
    newArgs.push_back("-foptimization-record-file=" + config.tempDir +
                      "/remarks");
  }

  return newArgs;
}

Expected<int> CompilerWrapper::execute(const std::vector<std::string> &args,
                                       RemarksConfig &config) {
  if (args.empty()) {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "No compiler arguments provided");
  }

  auto tempDirOrErr = createTempDir(config);
  if (!tempDirOrErr) {
    return tempDirOrErr.takeError();
  }

  std::string tempDir = *tempDirOrErr;
  config.tempDir = tempDir;
  config.outputDir = tempDir;

  auto remarksArgs = buildRemarksArgs(args, config);

  // Execute compiler
  auto compilerPath = sys::findProgramByName(compiler_);
  if (!compilerPath) {
    cleanup(tempDir, config.keepTempFiles);
    return createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        ("Compiler not found: " + compiler_).c_str());
  }

  std::vector<StringRef> execArgs;
  execArgs.push_back(compiler_);
  for (const auto &arg : remarksArgs) {
    execArgs.push_back(arg);
  }

  int result = sys::ExecuteAndWait(*compilerPath, execArgs);

  if (result != 0) {
    cleanup(tempDir, config.keepTempFiles);
    return createStringError(
        std::make_error_code(std::errc::io_error),
        ("Compiler execution failed with exit code: " + std::to_string(result))
            .c_str());
  }

  // Attempt runtime profiling
  auto executeResult = tryExecuteForProfiling(args, tempDir, config);
  if (!executeResult && config.verbose) {
    outs() << "Warning: " << executeResult.takeError() << "\n";
  }

  config.outputDir = tempDir;
  return 0;
}

Expected<int> CompilerWrapper::tryExecuteForProfiling(
    const std::vector<std::string> &originalArgs, const std::string &tempDir,
    const RemarksConfig &config) {
  std::string executablePath = findExecutableFromArgs(originalArgs);
  if (executablePath.empty() || !sys::fs::exists(executablePath)) {
    return createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "Executable not found for profiling");
  }

  if (!looksLikeOpenMPProgram(originalArgs)) {
    return createStringError(
        std::make_error_code(std::errc::operation_not_supported),
        "Program does not use OpenMP offloading");
  }

  // Prepare environment with profiling variables
  std::vector<StringRef> environment;

  // Get current environment variables
  char **envp = environ;
  while (*envp) {
    environment.emplace_back(*envp);
    ++envp;
  }

  // Add profiling environment variable
  std::string profilingEnv = "LIBOMPTARGET_PROFILE=profile.json";
  environment.emplace_back(profilingEnv);

  // Execute with custom environment
  std::string execPath = sys::path::is_absolute(executablePath)
                             ? executablePath
                             : "./" + executablePath;
  std::vector<StringRef> execArgs = {execPath};

  std::optional<StringRef> redirects[] = {std::nullopt, std::nullopt,
                                          std::nullopt};
  int result =
      sys::ExecuteAndWait(execPath, execArgs, environment, redirects, 10);

  if (result != 0) {
    return createStringError(std::make_error_code(std::errc::io_error),
                             "Program execution failed");
  }

  // Move profile.json to temp directory
  SmallString<128> currentProfile("profile.json");
  SmallString<128> tempProfile;
  sys::path::append(tempProfile, tempDir, "profile.json");

  if (sys::fs::exists(currentProfile)) {
    if (auto err = sys::fs::rename(currentProfile, tempProfile)) {
      return createStringError(err,
                               "Failed to move profile.json to temp directory");
    }
  }

  return 0;
}

std::string
CompilerWrapper::findExecutableFromArgs(const std::vector<std::string> &args) {
  // Look for -o flag to find output executable
  for (size_t i = 0; i < args.size() - 1; i++) {
    if (args[i] == "-o") {
      return args[i + 1];
    }
  }

  // Default executable name if no -o specified
  return "a.out";
}

bool CompilerWrapper::looksLikeOpenMPProgram(
    const std::vector<std::string> &args) {
  // Check compilation flags for OpenMP indicators
  for (const auto &arg : args) {
    if (arg.find("-fopenmp") != std::string::npos ||
        arg.find("-fopenmp-targets") != std::string::npos ||
        arg.find("-mp") != std::string::npos) { // Intel compiler flag
      return true;
    }
  }
  return false;
}

void CompilerWrapper::cleanup(const std::string &tempDir, bool keepFiles) {
  if (!keepFiles && !tempDir.empty()) {
    sys::fs::remove_directories(tempDir);
  }
}

Expected<std::string> ViewerLauncher::findPythonExecutable() {
  std::vector<std::string> candidates = {"python3", "python"};

  for (const auto &candidate : candidates) {
    if (auto path = sys::findProgramByName(candidate)) {
      return *path;
    }
  }

  return createStringError(
      std::make_error_code(std::errc::no_such_file_or_directory),
      "Python executable not found");
}

Expected<std::string> ViewerLauncher::getViewerScript() {
  SmallString<256> scriptPath;

  // Try to find the script relative to the executable
  auto mainExecutable = sys::fs::getMainExecutable(nullptr, nullptr);
  if (mainExecutable.empty()) {
    return createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "Cannot determine executable path");
  }

  // First try: relative to binary (development/build tree)
  sys::path::append(scriptPath, sys::path::parent_path(mainExecutable));
  sys::path::append(scriptPath, "view");
  sys::path::append(scriptPath, "cli");
  sys::path::append(scriptPath, "main.py");

  if (sys::fs::exists(scriptPath)) {
    return scriptPath.str().str();
  }

  // Second try: installed location
  scriptPath.clear();
  sys::path::append(scriptPath, sys::path::parent_path(mainExecutable));
  sys::path::append(scriptPath, "..");
  sys::path::append(scriptPath, "share");
  sys::path::append(scriptPath, "llvm-advisor");
  sys::path::append(scriptPath, "view");
  sys::path::append(scriptPath, "cli");
  sys::path::append(scriptPath, "main.py");

  if (sys::fs::exists(scriptPath)) {
    return scriptPath.str().str();
  }

  return createStringError(
      std::make_error_code(std::errc::no_such_file_or_directory),
      "Viewer script not found");
}

Expected<int> ViewerLauncher::launch(const std::string &remarksDir, int port) {
  auto pythonOrErr = findPythonExecutable();
  if (!pythonOrErr) {
    return pythonOrErr.takeError();
  }

  auto scriptOrErr = getViewerScript();
  if (!scriptOrErr) {
    return scriptOrErr.takeError();
  }

  // Get current working directory for source files
  SmallString<128> currentDir;
  if (sys::fs::current_path(currentDir)) {
    return createStringError(std::make_error_code(std::errc::io_error),
                             "Failed to get current working directory");
  }

  std::vector<StringRef> args = {
      *pythonOrErr, *scriptOrErr,         "--directory",
      remarksDir,   "--source-directory", currentDir.str(),
      "--port",     std::to_string(port)};

  outs() << "Launching viewer at http://localhost:" << port << "\n";
  outs() << "Loading remarks from: " << remarksDir << "\n";
  outs() << "Source files from: " << currentDir.str() << "\n";

  return sys::ExecuteAndWait(*pythonOrErr, args);
}

Expected<int> ViewSubcommand::execute(const std::vector<std::string> &args) {
  if (args.empty()) {
    return createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Usage: llvm-advisor view <compiler> [compiler-args...]");
  }

  std::string compiler = args[0];
  std::vector<std::string> compilerArgs(args.begin() + 1, args.end());

  RemarksConfig config;
  config.verbose = true;
  config.keepTempFiles = true; // Keep temp files for viewing
  config.tempDir = "";         // Will be created automatically
  config.outputDir = "";

  CompilerWrapper wrapper(compiler);
  auto compileResult = wrapper.execute(compilerArgs, config);
  if (!compileResult) {
    return compileResult.takeError();
  }

  // Launch viewer with the isolated temporary directory
  return ViewerLauncher::launch(config.outputDir, 8080);
}

Expected<int> CompileSubcommand::execute(const std::vector<std::string> &args) {
  if (args.empty()) {
    return createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Usage: llvm-advisor compile <compiler> [compiler-args...]");
  }

  std::string compiler = args[0];
  std::vector<std::string> compilerArgs(args.begin() + 1, args.end());

  RemarksConfig config;
  config.verbose = true;
  config.keepTempFiles = true; // Keep for user inspection
  config.tempDir = "";         // Will be created automatically
  config.outputDir = "";

  CompilerWrapper wrapper(compiler);
  auto result = wrapper.execute(compilerArgs, config);

  if (!result) {
    return result.takeError();
  }

  outs() << "Remarks and profile data saved to: " << config.outputDir << "\n";
  return result;
}

int main(int argc, char **argv) {
  // Handle special cases first before LLVM CommandLine parsing
  if (argc > 1) {
    std::string firstArg = argv[1];
    if (firstArg == "--help" || firstArg == "-h") {
      outs() << "LLVM Advisor Compiler Wrapper\n\n";
      outs() << "Usage:\n";
      outs() << "  llvm-advisor view <compiler> [compiler-args...]    - "
                "Compile with analysis and launch web advisor\n";
      outs() << "  llvm-advisor compile <compiler> [compiler-args...] - "
                "Compile with optimization analysis\n";
      outs() << "  llvm-advisor <compiler> [compiler-args...]         - Same "
                "as compile\n\n";
      outs() << "Examples:\n";
      outs() << "  llvm-advisor view clang -O2 -g -fopenmp main.c\n";
      outs() << "  llvm-advisor compile clang++ -O3 -std=c++17 app.cpp\n";
      return 0;
    }
  }

  // Determine subcommand and split arguments
  bool isView = false;
  bool isCompile = false;
  int startIdx = 1;

  if (argc > 1) {
    std::string firstArg = argv[1];
    if (firstArg == "view") {
      isView = true;
      startIdx = 2;
    } else if (firstArg == "compile") {
      isCompile = true;
      startIdx = 2;
    }
  }

  // Collect remaining arguments for the compiler
  std::vector<std::string> compilerArgs;
  for (int i = startIdx; i < argc; ++i) {
    compilerArgs.push_back(argv[i]);
  }

  // Create appropriate handler and execute
  std::unique_ptr<SubcommandHandler> handler;

  if (isView) {
    handler = std::make_unique<ViewSubcommand>();
  } else if (isCompile) {
    handler = std::make_unique<CompileSubcommand>();
  } else {
    // Default behavior - treat as compile
    handler = std::make_unique<CompileSubcommand>();
  }

  auto result = handler->execute(compilerArgs);
  if (!result) {
    errs() << "Error: " << toString(result.takeError()) << "\n";
    return 1;
  }

  return *result;
}
