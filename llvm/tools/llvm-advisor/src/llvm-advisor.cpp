#include "Config/AdvisorConfig.h"
#include "Core/CompilationManager.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::advisor;

static cl::opt<std::string> ConfigFile("config", cl::desc("Configuration file"),
                                       cl::value_desc("filename"));
static cl::opt<std::string> OutputDir("output-dir",
                                      cl::desc("Output directory"),
                                      cl::value_desc("directory"));
static cl::opt<bool> Verbose("verbose", cl::desc("Verbose output"));
static cl::opt<bool> KeepTemps("keep-temps", cl::desc("Keep temporary files"));
static cl::opt<bool> NoProfiler("no-profiler", cl::desc("Disable profiler"));

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Parse llvm-advisor options until we find the compiler
  std::vector<const char *> advisorArgs;
  advisorArgs.push_back(argv[0]);

  int compilerArgStart = 1;
  bool foundCompiler = false;

  for (int i = 1; i < argc; ++i) {
    StringRef arg(argv[i]);
    if (arg.starts_with("--") ||
        (arg.starts_with("-") && arg.size() > 1 && arg != "-")) {
      advisorArgs.push_back(argv[i]);
      if (arg == "--config" || arg == "--output-dir") {
        if (i + 1 < argc && !StringRef(argv[i + 1]).starts_with("-")) {
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
    errs() << "Error: No compiler command provided.\n";
    errs() << "Usage: llvm-advisor [options] <compiler> [compiler-args...]\n";
    return 1;
  }

  // Parse llvm-advisor options
  int advisorArgc = advisorArgs.size();
  cl::ParseCommandLineOptions(advisorArgc,
                              const_cast<char **>(advisorArgs.data()),
                              "LLVM Compilation Advisor");

  // Extract compiler and arguments
  std::string compiler = argv[compilerArgStart];
  std::vector<std::string> compilerArgs;
  for (int i = compilerArgStart + 1; i < argc; ++i) {
    compilerArgs.push_back(argv[i]);
  }

  // Configure advisor
  AdvisorConfig config;
  if (!ConfigFile.empty()) {
    if (auto Err = config.loadFromFile(ConfigFile).takeError()) {
      errs() << "Error loading config: " << toString(std::move(Err)) << "\n";
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
  if (auto EC = sys::fs::create_directories(config.getOutputDir())) {
    errs() << "Error creating output directory: " << EC.message() << "\n";
    return 1;
  }

  if (config.getVerbose()) {
    outs() << "LLVM Compilation Advisor\n";
    outs() << "Compiler: " << compiler << "\n";
    outs() << "Output: " << config.getOutputDir() << "\n";
  }

  // Execute with data collection
  CompilationManager manager(config);
  auto result = manager.executeWithDataCollection(compiler, compilerArgs);

  if (result) {
    if (config.getVerbose()) {
      outs() << "Compilation completed (exit code: " << *result << ")\n";
    }
    return *result;
  } else {
    errs() << "Error: " << toString(result.takeError()) << "\n";
    return 1;
  }
}
