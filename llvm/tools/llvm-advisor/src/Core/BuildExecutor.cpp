#include "BuildExecutor.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace advisor {

BuildExecutor::BuildExecutor(const AdvisorConfig &config) : config_(config) {}

Expected<int> BuildExecutor::execute(const std::string &compiler,
                                     const std::vector<std::string> &args,
                                     BuildContext &buildContext,
                                     const std::string &tempDir) {
  auto instrumentedArgs = instrumentCompilerArgs(args, buildContext, tempDir);

  auto compilerPath = sys::findProgramByName(compiler);
  if (!compilerPath) {
    return createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "Compiler not found: " + compiler);
  }

  std::vector<StringRef> execArgs;
  execArgs.push_back(compiler);
  for (const auto &arg : instrumentedArgs) {
    execArgs.push_back(arg);
  }

  if (config_.getVerbose()) {
    outs() << "Executing: " << compiler;
    for (const auto &arg : instrumentedArgs) {
      outs() << " " << arg;
    }
    outs() << "\n";
  }

  return sys::ExecuteAndWait(*compilerPath, execArgs);
}

std::vector<std::string>
BuildExecutor::instrumentCompilerArgs(const std::vector<std::string> &args,
                                      BuildContext &buildContext,
                                      const std::string &tempDir) {

  std::vector<std::string> result = args;
  std::set<std::string> existingFlags;

  // Scan existing flags to avoid duplication
  for (const auto &arg : args) {
    if (arg.find("-g") == 0)
      existingFlags.insert("debug");
    if (arg.find("-fsave-optimization-record") != std::string::npos)
      existingFlags.insert("remarks");
    if (arg.find("-fprofile-instr-generate") != std::string::npos)
      existingFlags.insert("profile");
  }

  // Add debug info if not present
  if (existingFlags.find("debug") == existingFlags.end()) {
    result.push_back("-g");
  }

  // Add optimization remarks with proper redirection
  if (existingFlags.find("remarks") == existingFlags.end()) {
    result.push_back("-fsave-optimization-record");
    result.push_back("-foptimization-record-file=" + tempDir +
                     "/remarks.opt.yaml");
    buildContext.expectedGeneratedFiles.push_back(tempDir +
                                                  "/remarks.opt.yaml");
  } else {
    // If user already specified remarks, find and redirect the file
    bool foundFileFlag = false;
    for (auto &arg : result) {
      if (arg.find("-foptimization-record-file=") != std::string::npos) {
        // Extract filename and redirect to temp
        StringRef existingPath = StringRef(arg).substr(26);
        StringRef filename = sys::path::filename(existingPath);
        arg = "-foptimization-record-file=" + tempDir + "/" + filename.str();
        buildContext.expectedGeneratedFiles.push_back(tempDir + "/" +
                                                      filename.str());
        foundFileFlag = true;
        break;
      }
    }
    // If no explicit file specified, add our own
    if (!foundFileFlag) {
      result.push_back("-foptimization-record-file=" + tempDir +
                       "/remarks.opt.yaml");
      buildContext.expectedGeneratedFiles.push_back(tempDir +
                                                    "/remarks.opt.yaml");
    }
  }

  // Add profiling if enabled and not present, redirect to temp directory
  if (config_.getRunProfiler() &&
      existingFlags.find("profile") == existingFlags.end()) {
    result.push_back("-fprofile-instr-generate=" + tempDir +
                     "/profile.profraw");
    result.push_back("-fcoverage-mapping");
    buildContext.expectedGeneratedFiles.push_back(tempDir + "/profile.profraw");
  }

  return result;
}

} // namespace advisor
} // namespace llvm
