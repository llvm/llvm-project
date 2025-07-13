#include "CommandAnalyzer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace advisor {

CommandAnalyzer::CommandAnalyzer(const std::string &command,
                                 const std::vector<std::string> &args)
    : command_(command), args_(args) {}

BuildContext CommandAnalyzer::analyze() const {
  BuildContext context;
  SmallString<256> cwd;
  sys::fs::current_path(cwd);
  context.workingDirectory = cwd.str().str();

  context.tool = detectBuildTool();
  context.phase = detectBuildPhase(context.tool);
  context.inputFiles = extractInputFiles();
  context.outputFiles = extractOutputFiles();
  detectBuildFeatures(context);

  return context;
}

BuildTool CommandAnalyzer::detectBuildTool() const {
  return StringSwitch<BuildTool>(sys::path::filename(command_))
      .StartsWith("clang", BuildTool::Clang)
      .StartsWith("gcc", BuildTool::GCC)
      .StartsWith("g++", BuildTool::GCC)
      .Case("cmake", BuildTool::CMake)
      .Case("make", BuildTool::Make)
      .Case("ninja", BuildTool::Ninja)
      .EndsWith("-ld", BuildTool::Linker)
      .Case("ld", BuildTool::Linker)
      .Case("ar", BuildTool::Archiver)
      .Case("llvm-ar", BuildTool::Archiver)
      .StartsWith("llvm-", BuildTool::LLVM_Tools)
      .Default(BuildTool::Unknown);
}

BuildPhase CommandAnalyzer::detectBuildPhase(BuildTool tool) const {
  if (tool == BuildTool::CMake) {
    for (const auto &arg : args_) {
      if (arg == "--build")
        return BuildPhase::CMakeBuild;
    }
    return BuildPhase::CMakeConfigure;
  }

  if (tool == BuildTool::Make || tool == BuildTool::Ninja) {
    return BuildPhase::MakefileBuild;
  }

  if (tool == BuildTool::Linker) {
    return BuildPhase::Linking;
  }

  if (tool == BuildTool::Archiver) {
    return BuildPhase::Archiving;
  }

  if (tool == BuildTool::Clang || tool == BuildTool::GCC) {
    for (const auto &arg : args_) {
      if (arg == "-E")
        return BuildPhase::Preprocessing;
      if (arg == "-S")
        return BuildPhase::Assembly;
      if (arg == "-c")
        return BuildPhase::Compilation;
    }

    bool hasObjectFile = false;
    for (const auto &Arg : args_) {
      StringRef argRef(Arg);
      if (argRef.ends_with(".o") || argRef.ends_with(".O") ||
          argRef.ends_with(".obj") || argRef.ends_with(".OBJ")) {
        hasObjectFile = true;
        break;
      }
    }
    if (hasObjectFile) {
      return BuildPhase::Linking;
    }

    bool hasSourceFile = false;
    for (const auto &Arg : args_) {
      StringRef argRef(Arg);
      if (argRef.ends_with(".c") || argRef.ends_with(".C") ||
          argRef.ends_with(".cpp") || argRef.ends_with(".CPP") ||
          argRef.ends_with(".cc") || argRef.ends_with(".CC") ||
          argRef.ends_with(".cxx") || argRef.ends_with(".CXX")) {
        hasSourceFile = true;
        break;
      }
    }
    if (hasSourceFile) {
      return BuildPhase::Compilation; // Default for source files
    }
  }

  return BuildPhase::Unknown;
}

void CommandAnalyzer::detectBuildFeatures(BuildContext &context) const {
  for (const auto &arg : args_) {
    if (arg == "-g" || StringRef(arg).starts_with("-g")) {
      context.hasDebugInfo = true;
    }

    if (StringRef(arg).starts_with("-O") && arg.length() > 2) {
      context.hasOptimization = true;
    }

    if (arg.find("openmp") != std::string::npos ||
        arg.find("openacc") != std::string::npos ||
        arg.find("cuda") != std::string::npos ||
        arg.find("offload") != std::string::npos) {
      context.hasOffloading = true;
    }

    if (StringRef(arg).starts_with("-march=")) {
      context.metadata["target_arch"] = arg.substr(7);
    }
    if (StringRef(arg).starts_with("-mtune=")) {
      context.metadata["tune"] = arg.substr(7);
    }
    if (StringRef(arg).starts_with("--offload-arch=")) {
      context.metadata["offload_arch"] = arg.substr(15);
    }
  }
}

std::vector<std::string> CommandAnalyzer::extractInputFiles() const {
  std::vector<std::string> inputs;
  for (size_t i = 0; i < args_.size(); ++i) {
    const auto &arg = args_[i];
    if (StringRef(arg).starts_with("-")) {
      if (arg == "-o" || arg == "-I" || arg == "-L" || arg == "-D") {
        i++;
      }
      continue;
    }
    if (sys::fs::exists(arg)) {
      inputs.push_back(arg);
    }
  }
  return inputs;
}

std::vector<std::string> CommandAnalyzer::extractOutputFiles() const {
  std::vector<std::string> outputs;
  for (size_t i = 0; i < args_.size(); ++i) {
    const auto &arg = args_[i];
    if (arg == "-o" && i + 1 < args_.size()) {
      outputs.push_back(args_[i + 1]);
      i++;
    }
  }
  return outputs;
}

} // namespace advisor
} // namespace llvm
