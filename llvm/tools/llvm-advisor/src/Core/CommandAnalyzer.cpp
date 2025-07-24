//===----------------- CommandAnalyzer.cpp - LLVM Advisor -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the CommandAnalyzer code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "CommandAnalyzer.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace advisor {

CommandAnalyzer::CommandAnalyzer(llvm::StringRef command,
                                 const llvm::SmallVectorImpl<std::string> &args)
    : command_(command.str()), args_(args.data(), args.data() + args.size()) {}

BuildContext CommandAnalyzer::analyze() const {
  BuildContext context;
  llvm::SmallString<256> cwd;
  llvm::sys::fs::current_path(cwd);
  context.workingDirectory = cwd.str().str();

  context.tool = detectBuildTool();
  context.phase = detectBuildPhase(context.tool);
  context.inputFiles = extractInputFiles();
  context.outputFiles = extractOutputFiles();
  detectBuildFeatures(context);

  return context;
}

BuildTool CommandAnalyzer::detectBuildTool() const {
  return llvm::StringSwitch<BuildTool>(llvm::sys::path::filename(command_))
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
      llvm::StringRef argRef(Arg);
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
      llvm::StringRef argRef(Arg);
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
    if (arg == "-g" || llvm::StringRef(arg).starts_with("-g")) {
      context.hasDebugInfo = true;
    }

    if (llvm::StringRef(arg).starts_with("-O") && arg.length() > 2) {
      context.hasOptimization = true;
    }

    llvm::StringRef argRef(arg);
    if (argRef.contains("openmp") || argRef.contains("openacc") ||
        argRef.contains("cuda") || argRef.contains("offload")) {
      context.hasOffloading = true;
    }

    if (llvm::StringRef(arg).starts_with("-march=")) {
      context.metadata["target_arch"] = arg.substr(7);
    }
    if (llvm::StringRef(arg).starts_with("-mtune=")) {
      context.metadata["tune"] = arg.substr(7);
    }
    if (llvm::StringRef(arg).starts_with("--offload-arch=")) {
      context.metadata["offload_arch"] = arg.substr(15);
    }
  }
}

llvm::SmallVector<std::string, 8> CommandAnalyzer::extractInputFiles() const {
  llvm::SmallVector<std::string, 8> inputs;
  for (size_t i = 0; i < args_.size(); ++i) {
    const auto &arg = args_[i];
    if (llvm::StringRef(arg).starts_with("-")) {
      if (arg == "-o" || arg == "-I" || arg == "-L" || arg == "-D") {
        i++;
      }
      continue;
    }
    if (llvm::sys::fs::exists(arg)) {
      inputs.push_back(arg);
    }
  }
  return inputs;
}

llvm::SmallVector<std::string, 8> CommandAnalyzer::extractOutputFiles() const {
  llvm::SmallVector<std::string, 8> outputs;
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
