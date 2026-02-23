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
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <memory>

namespace llvm {
namespace advisor {

namespace {

BuildPhase phaseFromAction(clang::driver::Action::ActionClass Kind) {
  switch (Kind) {
  case clang::driver::Action::CompileJobClass:
    return BuildPhase::Compilation;
  case clang::driver::Action::AssembleJobClass:
    return BuildPhase::Assembly;
  case clang::driver::Action::LinkJobClass:
    return BuildPhase::Linking;
  case clang::driver::Action::LipoJobClass:
    return BuildPhase::Linking;
  case clang::driver::Action::StaticLibJobClass:
    return BuildPhase::Archiving;
  default:
    return BuildPhase::Unknown;
  }
}

BuildPhase detectPhaseFromCompilation(const clang::driver::Compilation &C) {
  BuildPhase Detected = BuildPhase::Unknown;
  for (const auto &Cmd : C.getJobs()) {
    const auto *JA =
        llvm::dyn_cast<clang::driver::JobAction>(&Cmd.getSource());
    if (!JA)
      continue;
    BuildPhase Phase = phaseFromAction(JA->getKind());
    if (Phase == BuildPhase::Linking || Phase == BuildPhase::Archiving)
      return Phase;
    if (Phase == BuildPhase::Assembly && Detected != BuildPhase::Linking)
      Detected = Phase;
    if (Phase == BuildPhase::Compilation && Detected == BuildPhase::Unknown)
      Detected = Phase;
  }
  return Detected;
}

void collectFilesFromCompilation(const clang::driver::Compilation &C,
                                 llvm::SmallVector<std::string, 8> &Inputs,
                                 llvm::SmallVector<std::string, 8> &Outputs) {
  llvm::DenseSet<llvm::StringRef> SeenInputs;
  llvm::DenseSet<llvm::StringRef> SeenOutputs;
  for (const auto &Cmd : C.getJobs()) {
    const auto *JA =
        llvm::dyn_cast<clang::driver::JobAction>(&Cmd.getSource());
    if (!JA)
      continue;

    if (JA->getKind() == clang::driver::Action::CompileJobClass) {
      for (const auto &Info : Cmd.getInputInfos()) {
        if (!Info.isFilename())
          continue;
        if (SeenInputs.insert(Info.getFilename()).second)
          Inputs.emplace_back(Info.getFilename());
      }
    }

    for (const auto &Out : Cmd.getOutputFilenames()) {
      if (SeenOutputs.insert(Out).second)
        Outputs.emplace_back(Out);
    }
  }
}

void detectFeaturesFromArgs(llvm::ArrayRef<const char *> Argv,
                            BuildContext &Context) {
  for (const char *Arg : Argv) {
    if (!Arg)
      continue;
    llvm::StringRef ArgRef(Arg);
    if (ArgRef.starts_with("-g"))
      Context.hasDebugInfo = true;
    if (ArgRef.starts_with("-O") && ArgRef.size() > 2)
      Context.hasOptimization = true;
    if (ArgRef.contains("openmp") || ArgRef.contains("offload") ||
        ArgRef.contains("cuda"))
      Context.hasOffloading = true;
    if (ArgRef.starts_with("-march="))
      Context.metadata["target_arch"] = ArgRef.substr(7);
    if (ArgRef.starts_with("-mtune="))
      Context.metadata["tune"] = ArgRef.substr(7);
  }
}

} // namespace

CommandAnalyzer::CommandAnalyzer(llvm::StringRef command,
                                 const llvm::SmallVectorImpl<std::string> &args)
    : command(command.str()), args(args.data(), args.data() + args.size()) {}

BuildContext CommandAnalyzer::analyze() const {
  BuildContext context;
  llvm::SmallString<256> cwd;
  llvm::sys::fs::current_path(cwd);
  context.workingDirectory = std::string(cwd.str());

  context.tool = detectBuildTool();
  context.phase = detectBuildPhase(context.tool);
  context.inputFiles = extractInputFiles();
  context.outputFiles = extractOutputFiles();
  detectBuildFeatures(context);

  return context;
}

void CommandAnalyzer::refineWithCompilation(
    BuildContext &context,
    const clang::driver::Compilation *DriverCompilation) const {
  if (!DriverCompilation)
    return;
  context.tool = BuildTool::Clang;
  BuildPhase Phase = detectPhaseFromCompilation(*DriverCompilation);
  if (Phase != BuildPhase::Unknown)
    context.phase = Phase;

  if (context.inputFiles.empty() || context.outputFiles.empty()) {
    llvm::SmallVector<std::string, 8> Inputs;
    llvm::SmallVector<std::string, 8> Outputs;
    collectFilesFromCompilation(*DriverCompilation, Inputs, Outputs);
    if (context.inputFiles.empty())
      context.inputFiles = Inputs;
    if (context.outputFiles.empty())
      context.outputFiles = Outputs;
  }

  llvm::SmallVector<const char *, 32> ArgPointers;
  for (const auto &Arg : args)
    ArgPointers.push_back(Arg.c_str());
  detectFeaturesFromArgs(ArgPointers, context);
}

BuildTool CommandAnalyzer::detectBuildTool() const {
  return llvm::StringSwitch<BuildTool>(llvm::sys::path::filename(command))
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
      .StartsWith("llvm-", BuildTool::LlvmTools)
      .Default(BuildTool::Unknown);
}

BuildPhase CommandAnalyzer::detectBuildPhase(BuildTool tool) const {
  if (tool == BuildTool::CMake) {
    for (const auto &arg : args) {
      if (arg == "--build")
        return BuildPhase::CMakeBuild;
    }
    return BuildPhase::CMakeConfigure;
  }

  if (tool == BuildTool::Make || tool == BuildTool::Ninja)
    return BuildPhase::MakefileBuild;

  if (tool == BuildTool::Linker)
    return BuildPhase::Linking;

  if (tool == BuildTool::Archiver)
    return BuildPhase::Archiving;

  if (tool == BuildTool::Clang || tool == BuildTool::GCC) {
    for (const auto &arg : args) {
      if (arg == "-E")
        return BuildPhase::Preprocessing;
      if (arg == "-S")
        return BuildPhase::Assembly;
      if (arg == "-c")
        return BuildPhase::Compilation;
    }

    bool hasObjectFile = false;
    for (const auto &Arg : args) {
      llvm::StringRef argRef(Arg);
      if (argRef.ends_with(".o") || argRef.ends_with(".O") ||
          argRef.ends_with(".obj") || argRef.ends_with(".OBJ")) {
        hasObjectFile = true;
        break;
      }
    }
    if (hasObjectFile)
      return BuildPhase::Linking;

    // No explicit stop-flag (-c/-S/-E) was found.  If the command has source
    // files but no object-file inputs the driver will compile *and* link,
    // producing an executable.  Treat that as Linking so that
    // CompilationManager passes the invocation through unchanged — cc1 is
    // never invoked directly for the link step.
    // Note: hasObjectFile was already checked above and returned Linking, so
    // at this point hasObjectFile is always false; we only need to check for
    // source files here.
    bool hasSourceFile = false;
    for (const auto &Arg : args) {
      llvm::StringRef argRef(Arg);
      if (argRef.ends_with(".c") || argRef.ends_with(".C") ||
          argRef.ends_with(".cpp") || argRef.ends_with(".CPP") ||
          argRef.ends_with(".cc") || argRef.ends_with(".CC") ||
          argRef.ends_with(".cxx") || argRef.ends_with(".CXX")) {
        hasSourceFile = true;
        break;
      }
    }
    // A bare "clang foo.c -o foo" (no -c) drives both compilation and linking.
    // Classify as Linking so the invocation is forwarded without modification.
    if (hasSourceFile)
      return BuildPhase::Linking;
  }

  return BuildPhase::Unknown;
}

void CommandAnalyzer::detectBuildFeatures(BuildContext &context) const {
  for (const auto &arg : args) {
    if (arg == "-g" || llvm::StringRef(arg).starts_with("-g"))
      context.hasDebugInfo = true;

    if (llvm::StringRef(arg).starts_with("-O") && arg.length() > 2)
      context.hasOptimization = true;

    llvm::StringRef argRef(arg);
    if (argRef.contains("openmp") || argRef.contains("openacc") ||
        argRef.contains("cuda") || argRef.contains("offload"))
      context.hasOffloading = true;

    if (llvm::StringRef(arg).starts_with("-march="))
      context.metadata["target_arch"] = arg.substr(7);
    if (llvm::StringRef(arg).starts_with("-mtune="))
      context.metadata["tune"] = arg.substr(7);
    if (llvm::StringRef(arg).starts_with("--offload-arch="))
      context.metadata["offload_arch"] = arg.substr(15);
  }
}

llvm::SmallVector<std::string, 8> CommandAnalyzer::extractInputFiles() const {
  llvm::SmallVector<std::string, 8> inputs;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    if (llvm::StringRef(arg).starts_with("-")) {
      if (arg == "-o" || arg == "-I" || arg == "-L" || arg == "-D")
        i++;
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
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    if (arg == "-o" && i + 1 < args.size()) {
      outputs.push_back(args[i + 1]);
      i++;
    }
  }
  return outputs;
}

} // namespace advisor
} // namespace llvm
