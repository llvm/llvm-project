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
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"
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
    const auto *JA = llvm::dyn_cast<clang::driver::JobAction>(&Cmd.getSource());
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
    const auto *JA = llvm::dyn_cast<clang::driver::JobAction>(&Cmd.getSource());
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

CommandAnalyzer::CommandAnalyzer(llvm::StringRef Command,
                                 const llvm::SmallVectorImpl<std::string> &Args)
    : command(Command.str()), args(Args.data(), Args.data() + Args.size()) {}

BuildContext CommandAnalyzer::analyze() const {
  BuildContext Context;
  llvm::SmallString<256> Cwd;
  llvm::sys::fs::current_path(Cwd);
  Context.workingDirectory = std::string(Cwd.str());

  Context.tool = detectBuildTool();
  Context.phase = detectBuildPhase(Context.tool);
  Context.inputFiles = extractInputFiles();
  Context.outputFiles = extractOutputFiles();
  detectBuildFeatures(Context);

  return Context;
}

void CommandAnalyzer::refineWithCompilation(
    BuildContext &Context,
    const clang::driver::Compilation *DriverCompilation) const {
  if (!DriverCompilation)
    return;
  Context.tool = BuildTool::Clang;
  BuildPhase Phase = detectPhaseFromCompilation(*DriverCompilation);
  if (Phase != BuildPhase::Unknown)
    Context.phase = Phase;

  if (Context.inputFiles.empty() || Context.outputFiles.empty()) {
    llvm::SmallVector<std::string, 8> Inputs;
    llvm::SmallVector<std::string, 8> Outputs;
    collectFilesFromCompilation(*DriverCompilation, Inputs, Outputs);
    if (Context.inputFiles.empty())
      Context.inputFiles = Inputs;
    if (Context.outputFiles.empty())
      Context.outputFiles = Outputs;
  }

  llvm::SmallVector<const char *, 32> ArgPointers;
  for (const auto &Arg : args)
    ArgPointers.push_back(Arg.c_str());
  detectFeaturesFromArgs(ArgPointers, Context);
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

BuildPhase CommandAnalyzer::detectBuildPhase(BuildTool Tool) const {
  if (Tool == BuildTool::CMake) {
    for (const auto &Arg : args) {
      if (Arg == "--build")
        return BuildPhase::CMakeBuild;
    }
    return BuildPhase::CMakeConfigure;
  }

  if (Tool == BuildTool::Make || Tool == BuildTool::Ninja)
    return BuildPhase::MakefileBuild;

  if (Tool == BuildTool::Linker)
    return BuildPhase::Linking;

  if (Tool == BuildTool::Archiver)
    return BuildPhase::Archiving;

  if (Tool == BuildTool::Clang || Tool == BuildTool::GCC) {
    for (const auto &Arg : args) {
      if (Arg == "-E")
        return BuildPhase::Preprocessing;
      if (Arg == "-S")
        return BuildPhase::Assembly;
      if (Arg == "-c")
        return BuildPhase::Compilation;
    }

    bool HasObjectFile = false;
    for (const auto &Arg : args) {
      llvm::StringRef ArgRef(Arg);
      if (ArgRef.ends_with(".o") || ArgRef.ends_with(".O") ||
          ArgRef.ends_with(".obj") || ArgRef.ends_with(".OBJ")) {
        HasObjectFile = true;
        break;
      }
    }
    if (HasObjectFile)
      return BuildPhase::Linking;

    // No explicit stop-flag (-c/-S/-E) was found.  If the command has source
    // files but no object-file inputs the driver will compile *and* link,
    // producing an executable.  Treat that as Linking so that
    // CompilationManager passes the invocation through unchanged — cc1 is
    // never invoked directly for the link step.
    // Note: hasObjectFile was already checked above and returned Linking, so
    // at this point hasObjectFile is always false; we only need to check for
    // source files here.
    bool HasSourceFile = false;
    for (const auto &Arg : args) {
      llvm::StringRef ArgRef(Arg);
      if (ArgRef.ends_with(".c") || ArgRef.ends_with(".C") ||
          ArgRef.ends_with(".cpp") || ArgRef.ends_with(".CPP") ||
          ArgRef.ends_with(".cc") || ArgRef.ends_with(".CC") ||
          ArgRef.ends_with(".cxx") || ArgRef.ends_with(".CXX")) {
        HasSourceFile = true;
        break;
      }
    }
    // A bare "clang foo.c -o foo" (no -c) drives both compilation and linking.
    // Classify as Linking so the invocation is forwarded without modification.
    if (HasSourceFile)
      return BuildPhase::Linking;
  }

  return BuildPhase::Unknown;
}

void CommandAnalyzer::detectBuildFeatures(BuildContext &Context) const {
  for (const auto &Arg : args) {
    if (Arg == "-g" || llvm::StringRef(Arg).starts_with("-g"))
      Context.hasDebugInfo = true;

    if (llvm::StringRef(Arg).starts_with("-O") && Arg.length() > 2)
      Context.hasOptimization = true;

    llvm::StringRef ArgRef(Arg);
    if (ArgRef.contains("openmp") || ArgRef.contains("openacc") ||
        ArgRef.contains("cuda") || ArgRef.contains("offload"))
      Context.hasOffloading = true;

    if (llvm::StringRef(Arg).starts_with("-march="))
      Context.metadata["target_arch"] = Arg.substr(7);
    if (llvm::StringRef(Arg).starts_with("-mtune="))
      Context.metadata["tune"] = Arg.substr(7);
    if (llvm::StringRef(Arg).starts_with("--offload-arch="))
      Context.metadata["offload_arch"] = Arg.substr(15);
  }
}

llvm::SmallVector<std::string, 8> CommandAnalyzer::extractInputFiles() const {
  llvm::SmallVector<std::string, 8> Inputs;
  for (size_t I = 0; I < args.size(); ++I) {
    const auto &Arg = args[I];
    if (llvm::StringRef(Arg).starts_with("-")) {
      if (Arg == "-o" || Arg == "-I" || Arg == "-L" || Arg == "-D")
        I++;
      continue;
    }
    if (llvm::sys::fs::exists(Arg))
      Inputs.push_back(Arg);
  }
  return Inputs;
}

llvm::SmallVector<std::string, 8> CommandAnalyzer::extractOutputFiles() const {
  llvm::SmallVector<std::string, 8> Outputs;
  for (size_t I = 0; I < args.size(); ++I) {
    const auto &Arg = args[I];
    if (Arg == "-o" && I + 1 < args.size()) {
      Outputs.push_back(args[I + 1]);
      I++;
    }
  }
  return Outputs;
}

} // namespace advisor
} // namespace llvm
