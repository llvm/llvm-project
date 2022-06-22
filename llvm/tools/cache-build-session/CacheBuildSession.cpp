//===-- CacheBuildSession.cpp - cache-build-session tool ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Accepts a command that triggers a build and, while the command is running,
// all the compiler invocations that are part of that build will be sharing the
// same dependency scanning daemon.
//
//===----------------------------------------------------------------------===//

#include "CMakeFileAPI.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Config/config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;

static Error inferCMakePathPrefixes(StringRef CMakeBuildPath,
                                    SmallVectorImpl<StringRef> &PrefixMaps,
                                    StringSaver &Saver) {
  cmake_file_api::Index Index;
  if (Error E = cmake_file_api::Index::fromPath(CMakeBuildPath).moveInto(Index))
    return E;
  cmake_file_api::CodeModel CodeModel;
  if (Error E = Index.getCodeModel().moveInto(CodeModel))
    return E;
  StringRef BuildPath;
  if (Error E = CodeModel.getBuildPath().moveInto(BuildPath))
    return E;
  StringRef SourcePath;
  if (Error E = CodeModel.getSourcePath().moveInto(SourcePath))
    return E;
  SmallVector<StringRef, 8> ExtraSourcePaths;
  if (Error E = CodeModel.getExtraTopLevelSourcePaths(ExtraSourcePaths))
    return E;

  StringSet<> SeenMapNames;
  auto getUniqueMapName = [&](StringRef SourcePath) -> StringRef {
    StringRef Filename = sys::path::filename(SourcePath);
    StringRef MapName = Filename;
    unsigned Count = 0;
    while (SeenMapNames.contains(MapName)) {
      MapName = Saver.save(Filename + "-" + utostr(++Count));
    }
    SeenMapNames.insert(MapName);
    return MapName;
  };

  PrefixMaps.push_back(Saver.save(BuildPath + "=/^build"));
  PrefixMaps.push_back(Saver.save(SourcePath + "=/^src"));
  for (StringRef SourcePath : ExtraSourcePaths) {
    PrefixMaps.push_back(
        Saver.save(SourcePath + "=/^src-" + getUniqueMapName(SourcePath)));
  }
  return Error::success();
}

int main(int Argc, const char **Argv) {
  InitLLVM X(Argc, Argv);

  cl::OptionCategory OptCategory("cache-build-session options");
  cl::extrahelp MoreHelp("\n"
                         "Accepts a command that triggers a build and, while "
                         "the command is running,\n"
                         "all the compiler invocations that are part of that "
                         "build will share the same\n"
                         "dependency scanning daemon.\n"
                         "\n");

  cl::opt<bool> InferCMakePrefixMaps(
      "prefix-map-cmake",
      cl::desc("Infer source and build prefixes from a CMake build directory"),
      cl::cat(OptCategory));
  cl::opt<bool> Verbose("v", cl::desc("Verbose output"), cl::cat(OptCategory));
  // This is here only to improve the help message (for "USAGE:" line).
  cl::list<std::string> Inputs(cl::Positional, cl::desc("command ..."));

  int CmdArgsI = 1;
  while (CmdArgsI < Argc && Argv[CmdArgsI][0] == '-')
    ++CmdArgsI;

  cl::HideUnrelatedOptions(OptCategory);
  cl::ParseCommandLineOptions(CmdArgsI, Argv, "cache-build-session");

  SmallVector<const char *, 256> CmdArgs(Argv + CmdArgsI, Argv + Argc);
  if (CmdArgs.empty()) {
    cl::PrintHelpMessage();
    return 1;
  }

  auto setEnvVar = [&Verbose](const char *Var, const char *Value) {
#if HAVE_SETENV
    ::setenv(Var, Value, 1);
#elif defined(_WIN32)
    _putenv_s(Var, Value);
#else
#error "unsupported environment"
#endif
    if (Verbose) {
      errs() << "note: setting " << Var << '=' << Value << '\n';
    }
  };

  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);

  if (InferCMakePrefixMaps) {
    SmallString<128> WorkingDirectory;
    if (Error E = errorCodeToError(sys::fs::current_path(WorkingDirectory))) {
      errs() << "error: failed getting working directory: "
             << toString(std::move(E)) << '\n';
      return 1;
    }
    SmallVector<StringRef, 8> PrefixMaps;
    if (Error E = inferCMakePathPrefixes(WorkingDirectory, PrefixMaps, Saver)) {
      errs() << "error: could not infer prefix map for CMake build: "
             << toString(std::move(E)) << '\n';
      return 1;
    }
    if (!PrefixMaps.empty()) {
      constexpr const char *PrefixMapEnvVar = "LLVM_CACHE_PREFIX_MAPS";
      SmallString<128> PrefixMapsConcat;
      const char *PriorPrefixMaps = ::getenv(PrefixMapEnvVar);
      if (PriorPrefixMaps) {
        // Merge the derived prefix maps with the pre-existing ones from the
        // environment variable.
        PrefixMapsConcat.append(PriorPrefixMaps);
        PrefixMapsConcat.push_back(';');
      }
      for (StringRef PrefixMap : PrefixMaps) {
        PrefixMapsConcat.append(PrefixMap);
        PrefixMapsConcat.push_back(';');
      }
      assert(PrefixMapsConcat.back() == ';');
      PrefixMapsConcat.pop_back();
      setEnvVar(PrefixMapEnvVar, Saver.save(PrefixMapsConcat.str()).data());
    }
  }

  // Set 'LLVM_CACHE_BUILD_SESSION_ID' to a unique identifier so that compiler
  // invocations under the given command share the same depscan daemon while the
  // command is running.
  // Uses the process id to ensure parallel invocations of
  // `cache-build-session` will not share the same identifier, and
  // 'elapsed nanoseconds since epoch' to ensure the same for consecutive
  // invocations.
  SmallString<32> SessionId;
  raw_svector_ostream(SessionId)
      << sys::Process::getProcessId() << '-'
      << std::chrono::system_clock::now().time_since_epoch().count();
  setEnvVar("LLVM_CACHE_BUILD_SESSION_ID", SessionId.c_str());

  ErrorOr<std::string> ExecPathOrErr = sys::findProgramByName(CmdArgs.front());
  if (!ExecPathOrErr) {
    errs() << "error: cannot find executable " << CmdArgs.front() << '\n';
    return 1;
  }
  std::string ExecPath = std::move(*ExecPathOrErr);
  CmdArgs[0] = ExecPath.c_str();

  SmallVector<StringRef, 16> RefArgs;
  RefArgs.reserve(CmdArgs.size());
  for (const char *Arg : CmdArgs) {
    RefArgs.push_back(Arg);
  }

  std::string ErrMsg;
  int Result = sys::ExecuteAndWait(RefArgs.front(), RefArgs, /*Env*/ None,
                                   /*Redirects*/ {}, /*SecondsToWait*/ 0,
                                   /*MemoryLimit*/ 0, &ErrMsg);
  if (!ErrMsg.empty()) {
    errs() << "error: failed executing command: " << ErrMsg << '\n';
  }
  return Result;
}
