//===--- ClangAnalyzerUtils.cpp - LLVM Advisor ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Clang/ClangAnalyzerUtils.h"
#include "Utils/Normalization.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VersionTuple.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <mutex>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace llvm;
using namespace llvm::advisor;

static bool isSkipNextFlag(StringRef Arg) {
  return Arg == "-o" || Arg == "-MF" || Arg == "-MT" || Arg == "-MQ" ||
         Arg == "-MJ";
}

static bool isInlineSkipFlag(StringRef Arg) {
  return (Arg.starts_with("-o") && Arg.size() > 2) ||
         Arg.starts_with("-fdiagnostics-format=");
}

static bool shouldSkipArg(StringRef Arg, bool SkipNext) {
  if (SkipNext)
    return true;
  if (isSkipNextFlag(Arg))
    return true;
  if (isInlineSkipFlag(Arg))
    return true;
  if (Arg == "-c")
    return true;
  if (Arg.starts_with("-MMD") || Arg.starts_with("-MD"))
    return true;
  if (Arg == "-fsyntax-only")
    return true;
  return false;
}

static bool isSourceInputArg(const CapabilityContext &Context, StringRef Arg) {
  return Arg == Context.Unit.SourcePath || Arg == Context.SourcePath ||
         normalizePath(Arg, Context.WorkingDirectory) == Context.SourcePath;
}

static bool hasGCCConfigArg(ArrayRef<std::string> Args) {
  return llvm::any_of(Args, [](const std::string &Arg) {
    return StringRef(Arg).starts_with("--gcc-install-dir=") ||
           StringRef(Arg).starts_with("--gcc-toolchain=");
  });
}

static std::optional<std::string> detectGCCInstallDir() {
  SmallString<256> GCCRoot("/usr/lib/gcc");
  if (!sys::fs::is_directory(GCCRoot))
    return std::nullopt;

  std::optional<std::string> BestPath;
  VersionTuple BestVersion;
  bool HaveBest = false;
  std::error_code EC;
  for (sys::fs::directory_iterator ArchIt(GCCRoot, EC), ArchEnd;
       ArchIt != ArchEnd && !EC; ArchIt.increment(EC)) {
    if (!sys::fs::is_directory(ArchIt->path()))
      continue;
    std::error_code InnerEC;
    for (sys::fs::directory_iterator VerIt(ArchIt->path(), InnerEC), VerEnd;
         VerIt != VerEnd && !InnerEC; VerIt.increment(InnerEC)) {
      if (!sys::fs::is_directory(VerIt->path()))
        continue;
      StringRef Version = sys::path::filename(VerIt->path());
      VersionTuple Parsed;
      if (Parsed.tryParse(Version))
        continue;
      SmallString<256> IncludeDir("/usr/include/c++");
      sys::path::append(IncludeDir, Version);
      if (!sys::fs::is_directory(IncludeDir))
        continue;
      if (!HaveBest || Parsed > BestVersion) {
        BestVersion = Parsed;
        BestPath = VerIt->path();
        HaveBest = true;
      }
    }
  }
  return BestPath;
}

static void maybeAddGCCInstallDir(StringRef Program,
                                  SmallVectorImpl<std::string> &Args) {
  StringRef ProgramName = sys::path::filename(Program);
  if (!ProgramName.starts_with("clang"))
    return;
  if (hasGCCConfigArg(Args))
    return;
  if (std::optional<std::string> GCCInstallDir = detectGCCInstallDir())
    Args.push_back(("--gcc-install-dir=" + *GCCInstallDir));
}

namespace {

class ScopedWorkingDirectory {
public:
  explicit ScopedWorkingDirectory(StringRef NewDir) : Guard(getMutex()) {
    if (NewDir.empty())
      return;
    sys::fs::current_path(OriginalDir);
    if (auto EC = sys::fs::set_current_path(NewDir))
      consumeError(createStringError(EC, "failed to change working directory"));
    else
      Active = true;
  }

  ~ScopedWorkingDirectory() {
    if (Active)
      (void)sys::fs::set_current_path(OriginalDir);
  }

  bool succeeded() const { return Active; }

private:
  static std::mutex &getMutex() {
    static std::mutex M;
    return M;
  }

  std::unique_lock<std::mutex> Guard;
  SmallString<256> OriginalDir;
  bool Active = false;
};

#ifndef _WIN32
class ScopedStderrCapture {
public:
  ScopedStderrCapture() {
    if (auto EC = sys::fs::createTemporaryFile("advisor-clang-stderr", "tmp",
                                               CapturePath)) {
      consumeError(
          createStringError(EC, "failed to create stderr capture file"));
      return;
    }
    CaptureFD = ::open(CapturePath.c_str(), O_WRONLY | O_TRUNC);
    if (CaptureFD < 0)
      return;
    SavedFD = ::dup(STDERR_FILENO);
    if (SavedFD < 0)
      return;
    if (::dup2(CaptureFD, STDERR_FILENO) != -1)
      Active = true;
  }

  ~ScopedStderrCapture() { restore(); }

  std::string finish() {
    restore();
    if (!Active || CapturePath.empty())
      return {};
    std::string Output;
    if (auto Buf = MemoryBuffer::getFile(CapturePath))
      Output = (*Buf)->getBuffer().str();
    sys::fs::remove(CapturePath);
    CapturePath.clear();
    return Output;
  }

private:
  void restore() {
    if (Active && SavedFD >= 0)
      (void)::dup2(SavedFD, STDERR_FILENO);
    Active = false;
    if (SavedFD >= 0) {
      ::close(SavedFD);
      SavedFD = -1;
    }
    if (CaptureFD >= 0) {
      ::close(CaptureFD);
      CaptureFD = -1;
    }
  }

  SmallString<128> CapturePath;
  int SavedFD = -1;
  int CaptureFD = -1;
  bool Active = false;
};
#endif

static Expected<std::string> resolveProgram(StringRef Program) {
  if (sys::path::is_absolute(Program) && sys::fs::can_execute(Program))
    return Program.str();
  ErrorOr<std::string> Found = sys::findProgramByName(Program);
  if (!Found)
    return createStringError(Found.getError(), "cannot find program: %s",
                             Program.str().c_str());
  return *Found;
}

static SmallVector<std::string, 32>
buildCompileInvocation(const CapabilityContext &Context) {
  SmallVector<std::string, 32> Args;
  if (Context.Unit.Arguments.empty())
    return Args;

  bool SkipNext = false;
  for (size_t I = 1, E = Context.Unit.Arguments.size(); I != E; ++I) {
    StringRef Arg(Context.Unit.Arguments[I]);
    if (SkipNext) {
      SkipNext = false;
      continue;
    }
    if (isSourceInputArg(Context, Arg))
      continue;
    if (isSkipNextFlag(Arg)) {
      SkipNext = true;
      continue;
    }
    if (isInlineSkipFlag(Arg))
      continue;
    Args.push_back(Arg.str());
  }
  if (!Context.Unit.Arguments.empty())
    maybeAddGCCInstallDir(Context.Unit.Arguments.front(), Args);
  return Args;
}

static Expected<std::string>
runCompilerInvocation(const CapabilityContext &Context,
                      ArrayRef<std::string> ExtraArgs, StringRef OutPath) {
  if (Context.Unit.Arguments.empty())
    return createStringError(inconvertibleErrorCode(),
                             "missing compiler arguments");

  Expected<std::string> Program =
      resolveProgram(Context.Unit.Arguments.front());
  if (!Program)
    return Program.takeError();

  SmallVector<std::string, 32> Args = buildCompileInvocation(Context);
  for (const std::string &Arg : ExtraArgs)
    Args.push_back(Arg);
  Args.push_back(Context.SourcePath);

  SmallString<128> StdoutPath, StderrPath;
  if (auto EC = sys::fs::createTemporaryFile("advisor-compile-out", "tmp",
                                             StdoutPath))
    return createStringError(EC, "failed to create stdout temp file");
  if (auto EC = sys::fs::createTemporaryFile("advisor-compile-err", "tmp",
                                             StderrPath))
    return createStringError(EC, "failed to create stderr temp file");

  SmallVector<StringRef, 32> ExecArgs;
  ExecArgs.push_back(*Program);
  for (const std::string &Arg : Args)
    ExecArgs.push_back(Arg);

  std::optional<StringRef> Redirects[] = {std::nullopt, StringRef(StdoutPath),
                                          StringRef(StderrPath)};

  {
    ScopedWorkingDirectory CWD(Context.WorkingDirectory);
    if (!CWD.succeeded())
      return createStringError(inconvertibleErrorCode(),
                               "failed to set working directory");
    int ExitCode =
        sys::ExecuteAndWait(*Program, ExecArgs, std::nullopt, Redirects);
    std::string Stderr;
    if (auto Buf = MemoryBuffer::getFile(StderrPath))
      Stderr = (*Buf)->getBuffer().str();
    sys::fs::remove(StdoutPath);
    sys::fs::remove(StderrPath);
    if (ExitCode != 0)
      return createStringError(inconvertibleErrorCode(),
                               "compiler replay failed for %s: %s",
                               Context.SourcePath.c_str(), Stderr.c_str());
  }

  if (!sys::fs::exists(OutPath))
    return createStringError(inconvertibleErrorCode(),
                             "expected output was not produced: %s",
                             OutPath.str().c_str());
  return OutPath.str();
}

} // namespace

Expected<SmallVector<std::string, 32>>
llvm::advisor::buildBaseClangArgs(const CapabilityContext &Context) {
  if (Context.Unit.Arguments.empty())
    return createStringError(inconvertibleErrorCode(),
                             "missing compiler arguments");

  SmallVector<std::string, 32> Args;
  bool SkipNext = false;
  for (size_t I = 1, E = Context.Unit.Arguments.size(); I != E; ++I) {
    StringRef Arg(Context.Unit.Arguments[I]);
    if (isSourceInputArg(Context, Arg)) {
      SkipNext = false;
      continue;
    }
    bool Skip = shouldSkipArg(Arg, SkipNext);
    SkipNext = isSkipNextFlag(Arg);
    if (Skip)
      continue;
    Args.push_back(Arg.str());
  }
  if (!Context.Unit.Arguments.empty())
    maybeAddGCCInstallDir(Context.Unit.Arguments.front(), Args);
  return Args;
}

Expected<SmallVector<std::string, 32>>
llvm::advisor::buildClangArgs(const CapabilityContext &Context,
                              ArrayRef<StringRef> ExtraArgs) {
  Expected<SmallVector<std::string, 32>> Args = buildBaseClangArgs(Context);
  if (!Args)
    return Args;

  Args->push_back("-fsyntax-only");
  for (StringRef Arg : ExtraArgs)
    Args->push_back(Arg.str());
  return Args;
}

Expected<std::unique_ptr<clang::ASTUnit>>
llvm::advisor::buildASTUnit(const CapabilityContext &Context,
                            ArrayRef<StringRef> ExtraArgs) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Source =
      MemoryBuffer::getFile(Context.SourcePath);
  if (!Source)
    return createStringError(Source.getError(), "cannot read source: %s",
                             Context.SourcePath.c_str());

  Expected<SmallVector<std::string, 32>> Args =
      buildClangArgs(Context, ExtraArgs);
  if (!Args)
    return Args.takeError();
  std::vector<std::string> ToolArgs(Args->begin(), Args->end());
  ScopedWorkingDirectory CWD(Context.WorkingDirectory);
  if (!CWD.succeeded())
    return createStringError(inconvertibleErrorCode(),
                             "failed to set working directory");
#ifndef _WIN32
  ScopedStderrCapture Capture;
#endif
  std::unique_ptr<clang::ASTUnit> AST =
      clang::tooling::buildASTFromCodeWithArgs(Source.get()->getBuffer(),
                                               ToolArgs, Context.SourcePath,
                                               "llvm-advisor");
#ifndef _WIN32
  std::string CapturedStderr = Capture.finish();
#endif
  if (!AST)
    return createStringError(
        inconvertibleErrorCode(), "failed to build AST for: %s%s%s",
        Context.SourcePath.c_str(),
#ifndef _WIN32
        CapturedStderr.empty() ? "" : " — ", CapturedStderr.c_str()
#else
        "", ""
#endif
    );
  return AST;
}

Expected<std::string>
llvm::advisor::emitLLVMIR(const CapabilityContext &Context, StringRef OutPath) {
  SmallVector<std::string, 8> ExtraArgs = {"-S", "-emit-llvm", "-o",
                                           OutPath.str()};
  return runCompilerInvocation(Context, ExtraArgs, OutPath);
}

Expected<std::string>
llvm::advisor::emitAssembly(const CapabilityContext &Context,
                            StringRef OutPath) {
  SmallVector<std::string, 8> ExtraArgs = {"-S", "-o", OutPath.str()};
  return runCompilerInvocation(Context, ExtraArgs, OutPath);
}

Expected<std::string>
llvm::advisor::emitOptRemarks(const CapabilityContext &Context,
                              StringRef OutPath) {
  SmallString<128> ObjPath;
  if (auto EC = sys::fs::createTemporaryFile("advisor-remarks-obj", "tmp",
                                             ObjPath))
    return createStringError(EC, "failed to create temp object file");

  SmallVector<std::string, 8> ExtraArgs = {
      "-c", "-o", ObjPath.str().str(), "-fsave-optimization-record=yaml",
      ("-foptimization-record-file=" + OutPath).str()};
  Expected<std::string> Result =
      runCompilerInvocation(Context, ExtraArgs, OutPath);
  sys::fs::remove(ObjPath);
  return Result;
}

Expected<std::string>
llvm::advisor::emitObject(const CapabilityContext &Context, StringRef OutPath) {
  SmallVector<std::string, 8> ExtraArgs = {"-c", "-o", OutPath.str()};
  return runCompilerInvocation(Context, ExtraArgs, OutPath);
}
