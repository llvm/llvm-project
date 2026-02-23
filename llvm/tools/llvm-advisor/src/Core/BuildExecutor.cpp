//===---------------- BuildExecutor.cpp - LLVM Advisor --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the BuildExecutor code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "BuildExecutor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Options/Options.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

namespace llvm {
namespace advisor {

namespace {

bool isClangCompiler(llvm::StringRef CompilerPath) {
  llvm::StringRef Name = llvm::sys::path::filename(CompilerPath);
  return Name.starts_with("clang");
}

/// Resolve a compiler name/path to an absolute path on disk via getToolPath
/// and PATH lookup.  Returns an error if the compiler cannot be found.
llvm::Expected<std::string> resolveCompilerPath(const AdvisorConfig &Config,
                                                llvm::StringRef Compiler) {
  std::string Requested = Config.getToolPath(Compiler);
  if (auto Resolved = llvm::sys::findProgramByName(Requested))
    return *Resolved;
  if (llvm::sys::fs::exists(Requested))
    return Requested;
  return llvm::createStringError(
      std::make_error_code(std::errc::no_such_file_or_directory),
      "Compiler not found: " + Requested);
}

/// Build a clang::driver::Driver + Compilation from an argument list.
/// For non-Clang compilers the Driver and Compilation fields are left null and
/// InstrumentedArgs is set to a verbatim copy of Args.
///
/// \param SetCheckInputsExist  Pass true when the compilation will be executed
///                             (so the driver validates that input files
///                             exist), false when the result is used for
///                             analysis only.
llvm::Expected<BuildExecutor::PreparedBuild>
buildDriverCompilation(llvm::StringRef CompilerPath,
                       llvm::ArrayRef<std::string> Args,
                       bool SetCheckInputsExist) {
  BuildExecutor::PreparedBuild Build;
  Build.CompilerPath = CompilerPath.str();
  Build.InstrumentedArgs.assign(Args.begin(), Args.end());
  Build.UsesDriver = isClangCompiler(CompilerPath);
  if (!Build.UsesDriver)
    return Build;

  Build.DiagnosticsOptions = std::make_unique<clang::DiagnosticOptions>();
  auto DiagPrinter = std::make_unique<clang::TextDiagnosticPrinter>(
      llvm::errs(), *Build.DiagnosticsOptions);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(
      new clang::DiagnosticIDs());
  Build.Diagnostics = std::make_unique<clang::DiagnosticsEngine>(
      DiagIDs, *Build.DiagnosticsOptions, DiagPrinter.get(),
      /*ShouldOwnClient=*/false);

  auto Driver = std::make_unique<clang::driver::Driver>(
      Build.CompilerPath, llvm::sys::getDefaultTargetTriple(),
      *Build.Diagnostics);
  Driver->setTitle("llvm-advisor");
  Driver->setCheckInputsExist(SetCheckInputsExist);

  llvm::SmallVector<const char *, 32> Argv;
  Argv.push_back(Build.CompilerPath.c_str());
  for (const auto &Arg : Args)
    Argv.push_back(Arg.c_str());

  auto *Compilation = Driver->BuildCompilation(Argv);
  if (!Compilation)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Failed to build clang driver invocation for: " + CompilerPath.str());

  Build.DiagnosticClient = std::move(DiagPrinter);
  Build.Driver = std::move(Driver);
  Build.Compilation.reset(Compilation);
  return Build;
}

} // namespace

BuildExecutor::PreparedBuild::PreparedBuild() = default;
BuildExecutor::PreparedBuild::PreparedBuild(PreparedBuild &&) noexcept =
    default;
auto BuildExecutor::PreparedBuild::operator=(PreparedBuild &&) noexcept
    -> PreparedBuild & = default;
BuildExecutor::PreparedBuild::~PreparedBuild() = default;

BuildExecutor::BuildExecutor(const AdvisorConfig &Config) : config(Config) {}

llvm::Expected<int>
BuildExecutor::execute(llvm::StringRef Compiler,
                       const llvm::SmallVectorImpl<std::string> &Args,
                       BuildContext &BuildCtx, llvm::StringRef TempDir,
                       llvm::StringRef ArtifactRoot) {
  auto Prepared = prepareBuild(Compiler, Args, BuildCtx, TempDir, ArtifactRoot);
  if (!Prepared)
    return Prepared.takeError();
  return executePreparedBuild(*Prepared);
}

llvm::Expected<BuildExecutor::PreparedBuild>
BuildExecutor::buildOriginalCompilation(
    llvm::StringRef Compiler, const llvm::SmallVectorImpl<std::string> &Args) {
  auto CompilerPathOrErr = resolveCompilerPath(config, Compiler);
  if (!CompilerPathOrErr)
    return CompilerPathOrErr.takeError();

  // setCheckInputsExist=false: we only need the Driver model for analysis,
  // not a real compilation that validates input files.
  return buildDriverCompilation(*CompilerPathOrErr, Args,
                                /*SetCheckInputsExist=*/false);
}

llvm::Expected<BuildExecutor::PreparedBuild>
BuildExecutor::prepareBuild(llvm::StringRef Compiler,
                            const llvm::SmallVectorImpl<std::string> &Args,
                            BuildContext &BuildCtx, llvm::StringRef TempDir,
                            llvm::StringRef ArtifactRoot) {
  auto CompilerPathOrErr = resolveCompilerPath(config, Compiler);
  if (!CompilerPathOrErr)
    return CompilerPathOrErr.takeError();
  const std::string &ResolvedCompiler = *CompilerPathOrErr;

  // Instrument the user args (add debug info, remarks, profiling flags, etc.).
  auto InstrumentedArgsOrErr = instrumentCompilerArgs(
      ResolvedCompiler, Args, BuildCtx, TempDir, ArtifactRoot);
  if (!InstrumentedArgsOrErr)
    return InstrumentedArgsOrErr.takeError();

  PreparedBuild Build;
  Build.CompilerPath = ResolvedCompiler;
  Build.InstrumentedArgs = std::move(*InstrumentedArgsOrErr);
  Build.UsesDriver = false;
  return Build;
}

llvm::Expected<int> BuildExecutor::executePreparedBuild(PreparedBuild &Build) {
  if (config.getVerbose()) {
    llvm::outs() << "Executing: " << Build.CompilerPath;
    for (const auto &Arg : Build.InstrumentedArgs)
      llvm::outs() << " " << Arg;
    llvm::outs() << "\n";
  }

  llvm::SmallVector<llvm::StringRef, 16> ExecArgs;
  ExecArgs.push_back(Build.CompilerPath); // argv[0]
  for (const auto &Arg : Build.InstrumentedArgs)
    ExecArgs.push_back(Arg);
  return llvm::sys::ExecuteAndWait(Build.CompilerPath, ExecArgs);
}

// NOTE: We parse and rewrite compiler flags using the clang Driver's
// opt::ArgList / DerivedArgList infrastructure so that we manipulate arguments
// at a structured level rather than via raw string operations.  The Driver is
// instantiated with setCheckInputsExist(false) because we only need the parsed
// arg model here, not a real compilation.
llvm::Expected<llvm::SmallVector<std::string, 16>>
BuildExecutor::instrumentCompilerArgs(
    llvm::StringRef CompilerPath,
    const llvm::SmallVectorImpl<std::string> &Args, BuildContext &BuildCtx,
    llvm::StringRef TempDir, llvm::StringRef ArtifactRoot) {
  std::string OutputRoot =
      ArtifactRoot.empty() ? std::string(TempDir) : std::string(ArtifactRoot);
  auto RegisterExpectedFile = [&](llvm::StringRef Path) {
    if (!Path.empty())
      BuildCtx.expectedGeneratedFiles.push_back(Path.str());
  };

  const bool IsClangTool = isClangCompiler(CompilerPath);
  llvm::SmallVector<std::string, 16> Result(Args.begin(), Args.end());

  bool HasDebug = false;
  bool HasRemarksFlag = false;
  bool HasRemarksFile = false;
  bool HasProfile = false;
  bool HasCoverage = false;
  bool HasRpass = false;
  bool HasDiagFormat = false;
  bool HasParseableFixits = false;
  bool HasAbsolutePaths = false;

  for (auto &Arg : Result) {
    llvm::StringRef ArgRef(Arg);
    if (ArgRef.starts_with("-g"))
      HasDebug = true;

    if (ArgRef == "-fsave-optimization-record" ||
        ArgRef.starts_with("-fsave-optimization-record="))
      HasRemarksFlag = true;

    if (ArgRef.starts_with("-foptimization-record-file=")) {
      HasRemarksFile = true;
      // Keep explicit remarks output inside the artifact root.
      llvm::StringRef Requested =
          ArgRef.drop_front(sizeof("-foptimization-record-file=") - 1);
      llvm::StringRef FileName = llvm::sys::path::filename(Requested);
      std::string NewPath = (llvm::Twine(OutputRoot) + "/" + FileName).str();
      Arg = (llvm::Twine("-foptimization-record-file=") + NewPath).str();
      RegisterExpectedFile(NewPath);
    }

    if (ArgRef == "-fprofile-instr-generate" ||
        ArgRef.starts_with("-fprofile-instr-generate="))
      HasProfile = true;
    if (ArgRef == "-fcoverage-mapping")
      HasCoverage = true;
    if (ArgRef.starts_with("-Rpass="))
      HasRpass = true;
    if (ArgRef == "-fdiagnostics-format" ||
        ArgRef.starts_with("-fdiagnostics-format="))
      HasDiagFormat = true;
    if (ArgRef == "-fdiagnostics-parseable-fixits")
      HasParseableFixits = true;
    if (ArgRef == "-fdiagnostics-absolute-paths")
      HasAbsolutePaths = true;
  }

  if (!HasDebug) {
    Result.push_back("-g");
    BuildCtx.hasDebugInfo = true;
  }

  // Non-Clang compilers get only the debug-info guarantee.
  if (!IsClangTool)
    return Result;

  if (!HasRemarksFlag) {
    Result.push_back("-fsave-optimization-record");
    HasRemarksFlag = true;
  }

  if (!HasRemarksFile) {
    std::string RemarksPath =
        (llvm::Twine(OutputRoot) + "/remarks.opt.yaml").str();
    Result.push_back(
        (llvm::Twine("-foptimization-record-file=") + RemarksPath).str());
    RegisterExpectedFile(RemarksPath);
  }

  if (config.getRunProfiler() && !HasProfile) {
    std::string Profraw = (llvm::Twine(OutputRoot) + "/profile.profraw").str();
    Result.push_back(
        (llvm::Twine("-fprofile-instr-generate=") + Profraw).str());
    if (!HasCoverage)
      Result.push_back("-fcoverage-mapping");
    RegisterExpectedFile(Profraw);

    std::string Profdata =
        (llvm::Twine(OutputRoot) + "/profile.profdata").str();
    std::string Report = (llvm::Twine(OutputRoot) + "/coverage.json").str();
    RegisterExpectedFile(Profdata);
    RegisterExpectedFile(Report);

    CoverageProfileSite Site;
    Site.rawProfile = Profraw;
    Site.indexedProfile = Profdata;
    Site.reportPath = Report;
    BuildCtx.coverageSites.push_back(std::move(Site));
  }

  if (!HasRpass) {
    Result.push_back("-Rpass=kernel-info");
    Result.push_back("-Rpass=analysis");
  }

  if (!HasDiagFormat) {
    if (!HasParseableFixits)
      Result.push_back("-fdiagnostics-parseable-fixits");
    if (!HasAbsolutePaths)
      Result.push_back("-fdiagnostics-absolute-paths");
  }

  return Result;
}

} // namespace advisor
} // namespace llvm
